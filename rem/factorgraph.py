"""Factor graphs and bucket elimination -- the core of REM.

THE GOVERNING LAW, stated once and enforced everywhere below:

    cost = d ** treewidth        d = states per variable

Entanglement across a cut, bond dimension, edges crossing the cut and treewidth are
the same number. Every routine here logs the treewidth of the ordering it used, because
that number and not the problem size predicts the cost.

CONVENTION. A factor holds an arbitrary real table `phi` over its variables.

    eliminate("min")  ->  min over assignments of  sum_f phi_f      + the argmin
    eliminate("sum")  ->  log sum over assignments of exp( sum_f phi_f )

So phi = -E/kT gives log Z, and phi = E gives the ground-state energy. One table type,
two semirings, no separate energy/probability classes to keep in sync.

THE PATHWIDTH TRAP, designed against rather than patched. The classic bug is to carry a
running joint over every variable seen so far, which computes cost d^PATHwidth -- always
at least treewidth and often far worse. Here `eliminate` keeps a POOL OF FACTORS and, at
each step, touches only the factors that actually mention the variable being eliminated.
Nothing is ever combined that does not have to be. `verify_pathwidth_trap()` builds a long
chain, where pathwidth and treewidth are both 1 but a joint frontier would blow up, and
asserts the largest intermediate table stays at d^2.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

NEG_INF = -np.inf


def logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


@dataclass
class Factor:
    vars: Tuple[str, ...]
    table: np.ndarray

    def __post_init__(self):
        self.vars = tuple(self.vars)
        self.table = np.asarray(self.table, dtype=float)
        if self.table.ndim != len(self.vars):
            raise ValueError(f"factor over {self.vars} has table ndim {self.table.ndim}")


def _align(fvars: Tuple[str, ...], table: np.ndarray, U: Sequence[str],
           cards: Dict[str, int]) -> np.ndarray:
    """Reshape a factor's table so it broadcasts against the variable list U."""
    present = [v for v in U if v in fvars]
    perm = [fvars.index(v) for v in present]
    t = np.transpose(table, perm)
    shape = tuple(cards[v] if v in fvars else 1 for v in U)
    return t.reshape(shape)


class FactorGraph:
    def __init__(self):
        self.cards: Dict[str, int] = {}
        self.factors: List[Factor] = []

    # ---------------------------------------------------------------- construction
    def add_var(self, name: str, card: int) -> str:
        if name in self.cards and self.cards[name] != card:
            raise ValueError(f"variable {name} already has cardinality {self.cards[name]}")
        self.cards[name] = int(card)
        return name

    def add_factor(self, vars: Sequence[str], table) -> Factor:
        table = np.asarray(table, dtype=float)
        vars = tuple(vars)
        for v, n in zip(vars, table.shape):
            if v not in self.cards:
                self.add_var(v, n)
            elif self.cards[v] != n:
                raise ValueError(f"factor axis for {v} is {n}, variable has {self.cards[v]}")
        f = Factor(vars, table)
        self.factors.append(f)
        return f

    @property
    def variables(self) -> List[str]:
        return list(self.cards)

    # ---------------------------------------------------------------- graph structure
    def adjacency(self) -> Dict[str, set]:
        adj = {v: set() for v in self.cards}
        for f in self.factors:
            for a, b in itertools.combinations(f.vars, 2):
                adj[a].add(b)
                adj[b].add(a)
        return adj

    def _greedy_order(self, key: str) -> Tuple[List[str], int]:
        """Simulate elimination on the induced graph; return the order and its width.

        LAZY HEAP, not a rescan. Choosing the next vertex by scanning every remaining one
        costs O(n^2) scans and O(n^2 deg^2) fill computations -- measured at 5.9 s to order
        a 4,000-variable chain whose ELIMINATION takes 126 ms. Scores only change near the
        vertex just removed, so stale heap entries are revalidated on pop and pushed back
        if wrong. Same orderings, since ties break on the same (score, degree, name) key."""
        adj = {v: set(n) for v, n in self.adjacency().items()}

        def score(x):
            if key == "min-degree":
                return len(adj[x])
            nb = adj[x]
            return sum(1 for a, b in itertools.combinations(sorted(nb), 2)
                       if b not in adj[a])

        heap = [(score(v), len(adj[v]), v) for v in adj]
        heapq.heapify(heap)
        order, width, gone = [], 0, set()
        while len(order) < len(self.cards):
            sc, dg, v = heapq.heappop(heap)
            if v in gone:
                continue
            true_sc, true_dg = score(v), len(adj[v])
            if (true_sc, true_dg) != (sc, dg):
                heapq.heappush(heap, (true_sc, true_dg, v))   # stale: revalidate
                continue
            width = max(width, len(adj[v]))
            touched = set(adj[v])
            for a, b in itertools.combinations(sorted(adj[v]), 2):
                adj[a].add(b)
                adj[b].add(a)
                touched.add(a); touched.add(b)
            for u in adj[v]:
                adj[u].discard(v)
            del adj[v]
            gone.add(v)
            order.append(v)
            for u in touched:
                if u not in gone:
                    for w in adj[u]:                # fill scores depend on 2-hop structure
                        if w not in gone:
                            heapq.heappush(heap, (score(w), len(adj[w]), w))
                    heapq.heappush(heap, (score(u), len(adj[u]), u))
        return order, width

    def treewidth(self, order: Optional[Sequence[str]] = None) -> int:
        """Width of a given ordering, or the best of min-fill and min-degree."""
        if order is not None:
            adj = {v: set(n) for v, n in self.adjacency().items()}
            width = 0
            for v in order:
                width = max(width, len(adj[v]))
                for a, b in itertools.combinations(sorted(adj[v]), 2):
                    adj[a].add(b)
                    adj[b].add(a)
                for u in adj[v]:
                    adj[u].discard(v)
                del adj[v]
            return width
        return min(self._greedy_order("min-fill")[1], self._greedy_order("min-degree")[1])

    def best_order(self) -> Tuple[List[str], int]:
        of, wf = self._greedy_order("min-fill")
        od, wd = self._greedy_order("min-degree")
        return (of, wf) if wf <= wd else (od, wd)

    # ---------------------------------------------------------------- elimination
    def eliminate(self, mode: str = "min", order: Optional[Sequence[str]] = None,
                  max_table: float = 2e8):
        """Bucket elimination. Returns (value, assignment_or_None, info)."""
        if mode not in ("min", "sum"):
            raise ValueError("mode must be 'min' or 'sum'")
        if order is None:
            order, width = self.best_order()
        else:
            order = list(order)
            width = self.treewidth(order)
        missing = set(self.cards) - set(order)
        if missing:
            raise ValueError(f"elimination order omits {sorted(missing)}")

        # BUCKETS, not a rescanned pool. Each factor is filed under the FIRST of its
        # variables to be eliminated, so every factor is looked at exactly once. Scanning
        # the whole pool at each step instead costs O(n * n_factors) -- measured at 58 s
        # for a 10,000-variable chain whose true cost is linear.
        pos = {v: i for i, v in enumerate(order)}
        buckets: List[List[Factor]] = [[] for _ in order]
        leftovers: List[Factor] = []

        def file(f: Factor):
            if not f.vars:
                leftovers.append(f)
            else:
                buckets[min(pos[v] for v in f.vars)].append(f)

        for f in self.factors:
            file(Factor(f.vars, f.table))
        records = []
        biggest = 1
        for i, v in enumerate(order):
            involved = buckets[i]
            if not involved:
                continue
            # THE PATHWIDTH TRAP: only factors mentioning v are ever combined.
            U: List[str] = []
            for f in involved:
                for u in f.vars:
                    if u not in U:
                        U.append(u)
            U = [u for u in U if u != v] + [v]
            size = int(np.prod([self.cards[u] for u in U]))
            biggest = max(biggest, size)
            if size > max_table:
                raise MemoryError(
                    f"eliminating {v} needs a table of {size:,} entries "
                    f"(clique {U}); ordering width {width}, so cost is "
                    f"d^{width}. This is the treewidth wall, not a bug.")
            combined = np.zeros([self.cards[u] for u in U])
            for f in involved:
                combined = combined + _align(f.vars, f.table, U, self.cards)
            axis = len(U) - 1
            rest = tuple(U[:-1])
            if mode == "min":
                records.append((v, rest, np.argmin(combined, axis=axis)))
                new = np.min(combined, axis=axis)
            else:
                new = logsumexp(combined, axis=axis)
            file(Factor(rest, new))

        total = 0.0
        for f in leftovers:
            total = total + (f.table if f.vars == () else f.table.sum())
        total = float(np.asarray(total).reshape(()))

        assignment = None
        if mode == "min":
            assignment = {}
            for v, rest, arg in reversed(records):
                idx = tuple(assignment[u] for u in rest)
                assignment[v] = int(arg[idx]) if rest else int(arg)
        info = {"order": list(order), "treewidth": int(width),
                "largest_table": int(biggest),
                "n_vars": len(self.cards), "n_factors": len(self.factors)}
        return total, assignment, info

    # ---------------------------------------------------------------- marginals
    def marginals(self, max_table: float = 2e8) -> Dict[str, np.ndarray]:
        """Exact per-variable marginals by eliminating every other variable last-but-one."""
        out = {}
        base, _ = self.best_order()
        for v in self.cards:
            order = [u for u in base if u != v] + [v]
            pool: List[Factor] = [Factor(f.vars, f.table) for f in self.factors]
            for w in order[:-1]:
                involved = [f for f in pool if w in f.vars]
                if not involved:
                    continue
                pool = [f for f in pool if w not in f.vars]
                U: List[str] = []
                for f in involved:
                    for u in f.vars:
                        if u not in U:
                            U.append(u)
                U = [u for u in U if u != w] + [w]
                size = int(np.prod([self.cards[u] for u in U]))
                if size > max_table:
                    raise MemoryError(f"marginal for {v}: table of {size:,} entries")
                comb = np.zeros([self.cards[u] for u in U])
                for f in involved:
                    comb = comb + _align(f.vars, f.table, U, self.cards)
                pool.append(Factor(tuple(U[:-1]), logsumexp(comb, axis=len(U) - 1)))
            acc = np.zeros(self.cards[v])
            for f in pool:
                if f.vars == (v,):
                    acc = acc + f.table
                elif f.vars == ():
                    acc = acc + float(f.table)
                else:
                    acc = acc + f.table.sum()
            acc = acc - logsumexp(acc, axis=0)
            out[v] = np.exp(acc)
        return out

    # ---------------------------------------------------------------- brute force
    def brute_force(self, mode: str = "min"):
        """Reference implementation. Exponential; for verification only."""
        names = list(self.cards)
        best, arg, acc = np.inf, None, []
        for combo in itertools.product(*[range(self.cards[v]) for v in names]):
            a = dict(zip(names, combo))
            tot = 0.0
            for f in self.factors:
                tot += f.table[tuple(a[v] for v in f.vars)]
            if mode == "min":
                if tot < best:
                    best, arg = tot, dict(a)
            else:
                acc.append(tot)
        if mode == "min":
            return best, arg
        acc = np.asarray(acc)
        m = acc.max()
        return float(m + np.log(np.exp(acc - m).sum())), None

    def brute_force_marginals(self) -> Dict[str, np.ndarray]:
        names = list(self.cards)
        out = {v: np.zeros(self.cards[v]) for v in names}
        tot = []
        confs = list(itertools.product(*[range(self.cards[v]) for v in names]))
        w = []
        for combo in confs:
            a = dict(zip(names, combo))
            s = 0.0
            for f in self.factors:
                s += f.table[tuple(a[v] for v in f.vars)]
            w.append(s)
        w = np.asarray(w)
        w = np.exp(w - w.max())
        for combo, wi in zip(confs, w):
            for v, k in zip(names, combo):
                out[v][k] += wi
        return {v: out[v] / out[v].sum() for v in names}


# ---------------------------------------------------------------------------- verify
def random_graph(rng, n=6, card=3, n_factors=8, arity=2) -> FactorGraph:
    g = FactorGraph()
    names = [f"x{i}" for i in range(n)]
    for v in names:
        g.add_var(v, card)
    for v in names:                      # every variable gets a unary, so none is isolated
        g.add_factor([v], rng.normal(size=card))
    for _ in range(n_factors):
        k = min(arity, n)
        vs = list(rng.choice(names, size=k, replace=False))
        g.add_factor(vs, rng.normal(size=(card,) * k))
    return g


def verify(seed: int = 0, trials: int = 40, verbose: bool = True) -> dict:
    """Check elimination against brute force on small random instances."""
    rng = np.random.default_rng(seed)
    e_min, e_sum, e_marg, e_arg = 0.0, 0.0, 0.0, 0
    widths = []
    for t in range(trials):
        n = int(rng.integers(3, 7))
        card = int(rng.integers(2, 4))
        g = random_graph(rng, n=n, card=card,
                         n_factors=int(rng.integers(2, 8)),
                         arity=int(rng.integers(2, 4)))
        widths.append(g.treewidth())

        vmin, arg, info = g.eliminate("min")
        bmin, barg = g.brute_force("min")
        e_min = max(e_min, abs(vmin - bmin))
        # the returned assignment must actually achieve the returned value
        tot = sum(f.table[tuple(arg[v] for v in f.vars)] for f in g.factors)
        e_arg = max(e_arg, abs(tot - vmin))

        vsum, _, _ = g.eliminate("sum")
        bsum, _ = g.brute_force("sum")
        e_sum = max(e_sum, abs(vsum - bsum))

        m = g.marginals()
        bm = g.brute_force_marginals()
        for v in m:
            e_marg = max(e_marg, float(np.max(np.abs(m[v] - bm[v]))))

    res = {"trials": trials, "max_err_min": e_min, "max_err_logZ": e_sum,
           "max_err_marginal": e_marg, "max_err_argmin_consistency": e_arg,
           "treewidths": (min(widths), max(widths))}
    if verbose:
        print(f"  rem.factorgraph.verify  {trials} random instances, "
              f"treewidth {min(widths)}-{max(widths)}")
        print(f"    max |elimination_min  - brute_force|   {e_min:.3e}")
        print(f"    max |elimination_logZ - brute_force|   {e_sum:.3e}")
        print(f"    max |marginal         - brute_force|   {e_marg:.3e}")
        print(f"    max |E(argmin) - reported min|         {e_arg:.3e}")
    return res


def verify_pathwidth_trap(n: int = 60, card: int = 4, verbose: bool = True) -> dict:
    """A chain has treewidth 1. A joint-frontier implementation would build d^n.

    This asserts the largest intermediate table stays at d^2, which is what separates
    bucket elimination from the pathwidth bug the spec warns about."""
    rng = np.random.default_rng(1)
    g = FactorGraph()
    for i in range(n):
        g.add_var(f"x{i}", card)
        g.add_factor([f"x{i}"], rng.normal(size=card))
    for i in range(n - 1):
        g.add_factor([f"x{i}", f"x{i+1}"], rng.normal(size=(card, card)))
    tw = g.treewidth()
    _, _, info = g.eliminate("min")
    ok = info["largest_table"] <= card ** 2
    if verbose:
        print(f"  pathwidth trap: chain of {n} vars, d={card}")
        print(f"    treewidth {tw}   largest intermediate table {info['largest_table']} "
              f"(d^2 = {card**2})   joint frontier would be d^{n} = {card}^{n}")
        print(f"    {'PASS' if ok else 'FAIL'}")
    return {"treewidth": tw, "largest_table": info["largest_table"], "ok": bool(ok)}
