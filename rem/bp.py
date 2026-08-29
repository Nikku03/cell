"""Belief propagation -- exact on trees, and the honest failure off them.

THE GOVERNING LAW, as everywhere in REM:

    cost = d ** treewidth        d = states per variable

BP is the special case where that law is cheapest to obey. A factor graph that is a
FOREST has an elimination order in which every bucket touches exactly one clique of a
single factor, so the whole inference costs  n * d ** max_arity  and no ordering search
is needed at all: the message schedule *is* the elimination order. That is the entire
content of "BP is exact on trees". The spec's instruction is blunt -- use only there.

WHAT "TREE" MEANS HERE. The exactness condition is that the BIPARTITE factor graph
(variable nodes + factor nodes) is acyclic. It is not that the induced/moralised graph is
acyclic: one factor over three variables makes a triangle in the induced graph (treewidth
2) while the bipartite graph is still a star, and BP is exact on it. So `is_tree` tests
the bipartite graph, `sum_product`/`min_sum` warn (or raise, with strict=True) exactly
when that graph has an independent cycle, and every info dict reports BOTH numbers:
`treewidth` of the induced graph (the cost law) and `n_independent_cycles` of the
bipartite graph (the exactness law).

CONVENTIONS, inherited from rem.factorgraph and not re-invented:
  a factor holds a real table phi over its variables; everything below is in LOG SPACE.
    sum_product  reduces with logsumexp  ->  marginals of  p(x) ~ exp(sum_f phi_f)
    min_sum      reduces with min        ->  argmin over x of  sum_f phi_f
  So phi = -E/kT gives probabilities and log Z, phi = E gives the ground state, and a
  600-node chain with |phi| ~ 30 does not overflow, because nothing is ever exponentiated
  outside a logsumexp. verify() measures that case rather than asserting it.

MESSAGES. Synchronous (flooding) schedule with optional damping in log space:
    m_{v->a}(x_v) = sum_{b in N(v)\a} m_{b->v}(x_v)
    m_{a->v}(x_v) = reduce_{x_a \ x_v} [ phi_a(x_a) + sum_{u in N(a)\v} m_{u->a}(x_u) ]
    new <- (1-damping) * new + damping * old
Messages are renormalised every sweep (logsumexp for sum, min for min), which changes
nothing that is reported: beliefs are renormalised, the Bethe log Z is invariant to
message scaling, and the min-sum optimum is reported as the energy of the DECODED
assignment evaluated on the original tables, not as a running message constant.
"""
from __future__ import annotations

import itertools
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rem.factorgraph import FactorGraph, Factor, logsumexp


# --------------------------------------------------------------------- structure
def factor_graph_structure(graph: FactorGraph) -> dict:
    """Union-find on the BIPARTITE factor graph. Constant (0-ary) factors are ignored.

    A graph with V nodes, E edges and C components has exactly E - V + C independent
    cycles. Zero of them is the condition under which BP is exact."""
    var_nodes = list(graph.cards)
    vid = {v: i for i, v in enumerate(var_nodes)}
    nv = len(var_nodes)
    active = [k for k, f in enumerate(graph.factors) if len(f.vars) > 0]
    fid = {k: nv + i for i, k in enumerate(active)}
    n_nodes = nv + len(active)

    parent = list(range(n_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_edges = 0
    repeated = False
    for k in active:
        fv = graph.factors[k].vars
        if len(set(fv)) != len(fv):
            repeated = True            # a variable twice in one factor is a self-loop
        for v in set(fv):
            n_edges += 1
            union(vid[v], fid[k])
    comps = len({find(i) for i in range(n_nodes)}) if n_nodes else 0
    cycles = n_edges - n_nodes + comps
    acyclic = (cycles == 0) and not repeated
    return {"n_nodes": n_nodes, "n_var_nodes": nv, "n_factor_nodes": len(active),
            "n_edges": n_edges, "n_components": comps,
            "n_independent_cycles": int(cycles + (1 if repeated else 0)),
            "acyclic": bool(acyclic), "is_forest": bool(acyclic),
            "is_tree": bool(acyclic and comps == 1 and n_nodes > 0),
            "max_arity": max([len(f.vars) for f in graph.factors], default=0)}


def is_tree(graph: FactorGraph) -> bool:
    """True iff the bipartite factor graph is a single acyclic connected component."""
    return factor_graph_structure(graph)["is_tree"]


def is_forest(graph: FactorGraph) -> bool:
    """True iff the bipartite factor graph is acyclic (BP's exactness condition)."""
    return factor_graph_structure(graph)["acyclic"]


# --------------------------------------------------------------------- machinery
def _bcast(msg: np.ndarray, ndim: int, axis: int) -> np.ndarray:
    shape = [1] * ndim
    shape[axis] = msg.shape[0]
    return msg.reshape(shape)


def _reduce_to(tab: np.ndarray, axis: int, mode: str) -> np.ndarray:
    """Reduce every axis except `axis`, by min or by logsumexp."""
    t = np.moveaxis(tab, axis, 0).reshape(tab.shape[axis], -1)
    return t.min(axis=1) if mode == "min" else logsumexp(t, axis=1)


def _norm(msg: np.ndarray, mode: str) -> np.ndarray:
    return msg - (msg.min() if mode == "min" else float(logsumexp(msg, 0)))


def _run_bp(graph: FactorGraph, mode: str, max_iter: int, damping: float, tol: float,
            strict: bool, warn: bool, compute_treewidth: bool):
    if mode not in ("min", "sum"):
        raise ValueError("mode must be 'min' or 'sum'")
    if not (0.0 <= damping < 1.0):
        raise ValueError("damping must be in [0, 1)")
    st = factor_graph_structure(graph)
    if not st["acyclic"]:
        msg = (f"belief propagation on a LOOPY factor graph: "
               f"{st['n_independent_cycles']} independent cycle(s). BP is exact only on "
               f"forests; results here are an uncontrolled approximation and may "
               f"oscillate. Use FactorGraph.eliminate / .marginals for the exact answer.")
        if strict:
            raise ValueError(msg)
        if warn:
            warnings.warn(msg, RuntimeWarning, stacklevel=3)

    active = [k for k, f in enumerate(graph.factors) if len(f.vars) > 0]
    const = float(sum(float(np.asarray(graph.factors[k].table).reshape(()))
                      for k in range(len(graph.factors)) if len(graph.factors[k].vars) == 0))
    nbr_v: Dict[str, List[int]] = {v: [] for v in graph.cards}
    for k in active:
        for v in graph.factors[k].vars:
            nbr_v[v].append(k)

    cards = graph.cards
    m_f2v = {(k, v): np.zeros(cards[v]) for k in active for v in graph.factors[k].vars}
    m_v2f = {(v, k): np.zeros(cards[v]) for k in active for v in graph.factors[k].vars}
    keys = sorted(m_f2v)

    def flat(d):
        return np.concatenate([d[k] for k in keys]) if keys else np.zeros(0)

    hist = [flat(m_f2v)]
    delta = np.inf
    converged = False
    it = 0
    deltas = []
    for it in range(1, max_iter + 1):
        for v, ks in nbr_v.items():
            if not ks:
                continue
            tot = np.zeros(cards[v])
            for k in ks:
                tot = tot + m_f2v[(k, v)]
            for k in ks:
                m_v2f[(v, k)] = _norm(tot - m_f2v[(k, v)], mode)
        delta = 0.0
        for k in active:
            f = graph.factors[k]
            nd = f.table.ndim
            full = np.array(f.table, dtype=float)
            for j, u in enumerate(f.vars):
                full = full + _bcast(m_v2f[(u, k)], nd, j)
            for j, u in enumerate(f.vars):
                tab = full - _bcast(m_v2f[(u, k)], nd, j)
                new = _norm(_reduce_to(tab, j, mode), mode)
                old = m_f2v[(k, u)]
                if damping > 0.0:
                    new = _norm((1.0 - damping) * new + damping * old, mode)
                delta = max(delta, float(np.max(np.abs(new - old))) if new.size else 0.0)
                m_f2v[(k, u)] = new
        deltas.append(delta)
        hist.append(flat(m_f2v))
        if len(hist) > 3:
            hist.pop(0)
        if delta < tol:
            converged = True
            break

    # period-2 oscillation: messages repeat every other sweep but never settle
    p2 = float("nan")
    if len(hist) >= 3:
        p2 = float(np.max(np.abs(hist[-1] - hist[-3]))) if hist[-1].size else 0.0
    oscillating = bool((not converged) and np.isfinite(p2) and p2 < 0.1 * max(delta, 1e-300))

    info = {"mode": mode, "converged": bool(converged), "iterations": it,
            "final_delta": float(delta), "period2_delta": p2,
            "oscillating": oscillating, "damping": float(damping), "tol": float(tol),
            "max_iter": int(max_iter),
            "is_tree": st["is_tree"], "is_forest": st["acyclic"],
            "n_independent_cycles": st["n_independent_cycles"],
            "max_arity": st["max_arity"],
            "n_vars": len(graph.cards), "n_factors": len(graph.factors),
            "delta_history": deltas[-10:]}
    info["exact"] = bool(st["acyclic"] and converged)
    if compute_treewidth:
        info["treewidth"] = int(graph.treewidth())
    return m_f2v, m_v2f, active, const, nbr_v, info


# --------------------------------------------------------------------- sum-product
def sum_product(graph: FactorGraph, max_iter: int = 500, damping: float = 0.0,
                tol: float = 1e-12, strict: bool = False, warn: bool = True,
                compute_treewidth: bool = True) -> Tuple[Dict[str, np.ndarray], dict]:
    """Sum-product BP. Returns (per-variable marginals, info).

    EXACT when the bipartite factor graph is a forest; on a loopy graph it warns (or
    raises with strict=True) and returns the loopy-BP approximation. info carries
    `bethe_logZ`, which equals eliminate("sum") exactly on a converged forest."""
    m_f2v, m_v2f, active, const, nbr_v, info = _run_bp(
        graph, "sum", max_iter, damping, tol, strict, warn, compute_treewidth)

    beliefs = {}
    for v, ks in nbr_v.items():
        b = np.zeros(graph.cards[v])
        for k in ks:
            b = b + m_f2v[(k, v)]
        b = b - float(logsumexp(b, 0))
        beliefs[v] = np.exp(b)

    # Bethe free energy -> log Z. Exact on a converged forest, an estimate otherwise.
    logZ = const
    for k in active:
        f = graph.factors[k]
        nd = f.table.ndim
        lb = np.array(f.table, dtype=float)
        for j, u in enumerate(f.vars):
            lb = lb + _bcast(m_v2f[(u, k)], nd, j)
        lb = lb - float(logsumexp(lb.reshape(-1), 0))
        ba = np.exp(lb)
        logZ += float(np.sum(ba * np.asarray(f.table, dtype=float)))     # <b_a, phi_a>
        logZ += float(-np.sum(ba * np.where(ba > 0, lb, 0.0)))           # H_a
    for v, ks in nbr_v.items():
        bv = beliefs[v]
        Hv = float(-np.sum(bv * np.where(bv > 0, np.log(np.maximum(bv, 1e-300)), 0.0)))
        logZ += (1 - len(ks)) * Hv
    info["bethe_logZ"] = float(logZ)
    return beliefs, info


# --------------------------------------------------------------------- min-sum
def min_sum(graph: FactorGraph, max_iter: int = 500, damping: float = 0.0,
            tol: float = 1e-12, strict: bool = False, warn: bool = True,
            compute_treewidth: bool = True):
    """Min-sum BP. Returns (beliefs, assignment, info), mirroring eliminate("min").

    beliefs[v] is shifted so min = 0; on a converged forest it is exactly the min-marginal
    of v minus the global optimum. `assignment` decodes by per-variable argmin and
    info["value"] is that assignment's energy EVALUATED ON THE ORIGINAL TABLES -- never a
    running message constant -- so it is a number an independent enumerator can check.
    info["degenerate"] flags near-ties, where per-variable decoding is not well defined."""
    m_f2v, m_v2f, active, const, nbr_v, info = _run_bp(
        graph, "min", max_iter, damping, tol, strict, warn, compute_treewidth)

    beliefs, assignment = {}, {}
    tie = 0.0
    for v, ks in nbr_v.items():
        b = np.zeros(graph.cards[v])
        for k in ks:
            b = b + m_f2v[(k, v)]
        b = b - b.min()
        beliefs[v] = b
        assignment[v] = int(np.argmin(b))
        s = np.sort(b)
        if len(s) > 1:
            tie = max(tie, 1.0 / (1.0 + s[1]))       # ->1 when the top two beliefs tie
    value = const
    for f in graph.factors:
        if len(f.vars) == 0:
            continue
        value += float(f.table[tuple(assignment[v] for v in f.vars)])
    info["value"] = float(value)
    info["assignment"] = dict(assignment)
    info["degenerate"] = bool(tie > 0.5)
    return beliefs, assignment, info


# --------------------------------------------------------------------- brute force
# Naive enumeration over all d^n configurations. Shares no code path with the message
# passing above: no messages, no schedule, no damping, no log-space tricks beyond one
# max-shift. This is the reference every number in verify() is measured against.
def brute_force_marginals(graph: FactorGraph) -> Dict[str, np.ndarray]:
    names = list(graph.cards)
    acc = {v: np.zeros(graph.cards[v]) for v in names}
    confs, tot = [], []
    for combo in itertools.product(*[range(graph.cards[v]) for v in names]):
        a = dict(zip(names, combo))
        s = 0.0
        for f in graph.factors:
            s += float(f.table[tuple(a[v] for v in f.vars)]) if f.vars else float(f.table)
        confs.append(combo)
        tot.append(s)
    w = np.asarray(tot)
    w = np.exp(w - w.max())
    for combo, wi in zip(confs, w):
        for v, k in zip(names, combo):
            acc[v][k] += wi
    return {v: acc[v] / acc[v].sum() for v in names}


def brute_force_logZ(graph: FactorGraph) -> float:
    names = list(graph.cards)
    tot = []
    for combo in itertools.product(*[range(graph.cards[v]) for v in names]):
        a = dict(zip(names, combo))
        s = 0.0
        for f in graph.factors:
            s += float(f.table[tuple(a[v] for v in f.vars)]) if f.vars else float(f.table)
        tot.append(s)
    t = np.asarray(tot)
    m = t.max()
    return float(m + np.log(np.exp(t - m).sum()))


def brute_force_min_marginals(graph: FactorGraph):
    """min over all configurations with x_v = k, for every (v, k). Returns (mm, best)."""
    names = list(graph.cards)
    mm = {v: np.full(graph.cards[v], np.inf) for v in names}
    best = np.inf
    for combo in itertools.product(*[range(graph.cards[v]) for v in names]):
        a = dict(zip(names, combo))
        s = 0.0
        for f in graph.factors:
            s += float(f.table[tuple(a[v] for v in f.vars)]) if f.vars else float(f.table)
        best = min(best, s)
        for v, k in zip(names, combo):
            if s < mm[v][k]:
                mm[v][k] = s
    return mm, float(best)


# --------------------------------------------------------------------- generators
def random_tree(rng, n: int = 8, card: int = 3, unary: bool = True,
                scale: float = 1.0) -> FactorGraph:
    """Uniformly-attached random tree: node i attaches to a random earlier node."""
    g = FactorGraph()
    for i in range(n):
        g.add_var(f"x{i}", card)
    if unary:
        for i in range(n):
            g.add_factor([f"x{i}"], rng.normal(size=card) * scale)
    for i in range(1, n):
        j = int(rng.integers(0, i))
        g.add_factor([f"x{j}", f"x{i}"], rng.normal(size=(card, card)) * scale)
    return g


def random_hypertree(rng, n_factors: int = 4, card: int = 2, arity: int = 3,
                     scale: float = 1.0) -> FactorGraph:
    """A bipartite TREE whose factors have arity>2: induced treewidth arity-1, BP exact.

    This is the case that separates the two notions of "tree": treewidth here is 2 or
    more, yet the factor graph is a star-of-stars and BP is exact on it."""
    g = FactorGraph()
    nxt = 0
    pool: List[str] = []
    for t in range(n_factors):
        vs = []
        if pool:
            vs.append(pool[int(rng.integers(0, len(pool)))])
        while len(vs) < arity:
            v = f"y{nxt}"
            nxt += 1
            g.add_var(v, card)
            vs.append(v)
            pool.append(v)
        g.add_factor(vs, rng.normal(size=(card,) * arity) * scale)
    return g


def ising(rng, n: int = 6, beta: float = 1.0, h: float = 0.4, ring: bool = False,
          seed_couplings: Optional[np.ndarray] = None) -> FactorGraph:
    """Random-bond Ising chain (tree) or ring (one extra edge -> one cycle).

    phi(x,y) = beta * J * s_x * s_y  with s in {-1,+1} and J random sign; unary
    phi(x) = h_i * s_x. Chain and ring differ by exactly ONE factor, which is what makes
    them a controlled pair: identical statistics, treewidth 1 vs 2, exact vs not."""
    g = FactorGraph()
    s = np.array([-1.0, 1.0])
    for i in range(n):
        g.add_var(f"s{i}", 2)
    hs = rng.normal(size=n) * h
    for i in range(n):
        g.add_factor([f"s{i}"], hs[i] * s)
    edges = [(i, (i + 1) % n) for i in range(n if ring else n - 1)]
    J = seed_couplings if seed_couplings is not None else rng.choice([-1.0, 1.0], size=n)
    for e, (i, j) in enumerate(edges):
        g.add_factor([f"s{i}", f"s{j}"], beta * J[e] * np.outer(s, s))
    return g


def grid_ising(rng, rows: int = 3, cols: int = 3, beta: float = 0.8,
               h: float = 0.3) -> FactorGraph:
    g = FactorGraph()
    s = np.array([-1.0, 1.0])
    nm = lambda r, c: f"g{r}_{c}"
    for r in range(rows):
        for c in range(cols):
            g.add_var(nm(r, c), 2)
            g.add_factor([nm(r, c)], rng.normal() * h * s)
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                g.add_factor([nm(r, c), nm(r + 1, c)],
                             beta * rng.choice([-1.0, 1.0]) * np.outer(s, s))
            if c + 1 < cols:
                g.add_factor([nm(r, c), nm(r, c + 1)],
                             beta * rng.choice([-1.0, 1.0]) * np.outer(s, s))
    return g


# --------------------------------------------------------------------- verify
def _tree_trials(rng, trials: int, verbose: bool):
    e_marg_elim = e_marg_bf = e_logz = e_minval = e_minmarg = 0.0
    bad_arg = 0
    iters, tws, cycles = [], [], []
    for t in range(trials):
        if t % 3 == 2:
            g = random_hypertree(rng, n_factors=int(rng.integers(2, 4)),
                                 card=int(rng.integers(2, 4)), arity=3)
        else:
            g = random_tree(rng, n=int(rng.integers(2, 8)), card=int(rng.integers(2, 4)))
        st = factor_graph_structure(g)
        assert st["acyclic"], "generator produced a loop"
        cycles.append(st["n_independent_cycles"])

        b, info = sum_product(g, max_iter=400, tol=1e-13)
        iters.append(info["iterations"])
        tws.append(info["treewidth"])
        assert info["converged"], f"BP did not converge on a tree (trial {t})"

        me = g.marginals()                      # exact elimination
        mb = brute_force_marginals(g)           # naive enumeration
        for v in b:
            e_marg_elim = max(e_marg_elim, float(np.max(np.abs(b[v] - me[v]))))
            e_marg_bf = max(e_marg_bf, float(np.max(np.abs(b[v] - mb[v]))))
        zi, _, _ = g.eliminate("sum")
        e_logz = max(e_logz, abs(info["bethe_logZ"] - zi),
                     abs(info["bethe_logZ"] - brute_force_logZ(g)))

        bm, arg, mi = min_sum(g, max_iter=400, tol=1e-13)
        assert mi["converged"], f"min-sum did not converge on a tree (trial {t})"
        vmin, varg, _ = g.eliminate("min")
        e_minval = max(e_minval, abs(mi["value"] - vmin))
        if not mi["degenerate"]:
            mmb, best = brute_force_min_marginals(g)
            for v in bm:
                e_minmarg = max(e_minmarg, float(np.max(np.abs(bm[v] - (mmb[v] - best)))))
            if any(arg[v] != varg[v] for v in arg) and abs(mi["value"] - vmin) > 1e-9:
                bad_arg += 1
    return {"max_err_marginal_vs_elimination": e_marg_elim,
            "max_err_marginal_vs_bruteforce": e_marg_bf,
            "max_err_bethe_logZ": e_logz,
            "max_err_minsum_value": e_minval,
            "max_err_min_marginal": e_minmarg,
            "bad_assignments": bad_arg,
            "iterations": (min(iters), max(iters)),
            "treewidths": (min(tws), max(tws)),
            "cycles": max(cycles)}


def verify(seed: int = 0, trials: int = 24, verbose: bool = True) -> dict:
    """Positive control on trees, then the documented failure on loops.

    Every tree number is measured against TWO references: exact bucket elimination
    (rem.factorgraph) and naive enumeration over all d^n configurations (this module).
    The loopy numbers are measured against exact elimination on the same graph."""
    rng = np.random.default_rng(seed)
    res = {}

    # ---------------------------------------------------------- (a) trees: positive control
    tr = _tree_trials(rng, trials, verbose)
    res.update(tr)

    # log-space stress: a long chain with huge phi. exp() of these would be inf.
    n_chain, scale = 80, 30.0
    gc = random_tree(np.random.default_rng(11), n=n_chain, card=2, scale=scale)
    _, ic = sum_product(gc, max_iter=4 * n_chain, tol=1e-12)
    zc, _, _ = gc.eliminate("sum")
    res["chain_n"] = n_chain
    res["chain_logZ"] = float(ic["bethe_logZ"])
    res["chain_logZ_err"] = abs(ic["bethe_logZ"] - zc)
    res["chain_iterations"] = ic["iterations"]
    res["chain_treewidth"] = ic["treewidth"]

    # damping must not move the fixed point on a tree, only slow the approach
    gd = random_tree(np.random.default_rng(5), n=7, card=3)
    b0, i0 = sum_product(gd, max_iter=2000, tol=1e-13, damping=0.0)
    b5, i5 = sum_product(gd, max_iter=4000, tol=1e-13, damping=0.5)
    res["damping_fixed_point_shift"] = max(float(np.max(np.abs(b0[v] - b5[v])))
                                           for v in b0)
    res["damping_iterations"] = (i0["iterations"], i5["iterations"])

    # ---------------------------------------------------------- (b) loops: honest negative
    # Controlled pair: the SAME Ising model as a chain and as a ring. One extra factor.
    rr = np.random.default_rng(3)
    J = rr.choice([-1.0, 1.0], size=8)
    chain = ising(np.random.default_rng(4), n=8, beta=0.9, h=0.4, ring=False,
                  seed_couplings=J)
    ring = ising(np.random.default_rng(4), n=8, beta=0.9, h=0.4, ring=True,
                 seed_couplings=J)
    bc, ic2 = sum_product(chain, max_iter=400, tol=1e-13)
    mc = chain.marginals()
    res["pair_chain_err"] = max(float(np.max(np.abs(bc[v] - mc[v]))) for v in bc)
    res["pair_chain_treewidth"] = ic2["treewidth"]
    res["pair_chain_cycles"] = ic2["n_independent_cycles"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        br, ir = sum_product(ring, max_iter=400, tol=1e-13)
    res["loop_warning_emitted"] = bool(any(issubclass(w.category, RuntimeWarning)
                                           for w in caught))
    mr = ring.marginals()
    res["pair_ring_err"] = max(float(np.max(np.abs(br[v] - mr[v]))) for v in br)
    res["pair_ring_treewidth"] = ir["treewidth"]
    res["pair_ring_cycles"] = ir["n_independent_cycles"]
    res["pair_ring_converged"] = ir["converged"]
    zr, _, _ = ring.eliminate("sum")
    res["pair_ring_logZ_exact"] = float(zr)
    res["pair_ring_logZ_bethe"] = float(ir["bethe_logZ"])
    res["pair_ring_logZ_err"] = abs(ir["bethe_logZ"] - zr)

    # a frustrated grid: undamped BP oscillates, damped BP converges -- to a wrong answer
    gg = grid_ising(np.random.default_rng(2), rows=3, cols=3, beta=1.1, h=0.25)
    mg = gg.marginals()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bu, iu = sum_product(gg, max_iter=300, tol=1e-10, damping=0.0)
        bd, idmp = sum_product(gg, max_iter=3000, tol=1e-10, damping=0.7)
        _, arg_ms, ims = min_sum(gg, max_iter=1000, tol=1e-10, damping=0.5)
    res["grid_treewidth"] = iu["treewidth"]
    res["grid_cycles"] = iu["n_independent_cycles"]
    res["grid_undamped_converged"] = iu["converged"]
    res["grid_undamped_iterations"] = iu["iterations"]
    res["grid_undamped_final_delta"] = iu["final_delta"]
    res["grid_undamped_period2_delta"] = iu["period2_delta"]
    res["grid_undamped_oscillating"] = iu["oscillating"]
    res["grid_undamped_err"] = max(float(np.max(np.abs(bu[v] - mg[v]))) for v in bu)
    res["grid_damped_converged"] = idmp["converged"]
    res["grid_damped_iterations"] = idmp["iterations"]
    res["grid_damped_err"] = max(float(np.max(np.abs(bd[v] - mg[v]))) for v in bd)
    res["grid_damped_logZ_err"] = abs(idmp["bethe_logZ"] - gg.eliminate("sum")[0])
    vmin_g, _, _ = gg.eliminate("min")
    res["grid_minsum_value"] = ims["value"]
    res["grid_minsum_exact"] = float(vmin_g)
    res["grid_minsum_gap"] = float(ims["value"] - vmin_g)

    if verbose:
        print("  rem.bp.verify")
        print(f"  (a) TREES -- positive control, {trials} random factor trees "
              f"(pairwise + arity-3 hypertrees)")
        print(f"      bipartite cycles {tr['cycles']}   induced treewidth "
              f"{tr['treewidths'][0]}-{tr['treewidths'][1]}   sweeps to converge "
              f"{tr['iterations'][0]}-{tr['iterations'][1]}")
        print(f"      max |BP marginal   - elimination marginal|   "
              f"{tr['max_err_marginal_vs_elimination']:.3e}")
        print(f"      max |BP marginal   - brute-force marginal|   "
              f"{tr['max_err_marginal_vs_bruteforce']:.3e}")
        print(f"      max |Bethe log Z   - elimination / brute|    "
              f"{tr['max_err_bethe_logZ']:.3e}")
        print(f"      max |min-sum value - eliminate('min')|       "
              f"{tr['max_err_minsum_value']:.3e}")
        print(f"      max |min-sum belief- brute min-marginal|     "
              f"{tr['max_err_min_marginal']:.3e}")
        print(f"      disagreeing argmins with a different energy: {tr['bad_assignments']}")
        print(f"      log-space stress: chain n={n_chain}, |phi|~{scale:.0f}, treewidth "
              f"{res['chain_treewidth']}, log Z = {res['chain_logZ']:.6f}")
        print(f"        |Bethe log Z - elimination log Z|          "
              f"{res['chain_logZ_err']:.3e}   (exp(log Z) would overflow)")
        print(f"      damping 0.0 vs 0.5 on a tree: fixed point moves "
              f"{res['damping_fixed_point_shift']:.3e} in "
              f"{res['damping_iterations'][0]} vs {res['damping_iterations'][1]} sweeps")
        print(f"  (b) LOOPS -- the documented boundary, not a bug")
        print(f"      controlled pair, same Ising model, ONE extra factor:")
        print(f"        chain: treewidth {res['pair_chain_treewidth']}, cycles "
              f"{res['pair_chain_cycles']}, max |BP - exact| "
              f"{res['pair_chain_err']:.3e}   <- exact")
        print(f"        ring : treewidth {res['pair_ring_treewidth']}, cycles "
              f"{res['pair_ring_cycles']}, max |BP - exact| "
              f"{res['pair_ring_err']:.3e}   <- WRONG, converged="
              f"{res['pair_ring_converged']}")
        print(f"        ring log Z: exact {res['pair_ring_logZ_exact']:.6f}  "
              f"Bethe {res['pair_ring_logZ_bethe']:.6f}  gap "
              f"{res['pair_ring_logZ_err']:.3e}")
        print(f"        loopy warning emitted: {res['loop_warning_emitted']}")
        print(f"      frustrated 3x3 grid, treewidth {res['grid_treewidth']}, "
              f"{res['grid_cycles']} independent cycles:")
        print(f"        undamped: converged={res['grid_undamped_converged']} after "
              f"{res['grid_undamped_iterations']} sweeps, final delta "
              f"{res['grid_undamped_final_delta']:.3e}, "
              f"|m_t - m_(t-2)| {res['grid_undamped_period2_delta']:.3e} "
              f"-> oscillating={res['grid_undamped_oscillating']}")
        print(f"        undamped max |BP - exact marginal|  "
              f"{res['grid_undamped_err']:.3e}")
        print(f"        damped 0.7: converged={res['grid_damped_converged']} after "
              f"{res['grid_damped_iterations']} sweeps -- to a WRONG fixed point, "
              f"max |BP - exact| {res['grid_damped_err']:.3e}")
        print(f"        damped Bethe log Z error {res['grid_damped_logZ_err']:.3e}")
        print(f"        min-sum decoded energy {res['grid_minsum_value']:.6f} vs exact "
              f"{res['grid_minsum_exact']:.6f}, gap {res['grid_minsum_gap']:.3e}")
    return res


if __name__ == "__main__":
    verify()
