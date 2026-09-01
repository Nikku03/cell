"""Deliverable 1: the cost model, and the input hygiene that has to run before it.

WHY THIS BLOCKS EVERYTHING ELSE. The cost of a bucket in an elimination schedule is the
product of THE ACTUAL DOMAIN SIZES of its members. It is not d^(w+1) for a single
representative d. That substitution has been made twice in this project and moved answers
by ~80 orders both times, in the direction that makes an intractable problem look
tractable. So the primitive is:

    bucket_cost = sum(log10(domain[v]) for v in bag)          # log10 of the true state count
    memory_bytes_log10 = bucket_cost + log10(8)

A single representative d is only correct when every domain is equal, which is exactly the
case the rung system in this package is built to destroy.

AND THE HYGIENE HAS TO COME FIRST, because a genome-scale model contains things that are
not biochemistry. iJO1366's BIOMASS pseudo-reaction has 106 participants. It is an
objective function for flux balance, not a reaction, and as a factor it is a 106-node
clique that dominates every width calculation downstream of it. Real reactions in these
models max out around 13 participants with a median of 4. So any reaction of arity above
ARITY_CAP is flagged and excluded BY DEFAULT, with a printed warning naming what was
dropped -- never silently, because a silently dropped clique is indistinguishable from a
model that was always sparse.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN.
=================================================================================================

G1a  On iJO1366, currency-buffered, BIOMASS excluded:
         min-fill upper bound on treewidth = 31
         min-degree upper bound            = 39
         degeneracy (rigorous LOWER bound) = 6
     A CAVEAT THAT IS PART OF THE GATE, not an excuse added afterwards: "currency-buffered"
     names an operation but not a set, and the answer depends on which metabolites are
     called currency. The set used here is written out in CURRENCY below and the run
     reports its sensitivity to that choice, so a mismatch can be diagnosed as a different
     buffer set rather than a different algorithm.

G1b  Arity hygiene. iJO1366 must contain at least one reaction of arity > 20 (the spec says
     BIOMASS has 106 participants); iMB155 likewise (54). After exclusion the maximum arity
     of what remains must be small -- the spec says real reactions max out at 13 with median
     4. The run reports max and median arity before and after, and the count excluded.

G1c  Cost-model self-consistency, the point of the whole module. On a bag with mixed
     domains, sum(log10 d_v) must differ from the single-d formula, and the run prints the
     size of the discrepancy on a realistic mixed bag so the failure mode is visible rather
     than argued.

G1d  Cross-check against the existing engine. The width computed here must agree with
     rem.factorgraph.FactorGraph.treewidth on random graphs where both can run. This exists
     because two independent implementations of the same quantity is the cheapest control
     available, and this module deliberately reimplements the orderings rather than
     importing them (FactorGraph carries dense tables that a 1800-node metabolic graph
     cannot afford).
"""
from __future__ import annotations

import heapq
import itertools
import json
import math
from typing import Dict, Iterable, List, Sequence, Set, Tuple

ARITY_CAP = 20

# The currency set, written out because the answer depends on it and a name is not a set.
# Compartment suffixes (_c, _p, _e) are stripped before matching.
CURRENCY = frozenset("""
h2o h atp adp amp pi ppi nad nadh nadp nadph co2 o2 nh4 coa gtp gdp gmp
utp udp ump ctp cdp cmp fad fadh2 fmn fmnh2 q8 q8h2 mqn8 mql8 2dmmq8 2dmmql8
so4 so3 h2s na1 k cl ca2 mg2 fe2 fe3 mn2 zn2 cu2 cobalt2 ni2 mobd
thf mlthf methf 10fthf dhf h2o2 o2s
""".split())


# -------------------------------------------------------------------------------------
# the cost primitive
# -------------------------------------------------------------------------------------

def bucket_cost(bag: Iterable[str], domain: Dict[str, int]) -> float:
    """log10 of the true number of states in a bucket: sum of log10 of each domain size."""
    return float(sum(math.log10(max(1, int(domain[v]))) for v in bag))


def memory_bytes_log10(bag: Iterable[str], domain: Dict[str, int]) -> float:
    return bucket_cost(bag, domain) + math.log10(8.0)


def naive_uniform_cost(bag: Sequence[str], domain: Dict[str, int]) -> float:
    """THE MISTAKE, kept so the gate can measure it: d^|bag| with one representative d."""
    if not bag:
        return 0.0
    d = max(int(domain[v]) for v in bag)
    return len(bag) * math.log10(d)


# -------------------------------------------------------------------------------------
# model loading and hygiene
# -------------------------------------------------------------------------------------

def load_bigg(path: str) -> Dict[str, Dict[str, float]]:
    """BiGG / COBRA JSON -> {reaction_id: {metabolite_id: coefficient}}."""
    with open(path) as fh:
        m = json.load(fh)
    out = {}
    for r in m["reactions"]:
        out[r["id"]] = dict(r["metabolites"])
    return out


def _base(met: str) -> str:
    for suf in ("_c", "_p", "_e", "_m", "_n", "_r", "_x", "_g", "_v", "_h", "_u"):
        if met.endswith(suf):
            return met[: -len(suf)]
    return met


def arity_report(reactions: Dict[str, Dict[str, float]], cap: int = ARITY_CAP) -> dict:
    ar = {r: len(m) for r, m in reactions.items()}
    over = sorted(((n, r) for r, n in ar.items() if n > cap), reverse=True)
    kept = [n for r, n in ar.items() if n <= cap]
    vals = sorted(ar.values())
    med = vals[len(vals) // 2] if vals else 0
    kept_sorted = sorted(kept)
    return {"n_reactions": len(ar), "max_arity": max(ar.values()) if ar else 0,
            "median_arity": med, "n_over_cap": len(over), "over": over[:10],
            "max_arity_kept": max(kept) if kept else 0,
            "median_arity_kept": kept_sorted[len(kept_sorted) // 2] if kept_sorted else 0}


def build_graph(reactions: Dict[str, Dict[str, float]], currency: Iterable[str] = CURRENCY,
                cap: int = ARITY_CAP, verbose: bool = True) -> Dict[str, Set[str]]:
    """Moral graph of the reaction hypergraph: a clique per reaction, currency removed.

    Reactions above `cap` participants are EXCLUDED and named, because a pseudo-reaction
    clique is not biochemistry and silently keeping it makes every width meaningless.
    """
    cur = set(currency)
    rep = arity_report(reactions, cap)
    if verbose and rep["n_over_cap"]:
        names = ", ".join(f"{r}({n})" for n, r in rep["over"][:5])
        print(f"  WARNING arity hygiene: excluding {rep['n_over_cap']} reaction(s) with "
              f"arity > {cap}: {names}", flush=True)
    adj: Dict[str, Set[str]] = {}
    for rid, mets in reactions.items():
        if len(mets) > cap:
            continue
        nodes = sorted({m for m in mets if _base(m) not in cur})
        for v in nodes:
            adj.setdefault(v, set())
        for a, b in itertools.combinations(nodes, 2):
            adj[a].add(b)
            adj[b].add(a)
    return adj


# -------------------------------------------------------------------------------------
# width bounds -- reimplemented rather than imported, and cross-checked in G1d
# -------------------------------------------------------------------------------------

def _greedy_width(adj: Dict[str, Set[str]], key: str) -> Tuple[List[str], int]:
    a = {v: set(n) for v, n in adj.items()}

    def fill(v):
        nb = a[v]
        if key == "min-degree":
            return len(nb)
        miss = 0
        for x, y in itertools.combinations(sorted(nb), 2):
            if y not in a[x]:
                miss += 1
        return miss

    heap = [(fill(v), len(a[v]), v) for v in a]
    heapq.heapify(heap)
    order, width, gone = [], 0, set()
    while heap:
        _s, _d, v = heapq.heappop(heap)
        if v in gone or v not in a:
            continue
        cur = fill(v)
        if cur != _s:                       # stale key; reinsert with the fresh score
            heapq.heappush(heap, (cur, len(a[v]), v))
            continue
        nb = sorted(a[v])
        width = max(width, len(nb))
        for x, y in itertools.combinations(nb, 2):
            a[x].add(y)
            a[y].add(x)
        for u in nb:
            a[u].discard(v)
        del a[v]
        gone.add(v)
        order.append(v)
        for u in nb:
            if u in a:
                heapq.heappush(heap, (fill(u), len(a[u]), u))
    return order, width


def degeneracy(adj: Dict[str, Set[str]]) -> int:
    """Degeneracy: a RIGOROUS LOWER bound on treewidth (every graph has tw >= degeneracy)."""
    a = {v: set(n) for v, n in adj.items()}
    heap = [(len(a[v]), v) for v in a]
    heapq.heapify(heap)
    best = 0
    while heap:
        d, v = heapq.heappop(heap)
        if v not in a:
            continue
        if d != len(a[v]):
            heapq.heappush(heap, (len(a[v]), v))
            continue
        best = max(best, d)
        for u in a[v]:
            a[u].discard(v)
            heapq.heappush(heap, (len(a[u]), u))
        del a[v]
    return best


def width_bounds(adj: Dict[str, Set[str]]) -> dict:
    _o1, wf = _greedy_width(adj, "min-fill")
    _o2, wd = _greedy_width(adj, "min-degree")
    return {"n_nodes": len(adj), "n_edges": sum(len(v) for v in adj.values()) // 2,
            "min_fill": wf, "min_degree": wd, "degeneracy": degeneracy(adj)}


# -------------------------------------------------------------------------------------
# gates
# -------------------------------------------------------------------------------------

IJO = "data/models/iJO1366.json"
IMB = "data/models/iMB155_noUnqATP_lipdiomics_wPUNP5_noNBtransport.json"

EXPECT = {"min_fill": 31, "min_degree": 39, "degeneracy": 6}


def _gate(name, got, want, tol=0):
    ok = abs(got - want) <= tol
    print(f"  {name:<34s} expected {want:>8}   measured {got:>8}   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def verify(verbose: bool = True) -> dict:
    out = {}
    print("=" * 92)
    print("G1b  ARITY HYGIENE")
    print("=" * 92)
    for tag, path, want_over in (("iJO1366", IJO, 106), ("iMB155", IMB, 54)):
        rx = load_bigg(path)
        rep = arity_report(rx)
        print(f"  {tag}: {rep['n_reactions']} reactions   max arity {rep['max_arity']} "
              f"(expected a pseudo-reaction near {want_over})   median {rep['median_arity']}")
        print(f"          over cap({ARITY_CAP}): {rep['n_over_cap']}  -> "
              f"{', '.join(f'{r}={n}' for n, r in rep['over'][:4])}")
        print(f"          after exclusion: max arity {rep['max_arity_kept']}, "
              f"median {rep['median_arity_kept']}")
        out[f"arity_{tag}"] = rep

    print("\n" + "=" * 92)
    print("G1a  iJO1366 WIDTH BOUNDS, currency-buffered, arity-filtered")
    print("=" * 92)
    rx = load_bigg(IJO)
    adj = build_graph(rx)
    w = width_bounds(adj)
    print(f"  graph: {w['n_nodes']} metabolites, {w['n_edges']} edges "
          f"({len(CURRENCY)} currency species buffered out)")
    ok = [_gate("min-fill upper bound", w["min_fill"], EXPECT["min_fill"]),
          _gate("min-degree upper bound", w["min_degree"], EXPECT["min_degree"]),
          _gate("degeneracy (rigorous lower)", w["degeneracy"], EXPECT["degeneracy"])]
    out["ijo"] = w
    out["G1a"] = all(ok)

    # sensitivity to the buffer set, because "currency-buffered" names an operation, not a set
    print("\n  SENSITIVITY TO THE BUFFER SET (the gate value depends on it):")
    for label, cur in (("no buffering", frozenset()),
                       ("core only (h2o,h,atp,adp,pi,nad(p)(h))",
                        frozenset("h2o h atp adp pi nad nadh nadp nadph".split())),
                       ("as declared", CURRENCY)):
        a = build_graph(rx, currency=cur, verbose=False)
        ww = width_bounds(a)
        print(f"    {label:<42s} n={ww['n_nodes']:5d}  min-fill {ww['min_fill']:3d}  "
              f"min-degree {ww['min_degree']:3d}  degeneracy {ww['degeneracy']:3d}")

    print("\n" + "=" * 92)
    print("G1c  COST MODEL: true product of domains vs one representative d")
    print("=" * 92)
    bag = [f"v{i}" for i in range(12)]
    dom = {"v0": 600, "v1": 40, "v2": 20, "v3": 8, "v4": 8, "v5": 4,
           "v6": 4, "v7": 3, "v8": 3, "v9": 2, "v10": 2, "v11": 2}
    true, naive = bucket_cost(bag, dom), naive_uniform_cost(bag, dom)
    print(f"  mixed bag of {len(bag)} species, domains {sorted(dom.values(), reverse=True)}")
    print(f"    true   sum(log10 d_v)      = 10^{true:.2f} states")
    print(f"    naive  d_max^|bag|          = 10^{naive:.2f} states")
    print(f"    OVERSTATEMENT              = {naive - true:.1f} orders of magnitude")
    out["G1c"] = naive - true > 1.0
    print(f"  G1c {'PASS' if out['G1c'] else 'FAIL'} -- the two formulas must differ on "
          f"mixed domains, or the cost model is not being used")

    print("\n" + "=" * 92)
    print("G1d  CROSS-CHECK against rem.factorgraph on random graphs")
    print("=" * 92)
    import numpy as np
    from rem import factorgraph as FG
    rng = np.random.default_rng(0)
    agree = 0
    for t in range(12):
        g = FG.random_graph(rng, n=8, card=2, n_factors=10, arity=2)
        a = {v: set(n) for v, n in g.adjacency().items()}
        mine = min(_greedy_width(a, "min-fill")[1], _greedy_width(a, "min-degree")[1])
        theirs = g.treewidth()
        agree += (mine == theirs)
        if mine != theirs and verbose:
            print(f"    trial {t}: mine {mine} vs engine {theirs}")
    print(f"  {agree}/12 random graphs agree with the engine's own treewidth()")
    out["G1d"] = agree == 12
    print(f"  G1d {'PASS' if out['G1d'] else 'FAIL'}")
    return out


if __name__ == "__main__":
    verify()
