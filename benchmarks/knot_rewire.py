"""Is the knot real wiring, or just the degree sequence? Degree-preserving rewiring null.

WHY THIS IS THE TEST THAT SETTLES IT. Every control so far has been density-matched: the same
NUMBER of edges, drawn uniformly. That destroys the degree sequence along with everything else,
so a difference from it can always be explained by "some genes have many partners" -- which is
not a claim about regulatory wiring, only about how often a gene appears in the literature. The
hub result made that explanation the leading one: the >= 2-PMID subgraph is more
hub-concentrated than every uniform control, and stars are exactly what width-preserving
kernelization dissolves for free.

A DEGREE-PRESERVING REWIRING REMOVES THAT EXPLANATION BY CONSTRUCTION. Double-edge swaps --
take (a,b) and (c,d), rewrite as (a,d) and (c,b) -- change WHO is connected to WHOM while every
single node keeps exactly the degree it had. SP1 still has 479 partners; they are just different
partners. So the rewired ensemble holds fame fixed and varies only wiring, and comparing the
real knot against it asks precisely the question left open:

    knot LARGER than every rewiring   -> the wiring carries structure the degrees do not.
                                          There is real organisation in the core.
    knot INSIDE the rewiring spread   -> the degree sequence alone reproduces it. The knot is
                                          "some genes are famous" and nothing more, and no
                                          further filtering of this dataset will change that.
    knot SMALLER than every rewiring  -> the real network is LESS tangled than its own degree
                                          sequence implies, which would be a positive finding
                                          about modularity and is reported as such.

=================================================================================================
THE GATES, FIXED BEFORE ANY NUMBER IS RUN.
=================================================================================================

R1  THE REWIRING IS VALIDATED, NOT TRUSTED. After every rewiring, assert that each node's degree
    is EXACTLY unchanged, that no self-loop was created, and that no edge is duplicated. A swap
    routine that silently drops rejected swaps still preserves degrees; one with an indexing bug
    may not, and a null whose degrees have drifted is not the null being claimed.

R2  THE REWIRING MUST ACTUALLY MOVE. Report the fraction of ORIGINAL edges still present. With
    too few swaps the "null" is the original graph wearing a hat, and it would trivially match.
    GATE AS ORIGINALLY WRITTEN: retained fraction < 0.05.

R2 WAS UNREACHABLE BY CONSTRUCTION, AND THAT IS LEDGER DEFECT N AGAIN -- an absolute bar set
    above the achievable ceiling, written into a brand new gate one arc after recording the
    rule. Retention plateaus at 0.072 and will not go lower however long the shuffling runs:
    30 swaps/edge gives 0.0799, 1000 swaps/edge (8.3 million swaps) gives 0.0717. The floor is
    structural. Under a degree-matched random graph an edge recurs with probability about
    d_i d_j / 2m, and the hub-hub edges have d_i d_j far exceeding 2m -- NFKB1-SP1 is
    157,112 against 16,600 -- so they recur with probability 1. 109 edges (1.3%) are forced,
    and they are exactly the core. The measured Chung-Lu expectation for this degree sequence
    is 0.0738 against an observed 0.0717, so the shuffle is FULLY MIXED and the bar was simply
    below what any degree-preserving null can reach.

R2b THE REPAIR, declared separately and leaving R2's verdict standing. Compare the observed
    retention to the CONFIGURATION-MODEL EXPECTATION for this degree sequence rather than to a
    constant: mixed means observed / expected is near 1, which is scale-free and derived from
    the data's own structure. Measured 0.0717 / 0.0738 = 0.97.
    AND THE RESIDUAL SHARING BIASES THE TEST IN A KNOWN DIRECTION, which has to be stated
    because it decides how much the result is worth: the rewired graphs necessarily retain the
    forced hub-hub core, so they share the real graph's most tangled part. That can only make
    them look MORE like the real graph. Any difference that survives is therefore understated,
    never inflated.

R3  THE VERDICT IS RANK AMONG THE REWIRINGS, two-sided, never a ratio to their mean. Same rule
    that this arc has now had to repair twice.

R4  BOTH GRAPHS ARE TESTED: the full network, whose knot is 888, and the >= 2-PMID subgraph,
    whose knot is 71. The star reading predicts the >= 2 subgraph in particular should be
    reproduced by its degree sequence, since that is what "it is a star" means.

R5  KNOT SIZE FOR EVERY REWIRING, TREEWIDTH FOR A SUBSET. Kernelization is fast; min-fill on a
    ~900-node kernel is not. Knot size and component count are reported for all N_REWIRE
    rewirings, treewidth for the first N_TW of them, and the counts are stated so the treewidth
    comparison is not mistaken for having the same sample size.

R6  A NULL IS THE ANSWER, NOT A FAILURE. If the knot sits inside the rewired spread, that closes
    the question: the structure is the degree sequence, the 888 was never a statement about
    regulatory organisation, and this dataset cannot be made to yield one.
"""
from __future__ import annotations

import argparse, json, sys, time
sys.path.insert(0, ".")
import numpy as np

from benchmarks.kernelize import kernelize, minfill, components, n_edges
from benchmarks.knot_evidence import load_edges, graph_of

N_REWIRE = 20
N_TW = 5
SWAPS_PER_EDGE = 300
RETAINED_BAR = 0.05


def rewire(adj, seed=0, swaps_per_edge=SWAPS_PER_EDGE):
    """Degree-preserving double-edge swap on a simple undirected graph."""
    rng = np.random.default_rng(seed)
    nbr = {v: set(nb) for v, nb in adj.items()}
    edges = [(u, v) for u in nbr for v in nbr[u] if u < v]
    m = len(edges)
    target = swaps_per_edge * m
    done = 0
    for _ in range(target * 4):
        if done >= target:
            break
        i, j = int(rng.integers(m)), int(rng.integers(m))
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        if d in nbr[a] or b in nbr[c]:
            continue
        nbr[a].discard(b); nbr[b].discard(a)
        nbr[c].discard(d); nbr[d].discard(c)
        nbr[a].add(d); nbr[d].add(a)
        nbr[c].add(b); nbr[b].add(c)
        edges[i] = (a, d) if a < d else (d, a)
        edges[j] = (c, b) if c < b else (b, c)
        done += 1
    return nbr, done


def validate(orig, new):
    """R1: degrees exactly preserved, simple graph, and R2's retained fraction."""
    bad_deg = [v for v in orig if len(orig[v]) != len(new.get(v, ()))]
    self_loops = [v for v in new if v in new[v]]
    sym = all(u in new[v] for v in new for u in new[v])
    e0 = {(u, v) for u in orig for v in orig[u] if u < v}
    e1 = {(u, v) for u in new for v in new[u] if u < v}
    retained = len(e0 & e1) / max(1, len(e0))
    return {"degree_mismatches": len(bad_deg), "self_loops": len(self_loops),
            "symmetric": bool(sym), "retained_frac": float(retained),
            "n_edges_before": len(e0), "n_edges_after": len(e1)}


def expected_retention(adj):
    """Chung-Lu probability that each existing edge recurs in a degree-matched random graph.

    This is the floor the shuffle can reach, and it is what R2's absolute bar should have been
    compared against.
    """
    deg = {v: len(nb) for v, nb in adj.items()}
    twom = sum(deg.values())
    ps = [min(1.0, deg[u] * deg[v] / twom) for u in adj for v in adj[u] if u < v]
    return float(np.mean(ps)) if ps else float("nan")


def knot_of(adj, with_tw=False):
    ker, low, _t = kernelize(adj)
    if not ker:
        return {"knot": 0, "components": 0, "largest": 0, "tw": low}
    comps = components(ker)
    tw = max(low, minfill(ker)[0]) if with_tw else None
    return {"knot": len(ker), "components": len(comps), "largest": len(comps[0]), "tw": tw}


def rank_verdict(value, nulls):
    n = len(nulls)
    above = sum(1 for x in nulls if value > x)
    below = sum(1 for x in nulls if value < x)
    if above == n:
        return f"LARGER than all {n} rewirings"
    if below == n:
        return f"SMALLER than all {n} rewirings"
    return f"INSIDE the rewired spread [{min(nulls)}, {max(nulls)}] (percentile {100*above/n:.0f})"


def run(label, adj, n_rewire, n_tw, out):
    real = knot_of(adj, with_tw=True)
    print(f"\n  {label}: {len(adj):,} nodes / {n_edges(adj):,} edges")
    print(f"     REAL knot {real['knot']:,}  components {real['components']}  "
          f"tw {real['tw']}")
    nulls, tws, vals = [], [], []
    worst_retained, bad = 0.0, 0
    for s in range(n_rewire):
        new, done = rewire(adj, seed=s)
        v = validate(adj, new)
        bad += (v["degree_mismatches"] > 0 or v["self_loops"] > 0 or not v["symmetric"]
                or v["n_edges_before"] != v["n_edges_after"])
        worst_retained = max(worst_retained, v["retained_frac"])
        k = knot_of(new, with_tw=(s < n_tw))
        nulls.append(k["knot"]); vals.append(k)
        if k["tw"] is not None:
            tws.append(k["tw"])
        if s == 0:
            print(f"     R1 validation: degree mismatches {v['degree_mismatches']}, "
                  f"self-loops {v['self_loops']}, symmetric {v['symmetric']}, "
                  f"edges {v['n_edges_before']} -> {v['n_edges_after']}, "
                  f"{done:,} swaps")
    exp = expected_retention(adj)
    ratio = worst_retained / exp if exp > 0 else float("nan")
    print(f"     R1 across {n_rewire} rewirings: {bad} invalid")
    print(f"     R2b mixing: retained {worst_retained:.4f} vs configuration-model expectation "
          f"{exp:.4f}  ratio {ratio:.2f}  "
          f"({'FULLY MIXED' if ratio < 1.25 else 'NOT MIXED'})"
          f"   [R2's absolute bar of {RETAINED_BAR} was unreachable -- see docstring]")
    print(f"     rewired knots: min {min(nulls)}  median {int(np.median(nulls))}  "
          f"max {max(nulls)}")
    if tws:
        print(f"     rewired treewidth (n={len(tws)}): {sorted(tws)}   REAL {real['tw']}")
    print(f"     R3 knot verdict: {rank_verdict(real['knot'], nulls)}")
    if tws:
        print(f"     R3 tw   verdict: {rank_verdict(real['tw'], tws)}")
    out[label] = {"real": real, "null_knots": nulls, "null_tws": tws,
                  "worst_retained": worst_retained, "n_invalid": int(bad),
                  "knot_verdict": rank_verdict(real["knot"], nulls),
                  "tw_verdict": rank_verdict(real["tw"], tws) if tws else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewires", type=int, default=N_REWIRE)
    ap.add_argument("--tw", type=int, default=N_TW)
    ap.add_argument("--out", default="benchmarks/knot_rewire.json")
    a = ap.parse_args(argv)

    edges = load_edges()
    out = {}
    run("FULL network (>= 1 PMID)", graph_of(edges), a.rewires, a.tw, out)
    sub = [e for e in edges if len(e[3]) >= 2]
    run(">= 2 PMIDs", graph_of(sub), a.rewires, a.tw, out)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
