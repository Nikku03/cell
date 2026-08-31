"""Does replication track fame? Hub concentration in the >=2-PMID subgraph vs its controls.

WHY THIS DECIDES BETWEEN TWO READINGS OF THE SAME NUMBER. Requiring two independent PMIDs
collapses the TRRUST knot from 888 nodes to 71, and the knot is SMALLER than every
density-matched random cull (71 against 99, 103, 109, 111, 113). Two stories fit that:

  STAR ARTEFACT. Replication concentrates on famous genes, so the surviving graph is a few
  hubs with spokes. Stars kernelize away almost completely -- a leaf is simplicial, and a hub
  whose neighbours are all leaves goes next -- so a small knot would be what a star-shaped
  graph does under these reductions, not evidence about biology.

  PERIPHERAL EFFORT. Replication follows individually well-studied interactions, which sit off
  the densely interconnected core, so the retained edges are genuinely peripheral and the knot
  shrinks because the core's edges were the singly-reported ones.

These predict OPPOSITE hub concentration. Under the star reading the filtered graph is MORE
hub-concentrated than a random cull of equal size; under the peripheral reading it is LESS.

"HUB-CONCENTRATED" IS AMBIGUOUS ABOUT WHOSE HUBS, SO BOTH VERSIONS ARE MEASURED.
  SELF hubs   -- each graph's own top-10 by its own degree. Measures how star-shaped the
                 surviving graph is internally, which is what governs whether it kernelizes
                 away. Every graph has a top 10, so this is always defined.
  GLOBAL hubs -- the FULL network's top-10, held FIXED across the filtered graph and every
                 control. This is the direct test of "replication tracks fame": do retained
                 edges preferentially touch the genes that are famous in the whole network?
The two can disagree, and if they do that is the finding rather than a problem: a graph can be
internally star-shaped around hubs that are not the globally famous ones.

=================================================================================================
THE GATES, FIXED BEFORE ANY NUMBER IS RUN.
=================================================================================================

H1  THE COMPARISON IS RANK AMONG DENSITY-MATCHED CONTROLS, never a ratio to their median.
    Same rule the previous module had to be repaired for: a median hides the spread, and at
    these sizes the spread is the whole story. Controls are random edge subsets of the FULL
    edge list of exactly the filtered subgraph's edge count. N_CONTROL = 50 here rather than 5,
    because these statistics are cheap (no kernelization needed) and 5 draws cannot locate a
    value in a distribution.

H2  THE STATISTICS, all reported for filtered and controls side by side:
      share of edges incident to the graph's own top-10 degree nodes    (SELF)
      share of edges incident to the full network's top-10 degree nodes (GLOBAL, fixed)
      Gini coefficient of the degree distribution
      max degree / mean degree
      node count at fixed edge count -- fewer nodes for the same edges IS concentration, so
      this is reported rather than left implicit.

H3  THE VERDICT, per statistic, two-sided and predeclared:
      filtered ABOVE every control  -> MORE hub-concentrated  -> star artefact reading
      filtered BELOW every control  -> LESS hub-concentrated  -> peripheral-effort reading
      filtered inside the range     -> indistinguishable; that statistic decides nothing
    A verdict is only claimed where SELF and GLOBAL agree. If they point opposite ways, both
    are reported and neither reading is adopted.

H4  THE >= 3 LEVEL IS REPORTED BUT CARRIES NO VERDICT. Its knot-size controls ran 0, 0, 6, 15,
    32 around a filtered 14 -- no resolving power -- and nothing about hub statistics repairs a
    sample that thin. It is shown for completeness and explicitly excluded from conclusions.

H5  A NULL IS A REAL ANSWER. If the filtered graph is indistinguishable from its controls on
    every statistic, then the >= 2 filter carries no hub information either way, the collapse
    from 888 to 71 is sparsification and nothing more, and BOTH readings above are unsupported.
"""
from __future__ import annotations

import argparse, json, sys
sys.path.insert(0, ".")
import numpy as np

from benchmarks.knot_evidence import load_edges, graph_of

N_CONTROL = 50
TOPK = 10


def degrees(adj):
    return {v: len(nb) for v, nb in adj.items()}


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def edge_share_touching(edges, nodes):
    """Fraction of edges with at least one endpoint in `nodes`."""
    if not edges:
        return float("nan")
    S = set(nodes)
    return float(sum(1 for tf, tgt, _m, _p in edges if tf in S or tgt in S) / len(edges))


def stats(edges, global_hubs):
    adj = graph_of(edges)
    deg = degrees(adj)
    own = [v for v, _ in sorted(deg.items(), key=lambda kv: -kv[1])[:TOPK]]
    d = np.array(list(deg.values()), float)
    return {"nodes": len(adj), "edges": len(edges),
            "self_top10_share": edge_share_touching(edges, own),
            "global_top10_share": edge_share_touching(edges, global_hubs),
            "gini": gini(d),
            "max_over_mean": float(d.max() / d.mean()) if len(d) else float("nan")}


def verdict(value, controls, more="MORE", less="LESS"):
    c = sorted(float(x) for x in controls)
    if not np.isfinite(value) or not c:
        return "UNDEFINED", 0, len(c)
    above = sum(1 for x in c if value > x)
    below = sum(1 for x in c if value < x)
    n = len(c)
    if above == n:
        return f"{more} than every control", above, n
    if below == n:
        return f"{less} than every control", below, n
    pct = 100.0 * above / n
    return f"inside control range (percentile {pct:.0f})", above, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", type=int, default=N_CONTROL)
    ap.add_argument("--out", default="benchmarks/knot_hubness.json")
    a = ap.parse_args(argv)

    edges = load_edges()
    full_adj = graph_of(edges)
    fdeg = degrees(full_adj)
    global_hubs = [v for v, _ in sorted(fdeg.items(), key=lambda kv: -kv[1])[:TOPK]]
    print(f"  full network {len(full_adj):,} nodes / {len(edges):,} edges")
    print(f"  GLOBAL top-{TOPK} hubs (held fixed for every comparison): "
          f"{', '.join(f'{h}({fdeg[h]})' for h in global_hubs[:6])} ...")

    out = {"global_hubs": global_hubs, "levels": []}
    KEYS = ["self_top10_share", "global_top10_share", "gini", "max_over_mean", "nodes"]
    for thr in (2, 3):
        sub = [e for e in edges if len(e[3]) >= thr]
        f = stats(sub, global_hubs)
        ctl = []
        for s in range(a.controls):
            idx = np.random.default_rng(500 + s).choice(len(edges), size=len(sub),
                                                        replace=False)
            ctl.append(stats([edges[int(i)] for i in idx], global_hubs))
        print(f"\n  >= {thr} PMIDs   {f['edges']:,} edges")
        print(f"     {'statistic':>20s} {'filtered':>9s} {'control min':>12s} "
              f"{'median':>8s} {'max':>8s}   verdict")
        rows = {}
        for k in KEYS:
            cs = [c[k] for c in ctl]
            v, above, n = verdict(f[k], cs)
            rows[k] = {"filtered": f[k], "control_min": float(np.min(cs)),
                       "control_median": float(np.median(cs)),
                       "control_max": float(np.max(cs)), "verdict": v,
                       "n_above": above, "n_control": n}
            print(f"     {k:>20s} {f[k]:9.4f} {np.min(cs):12.4f} {np.median(cs):8.4f} "
                  f"{np.max(cs):8.4f}   {v}")
        out["levels"].append({"threshold": thr, "filtered": f, "stats": rows})
        json.dump(out, open(a.out, "w"), indent=1, default=float)

    lvl2 = next(l for l in out["levels"] if l["threshold"] == 2)
    s_self = lvl2["stats"]["self_top10_share"]["verdict"]
    s_glob = lvl2["stats"]["global_top10_share"]["verdict"]
    print(f"\n  H3 VERDICT at >= 2 PMIDs (the only level with power):")
    print(f"     SELF   hubs: {s_self}")
    print(f"     GLOBAL hubs: {s_glob}")
    more = lambda s: s.startswith("MORE")
    less = lambda s: s.startswith("LESS")
    if more(s_self) and more(s_glob):
        concl = ("STAR ARTEFACT -- the filtered graph is more hub-concentrated on both "
                 "readings, so the small knot is what a star does under kernelization and "
                 "replication tracks fame.")
    elif less(s_self) and less(s_glob):
        concl = ("PERIPHERAL EFFORT HOLDS -- the filtered graph is less hub-concentrated on "
                 "both readings, so retained edges are genuinely peripheral.")
    elif more(s_self) != more(s_glob) and (more(s_self) or more(s_glob)) and (
            less(s_self) or less(s_glob)):
        concl = ("SPLIT -- the two readings of 'hub' disagree, so neither story is adopted; "
                 "both numbers stand as reported.")
    else:
        concl = ("NULL -- at least one reading is indistinguishable from a random cull, so the "
                 "filter carries no decisive hub information and BOTH stories stay unsupported "
                 "(H5).")
    print(f"     -> {concl}")
    out["conclusion"] = concl
    print(f"\n  H4 the >= 3 level is reported above but carries no verdict.")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
