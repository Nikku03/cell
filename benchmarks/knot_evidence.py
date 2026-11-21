"""Is the 888-gene knot biology, or bibliometrics? Filter by evidence and re-kernelize.

THE QUESTION. Width-preserving kernelization of TRRUST leaves a knot of 888 genes in ONE
component at treewidth 200, and that single number is what closes the exact gene layer. TRRUST
is literature-curated from PubMed sentences, so an edge exists because somebody wrote it down.
If the knot is held together by singly-reported edges, it is a map of what has been studied
rather than of what regulates what, and the conclusion drawn from it is about publication
history.

THE TEST AS POSED IS CONFOUNDED, AND THE CONFOUND IS LARGE. Requiring >= 2 distinct PMIDs
removes 88.5% of the edges: 8,319 of 9,396 edges rest on exactly one paper, leaving 1,076.
A graph on ~2,861 nodes with 1,076 edges has mean degree below one, and ANY graph that sparse
fragments and dissolves under kernelization -- forests are exactly what these reductions
eliminate for free. So "the knot fragments under a >= 2-PMID filter" is the expected outcome
whether or not the original structure was bibliometric, and observing it would license nothing.

SO EVERY FILTER IS RUN AGAINST A DENSITY-MATCHED RANDOM CONTROL. For each evidence threshold,
random subsets of the FULL edge list of exactly the same size go through exactly the same
pipeline. The comparison is then between two graphs of identical size and identical node
universe, differing only in WHICH edges were kept, which is the one thing the question is about.

  the filtered knot is much LARGER than the density-matched control
      -> well-supported edges concentrate in the core. The structure is biology; the
         kernelization result stands.
  the filtered knot is the SAME SIZE as the control
      -> the filter did nothing a random cull would not have done. The evidence weighting
         carries no structural information, and the original knot cannot be attributed to
         well-supported biology on this data.
  the filtered knot is much SMALLER than the control
      -> well-supported edges are anti-concentrated in the periphery, which would be a strange
         and interesting result, and is reported as such rather than folded into either verdict.

WHAT A NULL HERE DOES AND DOES NOT MEAN. If the filtered knot matches its control, that does
not prove the 888-node knot is an artifact. It says this dataset cannot distinguish the two,
because at 11.5% edge retention the filter is mostly measuring sparsity. Saying more than that
would be the same error as reading a trend off a saturating detector.

=================================================================================================
THE GATES, FIXED BEFORE ANY NUMBER IS RUN.
=================================================================================================

E1  PROVENANCE. The raw file must reproduce the cached network exactly, or the filters are being
    applied to a different graph than the one the 888 came from. GATE: raw row count equals the
    cached JSON's edge count. Measured: 9,396 = 9,396, with 27 self-loops then dropped on both
    sides, leaving 9,369 edges.

E2  THE CONTROL IS DENSITY-MATCHED AND RUN EVERY TIME. For each threshold, N_CONTROL random
    edge subsets of the same size, same pipeline. A filter level without its control is not
    reported. THE VERDICT IS THE FILTERED VALUE'S RANK AMONG ITS CONTROLS, not a ratio to
    their median: a ratio hides the spread, and at these sizes the spread is the whole story.
    At >= 3 PMIDs the five controls came out 0, 0, 6, 15, 32, so a median of 6 made a filtered
    14 read as "2.33x, concentrated" when it sits squarely inside the control range. Ranks are
    distribution-free and cannot be fooled by a median of a handful of noisy draws.

E3  THE COMPARISON IS THE KNOT, NOT THE GRAPH. Report kernel node count, component count, the
    largest component's size, and treewidth -- for the filtered graph and its control side by
    side. A filtered graph is smaller than the original by construction; only the difference
    from its OWN control is evidence.

E4  KERNELIZATION IS UNCHANGED. The same two width-preserving rules, with the same controls
    (path and tree dissolve, expander does not peel) already gated in kernelize.py. Nothing
    about the reductions is re-tuned for this question.

E5  THE REVIEW FILTER IS SEPARATE AND MAY BE UNAVAILABLE. Dropping review-derived edges needs
    the publication type of each supporting PMID, which is not in the TRRUST file. If those
    types cannot be obtained, the level is reported as NOT MEASURED rather than approximated by
    a proxy, because a wrong review call would silently move the very edges under test.

RESULT.

    level        edges  nodes |  knot  comp  largest   tw |  controls (knot)         verdict
    >= 1 PMID    8,300  2,861 |   888     1      888  200 |  (is the whole graph)
    >= 2 PMIDs     883    609 |    71     1       71   19 |  99,103,109,111,113      BELOW ALL 5
    >= 3 PMIDs     293    239 |    14     1       14    7 |  0,0,6,15,32             INSIDE RANGE

THE 888-NODE KNOT DOES NOT SURVIVE A >= 2-PMID FILTER. It collapses to 71 nodes at treewidth
19. It never fragments -- every level stays a single component -- so the right description is
that it SHRINKS by an order of magnitude, not that it breaks apart.

BUT THE COLLAPSE IS NOT EVIDENCE THAT WELL-SUPPORTED BIOLOGY HOLDS THE CORE TOGETHER. A random
cull of the identical size leaves a LARGER knot: 99 to 113 across five draws, against the
evidence filter's 71, which is below every one of them. So requiring two independent reports
removes MORE core structure than deleting the same number of edges at random.

*** THE FIRST READING OF THAT WAS WRONG AND IS RETRACTED. *** This file originally concluded
that doubly-reported edges must be "anti-concentrated in the periphery", i.e. that replication
follows individually well-studied interactions sitting off the core. benchmarks/knot_hubness.py
tested that directly and found the opposite on every statistic that resolves:

    >= 2 PMIDs, filtered vs 50 density-matched controls
      share of edges touching the graph's own top-10 hubs   0.501  vs  0.257-0.328   ABOVE ALL
      share touching the FULL network's top-10 hubs         0.479  vs  0.258-0.322   ABOVE ALL
      Gini of the degree distribution                       0.538  vs  0.449-0.470   ABOVE ALL
      nodes carrying the same edge count                      609  vs  919-982       BELOW ALL

Half of all doubly-reported edges touch just ten genes -- SP1, NFKB1, RELA, TP53, JUN, MYC and
four others -- against 29% for a random cull. The filtered graph is MORE hub-concentrated, not
less, and it packs the same edges onto a third fewer nodes.

SO THE 71 IS A STAR-COLLAPSE ARTEFACT, AND THE FILTER INTERACTS WITH THE INSTRUMENT. A star is
precisely what these reductions dissolve for free: every leaf is simplicial, and a hub whose
neighbours have all been removed goes next. Requiring replication concentrates the surviving
graph onto a few famous regulators, which makes it star-shaped, which makes kernelization eat
it. The small knot is therefore what the filter's own bias does to the measurement, not a
statement about which edges hold the core together.

The >= 3 level resolves nothing: five controls spanning 0 to 32 around a filtered 14 is a
measurement with no power, and it is reported as such rather than as the "2.33x concentrated"
its median would have suggested.

WHAT THIS DOES AND DOES NOT SETTLE. Neither number survives as a measurement of biological
regulatory structure. The 888 rests on a graph that is 88.5% single-report edges. The 71 is
what kernelization does to a star, and the hub test shows the >= 2 filter manufactures exactly
that star. So the two figures fail for DIFFERENT reasons and neither can be quoted, and the
gap between them is not a bracket on the truth.

The sharper conclusion is about the method rather than the number: on a literature-curated
network, an evidence filter and a width-preserving kernelization are not independent
instruments. Filtering on replication selects for well-studied hubs, and hub-and-spoke
structure is the one thing these reductions remove for free, so the two compose into an
artefact that looks like a finding. Any future use of this pipeline on curated data has to
break that coupling -- by an assay-based network rather than a citation-based one, or by a
hub-preserving null -- before a knot size means anything.

RegulonDB was unreachable this session (HTTP 503), so the independently assayed network that
would settle it stays out of reach.
"""
from __future__ import annotations

import argparse, json, sys
sys.path.insert(0, ".")
import numpy as np

from benchmarks.kernelize import kernelize, minfill, cutwidth, components, n_edges

RAW = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/trrust_rawdata.human.tsv"
N_CONTROL = 5


def load_edges(path=RAW):
    """(tf, target, mode, [pmids]) with self-loops dropped, matching the cached build."""
    out = []
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 4:
            continue
        tf, tgt, mode, pm = f[0], f[1], f[2], f[3]
        if tf == tgt:
            continue
        ids = sorted({x for x in pm.replace(",", ";").split(";") if x.strip().isdigit()})
        out.append((tf, tgt, mode, ids))
    return out


def graph_of(edges):
    adj = {}
    for tf, tgt, _m, _p in edges:
        adj.setdefault(tf, set()).add(tgt)
        adj.setdefault(tgt, set()).add(tf)
    return adj


def knot_stats(adj, measure_tw=True):
    if not adj:
        return {"n": 0, "edges": 0, "kernel_n": 0, "components": 0, "largest": 0,
                "tw": 0, "low": 0}
    ker, low, tally = kernelize(adj)
    if not ker:
        return {"n": len(adj), "edges": n_edges(adj), "kernel_n": 0, "components": 0,
                "largest": 0, "tw": low, "low": low, "tally": tally}
    comps = components(ker)
    tw = minfill(ker)[0] if measure_tw else -1
    return {"n": len(adj), "edges": n_edges(adj), "kernel_n": len(ker),
            "kernel_edges": n_edges(ker), "components": len(comps),
            "largest": len(comps[0]), "tw": max(low, tw), "low": low, "tally": tally}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/knot_evidence.json")
    ap.add_argument("--controls", type=int, default=N_CONTROL)
    a = ap.parse_args(argv)

    edges = load_edges()
    cached = json.load(open("outputs/orphan/trrust_regulon.json"))
    n_cached = sum(len(v) for v in cached["tf_targets"].values())
    n_rows = sum(1 for line in open(RAW) if len(line.rstrip("\n").split("\t")) >= 4)
    print(f"  E1 provenance: raw {n_rows:,} rows vs cached {n_cached:,} "
          f"{'MATCH' if n_rows == n_cached else 'DIFFER'}; "
          f"{n_rows - len(edges):,} self-loops dropped -> {len(edges):,} edges")

    out = {"levels": []}
    print(f"\n  {'level':>14s} {'edges':>7s} {'nodes':>6s} | {'knot':>6s} {'comp':>5s} "
          f"{'largest':>8s} {'tw':>5s} | {'CONTROL knot':>13s} {'comp':>5s} {'largest':>8s} "
          f"{'tw':>5s}")
    rng = np.random.default_rng(0)
    for thr in (1, 2, 3):
        sub = [e for e in edges if len(e[3]) >= thr]
        g = graph_of(sub)
        st = knot_stats(g)
        # E2: density-matched control, same edge count drawn from the FULL list
        ctl = []
        for s in range(a.controls):
            idx = np.random.default_rng(100 + s).choice(len(edges), size=len(sub),
                                                        replace=False)
            ctl.append(knot_stats(graph_of([edges[int(i)] for i in idx])))
        med = {k: float(np.median([c[k] for c in ctl])) for k in
               ("kernel_n", "components", "largest", "tw")}
        out["levels"].append({"threshold": thr, "filtered": st, "control_median": med,
                              "controls": ctl})
        print(f"  {'>= ' + str(thr) + ' PMIDs':>14s} {st['edges']:7,d} {st['n']:6,d} | "
              f"{st['kernel_n']:6,d} {st['components']:5d} {st['largest']:8,d} "
              f"{st['tw']:5d} | {med['kernel_n']:13.0f} {med['components']:5.0f} "
              f"{med['largest']:8.0f} {med['tw']:5.0f}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1, default=float)

    print(f"\n  E5 review filter: NOT MEASURED -- publication types are not in the TRRUST file "
          f"and were not obtained; approximating them by a proxy would move the very edges "
          f"under test.")
    out["E5_review"] = "not measured"

    print(f"\n  E3 verdict, filtered knot against its OWN density-matched control:")
    for lv in out["levels"]:
        f_, c_ = lv["filtered"]["kernel_n"], lv["control_median"]["kernel_n"]
        if lv["threshold"] == 1:
            print(f"      >= 1 PMID  is the whole graph; its control is itself.")
            continue
        # THE VERDICT IS THE FILTERED VALUE'S RANK AMONG ITS CONTROLS, NOT A RATIO TO THEIR
        # MEDIAN. A ratio against a median hides the control spread, and at these sizes the
        # spread is the whole story: at >= 3 PMIDs the five controls ran 0, 0, 6, 15, 32, so a
        # median of 6 made a filtered 14 look like "2.33x, concentrated" when it sits squarely
        # inside the control range. Comparing ranks is distribution-free and cannot be fooled
        # by a median computed from a handful of noisy draws.
        ks = sorted(c["kernel_n"] for c in lv["controls"])
        below = sum(1 for x in ks if f_ < x)
        above = sum(1 for x in ks if f_ > x)
        n_c = len(ks)
        if below == n_c:
            v = (f"SMALLER THAN EVERY CONTROL ({below}/{n_c}) -- the evidence filter removes "
                 f"MORE core structure than a random cull of the same size")
        elif above == n_c:
            v = (f"LARGER THAN EVERY CONTROL ({above}/{n_c}) -- well-supported edges "
                 f"concentrate in the core")
        else:
            v = (f"INSIDE the control range [{ks[0]}, {ks[-1]}] -- indistinguishable from a "
                 f"random cull; this level has no resolving power")
        print(f"      >= {lv['threshold']} PMIDs: knot {f_:,} vs controls {ks}  -> {v}")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
