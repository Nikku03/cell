"""What does a REAL regulatory network's structure cost a tensor network?

WHY THIS AND NOT ANOTHER SYNTHETIC TOPOLOGY. tt_ordering.py established that rank across a cut
is a property of the graph PLUS its layout, and that a bad layout can inflate the rank by 2-3x.
That makes the layout-free structural widths of the actual network the thing to measure, because
they bound what ANY layout can achieve:

    treewidth   what a TREE tensor network following the graph's own hierarchy would pay
    cutwidth    what a TENSOR TRAIN, which must linearise the graph, would pay

If those differ substantially the choice of representation matters; if both are large the layer
is dead regardless of representation.

WHAT IS ACTUALLY IN THIS REPO, stated because it is not what one would choose. The network here
is TRRUST v2 -- HUMAN, literature-curated from PubMed sentences, 2,861 nodes and 9,369 edges.
It is not E. coli and it is not a minimal cell. Literature curation biases hard toward
well-studied regulators (max degree 479), which inflates hub structure relative to an unbiased
map. Every number below is a statement about this graph.

THE MEASUREMENT IS AN UPPER BOUND, AND ONLY IN ONE DIRECTION. Treewidth is NP-hard; min-fill is
a heuristic that returns a valid UPPER bound, so a reported width of 200 means "at most 200",
and the true width could be lower. That direction matters: an upper bound cannot be used to
prove a graph is HARD. What it can do is show that the widths differ from each other, and show
how width scales with the size of the subnetwork -- which is the decision-relevant question.

THE GAP THAT REMAINS, AND WHICH THIS MODULE DOES NOT CLOSE. Treewidth and cutwidth bound the
complexity of representing the GENERATOR. The quantity that governs cost is the rank of the
SOLUTION, and a master equation's stationary state is a null vector that does not inherit the
generator's factorisation -- this project has hit that gap repeatedly. tt_ordering.py measured
the empirical coupling between them on small systems: Spearman(middle-cut edges, r@1e-6) =
+0.712 over 132 orderings, so they ARE coupled, but the measured relationship is far shallower
than the generator bound (at 9 cut edges the bound is 2^9 = 512 and the median measured rank is
48). So structural width predicts solution rank in direction but not in magnitude, and a width
measured here cannot be turned into a rank by exponentiating it.

RESULT, recorded from the run (reproduce with `PYTHONPATH=. python benchmarks/tf_network_width.py`):

  WHOLE GRAPH, 2,861 nodes / 9,369 edges, degree mean 5.8, median 2, max 479
      treewidth (min-fill upper bound)      200
      cutwidth under that elimination order 5,479   (middle cut 1,668)
  The two differ by a factor of 27, so a tree tensor network following the hierarchy is a
  genuinely different proposition from a tensor train -- which is the useful half of the
  result. The unuseful half is that 2^200 is 1.6e60, so both are hopeless at face value.

  INDUCED SUBNETWORKS -- does a 500-gene module stay tractable?
      size   selection      edges   treewidth(UB)   tw/n
        50   hub-centred      333        22         0.44
       100   hub-centred      762        44         0.44
       200   hub-centred     1665        87         0.43
       400   hub-centred     3072       145         0.36
       600   hub-centred     4108       170         0.28
        50   random             3         1         0.02
       100   random             1         1         0.01
       200   random            55         2         0.01
       400   random           133         3         0.01
       600   random           314         8         0.01

  THE DICHOTOMY IS THE FINDING, and it is the same shape as this project's metabolism result
  (HumanGEM local neighbourhood treewidth 0.5-0.9 x n). Tractability is not a property of the
  gene count; it is a property of WHICH genes. The densely interacting core sits at treewidth
  ~0.3-0.44 n, so a 500-gene core is ~150-200 and out of reach. A random 500 genes sit at
  treewidth ~3 -- trivially tractable, and trivially uninteresting, because 400 random genes
  share only 133 edges and are very nearly independent. Where genes barely interact, a joint
  exact treatment is not needed; where they interact enough to need one, the width is too
  large. That is the gene-network analogue of this project's crowding crossover, and it points
  the same way: exactness earns its keep only in a regime that is, here, also too expensive.
"""
from __future__ import annotations

import itertools, json, sys, time
sys.path.insert(0, ".")
import numpy as np

SRC = "outputs/orphan/trrust_regulon.json"


def load_graph(path=SRC):
    d = json.load(open(path))
    adj = {}
    for tf, targs in d["tf_targets"].items():
        for e in targs:
            t = e[0] if isinstance(e, (list, tuple)) else e
            if t == tf:
                continue
            adj.setdefault(tf, set()).add(t)
            adj.setdefault(t, set()).add(tf)
    return adj, d.get("source", "?")


def minfill(adj0):
    """Min-fill elimination. Returns (width, order). The width is an UPPER bound on treewidth."""
    a = {k: set(v) for k, v in adj0.items()}
    w, order = 0, []
    while a:
        best, bestf = None, None
        for v, nb in a.items():
            miss = sum(1 for x, y in itertools.combinations(nb, 2) if y not in a[x])
            if bestf is None or miss < bestf:
                best, bestf = v, miss
            if bestf == 0:
                break
        nb = a[best]
        w = max(w, len(nb))
        order.append(best)
        for x, y in itertools.combinations(nb, 2):
            a[x].add(y); a[y].add(x)
        for x in nb:
            a[x].discard(best)
        del a[best]
    return w, order


def cutwidth(adj, order):
    pos = {g: i for i, g in enumerate(order)}
    cuts = np.zeros(max(len(order) - 1, 1), dtype=int)
    for u, nb in adj.items():
        for v in nb:
            if pos[u] < pos[v]:
                cuts[pos[u]:pos[v]] += 1
    return int(cuts.max()), int(cuts[len(order) // 2 - 1])


def induced(adj, nodes):
    S = set(nodes)
    return {v: {u for u in adj[v] if u in S} for v in S}


def main():
    adj, src = load_graph()
    deg = np.array([len(v) for v in adj.values()])
    E = int(deg.sum() // 2)
    print(f"  {src}")
    print(f"  {len(adj):,} nodes, {E:,} undirected edges; degree mean {deg.mean():.1f}, "
          f"median {np.median(deg):.0f}, max {deg.max()}")
    t = time.perf_counter()
    w, order = minfill(adj)
    cw, mid = cutwidth(adj, order)
    print(f"  treewidth (min-fill UPPER bound)  {w:,}      [{time.perf_counter()-t:.1f}s]")
    print(f"  cutwidth under that order         {cw:,}   (middle cut {mid:,})")
    print(f"  -> they differ by {cw / max(w, 1):.0f}x, so a tree tensor network following the "
          f"hierarchy is a different proposition from a tensor train.")

    print(f"\n  induced subnetworks -- is tractability about SIZE or about WHICH genes?")
    print(f"     {'size':>5s} {'selection':>12s} {'edges':>7s} {'tw(UB)':>7s} {'tw/n':>6s}")
    order_by_deg = sorted(adj, key=lambda v: -len(adj[v]))
    rng = np.random.default_rng(0)
    out = []
    for size in (50, 100, 200, 400, 600):
        for lbl, nodes in (("hub-centred", order_by_deg[:size]),
                           ("random", list(rng.choice(list(adj), size=size, replace=False)))):
            g = induced(adj, nodes)
            e = sum(len(v) for v in g.values()) // 2
            ww, _o = minfill(g)
            out.append({"size": size, "selection": lbl, "edges": e, "tw": ww})
            print(f"     {size:5d} {lbl:>12s} {e:7d} {ww:7d} {ww / size:6.2f}", flush=True)
    json.dump({"nodes": len(adj), "edges": E, "treewidth_ub": w, "cutwidth": cw,
               "mid_cut": mid, "induced": out},
              open("benchmarks/tf_network_width.json", "w"), indent=1)
    print("\n  wrote benchmarks/tf_network_width.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
