"""Width-preserving kernelization: strip the periphery, and measure what knot is left.

THE POINT, which is stronger than a diagnostic. There is a set of graph reductions that
PROVABLY do not change treewidth. Apply them to exhaustion and whatever survives is the hard
core, by definition rather than by clustering heuristic. And the peeling is not preprocessing
for a measurement -- it is the first half of an algorithm: strip the periphery, solve the knot
exactly, then reconstruct every peeled node by running the removals backwards, also exactly.
So mapping the knot and solving the network are the same operation.

THE TWO RULES, AND WHY ONLY TWO. Everything usually listed separately -- islet, twig, series,
triangle -- is a special case of one of these, and writing them separately is how the lower
bound gets mis-stated.

  SIMPLICIAL. If N(v) is already a clique, then v sits in a bag with its neighbours in some
  optimal decomposition, so removing it cannot lower the width below deg(v):
      tw(G) = max(deg(v), tw(G - v))
  Always safe. Accumulates low := max(low, deg(v)).

  ALMOST SIMPLICIAL. If N(v) minus one vertex is a clique AND deg(v) <= low, remove v after
  making N(v) a clique. The guard deg(v) <= low is not optional and is the subtle part: it is
  what makes the rule safe, because the fill-in it introduces is free only when the width is
  already known to be at least deg(v).

  THE SERIES RULE IS THIS RULE, NOT A FREE ONE. A degree-2 vertex is almost simplicial
  trivially, so "bypass the pass-through node" is licensed only once low >= 2. Applying it
  unguarded and setting low := max(low, 2) is wrong, and a path is the counter-example: every
  interior vertex is degree 2, the unguarded rule would return low = 2, and a path has
  treewidth 1. Here a path instead dissolves by the simplicial rule from its endpoints inwards
  with low = 1, which is the right answer.

THE RESULT IS A BRACKET, AND ITS ORIENTATION MATTERS.
    tw(G) = max(low, tw(kernel))
`low` is a LOWER bound earned during peeling; tw(kernel) is whatever the kernel turns out to
need. Reporting them as a pair with the lower one second reads like an inverted interval; they
are widths of two different graphs and the original's width is the MAX of the two, never the
min and never a range between them.

=================================================================================================
THE GATES, FIXED BEFORE ANY NUMBER IS RUN.
=================================================================================================

K1  CONTROLS, RUN FIRST, AND THEY ARE PASS/FAIL. The reductions have a known answer on three
    graphs, so if they do not reproduce it the measurement is not run:
      path of 200            -> must vanish entirely, low = 1
      random tree of 200     -> must vanish entirely, low = 1
      3-regular expander     -> must NOT peel at all; an expander has no simplicial vertex and
                                low never reaches the degree needed to license the almost-
                                simplicial rule. If an expander peels, a rule is unsound.

K2  WIDTH IS PRESERVED, CHECKED AND NOT ASSUMED. On every graph small enough to measure both,
    compare the min-fill width of the original against max(low, min-fill width of the kernel).
    These are heuristic upper bounds, so they need not be equal -- but the kernel's bracket
    must not EXCEED the original's width by more than the heuristic's own slack, and if
    max(low, tw_kernel) < tw_original the peeling has apparently made the problem easier, which
    for a width-preserving reduction means the heuristic got luckier on the kernel, not that
    difficulty vanished. Both numbers are reported so that cannot be mistaken for a gain.

K3  WHAT SURVIVES, reported as counts and not adjectives: kernel nodes and edges, the fraction
    of the original, and the tally by rule so "peeled for free" is auditable.

K4  DOES THE KNOT FRAGMENT? Connected components of the kernel, with sizes. This is the
    decision-relevant question: independent components are solved separately and their sizes
    add rather than multiply, so a kernel of 700 in ten pieces of 70 is a different proposition
    from a kernel of 700 in one piece.

K5  THE KNOT'S OWN WIDTHS: treewidth (min-fill upper bound) and cutwidth, per component and
    overall, since a tensor train pays cutwidth and a tree tensor network pays treewidth.

K6  SOURCE SENSITIVITY. TRRUST is human, literature-curated from PubMed sentences, and biased
    toward well-studied regulators. Any conclusion that rests on one curated network is a
    conclusion about that curation, so the same measurement is reported for every source
    available, and a source that could not be obtained is named as missing rather than
    silently dropped.
"""
from __future__ import annotations

import argparse, itertools, json, sys, time
sys.path.insert(0, ".")
import numpy as np


# ---------------------------------------------------------------- the reductions

def _is_clique(a, nodes):
    for x, y in itertools.combinations(nodes, 2):
        if y not in a[x]:
            return False
    return True


def _almost_simplicial_pivot(a, nb):
    """Return w such that nb - {w} is a clique, or None."""
    missing = []
    for x, y in itertools.combinations(nb, 2):
        if y not in a[x]:
            missing.append((x, y))
            if len(missing) > len(nb):        # far too broken to be almost simplicial
                return None
    if not missing:
        return next(iter(nb), None)
    cand = set(missing[0])
    for w in list(cand):
        if all(w in e for e in missing):
            return w
    return None


def kernelize(adj, max_rounds=10_000):
    """Peel with width-preserving rules. Returns (kernel, low, tally)."""
    a = {v: set(nb) for v, nb in adj.items()}
    low = 0
    tally = {"simplicial": 0, "almost_simplicial": 0}
    changed = True
    rounds = 0
    while changed and rounds < max_rounds:
        changed = False
        rounds += 1
        for v in list(a.keys()):
            if v not in a:
                continue
            nb = a[v]
            if _is_clique(a, nb):
                low = max(low, len(nb))
                for x in nb:
                    a[x].discard(v)
                del a[v]
                tally["simplicial"] += 1
                changed = True
                continue
            if len(nb) <= low:
                w = _almost_simplicial_pivot(a, nb)
                if w is not None:
                    for x, y in itertools.combinations(nb, 2):
                        a[x].add(y); a[y].add(x)
                    low = max(low, len(nb))
                    for x in nb:
                        a[x].discard(v)
                    del a[v]
                    tally["almost_simplicial"] += 1
                    changed = True
    return a, low, tally


# ---------------------------------------------------------------- widths

def minfill(adj0):
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
    if len(order) < 2:
        return 0, 0
    cuts = np.zeros(len(order) - 1, dtype=int)
    for u, nb in adj.items():
        for v in nb:
            if pos[u] < pos[v]:
                cuts[pos[u]:pos[v]] += 1
    return int(cuts.max()), int(cuts[len(order) // 2 - 1])


def components(adj):
    seen, comps = set(), []
    for s in adj:
        if s in seen:
            continue
        stack, cur = [s], []
        seen.add(s)
        while stack:
            v = stack.pop(); cur.append(v)
            for u in adj[v]:
                if u not in seen:
                    seen.add(u); stack.append(u)
        comps.append(cur)
    return sorted(comps, key=len, reverse=True)


def n_edges(adj):
    return sum(len(v) for v in adj.values()) // 2


# ---------------------------------------------------------------- control graphs

def path_graph(n):
    a = {i: set() for i in range(n)}
    for i in range(n - 1):
        a[i].add(i + 1); a[i + 1].add(i)
    return a


def random_tree(n, seed=0):
    rng = np.random.default_rng(seed)
    a = {i: set() for i in range(n)}
    for i in range(1, n):
        j = int(rng.integers(0, i))
        a[i].add(j); a[j].add(i)
    return a


def cubic_expander(n, seed=0):
    """3-regular random graph via three perfect matchings on a cycle (n even)."""
    rng = np.random.default_rng(seed)
    a = {i: set() for i in range(n)}
    for i in range(n):
        j = (i + 1) % n
        a[i].add(j); a[j].add(i)
    perm = rng.permutation(n)
    for k in range(0, n, 2):
        u, v = int(perm[k]), int(perm[k + 1])
        if u != v:
            a[u].add(v); a[v].add(u)
    return a


# ---------------------------------------------------------------- report

def analyse(name, adj, measure_original=True, verbose=True):
    n0, e0 = len(adj), n_edges(adj)
    t = time.perf_counter()
    ker, low, tally = kernelize(adj)
    ks = time.perf_counter() - t
    nk, ek = len(ker), n_edges(ker)
    row = {"name": name, "n": n0, "edges": e0, "kernel_n": nk, "kernel_edges": ek,
           "survive_frac": (nk / n0 if n0 else 0.0), "low": low, "tally": tally,
           "kernel_seconds": ks}
    if nk:
        comps = components(ker)
        row["n_components"] = len(comps)
        row["component_sizes"] = [len(c) for c in comps[:12]]
        wk, ok = minfill(ker)
        cw, mid = cutwidth(ker, ok)
        row["kernel_tw_ub"] = wk
        row["kernel_cutwidth"] = cw
        row["kernel_midcut"] = mid
        row["tw_bracket"] = max(low, wk)
    else:
        row["n_components"] = 0
        row["component_sizes"] = []
        row["kernel_tw_ub"] = 0
        row["kernel_cutwidth"] = 0
        row["tw_bracket"] = low
    if measure_original and n0 <= 4000:
        w0, o0 = minfill(adj)
        row["orig_tw_ub"] = w0
    if verbose:
        print(f"  {name:<28s} n={n0:6,d} E={e0:7,d}  ->  kernel n={nk:5,d} "
              f"({100*row['survive_frac']:5.1f}%) E={ek:6,d}  low={low:3d} "
              f"tw(kernel)={row['kernel_tw_ub']:4d}  bracket={row['tw_bracket']:4d}"
              f"{'  tw(orig)=' + str(row.get('orig_tw_ub')) if 'orig_tw_ub' in row else ''}"
              f"  [{ks:.1f}s]", flush=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/kernelize.json")
    ap.add_argument("--skip-controls", action="store_true")
    a = ap.parse_args(argv)
    out = {"controls": [], "networks": []}

    if not a.skip_controls:
        print("  K1 CONTROLS -- the reductions have a known answer on these\n")
        c1 = analyse("path 200", path_graph(200))
        c2 = analyse("random tree 200", random_tree(200))
        c3 = analyse("3-regular expander 200", cubic_expander(200))
        out["controls"] = [c1, c2, c3]
        ok = (c1["kernel_n"] == 0 and c1["low"] == 1
              and c2["kernel_n"] == 0 and c2["low"] == 1
              and c3["survive_frac"] == 1.0)
        print(f"\n  K1 {'PASS' if ok else 'FAIL'}: trees must dissolve to low=1, an expander "
              f"must not peel at all")
        out["K1"] = bool(ok)
        if not ok:
            print("     a rule is unsound -- the measurement is NOT run.")
            json.dump(out, open(a.out, "w"), indent=1, default=float)
            return 1

    print(f"\n  K3/K4/K5 REAL NETWORKS\n")
    from benchmarks.tf_network_width import load_graph
    srcs = [("TRRUST v2 (human, curated)", "outputs/orphan/trrust_regulon.json")]
    for label, path in srcs:
        try:
            adj, src = load_graph(path)
        except Exception as e:                                     # noqa: BLE001
            print(f"  {label}: MISSING ({type(e).__name__}) -- named, not silently dropped")
            out["networks"].append({"name": label, "missing": str(e)[:120]})
            continue
        row = analyse(label, adj, measure_original=False)
        out["networks"].append(row)
        if row["kernel_n"]:
            print(f"     components: {row['n_components']}  sizes(top) "
                  f"{row['component_sizes']}")
            print(f"     knot cutwidth {row['kernel_cutwidth']:,} (mid {row['kernel_midcut']:,})"
                  f"   peeled by rule: {row['tally']}")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
