"""Is the random-topology rank a property of the graph, or of the order it was laid out in?

THE CLAIM UNDER TEST. tt_rank.py measured, on a random one-regulator-per-gene topology, ranks
that ATTAIN the free bound d^floor(n/2) -- no compression at all -- and concluded that real
topologies are hard. But rank across a cut is not a property of a graph. It is a property of a
graph PLUS the linear ordering the tensor train lays it out in, and that module laid every
network out in gene-index order, which for a randomly wired graph is arbitrary and close to
the worst available.

The evidence that ordering dominates is already in this project's own results: a hub and a
chain have opposite costs for the same reason in mirror image -- the chain's coupling graph
agrees exactly with the tensor-train ordering, and the hub decouples under conditioning -- so
"same difficulty, opposite cost" is exactly what an ordering effect looks like.

And a one-regulator-per-gene graph is a FUNCTIONAL GRAPH: trees hanging off small cycles,
about as far from an expander as a graph gets. Measured, it has treewidth 2 and pathwidth 4.
If rank tracked the layout's cut structure, a good ordering would need d^pathwidth = 2^4 = 16,
against the 54, 41 and 85 measured in index order.

THE HONEST CAVEAT, WHICH IS WHY THIS IS A MEASUREMENT AND NOT A DERIVATION. Pathwidth bounds
the complexity of representing the GENERATOR, not the SOLUTION. This project has hit that gap
repeatedly -- a master equation's stationary state is a null vector, not a product of local
factors, and does not inherit the generator's factorisation. So reordering may not reduce the
solution's rank at all. The gap between 16 and 85 is large enough that the answer is decisive
either way, and either answer is worth having.

=================================================================================================
THE GATES, FIXED BEFORE ANY NUMBER IS RUN.
=================================================================================================

O1  THE ORDERING IS CHOSEN BY A GRAPH CRITERION, NEVER BY THE RANK. Minimising the reported
    quantity over orderings and then reporting its minimum would be fitting and scoring on the
    same rows (ledger defect J), and would guarantee an improvement whether or not one exists.
    The ordering is selected by simulated annealing on CUTWIDTH -- the maximum number of
    regulatory edges crossing any prefix boundary -- computed from the graph alone, with the
    stationary distribution never consulted. The rank is then measured under that ordering,
    once.

O2  NO RE-SOLVE, SO NOTHING ELSE CAN MOVE. Reordering genes is a permutation of the tensor's
    axes, so the SAME solved stationary distribution is reused and only the axis order changes.
    This makes the comparison exact rather than matched: any difference in rank is the
    ordering, because nothing else differs, not even floating-point noise in the solve.

O3  THE BASELINE IS THE ORDERING THAT WAS ACTUALLY USED. Index order, as tt_rank.py ran it, so
    the comparison answers the question that was actually asked of the earlier result.

O4  A RANDOM-ORDERING CONTROL. If index order happened to be unluckier than typical, some of
    the improvement is regression to the mean rather than the annealer's work. Rank is
    therefore also reported under RANDOM orderings (median over several), which is what "an
    arbitrary layout" really costs. The annealed ordering must beat the random median, not
    merely index order.

O5  THE VERDICT, in terms of the quantity the layer actually cares about.
      rank under the annealed ordering <= 1/2 the index-order rank, at BOTH 1e-3 and 1e-6
          -> ORDERING ARTIFACT. The earlier conclusion is withdrawn and the layer survives
             with a reordering step, whose cost must then be stated.
      rank essentially unchanged (>= 0.9x) under a genuinely better cutwidth
          -> INTRINSIC. The solution does not inherit the generator's structure, real
             topologies are hard, and the plan closes.
      anything between -> PARTIAL, reported as such with both numbers.

O6  THE CUTWIDTH MUST ACTUALLY IMPROVE, or O5 is vacuous. Report cutwidth and middle-cut edge
    count for index, annealed and random orderings. If annealing does not reduce cutwidth, the
    test never ran and says so instead of reporting a verdict.
"""
from __future__ import annotations

import argparse, json, sys, time
sys.path.insert(0, ".")
import numpy as np

from benchmarks.tt_rank import (binary_random, binary_cascade, stationary_robust, TOLS)

RANDOM_ORDERINGS = 9


def edges_of(net):
    """Regulatory edges as (target, regulator) pairs, from the graph alone."""
    return [(i, j) for i, j in enumerate(getattr(net, "regulators", []))
            if j is not None]


def cut_profile(edges, perm):
    """Edges crossing each prefix boundary under `perm`. pos[g] = where gene g sits."""
    n = len(perm)
    pos = np.empty(n, dtype=int)
    pos[np.asarray(perm)] = np.arange(n)
    cuts = np.zeros(n - 1, dtype=int)
    for a, b in edges:
        lo, hi = sorted((pos[a], pos[b]))
        cuts[lo:hi] += 1                      # the edge spans every boundary between them
    return cuts


def cutwidth(edges, perm):
    c = cut_profile(edges, perm)
    return int(c.max()) if len(c) else 0


def middle_cut_edges(edges, perm):
    c = cut_profile(edges, perm)
    return int(c[len(perm) // 2 - 1]) if len(c) else 0


def anneal_cutwidth(edges, n, seed=0, iters=40000):
    """Minimise CUTWIDTH over orderings. Uses the graph only -- never the distribution."""
    rng = np.random.default_rng(seed)
    perm = np.arange(n)
    best = perm.copy()
    cur = bestv = cutwidth(edges, perm)
    for k in range(iters):
        T = max(1e-3, 1.0 * (1.0 - k / iters))
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        cand = perm.copy(); cand[i], cand[j] = cand[j], cand[i]
        v = cutwidth(edges, cand)
        if v <= cur or rng.random() < np.exp(-(v - cur) / T):
            perm, cur = cand, v
            if v < bestv:
                best, bestv = cand.copy(), v
    return best, bestv


def ranks_under(p, n, perm, tols=TOLS):
    """Rank at the middle cut with the genes laid out in `perm` order.

    Reordering is a transpose of the state tensor, so the SAME solved distribution is reused
    and nothing but the axis order changes (gate O2).
    """
    T = np.asarray(p, float).reshape((2,) * n)
    T = np.transpose(T, axes=tuple(int(x) for x in perm))
    half = n // 2
    sv = np.linalg.svd(T.reshape(2 ** half, -1), compute_uv=False)
    sv = sv / sv[0]
    return {t: int((sv > t).sum()) for t in tols}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="10,12,14")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ratio", type=float, default=8.0)
    ap.add_argument("--out", default="benchmarks/tt_ordering.json")
    a = ap.parse_args(argv)
    ns = [int(x) for x in a.ns.split(",")]
    out = {"config": vars(a), "rows": []}

    print("  ORDERING TEST -- same solved distribution, different axis order (gate O2)")
    print(f"     {'n':>3s} {'seed':>4s} {'ordering':>9s} {'cutw':>5s} {'midE':>5s} "
          f"{'bound':>6s} {'r@1e-3':>7s} {'r@1e-6':>7s} {'r@1e-10':>8s}")
    for n in ns:
        half = n // 2
        bound = 2 ** half
        for seed in range(a.seeds):
            net = binary_random(n, ratio=a.ratio, seed=seed)
            edges = edges_of(net)
            t0 = time.perf_counter()
            p, _info = stationary_robust(net)
            solve_s = time.perf_counter() - t0

            idx = np.arange(n)
            ann, annv = anneal_cutwidth(edges, n, seed=seed)
            rnd = [np.random.default_rng(1000 + seed * 10 + k).permutation(n)
                   for k in range(RANDOM_ORDERINGS)]

            rows = []
            for label, perm in ([("index", idx), ("annealed", ann)]
                                + [(f"random{k}", q) for k, q in enumerate(rnd)]):
                r = ranks_under(p, n, perm)
                rows.append({"n": n, "seed": seed, "ordering": label,
                             "cutwidth": cutwidth(edges, perm),
                             "mid_edges": middle_cut_edges(edges, perm),
                             "bound": bound, "solve_s": solve_s,
                             "r": {str(k): v for k, v in r.items()}})
            # report index, annealed, and the MEDIAN random rather than nine noisy lines
            rnd_rows = [x for x in rows if x["ordering"].startswith("random")]
            med = {"n": n, "seed": seed, "ordering": "random~med",
                   "cutwidth": int(np.median([x["cutwidth"] for x in rnd_rows])),
                   "mid_edges": int(np.median([x["mid_edges"] for x in rnd_rows])),
                   "bound": bound, "solve_s": solve_s,
                   "r": {str(t): int(np.median([x["r"][str(t)] for x in rnd_rows]))
                         for t in TOLS}}
            for x in [rows[0], rows[1], med]:
                print(f"     {n:3d} {seed:4d} {x['ordering']:>9s} {x['cutwidth']:5d} "
                      f"{x['mid_edges']:5d} {bound:6d} {x['r'][str(1e-3)]:7d} "
                      f"{x['r'][str(1e-06)]:7d} {x['r'][str(1e-10)]:8d}", flush=True)
            out["rows"] += rows + [med]
        json.dump(out, open(a.out, "w"), indent=1, default=float)

    # ---- O6 then O5 ----
    def pick(lbl, n=None):
        return [x for x in out["rows"]
                if x["ordering"] == lbl and (n is None or x["n"] == n)]
    ci = np.mean([x["cutwidth"] for x in pick("index")])
    ca = np.mean([x["cutwidth"] for x in pick("annealed")])
    cr = np.mean([x["cutwidth"] for x in pick("random~med")])
    print(f"\n  O6  cutwidth: index {ci:.2f}, annealed {ca:.2f}, random median {cr:.2f}")
    if ca >= ci:
        print("      ANNEALING DID NOT REDUCE CUTWIDTH -- the test never ran. No verdict.")
        out["O6"] = False
        json.dump(out, open(a.out, "w"), indent=1, default=float)
        return 0
    out["O6"] = True

    print(f"\n  O5  verdict, per tolerance (annealed vs the index order actually used, and "
          f"vs a random layout)")
    verdicts = {}
    for t in (1e-3, 1e-6):
        ri = np.mean([x["r"][str(t)] for x in pick("index")])
        ra = np.mean([x["r"][str(t)] for x in pick("annealed")])
        rr = np.mean([x["r"][str(t)] for x in pick("random~med")])
        ratio = ra / ri if ri else float("nan")
        v = ("ORDERING ARTIFACT" if ratio <= 0.5 else
             "INTRINSIC" if ratio >= 0.9 else "PARTIAL")
        verdicts[t] = v
        print(f"      tol {t:.0e}: index {ri:.1f}  annealed {ra:.1f}  random {rr:.1f}   "
              f"annealed/index = {ratio:.2f}  -> {v}"
              f"{'' if ra <= rr else '   (WARNING: annealed is worse than random)'}")
        out.setdefault("O5", {})[str(t)] = {"index": ri, "annealed": ra, "random": rr,
                                            "ratio": float(ratio), "verdict": v}
    agree = verdicts[1e-3] == verdicts[1e-6]
    print(f"      tolerances {'agree' if agree else 'DISAGREE -- INCONCLUSIVE'}")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
