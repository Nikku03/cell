"""How wide a cut can the rank measurement actually reach? And why that is the whole answer.

THE REQUEST THIS ANSWERS. The kernelized TRRUST knot has treewidth 200 and cutwidth 3,574, and
the rank-versus-width relation had only been measured over 1-11 cut edges. The obvious next
step is to stop extrapolating and measure the rank at the knot's real width instead.

IT CANNOT BE DONE, AND THE OBSTRUCTION IS STRUCTURAL RATHER THAN BUDGETARY. Rank across a cut
is bounded for free by the smaller side of it, d^floor(n/2). To OBSERVE a rank of about 200
without that ceiling doing the work, the free bound must be comfortably above 200, which needs
roughly 16 two-state genes on each side of the cut -- about 2^400 states. So measuring the rank
at width 200 requires already being able to solve at width 200, which is precisely the thing
the rank number exists to tell us we can avoid. The requirement is circular, and no amount of
compute removes it.

WHAT WAS TRIED ANYWAY, because the circularity is worth demonstrating rather than asserting.
Widening the cut does not require more genes if the GRAPH is made denser at fixed n. Each gene
was gated by k = 2, 3 or 4 randomly chosen others, at n = 12 and n = 14, which pushed the
middle-cut edge count from 11 up to 31 -- a genuine threefold extension of the range.

THE RESULT, AND ITS VERDICT UNDER THIS PROJECT'S OWN ADMISSIBILITY RULE (G5a: a point counts
only with 4x room under its free bound):

    n    free bound   midE range   r@1e-6      r/bound      r@1e-10    admissible
    12       64         10-29       45-62     0.70-0.97      63-64        0/18
    14      128         14-31      72-112     0.56-0.88     126-128       0/18

    0 of 36 points admissible.

Every point sits between 56% and 97% of its free bound at 1e-6 and exactly AT the bound at
1e-10. The rank is not responding to cut width at all in this regime; it is pinned against the
state space. So the extension succeeded in widening the cut and failed to widen the
MEASUREMENT, which is the informative outcome: past roughly 10 cut edges there is no reachable
system in which the rank is free to be smaller than its ceiling.

AND THE FITS THE RUN PRINTED ARE VOID, recorded here because a number that has been computed
tends to get quoted. Pooled over these points, least squares returns

    LINEAR       r = 79.7 - 0.33 * midE     R^2 = 0.008
    EXPONENTIAL  r = 2^(6.22 - 0.005*midE)  R^2 = -0.014

A NEGATIVE slope would read as "rank falls as the graph gets denser", which is false. It is an
artifact of pooling two different ceilings -- 64 at n = 12 and 128 at n = 14 -- while the
response is stuck against both, so the fit is measuring the mix of n in the sample and nothing
else. Neither fit is evidence for a linear or an exponential law, and the R^2 values say so
plainly: one explains 0.8% of the variance and the other explains less than none.

WHAT STANDS AFTERWARDS. The widest honest observation remains the one from the ordering study:
at 9 middle-cut edges the median rank at 1e-6 is 48, against a generator bound of 2^9 = 512.
That says the solution does not inherit the generator's factorisation, and it says the
generator bound is loose. It does not license a projection to width 200 in either direction,
and this module is the record of why no such projection can currently be earned.
"""
from __future__ import annotations

import json, sys
sys.path.insert(0, ".")
import numpy as np

HEADROOM = 4.0        # G5a, quoted not redefined


def verdict(rows):
    """Apply G5a to the extension points and report, never fit an inadmissible set."""
    out, adm_total = [], 0
    for n in sorted({r["n"] for r in rows}):
        R = [r for r in rows if r["n"] == n]
        b = 2 ** (n // 2)
        adm = sum(1 for r in R if r["r6"] <= b / HEADROOM)
        adm_total += adm
        out.append({"n": n, "bound": b,
                    "midE": [min(r["midE"] for r in R), max(r["midE"] for r in R)],
                    "r6": [min(r["r6"] for r in R), max(r["r6"] for r in R)],
                    "r10": [min(r["r10"] for r in R), max(r["r10"] for r in R)],
                    "admissible": adm, "n_points": len(R)})
    return out, adm_total


def rng(pair):
    return f"{pair[0]}-{pair[1]}"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/tt_width_reach.json"
    try:
        rows = json.load(open(path))
    except Exception:
        print(f"  no data at {path}; this module records a completed measurement.")
        return 1
    per_n, adm = verdict(rows)
    print(f"  {'n':>3s} {'bound':>6s} {'midE':>8s} {'r@1e-6':>9s} {'r/bound':>11s} "
          f"{'r@1e-10':>9s} {'admissible':>11s}")
    for p in per_n:
        lo, hi = p["r6"]
        b = p["bound"]
        print(f"  {p['n']:3d} {b:6d} {rng(p['midE']):>8s} {rng(p['r6']):>9s} "
              f"{lo / b:5.2f}-{hi / b:<5.2f} {rng(p['r10']):>9s} "
              f"{p['admissible']:>5d}/{p['n_points']:<5d}")

    print(f"\n  {adm}/{len(rows)} points admissible under G5a.")
    if adm == 0:
        print("  VOID: the rank is pinned against its free bound at every reachable width, so")
        print("  no growth law is fitted and none is claimed. See this module's docstring for")
        print("  the fits the run printed and why they are artifacts of pooled ceilings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
