"""Partition the shared accumulation, propagate a bound per region WITH its probability accounting.

WHY PARTITION THE SHARED STATE RATHER THAN THE DRIVES. vertexbound established that the terminal's
looseness is not target-decoupling -- exploiting the coupling across targets was worth 1.33x and the
target-count exponent stayed at 1.16. What remains is the CURVATURE of log sigma across a drive
range of width about two, which no affine bound escapes. Narrowing that range means splitting the
domain, and splitting 200 drives independently is 2^200. But every drive is a function of the SAME
accumulation, so splitting ONE accumulation coordinate narrows EVERY drive at once. That is the
whole idea and it is why the region count can stay small.

THE PART THAT IS NOT BOOKKEEPING. A terminal enclosure that is piecewise in the accumulation is a
sum of indicator-weighted pieces, and the backward operator cannot simply drop those indicators.
Applying it soundly means either computing region-to-region transitions exactly, or enclosing them
while PRESERVING THE TRANSITION PROBABILITIES and accounting conservatively wherever a successor
crosses a region boundary. This does the second: the probabilities Pm are used exactly and only
region MEMBERSHIP is relaxed, by taking the extreme over every region the shifted region can reach.
Probability is never reassigned to whichever region gives the preferred answer -- the only freedom
taken is which bound within the reachable regions, and it is taken pessimistically on both sides.

WHAT THIS EXPERIMENT MUST DISTINGUISH, and it is the reason the diagnostic below is not optional:
terminal gaps shrinking while transition bounds widen looks like progress and is not. Only the final
interval shrinking advances anything, so both are tracked at every region count.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import time
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.enclose import prep, logg, exact_tail, reachable

BASE, GAIN = -1.0, 2.0


def region_boxes(cuts, k):
    """cuts: dict coordinate -> threshold. Region index is a bitmask over the split coordinates."""
    js = sorted(cuts)
    R = 1 << len(js)
    lo = np.zeros((R, k))
    hi = np.ones((R, k))
    for r in range(R):
        for b, j in enumerate(js):
            if (r >> b) & 1:
                lo[r, j] = cuts[j]
            else:
                hi[r, j] = cuts[j]
    return js, lo, hi


def overlap_map(js, lo, hi, k, delta):
    """Which regions can a source region reach once shifted by delta? Exact interval overlap."""
    R = len(lo)
    out = []
    for r in range(R):
        slo, shi = lo[r] + delta, hi[r] + delta
        ok = np.ones(R, dtype=bool)
        for j in js:
            ok &= (lo[:, j] <= shi[j] + 1e-15) & (hi[:, j] >= slo[j] - 1e-15)
        out.append(np.nonzero(ok)[0])
    return out


def terminal_bounds(S, lo, hi, k):
    """Per-region constant enclosure of log g, from per-target chord below and tangent above on that
    region's EXACT drive interval. Both are affine, so their extremes over the box are exact."""
    R = len(lo)
    L = np.zeros(R)
    U = np.zeros(R)
    gaps = np.zeros(R)
    ls = lambda z: -np.logaddexp(0.0, -z)
    for r in range(R):
        c, w = 0.5 * (lo[r] + hi[r]), 0.5 * (hi[r] - lo[r])
        mid = BASE + GAIN * (S @ c)
        rad = GAIN * (np.abs(S) @ w)
        a, b = mid - rad, mid + rad                     # exact drive range on this region
        m = 0.5 * (a + b)
        su = 1.0 / (1.0 + np.exp(m))                    # tangent slope
        wide = b - a > 1e-12
        sl = np.where(wide, (ls(b) - ls(a)) / np.where(wide, b - a, 1.0), su)
        # upper affine in z: ls(m) + su (z - m); lower: ls(a) + sl (z - a). Extremes over the box
        # are exact because z is affine in x.
        up = ls(m) + su * (mid - m) + np.abs(su) * rad
        lw = ls(a) + sl * (mid - a) - np.abs(sl) * rad
        U[r], L[r] = float(up.sum()), float(lw.sum())
        gaps[r] = float((up - lw).sum())
    return L, U, gaps


def curvature_score(S, lo, hi):
    """sum_j M/8 * (b-a)^2 with M bounding |f''| = sigma sigma(-.) <= 1/4 on the interval."""
    tot = []
    for r in range(len(lo)):
        c, w = 0.5 * (lo[r] + hi[r]), 0.5 * (hi[r] - lo[r])
        rad = GAIN * (np.abs(S) @ w)
        tot.append(float(np.sum(0.25 / 8.0 * (2 * rad) ** 2)))
    return float(np.sum(tot))


def run_regions(Pm, pi, n, hvec, S, actbit, L, cuts):
    """Backward propagation of a constant lower and upper bound per (state, region)."""
    k = actbit.shape[1]
    js, lo, hi = region_boxes(cuts, k)
    R = len(lo)
    Llo, Lhi, gaps = terminal_bounds(S, lo, hi, k)
    Lo = np.tile(np.exp(np.clip(Llo, -700, 700)), (n, 1))       # (state, region)
    Hi = np.tile(np.exp(np.clip(Lhi, -700, 700)), (n, 1))
    work = R * len(S) * 4
    cross_tot, cross_cnt = 0, 0
    for d in range(L - 1, -1, -1):
        Mlo = np.empty((n, R))
        Mhi = np.empty((n, R))
        for sp in range(n):
            omap = overlap_map(js, lo, hi, k, actbit[sp] * hvec[d + 1])
            for r in range(R):
                tgt = omap[r]
                cross_tot += len(tgt)
                cross_cnt += 1
                Mlo[sp, r] = Lo[sp, tgt].min()          # conservative on both sides
                Mhi[sp, r] = Hi[sp, tgt].max()
        Lo = Pm.T @ Mlo                                  # probabilities used EXACTLY
        Hi = Pm.T @ Mhi
        work += n * R + 2 * n * n * R
    X0 = actbit * hvec[0]
    idx = np.zeros(n, dtype=int)
    for b, j in enumerate(js):
        idx |= ((X0[:, j] >= cuts[j]).astype(int) << b)
    plo = float((pi * Lo[np.arange(n), idx]).sum())
    phi = float((pi * Hi[np.arange(n), idx]).sum())
    return plo, phi, work, float(np.sum(gaps)) / len(gaps), cross_tot / max(cross_cnt, 1)


def choose_cuts(S, k, m):
    """Greedy: add the coordinate whose split lowers the curvature score most."""
    cuts = {}
    for _ in range(m):
        best, bj = None, None
        for j in range(k):
            if j in cuts:
                continue
            trial = dict(cuts)
            trial[j] = 0.5
            _js, lo, hi = region_boxes(trial, k)
            sc = curvature_score(S, lo, hi)
            if best is None or sc < best:
                best, bj = sc, j
        if bj is None:
            break
        cuts[bj] = 0.5
    return cuts


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("PARTITION THE SHARED ACCUMULATION, PROPAGATE BOUNDS WITH THEIR PROBABILITY ACCOUNTING")
    P_(RULE)
    P_("  One split of one accumulation coordinate narrows EVERY target's drive at once, which is")
    P_("  why the region count can stay small where splitting 200 drives independently is 2^200.")
    P_("  The transition operator is applied with the probabilities used EXACTLY; only region")
    P_("  membership is relaxed, pessimistically on both sides.")

    # ---- R0  CORRECTNESS OF THE PIECES ---------------------------------------------------------
    P_("\n" + RULE)
    P_("R0  CORRECTNESS: DO THE TERMINAL AND TRANSITION ENCLOSURES HOLD?")
    P_(RULE)
    Pm, pi, n, hvec, S, actbit, sha = prep(4, 4)
    k = actbit.shape[1]
    cuts = choose_cuts(S, k, 3)
    js, lo, hi = region_boxes(cuts, k)
    Llo, Lhi, _g = terminal_bounds(S, lo, hi, k)
    rng = np.random.default_rng(3)
    bad_t = 0
    for r in range(len(lo)):
        X = lo[r] + rng.random((400, k)) * (hi[r] - lo[r])
        f = logg(X, S)
        bad_t += int(np.sum(f < Llo[r] - 1e-9)) + int(np.sum(f > Lhi[r] + 1e-9))
    P_(f"  terminal: {len(lo)} regions x 400 interior points, {bad_t} violations")
    SD, XD = reachable(Pm, n, hvec, actbit, 3)
    ridx = np.zeros(len(XD), dtype=int)
    for b, j in enumerate(js):
        ridx |= ((XD[:, j] >= cuts[j]).astype(int) << b)
    inbox = bool(np.all(XD >= lo[ridx] - 1e-12)) and bool(np.all(XD <= hi[ridx] + 1e-12))
    P_(f"  every reachable accumulation lies in the region its index selects: {'yes' if inbox else 'NO'}")
    P_(f"\n  R0: {'PASS' if bad_t == 0 and inbox else 'FAIL'}")

    # ---- R1  THE SWEEP -------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("R1  THE REGION-COUNT SWEEP, WITH THE FULL PROPAGATION COST")
    P_(RULE)
    P_("  Tracking the TERMINAL gap and the FINAL interval separately, because terminal gaps")
    P_("  shrinking while transition bounds widen is the failure mode this must detect.")
    ex = exact_tail(Pm, pi, n, hvec, S, actbit, 4)
    P_(f"\n  4 controllers, L = 4, 200 targets. exact tail {ex:.6e}")
    P_(f"\n    {'regions':>8} {'terminal gap':>14} {'final width':>13} {'contains':>9}"
       f" {'mean crossings':>15} {'work':>12} {'seconds':>9}")
    for m in range(0, 6):
        cuts = choose_cuts(S, k, m)
        t0 = time.time()
        plo, phi, work, tg, cross = run_regions(Pm, pi, n, hvec, S, actbit, 4, cuts)
        el = time.time() - t0
        good = plo <= ex * (1 + 1e-9) and ex <= phi * (1 + 1e-9)
        P_(f"    {1 << m:>8} {tg / np.log(10.0):>14.3f}"
           f" {np.log10(phi / max(plo, 1e-308)):>13.3f} {'yes' if good else 'NO':>9}"
           f" {cross:>15.2f} {work:>12,} {el:>9.2f}")
    P_("\n  The terminal-gap column is per region, in orders. The crossings column is how many")
    P_("  regions a shifted region reaches on average -- the transition ambiguity, and the thing")
    P_("  that grows if this approach is going to fail.")

    # ---- R2  THE GATES -------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("R2  THE GATES AT 200 TARGETS, AND THE COST AGAINST PATH ENUMERATION")
    P_(RULE)
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'regions':>8} {'paths':>13} {'width':>8} {'contains':>9}"
       f" {'work/paths':>12} {'seconds':>9}")
    for nc, Ld in ((4, 4), (6, 4), (8, 5), (10, 6)):
        Pm2, pi2, n2, hv2, S2, ab2, _s = prep(nc, Ld)
        k2 = ab2.shape[1]
        cuts2 = choose_cuts(S2, k2, 5)
        ex2 = exact_tail(Pm2, pi2, n2, hv2, S2, ab2, Ld) if (2 ** nc) ** (Ld + 1) <= 3e6 else None
        t0 = time.time()
        plo, phi, work, tg, cross = run_regions(Pm2, pi2, n2, hv2, S2, ab2, Ld, cuts2)
        el = time.time() - t0
        npaths = (2.0 ** nc) ** (Ld + 1)
        cont = "unavailable" if ex2 is None else ("yes" if plo <= ex2 * (1 + 1e-9)
                                                  and ex2 <= phi * (1 + 1e-9) else "NO")
        P_(f"    {nc:>6} {Ld:>3} {32:>8} {npaths:>13.3e}"
           f" {np.log10(phi / max(plo, 1e-308)):>8.3f} {cont:>9} {work / npaths:>12.3e} {el:>9.2f}")

    P_("\n" + RULE)
    P_("R3  WHAT THIS DOES NOT SETTLE")
    P_(RULE)
    P_("  1. The bound carried per region is a CONSTANT. That is the spec's letter and it is coarse")
    P_("     in ten dimensions: a piecewise-constant enclosure needs many regions to beat a single")
    P_("     affine one, and the sweep is where that shows.")
    P_("  2. Region membership is relaxed at boundaries by taking the extreme over reachable")
    P_("     regions. That is sound and it is the transition ambiguity the crossings column tracks.")
    P_("  3. Cuts are at coordinate midpoints, chosen greedily by curvature. Neither the position")
    P_("     nor the greedy order is optimised.")
    P_("  4. Rigour is modulo floating point, and containment is observed only where exact")
    P_("     enumeration runs.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_regions.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
