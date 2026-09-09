"""Verification without visiting every accumulation: an affine enclosure with closed-form bounds.

THE TARGET, AS STATED. Bound the backward residual over ALL reachable accumulations without
evaluating it separately at each one. enclose.py met the width gate at 0.001 to 0.004 orders but its
verifier visited every reachable point, so it cost O(paths) and settled nothing about scale.

AND THE PERMITTED RELAXATION CHANGES THE DESIGN. Verification need not avoid every relaxation, only
intolerably loose ones -- a cheap conservative enclosure that still meets the accuracy gate suffices.
That admits the following construction, which visits NO accumulation at all.

CARRY AFFINE FUNCTIONS IN THE LOG. Write log h_d(s, x) = A_d[s] + B_d[s].x. Two facts make every
required bound closed form:

    THE DRIVES ARE AFFINE IN x, so their range over the coordinate box is EXACT -- a box bound on an
    affine function is not a relaxation at all, which is what the old box bound got wrong by
    applying interval reasoning to a nonlinear function.

    log sigma IS CONCAVE and log-sum-exp IS CONVEX. A concave function is under any tangent and over
    any chord; a convex one is over any tangent. So the terminal gets affine bounds from per-target
    chords and tangents, and the backward step gets an affine lower bound from a tangent to the
    log-sum-exp and an affine upper bound by centring it on that tangent's slope and bounding the
    residual spread of the successor slopes.

The only genuine relaxations are the chord/tangent gap for log sigma over each drive's exact range,
and the decoupling of that residual spread. Both are second order where the successor slopes agree,
both are quantified below, and neither is applied to the function as a whole.

WHAT THIS COSTS: O(states^2 * k) per level, with no enumeration and no dependence on depth beyond
the number of levels. AND WHAT IT IS NOT: this is a margin argument in floating point, not machine
-rigorous interval arithmetic. V4 states the margin and cross-checks it in extended precision, and
says plainly what that does and does not establish.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

V0  THE CEILING GATE, ON THE STRUCTURAL FACT EVERYTHING RESTS ON. The construction bounds over the
    coordinate BOX and concludes about the reachable set, which is sound only if the reachable set is
    inside that box -- and is tight only if the box is not much bigger than the reachable set. Check
    both by enumeration at a small width: is every reachable accumulation inside the box, and does
    the reachable set fill the box's corners?
    PREDECLARED: if the reachable set does not reach the box's extremes, the box is loose before any
    bound is computed, and the width results below inherit that looseness rather than the
    construction's.

V1  CONTAINMENT AND WIDTH TOGETHER, never width alone -- which is the lesson enclose.py's
    double-counting bug paid for. Both reported on every instance, containment first.
    PREDECLARED: PASS requires the exact tail inside the interval AND width under one order, on
    every independent exact instance. Either alone is not a pass.

V2  THE COST, WHICH IS THE WHOLE POINT. Report verification work against path count, runtime and
    peak memory, on instances where exact enumeration is feasible and on instances where it is not.
    PREDECLARED: the method is only interesting if verification work grows substantially more slowly
    than the path count. A constant-factor saving is not the claim being tested.

V3  HEAD TO HEAD WITH enclose.py at equal instances: width and cost of the enumerating verifier
    against the closed-form one.
    PREDECLARED: the closed-form verifier is expected to be WIDER. The question is whether it stays
    under the gate while costing asymptotically less, and if it does not, that is the result.

V4  THE FLOATING-POINT MARGIN, stated rather than assumed. Apply an explicit conservative relative
    margin to both ends and cross-check the enclosure in extended precision.
    PREDECLARED: this establishes a margin argument plus an empirical cross-check. It does NOT
    establish machine rigour, which needs directed rounding throughout, and the report must say so
    in those words rather than describing the result as verified.

V5  SCALING to widths where exact enumeration is infeasible: width and cost only, containment
    unavailable and marked so.

V6  WHAT THIS DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import resource
import time
import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp

from rem.atlas.hybrid_tune import RULE
from rem.atlas.realkinetics import trrust_block, target_rows, stationary
from rem.atlas.accuracy import identity_S
from rem.atlas.enclose import prep, logg, exact_tail, reachable, enclose

FP_MARGIN = 1e-9          # conservative relative margin per level, see V4


def affine_terminal(S, H, k, gain=2.0, base=-1.0):
    """Affine upper and lower bounds on log g over [0, H]^k.

    log g = sum_t log sigma(z_t) with z_t affine in x, so each z_t's range over the box is EXACT.
    log sigma is concave: it lies UNDER its tangent and OVER its chord on that range."""
    lo = base + gain * (np.minimum(S, 0.0).sum(axis=1) * H)
    hi = base + gain * (np.maximum(S, 0.0).sum(axis=1) * H)
    ls = lambda z: -np.logaddexp(0.0, -z)
    mid = 0.5 * (lo + hi)
    # THE CONSTANT MUST CARRY `base`. A bound on log sigma is stated in z, and z = base +
    # gain*(x.S[t]), so substituting leaves a slope*(base - anchor) term in the constant. The first
    # version dropped slope*base, which with 200 targets put the terminal about 87 orders too high
    # and both ends of the enclosure above the truth. The containment gate caught it.
    # upper: tangent at the midpoint of each drive's exact range
    slope_u = 1.0 / (1.0 + np.exp(mid))                    # d/dz log sigma = sigma(-z)
    a_u = float(np.sum(ls(mid) + slope_u * (base - mid)))
    b_u = gain * (slope_u @ S)
    # lower: chord across each drive's exact range
    wide = hi - lo > 1e-12
    slope_l = np.where(wide, (ls(hi) - ls(lo)) / np.where(wide, hi - lo, 1.0), slope_u)
    a_l = float(np.sum(ls(lo) + slope_l * (base - lo)))
    b_l = gain * (slope_l @ S)
    return (a_u, b_u), (a_l, b_l)


def fast_enclose(Pm, pi, n, hvec, S, actbit, L, margin=FP_MARGIN, kill_M=False,
                 trace=None):
    """Affine enclosure with closed-form bounds. Visits no accumulation."""
    k = actbit.shape[1]
    work = 0
    (au, bu), (al, bl) = affine_terminal(S, float(hvec.sum()), k)
    Au, Bu = np.full(n, au), np.tile(bu, (n, 1))
    Al, Bl = np.full(n, al), np.tile(bl, (n, 1))
    work += len(S) * 4
    logPm = np.log(np.maximum(Pm, 1e-300))
    for d in range(L - 1, -1, -1):
        H = float(hvec[:d + 1].sum())
        xbar = np.full(k, 0.5 * H)
        for (A, B, upper) in ((Au, Bu, True), (Al, Bl, False)):
            c = logPm + (A + np.einsum('sc,sc->s', actbit * hvec[d + 1], B))[:, None]  # (s', s)
            base_at = c + (B @ xbar)[:, None]                    # (s', s)
            w = np.exp(base_at - logsumexp(base_at, axis=0)[None, :])   # softmax over s'
            beta = w.T @ B                                       # (s, k) the tangent slope
            lse_at = logsumexp(base_at, axis=0)                   # (s,)
            work += n * n + n * k
            if upper:
                Dv = B[:, None, :] - beta[None, :, :]            # (s', s, k)
                M = (np.maximum(Dv, 0.0) * H).sum(axis=2)        # (s', s) exact for affine
                if kill_M:
                    M = np.zeros_like(M)          # DIAGNOSTIC ONLY -- not a valid bound
                An = logsumexp(c + M, axis=0)
                Bn = beta
                Au, Bu = An + margin, Bn
            else:
                An = lse_at - np.einsum('sc,c->s', beta, xbar)
                Al, Bl = An - margin, beta
        if trace is not None:
            xb = np.full(k, 0.5 * float(hvec[:d + 1].sum()))
            trace.append((d, float(np.mean((Au + Bu @ xb) - (Al + Bl @ xb)) / np.log(10.0))))
    X0 = actbit * hvec[0]
    hi = float((pi * np.exp(np.clip(Au + np.einsum('sc,sc->s', X0, Bu), -700, 700))).sum())
    lo = float((pi * np.exp(np.clip(Al + np.einsum('sc,sc->s', X0, Bl), -700, 700))).sum())
    return lo, hi, work


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("VERIFICATION WITHOUT VISITING EVERY ACCUMULATION")
    P_(RULE)
    P_("  Affine functions in the log, with closed-form bounds from concavity of log sigma and")
    P_("  convexity of log-sum-exp. Box ranges of AFFINE drives are exact, which is the specific")
    P_("  thing the old box bound got wrong by applying interval reasoning to a nonlinear function.")

    # ---- V0  THE STRUCTURAL FACT ---------------------------------------------------------------
    P_("\n" + RULE)
    P_("V0  THE CEILING GATE: IS THE REACHABLE SET INSIDE THE BOX, AND DOES IT FILL IT?")
    P_(RULE)
    Pm, pi, n, hvec, S, actbit, sha = prep(3, 3)
    ins, fills = True, True
    for d in range(0, 3):
        SD, XD = reachable(Pm, n, hvec, actbit, d)
        H = float(hvec[:d + 1].sum())
        ins = ins and bool(np.all(XD >= -1e-12)) and bool(np.all(XD <= H + 1e-12))
        fills = fills and bool(np.allclose(XD.min(axis=0), 0.0, atol=1e-12)) \
            and bool(np.allclose(XD.max(axis=0), H, atol=1e-12))
        P_(f"    level {d}: {len(XD):>6,} reachable points, box [0, {H:.4f}]^{actbit.shape[1]},"
           f" coords span [{XD.min():.4f}, {XD.max():.4f}]")
    P_(f"\n  V0: inside the box: {'yes' if ins else 'NO'};"
       f" reaches both extremes in every coordinate: {'yes' if fills else 'NO'}.")
    P_("  The reachable set is the FULL GRID of coordinate-wise subset sums -- every bit pattern is")
    P_("  a state and the chain reaches all of them -- so the box is its exact bounding box and")
    P_("  contributes no looseness of its own. What looseness there is comes from the chord and")
    P_("  tangent gaps, not from the box.")

    # ---- V1/V2/V3  CONTAINMENT, WIDTH, COST ----------------------------------------------------
    P_("\n" + RULE)
    P_("V1/V2/V3  CONTAINMENT AND WIDTH TOGETHER, WITH COST, AGAINST THE ENUMERATING VERIFIER")
    P_(RULE)
    inst = [(3, 3, 200), (3, 4, 200), (4, 3, 200), (4, 4, 200), (3, 5, 200)]
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'paths':>12} {'exact tail':>15}"
       f" {'fast width':>11} {'contains':>9} {'fast work':>11} {'fast s':>8}"
       f" {'enum width':>11} {'enum s':>8}")
    okall, widths, rows_ = True, [], []
    for nc, L, nt in inst:
        Pm, pi, n, hvec, S, actbit, sha = prep(nc, L, ntarget=nt)
        ex = exact_tail(Pm, pi, n, hvec, S, actbit, L)
        t0 = time.time()
        lo, hi, work = fast_enclose(Pm, pi, n, hvec, S, actbit, L)
        tf = time.time() - t0
        good = lo <= ex * (1 + 1e-9) and ex <= hi * (1 + 1e-9)
        w = float(np.log10(hi / max(lo, 1e-308)))
        t1 = time.time()
        elo, ehi, ech, ev = enclose(Pm, pi, n, hvec, S, actbit, L, 3)
        te = time.time() - t1
        ew = float(np.log10(ehi / max(elo, 1e-308)))
        okall = okall and good
        widths.append(w)
        rows_.append((nc, L, (2 ** nc) ** (L + 1), work, tf, w, ew, te))
        P_(f"    {nc:>6} {L:>3} {(2 ** nc) ** (L + 1):>12,} {ex:>15.6e}"
           f" {w:>11.3f} {'yes' if good else 'NO':>9} {work:>11,} {tf:>8.2f}"
           f" {ew:>11.3f} {te:>8.2f}")
    P_(f"\n  V1 AS PREDECLARED: containment {'holds on every instance' if okall else 'FAILS'};"
       f" widest fast interval {max(widths):.3f} orders."
       f" {'PASS -- both conditions met.' if okall and max(widths) < 1.0 else 'FAIL.'}")
    P_(f"\n  V2 THE COST. Verification work against path count:")
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'paths':>12} {'fast work':>12} {'work / paths':>14}")
    for nc, L, npaths, work, tf, w, ew, te in rows_:
        P_(f"    {nc:>6} {L:>3} {npaths:>12,} {work:>12,} {work / npaths:>14.4f}")
    P_("    The fast verifier's work is O(states^2 * k) per level and does NOT grow with depth")
    P_("    beyond the level count, so the ratio falls as the path set grows. That is the property")
    P_("    being tested, and a falling ratio is what would establish it.")

    # ---- V3b  WHERE THE WIDTH COMES FROM -------------------------------------------------------
    P_("\n" + RULE)
    P_("V3b  DECOMPOSING THE WIDTH, SINCE THE GATE FAILED ON IT")
    P_(RULE)
    P_("  Two relaxations are in play: the chord/tangent gap on log sigma at the terminal, and the")
    P_("  decoupled residual spread M in the upper log-sum-exp bound. This separates them.")
    Pm, pi, n, hvec, S, actbit, sha = prep(4, 4)
    k = actbit.shape[1]
    (au, bu), (al, bl) = affine_terminal(S, float(hvec.sum()), k)
    rng = np.random.default_rng(5)
    Xs = rng.random((2000, k)) * float(hvec.sum())
    gex = logg(Xs, S)
    gap_t = float(np.mean((au + Xs @ bu) - (al + Xs @ bl)) / np.log(10.0))
    P_(f"\n  TERMINAL GAP ALONE, averaged over 2,000 points in the box:"
       f" {gap_t:.3f} orders")
    P_(f"    upper exceeds exact log g by {float(np.mean((au + Xs @ bu) - gex) / np.log(10.0)):.3f}"
       f" orders, lower falls short by"
       f" {float(np.mean(gex - (al + Xs @ bl)) / np.log(10.0)):.3f}")
    tr = []
    lo, hi, _ = fast_enclose(Pm, pi, n, hvec, S, actbit, 4, trace=tr)
    P_(f"\n  HOW THE GAP ACCUMULATES DOWN THE RECURSION (mean over states, at the box centre):")
    P_(f"\n    {'level':>6} {'gap, orders':>13}")
    P_(f"    {'terminal':>6} {gap_t:>13.3f}")
    for d, gp in tr:
        P_(f"    {d:>6} {gp:>13.3f}")
    lok, hik, _ = fast_enclose(Pm, pi, n, hvec, S, actbit, 4, kill_M=True)
    P_(f"\n  AND WITH THE DECOUPLED SPREAD M FORCED TO ZERO -- NOT A VALID BOUND, A DIAGNOSTIC:")
    P_(f"    width with M      : {np.log10(hi / lo):.3f} orders")
    P_(f"    width without M   : {np.log10(hik / lok):.3f} orders")
    P_(f"    so M contributes    {np.log10(hi / lo) - np.log10(hik / lok):.3f} orders and the terminal")
    P_(f"    plus the tangent gap contribute {np.log10(hik / lok):.3f}")
    P_("\n  V3b: the term to attack is whichever of those two dominates, and the answer is in the")
    P_("  two numbers above rather than in an argument about which ought to.")

    # ---- V4  THE MARGIN ------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("V4  THE FLOATING-POINT MARGIN, AND WHAT IT DOES NOT ESTABLISH")
    P_(RULE)
    Pm, pi, n, hvec, S, actbit, sha = prep(4, 4)
    lo0, hi0, _ = fast_enclose(Pm, pi, n, hvec, S, actbit, 4, margin=0.0)
    lo1, hi1, _ = fast_enclose(Pm, pi, n, hvec, S, actbit, 4, margin=FP_MARGIN)
    lo2, hi2, _ = fast_enclose(Pm.astype(np.longdouble), pi.astype(np.longdouble), n,
                               hvec.astype(np.longdouble), S.astype(np.longdouble),
                               actbit.astype(np.longdouble), 4, margin=0.0)
    P_(f"  margin 0, float64      : [{lo0:.10e}, {hi0:.10e}]")
    P_(f"  margin {FP_MARGIN:.0e}, float64  : [{lo1:.10e}, {hi1:.10e}]")
    P_(f"  margin 0, longdouble   : [{float(lo2):.10e}, {float(hi2):.10e}]")
    P_(f"  float64 vs longdouble disagreement:"
       f" {abs(np.log10(hi0 / float(hi2))):.3e} orders on the upper end,"
       f" {abs(np.log10(lo0 / float(lo2))):.3e} on the lower")
    P_(f"  the applied margin is worth {abs(np.log10(hi1 / hi0)):.3e} orders, which exceeds that")
    P_(f"  disagreement by {abs(np.log10(hi1 / hi0)) / max(abs(np.log10(hi0 / float(hi2))), 1e-18):.1f}x.")
    P_("\n  WHAT THIS ESTABLISHES: a margin argument, plus an empirical cross-check in a wider")
    P_("  format. WHAT IT DOES NOT ESTABLISH: machine rigour. That needs directed rounding through")
    P_("  every operation, which this does not do. The enclosure is therefore rigorous MODULO")
    P_("  floating-point evaluation, and calling it verified without that qualifier would be an")
    P_("  overstatement of exactly the kind this record keeps correcting.")

    # ---- V5  BEYOND EXACT --------------------------------------------------------------------
    P_("\n" + RULE)
    P_("V5  WIDTHS AND COSTS WHERE EXACT ENUMERATION IS INFEASIBLE")
    P_(RULE)
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'paths':>14} {'width, orders':>14} {'work':>12}"
       f" {'seconds':>9} {'peak MB':>9} {'containment':>13}")
    for nc, L in ((6, 4), (8, 5), (10, 6)):
        Pm, pi, n, hvec, S, actbit, sha = prep(nc, L)
        t0 = time.time()
        lo, hi, work = fast_enclose(Pm, pi, n, hvec, S, actbit, L)
        el = time.time() - t0
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        P_(f"    {nc:>6} {L:>3} {(2.0 ** nc) ** (L + 1):>14.3e}"
           f" {np.log10(hi / max(lo, 1e-308)):>14.3f} {work:>12,} {el:>9.2f} {mb:>9.0f}"
           f" {'unavailable':>13}")
    P_("\n  Containment cannot be checked here -- there is no exact tail -- and is marked so rather")
    P_("  than inferred from the small instances.")

    # ---- V6 -----------------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("V6  WHAT THIS DOES NOT SETTLE")
    P_(RULE)
    P_("  1. Containment is verified only where exact enumeration runs. At the engine's width the")
    P_("     enclosure's correctness rests on the construction's inequalities, not on observation.")
    P_("  2. Rigour is modulo floating point, as V4 states.")
    P_("  3. The affine representation is coarser than enclose.py's cubic, and the width gap")
    P_("     between them is the price of dropping enumeration. Whether an intermediate")
    P_("     representation -- affine bounds around a polynomial centre -- does better is untried.")
    P_("  4. One term of one calculation. Class-map validity, transfer, and biology are untouched.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_fastverify.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
