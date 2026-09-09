"""How accurate is backpoly's full-tail estimate, really?

WHAT IS RESTING ON IT. backpoly's engine-width estimate of 1.58e-95 was used to argue that the
remaining pruning rise is 2.17 orders rather than the 10.22 accuracy.py's slope extrapolation
gave, and therefore that the pruning error never reaches the class map's 5.08 orders and the
RANKING OF THE ENGINE'S ERROR TERMS REVERSES. That is a large conclusion resting on a number whose
accuracy was measured at ONE width and ONE depth -- four controllers, L = 4 -- and asserted nowhere
else. backpoly's own G5 said so. This measures it.

THREE THINGS HAVE TO BE TRUE FOR THAT ESTIMATE TO CARRY THE ARGUMENT, and they are separable:
the estimate must be stable to its own arbitrary choices; it must have converged in the
representation; and its error must not grow so fast with width and depth that four controllers
says nothing about ten. Each gets a gate, and the third is where the argument is most exposed.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

H0  THE CEILING GATE: IS THE ESTIMATE STABLE TO CHOICES THAT SHOULD NOT MATTER? The collocation
    points are drawn at random and their number is a free multiple of the coefficient count.
    Neither should move the answer.
    PREDECLARED: the reversal argument needs the estimate to distinguish 2.17 orders from 5.08, a
    margin of 2.91 orders. If the spread across seeds and point counts is comparable to that
    margin, the estimate cannot support the reversal WHATEVER its bias, and this module stops
    there and says so.

H1  CONVERGENCE IN THE REPRESENTATION, at the engine's own width. backpoly reported degrees 2 and
    3 agreeing to 2.4% and called the representation converged. Two points do not establish
    convergence.
    PREDECLARED: convergence is claimed only if successive degrees move by a SHRINKING amount. If
    the changes are flat or growing, 2.4% was a coincidence between two adjacent degrees and the
    estimate is not converged.

H2  THE ERROR LAW IN WIDTH AND DEPTH, which is the gate that matters. Measure the error against
    exact enumeration at every (nCtrl, L) where exact is computable, and fit it in BOTH variables
    separately -- one variable giving two exponents is the confound this record has now corrected
    three times.
    PREDECLARED: extrapolate the law to ten controllers and L = 6 and report an INTERVAL for the
    engine-width error. If that interval reaches 2.91 orders the reversal is not supported; if it
    stays far below, it is. The interval is the deliverable, not the point.

H3  THE DIRECTION OF THE ERROR. The reversal needs the estimate not to be biased LOW: a low
    estimate understates the true tail and so understates the pruning error, which is exactly the
    quantity the argument turns on.
    PREDECLARED: report the SIGN of the error at every measured point. A consistent sign is a bias
    and must be stated as one; if the bias is negative the reversal argument is weakened by its own
    instrument and that has to be said plainly.

H4  THE HONEST NUMBER, assembled: the engine-width estimate with an interval, and what that does
    to the reversal.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.backpoly import setup, logF, poly_backward, mono

PRUNED_BOUND = 1.0640e-97
CLASS_LADDER = 5.08
SLOPE_MARGIN = 5.08 - 2.17      # the margin the reversal argument needs the estimate to resolve


def exact_tail(Pm, pi, n, hvec, S, actbit, L):
    last = np.arange(n)
    w = pi.copy()
    acc = actbit * hvec[0]
    for dd in range(1, L + 1):
        col = Pm[:, last].T * w[:, None]
        w = col.ravel()
        par = np.repeat(np.arange(len(last)), n)
        chd = np.tile(np.arange(n), len(last))
        acc = acc[par] + actbit[chd] * hvec[dd]
        last = chd
    return float((w * np.exp(logF(acc, S))).sum())


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("HOW ACCURATE IS backpoly's FULL-TAIL ESTIMATE?")
    P_(RULE)
    P_(f"  The reversal argument needs the estimate to distinguish 2.17 orders from"
       f" {CLASS_LADDER}, a margin of {SLOPE_MARGIN:.2f} orders. Everything below is measured against that.")

    # ---- H0  STABILITY -------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("H0  THE CEILING GATE: IS IT STABLE TO CHOICES THAT SHOULD NOT MATTER?")
    P_(RULE)
    Pm2, pi2, n2, hv2, S2, ab2, rows2, sha2 = setup(10, 6)
    P_(f"  10 controllers, {n2} states, L = 6. sha {sha2[:12]}.")
    P_(f"\n    {'degree':>7} {'points/coef':>12} {'seed':>6} {'tail':>15} {'log10':>10}")
    vals = {}
    for deg in (2, 3):
        for mult in (3, 4, 6):
            for seed in (7, 101, 20260909):
                if deg == 3 and (mult != 4 or seed == 20260909):
                    continue                      # degree 3 is expensive; sample its grid
                t, ncf, npt = poly_backward(Pm2, pi2, n2, hv2, S2, ab2, 6, deg,
                                            npts_mult=mult, seed=seed)
                vals.setdefault(deg, []).append(t)
                P_(f"    {deg:>7} {mult:>12} {seed:>6} {t:>15.6e} {np.log10(t):>10.4f}")
    spreads = {d: float(np.log10(max(v)) - np.log10(min(v))) for d, v in vals.items() if len(v) > 1}
    worst = max(spreads.values()) if spreads else 0.0
    for d, sp in sorted(spreads.items()):
        P_(f"\n    degree {d}: spread across choices = {sp:.4f} orders")
    P_(f"\n  H0: worst spread {worst:.4f} orders against the {SLOPE_MARGIN:.2f}-order margin --"
       f" {'STABLE, the arbitrary choices do not move the answer.' if worst < 0.1 * SLOPE_MARGIN else 'NOT STABLE ENOUGH; the estimate cannot support the reversal whatever its bias.'}")
    if worst >= 0.1 * SLOPE_MARGIN:
        with open(os.path.join(os.path.dirname(__file__), "RESULTS_backpolyerr.txt"), "w") as fh:
            fh.write("\n".join(out) + "\n")
        return

    # ---- H1  CONVERGENCE -----------------------------------------------------------------------
    P_("\n" + RULE)
    P_("H1  CONVERGENCE IN DEGREE, AT THE ENGINE'S WIDTH")
    P_(RULE)
    P_("  Two adjacent degrees agreeing is not convergence. The successive changes must SHRINK.")
    P_(f"\n    {'degree':>7} {'coefficients':>13} {'tail':>15} {'change, orders':>16} {'seconds':>9}")
    seq = []
    for deg in (1, 2, 3, 4):
        t0 = time.time()
        try:
            t, ncf, npt = poly_backward(Pm2, pi2, n2, hv2, S2, ab2, 6, deg,
                                        npts_mult=(2 if deg == 4 else 4))
        except Exception as e:                                  # pragma: no cover
            P_(f"    {deg:>7}  failed: {e}")
            continue
        el = time.time() - t0
        ch = "" if not seq else f"{np.log10(t) - np.log10(seq[-1][1]):+.4f}"
        seq.append((deg, t, ncf))
        P_(f"    {deg:>7} {ncf:>13} {t:>15.6e} {ch:>16} {el:>9.1f}")
    if len(seq) >= 3:
        d = [abs(np.log10(seq[i + 1][1]) - np.log10(seq[i][1])) for i in range(len(seq) - 1)]
        shrink = all(b <= a * 1.05 for a, b in zip(d, d[1:]))
        P_(f"\n    successive changes: " + ", ".join(f"{x:.4f}" for x in d))
        P_(f"  H1: {'the changes SHRINK -- converged.' if shrink else 'the changes do NOT shrink monotonically, so the representation is not demonstrably converged and the 2.4%% agreement backpoly reported was between two adjacent degrees only.'}")

    # ---- H2  THE ERROR LAW ---------------------------------------------------------------------
    P_("\n" + RULE)
    P_("H2  THE ERROR LAW IN WIDTH AND DEPTH, AGAINST EXACT ENUMERATION")
    P_(RULE)
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'paths':>12} {'exact':>15} {'poly deg3':>15} {'err, orders':>12} {'sign':>6}")
    pts = []
    for nc, ll in ((3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (5, 3)):
        npaths = (2 ** nc) ** (ll + 1)
        if npaths > 3_000_000:
            continue
        Pm, pi, n, hvec, S, actbit, rows, sha = setup(nc, ll)
        ex = exact_tail(Pm, pi, n, hvec, S, actbit, ll)
        tl, ncf, npt = poly_backward(Pm, pi, n, hvec, S, actbit, ll, 3)
        er = float(np.log10(tl) - np.log10(ex))
        pts.append((nc, ll, ex, tl, er))
        P_(f"    {nc:>6} {ll:>3} {npaths:>12,} {ex:>15.6e} {tl:>15.6e} {abs(er):>12.4f}"
           f" {'+' if er > 0 else '-':>6}")
    A = np.column_stack([np.ones(len(pts)),
                         [p[0] for p in pts],
                         [p[1] for p in pts]])
    y = np.log10([abs(p[4]) for p in pts])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1.0 - np.sum((y - A @ coef) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
    P_(f"\n    log10|error| = {coef[0]:.3f} + {coef[1]:.3f}*nCtrl + {coef[2]:.3f}*L      R^2 = {r2:.3f}")
    pred = 10 ** (coef[0] + coef[1] * 10 + coef[2] * 6)
    resid = float(np.std(y - A @ coef))
    lo, hi = pred * 10 ** (-2 * resid), pred * 10 ** (2 * resid)
    P_(f"\n    extrapolated to 10 controllers and L = 6: {pred:.4f} orders,"
       f" 2-sigma interval [{lo:.4f}, {hi:.4f}]")
    P_(f"\n  H2 AS PREDECLARED: the interval's upper end is {hi:.4f} orders against the"
       f" {SLOPE_MARGIN:.2f}-order margin.")
    P_(f"  {'IT STAYS FAR BELOW, so the reversal is supported by the error law.' if hi < SLOPE_MARGIN else 'IT REACHES THE MARGIN, so the error law does NOT support the reversal and backpoly s conclusion must be withdrawn to a direction rather than a number.'}")
    P_("  The fit spans nCtrl 3 to 5 and L 3 to 5, and the engine sits at 10 and 6 -- five")
    P_("  controllers beyond the widest point. This is an extrapolation, and its whole purpose is")
    P_("  to say whether the margin is reached, not to predict a value.")

    # ---- H3  THE DIRECTION ---------------------------------------------------------------------
    P_("\n" + RULE)
    P_("H3  THE DIRECTION OF THE ERROR, WHICH THE ARGUMENT IS SENSITIVE TO")
    P_(RULE)
    signs = [np.sign(p[4]) for p in pts]
    P_(f"  signs at the {len(pts)} measured points: " + " ".join('+' if s > 0 else '-' for s in signs))
    allpos, allneg = all(s > 0 for s in signs), all(s < 0 for s in signs)
    P_(f"\n  H3: {'the estimate is consistently HIGH.' if allpos else ('the estimate is consistently LOW.' if allneg else 'the sign is mixed -- no consistent bias.')}")
    if allneg:
        P_("  A LOW estimate understates the true tail and so understates the pruning error, which")
        P_("  is exactly the quantity the reversal turns on. The argument is weakened by its own")
        P_("  instrument and the reversal should be read as an upper limit on the pruning error,")
        P_("  not an estimate of it.")
    elif allpos:
        P_("  A HIGH estimate overstates the true tail and so OVERSTATES the pruning error. The")
        P_("  reversal argument survives that direction: the true pruning error would be smaller")
        P_("  still, which strengthens rather than weakens the conclusion that it never reaches")
        P_(f"  {CLASS_LADDER} orders.")

    # ---- H4  THE HONEST NUMBER -----------------------------------------------------------------
    P_("\n" + RULE)
    P_("H4  THE HONEST NUMBER")
    P_(RULE)
    base = seq[-1][1] if seq else float("nan")
    P_(f"  engine-width estimate: {base:.4e}, stable to {worst:.4f} orders across collocation")
    P_(f"  choices, with an extrapolated error interval of up to {hi:.4f} orders.")
    imp = float(np.log10(base) - np.log10(PRUNED_BOUND))
    P_(f"\n  implied remaining pruning rise: {imp:.2f} orders, +/- {hi:.2f} from the error law")
    P_(f"  the class map's ladder, which it must not reach for the reversal to hold: {CLASS_LADDER}")
    P_(f"\n  So the pruning error is {max(imp - hi, 0):.2f} to {imp + hi:.2f} orders --"
       f" {'entirely below the class ladder, and the reversal holds.' if imp + hi < CLASS_LADDER else 'and the interval REACHES the class ladder, so the ranking cannot be called either way.'}")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_backpolyerr.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
