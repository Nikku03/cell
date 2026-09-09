"""Does the EXACT bound close the certificate? And can any bound of that shape?

WHERE THIS SITS. boundprune wired the target-aware bound into the pruner and won 6x to 8x in
retained paths, but its certificate -- summed dropped bound -- came out 25 orders LARGER than the
tail it is supposed to bound. Two things could be responsible. The engine-width pruner uses the
TANGENT relaxation of the bound, which gives away orders that the exact bound does not. Or the
looseness is inherent to the shape of the bound and no version of it closes.

THE SECOND QUESTION IS THE ONE THAT MATTERS, AND IT IS CHEAP TO ANSWER. Every bound in this family
has the same shape: the subtree's mass times the best ON-probability any single completion could
reach. The mass factor is EXACT -- the subtree's weights sum to the prefix weight because the
transition matrix is column stochastic. So all the looseness lives in replacing a WEIGHTED AVERAGE
of ON-probabilities over the subtree by their MAXIMUM. At a width where every completion can be
enumerated, that substitution can be priced exactly, and its price is a floor under the certificate
of any bound of this shape -- including the exact one, and including any tighter one yet to be
invented.

So this module prices the substitution first and only then spends the engine-width run.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

E0  THE CEILING GATE, WHICH IS A FLOOR UNDER THE CERTIFICATE. At 4 controllers and L = 4, where
    all 1,048,576 paths enumerate, decompose the certificate's looseness into its three steps for
    the same prefix set:
        TANGENT   / EXACT   -- what the relaxation boundprune used gives away
        EXACT     / BESTCOMP -- what relaxing real completions to the box gives away, where
                    BESTCOMP is the true maximum over ACTUAL completions, the tightest bound of
                    this shape that could ever exist
        BESTCOMP  / TRUTH   -- what replacing the subtree's weighted average by its maximum costs,
                    which no bound of this shape can avoid
    PREDECLARED: if BESTCOMP / TRUTH is itself above 1e3 -- the bar pathbound set for a useful
    certificate -- then NO bound of this shape closes the certificate, the exact bound cannot, and
    that is the answer regardless of what the engine-width run shows. If it is below 1e3, the
    shape is capable of a useful certificate and the remaining looseness is the relaxations'
    fault, which is fixable.

E1  THE LITERAL QUESTION, ANSWERED AT THE ENGINE'S WIDTH REGARDLESS OF E0, because it was asked
    and because E0's width is not the engine's. Run the pruner with rank="exact" at 10 controllers
    and L = 6 and compare its certificate against the tangent's, both as a ratio to the reported
    tail.
    PREDECLARED: the certificate CLOSES only if certificate / tail falls below 1. A certificate
    larger than the quantity it bounds is not a guarantee at any margin.

E2  AND THE RANKING, WHICH IS A SEPARATE QUESTION FROM THE CERTIFICATE. A tighter bound should
    also rank at least as well. Report the tail each ranking recovers at equal retained count.
    PREDECLARED: if the exact bound ranks WORSE than the tangent relaxation anywhere, that is a
    real and surprising result and must be reported rather than smoothed over -- a looser bound
    can rank differently, and nothing guarantees the tighter one orders candidates better.

E3  WHAT THIS DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.realkinetics import trrust_block, target_rows, engine_budget
from rem.atlas.pathbound import setup, enumerate_all, prefix_level, logon, bound_of


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("DOES THE EXACT BOUND CLOSE THE CERTIFICATE? AND CAN ANY BOUND OF THAT SHAPE?")
    P_(RULE)

    # ---- E0  THE CEILING GATE ------------------------------------------------------------------
    P_("\n" + RULE)
    P_("E0  THE CEILING GATE: PRICING THE SUBSTITUTION THAT EVERY BOUND OF THIS SHAPE MAKES")
    P_(RULE)
    nC, L = 4, 4
    Q, Pm, pi, n, hvec, S, actbit, rows, sha = setup(nC, L)
    Splus = np.maximum(S, 0.0).sum(axis=1)
    Bst = actbit @ S.T
    wts, wact = enumerate_all(Pm, pi, n, L, hvec, actbit)
    lo = logon(wact, S)
    contrib = wts * np.exp(lo)
    onprob = np.exp(lo)
    tail = float(contrib.sum())
    P_(f"  {nC} controllers, n = {n}, L = {L}, {n ** (L + 1):,} paths, {len(rows)} targets.")
    P_(f"  exact tail {tail:.6e}. Every quantity below is computed on the SAME prefix set at each")
    P_(f"  level, so the three ratios multiply to the total looseness.")
    P_(f"\n    {'level':>6} {'prefixes':>9} {'TANGENT/EXACT':>15} {'EXACT/BESTCOMP':>16}"
       f" {'BESTCOMP/TRUTH':>16} {'total':>12}")
    floors = []
    for d in range(1, L + 1):
        wp, ap = prefix_level(Pm, pi, n, L, hvec, actbit, d)
        H = float(hvec[d + 1:].sum())
        exact = bound_of(ap, wp, S, Splus, H)
        # the tangent relaxation, taken at the PARENT's drive exactly as the pruner takes it
        wpar, apar = prefix_level(Pm, pi, n, L, hvec, actbit, d - 1)
        u = -1.0 + 2.0 * (apar @ S.T) + 2.0 * H * Splus[None, :]
        ls = (-np.logaddexp(0.0, -u)).sum(axis=1)
        g = 1.0 / (1.0 + np.exp(u))
        lgt = ls[:, None] + g @ (2.0 * hvec[d] * Bst).T           # (P_parent, n)
        chmass = wp.reshape(len(wpar), n)
        tang = (chmass * np.exp(lgt)).ravel()
        blk = n ** (L - d)
        true = contrib.reshape(-1, blk).sum(axis=1)
        bestcomp = wp * onprob.reshape(-1, blk).max(axis=1)
        r1 = float(np.median(tang / np.maximum(exact, 1e-300)))
        r2 = float(np.median(exact / np.maximum(bestcomp, 1e-300)))
        r3 = float(np.median(bestcomp / np.maximum(true, 1e-300)))
        floors.append(r3)
        P_(f"    {d:>6} {len(wp):>9} {r1:>15.3e} {r2:>16.3e} {r3:>16.3e}"
           f" {r1 * r2 * r3:>12.3e}")
    fl = float(np.median(floors))
    P_(f"\n  medians across levels: the IRREDUCIBLE step -- best single completion over the")
    P_(f"  subtree's weighted average -- is {fl:.3e}.")
    P_(f"\n  E0 AS PREDECLARED: {'this is ABOVE the 1e3 bar, so NO bound of this shape closes the certificate.' if fl > 1e3 else 'this is BELOW the 1e3 bar, so the shape is capable of a useful certificate.'}")
    if fl > 1e3:
        P_("  The mass factor in these bounds is EXACT. All the looseness is in replacing a")
        P_("  weighted average of ON-probabilities by a maximum, and that substitution alone")
        P_("  costs more than the whole budget for a useful guarantee. Making the bound tighter")
        P_("  WITHIN this shape -- exact instead of tangent, coupled instead of decoupled, or any")
        P_("  refinement yet to be invented -- cannot get under this floor. A closing certificate")
        P_("  requires bounding the SUM over the subtree, not its largest member, which is a")
        P_("  different object and not a tighter version of this one.")

    # ---- E1  THE ENGINE-WIDTH RUN --------------------------------------------------------------
    P_("\n" + RULE)
    P_("E1  THE EXACT BOUND AT THE ENGINE'S WIDTH, RUN REGARDLESS OF E0")
    P_(RULE)
    P_("  E0's width is not the engine's, and the question was asked directly, so it is answered")
    P_("  directly. rank='exact' forms the true per-candidate bound, chunked over parents so the")
    P_("  parents x children x targets tensor never exists in full.")
    nCtrl, Le, dt = 10, 6, 0.5
    Q2, ctrl, cidx, sha2, ne, nw = trrust_block(nCtrl)
    rows2 = target_rows(cidx, 200)
    P_(f"\n  {nCtrl} controllers, {Q2.shape[0]} states, L = {Le}, {len(rows2)} target rows.")
    P_(f"\n    {'cap':>7} {'rank':>8} {'tail':>15} {'cert/tail':>13} {'seconds':>9}")
    res = {}
    for cp in (100, 1000, 10000, 19531):
        for rk in ("mass", "bound", "exact"):
            t0 = time.time()
            t, dr, _tou, kept, _r = engine_budget(Q2, nCtrl, rows2, Le, dt, 0.0, cap=cp, rank=rk)
            el = time.time() - t0
            res[(cp, rk)] = (t, dr)
            P_(f"    {cp:>7} {rk:>8} {t:>15.4e} {(dr / t if t > 0 else float('inf')):>13.3e}"
               f" {el:>9.1f}")
    P_("\n    (for rank='mass' the certificate is dropped MASS, which bounds the wrong thing at")
    P_("     all; it is listed so the three are in one table.)")
    ct = [res[(c, 'exact')][1] / res[(c, 'exact')][0] for c in (100, 1000, 10000, 19531)
          if res[(c, 'exact')][0] > 0]
    P_(f"\n  E1 AS PREDECLARED: the exact bound's certificate / tail is"
       f" {min(ct):.3e} to {max(ct):.3e}.")
    P_(f"  It {'CLOSES -- below 1.' if max(ct) < 1 else 'does NOT close: a certificate larger than the tail it bounds is not a guarantee.'}")
    for cp in (100, 1000, 10000, 19531):
        tb, db = res[(cp, "bound")]
        te, de = res[(cp, "exact")]
        if tb > 0 and te > 0:
            P_(f"    at cap {cp}: exact is {(db / tb) / (de / te):.2f}x tighter than the tangent"
               f" relaxation.")

    # ---- E2  THE RANKING -----------------------------------------------------------------------
    P_("\n" + RULE)
    P_("E2  DOES THE TIGHTER BOUND ALSO RANK BETTER?")
    P_(RULE)
    P_("  Nothing guarantees it does. A looser bound can order candidates differently and the")
    P_("  ordering is what determines the recovered tail.")
    P_(f"\n    {'cap':>7} {'tail, tangent':>15} {'tail, exact':>15} {'orders, exact - tangent':>25}")
    worse = []
    for cp in (100, 1000, 10000, 19531):
        tb = res[(cp, "bound")][0]
        te = res[(cp, "exact")][0]
        if tb > 0 and te > 0:
            dl = np.log10(te) - np.log10(tb)
            worse.append(dl)
            P_(f"    {cp:>7} {tb:>15.4e} {te:>15.4e} {dl:>25.3f}")
    if worse:
        P_(f"\n  E2: the exact bound ranks"
           f" {'BETTER everywhere.' if min(worse) > 0.001 else ('WORSE somewhere, which is the surprising case and is reported as predeclared.' if min(worse) < -0.001 else 'essentially IDENTICALLY -- the two orderings agree to within a thousandth of an order.')}")
        if abs(max(worse, key=abs)) < 0.01:
            P_("  That is itself informative: the tangent relaxation gives away orders in the")
            P_("  CERTIFICATE while barely changing the ORDER, which is why boundprune's 6x-8x")
            P_("  ranking win did not need the exact bound and the certificate could not be")
            P_("  rescued by it.")

    # ---- E3  LIMITS ----------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("E3  WHAT THIS DOES NOT SETTLE")
    P_(RULE)
    P_("  1. E0's floor is measured at 4 controllers and L = 4. The substitution it prices gets")
    P_("     WORSE with width and depth, not better -- more completions to take a maximum over --")
    P_("     so it is a floor at the engine's width too, but its VALUE there is not measured.")
    P_("  2. BESTCOMP is the tightest bound of the form mass x best-single-completion. A bound")
    P_("     that distributes the ON-probability over the subtree is outside the family and is")
    P_("     not bounded by E0's floor. That is where a closing certificate would have to come")
    P_("     from, and it is not built here.")
    P_("  3. Certificate / tail is the readable ratio at the engine's width because the true")
    P_("     deficit is unknowable there. It understates the guarantee's quality when the tail")
    P_("     itself is far below the truth, which it is.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_exactcert.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
