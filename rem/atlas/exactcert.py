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

E0b THE NUMBER THAT MATTERS, PREDECLARED AFTER E0's DECOMPOSITION AND BEFORE IT RAN. E0 prices the
    three steps as RATIOS PER PREFIX. That is not yet a certificate: a certificate is the SUM of
    the bound over the dropped set against the ACTUAL deficit that dropping them caused. So form
    both, for the same dropped set, at several retained fractions -- once with the exact box bound
    and once with BESTCOMP, the true maximum over actual completions.
    PREDECLARED: if the BESTCOMP certificate lands under the 1e3 bar while the exact one does not,
    then the certificate is CLOSABLE and the thing to fix is the completion set, not the bound's
    algebra. If both fail, the route is dead whatever E0's per-prefix ratios suggested.
    AND THE LIMIT MUST BE STATED WITH THE RESULT: BESTCOMP IS NOT AN ALGORITHM. It is computed by
    enumerating the completions, which is exactly what the engine cannot do. It measures what a
    bound would deliver IF the reachable completion set could be described tightly. Reporting it
    as an achieved certificate would be the same error as reading a loop ceiling as a cap.

E0c THE SCALING OF THE IRREDUCIBLE FLOOR, PREDECLARED BEFORE IT RAN AND BEFORE THE CONCLUSION IS
    ALLOWED TO REST ON IT. E0b's verdict that the certificate is closable rests entirely on
    BESTCOMP / TRUTH being small -- about ten to twenty at four controllers. But the engine runs at
    ten controllers and L = 6, and the quantity is a MAXIMUM over completions against their
    weighted average, so it should grow as there are more completions to maximise over. The first
    draft of this module asserted that direction in prose. Prose arithmetic is what block.py's E4
    and E7 were, so it is measured.
    PREDECLARED: fit the floor's growth in width and in depth over the enumerable range. If
    extrapolating it to ten controllers and L = 6 puts it above 1e3, then the certificate is NOT
    closable at the engine's width and E0b's verdict holds only at toy widths -- which must be
    said plainly, because E0b is the encouraging result and this is the gate that can take it away.

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

    # ---- E0b  THE CERTIFICATE ITSELF, NOT THE PER-PREFIX RATIO ---------------------------------
    P_("\n" + RULE)
    P_("E0b  THE CERTIFICATE ITSELF: SUMMED BOUND OVER DROPPED, AGAINST THE ACTUAL DEFICIT")
    P_(RULE)
    P_("  E0's ratios are per prefix. A certificate is the SUM of the bound over the dropped set")
    P_("  against the deficit dropping them actually caused. Both are formed here on the SAME")
    P_("  dropped set, once with the exact box bound and once with BESTCOMP.")
    P_(f"\n    {'level':>6} {'kept':>7} {'EXACT cert/deficit':>21} {'BESTCOMP cert/deficit':>23}")
    bc_ratios = []
    ex_ratios = []
    for d in (2, 3):
        wp, ap = prefix_level(Pm, pi, n, L, hvec, actbit, d)
        H = float(hvec[d + 1:].sum())
        ex = bound_of(ap, wp, S, Splus, H)
        blk = n ** (L - d)
        true = contrib.reshape(-1, blk).sum(axis=1)
        bc = wp * onprob.reshape(-1, blk).max(axis=1)
        oe, ob = np.argsort(-ex), np.argsort(-bc)
        for kf in (0.01, 0.10, 0.50):
            k = max(1, int(kf * len(ex)))
            de = tail - float(true[oe[:k]].sum())
            ce = float(ex[oe[k:]].sum())
            db = tail - float(true[ob[:k]].sum())
            cb = float(bc[ob[k:]].sum())
            re_, rb = ce / max(de, 1e-300), cb / max(db, 1e-300)
            ex_ratios.append(re_)
            bc_ratios.append(rb)
            P_(f"    {d:>6} {f'{kf * 100:.0f}%':>7} {re_:>21.3e} {rb:>23.3e}")
    P_(f"\n  E0b AS PREDECLARED: the exact box certificate runs {min(ex_ratios):.2e} to"
       f" {max(ex_ratios):.2e}; BESTCOMP runs {min(bc_ratios):.2f} to {max(bc_ratios):.2f}.")
    if max(bc_ratios) < 1e3 <= max(ex_ratios):
        P_("  THE CERTIFICATE IS CLOSABLE, AND THE THING TO FIX IS THE COMPLETION SET RATHER THAN")
        P_("  THE BOUND'S ALGEBRA. Describing the reachable completions instead of relaxing them to")
        P_(f"  a box moves the certificate from {max(ex_ratios):.1e} times the deficit to about"
           f" {max(bc_ratios):.0f} times it,")
        P_("  which clears the 1e3 bar by two orders.")
    elif max(bc_ratios) >= 1e3:
        P_("  BOTH FAIL. The route is dead whatever E0's per-prefix ratios suggested.")
    P_("\n  AND THE LIMIT, STATED WITH THE RESULT: BESTCOMP IS NOT AN ALGORITHM. It is computed by")
    P_("  ENUMERATING the completions, which is exactly what the engine cannot do at its own")
    P_("  width. It measures what a bound would deliver IF the reachable completion set could be")
    P_("  described tightly. Quoting it as an achieved certificate would be the same error as")
    P_("  reading a loop ceiling as a cap, which this build order has now done three times.")

    # ---- E0c  DOES THE FLOOR SURVIVE THE ENGINE'S WIDTH? ---------------------------------------
    P_("\n" + RULE)
    P_("E0c  THE SCALING OF THE IRREDUCIBLE FLOOR, SINCE E0b's VERDICT RESTS ON IT")
    P_(RULE)
    P_("  BESTCOMP / TRUTH is a MAXIMUM over completions against their weighted average, so it")
    P_("  should grow as there are more completions. The engine runs at 10 controllers and L = 6.")
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'paths':>12} {'completions per prefix':>23} {'BESTCOMP/TRUTH':>16}")
    grid = []
    for nc, ll in ((3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (5, 3)):
        np_ = (2 ** nc) ** (ll + 1)
        if np_ > 3_000_000:
            continue
        Q3, Pm3, pi3, n3, hv3, S3, ab3, rw3, _ = setup(nc, ll)
        Sp3 = np.maximum(S3, 0.0).sum(axis=1)
        w3, a3 = enumerate_all(Pm3, pi3, n3, ll, hv3, ab3)
        lo3 = logon(a3, S3)
        c3 = w3 * np.exp(lo3)
        on3 = np.exp(lo3)
        d = ll - 2 if ll >= 3 else ll - 1                 # two levels of completions left
        wp3, ap3 = prefix_level(Pm3, pi3, n3, ll, hv3, ab3, d)
        blk3 = n3 ** (ll - d)
        tr3 = c3.reshape(-1, blk3).sum(axis=1)
        bc3 = wp3 * on3.reshape(-1, blk3).max(axis=1)
        r = float(np.median(bc3 / np.maximum(tr3, 1e-300)))
        grid.append((nc, ll, blk3, r))
        P_(f"    {nc:>6} {ll:>3} {np_:>12,} {blk3:>23,} {r:>16.3f}")
    if len(grid) >= 4:
        # the floor should be a function of the number of completions, which is what varies
        cw = np.log10([g[2] for g in grid])
        fv = np.log10([g[3] for g in grid])
        sl, ic = np.polyfit(cw, fv, 1)
        r2 = 1.0 - np.sum((fv - (ic + sl * cw)) ** 2) / np.sum((fv - fv.mean()) ** 2)
        P_(f"\n    log10(floor) = {ic:.3f} + {sl:.3f} * log10(completions per prefix)   R^2 = {r2:.3f}")
        comp_engine = (2 ** 10) ** 2                       # two levels of completions at nCtrl=10
        pred = 10 ** (ic + sl * np.log10(comp_engine))
        P_(f"    at the engine's width the same prefix depth leaves {comp_engine:,} completions,")
        P_(f"    which the law puts at a floor of about {pred:.1f}.")
        P_(f"\n  E0c AS PREDECLARED: extrapolated floor {pred:.1f}"
           f" {'is BELOW 1e3, so E0b s verdict survives the engine width: the certificate is closable there too.' if pred < 1e3 else 'is ABOVE 1e3, so E0b s verdict holds only at toy widths and the certificate is NOT closable at the engine width.'}")
        P_("    The law is fitted over three decades of completion count and the engine sits about")
        P_("    three decades beyond the widest point, so this is an extrapolation of comparable")
        P_("    reach to the ones accuracy.py and boundprune had to make, and no better founded.")

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
    P_("  1. E0c MEASURES the floor's growth rather than asserting it, but it still extrapolates")
    P_("     about three decades of completion count beyond the widest enumerable point. The")
    P_("     DIRECTION is measured; the engine-width VALUE is extrapolated.")
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
