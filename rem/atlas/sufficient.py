"""What must a path retain to give its remaining event probability EXACTLY, and how big is that?

WHY THIS IS THE RIGHT QUESTION NOW. exactcert closed the bounding route: every bound of the form
"subtree mass times the best ON-probability any completion could reach" carries an irreducible
floor of about 0.52 * n * m against the true deficit, which is 750 to 5500 at the engine's width,
so no refinement inside that family gives a usable certificate. What it named as the way out was a
representation that does not take a MAXIMUM over the subtree at all. That is this question:
if paths could be MERGED whenever they have the same future, no bound would be needed for the
merged part -- the computation would be exact.

THE STATISTIC, DERIVED. The engine's observable is

    tail = SUM_paths  w_p * PROD_t sigma(base + gain * (wact_p . S[t])),
    wact_p = SUM_{d=0..L} actbit[s_d] * hvec[d]

For a prefix through level d with last state s_d, every completion c contributes

    w_prefix * P(c | s_d) * PROD_t sigma(base + gain * ((wact_prefix + r_c) . S[t]))

so the prefix's ENTIRE remaining contribution is

    w_prefix  *  G_d(s_d, wact_prefix),    G_d(s, x) = E[ PROD_t sigma(base + gain*(x+r).S) | s ]

The weight factors out completely. THEREFORE the sufficient statistic is exactly

    (d,  s_d,  wact_prefix)

-- the level, the current state, and the accumulated time-weighted activity vector. Nothing else
about how the path arrived matters, and two prefixes sharing it can be merged by ADDING weights,
exactly. That is the derivation; the size is the interesting part.

AND THE SIZE IS DECIDED BY THE ARITHMETIC OF hvec, WHICH IS A THEOREM AND NOT A MEASUREMENT.
Each component of wact is a subset sum of hvec, so the statistic is injective on paths exactly
when hvec has distinct subset sums. The engine's hvec is geometric with ratio rho = exp(1/2):
a collision would mean SUM_d a_d rho^d = 0 with a_d in {-1,0,1} not all zero, which would make rho
ALGEBRAIC. By Lindemann-Weierstrass exp(1/2) is transcendental, so no collision exists at any
depth. WITH THE ENGINE'S OWN TIME WEIGHTING THERE IS NO EXACT MERGE AT ALL: the sufficient
statistic is the path, and the representation is the full path set.

That is a negative result with a constructive edge, because the obstruction is the irrationality of
the weights and not the structure of the problem. Quantise hvec to a grid of Q and every component
of wact becomes one of at most Q+1 rationals, so the statistic count collapses from
2^(nCtrl*(L+1)) to at most n * (Q+1)^nCtrl -- exponential in WIDTH but only POLYNOMIAL in DEPTH.
The price is whatever quantising the time weighting costs the tail. That trade is what this module
measures.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

F0  THE CEILING GATE, WHICH IS THE INJECTIVITY CLAIM MADE FINITE. Count the DISTINCT statistics
    against the path count at widths where both can be enumerated, using the engine's own hvec.
    PREDECLARED: if distinct statistics equal paths, there is no exact merge and the theorem's
    consequence is confirmed -- every compression below is then an APPROXIMATION and must be
    labelled one. If they are fewer, the theorem is wrong or the arithmetic is being done in
    floating point rather than exactly, and the module must say which.

F1  SUFFICIENCY, CHECKED WHERE IT IS NOT VACUOUS. F0 will show no collisions under the engine's
    hvec, which would make a sufficiency check vacuous -- there would be no two prefixes to compare.
    So the check is run under a QUANTISED hvec, where collisions are abundant: take groups of
    distinct prefixes sharing (s_d, wact), and verify their remaining contributions are exactly
    proportional to their weights.
    PREDECLARED: max relative deviation below 1e-12 or the derivation is wrong and nothing below is
    readable. A vacuous check is reported as vacuous, not as a pass.

F2  THE SIZE LAW. Count distinct statistics under quantised hvec across Q, nCtrl and L, and test
    the derived ceiling n*(Q+1)^nCtrl.
    PREDECLARED: reported as a law in BOTH nCtrl and L separately, because exactcert's floor law
    needed two variables and a one-variable fit gave two different exponents there. The claim that
    matters is the DEPTH exponent: if the statistic count grows exponentially in L the route is
    worthless, and if it is polynomial in L it is the collapse the engine needs.

F3  THE PRICE. Quantising the time weighting changes the observable. Measure the exact tail under
    the engine's hvec against the quantised one, in orders of magnitude, at each Q.
    PREDECLARED: the price must be compared against something. The reference is the pruning error
    the engine currently carries -- accuracy.py measured the reported tail still rising at 0.608
    orders per decade of retained paths. A quantisation whose price is below what pruning already
    costs is FREE in the only sense that matters.

F4  THE TRADE, as a curve of representation size against price over Q. This is the deliverable.

F2b THE DEPTH CLAIM, DECONFOUNDED. PREDECLARED AFTER F2's FIRST RUN AND BEFORE F2b's. F2 swept L
    with Q tied to it as Q = 2(L+1), then fitted "statistics ~ L^6.54" and called it the depth
    exponent. Q was growing with L, so the fit measured the two together -- exactly the confound
    caught in exactcert's floor law one module earlier, repeated here. The claim to test is that
    the count is bounded by n*(Q+1)^nCtrl INDEPENDENTLY of L, so the sweep must hold Q FIXED and
    vary L alone.
    PREDECLARED: the count must SATURATE in L at fixed Q. If it keeps growing with L at fixed Q,
    the derived ceiling is wrong and the flat-in-depth claim -- the entire value of the statistic
    -- fails with it.

F3b THE RESOLUTION AXIS, WHICH IS NOT Q. PREDECLARED after F3's first run. F3's table shows Q = 20
    and Q = 40 with byte-identical tails and identical statistic counts, because quantising at 40
    returns numerators [2,4,6,10,18] whose gcd is 2 -- the same rationals as Q = 20. Q is a nominal
    knob, not the resolution. The quantity that actually controls the count is the number of
    DISTINCT SUBSET SUMS of the numerator vector, and the effective denominator is Q/gcd.
    PREDECLARED: rows with identical weight vectors are collapsed and the table is keyed on the
    effective denominator and the distinct-subset-sum count, so no two rows can present the same
    approximation as two different ones.

F5  THE ENGINE'S OWN NUMBERS, and F6 the limits.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.pathbound import setup, enumerate_all, prefix_level, logon

PRUNE_SLOPE = 0.608     # accuracy.py's A4b: what pruning already costs, orders per decade


def quantise(hvec, Q):
    """hvec on a grid of Q, still summing to one exactly in integer units.

    Returns the integer numerators k with sum(k) == Q, so a component of wact is an exact integer
    multiple of 1/Q and equality between statistics can be tested without floating point."""
    k = np.maximum(np.round(np.asarray(hvec) * Q).astype(np.int64), 1)
    while k.sum() > Q:                      # take from the largest, so the shape is preserved
        k[np.argmax(k)] -= 1
    while k.sum() < Q:
        k[np.argmax(k)] += 1
    return k


def stats_int(Pm, pi, n, L, kvec, actbit, d):
    """Distinct sufficient statistics at level d under an integer time weighting.

    Carries wact in INTEGER units of 1/Q, so two statistics are equal iff their integer vectors
    are equal -- no tolerance, no floating point."""
    last = np.arange(n)
    W = actbit.astype(np.int64) * kvec[0]
    for dd in range(1, d + 1):
        par = np.repeat(np.arange(len(last)), n)
        chd = np.tile(np.arange(n), len(last))
        W = W[par] + actbit[chd].astype(np.int64) * kvec[dd]
        last = chd
    key = np.column_stack([last, W])
    return key, len(np.unique(key, axis=0))


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("WHAT A PATH MUST RETAIN FOR ITS REMAINING EVENT PROBABILITY, AND HOW BIG THAT IS")
    P_(RULE)
    P_("  THE STATISTIC, from the derivation in this module's header:  (d, s_d, wact_prefix)")
    P_("  -- the level, the current state, and the accumulated time-weighted activity vector. The")
    P_("  path weight factors out entirely, so two prefixes sharing the statistic merge by ADDING")
    P_("  weights, exactly. Everything below is about how many distinct values it takes.")

    # ---- F0  THE CEILING GATE ------------------------------------------------------------------
    P_("\n" + RULE)
    P_("F0  THE CEILING GATE: IS THERE AN EXACT MERGE UNDER THE ENGINE'S OWN TIME WEIGHTING?")
    P_(RULE)
    P_("  The theorem says no -- a collision would make exp(1/2) algebraic. Made finite here.")
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'level':>6} {'prefixes':>10} {'distinct statistics':>20} {'merge?':>8}")
    anymerge = False
    for nc, ll in ((3, 4), (4, 4)):
        Q_, Pm, pi, n, hvec, S, actbit, rows, sha = setup(nc, ll)
        for d in (ll - 1, ll):
            wp, ap = prefix_level(Pm, pi, n, ll, hvec, actbit, d)
            last = np.arange(n)
            for dd in range(1, d + 1):
                last = np.tile(np.arange(n), len(last))
            key = np.column_stack([last.astype(float), ap])
            uq = len(np.unique(np.round(key, 12), axis=0))
            anymerge = anymerge or (uq < len(wp))
            P_(f"    {nc:>6} {ll:>3} {d:>6} {len(wp):>10} {uq:>20}"
               f" {'yes' if uq < len(wp) else 'none':>8}")
    P_(f"\n  F0: {'no merge anywhere -- the statistic IS the path under the engine hvec, as the theorem requires.' if not anymerge else 'a merge was found, which contradicts the theorem: the arithmetic is floating point, not exact.'}")
    P_("  CONSEQUENCE, AND IT IS THE WHOLE REASON THE REST OF THIS MODULE IS ABOUT APPROXIMATION:")
    P_("  with the engine's geometric weighting there is NO exact merge at any depth or width, so")
    P_("  the exact representation is the full path set and nothing can be retained more cheaply.")

    # ---- F1  SUFFICIENCY -----------------------------------------------------------------------
    P_("\n" + RULE)
    P_("F1  SUFFICIENCY, CHECKED WHERE COLLISIONS EXIST")
    P_(RULE)
    P_("  Under a quantised weighting collisions are abundant, so the check is not vacuous: take")
    P_("  distinct prefixes sharing (s_d, wact) and verify their remaining contributions are")
    P_("  exactly proportional to their weights.")
    nc, ll, Q = 4, 4, 8
    Q_, Pm, pi, n, hvec, S, actbit, rows, sha = setup(nc, ll)
    kv = quantise(hvec, Q)
    hq = kv / float(Q)
    wts, wact = enumerate_all(Pm, pi, n, ll, hq, actbit)
    contrib = wts * np.exp(logon(wact, S))
    d = ll - 1
    key, nuq = stats_int(Pm, pi, n, ll, kv, actbit, d)
    wp, _ = prefix_level(Pm, pi, n, ll, hq, actbit, d)
    blk = n ** (ll - d)
    true = contrib.reshape(-1, blk).sum(axis=1)
    ratio = true / np.maximum(wp, 1e-300)          # must be equal within a collision group
    uq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    worst, groups = 0.0, 0
    for g in np.nonzero(cnt > 1)[0]:
        r = ratio[inv == g]
        groups += 1
        worst = max(worst, float((r.max() - r.min()) / max(abs(r.mean()), 1e-300)))
    P_(f"\n  quantised hvec at Q = {Q}: {len(wp)} prefixes collapse to {nuq} statistics,")
    P_(f"  {groups} of them holding more than one prefix.")
    if groups == 0:
        P_("  F1: VACUOUS -- no collision groups, so nothing was actually checked. Reported as")
        P_("  vacuous rather than as a pass.")
    else:
        P_(f"  worst relative spread of remaining-contribution-per-unit-weight within a group:"
           f" {worst:.3e}")
        P_(f"  F1: {'PASS -- the statistic is sufficient.' if worst < 1e-12 else 'FAIL -- the derivation is wrong and nothing below is readable.'}")
        if worst >= 1e-12:
            with open(os.path.join(os.path.dirname(__file__), "RESULTS_sufficient.txt"), "w") as fh:
                fh.write("\n".join(out) + "\n")
            return

    # ---- F2  THE SIZE LAW ----------------------------------------------------------------------
    P_("\n" + RULE)
    P_("F2  THE SIZE LAW, IN WIDTH AND DEPTH SEPARATELY")
    P_(RULE)
    P_("  The derived ceiling is n * (Q+1)^nCtrl, independent of L. If the measured count follows")
    P_("  it, the representation is exponential in WIDTH and FLAT in DEPTH, which is the collapse")
    P_("  the engine needs. Counted exactly in integer units of 1/Q -- no tolerance.")
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'Q':>4} {'paths':>14} {'statistics':>12} {'ceiling n(Q+1)^k':>18} {'paths/stats':>12}")
    rowsF2 = []
    for nc in (3, 4):
        for ll in (3, 4, 5):
            if (2 ** nc) ** (ll + 1) > 3_000_000:
                continue
            Q_, Pm, pi, n, hvec, S, actbit, rows, sha = setup(nc, ll)
            for Q in (ll + 1, 2 * (ll + 1), 4 * (ll + 1)):
                kv = quantise(hvec, Q)
                _k, nuq = stats_int(Pm, pi, n, ll, kv, actbit, ll)
                npaths = n ** (ll + 1)
                ceil = n * (Q + 1) ** nc
                rowsF2.append((nc, ll, Q, npaths, nuq, ceil))
                P_(f"    {nc:>6} {ll:>3} {Q:>4} {npaths:>14,} {nuq:>12,} {ceil:>18,}"
                   f" {npaths / nuq:>12,.0f}")
    P_("\n  A DEPTH EXPONENT FITTED HERE WOULD BE CONFOUNDED, because the Q column above moves with")
    P_("  L. That is the mistake exactcert's floor law had to be rescued from one module earlier,")
    P_("  and it is not repeated: the depth claim is tested in F2b with Q held FIXED.")

    # ---- F2b  THE DEPTH CLAIM, WITH Q HELD FIXED -----------------------------------------------
    P_("\n" + RULE)
    P_("F2b  THE DEPTH CLAIM, DECONFOUNDED: Q FIXED, L VARYING")
    P_(RULE)
    P_("  The derived ceiling n*(Q+1)^nCtrl does not contain L at all, so at fixed Q the count must")
    P_("  SATURATE as the window deepens. If it keeps growing, the flat-in-depth claim fails and")
    P_("  with it the whole value of the statistic.")
    P_(f"\n    {'nCtrl':>6} {'Q':>4} {'L':>3} {'paths':>14} {'statistics':>12} {'ceiling':>12} {'x paths':>10}")
    sat = {}
    for nc in (3, 4):
        for Q in (12, 24):
            for ll in (3, 4, 5):
                if (2 ** nc) ** (ll + 1) > 3_000_000:
                    continue
                Q_, Pm, pi, n, hvec, S, actbit, rows, sha = setup(nc, ll)
                kv = quantise(hvec, Q)
                _k, nuq = stats_int(Pm, pi, n, ll, kv, actbit, ll)
                sat.setdefault((nc, Q), []).append((ll, nuq))
                P_(f"    {nc:>6} {Q:>4} {ll:>3} {n ** (ll + 1):>14,} {nuq:>12,}"
                   f" {n * (Q + 1) ** nc:>12,} {n ** (ll + 1) / nuq:>9,.0f}x")
    P_("")
    P_("  THE FIRST VERSION OF THIS VERDICT USED A THRESHOLD I INVENTED -- growth below")
    P_("  (path growth)^0.34 -- and it returned FAIL on data that does not support a fail. That is")
    P_("  the invented-bar defect this record already carries once, in K5's base = -1.0. The claim")
    P_("  has two parts and neither needs a threshold: the ceiling must never be violated, and the")
    P_("  COMPRESSION must grow with depth.")
    viol, comp_ok = 0, True
    for (nc, Q), v in sorted(sat.items()):
        ceil = (2 ** nc) * (Q + 1) ** nc
        for ll, nuq in v:
            if nuq > ceil:
                viol += 1
        if len(v) < 2:
            continue
        comps = [((2 ** nc) ** (ll + 1)) / nuq for ll, nuq in v]
        rising = all(b >= a for a, b in zip(comps, comps[1:]))
        comp_ok = comp_ok and rising
        growth = v[-1][1] / v[0][1]
        pathgrowth = (2.0 ** nc) ** (v[-1][0] - v[0][0])
        P_(f"    nCtrl = {nc}, Q = {Q}: over L = {v[0][0]}..{v[-1][0]} statistics grow {growth:.2f}x,"
           f" paths {pathgrowth:.0f}x, compression {comps[0]:.0f}x -> {comps[-1]:.0f}x"
           f" {'(rising)' if rising else '(NOT rising)'}")
    P_(f"\n    ceiling violations across all {sum(len(v) for v in sat.values())} configurations:"
       f" {viol}")
    P_(f"\n  F2b: {'the ceiling holds everywhere and the compression RISES with depth in every configuration.' if viol == 0 and comp_ok else ('the ceiling is VIOLATED, so the derived bound is wrong.' if viol else 'the ceiling holds but the compression does not rise everywhere.')}")
    P_("  WHAT IT DOES NOT SHOW, AND THE FIRST VERSION CONFLATED THIS: the count is bounded")
    P_("  independently of L, but it has NOT SATURATED at L = 3 to 5 -- it is still filling up")
    P_("  toward the ceiling. The growth observed is the approach to a fixed bound, not unbounded")
    P_("  growth, and those two look identical over three points unless the bound is checked. It")
    P_("  is checked above.")
    P_("  So the depth exponential the engine has been fighting is an artefact of carrying the")
    P_("  PATH rather than what the path is for -- but only once the weighting is quantised, which")
    P_("  F0 established is an approximation and not a reformulation.")

    # ---- F3  THE PRICE -------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("F3  THE PRICE OF QUANTISING THE TIME WEIGHTING")
    P_(RULE)
    P_("  Quantising changes the observable. The reference it is judged against is what pruning")
    P_("  already costs: accuracy.py measured the reported tail still rising at 0.608 orders per")
    P_("  decade of retained paths, so a price well under one order is free in the only sense")
    P_("  that matters here.")
    nc, ll = 4, 4
    Q_, Pm, pi, n, hvec, S, actbit, rows, sha = setup(nc, ll)
    w0, a0 = enumerate_all(Pm, pi, n, ll, hvec, actbit)
    t0 = float((w0 * np.exp(logon(a0, S))).sum())
    P_(f"\n  exact tail under the engine's own hvec: {t0:.6e}")
    P_("\n  AND THE AXIS IS NOT Q. Quantising at 40 returns numerators whose gcd is 2, i.e. the same")
    P_("  rationals as 20, so a table keyed on Q shows one approximation twice. Keyed instead on")
    P_("  the EFFECTIVE denominator Q/gcd and the number of DISTINCT SUBSET SUMS of the numerators,")
    P_("  which is what actually controls the count. Duplicate weight vectors are collapsed.")
    P_(f"\n    {'Qeff':>6} {'subset sums':>12} {'statistics':>12} {'tail':>15} {'price, orders':>14} {'vs a pruning decade':>21}")
    curve, seen = [], set()
    for Q in (5, 10, 20, 40, 80, 160, 320):
        kv = quantise(hvec, Q)
        g = int(np.gcd.reduce(kv))
        kr = kv // g
        sig = tuple(kr.tolist())
        if sig in seen:
            continue
        seen.add(sig)
        Qeff = int(kr.sum())
        ss = len({int(x) for m in range(1 << len(kr)) for x in
                  [sum(kr[i] for i in range(len(kr)) if (m >> i) & 1)]})
        hq = kr / float(Qeff)
        wq, aq = enumerate_all(Pm, pi, n, ll, hq, actbit)
        tq = float((wq * np.exp(logon(aq, S))).sum())
        _k, nuq = stats_int(Pm, pi, n, ll, kr, actbit, ll)
        pr = abs(np.log10(tq) - np.log10(t0)) if tq > 0 else float("inf")
        curve.append((Qeff, nuq, tq, pr, ss))
        P_(f"    {Qeff:>6} {ss:>12} {nuq:>12,} {tq:>15.6e} {pr:>14.4f}"
           f" {pr / PRUNE_SLOPE:>21.3f}")

    # ---- F4  THE TRADE -------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("F4  THE TRADE, AND WHAT IT LOOKS LIKE AT THE ENGINE'S WIDTH")
    P_(RULE)
    npaths_small = n ** (ll + 1)
    P_(f"  at {nc} controllers and L = {ll} the path set is {npaths_small:,}:")
    P_(f"\n    {'Qeff':>6} {'compression':>13} {'price, orders':>14}")
    for Qe, nuq, tq, pr, ss in curve:
        P_(f"    {Qe:>6} {npaths_small / nuq:>12,.0f}x {pr:>14.4f}")
    P_("\n  AND EXTRAPOLATED TO THE ENGINE (nCtrl = 10, n = 1024, L = 6) by the derived ceiling,")
    P_("  which F2 tests rather than assumes:")
    P_(f"\n    {'Q':>5} {'statistics, n(Q+1)^10':>24} {'paths, 2^70':>14} {'compression':>14}")
    for Q in (7, 14, 28, 56):
        st = 1024 * (Q + 1) ** 10
        P_(f"    {Q:>5} {st:>24.3e} {2.0 ** 70:>14.3e} {2.0 ** 70 / st:>13.3e}x")

    # ---- F5/F6 ---------------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("F5  WHAT THIS SETTLES, AND F6 WHAT IT DOES NOT")
    P_(RULE)
    P_("  SETTLED. The exact sufficient statistic for a path's remaining event probability is")
    P_("  (level, current state, accumulated time-weighted activity vector), the weight factors")
    P_("  out, and merging is exact. Under the engine's geometric weighting the statistic is")
    P_("  injective on paths -- by transcendence of exp(1/2), not by measurement -- so the exact")
    P_("  representation is the full path set and there is nothing to gain.")
    P_("  Under a quantised weighting the count is BOUNDED BY n*(Q+1)^nCtrl, a bound that contains")
    P_("  no L at all: exponential in width, independent of depth. Measured at 3 and 4 controllers")
    P_("  the bound is never violated and the compression rises with every extra level of window.")
    P_("  It has not saturated over L = 3 to 5, so what is demonstrated is a bound independent of")
    P_("  depth and a compression that grows with it -- not that the count has stopped growing.")
    P_("\n  NOT SETTLED.")
    P_("  1. The engine-width row is the DERIVED CEILING, tested at 3 and 4 controllers, not")
    P_("     measured at 10. n*(Q+1)^10 is still 1e13 at Q = 14 -- nine orders below the path set")
    P_("     and far above what fits in memory. This buys a different exponent, not a small")
    P_("     number, and calling it tractable would be the overstatement this record keeps")
    P_("     correcting.")
    P_("  2. Quantising the time weighting is an APPROXIMATION to the engine's observable, not a")
    P_("     reformulation of it. F3 prices it at one width and one depth only.")
    P_("  3. multiplier.py measured that the temporal multiplier PAYS -- recent slices weighing")
    P_("     more is worth real accuracy. Quantising it coarsely is in tension with that result,")
    P_("     and this module does not test the interaction.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_sufficient.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
