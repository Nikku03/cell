"""How do we know a path is irrelevant to the TARGET before deleting it?

THE QUESTION IS THE ENGINE'S BINDING DEFECT, NOW MEASURED. accuracy.py ranked the engine's error
terms in the unit it actually reports -- orders of magnitude of the rare-event tail -- and found
the pruner binding by 8.4 decades: the reported tail is a LOWER BOUND still rising at 0.608 orders
per decade of retained paths. The reason is exactly this question. The pruner ranks paths by MASS
and certifies the mass it dropped, but the reported quantity is not mass:

    tail = SUM over paths of  w_p * PROD over targets of sigma(Z_p[t])

A path with negligible mass can carry a large ON-probability, and a path with large mass can carry
none. Ranking by w alone therefore answers a different question from the one being asked, and
realkinetics measured the gap between the mass certificate and the tail at 1.9e29.

THE ANSWER THIS MODULE TESTS. You cannot know a path is irrelevant from its mass. You CAN know it
from an admissible upper bound on the best contribution any completion of its prefix could make,
and that bound exists in closed form. For a prefix through level d:

    the remaining activity is a sum of 0/1 vectors weighted by hvec[d+1..L], so it lies in the BOX
    [0, H_d]^nCtrl with H_d = sum of hvec[d+1..L];
    sigma is increasing and Z is linear in the activity, so target t is maximised by turning on
    exactly the controllers with positive weight, giving Z <= base + gain*(wact_prefix.S[t] +
    H_d * S+[t]) with S+[t] = sum_c max(S[t,c], 0);
    the subtree's total mass is exactly w_prefix, because the transition matrix is column
    stochastic and children's weights sum to the parent's.

    BOUND(prefix) = w_prefix * PROD_t sigma(base + gain*(wact_prefix.S[t] + H_d * S+[t]))

That bounds the WHOLE SUBTREE, not one completion, which is what a branch-and-bound needs. The
certificate becomes the summed dropped bound and it bounds the TAIL DEFICIT directly rather than
the mass. Whether it is tight enough to be worth anything is not arguable and is measured below.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

B0  THE CEILING GATE, AND IT COULD KILL THE WHOLE ROUTE. Before asking whether a better bound
    finds the important paths, ask whether there ARE few important paths. At a width small enough
    to enumerate EVERY path exactly, sort paths by their true contribution and measure how much
    of the tail the top fraction carries.
    PREDECLARED: if the top 1% of paths carry less than 50% of the tail, the tail is NOT
    concentrated, no pruner can be both cheap and accurate whatever it ranks by, and the bound
    question is moot -- the answer would be that the engine's observable is not amenable to
    pruning at all, which is a worse answer than the one this module hopes for and must be
    reported if it is what the data says. If the top 1% carry more than 90%, pruning is viable
    and the bound is worth building.

B1  ADMISSIBILITY, CHECKED EXHAUSTIVELY RATHER THAN ARGUED. The derivation above is a proof, and
    this build order has shipped false prose arithmetic before -- block.py's E4 and E7. So: at
    the enumerable width, for EVERY prefix at every level, compare BOUND(prefix) against the
    exact total contribution of its subtree.
    PREDECLARED: ONE violation anywhere and the bound is not admissible, the module stops, and
    nothing below it is reported. There is no tolerance band; admissibility is not statistical.

B2  TIGHTNESS, WHICH IS A DIFFERENT QUESTION FROM ADMISSIBILITY. The mass bound is admissible too
    and it is useless. Report the ratio of BOUND to true subtree contribution, per level.
    PREDECLARED: the bound is worth using only if its ratio is smaller than the mass bound's by
    many orders; a bound that is admissible and 1e29 loose is the bound the engine already has.

B3  THE HEAD-TO-HEAD, AT EQUAL COST. Retain the same NUMBER of paths under mass ranking and under
    bound ranking, and compare the fraction of the exact tail each recovers. Equal retained count
    is the fair comparison because that is what the engine pays for.
    PREDECLARED: the bound wins only if it recovers more tail at every retained count tested, not
    on average -- a rule that wins sometimes is a rule that has to be chosen, and the engine has
    no basis for choosing.

B4  THE CERTIFICATE, WHICH IS THE POINT. For each pruner, compare its certificate against the
    ACTUAL deficit it incurred. The mass pruner's certificate bounds mass and is known to miss.
    PREDECLARED: the new certificate is USEFUL if certificate / actual deficit stays below 1e3;
    it is worthless at 1e29, the figure realkinetics recorded for the mass certificate.

B5  THE SCALING LAW, because ranking at a point is the error corrected four times in this build
    order. Repeat across widths and window depths and report how the advantage MOVES.
    PREDECLARED: an advantage that shrinks with depth is not an answer for the engine, which
    exists to run deep windows.

B6  WHAT THIS DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.linalg import expm

from rem.atlas.hybrid_tune import RULE
from rem.atlas.realkinetics import trrust_block, target_rows, stationary
from rem.atlas.accuracy import identity_S


def setup(nCtrl, L, dt=0.5, ntarget=200):
    Q, ctrl, cidx, sha, ne, nw = trrust_block(nCtrl)
    rows = target_rows(cidx, ntarget)
    pi, res, _ = stationary(Q)
    n = Q.shape[0]
    Pm = expm((Q.T * dt).toarray())
    hvec = np.exp(-0.5 * np.arange(L, -1, -1))
    hvec = hvec / hvec.sum()
    S = identity_S(rows, nCtrl)
    actbit = np.array([[(m >> c) & 1 for c in range(nCtrl)] for m in range(n)], dtype=float)
    return Q, Pm, pi, n, hvec, S, actbit, rows, sha


def logon(wact, S, base=-1.0, gain=2.0):
    """log PROD_t sigma(Z[t]) for each row of wact. Computed in log space throughout."""
    Z = base + gain * (wact @ S.T)
    return -np.logaddexp(0.0, -Z).sum(axis=1)


def enumerate_all(Pm, pi, n, L, hvec, actbit):
    """Every path of L+1 levels, with its exact weight and time-weighted activity."""
    last = np.arange(n)
    wts = pi.copy()
    wact = actbit * hvec[0]
    for d in range(1, L + 1):
        col = Pm[:, last] * wts[None, :]              # (child, parent)
        wts = col.ravel(order="F")                    # parent-major, so child varies fastest
        par = np.repeat(np.arange(len(last)), n)
        chd = np.tile(np.arange(n), len(last))
        wact = wact[par] + actbit[chd] * hvec[d]
        last = chd
    return wts, wact


def bound_of(wact_prefix, w_prefix, S, Splus, H, base=-1.0, gain=2.0):
    """Admissible upper bound on the TOTAL contribution of a prefix's whole subtree."""
    Z = base + gain * (wact_prefix @ S.T + H * Splus[None, :])
    return w_prefix * np.exp(-np.logaddexp(0.0, -Z).sum(axis=1))


def prefix_level(Pm, pi, n, L, hvec, actbit, d):
    """Every prefix through level d, with its weight and partial activity."""
    last = np.arange(n)
    wts = pi.copy()
    wact = actbit * hvec[0]
    for dd in range(1, d + 1):
        col = Pm[:, last] * wts[None, :]
        wts = col.ravel(order="F")
        par = np.repeat(np.arange(len(last)), n)
        chd = np.tile(np.arange(n), len(last))
        wact = wact[par] + actbit[chd] * hvec[dd]
        last = chd
    return wts, wact


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("HOW DO WE KNOW A PATH IS IRRELEVANT TO THE TARGET BEFORE DELETING IT?")
    P_(RULE)
    P_("  The pruner ranks by MASS and the engine reports a TAIL. accuracy.py measured the")
    P_("  consequence: pruning binds by 8.4 decades. This tests the bound that would fix it.")

    nCtrl, L = 4, 4
    Q, Pm, pi, n, hvec, S, actbit, rows, sha = setup(nCtrl, L)
    Splus = np.maximum(S, 0.0).sum(axis=1)
    P_(f"\n  enumerable width: {nCtrl} controllers, n = {n} states, L = {L},"
       f" {n ** (L + 1):,} paths, {len(rows)} targets.")
    wts, wact = enumerate_all(Pm, pi, n, L, hvec, actbit)
    contrib = wts * np.exp(logon(wact, S))
    tail = float(contrib.sum())
    P_(f"  EXACT tail by full enumeration: {tail:.6e}")

    # ---- B0  THE CEILING GATE ------------------------------------------------------------------
    P_("\n" + RULE)
    P_("B0  THE CEILING GATE: IS THE TAIL CONCENTRATED ON FEW PATHS AT ALL?")
    P_(RULE)
    P_("  If it is not, no pruner can be cheap and accurate whatever it ranks by, and the bound")
    P_("  question is moot. PREDECLARED: top 1% carrying < 50% kills the route; > 90% clears it.")
    order = np.argsort(-contrib)
    cum = np.cumsum(contrib[order]) / tail
    P_(f"\n    {'top fraction of paths':>23} {'share of the tail':>19}")
    for f in (1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1):
        k = max(1, int(f * len(contrib)))
        P_(f"    {f * 100:>22.2f}% {cum[k - 1] * 100:>18.2f}%")
    k1 = max(1, int(0.01 * len(contrib)))
    top1 = float(cum[k1 - 1])
    P_(f"\n  B0: top 1% of paths carry {top1 * 100:.2f}% of the tail --"
       f" {'CLEARS the gate, pruning is viable' if top1 > 0.90 else ('KILLS the route' if top1 < 0.50 else 'INCONCLUSIVE band between the two predeclared thresholds')}.")
    if top1 < 0.50:
        P_("  As predeclared the module stops here: the observable is not amenable to pruning.")
        with open(os.path.join(os.path.dirname(__file__), "RESULTS_pathbound.txt"), "w") as fh:
            fh.write("\n".join(out) + "\n")
        return

    # ---- B1  ADMISSIBILITY ---------------------------------------------------------------------
    P_("\n" + RULE)
    P_("B1  ADMISSIBILITY, CHECKED EXHAUSTIVELY AT EVERY LEVEL")
    P_(RULE)
    P_("  ONE violation anywhere and the bound is not admissible and this module stops.")
    P_(f"\n    {'level':>6} {'prefixes':>10} {'violations':>12} {'worst ratio bound/true':>24}")
    ok = True
    per_level = {}
    for d in range(0, L):
        wp, ap = prefix_level(Pm, pi, n, L, hvec, actbit, d)
        H = float(hvec[d + 1:].sum())
        bd = bound_of(ap, wp, S, Splus, H)
        # exact subtree contribution: paths are enumerated parent-major, so a prefix at level d
        # owns a contiguous block of n^(L-d) leaves
        blk = n ** (L - d)
        true = contrib.reshape(-1, blk).sum(axis=1)
        assert true.shape == bd.shape, (true.shape, bd.shape)
        viol = int(np.sum(bd < true * (1 - 1e-9)))
        ratio = bd / np.maximum(true, 1e-300)
        per_level[d] = (float(np.median(ratio)), float(ratio.max()), wp, ap, bd, true, H)
        ok = ok and viol == 0
        P_(f"    {d:>6} {len(bd):>10} {viol:>12} {ratio.max():>24.3e}")
    P_(f"\n  B1: {'PASS -- admissible at every prefix at every level.' if ok else 'FAIL -- the bound is not admissible. STOPPING.'}")
    if not ok:
        with open(os.path.join(os.path.dirname(__file__), "RESULTS_pathbound.txt"), "w") as fh:
            fh.write("\n".join(out) + "\n")
        return

    # ---- B2  TIGHTNESS -------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("B2  TIGHTNESS AGAINST THE BOUND THE ENGINE ALREADY HAS")
    P_(RULE)
    P_("  The mass bound is admissible too, and useless. Its ratio is w_prefix / true.")
    P_(f"\n    {'level':>6} {'median ratio, NEW bound':>25} {'median ratio, MASS bound':>26} {'gain':>12}")
    for d in range(0, L):
        med, mx, wp, ap, bd, true, H = per_level[d]
        mass_ratio = float(np.median(wp / np.maximum(true, 1e-300)))
        P_(f"    {d:>6} {med:>25.3e} {mass_ratio:>26.3e} {mass_ratio / med:>12.3e}")

    # ---- B3  HEAD TO HEAD ----------------------------------------------------------------------
    P_("\n" + RULE)
    P_("B3  HEAD TO HEAD AT EQUAL RETAINED-PATH COUNT")
    P_(RULE)
    P_("  Same number of paths kept under each ranking; compare the fraction of the exact tail")
    P_("  recovered. PREDECLARED: the bound wins only if it wins at EVERY retained count.")
    d = L - 1
    _, _, wp, ap, bd, true, H = per_level[d]
    P_(f"\n  ranking prefixes at level {d} ({len(bd):,} of them, each owning {n} leaves)")
    P_(f"\n    {'kept':>10} {'tail via MASS':>16} {'tail via BOUND':>16} {'mass %':>9} {'bound %':>9}")
    om, ob = np.argsort(-wp), np.argsort(-bd)
    wins = []
    for kf in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5):
        k = max(1, int(kf * len(bd)))
        tm = float(true[om[:k]].sum())
        tb = float(true[ob[:k]].sum())
        wins.append(tb >= tm)
        P_(f"    {k:>10} {tm:>16.4e} {tb:>16.4e} {tm / tail * 100:>8.2f}% {tb / tail * 100:>8.2f}%")
    P_(f"\n  B3: {'the bound ranking wins at EVERY retained count.' if all(wins) else 'the bound ranking does NOT win everywhere -- as predeclared that is not a rule the engine can adopt.'}")

    # ---- B4  THE CERTIFICATE -------------------------------------------------------------------
    P_("\n" + RULE)
    P_("B4  THE CERTIFICATE AGAINST THE DEFICIT IT IS SUPPOSED TO BOUND")
    P_(RULE)
    P_(f"\n    {'kept':>10} {'MASS cert/deficit':>20} {'BOUND cert/deficit':>21}")
    ratios = []
    for kf in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5):
        k = max(1, int(kf * len(bd)))
        dm = tail - float(true[om[:k]].sum())
        db = tail - float(true[ob[:k]].sum())
        cm = float(wp[om[k:]].sum())
        cb = float(bd[ob[k:]].sum())
        rm = cm / max(dm, 1e-300)
        rb = cb / max(db, 1e-300)
        ratios.append(rb)
        P_(f"    {k:>10} {rm:>20.3e} {rb:>21.3e}")
    P_(f"\n  B4: worst BOUND certificate ratio {max(ratios):.3e} --"
       f" {'USEFUL, under the predeclared 1e3.' if max(ratios) < 1e3 else 'above the predeclared 1e3 bar; it is far tighter than mass but is NOT yet a tight certificate.'}")

    # ---- B5  THE SCALING LAW -------------------------------------------------------------------
    P_("\n" + RULE)
    P_("B5  DOES THE ADVANTAGE SURVIVE DEPTH? THE SCALING LAW")
    P_(RULE)
    P_("  An advantage that shrinks with depth is not an answer for an engine built to run deep")
    P_("  windows. PREDECLARED before running.")
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'paths':>12} {'top-1% share':>14} {'bound/mass tail at 1% kept':>28}")
    for nc, ll in ((4, 3), (4, 4), (4, 5), (5, 3), (5, 4)):
        if nc ** (ll + 1) > 3_000_000:
            P_(f"    {nc:>6} {ll:>3} {nc ** (ll + 1):>12,}  not enumerable within this module's budget")
            continue
        Q2, Pm2, pi2, n2, hv2, S2, ab2, rw2, _ = setup(nc, ll)
        Sp2 = np.maximum(S2, 0.0).sum(axis=1)
        w2, a2 = enumerate_all(Pm2, pi2, n2, ll, hv2, ab2)
        c2 = w2 * np.exp(logon(a2, S2))
        t2 = float(c2.sum())
        o2 = np.argsort(-c2)
        cm2 = np.cumsum(c2[o2]) / t2
        top = float(cm2[max(1, int(0.01 * len(c2))) - 1])
        dd = ll - 1
        wp2, ap2 = prefix_level(Pm2, pi2, n2, ll, hv2, ab2, dd)
        H2 = float(hv2[dd + 1:].sum())
        bd2 = bound_of(ap2, wp2, S2, Sp2, H2)
        blk2 = n2 ** (ll - dd)
        tr2 = c2.reshape(-1, blk2).sum(axis=1)
        k = max(1, int(0.01 * len(bd2)))
        tm = float(tr2[np.argsort(-wp2)[:k]].sum())
        tb = float(tr2[np.argsort(-bd2)[:k]].sum())
        P_(f"    {nc:>6} {ll:>3} {nc ** (ll + 1):>12,} {top * 100:>13.2f}%"
           f" {(tb / max(tm, 1e-300)):>28.3e}")

    # ---- B6  LIMITS ----------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("B6  WHAT THIS DOES NOT SETTLE")
    P_(RULE)
    P_("  1. Everything here is at widths small enough to enumerate EXACTLY, which is the only")
    P_("     way to check admissibility and to know the true tail. The engine runs at widths")
    P_("     where neither is available; that the bound stays admissible there follows from the")
    P_("     derivation, but that it stays USEFUL there does not, and B5 is the only evidence.")
    P_("  2. The bound relaxes the coupling between targets -- each target is allowed its own")
    P_("     best completion. A tighter bound exists and is not built here.")
    P_("  3. Admissibility is a property of the bound, not of the engine's implementation. This")
    P_("     module does not modify the pruner; wiring the bound into engine_budget is the next")
    P_("     step and is not taken here.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_pathbound.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
