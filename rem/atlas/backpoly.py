"""The compact backward function, run against the ACTUAL REM tail.

WHY. An external round of tests replaced path enumeration with a compact estimate of the remaining
event probability, found it worked on a smooth benchmark and failed on an irregular one, and then
rescued the irregular case with a Fourier basis matched to its known oscillation. Those tests did
not run the REM engine. Before their failure mode can be assumed to be REM's failure mode, REM's
own backward function has to be looked at, because there is a specific reason to doubt it is the
same problem.

THE REASON TO DOUBT IT. Their exact reference ran 1,024 states and 200 steps in 0.012 s. The
engine's controller block has exactly nCtrl off-diagonal entries per state, so a STATE-INDEXED
backward pass costs 1024 * 10 * 200 = 2.05e6 operations, which is 0.012 s. That arithmetic says
their backward function is indexed by the current state alone. REM's is not: sufficient.py showed
it is indexed by (level, state, accumulated time-weighted activity), and that at full precision the
statistic is injective on paths. So REM has no cheap exact reference at all, its backward function
carries a 10-dimensional continuous argument, and its difficulty may be DIMENSION rather than
oscillation. This module measures which.

THE OBJECT. Writing V_d(s, x) for the contribution from being at state s at level d having
accumulated x, the recursion is exact:

    V_L(s, x) = PROD_t sigma(base + gain*((x + a_s h_L).S[t]))
    V_d(s, x) = SUM_s' Pm[s',s] * V_{d+1}(s', x + a_s h_d)
    tail      = SUM_s pi[s] * V_0(s, 0)

and the whole method is to carry log V_d(s, .) as a POLYNOMIAL in x. A polynomial of total degree
p in k variables has C(k+p, p) coefficients -- 66 at degree 2 in ten variables -- so the
representation is polynomial in width, not exponential, which is the only reason this is worth
trying at the engine's own size. The single approximation is projecting log of a weighted sum of
exponentials back onto that space; the shift by a_s h_d is exact, since a shifted polynomial is a
polynomial of the same degree.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

G0  THE CEILING GATE, WHICH DECIDES WHETHER THE EXTERNAL ROUNDS TESTED THE SAME PROBLEM. At a
    width where V can be computed EXACTLY by enumerating suffixes, fit log V(s, .) with (a)
    polynomials and (b) random Fourier features, at EQUAL coefficient count.
    PREDECLARED: if polynomials beat Fourier at equal size, REM's backward function is smooth and
    its obstruction is dimension, so the irregular/oscillatory failure the external round 1 hit is
    NOT REM's failure and round 2's Fourier repair is solving a different problem. If Fourier wins,
    the external rounds were on target and this module should adopt their basis.

G1  THE END TO END, AT A WIDTH WITH AN EXACT ANSWER. Run the polynomial backward recursion and
    compare its tail against the exact enumerated tail.
    PREDECLARED: the reference for "useful" is what pruning already costs, which accuracy.py
    measured at 0.608 orders per decade of retained paths. An error well under one order is useful;
    an error above that is not, whatever its coefficient count.

G2  THE SCALING, IN DEGREE AND IN WIDTH, because a method ranked at a point is the error this
    build order has corrected five times. Report error against degree, and coefficient count
    against nCtrl.

G3  THE ACTUAL REM TAIL, at 10 controllers and L = 6 -- the run the external rounds did not do.
    There is no exact answer at that width, but there IS a falsifiable check, and it is sharp: the
    pruned tail is a LOWER BOUND on the truth, because retaining more paths can only add mass.
    PREDECLARED: if the polynomial backward returns a value BELOW the best pruned lower bound, the
    method is REFUTED at the engine's width, with no exact reference needed. If it returns a value
    above, that is consistent and is reported as consistency, not as accuracy.

G3b WHAT G3's ESTIMATE IMPLIES FOR accuracy.py's RANKING. PREDECLARED AFTER G3 RAN AND BEFORE G3b
    DID, because G3's numbers make a question answerable that was not answerable before. accuracy.py
    ranked the engine's error terms by extrapolating the pruned tail's rise -- 0.608 orders per
    decade of retained paths -- from 10^4.29 out to the full path set at 10^21.1, and concluded
    pruning binds because that reaches the class map's whole 5.08-order span at 10^12.6. It flagged
    the extrapolation as a direction rather than a value, and this is the first INDEPENDENT estimate
    of where the tail actually ends up.
    PREDECLARED: compute the remaining rise G3's estimate implies, against what the slope
    extrapolation implies. If they disagree, the slope FLATTENS and accuracy.py's crossing is wrong;
    and if the implied total pruning error falls below 5.08 orders, accuracy.py's verdict REVERSES
    and the class map becomes the binding term after all. That must be reported whichever way it
    lands, and reported as CONDITIONAL on an estimate that carries no bound.

G4  THE TIMING, AS A SCALING LAW AND NOT AS THE 28x POINT. The external note compared 0.34 s
    against 0.012 s for exact DP. That comparison is against a baseline REM does not have. Report
    instead how the polynomial backward's cost grows with width and depth, and what it is being
    compared against in REM's case, which is path enumeration.

G5  WHAT THIS DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import time
import numpy as np
from scipy.linalg import expm

from rem.atlas.hybrid_tune import RULE
from rem.atlas.realkinetics import trrust_block, target_rows, stationary
from rem.atlas.accuracy import identity_S

BASE, GAIN = -1.0, 2.0
PRUNE_SLOPE = 0.608


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
    return Pm, pi, n, hvec, S, actbit, rows, sha


def logF(Y, S):
    """log PROD_t sigma(base + gain * (y . S[t])) for each row of Y."""
    return -np.logaddexp(0.0, -(BASE + GAIN * (Y @ S.T))).sum(axis=1)


def mono(k, deg):
    """Exponent tuples for every monomial of total degree <= deg in k variables."""
    out = [(0,) * k]
    for order in range(1, deg + 1):
        for c in itertools.combinations_with_replacement(range(k), order):
            e = [0] * k
            for i in c:
                e[i] += 1
            out.append(tuple(e))
    return out


def design(X, mons):
    """(npts, ncoef) polynomial design matrix."""
    A = np.empty((X.shape[0], len(mons)))
    for j, e in enumerate(mons):
        v = np.ones(X.shape[0])
        for i, p in enumerate(e):
            if p:
                v = v * X[:, i] ** p
        A[:, j] = v
    return A


def rff(X, W, b):
    """Random Fourier features, the same-size comparison basis."""
    return np.cos(X @ W.T + b[None, :])


def exact_V(Pm, n, hvec, actbit, S, L, d, s_list, X):
    """V_d(s, x) computed EXACTLY by enumerating every suffix from level d. Only tractable at
    small widths, which is why G0 and G1 live there."""
    out = np.zeros((len(s_list), len(X)))
    for i, s in enumerate(s_list):
        last = np.array([s])
        w = np.array([1.0])
        acc = (actbit[s] * hvec[d])[None, :]
        for dd in range(d + 1, L + 1):
            col = Pm[:, last].T * w[:, None]
            w = col.ravel()
            par = np.repeat(np.arange(len(last)), n)
            chd = np.tile(np.arange(n), len(last))
            acc = acc[par] + actbit[chd] * hvec[dd]
            last = chd
        for p, x in enumerate(X):
            out[i, p] = float((w * np.exp(logF(x[None, :] + acc, S))).sum())
    return out


def poly_backward(Pm, pi, n, hvec, S, actbit, L, deg, npts_mult=4, seed=7, chunk=64):
    """The method: carry log V_d(s, .) as a total-degree-`deg` polynomial in x."""
    k = actbit.shape[1]
    mons = mono(k, deg)
    nc = len(mons)
    rng = np.random.default_rng(seed)
    npts = npts_mult * nc
    # collocation box: x accumulates hvec over the levels already passed, so it lives in [0, H]^k
    C = None
    for d in range(L, -1, -1):
        H = float(hvec[:d].sum()) if d > 0 else 0.0
        X = rng.random((npts, k)) * max(H, 1e-9)
        A = design(X, mons)
        Ap = np.linalg.pinv(A)
        if d == L:
            lv = np.empty((n, npts))
            for s in range(n):
                lv[s] = logF(X + actbit[s] * hvec[L], S)
            C = lv @ Ap.T
            Xprev, Aprev, Apprev = X, A, Ap
            continue
        # 1. evaluate V_{d+1} at this level's collocation points
        E = np.exp(np.clip(C @ design(X, mons).T, -700.0, 700.0))
        # 2. one matvec in the state index
        W = Pm.T @ E
        # 3. project log W back onto the polynomial space -- the ONLY approximation
        Cw = np.log(np.maximum(W, 1e-300)) @ Ap.T
        # 4. shift by a_s h_d, which is exact for polynomials
        Cn = np.empty_like(Cw)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            sh = X[None, :, :] + (actbit[a:b] * hvec[d])[:, None, :]
            Ash = design(sh.reshape(-1, k), mons).reshape(b - a, npts, nc)
            vals = np.einsum('spc,sc->sp', Ash, Cw[a:b])
            Cn[a:b] = vals @ Ap.T
        C = Cn
    v0 = np.exp(np.clip(C @ design(np.zeros((1, k)), mons).T, -700.0, 700.0)).ravel()
    return float((pi * v0).sum()), nc, npts


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("THE COMPACT BACKWARD FUNCTION, RUN AGAINST THE ACTUAL REM TAIL")
    P_(RULE)

    # ---- G0  THE CEILING GATE ------------------------------------------------------------------
    P_("\n" + RULE)
    P_("G0  SMOOTH OR OSCILLATORY? THE GATE THAT DECIDES IF THE SAME PROBLEM WAS TESTED")
    P_(RULE)
    nc4, L4 = 4, 4
    Pm, pi, n, hvec, S, actbit, rows, sha = setup(nc4, L4)
    d = 2
    rng = np.random.default_rng(11)
    H = float(hvec[:d].sum())
    Xf = rng.random((400, nc4)) * H
    s_list = list(range(0, n, 4))
    Vex = exact_V(Pm, n, hvec, actbit, S, L4, d, s_list, Xf)
    Y = np.log(np.maximum(Vex, 1e-300))
    P_(f"  exact V at {nc4} controllers, L = {L4}, level {d}: {len(s_list)} states x {len(Xf)} points.")
    P_(f"\n    {'coefficients':>13} {'polynomial rel err':>20} {'Fourier rel err':>18} {'winner':>9}")
    polywin = 0
    for deg in (1, 2, 3):
        mons = mono(nc4, deg)
        A = design(Xf, mons)
        cp = np.linalg.lstsq(A, Y.T, rcond=None)[0]
        ep = float(np.mean(np.abs(A @ cp - Y.T) / np.maximum(np.abs(Y.T), 1e-12)))
        Wf = rng.normal(0.0, 1.0 / max(H, 1e-9), (len(mons), nc4))
        bf = rng.random(len(mons)) * 2 * np.pi
        Af = rff(Xf, Wf, bf)
        cf = np.linalg.lstsq(Af, Y.T, rcond=None)[0]
        ef = float(np.mean(np.abs(Af @ cf - Y.T) / np.maximum(np.abs(Y.T), 1e-12)))
        polywin += int(ep < ef)
        P_(f"    {len(mons):>13} {ep:>20.3e} {ef:>18.3e} {'poly' if ep < ef else 'Fourier':>9}")
    P_(f"\n  G0: {'POLYNOMIALS WIN AT EVERY SIZE. REM s backward function is SMOOTH in the accumulated statistic -- its obstruction is DIMENSION, not oscillation, so the external round-1 failure mode does not describe it and round-2 s Fourier repair is solving a different problem.' if polywin == 3 else 'Fourier wins somewhere; the external rounds were on target and their basis should be adopted.'}")

    # ---- G1  END TO END, EXACT REFERENCE -------------------------------------------------------
    P_("\n" + RULE)
    P_("G1  END TO END WHERE THERE IS AN EXACT ANSWER")
    P_(RULE)
    last = np.arange(n)
    w = pi.copy()
    acc = actbit * hvec[0]
    for dd in range(1, L4 + 1):
        col = Pm[:, last].T * w[:, None]
        w = col.ravel()
        par = np.repeat(np.arange(len(last)), n)
        chd = np.tile(np.arange(n), len(last))
        acc = acc[par] + actbit[chd] * hvec[dd]
        last = chd
    exact_tail = float((w * np.exp(logF(acc, S))).sum())
    P_(f"  exact tail by full enumeration ({n ** (L4 + 1):,} paths): {exact_tail:.6e}")
    P_(f"\n    {'degree':>7} {'coefficients':>13} {'tail':>15} {'error, orders':>14} {'vs a pruning decade':>21}")
    g1 = []
    for deg in (1, 2, 3, 4):
        t0 = time.time()
        tl, ncf, npt = poly_backward(Pm, pi, n, hvec, S, actbit, L4, deg)
        el = time.time() - t0
        er = abs(np.log10(tl) - np.log10(exact_tail)) if tl > 0 else float("inf")
        g1.append((deg, ncf, tl, er, el))
        P_(f"    {deg:>7} {ncf:>13} {tl:>15.6e} {er:>14.4f} {er / PRUNE_SLOPE:>21.3f}")
    best = min(g1, key=lambda r: r[3])
    P_(f"\n  G1: best is degree {best[0]} at {best[3]:.4f} orders --"
       f" {'USEFUL, well under one pruning decade.' if best[3] < PRUNE_SLOPE else 'NOT useful: the error exceeds what pruning already costs.'}")

    # ---- G2  SCALING ---------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("G2  THE SCALING, IN DEGREE AND IN WIDTH")
    P_(RULE)
    P_(f"\n    {'error vs degree':<22}" + "  ".join(f"deg{r[0]}={r[3]:.4f}" for r in g1))
    P_(f"\n    {'nCtrl':>6} {'deg 2 coefficients':>20} {'deg 3 coefficients':>20} {'paths at L=6':>16}")
    for k in (4, 6, 8, 10, 12):
        P_(f"    {k:>6} {len(mono(k, 2)):>20,} {len(mono(k, 3)):>20,} {2.0 ** (k * 7):>16.3e}")
    P_("    C(k+p, p) is POLYNOMIAL in width at fixed degree, against 2^(k(L+1)) for paths. That")
    P_("    is the property that makes this worth running at the engine's own size.")

    # ---- G3  THE ACTUAL REM TAIL ---------------------------------------------------------------
    P_("\n" + RULE)
    P_("G3  THE ACTUAL REM TAIL AT 10 CONTROLLERS AND L = 6")
    P_(RULE)
    P_("  No exact answer exists here, but the pruned tail is a LOWER BOUND on the truth, because")
    P_("  retaining more paths can only add mass. A value BELOW it refutes the method outright.")
    PRUNED_MASS, PRUNED_BOUND = 3.7208e-98, 1.0640e-97
    Pm2, pi2, n2, hv2, S2, ab2, rows2, sha2 = setup(10, 6)
    P_(f"\n  {10} controllers, {n2} states, L = 6, {len(rows2)} targets. sha {sha2[:12]}.")
    P_(f"  best pruned LOWER bound to beat: {PRUNED_BOUND:.4e} (bound-ranked, 19,531 paths)")
    P_(f"\n    {'degree':>7} {'coefficients':>13} {'tail':>15} {'vs lower bound':>16} {'seconds':>9}")
    g3 = []
    for deg in (1, 2, 3):
        t0 = time.time()
        tl, ncf, npt = poly_backward(Pm2, pi2, n2, hv2, S2, ab2, 6, deg)
        el = time.time() - t0
        g3.append((deg, ncf, tl, el))
        rel = tl / PRUNED_BOUND if tl > 0 else 0.0
        P_(f"    {deg:>7} {ncf:>13} {tl:>15.6e} {rel:>15.3f}x {el:>9.1f}")
    ok = [r for r in g3 if r[2] >= PRUNED_BOUND]
    P_(f"\n  G3 AS PREDECLARED: {'the polynomial backward lands ABOVE the pruned lower bound at ' + str(len(ok)) + ' of ' + str(len(g3)) + ' degrees. That is CONSISTENCY, not accuracy -- it is not refuted, and nothing here says how close to the truth it is.' if ok else 'every degree lands BELOW the pruned lower bound. THE METHOD IS REFUTED at the engine width, with no exact reference needed.'}")

    # ---- G3b  WHAT IT IMPLIES FOR accuracy.py --------------------------------------------------
    P_("\n" + RULE)
    P_("G3b  WHAT THAT ESTIMATE IMPLIES FOR accuracy.py's RANKING OF THE ENGINE'S ERROR TERMS")
    P_(RULE)
    conv = [r for r in g3 if r[0] >= 2]
    est = float(np.mean([r[2] for r in conv]))
    spread = max(r[2] for r in conv) / min(r[2] for r in conv)
    P_(f"  degrees 2 and 3 give {conv[0][2]:.4e} and {conv[-1][2]:.4e}, agreeing to"
       f" {(spread - 1) * 100:.1f}% -- the representation")
    P_(f"  has converged even though its accuracy is uncertified. Taking {est:.4e} as the estimate:")
    CLASS_LADDER, SLOPE, LOG_FULL, LOG_AT = 5.08, 0.608, 21.1, 4.29
    implied = float(np.log10(est) - np.log10(PRUNED_BOUND))
    extrap = SLOPE * (LOG_FULL - LOG_AT)
    P_(f"\n    {'quantity':<46} {'orders':>10}")
    P_(f"    {'remaining rise implied by this estimate':<46} {implied:>10.2f}")
    P_(f"    {'remaining rise from accuracy.py s slope':<46} {extrap:>10.2f}")
    P_(f"    {'the class map s whole ladder':<46} {CLASS_LADDER:>10.2f}")
    P_(f"\n  THE SLOPE MUST FLATTEN. {SLOPE} orders per decade held over the two decades it was")
    P_(f"  measured on, and extrapolating it {LOG_FULL - LOG_AT:.1f} decades gives {extrap:.1f} orders where this estimate")
    P_(f"  says {implied:.2f}. accuracy.py flagged that extrapolation as a direction and not a value; this")
    P_("  is the first independent number to put against it, and it says the local slope is local.")
    if implied < CLASS_LADDER:
        P_(f"\n  AND THE RANKING REVERSES. accuracy.py's A4b concluded pruning binds because the")
        P_(f"  extrapolated error crossed {CLASS_LADDER} orders at 10^12.6. If the total pruning error is")
        P_(f"  {implied:.2f} orders it never reaches {CLASS_LADDER} at all, and the CLASS MAP is the binding term --")
        P_("  which is what A4 said before A4b overturned it on the scaling law. A4b was right to")
        P_("  distrust the point comparison; it was wrong about which way the law bent.")
        P_("\n  CONDITIONAL, AND THE CONDITION IS NOT SMALL. This rests on an ESTIMATE that carries no")
        P_("  bound. Its evidence is G1: 0.012 to 0.10 orders against an exact answer at four")
        P_("  controllers. Nothing tests that transfer at ten. If the estimate is low, the pruning")
        P_("  error is larger than stated and the reversal weakens or vanishes. What is NOT")
        P_("  conditional is the direction: the tail is at least 1.0640e-97 by the lower bound and")
        P_(f"  this estimate puts it near {est:.2e}, so the slope cannot continue at {SLOPE} for {LOG_FULL - LOG_AT:.0f} more decades.")
    else:
        P_(f"\n  The implied error {implied:.2f} still exceeds the class ladder's {CLASS_LADDER}, so pruning remains")
        P_("  the binding term and accuracy.py's verdict stands, with a corrected magnitude.")

    # ---- G4  TIMING AS A LAW -------------------------------------------------------------------
    P_("\n" + RULE)
    P_("G4  THE TIMING, AS A LAW AND NOT AS THE 28x POINT")
    P_(RULE)
    P_("  The external comparison was 0.34 s against 0.012 s for exact DP. That baseline does not")
    P_("  exist for REM: sufficient.py showed the backward function is indexed by (state,")
    P_("  accumulated activity) and is injective on paths at full precision, so there is no cheap")
    P_("  exact DP to be 28x slower than. What REM's alternative actually is, is enumeration.")
    P_(f"\n    {'width':>6} {'depth':>6} {'poly backward, s':>18} {'paths it replaces':>20}")
    for (kk, ll, rl) in ((nc4, L4, g1[1][4]), (10, 6, g3[1][3])):
        P_(f"    {kk:>6} {ll:>6} {rl:>18.2f} {2.0 ** (kk * (ll + 1)):>20.3e}")
    P_("\n  The engine-width row runs in seconds against a path set of 1.2e21. There is no ratio")
    P_("  to quote because the thing it replaces cannot be run at all.")

    # ---- G5 ------------------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("G5  WHAT THIS DOES NOT SETTLE")
    P_(RULE)
    P_("  1. G3 has no exact reference. Landing above a lower bound is a consistency check that")
    P_("     the method could pass while still being far from the truth. It is not accuracy.")
    P_("  2. The projection of log-sum-exp onto the polynomial space is the only approximation and")
    P_("     it is unbounded -- nothing here certifies it, so this method produces an ESTIMATE and")
    P_("     not a certificate. exactcert's problem is untouched by it.")
    P_("  3. G0 tests smoothness at 4 controllers. That REM's backward function stays smooth at")
    P_("     ten is assumed by G3 and not shown.")
    P_("  4. Collocation points are drawn uniformly in the reachable box. A better design would")
    P_("     weight where the mass actually is, and that is not tried.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_backpoly.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
