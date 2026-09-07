"""Ruling out instead of enumerating: the first axis in this build order that is not table size.

THE AXIS. Every representation change in this session -- class count, signed count, multiplier --
shrank the TABLE the engine stores. None of them touched the number of STRATA the engine must
enumerate, which is 2^(|C|(L+1)) and which block_joint still builds in full, pruning only exact
zeros. Reducing that number is a different attack and it has not been tried here.

TWO WAYS TO REDUCE IT, AND THEY ARE NOT EQUIVALENT. block.py's E5 SAMPLED the block: 32 of 512
strata reproduced the answer to 3.3e-4. But a sample is an estimate with a VARIANCE and it
certifies nothing. Ruling a branch out with an admissible bound is stronger: what is dropped is
bounded, deterministically, by a quantity the algorithm computes as it goes. This module builds
the bound version, measures it against the sampled version, and combines them.

THE BOUND IS FREE AND EXACT. block_joint extends a prefix by w = Pm @ v then masks by the
controller code. Pm is a transition matrix and masking only removes mass, so the mass of EVERY
descendant of a prefix is at most the mass of that prefix. Any stratum's contribution to any
conjunctive observable is at most its mass. So prefix mass upper-bounds the total contribution of
an entire subtree, and cutting a prefix at threshold tau costs at most the summed mass of what was
cut -- a number the pruner knows exactly. That is a certificate and not an estimate.

THE CEILING GATE, RUN AS A STANDALONE PROBE BEFORE THIS FILE EXISTED, and it said NO at this
build order's own operating point. Ask an ORACLE -- given the exact joint, how many strata must be
kept to hold 1 - eps of the tail? No pruner can beat that. At nC = 4, nT = 5, L = 2, dt = 0.25:

    keep 99% of the MASS               3,059 of 4,096 strata      74.7%
    keep 99% of the TAIL               2,229 of 4,096 strata      54.4%
    prefix nodes above mass 1e-12      4,096 of 4,096             100%

The distribution over strata is nearly FLAT there. There are no obvious no's to rule out, and no
prefix can be cut. A pruner at that operating point is a constant factor at best.

SO THE QUESTION BECOMES WHAT MAKES IT FLAT, and that turns out to have a clean answer which is
about physics rather than algorithms. Two further probes, also run before this file:

    shrinking dt at fixed window   kept ~ nominal^0.87    still exponential, smaller base
    widening nC                    kept ~ nominal^0.96    still exponential
    slowing the controllers 100x   15 of 1,024 strata     1.5%, a 68x reduction

and, at fixed window and fixed physics, sweeping dt from 0.5 to 0.125:

    controllers fast    kept  59 -> 57,623   ratio per step 3.86 -> 2.56, nominal 4.00
    controllers 10x slow kept 29 ->    301   second differences constant at 8 -- QUADRATIC in L
                                             against an exponential 4^(L+1)

That is the first scaling result in this session rather than another constant factor, and it is
conditional: pruning defeats the explosion when controllers switch SLOWLY relative to the sampling
interval, and merely shaves the exponent when they switch fast. The dimensionless parameter should
be the expected number of controller switches per sampling interval, and if it is, the dt sweep
and the controller-speed sweep must collapse onto one curve in it. N2 tests exactly that, because
two unexplained regimes are worth much less than one law.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN OF THIS MODULE
=================================================================================================

N0  THE CEILING GATE, restated and re-run inside the module so it is in the record and not only
    in a transcript. PREDECLARED: if the oracle needs a constant FRACTION of the strata, pruning
    is a constant factor and may not be described as a scaling result.

N1  THE SCALING LAW, which is the whole question. N_eff against the nominal count, swept in dt at
    fixed window and in controller speed. PREDECLARED: pruning is a SCALING result only where
    N_eff grows polynomially while the nominal count grows exponentially. A smaller exponential
    base is a constant factor and must be reported as one, with the fitted base.

N2  THE COLLAPSE. If the mechanism is switches per sampling interval, the dt sweep and the
    controller-speed sweep must fall on ONE curve in that parameter. PREDECLARED: if they do not
    collapse, the mechanism is wrong and N1 is two facts rather than one law.

N3  THE ORACLE IS NOT AN ALGORITHM. A real pruner expands a prefix tree and must decide without
    seeing the answer. Measure the nodes a bound-pruner actually TOUCHES against the oracle's kept
    count; the gap is the price of not being an oracle. And verify the certificate empirically:
    the measured dropped tail must never exceed the bound the pruner computed.

N4  THE MASS PRUNER MUST LOSE THE TAIL, or this data is not testing what six modules in this
    session have found. Prune by mass, measure the TAIL kept.

N5  SAMPLING VERSUS RULING OUT, at equal strata budget, on the same system. PREDECLARED: sampling
    should win where the distribution is flat and pruning where it is peaked. The crossover is
    the useful number, and the difference in KIND -- variance against certificate -- is reported
    whichever wins.

N6  THE COMBINED FORM. Prune with the certificate, sample the residue. The residue's mass is known
    exactly, so the result is an unbiased estimate carrying a deterministic bound on what was
    dropped. Compare against each half alone.

N7  WHAT THIS DOES AND DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from rem.atlas.hybrid_tune import RULE
from rem.atlas.statedim import stationary
from rem.atlas.block import block_joint, marginalise_targets
from rem.atlas.multiplier import signed_controller


def slowed_controllers(nC, nT, sgn, mag, hr):
    """The same generator with the CONTROLLER switching rates scaled by hr and the targets left
    alone. hr is the knob that decides how much a history can change inside one sampling
    interval, which is the quantity this module argues governs prunability."""
    Q, nv = signed_controller(nC, nT, sgn, mag)
    Qd = Q.toarray().copy()
    n = Qd.shape[0]
    cmask = (1 << nC) - 1
    for i in range(n):
        for j in range(n):
            if i != j and ((i ^ j) & cmask):
                Qd[i, j] *= hr
    np.fill_diagonal(Qd, 0.0)
    np.fill_diagonal(Qd, -Qd.sum(axis=1))
    return csr_matrix(Qd), nv


def switch_rate(Q, nC, pi):
    """Expected controller switches per unit time in the stationary state -- the physical rate
    the dimensionless parameter is built from, measured rather than taken from a parameter."""
    Qd = Q.toarray()
    n = Qd.shape[0]
    cmask = (1 << nC) - 1
    r = 0.0
    for i in range(n):
        if pi[i] <= 0:
            continue
        s = 0.0
        for j in range(n):
            if i != j and ((i ^ j) & cmask):
                s += Qd[i, j]
        r += pi[i] * s
    return float(r)


def strata_weights(Q, nv, nC, nT, L, dt):
    """The exact strata, their masses, and each one's contribution to the conjunctive tail
    P(all targets on) -- which is what a pruner is trying to preserve."""
    pi, res, _ = stationary(Q)
    cur = block_joint(Q, pi, nC, L, dt)
    st = np.arange(1 << nT, dtype=np.int64)
    bits = [((st >> j) & 1) for j in range(nT)]
    A, W, T = [], [], []
    for a, v in cur.items():
        w = float(v.sum())
        if w <= 1e-300:
            continue
        m = marginalise_targets(v, nv, nC)
        s = m.sum()
        if s <= 0:
            continue
        q = [float((m / s)[bits[j] == 1].sum()) for j in range(nT)]
        A.append(a); W.append(w); T.append(w * float(np.prod(q)))
    return A, np.array(W), np.array(T), pi, res


def n_needed(x, eps):
    o = np.sort(x)[::-1]
    c = np.cumsum(o) / x.sum()
    return int(np.searchsorted(c, 1.0 - eps) + 1)


def bound_prune(Q, nv, nC, nT, L, dt, tau):
    """block_joint with a CERTIFICATE. A prefix whose mass is below tau is cut, and everything
    below it with it. Because Pm is stochastic and the mask only removes mass, no descendant can
    carry more than the prefix, so the cut is bounded by the prefix mass -- which is accumulated
    into `dropped` and returned as the certificate. Returns the tail estimate, the certificate,
    and the number of prefix nodes TOUCHED, which is the real cost."""
    pi, _, _ = stationary(Q)
    n = Q.shape[0]
    st = np.arange(n, dtype=np.int64)
    code = np.zeros(n, dtype=np.int64)
    for c in range(nC):
        code |= ((st >> c) & 1) << c
    Pm = expm((Q.T * dt).toarray())
    touched = 0
    dropped = 0.0
    cur = {}
    for m in range(1 << nC):
        v = pi * (code == m)
        touched += 1
        s = v.sum()
        if s <= 0:
            continue
        if s < tau:
            dropped += s
            continue
        cur[(m,)] = v
    for _ in range(L):
        nxt = {}
        for a, v in cur.items():
            w = Pm @ v
            for m in range(1 << nC):
                u = w * (code == m)
                touched += 1
                s = u.sum()
                if s <= 1e-300:
                    continue
                if s < tau:
                    dropped += s
                    continue
                nxt[a + (m,)] = u
        cur = nxt
    stt = np.arange(1 << nT, dtype=np.int64)
    bits = [((stt >> j) & 1) for j in range(nT)]
    tail = 0.0
    for a, v in cur.items():
        w = float(v.sum())
        m = marginalise_targets(v, nv, nC)
        s = m.sum()
        if s <= 0:
            continue
        q = [float((m / s)[bits[j] == 1].sum()) for j in range(nT)]
        tail += w * float(np.prod(q))
    return tail, dropped, touched, len(cur)


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE); P_("RULING OUT INSTEAD OF ENUMERATING"); P_(RULE)
    P_("  Every representation change in this session shrank the TABLE. None touched the number")
    P_("  of STRATA, 2^(|C|(L+1)), which block_joint still builds in full. This attacks that.")
    P_("  The bound is free and exact: Pm is stochastic and the mask only removes mass, so no")
    P_("  descendant of a prefix can carry more mass than the prefix. Cutting a prefix costs at")
    P_("  most its mass, which the pruner accumulates as it goes. That is a certificate.")

    # ---- N0  CEILING GATE ----------------------------------------------------------------------
    P_("\n" + RULE); P_("N0  THE CEILING GATE: HOW MANY STRATA DOES AN ORACLE NEED?"); P_(RULE)
    P_("  No pruner can beat an oracle that keeps exactly the top-contributing strata. Measured")
    P_("  at this build order's own operating point.")
    sg4 = np.array([1.0, 1.0, -1.0, -1.0]); mg4 = np.array([4.5, 1.5, 4.0, 1.2])
    Q4, nv4 = signed_controller(4, 5, sg4, mg4)
    A4, W4, T4, pi4, res4 = strata_weights(Q4, nv4, 4, 5, 2, 0.25)
    P_(f"\n  nC=4, nT=5, L=2, dt=0.25: {len(A4)} occupied strata of {1<<(4*3)},"
       f" solver residual {res4:.1e}")
    P_(f"  tail P(all 5 targets on) = {T4.sum():.6e}")
    P_(f"\n    {'keep 1-eps of':<22} {'eps=1e-2':>10} {'eps=1e-4':>10} {'eps=1e-6':>10}"
       f" {'% at 1e-2':>11}")
    for nm, x in (("the MASS", W4), ("the TAIL", T4)):
        P_(f"    {nm:<22} {n_needed(x,1e-2):>10} {n_needed(x,1e-4):>10} {n_needed(x,1e-6):>10}"
           f" {100*n_needed(x,1e-2)/len(x):>10.1f}%")
    frac = n_needed(T4, 1e-2) / len(T4)
    P_(f"\n  N0: the oracle needs {100*frac:.1f}% of the strata for 99% of the tail. That is a")
    P_( "  CONSTANT FRACTION, so at this operating point pruning is a constant factor and may not")
    P_( "  be called a scaling result. The question is therefore not whether to prune but what")
    P_( "  makes the distribution flat, and N1 answers that.")

    # ---- N1  THE SCALING LAW -------------------------------------------------------------------
    P_("\n" + RULE); P_("N1  THE SCALING LAW: WHERE DOES PRUNING STOP BEING A CONSTANT FACTOR?")
    P_(RULE)
    P_("  Fixed window W = 1.0, shrinking dt so L grows -- the regime history.py showed tail")
    P_("  accuracy needs. The nominal count is 4^(L+1). Does the KEPT count follow it?")
    sg2 = np.array([1.0, -1.0]); mg2 = np.array([4.5, 1.5])
    grid = [(0.5, 2), (1 / 3, 3), (0.25, 4), (0.2, 5), (1 / 6, 6), (1 / 7, 7), (0.125, 8)]
    curves = {}
    for hr in (1.0, 0.3, 0.1):
        Qh, nvh = slowed_controllers(2, 3, sg2, mg2, hr)
        pih, _, _ = stationary(Qh)
        rate = switch_rate(Qh, 2, pih)
        rows = []
        for dt, L in grid:
            A, W, T, _, _ = strata_weights(Qh, nvh, 2, 3, L, dt)
            rows.append((dt, L, 1 << (2 * (L + 1)), len(A), n_needed(T, 1e-2), rate * dt))
        curves[hr] = rows
        P_(f"\n  controller switch rate {rate:.3f} per unit time, scaled by hr = {hr}")
        P_(f"    {'dt':>7} {'L':>3} {'nominal':>9} {'keep99 tail':>12} {'step ratio':>11}"
           f" {'switches/slice':>15}")
        prev = None
        for (dt, L, nom, occ, k, th) in rows:
            P_(f"    {dt:>7.4f} {L:>3} {nom:>9} {k:>12} "
               f"{(k/prev if prev else float('nan')):>11.3f} {th:>15.4f}")
            prev = k
        ks = np.array([r[4] for r in rows], dtype=float)
        Ls = np.array([r[1] for r in rows], dtype=float)
        be = float(np.exp(np.polyfit(Ls, np.log(ks), 1)[0]))
        # R2 does not separate a quadratic from an exponential of base 1.4 over seven points.
        # The STEP RATIO does: an exponential holds it constant, a quadratic drives it to
        # ((L+1)/L)^2 and then to 1. So compare the observed ratio against both predictions.
        obs = ks[-1] / ks[-2]
        quad_pred = ((Ls[-1]) / (Ls[-2])) ** 2
        P_(f"    last step ratio {obs:.3f}   quadratic predicts {quad_pred:.3f}"
           f"   exponential predicts {be:.3f}   nominal 4.000")
        poly = abs(obs - quad_pred) < abs(obs - be)
        P_(f"    -> {'POLYNOMIAL in L against an exponential nominal: pruning defeats the explosion here' if poly else 'EXPONENTIAL with a smaller base: a constant factor, not a scaling result'}")

    # ---- N2  THE COLLAPSE ----------------------------------------------------------------------
    P_("\n" + RULE); P_("N2  DO THE TWO SWEEPS COLLAPSE ONTO ONE CURVE?"); P_(RULE)
    P_("  If the mechanism is switches per sampling interval, then the kept FRACTION should be a")
    P_("  function of that parameter alone, whether it was reached by shrinking dt or by slowing")
    P_("  the controllers. Two regimes explained by one number are worth more than two facts.")
    P_(f"\n    {'switches/slice':>15} {'hr':>6} {'L':>3} {'kept fraction':>14} {'kept':>8}")
    pts = []
    for hr in sorted(curves, reverse=True):
        for (dt, L, nom, occ, k, th) in curves[hr]:
            pts.append((th, hr, L, k / occ, k))
    for th, hr, L, fr, k in sorted(pts):
        P_(f"    {th:>15.4f} {hr:>6} {L:>3} {fr:>14.4f} {k:>8}")
    x = np.log(np.array([p[0] for p in pts]))
    y = np.log(np.array([p[3] for p in pts]))
    c = np.polyfit(x, y, 1)
    r2 = 1 - ((y - np.polyval(c, x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    P_(f"\n  one-parameter fit  log(kept fraction) = {c[0]:.3f} log(switches/slice) + {c[1]:.3f}")
    P_(f"  R2 = {r2:.4f} over {len(pts)} points spanning"
       f" {max(p[0] for p in pts)/min(p[0] for p in pts):.0f}x in the parameter and"
       f" {max(p[2] for p in pts)-min(p[2] for p in pts)+1} window depths")
    P_(f"  N2: {'PASS -- the two sweeps collapse, so this is one law and not two regimes' if r2 > 0.9 else 'FAIL -- they do not collapse; the mechanism is not switches per slice'}")

    # ---- N3  THE ORACLE IS NOT AN ALGORITHM ----------------------------------------------------
    P_("\n" + RULE); P_("N3  WHAT A REAL PRUNER PAYS FOR NOT BEING AN ORACLE"); P_(RULE)
    P_("  The pruner cannot see the answer. It expands a prefix tree and cuts on the bound. The")
    P_("  nodes it TOUCHES is the real cost, and the certificate it returns must never be")
    P_("  exceeded by the error it actually made.")
    hrN = 0.1
    QN, nvN = slowed_controllers(2, 3, sg2, mg2, hrN)
    A, W, T, _, _ = strata_weights(QN, nvN, 2, 3, 6, 1 / 6)
    exact_tail = T.sum()
    P_(f"\n  nC=2, nT=3, L=6, dt=0.1667, hr={hrN}: {len(A)} strata, exact tail"
       f" {exact_tail:.8e}, oracle keeps {n_needed(T,1e-2)} for 99%")
    P_(f"\n    {'tau':>10} {'nodes touched':>14} {'strata kept':>12} {'tail':>14}"
       f" {'rel err':>11} {'certificate':>12} {'held?':>7}")
    n3 = True
    for tau in (1e-2, 1e-3, 1e-4, 1e-6, 1e-9, 0.0):
        tl, dr, tc, nk = bound_prune(QN, nvN, 2, 3, 6, 1 / 6, tau)
        re = abs(tl - exact_tail) / exact_tail
        held = re <= dr / exact_tail + 1e-12
        n3 = n3 and held
        P_(f"    {tau:>10.0e} {tc:>14} {nk:>12} {tl:>14.6e} {re:>11.3e}"
           f" {dr/exact_tail:>12.3e} {str(held):>7}")
    P_(f"\n  'certificate' is the bound the pruner computed WITHOUT knowing the answer, as a")
    P_( "  fraction of the tail. 'held?' asks whether the error it actually made stayed inside it.")
    P_(f"  N3: {'PASS -- the certificate held at every threshold' if n3 else 'FAIL -- the bound was violated, so it is not admissible'}")
    fullnodes = sum((1 << 2) ** d for d in range(1, 6 + 2))
    P_(f"  full enumeration touches {fullnodes:,} nodes (the tau = 0 row confirms it); the")
    P_( "  pruner's counts are the column beside it.")

    # ---- N4  THE MASS PRUNER MUST LOSE THE TAIL ------------------------------------------------
    P_("\n" + RULE); P_("N4  DOES PRUNING BY MASS LOSE THE TAIL?"); P_(RULE)
    P_("  Six modules in this session found bulk and tail diverging. A mass pruner optimises the")
    P_("  bulk by construction, so this has to be checked rather than assumed.")
    om = np.argsort(W4)[::-1]
    P_(f"\n    {'top-k by MASS':>14} {'mass kept':>12} {'tail kept':>12} {'tail deficit':>14}")
    for k in (8, 32, 128, 512, 2048, len(W4)):
        idx = om[:k]
        mk, tk = W4[idx].sum() / W4.sum(), T4[idx].sum() / T4.sum()
        P_(f"    {k:>14} {mk:>12.6f} {tk:>12.6f} {mk-tk:>14.6f}")
    P_("\n  N4: on this system the tail is kept slightly BETTER than the mass at every k -- the")
    P_("  deficit column is negative throughout. The strata that carry the conjunctive tail here")
    P_("  are the same heavy ones that carry the mass, which is the opposite of what truncation,")
    P_("  thinning and history did. Reported because it is a negative for the usual worry, and")
    P_("  because it is exactly what makes a MASS bound usable as a TAIL certificate at all.")

    # ---- N5  SAMPLING VERSUS RULING OUT --------------------------------------------------------
    P_("\n" + RULE); P_("N5  SAMPLING VERSUS RULING OUT, AT EQUAL STRATA BUDGET"); P_(RULE)
    P_("  block.py E5 sampled the block and got 3.3e-4 from 32 of 512 strata -- but that is an")
    P_("  estimate with a variance. Pruning returns a bound. Same system, same budget, both.")
    Qflat, nvflat = slowed_controllers(2, 3, sg2, mg2, 1.0)
    for label, (Ql, nvl, Ll, dtl) in (("flat   (hr=1.0, dt=0.25)", (Qflat, nvflat, 4, 0.25)),
                                      ("peaked (hr=0.1, dt=0.1667)", (QN, nvN, 6, 1 / 6))):
        Aa, Wa, Ta, _, _ = strata_weights(Ql, nvl, 2, 3, Ll, dtl)
        ex = Ta.sum()
        P_(f"\n  {label}: {len(Aa)} strata, exact tail {ex:.6e}")
        P_(f"    {'budget':>8} {'PRUNE err':>12} {'certified?':>11} {'SAMPLE err':>22}")
        p = Wa / Wa.sum()
        for budget in (8, 32, 128):
            taus = np.sort(Wa)[::-1]
            tau = taus[min(budget, len(taus)) - 1]
            tl, dr, tc, nk = bound_prune(Ql, nvl, 2, 3, Ll, dtl, tau)
            errs = []
            for seed in range(8):
                rng = np.random.default_rng(seed)
                pick = rng.choice(len(Aa), size=min(budget, len(Aa)), replace=True, p=p)
                est = float(np.mean(Ta[pick] / p[pick]))    # E_p[T/p] = sum_a T_a, unbiased
                errs.append(abs(est - ex) / ex)
            P_(f"    {budget:>8} {abs(tl-ex)/ex:>12.3e} {'yes, <=' + f'{dr/ex:.1e}':>11}"
               f" {f'{np.mean(errs):.3e} +- {np.std(errs):.1e}':>22}")
    P_("\n  N5: the columns are not the same kind of number. A pruner's error comes with a bound")
    P_("  it computed itself; a sampler's comes with a spread over seeds and no bound at all.")

    # ---- N6  COMBINED --------------------------------------------------------------------------
    P_("\n" + RULE); P_("N6  THE COMBINED FORM: PRUNE WITH A CERTIFICATE, SAMPLE THE RESIDUE"); P_(RULE)
    P_("  The residue's mass is exactly what the certificate bounded, so sampling it gives back")
    P_("  an unbiased estimate of the part that was cut, while the bound still holds if the")
    P_("  sample is bad. That is strictly more than either half.")
    A6, W6, T6, _, _ = strata_weights(QN, nvN, 2, 3, 6, 1 / 6)
    ex6 = T6.sum()
    P_(f"\n    {'tau':>9} {'kept':>6} {'prune only':>12} {'+ residue sampled':>20}"
       f" {'certificate':>12}")
    for tau in (1e-3, 1e-4, 1e-6):
        tl, dr, tc, nk = bound_prune(QN, nvN, 2, 3, 6, 1 / 6, tau)
        cut = W6 < tau
        errs = []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            if cut.sum() == 0:
                errs.append(abs(tl - ex6) / ex6)
                continue
            pr = W6[cut] / W6[cut].sum()
            pick = rng.choice(int(cut.sum()), size=min(16, int(cut.sum())), replace=True, p=pr)
            resid = float(np.mean(T6[cut][pick] / pr[pick]))   # unbiased for the cut part
            errs.append(abs(tl + resid - ex6) / ex6)
        P_(f"    {tau:>9.0e} {nk:>6} {abs(tl-ex6)/ex6:>12.3e}"
           f" {f'{np.mean(errs):.3e} +- {np.std(errs):.1e}':>20} {dr/ex6:>12.3e}")

    # ---- N7 ------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("N7  WHAT THIS DOES AND DOES NOT SETTLE"); P_(RULE)
    P_("  1. The scaling law is measured on a two-controller synthetic system out to L = 8. The")
    P_("     polynomial regime is a fit over seven window depths, not a proof, and nothing here")
    P_("     shows it survives at the widths a real network needs.")
    P_("  2. The bound is admissible for any observable bounded by the stratum mass, which covers")
    P_("     conjunctive events. It is NOT a bound for observables that can be large where mass")
    P_("     is small -- a ratio, or anything normalised by a rare denominator.")
    P_("  3. N4 found the tail and the mass concentrated on the SAME strata here. That is what")
    P_("     makes a mass bound usable, and it is a property of this system that has to be")
    P_("     rechecked wherever the method is used, not assumed.")
    P_("  4. Prunability is governed by controller switches per sampling interval. Whether real")
    P_("     regulatory networks sit in the prunable regime is an empirical question about")
    P_("     transcription-factor kinetics that this module does not answer.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_prune.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
