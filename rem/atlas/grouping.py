"""Does the architecture's merge/split criterion control the error it is supposed to control?

WHAT IS UNDER TEST. The REM cell architecture decides when biological variables may be treated
INDEPENDENTLY and when they must be joined into a coupled group. Three of its criteria are:

    section 4   coupling score built on the Jacobian   I_ij = |dF_i/dx_j|
    section 14  split when the mutual information      I(A;B) ~ 0
    section 23  error budget                           eps_Y <~ sum_G |dY/dX_G| eps_G

All three are BULK quantities. A Jacobian is a local mean-field derivative; a mutual information
is an average of log[P_AB/(P_A P_B)] taken UNDER P, so it is dominated by the high-probability
bulk; a first-order sensitivity is a derivative of a mean.

Many biologically decisive observables are not means. They are conjunctive rare events: does ANY
lesion fail to sterilise, does ANY cell survive, does a memory element EVER flip. Those live in a
region of vanishing measure, which is precisely the region a bulk average does not see.

This build order has already measured five independent instances of the same phenomenon -- a bulk
quantity exactly right while the tail is wrong by orders:

    rem/atlas/RESULTS_pool.txt       mean preserved to 0.0000%, tail ratio 7909x
    rem/atlas/RESULTS_gapdetect.txt  mean identical to 0.000%, deep tail 78.51 orders
    rem/atlas/RESULTS_katg.txt       mean held to 2.9e-15, 8-week survival moves 19-58 orders
    rem/atlas/RESULTS_floquet.txt    mean exact to 1.01e-12, tail ratio 19.09x
    rem/atlas/RESULTS_candidates.txt mean-field relapse risk off by n*Cov(q,N)

So the question is not whether bulk and tail can diverge. It is whether the SPLIT CRITERION
inherits that divergence -- i.e. whether a mutual-information threshold tuned to keep a joint MEAN
accurate also keeps a joint TAIL accurate. If it does not, section 14 is unsound for the class of
question the engine most needs to answer, and the fix has to be part of the architecture rather
than bolted on afterwards.

THE SYSTEM. Two reporters A and B driven by one fluctuating resource pool P:

    pool     birth in bursts of mean b at rate alpha, death mu_R * P
    A        birth c_A * P,  death mu_A * A
    B        birth c_B * P,  death mu_B * B

A and B are conditionally independent GIVEN the pool and marginally dependent through it. This is
the canonical extrinsic-noise structure and it is exactly the resource-competition case the
architecture is built for. The coupling knob is the pool's burst size, swept with the pool MEAN
HELD EXACTLY FIXED, so that everything measured is attributable to correlation and not to a shift
in level. The exact joint P(P,A,B) is solved as a stationary null vector, so ground truth is
available and no sampling error enters.

THE THREE SCORES COMPARED.
    MI          = sum P(a,b) log[ P(a,b) / (P_A(a) P_B(b)) ]        -- section 14's criterion
    MI_tilt     = the same functional evaluated under the EXPONENTIALLY TILTED measure
                  P_theta(a,b) ~ P(a,b) exp(theta (a+b)), with theta fixed so that
                  E_theta[A+B] = T_A + T_B                          -- the proposed repair
    Lambda_tail = log[ P(A>=T_A, B>=T_B) / (P(A>=T_A) P(B>=T_B)) ]  -- the exact answer, and
                  therefore the reference to rank against, NOT a candidate criterion

The tilt is the standard large-deviations device: a rare event is governed by the measure under
which it is typical. MI_tilt is computable from a group's own joint, which is the object the
architecture already holds while the group exists, so it costs nothing extra to evaluate.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN. Deciding statistic: worst case over the sweep.
=================================================================================================

G1  EXACTNESS OF THE GROUND TRUTH. The stationary solution must satisfy pi^T Q = 0 to < 1e-10
    (max abs residual), and the marginals extracted from the 3-D joint must reproduce the mean of
    each species to < 1e-10 against its independently computed value. Without this every number
    below is a solver artefact.

G2  THE CORE TEST, AND IT CAN FAIL. Calibrate a threshold tau on MI as the architecture would:
    tau is the largest MI at which the joint MEAN observable E[A*B] is still accurate to 1%.
    Then, holding that same tau, report the WORST joint TAIL error over the rows that the
    criterion admits as splittable. Predeclared readings:
        worst tail error < 1%          -> section 14's criterion is SOUND for tails; report that
        worst tail error > 10x         -> section 14's criterion is UNSOUND for tails
    Anything between is reported as measured with no verdict attached.

G3  DISCRIMINATION. There must exist at least one swept point where MI is below tau AND the tail
    error exceeds 10x. If no such point exists anywhere in the sweep, G2 could not have failed and
    its result is VOID rather than a pass. This gate exists because a test that cannot fail has
    already been built once in this session.

G4  THE REPAIR. Rank the swept points by each score and correlate with the true tail error.
    Predeclared: MI_tilt must achieve Spearman rho > 0.9 against |Lambda_tail| in the same sweep
    where plain MI achieves rho < 0.5. Both are reported whichever way they fall. If MI already
    achieves rho > 0.9 there is nothing to repair and the repair is withdrawn.

G5  ZERO-COUPLING CONTROL, MANDATORY. With the pool made effectively deterministic (burst size 1
    and a fast pool), A and B become independent: MI, MI_tilt, Lambda_tail, the mean error and the
    tail error must ALL vanish together, each < 1e-6. If any survives at zero coupling, the
    testbed is measuring truncation or the solver rather than dependence, and every row above is
    that artefact instead.

G6  TAIL NON-VACUITY. Every joint tail probability entering a reported ratio must lie inside
    (1e-12, 0.1). A ratio between two numbers that are both effectively zero is not a measurement,
    and a probability pressed against 1 cannot move. Rows failing this are marked VOID, not passed.

G7  MEAN-EXACTNESS OF THE SWEEP. The pool mean must be identical across the whole burst sweep to
    < 1e-9 relative, so that no reported change can be attributed to a change in level.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.optimize import brentq
from scipy.stats import spearmanr

RULE = "=" * 97


def build_joint(pcap, acap, bcap, alpha, burst, muR, cA, muA, cB, muB):
    """Exact stationary joint P(pool, A, B). Bursty pool, two linear reporters."""
    nA, nB = acap + 1, bcap + 1
    idx = lambda p, a, b: (p * nA + a) * nB + b
    n = (pcap + 1) * nA * nB
    rows, cols, vals = [], [], []

    def add(i, j, r):
        if r > 0:
            rows.append(i); cols.append(j); vals.append(r)

    q = 1.0 / burst                      # geometric burst on {1,2,...}, mean = 1/q = burst
    for p in range(pcap + 1):
        room = pcap - p
        for a in range(nA):
            for b in range(nB):
                i = idx(p, a, b)
                # bursty pool production, with the tail above the cap lumped onto the cap so that
                # no production flux is silently discarded (this is what holds G7)
                tail = 1.0
                for j in range(1, room):
                    pj = q * (1.0 - q) ** (j - 1)
                    tail -= pj
                    add(i, idx(p + j, a, b), alpha * pj)
                if room >= 1 and tail > 0:
                    add(i, idx(p + room, a, b), alpha * tail)
                if p > 0:
                    add(i, idx(p - 1, a, b), muR * p)
                if a + 1 < nA:
                    add(i, idx(p, a + 1, b), cA * p)
                if a > 0:
                    add(i, idx(p, a - 1, b), muA * a)
                if b + 1 < nB:
                    add(i, idx(p, a, b + 1), cB * p)
                if b > 0:
                    add(i, idx(p, a, b - 1), muB * b)

    r = np.array(rows); c = np.array(cols); v = np.array(vals, float)
    diag = np.bincount(r, weights=v, minlength=n)
    Q = sp.coo_matrix((np.concatenate([v, -diag]),
                       (np.concatenate([r, np.arange(n)]),
                        np.concatenate([c, np.arange(n)]))), shape=(n, n)).tocsr()
    # stationary: normalisation row on the HIGHEST-PROBABILITY state, per the standing rule
    A0 = Q.T.tolil(); A0[0, :] = 1.0
    rhs = np.zeros(n); rhs[0] = 1.0
    p0 = np.maximum(spl.spsolve(A0.tocsr(), rhs), 0.0)
    mode = int(np.argmax(p0))
    A1 = Q.T.tolil(); A1[mode, :] = 1.0
    rhs = np.zeros(n); rhs[mode] = 1.0
    pi = np.maximum(spl.spsolve(A1.tocsr(), rhs), 0.0)
    pi = pi / pi.sum()
    resid = float(np.max(np.abs(Q.T @ pi)))
    return pi.reshape(pcap + 1, nA, nB), resid


def scores(PAB, TA, TB):
    """MI, tilted MI, tail lift, and the joint mean / joint tail errors from factorising."""
    PA = PAB.sum(axis=1); PB = PAB.sum(axis=0)
    A = np.arange(PAB.shape[0], dtype=float); B = np.arange(PAB.shape[1], dtype=float)
    outer = np.outer(PA, PB)

    m = (PAB > 0) & (outer > 0)
    MI = float(np.sum(PAB[m] * np.log(PAB[m] / outer[m])))

    # joint MEAN observable: E[A*B]. E[A+B] is exact under factorisation, so it cannot test it.
    exact_AB = float(A @ PAB @ B)
    fact_AB = float((A @ PA) * (B @ PB))
    mean_err = abs(exact_AB - fact_AB) / abs(exact_AB)

    # joint TAIL observable: P(A>=TA, B>=TB)
    exact_tail = float(PAB[TA:, TB:].sum())
    fact_tail = float(PA[TA:].sum() * PB[TB:].sum())
    tail_err = abs(exact_tail - fact_tail) / exact_tail if exact_tail > 0 else np.nan
    lam = np.log(exact_tail / fact_tail) if exact_tail > 0 and fact_tail > 0 else np.nan

    # exponential tilt: theta such that E_theta[A+B] = TA + TB
    S = A[:, None] + B[None, :]
    target = float(TA + TB)

    def mean_tilt(th):
        w = PAB * np.exp(th * (S - S.max()))
        return float((w * S).sum() / w.sum())

    try:
        if mean_tilt(0.0) >= target:
            th = 0.0
        else:
            hi = 0.1
            while mean_tilt(hi) < target and hi < 50:
                hi *= 1.7
            th = brentq(lambda t: mean_tilt(t) - target, 0.0, hi, xtol=1e-13)
    except Exception:
        th = np.nan

    if np.isfinite(th):
        W = PAB * np.exp(th * (S - S.max())); W /= W.sum()
        WA = W.sum(axis=1); WB = W.sum(axis=0); wo = np.outer(WA, WB)
        mw = (W > 0) & (wo > 0)
        MI_tilt = float(np.sum(W[mw] * np.log(W[mw] / wo[mw])))
    else:
        MI_tilt = np.nan

    return dict(MI=MI, MI_tilt=MI_tilt, theta=th, lam=lam, mean_err=mean_err,
                tail_err=tail_err, exact_tail=exact_tail, fact_tail=fact_tail,
                meanA=float(A @ PA), meanB=float(B @ PB))


# -------------------------------------------------------------------------------------------
# the sweep and the report
# -------------------------------------------------------------------------------------------

POOL_MEAN = 6.0
MU_R = 1.0
CA, MUA = 0.9, 0.8
CB, MUB = 0.9, 0.8
PCAP, ACAP, BCAP = 60, 30, 30
TA, TB = 16, 16
BURST = 2.0
# CORRECTION 1. The first sweep varied the pool's BURST SIZE. Two things went wrong.
#   (a) G7 FAILED: at pcap=34 the burst truncation lost production flux and the pool mean drifted
#       5.7594 -> 6.0000, a 4% span, so "only the correlation changed" was false.
#   (b) G2 was VOID: every row sat in the strongly-coupled corner (the weakest gave a 6.9% joint
#       mean error), so the sweep never entered the regime where a split would be ADMITTED and
#       the criterion could be calibrated at all.
# Sweeping the pool's SPEED instead fixes both. Scaling a generator leaves its null vector
# untouched, so the pool's stationary mean AND variance are exactly invariant along the sweep and
# only its correlation time moves -- from fast (A and B average over it and decouple) to slow
# (they see a frozen common value and are strongly correlated). That spans independence to strong
# coupling with an exactly controlled driver.
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def run_row(burst, pool_mean=POOL_MEAN, muR=MU_R, fast=1.0):
    alpha = muR * pool_mean / burst * fast
    PPAB, resid = build_joint(PCAP, ACAP, BCAP, alpha, burst, muR * fast,
                              CA, MUA, CB, MUB)
    PAB = PPAB.sum(axis=0)
    Ppool = PPAB.sum(axis=(1, 2))
    pm = float(np.arange(PCAP + 1) @ Ppool)
    s = scores(PAB, TA, TB)
    s["resid"] = resid
    s["pool_mean"] = pm
    return s


def report():
    out = []; P = out.append
    P(RULE)
    P("DOES A MUTUAL-INFORMATION SPLIT CRITERION CONTROL TAIL ERROR?")
    P(RULE)
    P("  Two reporters on one fluctuating resource pool. The pool's SPEED is swept; scaling a")
    P("  generator leaves its stationary law untouched, so the pool's mean and variance are")
    P("  exactly invariant and only its correlation time moves. Fast pool -> A and B average over")
    P("  it and decouple. Slow pool -> they see a frozen common value and are strongly coupled.")
    P(f"  Joint mean observable E[A*B]; joint tail observable P(A>={TA}, B>={TB}).")
    P("")

    rows = [(sp_, run_row(BURST, fast=sp_)) for sp_ in SPEEDS]

    # ---- G1 / G7 ----
    P(RULE)
    P("G1 / G7  GROUND TRUTH AND MEAN-EXACTNESS -- before any claim")
    P(RULE)
    worst_res = max(r["resid"] for _, r in rows)
    pms = [r["pool_mean"] for _, r in rows]
    span = (max(pms) - min(pms)) / np.mean(pms)
    P(f"  worst stationary residual max|Q^T pi| = {worst_res:.3e}   "
      f"{'PASS' if worst_res < 1e-10 else 'FAIL'} (bar 1e-10)")
    P(f"  pool mean across the sweep: {min(pms):.10f} to {max(pms):.10f}, "
      f"relative span {span:.3e}   {'PASS' if span < 1e-9 else 'FAIL'} (bar 1e-9)")
    P("")

    # ---- the sweep ----
    P(RULE)
    P("THE SWEEP")
    P(RULE)
    P(f"  {'speed':>6s} {'MI':>10s} {'MI_tilt':>10s} {'theta':>8s} {'E[AB] err':>11s}"
      f" {'P_joint tail':>13s} {'tail err':>11s} {'Lambda':>9s}")
    for b, r in rows:
        P(f"  {b:6.2f} {r['MI']:10.6f} {r['MI_tilt']:10.6f} {r['theta']:8.4f}"
          f" {r['mean_err']:11.3e} {r['exact_tail']:13.4e} {r['tail_err']:11.3e}"
          f" {r['lam']:+9.4f}")
    P("")

    # ---- G6 vacuity ----
    P(RULE)
    P("G6  TAIL NON-VACUITY")
    P(RULE)
    bad = [(b, r) for b, r in rows if not (1e-12 < r["exact_tail"] < 0.1)]
    P(f"  every joint tail probability inside (1e-12, 0.1): "
      f"{'YES -- all rows usable' if not bad else 'NO -- ' + str([b for b, _ in bad])}")
    P(f"  range of joint tail probability: {min(r['exact_tail'] for _, r in rows):.3e} to "
      f"{max(r['exact_tail'] for _, r in rows):.3e}")
    P(f"  {'PASS' if not bad else 'FAIL'}")
    P("")

    # ---- G2 core test ----
    P(RULE)
    P("G2  THE CORE TEST -- calibrate the threshold on the MEAN, then read the TAIL")
    P(RULE)
    ok_mean = [(b, r) for b, r in rows if r["mean_err"] < 0.01]
    if not ok_mean:
        P("  no row keeps the joint mean to 1%; the criterion cannot be calibrated. VOID.")
        tau = np.nan; admitted = []
    else:
        tau = max(r["MI"] for _, r in ok_mean)
        admitted = [(b, r) for b, r in rows if r["MI"] <= tau]
        P(f"  tau calibrated on the mean: largest MI with E[A*B] error < 1%  ->  tau = {tau:.6f}")
        P(f"  rows the criterion admits as splittable (MI <= tau): "
          f"{[b for b, _ in admitted]}")
        P("")
        P(f"  {'speed':>6s} {'MI':>10s} {'mean err':>11s} {'tail err':>11s} {'tail x':>10s}")
        for b, r in admitted:
            P(f"  {b:6.2f} {r['MI']:10.6f} {r['mean_err']:11.3e} {r['tail_err']:11.3e}"
              f" {1.0 + r['tail_err']:10.2f}x")
        wt = max(r["tail_err"] for _, r in admitted)
        wm = max(r["mean_err"] for _, r in admitted)
        P("")
        P(f"  worst MEAN error among admitted rows: {wm:.3e}  (this is what tau controls)")
        P(f"  worst TAIL error among admitted rows: {wt:.3e}  = {1.0 + wt:.2f}x")
        P(f"  ratio tail/mean error at the same threshold: {wt / wm:.1f}x")
        if wt < 0.01:
            P("  VERDICT: section 14's criterion is SOUND for tails in this system.")
        elif wt > 10.0:
            P("  VERDICT: section 14's criterion is UNSOUND for tails. A threshold that keeps the")
            P("  joint MEAN accurate to 1% admits splits that corrupt the joint TAIL by the")
            P("  factor above, and nothing in the criterion sees it.")
        else:
            P("  VERDICT: between the predeclared readings; reported as measured, no verdict.")
    P("")

    # ---- G3 discrimination ----
    P(RULE)
    P("G3  DISCRIMINATION -- could G2 have failed?")
    P(RULE)
    disc = [(b, r) for b, r in rows if np.isfinite(tau) and r["MI"] <= tau and r["tail_err"] > 10.0]
    P(f"  rows with MI <= tau AND tail error > 10x: {[b for b, _ in disc]}")
    P(f"  {'PASS -- the test could have failed and did' if disc else 'no such row'}")
    if not disc and np.isfinite(tau):
        P("  If G2 reported SOUND, that verdict is VOID rather than a pass: no point in the sweep")
        P("  could have produced the failing outcome.")
    P("")

    # ---- G4 the repair ----
    P(RULE)
    P("G4  THE REPAIR -- which score ranks the true tail error?")
    P(RULE)
    lam = np.array([abs(r["lam"]) for _, r in rows])
    mi = np.array([r["MI"] for _, r in rows])
    mit = np.array([r["MI_tilt"] for _, r in rows])
    good = np.isfinite(lam) & np.isfinite(mi) & np.isfinite(mit)
    rho_mi = spearmanr(mi[good], lam[good]).statistic
    rho_mit = spearmanr(mit[good], lam[good]).statistic
    P(f"  Spearman rho against |Lambda_tail| (the true tail error, in logs):")
    P(f"    plain MI      (section 14's criterion) : rho = {rho_mi:+.4f}")
    P(f"    tilted MI     (the proposed repair)    : rho = {rho_mit:+.4f}")
    if rho_mi > 0.9:
        P("  MI already ranks the tail error. There is nothing to repair and the repair is")
        P("  WITHDRAWN -- reported because it is the unwelcome outcome for the proposal.")
    elif rho_mit > 0.9 and rho_mi < 0.5:
        P("  PASS as predeclared: the tilt repairs the ranking where the bulk average does not.")
    else:
        P("  Neither predeclared branch is met; reported as measured.")
    P("")
    P("  MAGNITUDE, which ranking alone does not show:")
    P(f"  {'speed':>6s} {'MI':>10s} {'MI_tilt':>10s} {'tilt/MI':>9s} {'|Lambda|':>9s}")
    for b, r in rows:
        ratio = r["MI_tilt"] / r["MI"] if r["MI"] > 0 else np.nan
        P(f"  {b:6.2f} {r['MI']:10.6f} {r['MI_tilt']:10.6f} {ratio:9.2f} {abs(r['lam']):9.4f}")
    P("")

    # ---- G5 control ----
    P(RULE)
    P("G5  ZERO-COUPLING CONTROL -- everything must vanish together")
    P(RULE)
    P(f"  {'pool speed':>11s} {'MI':>12s} {'MI_tilt':>12s} {'mean err':>11s} {'tail err':>11s}"
      f" {'|Lambda|':>10s}")
    worst_ctrl = 0.0
    for fast in (64.0, 256.0, 1024.0, 4096.0):
        r = run_row(BURST, fast=fast)
        vals = [r["MI"], r["MI_tilt"], r["mean_err"], r["tail_err"], abs(r["lam"])]
        worst_ctrl = max(v for v in vals if np.isfinite(v))
        P(f"  {fast:11.0f} {r['MI']:12.3e} {r['MI_tilt']:12.3e} {r['mean_err']:11.3e}"
          f" {r['tail_err']:11.3e} {abs(r['lam']):10.3e}")
    P(f"  worst surviving quantity at the fastest pool: {worst_ctrl:.3e}   "
      f"{'PASS' if worst_ctrl < 1e-6 else 'FAIL'} (bar 1e-6)")
    P("  A fast pool is effectively deterministic, so A and B are independent and every")
    P("  dependence measure and every factorisation error must go to zero together. If any")
    P("  survived, the sweep above would be measuring truncation rather than correlation.")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
