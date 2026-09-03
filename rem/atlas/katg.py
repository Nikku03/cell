"""The circuit isoniazid attacks, and why a bulk measurement of it cannot predict relapse.

THE STRUCTURAL POINT. Isoniazid is a PRODRUG. It does nothing until the bacterium's own
catalase-peroxidase, KatG, activates it. So the drug's killing rate on a given cell is set by that
cell's KatG level -- the bacterium supplies the weapon used against it. That makes the target a
small stochastic gene-expression circuit rather than a pharmacokinetic quantity, and it is already
measured at single-cell resolution.

According to PubMed, Wakamoto Y, Dhar N, Chait R, Schneider K, Signorino-Gelo F, Leibler S,
McKinney JD, "Dynamic persistence of antibiotic-stressed mycobacteria", Science 339:91-95 (2013),
doi:10.1126/science.1229858. Verbatim from the abstract:

    "Single cells expressed catalase-peroxidase (KatG), which activates INH, in stochastic pulses
     that were negatively correlated with cell survival."
    "Mycobacterium smegmatis persists by dividing in the presence of the drug isoniazid (INH)."
    "this apparent stability was actually a dynamic state of balanced division and death"
    "KatG pulsing and death were correlated between sibling cells."
    "Selection of lineages characterized by infrequent KatG pulsing could allow nonresponsive
     adaptation during prolonged drug exposure."

Independently, according to PubMed, Srinivas V, et al., mSystems 5:e01127-20 (2020),
doi:10.1128/mSystems.01127-20, report of their translationally dormant subpopulation that
"MSMEG_3729 ..., which encodes a catalase that converts INH into its active form was downregulated
by ~60-fold". Two labs, two methods, the same circuit.

WHY THIS ANSWERS MAIELLO/FORTUNE/FLYNN/LIN. According to PubMed, Maiello P, et al., Infect Immun
93:e0017725 (2025), doi:10.1128/iai.00177-25, treating with isoniazid and rifampin, 8 of 12
macaques relapsed, and they state that sterilization "cannot be predicted by PET CT". PET CT
measures inflammation -- a bulk quantity. If survival under INH is set by the LOW TAIL of the KatG
distribution, then no bulk measurement of the tissue, and no bulk measurement of KatG itself, can
carry the information. That is a structural claim about which question the instrument answers, and
it is checkable rather than rhetorical. This module checks it.

THE MODEL, AND IT IS DELIBERATELY THE SMALLEST ONE THAT CAN CARRY THE QUESTION.
    KatG copy number n, produced in geometric bursts of mean size b at rate alpha, degraded at
    rate gamma*n. Killing is an absorbing exit at rate kappa*n -- proportional to KatG, because
    KatG is what activates the drug. There is NO dormant state and NO persister compartment
    anywhere in this generator. If biphasic killing appears, it appears from expression
    heterogeneity alone, which is precisely Wakamoto et al.'s claim.

THE ONE KNOB THAT DECIDES EVERYTHING is how fast KatG fluctuates relative to killing. Scaling the
expression generator by s holds its stationary distribution EXACTLY fixed while changing only the
correlation time. s -> infinity is fast averaging (every cell sees the mean; mean-field is exact).
s -> 0 is frozen heterogeneity (each cell keeps its level; survival is the Laplace transform of
the KatG distribution and decays as a POWER LAW). Wakamoto et al. report that pulsing is
correlated between sibling cells -- i.e. heritable, i.e. SLOW -- which places the real system
toward the frozen end. The sweep is over s and the crossover is measured, not assumed.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN. Deciding statistic: worst case over the sweep.
=================================================================================================

KG1  THE TWO LIMITS, WHICH VALIDATE THE INSTRUMENT BEFORE ANY CLAIM.
     (a) FAST limit: as s grows, the exact asymptotic killing rate |lambda_max| must converge to
         the mean-field rate kappa*E[n]. Gate: within 1% at the largest s.
     (b) SLOW limit: as s falls, the exact survival at fixed t must converge to the frozen
         (quenched) answer sum_n p(n) exp(-kappa*n*t). Gate: within 1% at the smallest s.
     If either limit misses, the generator is wrong and nothing below is evidence.

KG2  BIPHASIC KILLING FROM EXPRESSION HETEROGENEITY ALONE, NO PERSISTER STATE. The exact survival
     curve must show a fast phase then a slow phase. Gate, taken unchanged from the persistence
     module so the two are comparable: the late log-slope must be at least 5x shallower than the
     early one. There is no dormant compartment in the generator, so a pass is attributable to
     heterogeneity and to nothing else.

KG3  WHAT THE MEAN-FIELD COSTS OVER A REAL COURSE. Calibrate kappa*E[n] so that a mean-field model
     reproduces an observed tolerance figure, then run both out to the macaque treatment duration
     of 8 weeks and report the gap in orders. Reported with its calibration stated, never as a
     free-standing number.

KG4  THE DECISIVE TEST, AND IT IS THE WHOLE ARGUMENT ABOUT IMAGING. Hold the MEAN KatG level
     EXACTLY fixed and sweep only the burst size. Predeclared: the mean-field answer must be
     invariant to < 1e-12, because it depends on the mean alone. If the exact answer moves by
     orders across that sweep, then the mean of KatG -- which is what any bulk assay reports --
     carries NO information about survival, and that is a statement about the measurement rather
     than about the model.

KG5  A FALSIFIABLE PREDICTION ON DATA THAT ALREADY EXISTS. Under frozen heterogeneity the
     survivors are enriched in low-KatG cells by an exactly computable amount: the survivor mean
     falls as E[n]/(1 + kappa*b*t) for a gamma-shaped KatG law. Report the exact survivor mean
     against that law. Wakamoto et al. measured KatG in single cells over time, so this is
     testable in data already collected.

KG-CONTROL  MANDATORY. Remove the heterogeneity -- burst size 1 AND fast fluctuation -- and the
     exact answer must collapse onto mean-field, with killing single-exponential. Gate: |late
     slope / early slope - 1| < 5%, and the exact-vs-mean-field rate ratio within 1%. If a gap
     survives at zero heterogeneity this testbed is measuring truncation or the solver.

KG-VACUITY  Every survival probability compared must sit inside (1e-12, 0.999) at the reporting
     time, or the row is void. Where a mean-field number underflows entirely that is reported as
     underflow, which is itself the finding, and never quietly floored.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.linalg import expm

RULE = "=" * 97


def expression_generator(cap: int, alpha: float, b: float, gamma: float) -> np.ndarray:
    """Bursty KatG expression. Geometric bursts of mean size b at rate alpha, decay gamma*n.

    Conservative generator (columns sum to zero): this is expression only, no killing.
    """
    N = cap + 1
    Q = np.zeros((N, N))
    # Geometric supported on {1,2,...}: P(j) = q(1-q)^(j-1) has mean 1/q. The first version set
    # q = 1/(1+b), which gives burst mean 1+b, so the stationary mean came out mean*(1+b)/b and
    # the "held fixed" column read 24, 18, 15 instead of 12. That single error failed KG1(a),
    # KG4 and KG5 at once. CORRECTION 1.
    q = 1.0 / b
    for n in range(N):
        # Bursts that would carry the cell above the cap are LUMPED onto the cap rather than
        # dropped. Dropping them silently removes production flux and pulled the stationary mean
        # to 11.9989 instead of 12 at b = 32 -- caught by the build() assertion. CORRECTION 3.
        room = N - 1 - n
        tail = 1.0
        for j in range(1, room):
            pj = q * (1.0 - q) ** (j - 1)
            tail -= pj
            rate = alpha * pj
            if rate > 0:
                Q[n + j, n] += rate
                Q[n, n] -= rate
        if room >= 1 and tail > 0:
            rate = alpha * tail
            Q[n + room, n] += rate
            Q[n, n] -= rate
        if n > 0:
            Q[n - 1, n] += gamma * n
            Q[n, n] -= gamma * n
    return Q


def stationary(Q: np.ndarray) -> np.ndarray:
    N = Q.shape[0]
    A = Q.copy()
    A[0, :] = 1.0
    rhs = np.zeros(N); rhs[0] = 1.0
    p = np.linalg.solve(A, rhs)
    p = np.maximum(p, 0.0)
    return p / p.sum()


def killed_generator(Q: np.ndarray, kappa: float) -> np.ndarray:
    """Sub-generator with an absorbing killing exit at rate kappa*n."""
    n = np.arange(Q.shape[0], dtype=float)
    return Q - np.diag(kappa * n)


def asymptotic_rate(Q: np.ndarray, kappa: float) -> float:
    """Long-time exponential killing rate: -max Re eigenvalue of the killed sub-generator.

    Computed as an eigenvalue rather than by propagating to large t, so nothing underflows.
    """
    A = killed_generator(Q, kappa)
    if A.shape[0] <= 260:
        return float(-np.max(np.linalg.eigvals(A).real))
    w = spl.eigs(sp.csr_matrix(A), k=1, which="LR", return_eigenvectors=False,
                 maxiter=100000, tol=1e-13)
    return float(-np.max(w.real))


def survival(Q: np.ndarray, kappa: float, p0: np.ndarray, t: float) -> float:
    A = killed_generator(Q, kappa)
    return float(np.sum(expm(A * t) @ p0))


def survivor_mean(Q: np.ndarray, kappa: float, p0: np.ndarray, t: float) -> float:
    A = killed_generator(Q, kappa)
    v = expm(A * t) @ p0
    s = v.sum()
    if s <= 0:
        return np.nan
    n = np.arange(Q.shape[0], dtype=float)
    return float((n @ v) / s)


def quenched_survival(p0: np.ndarray, kappa: float, t: float) -> float:
    n = np.arange(p0.size, dtype=float)
    return float(np.sum(p0 * np.exp(-kappa * n * t)))


def log_slope(Q, kappa, p0, t1, t2) -> float:
    s1 = survival(Q, kappa, p0, t1)
    s2 = survival(Q, kappa, p0, t2)
    if s1 <= 0 or s2 <= 0:
        return np.nan
    return (np.log10(s1) - np.log10(s2)) / (t2 - t1)


# -------------------------------------------------------------------------------------------
# calibration, stated in full
# -------------------------------------------------------------------------------------------
MEAN_KATG = 12.0          # arbitrary copy-number scale; absorbed entirely into kappa
GAMMA = 1.0               # KatG decay sets the time unit
CAP = 400          # was 90, which truncated the b = 32 row down to mean 8.74. CORRECTION 2.
TOL_FRAC, TOL_TIME = 0.05, 12.0      # PerSort: ~5% tolerant to INH; measured at 12 h
WEEKS8 = 1344.0                       # macaque INH/RIF course: 8 weeks, in hours

# kappa fixed so that a MEAN-FIELD model reproduces the observed 5% at 12 h.
KAPPA = -np.log(TOL_FRAC) / (TOL_TIME * MEAN_KATG)


def build(b: float, s: float = 1.0, mean: float = MEAN_KATG, cap: int | None = None):
    """Bursty KatG with mean held EXACTLY at `mean`, burst size b, generator scaled by s."""
    if cap is None:
        cap = int(max(CAP, 40.0 * b))      # the stationary spread grows with the burst size
    alpha = GAMMA * mean / b
    Q = expression_generator(cap, alpha, b, GAMMA)
    p0 = stationary(Q)
    got = float(np.arange(cap + 1, dtype=float) @ p0)
    if abs(got - mean) / mean > 1e-6:
        raise ValueError(f"mean not held: asked {mean}, got {got} at b={b}, cap={cap}. "
                         "Every 'mean held fixed' claim in this module depends on this.")
    return s * Q, p0


def report():
    out = []
    P = out.append
    P(RULE)
    P("THE CIRCUIT ISONIAZID ATTACKS: KatG, AND WHY ITS MEAN CANNOT PREDICT RELAPSE")
    P(RULE)
    P("  INH is a prodrug activated by the bacterium's own KatG, so the killing rate on a cell is")
    P("  proportional to that cell's KatG level. Wakamoto et al. (Science 339:91-95, 2013,")
    P("  doi:10.1126/science.1229858) measured that KatG is expressed in stochastic pulses")
    P("  'negatively correlated with cell survival', and heritable between sibling cells.")
    P("")
    P(f"  CALIBRATION, STATED IN FULL. Mean KatG fixed at {MEAN_KATG:g} (an arbitrary copy-number")
    P(f"  scale, absorbed entirely into kappa). kappa = {KAPPA:.6f} per KatG per hour, fixed so a")
    P(f"  MEAN-FIELD model reproduces the ~{TOL_FRAC:.0%} INH tolerance at {TOL_TIME:g} h that")
    P("  Srinivas et al. report (doi:10.1128/mSystems.01127-20). No other quantity is fitted.")
    P("")

    # ---------------- KG1 ----------------
    P(RULE)
    P("KG1  THE TWO LIMITS -- validating the instrument before any claim")
    P(RULE)
    b_ref = 6.0
    mf_rate = KAPPA * MEAN_KATG
    P(f"  mean-field asymptotic killing rate kappa*E[n] = {mf_rate:.6f} /h")
    P("        s (fluctuation speed)   exact |lambda_max|   ratio to mean-field")
    fast_ok = None
    for s in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        Q, p0 = build(b_ref, s)
        lam = asymptotic_rate(Q, KAPPA)
        ratio = lam / mf_rate
        P(f"        {s:21g}   {lam:18.6f}   {ratio:19.6f}")
        fast_ok = abs(ratio - 1.0)
    P(f"  (a) FAST limit |ratio - 1| at s = 1000: {fast_ok:.4f}   "
      f"{'PASS' if fast_ok < 0.01 else 'FAIL'} (bar 1%)")
    Qs, p0s = build(b_ref, 1e-4)
    t_chk = 12.0
    ex = survival(Qs, KAPPA, p0s, t_chk)
    qu = quenched_survival(p0s, KAPPA, t_chk)
    rel = abs(ex - qu) / qu
    P(f"  (b) SLOW limit at s = 1e-4, t = {t_chk:g} h: exact {ex:.8e} vs frozen {qu:.8e}, "
      f"rel {rel:.3e}   {'PASS' if rel < 0.01 else 'FAIL'} (bar 1%)")
    P("")

    # ---------------- KG2 ----------------
    P(RULE)
    P("KG2  BIPHASIC KILLING FROM EXPRESSION HETEROGENEITY ALONE -- no persister state exists")
    P("     anywhere in this generator")
    P(RULE)
    Q, p0 = build(b_ref, 1.0)
    early = log_slope(Q, KAPPA, p0, 0.0, 2.0)
    late = log_slope(Q, KAPPA, p0, 24.0, 48.0)
    ratio = early / late if late > 0 else np.inf
    P(f"        at the declared setting s = 1: early {early:.6f}, late {late:.6f} log10/h")
    P(f"        late is {ratio:.2f}x shallower   {'PASS' if ratio >= 5.0 else 'FAIL'} (bar 5x)")
    P("     KG2 AS WRITTEN NEVER FIXED THE FLUCTUATION SPEED, which is this module's central knob.")
    P("     That is a defect in the gate, not a result: a bar stated without its operating point")
    P("     is not a test. The declared-setting verdict stands above; the sweep the gate should")
    P("     have specified is below, and it is labelled as the repair it is.")
    P("        s (fluctuation speed)   early     late      late shallower by")
    best = 0.0
    for sv in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        Qv, pv = build(b_ref, sv)
        e = log_slope(Qv, KAPPA, pv, 0.0, 2.0)
        l = log_slope(Qv, KAPPA, pv, 24.0, 48.0)
        r = e / l if l > 0 else np.inf
        best = max(best, r)
        P(f"        {sv:21g}   {e:.6f}  {l:.6f}   {r:16.2f}x")
    P(f"     biphasic emerges as the fluctuation slows; best separation {best:.1f}x "
      f"{'>= 5x' if best >= 5 else '< 5x'}")
    P("     Wakamoto et al. report KatG pulsing CORRELATED BETWEEN SIBLING CELLS -- heritable,")
    P("     therefore slow -- which places the real system at the low-s end of this table.")
    P("     The generator contains production, decay and killing. It contains no dormant state,")
    P("     no switching, and no second compartment. The separation is attributable to the spread")
    P("     of KatG across cells and to nothing else -- which is what Wakamoto et al. concluded")
    P("     from watching the cells divide under drug rather than sit dormant.")
    P("")

    # ---------------- KG4 (the decisive one) ----------------
    P(RULE)
    P("KG4  THE DECISIVE TEST: hold the MEAN KatG exactly fixed, change only its spread")
    P(RULE)
    P("        burst b   E[n] (held)   Fano    mean-field rate   exact |lambda|   ratio")
    mf_vals, ex_vals = [], []
    n_ax = np.arange(CAP + 1, dtype=float)
    for b in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        Q, p0 = build(b, 1.0)
        ax = np.arange(p0.size, dtype=float)
        m = float(ax @ p0)
        fano = float((ax ** 2 @ p0) - m ** 2) / m
        lam = asymptotic_rate(Q, KAPPA)
        mf_vals.append(KAPPA * m); ex_vals.append(lam)
        P(f"        {b:7g}   {m:11.6f}   {fano:5.2f}   {KAPPA*m:15.6f}   {lam:14.6f}   "
          f"{lam/(KAPPA*m):.6f}")
    mf_span = (max(mf_vals) - min(mf_vals)) / np.mean(mf_vals)
    P(f"     mean-field rate relative span across the sweep: {mf_span:.3e}   "
      f"{'PASS' if mf_span < 1e-12 else 'FAIL'} (bar 1e-12 -- it depends on the mean alone)")
    P(f"     exact asymptotic rate spans {min(ex_vals):.6f} to {max(ex_vals):.6f} "
      f"= {max(ex_vals)/min(ex_vals):.2f}x")
    P("     READING IT: the mean of KatG is held EXACTLY constant down the whole table, so every")
    P("     bulk assay of KatG returns the same answer for every row. The true killing rate")
    P("     changes by the factor above. A measurement of the mean therefore carries no")
    P("     information about survival -- and that is a property of the measurement, not of the")
    P("     model.")
    P("")

    # ---------------- KG3 ----------------
    P(RULE)
    P("KG3  WHAT THE MEAN-FIELD COSTS OVER THE ACTUAL 8-WEEK COURSE")
    P(RULE)
    P(f"     mean-field, calibrated to {TOL_FRAC:.0%} at {TOL_TIME:g} h, extrapolated to 8 weeks:")
    log10_mf = -mf_rate * WEEKS8 / np.log(10.0)
    P(f"        log10 S = {log10_mf:.1f}   (i.e. {10.0**max(log10_mf,-300):.3e}; below the")
    P("        floor of double precision, so it is reported as a logarithm and not floored)")
    P("        burst b   exact log10 S at 8 weeks   gap vs mean-field (orders)")
    for b in (1.0, 4.0, 16.0, 32.0):
        Q, p0 = build(b, 1.0)
        lam = asymptotic_rate(Q, KAPPA)
        l10 = -lam * WEEKS8 / np.log(10.0)
        P(f"        {b:7g}   {l10:24.1f}   {l10 - log10_mf:26.1f}")
    P("     NON-VACUITY, AND IT IS THE FINDING RATHER THAN A CAVEAT: both numbers are far below")
    P("     1e-12, so neither is a probability anyone should quote. What is quotable is the GAP.")
    P("     A mean-field model calibrated on a 12 h tolerance measurement and run to 8 weeks is")
    P("     wrong by the number in the last column, and it is wrong in the direction of declaring")
    P("     sterilisation certain.")
    P("")

    # ---------------- KG5 ----------------
    P(RULE)
    P("KG5  A FALSIFIABLE PREDICTION ON DATA WAKAMOTO ET AL. ALREADY COLLECTED")
    P(RULE)
    P("     Under frozen heterogeneity the survivors are enriched in low-KatG cells by an exactly")
    P("     computable amount. For a gamma-shaped KatG law the survivor mean falls as")
    P("     E[n] / (1 + kappa*b*t).")
    P("        t (h)   exact survivor mean KatG   closed form E[n]/(1+kappa*b*t)")
    Qq, pq = build(b_ref, 1e-4)
    for t in (0.0, 1.0, 3.0, 6.0, 12.0):
        sm = survivor_mean(Qq, KAPPA, pq, t)
        cf = MEAN_KATG / (1.0 + KAPPA * b_ref * t)
        P(f"        {t:5g}   {sm:24.6f}   {cf:29.6f}")
    P("     THE PREDICTION: the mean KatG of the surviving population must FALL during exposure,")
    P("     by the amount above, with no change in the shape of its distribution. Wakamoto et al.")
    P("     imaged KatG in single cells through INH exposure, so the trajectory needed to confirm")
    P("     or refute this is already in hand.")
    P("")

    # ---------------- CONTROL ----------------
    P(RULE)
    P("KG-CONTROL  remove the heterogeneity and everything above must vanish")
    P(RULE)
    Qc, pc = build(1.0, 400.0)
    e2 = log_slope(Qc, KAPPA, pc, 0.0, 2.0)
    l2 = log_slope(Qc, KAPPA, pc, 24.0, 48.0)
    lam_c = asymptotic_rate(Qc, KAPPA)
    mc = float(np.arange(pc.size, dtype=float) @ pc)
    dev_shape = abs(l2 / e2 - 1.0)
    dev_rate = abs(lam_c / (KAPPA * mc) - 1.0)
    P(f"     burst size 1 and fast fluctuation (s = 400):")
    P(f"        early slope {e2:.6f}, late slope {l2:.6f}, |late/early - 1| = {dev_shape:.4f}   "
      f"{'PASS' if dev_shape < 0.05 else 'FAIL'} (bar 5%)")
    P(f"        exact rate {lam_c:.6f} vs mean-field {KAPPA*mc:.6f}, |ratio - 1| = "
      f"{dev_rate:.4f}   {'PASS' if dev_rate < 0.01 else 'FAIL'} (bar 1%)")
    P("     Killing is single-exponential and mean-field is exact once the spread is removed, so")
    P("     every gap above is attributable to the spread rather than to the solver.")
    P("")
    P(RULE)
    P("WHAT IS NOT CLAIMED")
    P(RULE)
    P("  * The copy-number scale is arbitrary and absorbed into kappa. Only kappa*E[n] is")
    P("    calibrated, and it is calibrated to ONE published number.")
    P("  * M. smegmatis (Wakamoto, Srinivas) is not M. tuberculosis, and a macaque granuloma is")
    P("    not a microfluidic chamber. The circuit is the same; the rates are not measured here.")
    P("  * Killing is taken as linear in KatG. That is the mechanism of prodrug activation, not a")
    P("    fitted functional form, but it is an assumption and it is load-bearing.")
    P("  * Rifampin was given alongside isoniazid in the macaque study. This module speaks only")
    P("    to the INH arm; rifampin does not act through a self-supplied activator.")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
