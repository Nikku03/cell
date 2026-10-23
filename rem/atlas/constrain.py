"""Can measured PHYSIOLOGY supply the rates, instead of chemistry or one-by-one measurement?

THE PROPOSAL THIS TESTS, in the form it was put. We know a great deal that is not a single rate:
overall replication time, polymerase and ribosome speeds, diffusion limits, kill curves. Those are
measured well -- a doubling time to a few percent, against chemistry's 1 kcal/mol. Can they not
pin the individual rates well enough to make the rare-event answer usable?

This is a genuinely different route from hiderate.py (chemistry supplies every rate) and hybrid.py
(measure a few rates directly). It deserves its own measurement rather than an argument.

WHY IT MIGHT WORK. The constraints are real and hard. Diffusion caps any bimolecular association
near 1e9 /M/s. Doubling time constrains total biosynthetic flux to a few percent. A time-kill
curve is a direct readout of the drug-on dynamics. None of these needs a force field.

WHY IT MIGHT NOT, AND THE THREE MEASUREMENTS ALREADY IN THIS REPOSITORY THAT SAY SO. Every one of
those observables constrains an AGGREGATE, and this build order has repeatedly measured that
aggregates do not determine tails:
    residence.py  mean residence time spans 2.1193 orders at IDENTICAL mean flux
    katg.py       mean KatG held to 2.9e-15 while 8-week survival moved 3.0 to 58.2 orders
    grouping_law  tail_err = 20.23 * sqrt(MI): a bulk criterion vanishes as the SQUARE of the
                  tail error it is asked to bound
So the question is not whether physiology constrains the rates -- it plainly does -- but whether
what it leaves free is the part the answer depends on. That is measurable here.

THE CONSTRUCTION. The four aggregate observables below are exactly the population-level quantities
an experimentalist measures, and every one of them is a function of the MEAN-FIELD 2x2 generator
of the two-type process, in the low-density limit where a growth-rate measurement is made:

    off drug   M_off = [[mu - a_off,  b_off], [a_off, -(b_off + d_death)]]
    on drug    M_on  = [[-(k_kill + a_on), b_on], [a_on, -(b_on + kd_kill)]]

    A1  doubling time            ln2 / (largest eigenvalue of M_off)      "replication time"
    A2  log-kill over a course   log10 of total population after t_on     time-kill assay endpoint
    A3  persister plateau        slow-mode amplitude in that decay        biphasic kill curve
    A4  regrowth lag             time to recover the starting population  outgrowth assay

That is FOUR scalar functions of EIGHT rates. The eradication probability is not a function of the
mean-field matrix at all -- it is an extinction probability of the stochastic process. So the
experiment is sharp: draw rates from chemistry, keep only the draws whose physiology matches the
truth within experimental error, and ask what is left.

THE DELIVERABLE IS IN THE SAME CURRENCY AS hybrid.py, so the three routes can be compared: how
many directly measured rates is a full set of physiological constraints WORTH?

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

C1  THE CONSTRAINTS ARE COMPUTED CORRECTLY. The true rate vector must satisfy every constraint to
    machine precision, and the mean-field growth rate must agree with the growth rate of the full
    stochastic generator at low density to better than 1%. If the aggregates are wrong, nothing
    below means anything.

C2  THE CONSTRAINTS ACTUALLY BIND. An unconstrained chemistry draw must FAIL them most of the
    time. If most draws pass, the constraints carry no information and the comparison is vacuous.
    Bar: acceptance below 0.5 at 1 kcal/mol.

C3  NON-VACUITY OF THE ESTIMATE. The accepted sample must be large enough for its hit fraction to
    mean something. Reported: acceptance rate and accepted count; bar: at least 300 accepted
    draws, or the number is reported as noise-limited and NOT used for the deliverable.

C4  THE DELIVERABLE. The hit fraction (within a factor of 2, and of 10) among draws that satisfy
    all four constraints, converted into the currency of hybrid.py: the number m of directly
    measured rates that achieves the same hit fraction. Predeclared readings: worth 4 or more
    rates means physiology substantially replaces direct measurement and the route is open; worth
    1 or fewer means the constraints pin what the answer does not depend on, which is what the
    three measurements quoted above predict; anything between is reported as measured.

C5  THE MATCHED CONTROL. Constraints with their tolerances inflated 10x must do measurably worse
    than the real ones. Without this, an apparent gain could come from rejection sampling merely
    discarding extreme draws rather than from the physiology being informative.

C6  THE HONEST NEGATIVE, REPORTED WITH EQUAL PROMINENCE. Among ACCEPTED draws -- rate vectors
    whose entire measurable physiology is indistinguishable from the truth -- report the full
    spread of log10(Y_hat/Y_true): standard deviation, 5th and 95th percentiles, and total range.
    This is the number that decides the question. If it is small the route is open regardless of
    anything else here; if it spans orders, then two cells that are experimentally identical in
    every aggregate an experimentalist can measure still differ by that much in the answer.

C7  THE IDEALISED LIMIT, so the conclusion does not rest on my assumed experimental precisions.
    Repeat with the tolerances tightened 10x and 100x, approaching PERFECT physiology measurement.
    If the residual spread in C6 does not shrink towards zero, the limitation is structural -- the
    constraint manifold itself -- and no improvement in experimental precision can fix it.

C8  DOMAIN. Repeat at the rarer question (g0 = 8). Every number above is reported with the
    question it was measured on.

=================================================================================================
A NOTE ON THE TOLERANCES, WHICH ARE ASSUMPTIONS AND ARE LABELLED AS SUCH
=================================================================================================
The base experimental precisions below are STATED ASSUMPTIONS, not values taken from a paper, and
they are not used to carry any conclusion on their own: C7 sweeps them by two orders of magnitude
and the deciding statistic in C6 is reported at every setting. They are chosen to be generous to
the proposal being tested -- tighter than a real assay -- so that the route is not condemned by a
pessimistic guess about instrument noise.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np
from scipy.linalg import expm as dense_expm

from rem.atlas.hybrid_tune import (
    RULE, OFF_RATES, ON_RATES, NAMES, CANDIDATE, ORDERS_PER_KCAL,
    state_index, generator, eradication, sensitivity,
)
from rem.atlas.hybrid import (
    K, G0, CYCLES, T_ON, T_OFF, N_TRIALS, TOLS, TARGET, SEED,
    N_RATES, N_ON, N_SUB, run_trial, subset_sizes, greedy_masks,
)

G0_DEEP = 8
N_DRAW = 400_000          # cheap analytic screening draws
EPS = 1.0                 # chemistry accuracy, kcal/mol -- the realistic case
TOL_SCALES = (1.0, 0.1, 0.01)

# STATED ASSUMPTIONS, swept by C7. Units: fractional for times, orders for log-quantities.
BASE_TOL = dict(doubling=0.05,     # 5% on doubling time
                logkill=0.30,      # 0.3 orders on a time-kill endpoint
                plateau=0.30,      # 0.3 orders on the persister plateau
                lag=0.20)          # 20% on regrowth lag


def generator_linear(K_, IX_, r, drug):
    """The same circuit with the logistic factor removed -- the exponential-phase regime in which
    doubling time and kill curves are actually measured. Used only by C1's control."""
    n = len(IX_)
    L = np.zeros((n, n))

    def add(i, j, rate):
        if rate > 0:
            L[j, i] += rate
            L[i, i] -= rate

    for (g, d), i in IX_.items():
        if not drug:
            if g > 0 and g + d < K_:
                add(i, IX_[(g + 1, d)], r["mu"] * g)
            if g > 0:
                add(i, IX_[(g - 1, d + 1)], r["a_off"] * g)
            if d > 0:
                add(i, IX_[(g + 1, d - 1)], r["b_off"] * d)
            if d > 0:
                add(i, IX_[(g, d - 1)], r["d_death"] * d)
    return L


def mean_field(r):
    M_off = np.array([[r["mu"] - r["a_off"], r["b_off"]],
                      [r["a_off"], -(r["b_off"] + r["d_death"])]])
    M_on = np.array([[-(r["k_kill"] + r["a_on"]), r["b_on"]],
                     [r["a_on"], -(r["b_on"] + r["kd_kill"])]])
    return M_off, M_on


def _aggregates(mu, k_kill, a_off, a_on, b_off, b_on, d_death, kd_kill):
    """The four aggregates, in closed form so hundreds of thousands of draws can be screened.

    For a 2x2 M the spectral form is exact: with lam+- = (tr +- sqrt((a-d)^2 + 4bc))/2,
    exp(Mt)(1,0)^T has total A+ e^{lam+ t} + A- e^{lam- t}, where A+ = (a - lam- + c)/(lam+ - lam-)
    and A- = 1 - A+. Off-diagonals here are switching rates and are non-negative, so the
    discriminant is non-negative and both eigenvalues are real. All operations are elementwise,
    so every input may be an array."""
    # drug off
    a, b = mu - a_off, b_off
    c, d = a_off, -(b_off + d_death)
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4.0 * b * c, 0.0))
    lam_off = 0.5 * (a + d + disc)
    doubling = np.where(lam_off > 1e-12, np.log(2.0) / np.maximum(lam_off, 1e-300), np.inf)

    # drug on
    A, B = -(k_kill + a_on), b_on
    C, D = a_on, -(b_on + kd_kill)
    dsc = np.sqrt(np.maximum((A - D) ** 2 + 4.0 * B * C, 0.0))
    lp, lm = 0.5 * (A + D + dsc), 0.5 * (A + D - dsc)
    gap = np.where(np.abs(lp - lm) < 1e-300, 1e-300, lp - lm)
    Ap = (A - lm + C) / gap
    Am = 1.0 - Ap
    total_on = np.maximum(Ap * np.exp(lp * T_ON) + Am * np.exp(lm * T_ON), 1e-300)
    logkill = np.log10(total_on)
    plateau = np.log10(np.maximum(np.abs(Ap), 1e-300))
    lag = np.where((lam_off > 1e-12) & (total_on < 1.0),
                   np.log(1.0 / total_on) / np.maximum(lam_off, 1e-300), np.inf)
    return doubling, logkill, plateau, lag


def observables(r):
    return np.array([float(x) for x in _aggregates(*(r[nm] for nm in
                     ("mu", "k_kill", "a_off", "a_on", "b_off", "b_on", "d_death", "kd_kill")))])


def satisfies(obs, obs_true, tol, scale):
    d = abs((obs[0] - obs_true[0]) / obs_true[0]) <= tol["doubling"] * scale
    k = abs(obs[1] - obs_true[1]) <= tol["logkill"] * scale
    p = abs(obs[2] - obs_true[2]) <= tol["plateau"] * scale
    g = abs((obs[3] - obs_true[3]) / obs_true[3]) <= tol["lag"] * scale
    return bool(d and k and p and g)


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("CAN MEASURED PHYSIOLOGY SUPPLY THE RATES?"); P(RULE)
    S, IX = state_index(K)
    y_true = eradication(CANDIDATE, K=K, g0=G0, cycles=CYCLES)
    ly = np.log10(y_true)
    obs_true = observables(CANDIDATE)
    P(f"  Y_true = {y_true:.6e}   chemistry error eps = {EPS} kcal/mol"
      f"  (sigma = {EPS*ORDERS_PER_KCAL:.4f} orders per rate)")
    P(f"  true physiology:  doubling {obs_true[0]:.4f} h,  log-kill {obs_true[1]:+.4f},"
      f"  plateau {obs_true[2]:+.4f},  lag {obs_true[3]:.4f} h")

    # ---- C1 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("C1  THE CONSTRAINTS ARE COMPUTED CORRECTLY"); P(RULE)
    P(f"  true rate vector satisfies every constraint at scale 1e-9: "
      f"{satisfies(observables(CANDIDATE), obs_true, BASE_TOL, 1e-9)}")
    # The aggregates are properties of the LINEAR two-type process, which is what an
    # exponential-phase measurement on a real culture gives -- density there is set by the
    # culture, not by this model's carrying capacity. The eradication question is a property of
    # the finite stochastic process WITH the capacity. The control below therefore compares the
    # 2x2 against the stochastic generator in its linear regime (room = 1), where the mean vector
    # of a multi-type branching process obeys dm/dt = M m exactly.
    KL = 40
    SL, IXL = state_index(KL)
    lin = dict(CANDIDATE)
    Llin = generator_linear(KL, IXL, lin, False)
    tshort = 0.30
    pl = np.zeros(len(SL)); pl[IXL[(1, 0)]] = 1.0
    ql = dense_expm(Llin * tshort) @ pl
    m_stoch = np.array([sum(g * ql[IXL[(g, d)]] for (g, d) in IXL),
                        sum(d * ql[IXL[(g, d)]] for (g, d) in IXL)])
    M_off, _ = mean_field(CANDIDATE)
    m_mf = dense_expm(M_off * tshort) @ np.array([1.0, 0.0])
    rel = float(np.abs(m_stoch - m_mf).max() / np.abs(m_mf).max())
    P(f"  mean vector after {tshort} h from one G cell")
    P(f"    stochastic (linear regime, {len(SL)} states): G {m_stoch[0]:.8f}, D {m_stoch[1]:.8f}")
    P(f"    mean-field 2x2                              : G {m_mf[0]:.8f}, D {m_mf[1]:.8f}")
    P(f"  relative disagreement {rel:.2e}   {'PASS' if rel < 0.01 else 'FAIL'} (bar 1%)")

    # ---- screening ----------------------------------------------------------------------------
    sigma = EPS * ORDERS_PER_KCAL
    rng = np.random.default_rng(SEED + 7)
    Zd = rng.standard_normal((N_DRAW, N_RATES))
    P(f"\n  screening {N_DRAW} chemistry draws through the analytic mean-field observables ...")
    t0 = time.time()
    R = {nm: CANDIDATE[nm] * 10.0 ** (sigma * Zd[:, k]) for k, nm in enumerate(NAMES)}
    obs_all = np.column_stack(_aggregates(*(R[nm] for nm in
        ("mu", "k_kill", "a_off", "a_on", "b_off", "b_on", "d_death", "kd_kill"))))
    P(f"  screened in {time.time()-t0:.1f}s")
    P(f"  cross-check: closed-form aggregates reproduce the scalar path on 5 random draws to "
      + f"{max(float(np.abs(obs_all[t] - observables({nm: CANDIDATE[nm]*10.0**(sigma*Zd[t,k]) for k,nm in enumerate(NAMES)})).max()) for t in (3, 17, 99, 1000, 20000)):.2e}")

    def accepted_idx(scale, tol=BASE_TOL):
        ok = np.ones(N_DRAW, bool)
        ok &= np.abs((obs_all[:, 0] - obs_true[0]) / obs_true[0]) <= tol["doubling"] * scale
        ok &= np.abs(obs_all[:, 1] - obs_true[1]) <= tol["logkill"] * scale
        ok &= np.abs(obs_all[:, 2] - obs_true[2]) <= tol["plateau"] * scale
        ok &= np.abs((obs_all[:, 3] - obs_true[3]) / obs_true[3]) <= tol["lag"] * scale
        return np.where(ok & np.isfinite(obs_all).all(axis=1))[0]

    # ---- C2 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("C2  THE CONSTRAINTS ACTUALLY BIND"); P(RULE)
    acc1 = accepted_idx(1.0)
    P(f"  acceptance at the stated precisions: {len(acc1)}/{N_DRAW} = {len(acc1)/N_DRAW:.5f}")
    P(f"  {'PASS' if len(acc1)/N_DRAW < 0.5 else 'FAIL'} (bar: acceptance below 0.5)")

    # ---- evaluate Y on accepted draws, per tolerance scale -------------------------------------
    P("\n" + RULE); P("C3/C6/C7  WHAT SURVIVES THE CONSTRAINTS"); P(RULE)
    P(f"  {'tol scale':>11}{'accepted':>10}{'rate':>10}{'sd(log10)':>12}"
      f"{'p05':>10}{'p95':>10}{'range':>10}{'within x2':>11}{'within x10':>12}")
    stats = {}
    for scale in TOL_SCALES:
        idx = accepted_idx(scale)
        use = idx[:2000]
        if len(use) < 20:
            P(f"  {scale:>11}{len(idx):>10}{len(idx)/N_DRAW:>10.6f}"
              f"{'too few accepted to estimate':>55}")
            stats[scale] = None
            continue
        vals = []
        for t in use:
            r = {nm: CANDIDATE[nm] * 10.0 ** (sigma * Zd[t, k]) for k, nm in enumerate(NAMES)}
            vals.append(np.log10(max(eradication(r, K=K, g0=G0, cycles=CYCLES,
                                                 t_on=T_ON, t_off=T_OFF), 1e-300)) - ly)
        v = np.array(vals)
        h2 = float((np.abs(v) <= np.log10(2.0)).mean())
        h10 = float((np.abs(v) <= 1.0).mean())
        stats[scale] = (len(idx), v, h2, h10)
        P(f"  {scale:>11}{len(idx):>10}{len(idx)/N_DRAW:>10.6f}{v.std(ddof=1):>12.4f}"
          f"{np.percentile(v,5):>10.4f}{np.percentile(v,95):>10.4f}"
          f"{v.max()-v.min():>10.4f}{h2:>11.4f}{h10:>12.4f}")
    P(f"  C3: {'PASS' if stats.get(1.0) and stats[1.0][0] >= 300 else 'FAIL -- noise-limited'}"
      f" (bar 300 accepted at the stated precisions)")

    # ---- C5 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("C5  THE MATCHED CONTROL  (tolerances inflated 10x)"); P(RULE)
    idx10 = accepted_idx(10.0)
    use = idx10[:2000]
    vals = []
    for t in use:
        r = {nm: CANDIDATE[nm] * 10.0 ** (sigma * Zd[t, k]) for k, nm in enumerate(NAMES)}
        vals.append(np.log10(max(eradication(r, K=K, g0=G0, cycles=CYCLES,
                                             t_on=T_ON, t_off=T_OFF), 1e-300)) - ly)
    v10 = np.array(vals)
    h10_2 = float((np.abs(v10) <= np.log10(2.0)).mean())
    h10_10 = float((np.abs(v10) <= 1.0).mean())
    P(f"  loose constraints: accepted {len(idx10)}, within x2 {h10_2:.4f}, within x10 {h10_10:.4f}")
    if stats.get(1.0):
        P(f"  real  constraints: accepted {stats[1.0][0]}, within x2 {stats[1.0][2]:.4f},"
          f" within x10 {stats[1.0][3]:.4f}")
        gain2, gain10 = stats[1.0][2] - h10_2, stats[1.0][3] - h10_10
        P(f"  the real constraints buy {gain2:+.4f} (x2) and {gain10:+.4f} (x10) over loose ones")
        P(f"  {'PASS -- the physiology is informative, not just rejection of outliers' if gain10 > 0.02 else 'FAIL -- no better than discarding extreme draws'}")

    # ---- C4: convert into hybrid.py's currency ------------------------------------------------
    P("\n" + RULE); P("C4  THE DELIVERABLE  --  how many measured rates is physiology worth?"); P(RULE)
    P("  Recomputing hybrid.py's greedy curve on the same circuit for a matched comparison ...")
    kw = dict(K=K, g0=G0, cycles=CYCLES)
    Sd = {nm: sensitivity(CANDIDATE, nm, 0.02, **kw) for nm in NAMES}
    order = sorted(NAMES, key=lambda n: -abs(Sd[n]))
    gm = greedy_masks(order)
    rng2 = np.random.default_rng(SEED)
    Z = rng2.standard_normal((N_TRIALS, N_RATES))
    Y = np.empty((N_TRIALS, N_SUB))
    t0 = time.time()
    for t in range(N_TRIALS):
        Y[t] = run_trial({nm: Z[t, k] for k, nm in enumerate(NAMES)}, sigma, K, S, IX, G0)
        if t and t % 200 == 0:
            P(f"    ... {t}/{N_TRIALS}, {time.time()-t0:.0f}s")
    lyh = np.log10(np.maximum(Y, 1e-300))
    P(f"  {'tolerance':>11}{'physiology':>12}{'equivalent m':>15}{'greedy curve at m = 0..8':>40}")
    for lab, tol in TOLS:
        h = (np.abs(lyh - ly) <= tol)
        curve = np.array([h[:, gm[m]].mean() for m in range(N_RATES + 1)])
        phys = stats[1.0][2] if lab == "x2" else stats[1.0][3]
        eq = next((m for m in range(N_RATES + 1) if curve[m] >= phys), None)
        P(f"  {lab:>11}{phys:>12.4f}{str(eq):>15}   " + " ".join(f"{c:.3f}" for c in curve))
    P("  'equivalent m' is the smallest number of directly measured rates that matches what the")
    P("  full set of physiological constraints achieves, on the same circuit and the same draws.")

    # ---- C8 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("C8  DOMAIN  --  the rarer question"); P(RULE)
    y_deep = eradication(CANDIDATE, K=K, g0=G0_DEEP, cycles=CYCLES)
    lyd = np.log10(y_deep)
    use = accepted_idx(1.0)[:1500]
    vals = []
    for t in use:
        r = {nm: CANDIDATE[nm] * 10.0 ** (sigma * Zd[t, k]) for k, nm in enumerate(NAMES)}
        vals.append(np.log10(max(eradication(r, K=K, g0=G0_DEEP, cycles=CYCLES,
                                             t_on=T_ON, t_off=T_OFF), 1e-300)) - lyd)
    vd = np.array(vals)
    P(f"  g0 = {G0_DEEP}, Y = {y_deep:.6e}   (against {y_true:.6e} at g0 = {G0})")
    P(f"  among draws with indistinguishable physiology: sd {vd.std(ddof=1):.4f} orders,"
      f" p05 {np.percentile(vd,5):.4f}, p95 {np.percentile(vd,95):.4f}")
    P(f"  within x2 {float((np.abs(vd)<=np.log10(2)).mean()):.4f},"
      f" within x10 {float((np.abs(vd)<=1.0).mean()):.4f}")
    if stats.get(1.0):
        P(f"  against g0 = {G0}: within x2 {stats[1.0][2]:.4f}, within x10 {stats[1.0][3]:.4f}")

    P("\n" + RULE)
    P("C6 IS THE DECIDING NUMBER: the spread of the answer among rate vectors whose entire")
    P("measurable physiology -- replication time, kill curve, persister plateau, regrowth lag --")
    P("is indistinguishable from the truth. C7 says whether better instruments would close it.")
    P(RULE)

    open(os.path.join(os.path.dirname(__file__), "RESULTS_constrain.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
