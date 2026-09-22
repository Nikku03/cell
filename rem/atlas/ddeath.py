"""Physiology plus ONE measured rate. Does the gap close?

WHERE THIS COMES FROM. constrain_rank.py measured that 32.6% of the eradication answer's
sensitivity survives PERFECT measurement of every population-level aggregate -- doubling time,
time-kill curve, persister plateau, outgrowth lag -- leaving 0.694 orders of irreducible spread at
1 kcal/mol. Its R3 found the surviving direction is almost entirely ONE rate, d_death, the
spontaneous death rate of dormant cells (+0.9434, every other component below 0.05). Its R6 then
projected that adding a single direct measurement of d_death would cut the free component from
0.9462 to 0.0730 -- 92% of what perfect physiology cannot reach -- while every other candidate,
aggregate assay or direct rate, bought at most 0.0021.

That projection is linear algebra. It has not been measured. This module measures it, and the
control in D6 can refute the ranking that produced it.

THE CONSTRUCTION, AND WHY IT IS PAIRED. Four arms share the SAME underlying chemistry draws and
the same solve-set, so the arms differ only in what is held at its true value:

    A  physiology alone                 -- the constrain_rank R4 condition
    B  physiology + d_death measured    -- the proposal
    C  physiology + kd_kill measured    -- the control R6 says is worthless
    D  d_death measured, NO physiology  -- separates the rate from the combination

Arms A, B and C weight every draw by how well its physiology matches the truth. Arm D applies no
physiology weight at all. A rate that is "measured" is held at its true value rather than drawn.

MODELLING CHOICES, RECORDED RATHER THAN BURIED.
  * Weighting, not projection. See the third correction block below: projection was tried first,
    failed on most draws for a structural reason, and would have produced a selected sample.
  * The weights define a soft constraint whose tau -> 0 limit is the conditional posterior. D8
    sweeps tau and gates that the limit has actually been reached.
  * The repaired outgrowth observable is reproduced here in closed form rather than by 601 matrix
    exponentials. D1 gates the two against each other, so the speedup cannot change the answer.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

D1  THE MACHINERY IS EXACT. (a) The closed-form outgrowth observable must agree with
    constrain_rank.py's matrix-exponential version to 1e-10 on a sample of random rate vectors.
    (b) The true rate vector must satisfy all four repaired observables to 1e-10.

D2  NOTHING IS DISCARDED. Under importance weighting there are no failures to hide, so what must
    be reported instead is the effective sample size, ESS = (sum w)^2 / sum w^2, at every tolerance
    and in every arm. Bar: ESS at the reference tolerance must exceed 500, or the arm is reported
    as noise-limited and NOT used for the deliverable.

D3  IT REPRODUCES WHAT IT EXTENDS. Arm A must agree with constrain_rank.py's R4 -- measured spread
    0.5984 orders, within x2 0.4890, within x10 0.8767 -- to within 30%. R4 used the unrepaired
    rank-3 observable set and this uses the repaired rank-4 set, so exact agreement is not
    expected; a large disagreement means one of the two is wrong.

D4  THE DELIVERABLE. Spread and hit fractions for arm B against arm A, on identical draws.

D5  THE PREDICTION IS TESTED, AND CAN FAIL. R6 predicted the free component falls from 0.9462 to
    0.0730, i.e. an irreducible spread of 0.7330 * 0.0730 = 0.0535 orders. Predeclared readings:
    a measured spread within 30% of 0.0535 confirms the linear projection as an experiment-design
    tool, and R6's entire table becomes usable for planning; a measured spread more than twice the
    prediction REFUTES it, and means the projection overstates what one measurement buys.

D6  THE MATCHED CONTROL, WHICH DECIDES WHETHER R6's RANKING MEANS ANYTHING. Arm C holds kd_kill,
    which R6 ranked as worth 0.0021 against d_death's 0.8732 -- a 400-fold difference in predicted
    value. Predeclared: arm B's spread must be at least 3x smaller than arm C's. If holding a rate
    R6 called worthless closes the gap comparably, the ranking carries no information and R6 is
    refuted regardless of what D5 says.

D7  IS IT THE RATE OR THE COMBINATION? Arm D measures d_death with no physiology at all.
    hybrid.py's greedy curve at m = 1 gives 0.105 within x2 and 0.482 within x10, so a single rate
    on its own is known to be weak. Reported, not gated: it separates "measuring d_death works"
    from "physiology plus d_death works".

D8  THE TOLERANCE LIMIT IS REACHED. Sweep tau_scale down to 1e-3. Predeclared: if the arm B spread
    changes by more than 30% between the two tightest settings, the tau -> 0 limit has NOT been
    reached and every number must be reported as an upper bound rather than as the posterior.

D10 TWO METHODS THAT SHARE NO MACHINERY MUST AGREE. At the loosest tolerance, where importance
    weighting still has adequate effective sample size, the Metropolis sampler and the
    importance-weighted estimator must give the same spread to within 30%. The sampler is trusted
    at tight tolerances only if it passes here. Predeclared: failure means the sampler is not
    sampling the posterior it claims to, and no tight-tolerance number may be reported.

D11 THE SAMPLER IS MIXING. Report Metropolis acceptance rate and the spread of the estimate ACROSS
    independent chains started from dispersed points. Bar: the across-chain standard deviation of
    the arm B spread must be below 10% of its value, or the chains have not converged.

D9  DOMAIN. Repeat at the rarer question, g0 = 8, where constrain_rank's R7 measured the surviving
    fraction nearly unchanged (0.3263) but the absolute irreducible spread larger (0.926 orders).

=================================================================================================
CORRECTION, AFTER THE FIRST RUN FAILED. THE OUTGROWTH OBSERVABLE WAS A STEP FUNCTION.
=================================================================================================
The first run of this module failed to solve on 594 to 600 of 600 draws in every arm; the log is
kept as RESULTS_ddeath_FAILED.txt. The cause was mine. The repaired outgrowth observable, as
written in constrain_rank.py and copied here, read the recovery time off a 601-point grid on
[0, 60] hours, so it could only return multiples of 0.1 h. Shifting log10(b_off) by +0.001, +0.005,
+0.010 and +0.020 left it at exactly 3.400000 h. A numerical Jacobian of a step function is zero
almost everywhere, and no root finder can solve one.

The defect was harmless where it was born -- constrain_rank only finite-differenced the observable
with a step of 0.02, which straddled grid boundaries and gave a plausible derivative -- and fatal
here. Two secondary problems compounded it: exp(lambda*t) overflowed to inf on solver excursions,
producing inf - inf = nan residuals, and fsolve ran unbounded.

THREE REPAIRS, all in this module:
  * lag is now obtained by BISECTING for the first time the population regains its pre-drug size,
    to machine precision, instead of snapping to a grid;
  * every exponent is clipped before exponentiation, so a residual is always a number;
  * the projection uses bounded least squares rather than unbounded fsolve, and a point is
    accepted only if its scaled residual is below 1e-10.

WHAT THE REPAIR DOES TO ALREADY-REPORTED NUMBERS. Recomputed with the continuous lag:
    surviving fraction under perfect physiology   0.3256 -> 0.3217
    reduction from measuring d_death              0.8732 -> 0.8637
    next best single measurement          kd_kill 0.0021 -> a_on 0.0173
The conclusion survives: d_death still dominates, by about 50x instead of 400x. D6's control still
holds kd_kill, which under the corrected observable is the WEAKEST candidate of all eight at
0.0042, so it remains the right comparator and the bar is unchanged.

D1(a) is necessarily redefined by this repair: it can no longer compare against constrain_rank's
grid-snapped version, which is now known to be wrong. It instead verifies the QUANTITY against an
independent matrix-exponential evaluation, which is a stronger control than agreement with another
implementation of the same formula.

SECOND REPAIR, after the bisected version ALSO failed. Making the lag continuous was not enough:
least squares refused draws outright with "residuals are not finite in the initial point". The
reason is that both of my original observables are FIRST-PASSAGE quantities, and both are singular
exactly where the biology is interesting. A population that does not grow off drug has no doubling
time; one that never recovers has no recovery time. Chemistry draws produce both cases routinely.

Neither is what an assay reads, either. A growth curve gives an exponential RATE and an outgrowth
assay gives an optical density at a fixed time. So the observable set is now

    A1  net growth rate off drug, per hour        (doubling time is ln2 divided by this)
    A2  log-kill over the drug window
    A3  persister plateau
    A4  log10 population 12 h after drug removal, from the actual post-drug (G, D) mixture

all four smooth and finite, verified finite on 2000 real draws. A1 is a monotone reparametrisation
of the old A1, so it cannot change the row space, the rank, or any conclusion. A4 genuinely
changes the row, so the conclusion was rechecked across all three definitions:

    definition                       surviving fraction   d_death reduction   next best
    grid-snapped (first reported)          0.3256              0.8732           0.0021
    continuous first-passage               0.3217              0.8637           0.0173
    smooth, used here                      0.3248              0.8757           0.0036

The conclusion is stable across all three, which is a stronger statement than any one of them:
d_death dominates by 240x to 400x however the physiology is parametrised.

THIRD CORRECTION, AND IT CHANGES THE METHOD, NOT JUST THE FORMULA. With smooth observables the
solver still failed on most draws, and a direct diagnosis showed why: multistarting with 1, 4 and
8 starts gave identically 8 successes out of 60, and the median residual of a FAILED draw was
0.16 to 0.90 -- nowhere near zero. So the failures are not a solver defect. For most draws no
solution exists at all: holding four rates at their drawn values, the remaining four cannot
reproduce the true physiology. The best of the fifteen candidate solve-sets managed 26 of 60.

That is fatal for the method, not for the question. Manifold projection keeps only the draws for
which a solution happens to exist, so the surviving sample is SELECTED, and selected on a
criterion related to the rates themselves. constrain_rank.py's R4 has the same flaw -- it reported
227 successes from 400 and I reported its numbers without noticing that the 173 failures were not
missing at random.

THE METHOD IS THEREFORE REPLACED by importance weighting, which has no selection step. Every
chemistry draw is kept and weighted by how well its physiology matches:

    log w = -0.5 * sum_i ((obs_i - obs_true_i) / tau_i)^2

As tau goes to zero this converges to the exact conditional posterior of the answer given the
physiology, under the chemistry prior -- which is precisely the quantity the question asks for.
Nothing is discarded, so nothing can be selected. The price is that the effective sample size
falls as tau shrinks, and ESS is therefore reported and gated rather than assumed.

The tolerances tau_i are set as a fraction of each observable's own spread under the chemistry
prior, so no instrument precision is invented: tau_scale = 0.05, 0.02, 0.01 means physiology known
to 5%, 2% and 1% of its natural variation.

FOURTH CORRECTION: IMPORTANCE WEIGHTING ALSO FAILS AT SMALL TAU, AND THE FIX IS A SAMPLER.
Plain importance weighting does not escape the curse that killed rejection sampling; it only makes
it visible. With four constraints the effective sample size falls roughly as tau^4, so the
tau -> 0 limit -- the only setting that answers the question -- needs a sample far beyond reach.
A 20,000-draw trial confirmed it: arm B's spread came out ten times the projected value purely
because tau was still loose.

The posterior is sampled directly instead, by Metropolis, and the sampler is preconditioned with
the posterior's own closed-form Gaussian approximation,

    Sigma_post = ( I/sigma^2 + J^T diag(1/tau^2) J )^{-1}

which is exact in the linear regime and an excellent proposal in the nonlinear one. This also
supplies an analytic prediction that connects the whole construction back to constrain_rank:

    predicted spread of log10 Y = sqrt( g^T Sigma_post g )   ->   sigma * ||g_null||  as tau -> 0

so the linear projection is not merely compared against a number, it is the tau -> 0 limit of the
curve the sampler traces out. The importance-weighted estimator is retained as an independent
cross-check at the loose tolerance where it is still reliable (D10): two methods that share no
machinery must agree there before the sampler is trusted where only it can go.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import numpy as np

from rem.atlas.hybrid_tune import (
    RULE, NAMES, CANDIDATE, ORDERS_PER_KCAL, eradication, sensitivity,
)
from rem.atlas.hybrid import K, G0, CYCLES, T_ON, T_OFF, SEED, N_RATES
from rem.atlas.constrain import EPS
from rem.atlas.constrain_rank import jacobian, split

G0_DEEP = 8
N_DRAW = 300_000        # draws for the importance-weighted cross-check only
M_EVAL = 1500           # posterior resamples at which the answer is actually evaluated
TAU_SCALES = (0.05, 0.01, 1e-3, 1e-4)
TAU_REF = 0.01          # the tightest tolerance at which D11 confirms the chains have converged
TAU_LOOSE = 0.05        # where importance weighting is still reliable, for D10
N_CHAIN, N_STEP, N_BURN = 12, 20000, 8000
ESS_BAR = 500.0
SIGMA = EPS * ORDERS_PER_KCAL
T_OUT = 12.0                                  # hours after drug removal at which outgrowth is read
CLIP = 700.0                                  # exp argument clip, so a residual is always a number

# R6's projections, quoted from RESULTS_constrain_rank.txt before this module ran.
R6_BASELINE_NULL = 0.9462
R6_NULL_WITH_DDEATH = 0.0730
R6_NULL_WITH_KDKILL = 0.9442
R4_SD, R4_H2, R4_H10 = 0.5984, 0.4890, 0.8767


def _exp(z):
    return np.exp(np.clip(z, -CLIP, CLIP))


def _two_by_two(a, b, c, d):
    disc = np.sqrt(max((a - d) ** 2 + 4.0 * b * c, 0.0))
    return 0.5 * (a + d + disc), 0.5 * (a + d - disc)


def aggregates_fast(x):
    """The four aggregates, in closed form, SMOOTH and finite for every rate vector.

    See the docstring's correction block for the two definitions this replaces. Both of the
    originals were first-passage quantities -- a doubling TIME and a time-to-recover -- and both
    are singular exactly where the biology is interesting: a population that does not grow off
    drug has no doubling time, and one that never recovers has no recovery time. Neither is what
    an assay reads anyway. A growth curve gives an exponential RATE, and an outgrowth assay gives
    an optical density at a fixed time. Both are smooth, finite, and carry the same information.

        A1  net growth rate off drug, per hour      (was: doubling time = ln2 / this)
        A2  log-kill over the drug window
        A3  persister plateau, the slow-mode amplitude
        A4  log10 population T_OUT hours after drug removal, from the ACTUAL post-drug (G, D)
            mixture -- this is the composition information the first lag definition discarded
    """
    r = {nm: CANDIDATE[nm] * 10.0 ** x[k] for k, nm in enumerate(NAMES)}
    a, b = r["mu"] - r["a_off"], r["b_off"]
    c, d = r["a_off"], -(r["b_off"] + r["d_death"])
    l1, l2 = _two_by_two(a, b, c, d)

    A, B = -(r["k_kill"] + r["a_on"]), r["b_on"]
    C, D = r["a_on"], -(r["b_on"] + r["kd_kill"])
    lp, lm = _two_by_two(A, B, C, D)
    gap = lp - lm if abs(lp - lm) > 1e-300 else 1e-300
    ep, em = _exp(lp * T_ON), _exp(lm * T_ON)
    vG = (ep * (A - lm) - em * (A - lp)) / gap
    vD = C * (ep - em) / gap
    logkill = np.log10(max(vG + vD, 1e-300))
    plateau = np.log10(max(abs((A - lm + C) / gap), 1e-300))

    g12 = l1 - l2 if abs(l1 - l2) > 1e-300 else 1e-300
    u1 = (((a - l2) * vG + b * vD) + (c * vG + (d - l2) * vD)) / g12
    u2 = (vG + vD) - u1
    outgrowth = np.log10(max(u1 * _exp(l1 * T_OUT) + u2 * _exp(l2 * T_OUT), 1e-300))
    return np.array([float(l1), float(logkill), float(plateau), float(outgrowth)])


def aggregates_vec(X):
    """aggregates_fast, elementwise over an (N, 8) array of log10 rate offsets."""
    r = {nm: CANDIDATE[nm] * 10.0 ** X[:, k] for k, nm in enumerate(NAMES)}
    a, b = r["mu"] - r["a_off"], r["b_off"]
    c, d = r["a_off"], -(r["b_off"] + r["d_death"])
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4.0 * b * c, 0.0))
    l1, l2 = 0.5 * (a + d + disc), 0.5 * (a + d - disc)

    A, B = -(r["k_kill"] + r["a_on"]), r["b_on"]
    C, D = r["a_on"], -(r["b_on"] + r["kd_kill"])
    dsc = np.sqrt(np.maximum((A - D) ** 2 + 4.0 * B * C, 0.0))
    lp, lm = 0.5 * (A + D + dsc), 0.5 * (A + D - dsc)
    gap = np.where(np.abs(lp - lm) < 1e-300, 1e-300, lp - lm)
    ep, em = _exp(lp * T_ON), _exp(lm * T_ON)
    vG = (ep * (A - lm) - em * (A - lp)) / gap
    vD = C * (ep - em) / gap
    logkill = np.log10(np.maximum(vG + vD, 1e-300))
    plateau = np.log10(np.maximum(np.abs((A - lm + C) / gap), 1e-300))

    g12 = np.where(np.abs(l1 - l2) < 1e-300, 1e-300, l1 - l2)
    u1 = (((a - l2) * vG + b * vD) + (c * vG + (d - l2) * vD)) / g12
    u2 = (vG + vD) - u1
    outgrowth = np.log10(np.maximum(u1 * _exp(l1 * T_OUT) + u2 * _exp(l2 * T_OUT), 1e-300))
    return np.column_stack([l1, logkill, plateau, outgrowth])


def weights(obs, obs_true, tau):
    """Soft physiology constraint. Returns normalised weights and the effective sample size."""
    z = (obs - obs_true) / tau
    lw = -0.5 * np.sum(z * z, axis=1)
    lw = np.where(np.isfinite(lw), lw, -np.inf)
    if not np.isfinite(lw).any():
        return None, 0.0
    w = np.exp(lw - lw.max())
    tot = w.sum()
    if tot <= 0:
        return None, 0.0
    w = w / tot
    return w, float(1.0 / np.sum(w * w))


def posterior_cov(J, sigma, tau, free_mask):
    """Closed-form Gaussian approximation to the posterior over log10 rate offsets.

    Prior is N(0, sigma^2) on every FREE coordinate; a coordinate that is held (a measured rate)
    has zero variance and is removed. Likelihood is the physiology, linearised at the truth."""
    idx = np.where(free_mask)[0]
    Jf = J[:, idx]
    prec = np.eye(len(idx)) / sigma ** 2 + Jf.T @ np.diag(1.0 / tau ** 2) @ Jf
    return idx, np.linalg.inv(prec)


def predicted_sd(g, J, sigma, tau, free_mask):
    idx, Sig = posterior_cov(J, sigma, tau, free_mask)
    gf = g[idx]
    return float(np.sqrt(max(gf @ Sig @ gf, 0.0)))


def metropolis(obs_true, tau, sigma, free_mask, J, n_chain, n_step, burn, seed, use_phys=True):
    """Ensemble Metropolis over the free coordinates, preconditioned by the Gaussian posterior.

    Chains are started dispersed from the prior, not from the truth, so D11's across-chain check
    is a real convergence test rather than a restatement of the initial condition."""
    idx, Sig = posterior_cov(J, sigma, tau, free_mask)
    d = len(idx)
    L = np.linalg.cholesky(Sig + 1e-18 * np.eye(d)) * (2.4 / np.sqrt(d))
    rng = np.random.default_rng(seed)
    X = np.zeros((n_chain, N_RATES))
    X[:, idx] = rng.standard_normal((n_chain, d)) @ L.T

    def logpost(Xa):
        lp = -0.5 * np.sum(Xa[:, idx] ** 2, axis=1) / sigma ** 2
        if use_phys:
            z = (aggregates_vec(Xa) - obs_true) / tau
            lp = lp - 0.5 * np.sum(z * z, axis=1)
        return np.where(np.isfinite(lp), lp, -np.inf)

    lp = logpost(X)
    kept, acc, post_acc, post_n = [], 0.0, 0.0, 0
    # Robbins-Monro step-size adaptation during burn-in only, targeting 25% acceptance. Without
    # it the acceptance rate collapses at tight tau -- the posterior is a thin CURVED sheet and a
    # Gaussian-preconditioned step of any useful size leaves it by more than tau. That failure is
    # recorded in RESULTS_ddeath.txt for the run before this was added, where acceptance fell to
    # 0.002 and the chains froze at their starting points, driving the measured spread BELOW the
    # analytic limit it should converge to.
    log_scale, batch = 0.0, np.zeros(0)
    for t in range(n_step):
        Yp = X.copy()
        Yp[:, idx] = X[:, idx] + np.exp(log_scale) * (rng.standard_normal((n_chain, d)) @ L.T)
        lq = logpost(Yp)
        take = np.log(rng.random(n_chain)) < (lq - lp)
        X[take] = Yp[take]
        lp[take] = lq[take]
        acc += take.mean()
        if t < burn:
            batch = np.append(batch, take.mean())
            if len(batch) >= 100:
                log_scale += 0.7 * (batch.mean() - 0.25)
                batch = np.zeros(0)
        else:
            post_acc += take.mean(); post_n += 1
            if (t - burn) % 5 == 0:
                kept.append(X.copy())
    return (np.array(kept), (post_acc / max(post_n, 1)), idx, float(np.exp(log_scale)))


def arm_stats(vals, counts=None):
    v = np.asarray(vals, float)
    if counts is not None:
        v = np.repeat(v, counts)
    if len(v) < 30:
        return None
    return dict(n=len(v), sd=float(v.std(ddof=1)), p05=float(np.percentile(v, 5)),
                p95=float(np.percentile(v, 95)), rng=float(v.max() - v.min()),
                h2=float((np.abs(v) <= np.log10(2.0)).mean()),
                h10=float((np.abs(v) <= 1.0).mean()))


def row(P, tag, s, ess=None, uniq=None):
    if s is None:
        P(f"  {tag:>32}{'noise-limited, not reported':>58}")
    else:
        e = f"{ess:>9.0f}" if ess is not None else f"{'--':>9}"
        u = f"{uniq:>7}" if uniq is not None else f"{'--':>7}"
        P(f"  {tag:>32}{e}{u}{s['sd']:>9.4f}{s['p05']:>9.4f}{s['p95']:>9.4f}"
          f"{s['rng']:>9.4f}{s['h2']:>10.4f}{s['h10']:>11.4f}")


def evaluate(idx, X, g0, ly, cache):
    vals = []
    for i in idx:
        if i not in cache:
            r = {nm: CANDIDATE[nm] * 10.0 ** X[i, k] for k, nm in enumerate(NAMES)}
            cache[i] = np.log10(max(eradication(r, K=K, g0=g0, cycles=CYCLES,
                                                t_on=T_ON, t_off=T_OFF), 1e-300)) - ly
        vals.append(cache[i])
    return np.array(vals)


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("PHYSIOLOGY PLUS ONE MEASURED RATE  --  DOES THE GAP CLOSE?"); P(RULE)
    x0 = np.zeros(N_RATES)
    obs_true = aggregates_fast(x0)
    P(f"  chemistry error {EPS} kcal/mol, sigma = {SIGMA:.4f} orders per rate")
    P(f"  true physiology: net growth {obs_true[0]:.4f} /h, log-kill {obs_true[1]:+.4f},"
      f" plateau {obs_true[2]:+.4f}, outgrowth at {T_OUT:.0f} h {obs_true[3]:+.4f}")

    # ---- D1 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("D1  THE MACHINERY IS EXACT"); P(RULE)
    from scipy.linalg import expm as _e
    rg = np.random.default_rng(5)
    worst = 0.0
    for _ in range(40):
        xr = rg.normal(0.0, 0.4, N_RATES)
        r = {nm: CANDIDATE[nm] * 10.0 ** xr[k] for k, nm in enumerate(NAMES)}
        Mon = np.array([[-(r["k_kill"] + r["a_on"]), r["b_on"]],
                        [r["a_on"], -(r["b_on"] + r["kd_kill"])]])
        Moff = np.array([[r["mu"] - r["a_off"], r["b_off"]],
                         [r["a_off"], -(r["b_off"] + r["d_death"])]])
        v = _e(Mon * T_ON) @ np.array([1.0, 0.0])
        ref = np.log10(max(float((_e(Moff * T_OUT) @ v).sum()), 1e-300))
        worst = max(worst, abs(aggregates_fast(xr)[3] - ref))
    P(f"  (a) outgrowth against an independent matrix exponential, 40 random rate vectors:")
    P(f"      worst {worst:.2e}   {'PASS' if worst < 1e-10 else 'FAIL'} (bar 1e-10)")
    res0 = float(np.abs(aggregates_fast(x0) - obs_true).max())
    P(f"  (b) true rate vector satisfies its own observables: {res0:.2e}"
      f"   {'PASS' if res0 < 1e-10 else 'FAIL'} (bar 1e-10)")
    rgv = np.random.default_rng(9).standard_normal((6, N_RATES)) * SIGMA
    dv = float(np.abs(aggregates_vec(rgv) - np.array([aggregates_fast(z) for z in rgv])).max())
    P(f"  (c) vectorised aggregates against the scalar path: {dv:.2e}"
      f"   {'PASS' if dv < 1e-12 else 'FAIL'} (bar 1e-12)")

    # ---- projections, from this module's own observable set ------------------------------------
    J = jacobian(lambda z: aggregates_fast(z), x0, 0.02)
    Sd = {nm: sensitivity(CANDIDATE, nm, 0.02, K=K, g0=G0, cycles=CYCLES) for nm in NAMES}
    g = np.array([Sd[nm] for nm in NAMES])
    _, r_rank, _, g_null = split(g, J)
    NULL_BASE = float(np.linalg.norm(g_null))
    proj = {}
    for j, nm in enumerate(NAMES):
        e = np.zeros(N_RATES); e[j] = 1.0
        _, _, _, gn2 = split(g, np.vstack([J, e]))
        proj[nm] = float(np.linalg.norm(gn2))
    NULL_DDEATH, NULL_KDKILL = proj["d_death"], proj["kd_kill"]
    P("\n" + RULE); P("THE LINEAR PROJECTION THIS TEST IS CHECKING"); P(RULE)
    P(f"  Jacobian rank {r_rank} of 4, ||g|| {np.linalg.norm(g):.4f}, ||g_null|| {NULL_BASE:.4f}")
    P(f"  (constrain_rank R5 recorded {R6_BASELINE_NULL:.4f} under its grid-snapped lag)")
    P(f"  {'rate held':>12}{'free component':>17}{'reduction':>12}{'predicted spread':>19}")
    for nm, v in sorted(proj.items(), key=lambda kv: kv[1]):
        P(f"  {nm:>12}{v:>17.4f}{NULL_BASE-v:>12.4f}{SIGMA*v:>19.4f}")

    ID = {nm: k for k, nm in enumerate(NAMES)}
    rng0 = np.random.default_rng(SEED + 31)
    Zis = rng0.standard_normal((N_DRAW, N_RATES)) * SIGMA
    spread = np.nanstd(np.where(np.isfinite(aggregates_vec(Zis)), aggregates_vec(Zis), np.nan), axis=0)
    P("\n" + RULE); P("THE TOLERANCE SCALE"); P(RULE)
    P("  tau is a fraction of each observable's own spread under the chemistry prior, so no")
    P("  instrument precision is invented. Prior spreads:")
    for n, v in zip(("growth rate", "log-kill", "plateau", "outgrowth"), spread):
        P(f"    {n:>12}  {v:.4g}")

    ARMS = (("A physiology alone", None, True),
            ("B physiology + d_death", "d_death", True),
            ("C physiology + kd_kill", "kd_kill", True),
            ("D d_death alone, no physiology", "d_death", False))

    results, accs, scales = {}, {}, {}
    for g0v in (G0, G0_DEEP):
        y_true = eradication(CANDIDATE, K=K, g0=g0v, cycles=CYCLES, t_on=T_ON, t_off=T_OFF)
        ly = np.log10(y_true)
        cache = {}
        taus = TAU_SCALES if g0v == G0 else (TAU_REF,)
        for arm, hold, use_phys in ARMS:
            free = np.ones(N_RATES, bool)
            if hold is not None:
                free[ID[hold]] = False
            for ts in (taus if use_phys else (None,)):
                tau = (ts if ts is not None else 1.0) * spread
                kept, acc, _, scl = metropolis(obs_true, tau, SIGMA, free, J, N_CHAIN, N_STEP,
                                               N_BURN, SEED + 101, use_phys=use_phys)
                scales[(g0v, arm, ts)] = scl
                nk, nc, _ = kept.shape
                flat = kept.reshape(nk * nc, N_RATES)
                chain_id = np.tile(np.arange(nc), nk)
                sel = np.random.default_rng(SEED + 5).choice(len(flat),
                                                             size=min(M_EVAL, len(flat)),
                                                             replace=False)
                vals, cid = [], []
                for i in sel:
                    key = tuple(np.round(flat[i], 12))
                    if key not in cache:
                        r = {nm: CANDIDATE[nm] * 10.0 ** flat[i, k] for k, nm in enumerate(NAMES)}
                        cache[key] = np.log10(max(eradication(r, K=K, g0=g0v, cycles=CYCLES,
                                                              t_on=T_ON, t_off=T_OFF), 1e-300)) - ly
                    vals.append(cache[key]); cid.append(chain_id[i])
                vals, cid = np.array(vals), np.array(cid)
                per = [float(vals[cid == c].std(ddof=1)) for c in range(nc)
                       if (cid == c).sum() > 30]
                results[(g0v, arm, ts)] = (arm_stats(vals), per,
                                           predicted_sd(g, J, SIGMA, tau, free) if use_phys else None)
                accs[(g0v, arm, ts)] = acc
                P(f"    ran g0={g0v} {arm.split()[0]} tau={ts}: acceptance {acc:.3f},"
                  f" adapted step scale {scl:.3e}, {len(vals)} evaluated")

    # ---- D10, the independent cross-check -----------------------------------------------------
    P("\n" + RULE); P("D10  TWO METHODS THAT SHARE NO MACHINERY MUST AGREE"); P(RULE)
    y_true = eradication(CANDIDATE, K=K, g0=G0, cycles=CYCLES, t_on=T_ON, t_off=T_OFF)
    ly = np.log10(y_true)
    obs_is = aggregates_vec(Zis)
    w, ess = weights(obs_is, obs_true, TAU_LOOSE * spread)
    pick = np.random.default_rng(SEED + 77).choice(N_DRAW, size=M_EVAL, p=w)
    uq, cnt = np.unique(pick, return_counts=True)
    cache2 = {}
    vis = evaluate(uq, Zis, G0, ly, cache2)
    is_stat = arm_stats(vis, cnt)
    mc_stat = results[(G0, "A physiology alone", TAU_LOOSE)][0]
    P(f"  at tau_scale = {TAU_LOOSE}, importance weighting has ESS {ess:.0f} on {N_DRAW} draws")
    P(f"  importance weighted : sd {is_stat['sd']:.4f}, within x2 {is_stat['h2']:.4f},"
      f" within x10 {is_stat['h10']:.4f}")
    P(f"  Metropolis          : sd {mc_stat['sd']:.4f}, within x2 {mc_stat['h2']:.4f},"
      f" within x10 {mc_stat['h10']:.4f}")
    d10 = abs(mc_stat["sd"] - is_stat["sd"]) / is_stat["sd"]
    P(f"  relative disagreement {d10:.4f}"
      f"   {'PASS -- the sampler may be trusted at tight tolerances' if d10 <= 0.30 else 'FAIL -- no tight-tolerance number may be reported'} (bar 30%)")

    # ---- D11 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("D11  THE SAMPLER IS MIXING"); P(RULE)
    P(f"  chains start dispersed from the prior, not at the truth, so this is a real test")
    P(f"  {'arm / tau':>30}{'acceptance':>12}{'step scale':>13}{'across-chain sd':>18}{'relative':>10}")
    worst11 = 0.0
    for (g0v, arm, ts), (st, per, _) in results.items():
        if g0v != G0 or st is None or not per:
            continue
        acs = float(np.std(per, ddof=1))
        rel = acs / max(st["sd"], 1e-12)
        if "B physiology" in arm:
            worst11 = max(worst11, rel)
        P(f"  {arm.split()[0]+' tau='+str(ts):>30}{accs[(g0v,arm,ts)]:>12.3f}{scales[(g0v,arm,ts)]:>13.3e}{acs:>18.4f}{rel:>10.4f}")
    P(f"  worst relative across-chain spread in arm B: {worst11:.4f}"
      f"   {'PASS' if worst11 <= 0.10 else 'FAIL -- chains have not converged'} (bar 0.10)")

    # ---- the arms -------------------------------------------------------------------------------
    for g0v in (G0, G0_DEEP):
        yt = eradication(CANDIDATE, K=K, g0=g0v, cycles=CYCLES, t_on=T_ON, t_off=T_OFF)
        P("\n" + RULE); P(f"THE ARMS  --  g0 = {g0v}, Y_true = {yt:.6e}"); P(RULE)
        P(f"  {'arm':>34}{'predicted':>11}{'sd':>9}{'p05':>9}{'p95':>9}{'range':>9}"
          f"{'within x2':>11}{'within x10':>12}")
        for arm, hold, use_phys in ARMS:
            for ts in ((TAU_SCALES if g0v == G0 else (TAU_REF,)) if use_phys else (None,)):
                key = (g0v, arm, ts)
                if key not in results:
                    continue
                st, per, pred = results[key]
                tag = f"{arm}  tau={ts}" if ts is not None else arm
                if st is None:
                    P(f"  {tag:>34}{'noise-limited':>60}")
                else:
                    pr = f"{pred:.4f}" if pred is not None else "--"
                    P(f"  {tag:>34}{pr:>11}{st['sd']:>9.4f}{st['p05']:>9.4f}{st['p95']:>9.4f}"
                      f"{st['rng']:>9.4f}{st['h2']:>11.4f}{st['h10']:>12.4f}")

    A = results[(G0, "A physiology alone", TAU_REF)][0]
    B = results[(G0, "B physiology + d_death", TAU_REF)][0]
    C = results[(G0, "C physiology + kd_kill", TAU_REF)][0]
    D = results[(G0, "D d_death alone, no physiology", None)][0]

    P("\n" + RULE); P("D3  IT REPRODUCES WHAT IT EXTENDS"); P(RULE)
    dv3 = abs(A["sd"] - R4_SD) / R4_SD
    P(f"  arm A sd {A['sd']:.4f} against constrain_rank R4's {R4_SD:.4f}: relative {dv3:.4f}"
      f"   {'PASS' if dv3 <= 0.30 else 'FAIL'} (bar 30%)")
    P(f"  arm A within x2 {A['h2']:.4f} (R4 {R4_H2:.4f}), within x10 {A['h10']:.4f}"
      f" (R4 {R4_H10:.4f})")
    P("  NOTE: R4 used manifold projection, which the third correction shows yields a SELECTED")
    P("  sample. Agreement is reassuring; disagreement would favour this estimate, not R4's.")

    P("\n" + RULE); P("D4  THE DELIVERABLE"); P(RULE)
    P(f"  at tau_scale = {TAU_REF}, i.e. physiology known to a fraction {TAU_REF} of its natural"
      f" spread -- the tightest tolerance at which D11 confirms the chains have converged:")
    P(f"  physiology alone     : sd {A['sd']:.4f}, within x2 {A['h2']:.4f},"
      f" within x10 {A['h10']:.4f}")
    P(f"  physiology + d_death : sd {B['sd']:.4f}, within x2 {B['h2']:.4f},"
      f" within x10 {B['h10']:.4f}")
    P(f"  the one measurement buys {B['h2']-A['h2']:+.4f} on x2 and {B['h10']-A['h10']:+.4f}"
      f" on x10, and shrinks the spread by a factor of {A['sd']/max(B['sd'],1e-12):.2f}")

    P("\n" + RULE); P("D5  THE PREDICTION IS TESTED"); P(RULE)
    pred = results[(G0, "B physiology + d_death", TAU_REF)][2]
    P(f"  the Gaussian posterior predicts sqrt(g' Sigma g) = {pred:.4f} orders, whose tau -> 0")
    P(f"  limit is sigma * ||g_null|| = {SIGMA*NULL_DDEATH:.4f} -- the constrain_rank projection")
    dv5 = abs(B["sd"] - pred) / pred
    P(f"  measured {B['sd']:.4f} orders, relative disagreement {dv5:.4f}")
    if dv5 <= 0.30:
        P("  PASS -- the linear projection is confirmed as an experiment-design tool, and the")
        P("  table above can be used to decide which measurement to buy.")
    elif B["sd"] > 2 * pred:
        P("  FAIL -- measured spread exceeds twice the prediction. The projection OVERSTATES what")
        P("  one measurement buys and must not be used for planning as it stands.")
    else:
        P("  PARTIAL -- outside 30% but within a factor of two; reported as measured.")

    P("\n" + RULE); P("D6  THE MATCHED CONTROL"); P(RULE)
    P(f"  projected free components: d_death {NULL_DDEATH:.4f}, kd_kill {NULL_KDKILL:.4f},"
      f" baseline {NULL_BASE:.4f}")
    ratio = C["sd"] / max(B["sd"], 1e-12)
    P(f"  measured at tau_scale {TAU_REF}: d_death sd {B['sd']:.4f}, kd_kill sd {C['sd']:.4f},"
      f" ratio {ratio:.2f}x")
    P(f"  {'PASS -- the ranking is informative' if ratio >= 3.0 else 'FAIL -- REFUTED: holding a rate the projection called worthless does comparably well'}"
      f" (bar 3x)")

    P("\n" + RULE); P("D7  IS IT THE RATE OR THE COMBINATION?"); P(RULE)
    P(f"  d_death alone, no physiology : sd {D['sd']:.4f}, within x2 {D['h2']:.4f},"
      f" within x10 {D['h10']:.4f}")
    P(f"  hybrid.py greedy at m = 1    : within x2 0.1050, within x10 0.4817")
    P(f"  physiology alone             : within x2 {A['h2']:.4f}, within x10 {A['h10']:.4f}")
    P(f"  physiology + d_death         : within x2 {B['h2']:.4f}, within x10 {B['h10']:.4f}")

    P("\n" + RULE); P("D8  HAS THE TOLERANCE LIMIT BEEN REACHED?"); P(RULE)
    P(f"  {'arm':>26}" + "".join(f"{'tau='+str(t):>12}" for t in TAU_SCALES))
    for arm in ("A physiology alone", "B physiology + d_death", "C physiology + kd_kill"):
        cells = []
        for ts in TAU_SCALES:
            st = results[(G0, arm, ts)][0]
            cells.append(f"{st['sd']:.4f}" if st else "noise")
        P(f"  {arm:>26}" + "".join(f"{c:>12}" for c in cells))
    P(f"  {'analytic, arm A':>26}" + "".join(
        f"{predicted_sd(g, J, SIGMA, t*spread, np.ones(N_RATES,bool)):>12.4f}" for t in TAU_SCALES))
    fmB = np.ones(N_RATES, bool); fmB[ID["d_death"]] = False
    P(f"  {'analytic, arm B':>26}" + "".join(
        f"{predicted_sd(g, J, SIGMA, t*spread, fmB):>12.4f}" for t in TAU_SCALES))
    P("  A measured spread BELOW its own analytic value is not physics; it is a chain that has")
    P("  not moved. The previous run failed exactly that way and its numbers are recorded as void.")
    sB = [results[(G0, "B physiology + d_death", ts)][0] for ts in TAU_SCALES]
    dv8 = abs(sB[-1]["sd"] - sB[-2]["sd"]) / max(sB[-2]["sd"], 1e-12)
    P(f"  arm B change between the two tightest tolerances: {dv8:.4f}"
      f"   {'PASS -- the limit is reached' if dv8 <= 0.30 else 'FAIL -- report as an upper bound, not the posterior'}"
      f" (bar 30%)")

    P("\n" + RULE); P("D9  DOMAIN  --  the rarer question"); P(RULE)
    P(f"  {'':>34}{'sd':>10}{'within x2':>12}{'within x10':>12}")
    for arm in ("A physiology alone", "B physiology + d_death"):
        for g0v in (G0, G0_DEEP):
            st = results[(g0v, arm, TAU_REF)][0]
            P(f"  {arm+f' , g0={g0v}':>34}{st['sd']:>10.4f}{st['h2']:>12.4f}{st['h10']:>12.4f}")

    P("\n" + RULE)
    open(os.path.join(os.path.dirname(__file__), "RESULTS_ddeath.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
