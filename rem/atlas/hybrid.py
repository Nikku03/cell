"""How many rates must actually be MEASURED, if chemistry supplies the rest?

WHERE THIS COMES FROM. The hide-the-rate test (hiderate.py) hid all four rates of a persister
circuit behind chemistry's own accuracy and found that at 1 kcal/mol only 20.4% of trials landed
within a factor of ten of the true eradication probability. But it also found, in gate H3, that
ranking rates by a cheap linear sensitivity and ranking them by the damage they actually do give
the SAME order. The ranking works even where the error bar does not.

That suggests a hybrid rather than a verdict: spend a small experimental budget measuring the
few rates the ranking flags, and let chemistry supply the rest, where H3 says it costs little.
The question this module answers is the only one that matters for planning an experiment --

    how many of the rates do we actually need to measure?

THE CIRCUIT AND THE BASE POINT are chosen in hybrid_tune.py, which was committed and run first,
and whose output is RESULTS_hybrid_tune.txt. Eight distinct physical processes, a carrying
capacity so the state space is exactly closed, K = 20, g0 = 6, three on/off cycles, Y = 1.7237e-02.
That script also records a CORRECTION: its first version used unbounded birth truncated at a cap,
which failed its own convergence criterion and, worse, flipped 2-3% of factor-of-2 verdicts
between caps. The fix was to the model, not to the bar.

THE METHOD. For a subset M of rates declared MEASURED, those take their true values and the rest
are drawn as log10(k_hat) = log10(k_true) + N(0, sigma), sigma = eps * 0.7328 orders, which is
transition-state theory's conversion of an eps kcal/mol barrier error into a rate error. Every
subset sees the SAME draws -- common random numbers -- so a subset-versus-subset comparison is
paired and its standard error is far below the unpaired one. All 2^8 = 256 subsets are enumerated
exhaustively, which is affordable only because the eight rates split 4/4 by phase with no rate in
both, so the drug-on and drug-off propagators depend on disjoint halves of the rate vector and
16 + 16 = 32 matrix exponentials per trial cover the entire lattice instead of 512.

THE LINEAR PREDICTION, ALREADY ON THE RECORD in RESULTS_hybrid_tune.txt before this ran. Measuring
the top m by |S| leaves residual spread sd(m) = sigma * sqrt(sum of S^2 over the unmeasured rates);
a two-sided 90% band needs 1.645*sd(m) <= delta. That predicts

    eps = 0.5 kcal/mol :  m* = 4 for a factor of 2,   m* = 2 for a factor of 10
    eps = 1.0 kcal/mol :  m* = 4 for a factor of 2,   m* = 4 for a factor of 10
    eps = 2.0 kcal/mol :  m* = 6 for a factor of 2,   m* = 4 for a factor of 10

This Monte-Carlo is nonlinear, clipped at 0 and 1, and free to refute all six numbers.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN. Deciding statistic: worst case over the sweep.
=================================================================================================

H1  THE STATE SPACE IS CLOSED, ON PERTURBED DRAWS AND NOT JUST AT THE BASE POINT. Propagated mass
    must stay at 1 to within 1e-12 with no renormalisation anywhere, for a sample of actual
    chemistry draws. The base point proving conservative is worth nothing; the draws are where the
    previous model leaked.

H2  ZERO-ERROR CONTROL. The subset with all eight rates measured must reproduce Y_true on every
    trial: worst |log10(Y_hat/Y_true)| < 1e-12. If the "measure everything" corner is not exact,
    nothing else in the table means anything.

H3  THE FACTORISATION IS EXACT, NOT MERELY FAST. The 32-exponential evaluator must agree with the
    naive per-subset evaluator in hybrid_tune.py to 1e-10 relative on a random sample of
    (trial, subset) pairs. A 16x speedup that changes the answer is worth nothing.

H4  NON-VACUITY. hit(m = 0) within a factor of 10 must be below 0.6, and hit(m = 8) must be 1.0,
    or the curve has no room to show a rise and the deliverable is meaningless.

H5  MONOTONICITY. The greedy hit fraction must not DECREASE in m beyond the paired band. Measuring
    one more rate cannot make the answer worse; a real decrease is a harness bug, not biology.

H6  IS GREEDY ACTUALLY OPTIMAL? This is the falsification test, and it can refute the hybrid's
    whole premise. For every m, compare the greedy-by-|S| subset against the BEST of all subsets
    of size m, on identical draws. Predeclared readings: a shortfall within the paired band at
    every m confirms |S|-ranking as a subset-selection rule and means an experimentalist can pick
    which rates to measure from cheap derivatives alone; a shortfall beyond the band at any m
    means greedy is NOT optimal, and the size of the shortfall is reported as measured. The count
    of m at which greedy is the STRICT argmax is reported separately, since ties are expected --
    hybrid_tune flagged d_death vs kd_kill at a ratio of 1.034 and b_on vs mu at 1.084.

H7  THE MATCHED CONTROL. Greedy is compared against the MEAN over all subsets of the same size,
    which is exactly the expected performance of choosing m rates at random. Two conditions:
    (a) greedy is never worse beyond the band, for 1 <= m <= 7; (b) greedy beats random by at
    least 0.15 somewhere. Gating the worst case over ALL m would be unreachable on any evidence,
    because at m = 0 and m = 8 there is one subset and the difference is identically zero. That
    would be the fourth unreachable bar in this build order, and it is avoided by construction
    rather than discovered after the fact.

H8  THE DELIVERABLE. m*(tolerance, eps), the smallest m whose hit fraction reaches 0.90, for
    tolerances of a factor of 2 and a factor of 10, at 0.5, 1 and 2 kcal/mol. Reported with a
    binomial band, as a fraction of the eight rates.

H9  SATURATION CHECK. At the reported m*, fewer than 1% of trials may be pinned at Y_hat = 0 or 1.
    A saturated observable has now cost this build order four reruns; the deliverable does not get
    to be a fifth.

H10 CAN THE CHEAP FORMULA PLAN THE EXPERIMENT WITHOUT THIS MONTE-CARLO? Compare measured m*
    against the six predictions above. Reported, not gated. If the formula is right, the whole
    exhaustive enumeration becomes unnecessary for the next circuit, which is the practical point.

H11 DEPTH DEPENDENCE, AND THE DOMAIN OF THE ANSWER. Repeat at g0 = 8, where Y = 4.476e-03 and the
    question is rarer. rateneed's N6 already showed sensitivity grows with tail depth, so m* is
    expected to grow. If it does, no single number for "how many rates" may be quoted without the
    question it was measured on, and that qualification is part of the deliverable.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import time
import numpy as np
from scipy.linalg import expm

from rem.atlas.hybrid_tune import (
    RULE, OFF_RATES, ON_RATES, NAMES, CANDIDATE, ORDERS_PER_KCAL,
    state_index, generator, eradication, sensitivity,
)

K, G0, CYCLES, T_ON, T_OFF = 20, 6, 3, 6.0, 3.0
G0_DEEP = 8
N_TRIALS = 600
EPS_LEVELS = (0.5, 1.0, 2.0)
TOLS = (("x2", np.log10(2.0)), ("x10", 1.0))
TARGET = 0.90
SEED = 20260905

# Predictions from RESULTS_hybrid_tune.txt, written down before this module ran.
LINEAR_PREDICTION = {(0.5, "x2"): 4, (0.5, "x10"): 2,
                     (1.0, "x2"): 4, (1.0, "x10"): 4,
                     (2.0, "x2"): 6, (2.0, "x10"): 4}

N_OFF, N_ON = len(OFF_RATES), len(ON_RATES)
N_RATES = len(NAMES)
N_SUB = 2 ** N_RATES


def masked_rates(group, mask, z, sigma):
    """Rates for one phase. Bit k set in `mask` means rate k is MEASURED, so it keeps its true
    value; an unset bit means chemistry supplied it and it carries the drawn log-error."""
    out = {}
    for k, nm in enumerate(group):
        out[nm] = CANDIDATE[nm] if (mask >> k) & 1 else CANDIDATE[nm] * 10.0 ** (sigma * z[nm])
    return out


def propagators(z, sigma, K_, IX):
    """The 32 matrix exponentials that cover all 256 subsets, by the 4/4 phase split."""
    base = dict(CANDIDATE)
    A_off, A_on = [], []
    for mask in range(2 ** N_OFF):
        r = dict(base); r.update(masked_rates(OFF_RATES, mask, z, sigma))
        A_off.append(expm(generator(K_, IX, r, False) * T_OFF))
    for mask in range(2 ** N_ON):
        r = dict(base); r.update(masked_rates(ON_RATES, mask, z, sigma))
        A_on.append(expm(generator(K_, IX, r, True) * T_ON))
    return A_off, A_on


def run_trial(z, sigma, K_, S, IX, g0, want_mass=False):
    """Return Y for all 256 subsets on one draw. Subset index is off_mask*16 + on_mask."""
    A_off, A_on = propagators(z, sigma, K_, IX)
    n = len(S)
    P = np.zeros((n, 2 ** N_OFF, 2 ** N_ON))
    P[IX[(min(g0, K_), 0)], :, :] = 1.0
    for _ in range(CYCLES):
        for j in range(2 ** N_ON):
            P[:, :, j] = A_on[j] @ P[:, :, j]
        for i in range(2 ** N_OFF):
            P[:, i, :] = A_off[i] @ P[:, i, :]
    Y = P[IX[(0, 0)], :, :].reshape(-1).copy()
    if want_mass:
        return Y, P.sum(axis=0).reshape(-1).copy()
    return Y


def subset_sizes():
    pc = lambda x: bin(x).count("1")
    return np.array([pc(i) + pc(j) for i in range(2 ** N_OFF) for j in range(2 ** N_ON)])


def greedy_masks(order):
    """Subset index of the top-m rates by |S|, for each m."""
    out = []
    for m in range(N_RATES + 1):
        chosen = set(order[:m])
        i = sum(1 << k for k, nm in enumerate(OFF_RATES) if nm in chosen)
        j = sum(1 << k for k, nm in enumerate(ON_RATES) if nm in chosen)
        out.append(i * (2 ** N_ON) + j)
    return out


def sweep(eps, g0, S, IX, Z, y_true, P):
    """All 256 subsets x N_TRIALS at one error level. Returns hit indicators and saturation."""
    sigma = eps * ORDERS_PER_KCAL
    ly = np.log10(y_true)
    Y = np.empty((N_TRIALS, N_SUB))
    worst_mass = 0.0
    t0 = time.time()
    for t in range(N_TRIALS):
        z = {nm: Z[t, k] for k, nm in enumerate(NAMES)}
        if t % 50 == 0:
            Yt, mass = run_trial(z, sigma, K, S, IX, g0, want_mass=True)
            worst_mass = max(worst_mass, float(np.abs(mass - 1.0).max()))
        else:
            Yt = run_trial(z, sigma, K, S, IX, g0)
        Y[t] = Yt
        if t and t % 100 == 0:
            P(f"    ... eps={eps} g0={g0}: {t}/{N_TRIALS} trials, {time.time()-t0:.0f}s")
    lyh = np.log10(np.maximum(Y, 1e-300))
    hits = {lab: (np.abs(lyh - ly) <= tol) for lab, tol in TOLS}
    pinned = (Y <= 1e-300) | (Y >= 1.0 - 1e-12)
    return Y, hits, pinned, worst_mass


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE)
    P("HOW MANY RATES MUST BE MEASURED, IF CHEMISTRY SUPPLIES THE REST?")
    P(RULE)
    S, IX = state_index(K)
    y_true = eradication(CANDIDATE, K=K, g0=G0, cycles=CYCLES)
    P(f"  circuit: 8 rates, carrying capacity K = {K} ({len(S)} states), g0 = {G0},"
      f" {CYCLES} on/off cycles")
    P(f"  true answer Y = P(eradication) = {y_true:.6e}, upward headroom"
      f" {np.log10(1.0/y_true):.3f} orders")
    P(f"  {N_SUB} subsets x {N_TRIALS} trials at each of {len(EPS_LEVELS)} error levels,"
      f" common random numbers")

    kw = dict(K=K, g0=G0, cycles=CYCLES)
    Sd = {nm: sensitivity(CANDIDATE, nm, 0.02, **kw) for nm in NAMES}
    order = sorted(NAMES, key=lambda n: -abs(Sd[n]))
    gm = greedy_masks(order)
    sizes = subset_sizes()
    P(f"  greedy order by |S|: {order}")

    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal((N_TRIALS, N_RATES))

    # ---- H3, before anything expensive depends on the fast path being right -----------------
    P("\n" + RULE); P("H3  THE FACTORISATION IS EXACT, NOT MERELY FAST"); P(RULE)
    chk = np.random.default_rng(11).integers(0, N_SUB, 8)
    worst_fac = 0.0
    for t in (0, 1, 2):
        z = {nm: Z[t, k] for k, nm in enumerate(NAMES)}
        Yt = run_trial(z, 1.0 * ORDERS_PER_KCAL, K, S, IX, G0)
        for s in chk:
            i, j = divmod(int(s), 2 ** N_ON)
            r = dict(CANDIDATE)
            r.update(masked_rates(OFF_RATES, i, z, 1.0 * ORDERS_PER_KCAL))
            r.update(masked_rates(ON_RATES, j, z, 1.0 * ORDERS_PER_KCAL))
            naive = eradication(r, K=K, g0=G0, cycles=CYCLES, t_on=T_ON, t_off=T_OFF)
            worst_fac = max(worst_fac, abs(Yt[s] - naive) / max(naive, 1e-300))
    P(f"  worst relative disagreement over 24 (trial, subset) pairs: {worst_fac:.2e}"
      f"   {'PASS' if worst_fac < 1e-10 else 'FAIL'} (bar 1e-10)")

    results, mass_worst = {}, 0.0
    for eps in EPS_LEVELS:
        P(f"\n  running eps = {eps} kcal/mol ...")
        Y, hits, pinned, wm = sweep(eps, G0, S, IX, Z, y_true, P)
        results[eps] = (Y, hits, pinned)
        mass_worst = max(mass_worst, wm)

    # ---- H1, H2 -----------------------------------------------------------------------------
    P("\n" + RULE); P("H1  THE STATE SPACE IS CLOSED, ON PERTURBED DRAWS"); P(RULE)
    P(f"  worst |propagated mass - 1| over sampled draws, no renormalisation: {mass_worst:.2e}"
      f"   {'PASS' if mass_worst < 1e-12 else 'FAIL'} (bar 1e-12)")

    P("\n" + RULE); P("H2  ZERO-ERROR CONTROL  (the all-measured corner must be exact)"); P(RULE)
    full = gm[N_RATES]
    worst_zero = max(float(np.abs(np.log10(results[e][0][:, full] / y_true)).max()) for e in EPS_LEVELS)
    P(f"  worst |log10(Y_hat/Y_true)| at m = 8: {worst_zero:.2e}"
      f"   {'PASS' if worst_zero < 1e-12 else 'FAIL'} (bar 1e-12)")

    # ---- H4, H5, H6, H7 ---------------------------------------------------------------------
    band = {}
    for lab, _ in TOLS:
        P("\n" + RULE)
        P(f"H4/H5/H6/H7  THE CURVE, ITS OPTIMALITY AND ITS CONTROL  --  tolerance {lab}")
        P(RULE)
        for eps in EPS_LEVELS:
            Y, hits, pinned = results[eps]
            H = hits[lab]
            g = np.array([H[:, gm[m]].mean() for m in range(N_RATES + 1)])
            bestv, bestm, randv, short, se = [], [], [], [], []
            for m in range(N_RATES + 1):
                cols = np.where(sizes == m)[0]
                fr = H[:, cols].mean(axis=0)
                b = int(cols[int(np.argmax(fr))])
                bestv.append(float(fr.max())); bestm.append(b)
                randv.append(float(fr.mean()))
                d = H[:, b].astype(float) - H[:, gm[m]].astype(float)
                se.append(float(d.std(ddof=1) / np.sqrt(N_TRIALS)) if d.std() > 0 else 0.0)
                short.append(float(fr.max() - g[m]))
            band[(eps, lab)] = (g, np.array(bestv), np.array(randv), np.array(short), np.array(se), bestm)
            P(f"\n  eps = {eps} kcal/mol")
            P(f"  {'m':>3}{'greedy':>9}{'best':>9}{'shortfall':>11}{'paired se':>11}"
              f"{'random':>9}{'greedy-rand':>13}{'best subset is greedy?':>24}")
            for m in range(N_RATES + 1):
                cols = np.where(sizes == m)[0]
                P(f"  {m:>3}{g[m]:>9.4f}{bestv[m]:>9.4f}{short[m]:>11.4f}{se[m]:>11.4f}"
                  f"{randv[m]:>9.4f}{g[m]-randv[m]:>13.4f}"
                  f"{('yes' if bestm[m]==gm[m] else 'NO ('+str(len(cols))+' subsets)'):>24}")

    P("\n" + RULE); P("H4  NON-VACUITY"); P(RULE)
    h0 = band[(1.0, "x10")][0][0]; h8 = band[(1.0, "x10")][0][N_RATES]
    P(f"  hit(m=0) within x10 at 1 kcal/mol = {h0:.4f}   (bar < 0.6)")
    P(f"  hit(m=8) within x10 at 1 kcal/mol = {h8:.4f}   (bar = 1.0)")
    P(f"  {'PASS' if h0 < 0.6 and h8 == 1.0 else 'FAIL'}")

    P("\n" + RULE); P("H5  MONOTONICITY  (measuring more cannot hurt)"); P(RULE)
    worst_drop = 0.0
    for key, (g, _, _, _, se, _) in band.items():
        for m in range(N_RATES):
            worst_drop = max(worst_drop, float(g[m] - g[m + 1]))
    P(f"  worst decrease in the greedy curve over all levels and tolerances: {worst_drop:.4f}")
    P(f"  {'PASS' if worst_drop <= 0.03 else 'FAIL'} (bar 0.03, about 2 paired standard errors)")

    P("\n" + RULE); P("H6  IS GREEDY OPTIMAL?  (the falsification test)"); P(RULE)
    worst_short, worst_key, strict = 0.0, None, 0
    total = 0
    for key, (g, bv, rv, sh, se, bm) in band.items():
        for m in range(N_RATES + 1):
            total += 1
            if bm[m] == gm[m]:
                strict += 1
            if sh[m] > worst_short:
                worst_short, worst_key = float(sh[m]), (key, m)
    P(f"  worst shortfall (best subset minus greedy) over all m, tolerances and error levels:"
      f" {worst_short:.4f}")
    P(f"  attained at eps = {worst_key[0][0]}, tolerance {worst_key[0][1]}, m = {worst_key[1]}")
    P(f"  {'PASS' if worst_short <= 0.03 else 'FAIL'} (bar 0.03)")
    P(f"  greedy was the STRICT argmax in {strict} of {total} (m, tolerance, eps) cells")
    P("  READING: a shortfall inside the band means |S|-ranking picks a subset no worse than the")
    P("  best available one, so cheap derivatives are enough to choose which rates to measure.")

    P("\n" + RULE); P("H7  THE MATCHED CONTROL  (greedy versus m rates chosen at random)"); P(RULE)
    worst_neg, best_gain = 0.0, 0.0
    for key, (g, bv, rv, sh, se, bm) in band.items():
        for m in range(1, N_RATES):
            worst_neg = min(worst_neg, float(g[m] - rv[m]))
            best_gain = max(best_gain, float(g[m] - rv[m]))
    P(f"  (a) worst (greedy - random) for 1 <= m <= 7: {worst_neg:+.4f}"
      f"   {'PASS' if worst_neg >= -0.03 else 'FAIL'} (bar -0.03)")
    P(f"  (b) largest (greedy - random) anywhere:      {best_gain:+.4f}"
      f"   {'PASS' if best_gain >= 0.15 else 'FAIL'} (bar 0.15)")

    # ---- H8, H9, H10 ------------------------------------------------------------------------
    P("\n" + RULE); P("H8  THE DELIVERABLE  --  how many of the 8 rates must be measured?"); P(RULE)
    P(f"  m* = the smallest m whose greedy hit fraction reaches {TARGET:.2f}")
    P(f"  {'eps kcal/mol':>14}{'tolerance':>12}{'m*':>6}{'of':>4}{'hit at m*':>12}"
      f"{'binomial se':>13}{'hit at m*-1':>13}")
    mstar = {}
    for eps in EPS_LEVELS:
        for lab, _ in TOLS:
            g = band[(eps, lab)][0]
            ms = next((m for m in range(N_RATES + 1) if g[m] >= TARGET), None)
            mstar[(eps, lab)] = ms
            if ms is None:
                P(f"  {eps:>14}{lab:>12}{'none':>6}{N_RATES:>4}{'--':>12}{'--':>13}{'--':>13}")
            else:
                se = np.sqrt(g[ms] * (1 - g[ms]) / N_TRIALS)
                prev = f"{g[ms-1]:.4f}" if ms > 0 else "--"
                P(f"  {eps:>14}{lab:>12}{ms:>6}{N_RATES:>4}{g[ms]:>12.4f}{se:>13.4f}{prev:>13}")

    P("\n" + RULE); P("H9  SATURATION CHECK AT THE REPORTED m*"); P(RULE)
    worst_pin = 0.0
    P(f"  {'eps':>6}{'tolerance':>12}{'m*':>6}{'fraction pinned at 0 or 1':>28}")
    for (eps, lab), ms in mstar.items():
        if ms is None:
            continue
        pin = float(results[eps][2][:, gm[ms]].mean())
        worst_pin = max(worst_pin, pin)
        P(f"  {eps:>6}{lab:>12}{ms:>6}{pin:>28.4f}")
    P(f"  worst {worst_pin:.4f}   {'PASS' if worst_pin < 0.01 else 'FAIL'} (bar 0.01)")

    P("\n" + RULE); P("H10  CAN THE CHEAP FORMULA PLAN THE EXPERIMENT WITHOUT THIS?"); P(RULE)
    P(f"  {'eps':>6}{'tolerance':>12}{'predicted m*':>15}{'measured m*':>14}{'agree?':>9}")
    agree = 0
    for eps in EPS_LEVELS:
        for lab, _ in TOLS:
            pr, me = LINEAR_PREDICTION[(eps, lab)], mstar[(eps, lab)]
            ok = pr == me
            agree += ok
            P(f"  {eps:>6}{lab:>12}{pr:>15}{str(me):>14}{('yes' if ok else 'NO'):>9}")
    P(f"  {agree} of 6 predictions exact.")

    # ---- H11 --------------------------------------------------------------------------------
    P("\n" + RULE); P("H11  DEPTH DEPENDENCE  --  m* is a property of the QUESTION"); P(RULE)
    y_deep = eradication(CANDIDATE, K=K, g0=G0_DEEP, cycles=CYCLES)
    kwd = dict(K=K, g0=G0_DEEP, cycles=CYCLES)
    Sdeep = {nm: sensitivity(CANDIDATE, nm, 0.02, **kwd) for nm in NAMES}
    order_d = sorted(NAMES, key=lambda n: -abs(Sdeep[n]))
    gm_d = greedy_masks(order_d)
    P(f"  rarer question: g0 = {G0_DEEP}, Y = {y_deep:.6e}"
      f"  (against {y_true:.6e} at g0 = {G0})")
    P(f"  greedy order there: {order_d}")
    P(f"  order unchanged from g0 = {G0}: {order_d == order}")
    P(f"\n  running the depth sweep at eps = 1.0 ...")
    Yd, hitsd, pind, _ = sweep(1.0, G0_DEEP, S, IX, Z, y_deep, P)
    P(f"\n  {'tolerance':>12}{'m* at g0=6':>13}{'m* at g0=8':>13}{'hit at that m*':>16}")
    for lab, _ in TOLS:
        gd = np.array([hitsd[lab][:, gm_d[m]].mean() for m in range(N_RATES + 1)])
        msd = next((m for m in range(N_RATES + 1) if gd[m] >= TARGET), None)
        P(f"  {lab:>12}{str(mstar[(1.0, lab)]):>13}{str(msd):>13}"
          f"{(f'{gd[msd]:.4f}' if msd is not None else '--'):>16}")
        P(f"               curve: " + " ".join(f"{x:.3f}" for x in gd))

    P("\n" + RULE)
    P("The number of rates that must be measured is not a property of the circuit alone.")
    P("It is a property of the circuit, the tolerance demanded, the accuracy chemistry brings,")
    P("and the rarity of the question. Every m* above carries all four.")
    P(RULE)

    open(os.path.join(os.path.dirname(__file__), "RESULTS_hybrid.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
