"""Hide the rates, let "chemistry" supply them, and see how close the answer gets.

THE TEST. Take a circuit whose rates are known. Hide them. Replace each with what a free-energy
calculation would have returned -- i.e. the true value perturbed by that method's known error --
and ask how far the rare-event answer moves. This is the hide-the-rate experiment, executed the
only honest way available here: no free-energy calculation is run, so the chemistry step is
represented by its MEASURED ERROR DISTRIBUTION rather than pretended.

WHAT IS BEING FALSIFIED, and it is my own claim. rem/atlas/RESULTS_rateneed.txt used LINEAR
sensitivities added in quadrature to predict that at 1 kcal/mol the answer would be uncertain by

    3.891 orders  (a factor of 7,785)

That prediction is on the record before this module was written. It is linear and it assumes
independent errors. This Monte-Carlo is fully nonlinear and can therefore refute it. If the
measured spread comes out far smaller, the sensitivity analysis was wrong and the chemistry route
is in better shape than I said.

THE CONVERSION, unchanged: k ~ exp(-dG/RT), so at 298 K an error of eps kcal/mol in a barrier is
eps * 0.7330 orders in the rate. A draw is therefore log10(k_hat) = log10(k_true) + N(0, sigma)
with sigma = eps * 0.7330.

WHAT THIS DELIBERATELY DOES NOT MODEL. Force-field error is partly SYSTEMATIC -- a functional that
over-stabilises a charged transition state does so for every reaction of that class -- so real
errors are correlated across rates, not independent. Independent draws are the GENEROUS case.
H5 tests a correlated variant so the size of that generosity is visible rather than assumed.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN. Deciding statistic: worst case over the sweep.
=================================================================================================

H1  THE FALSIFICATION TEST. At 1 kcal/mol the standard deviation of log10(Y_hat / Y_true) must be
    compared against the predicted 3.891 orders. Predeclared readings: agreement within 30%
    confirms the linear analysis; a measured spread below 1.0 order refutes it and means the
    chemistry route is viable where I said it was not. Anything between is reported as measured.

H2  NON-VACUITY, AND SATURATION IS ITSELF A RESULT. Y_hat is a probability, so it cannot move
    below 0 or above 1; a spread of several orders will therefore be CLIPPED and the measured
    standard deviation will understate the true uncertainty. Report the fraction of draws landing
    within 1e-12 of 0 or within 1e-3 of 1. If that fraction is large, the honest statement is not
    "the error is N orders" but "the answer is uninformative", and it is reported that way.

H3  WHICH RATE DOES THE DAMAGE. Hide ONE rate at a time, all others exact. This directly tests
    the claim in RESULTS_rateneed.txt that chemistry would supply good numbers for the rates that
    do not matter: the per-rate spreads must rank in the same order as the measured |S|.

H4  ZERO-ERROR CONTROL. At 0 kcal/mol every draw must return Y_true exactly, to < 1e-12 relative.
    If not, the harness is measuring its own noise.

H5  THE GENEROSITY OF INDEPENDENCE. Repeat with a single shared bias applied to all four rates
    (fully correlated error, the systematic-force-field case). Report both spreads. Predeclared:
    correlated error is expected to be WORSE for a quantity that depends on rate ratios, and the
    comparison is reported whichever way it falls.

H6  THE DELIVERABLE. The fraction of hidden-rate trials landing within a factor of 2, 10 and 100
    of the true answer, at 0.5, 1 and 2 kcal/mol. That is the number a biologist would ask for.
"""

from __future__ import annotations
import numpy as np

from rem.atlas.rateneed import eradication, ORDERS_PER_KCAL, BASE, CYCLES, G0

RULE = "=" * 97
NAMES = ("mu", "k_kill", "a", "b")
PREDICTED_ORDERS = 3.891          # from RESULTS_rateneed.txt, on the record before this ran
# COST CORRECTION. 4000 trials per level meant ~26,000 eradication() calls, each doing two expm
# on 225x225, which I estimated at 40 ms and which actually runs nearer 90-130 ms wall even at
# 395% CPU. That run was heading past its 50-minute ceiling and would have produced nothing.
# 800 trials still pins a standard deviation to about 2.5% relative, which is far finer than any
# gate here needs, and BLAS is pinned to one thread because 225x225 matrices oversubscribe.
N_TRIALS = 800


def draw(kcal, rng, names=NAMES, correlated=False):
    sigma = kcal * ORDERS_PER_KCAL
    r = dict(BASE)
    if correlated:
        z = rng.normal()
        for nm in names:
            r[nm] = BASE[nm] * 10.0 ** (sigma * z)
    else:
        for nm in names:
            r[nm] = BASE[nm] * 10.0 ** (sigma * rng.normal())
    return r


def trial_set(kcal, n, seed, names=NAMES, correlated=False):
    rng = np.random.default_rng(seed)
    y0 = eradication(BASE, cycles=CYCLES, g0=G0)
    ys = np.array([eradication(draw(kcal, rng, names, correlated), cycles=CYCLES, g0=G0)
                   for _ in range(n)])
    return y0, ys


def summarise(y0, ys):
    safe = np.clip(ys, 1e-300, 1.0)
    lr = np.log10(safe / y0)
    return dict(sd=float(np.std(lr)), med=float(np.median(lr)),
                p05=float(np.percentile(lr, 5)), p95=float(np.percentile(lr, 95)),
                frac0=float(np.mean(ys < 1e-12)), frac1=float(np.mean(ys > 1 - 1e-3)),
                w2=float(np.mean(np.abs(lr) < np.log10(2))),
                w10=float(np.mean(np.abs(lr) < 1.0)),
                w100=float(np.mean(np.abs(lr) < 2.0)))


def report():
    out = []; P = out.append
    y0 = eradication(BASE, cycles=CYCLES, g0=G0)
    P(RULE)
    P("HIDE THE RATES, LET CHEMISTRY SUPPLY THEM, AND SEE HOW CLOSE THE ANSWER GETS")
    P(RULE)
    P(f"  True answer Y = P(eradication) = {y0:.6e}")
    P(f"  Chemistry represented by its measured error: eps kcal/mol -> eps * {ORDERS_PER_KCAL:.4f}")
    P(f"  orders of log-rate error, drawn independently per rate. {N_TRIALS} trials per level.")
    P(f"  PREDICTION ON THE RECORD (linear sensitivities in quadrature): {PREDICTED_ORDERS} orders")
    P("  at 1 kcal/mol. This Monte-Carlo is nonlinear and can refute it.")
    P("")

    P(RULE)
    P("H4  ZERO-ERROR CONTROL")
    P(RULE)
    _, yz = trial_set(0.0, 50, 1)
    dev = float(np.max(np.abs(yz / y0 - 1.0)))
    P(f"  worst relative deviation at 0 kcal/mol: {dev:.3e}   "
      f"{'PASS' if dev < 1e-12 else 'FAIL'} (bar 1e-12)")
    P("")

    P(RULE)
    P("H1 / H2  THE SPREAD, AND WHETHER IT IS EVEN MEASURABLE")
    P(RULE)
    P(f"  {'kcal/mol':>9s} {'sd (orders)':>12s} {'median':>9s} {'p05':>9s} {'p95':>9s}"
      f" {'frac at 0':>10s} {'frac at 1':>10s}")
    res = {}
    for kc in (0.5, 1.0, 2.0):
        s = summarise(*trial_set(kc, N_TRIALS, 100 + int(kc * 10)))
        res[kc] = s
        P(f"  {kc:9.1f} {s['sd']:12.3f} {s['med']:+9.3f} {s['p05']:+9.3f} {s['p95']:+9.3f}"
          f" {s['frac0']:10.4f} {s['frac1']:10.4f}")
    s1 = res[1.0]
    rel = abs(s1["sd"] - PREDICTED_ORDERS) / PREDICTED_ORDERS
    P("")
    P(f"  H1 at 1 kcal/mol: measured sd {s1['sd']:.3f} orders vs predicted "
      f"{PREDICTED_ORDERS} -> relative gap {rel:.3f}")
    if rel < 0.30:
        P("     CONFIRMS the linear analysis within 30%.")
    elif s1["sd"] < 1.0:
        P("     REFUTES it: the spread is under 1 order, so the chemistry route is viable where")
        P("     the sensitivity analysis said it was not. The linear prediction was wrong.")
    else:
        P("     Between the predeclared readings; reported as measured.")
    clipped = max(s["frac0"] + s["frac1"] for s in res.values())
    P(f"  H2 worst fraction of draws pinned at 0 or 1: {clipped:.4f}")
    if clipped > 0.05:
        P("     A material fraction is CLIPPED, so the measured sd UNDERSTATES the true spread.")
        P("     The honest statement is not 'the error is N orders' but that the answer is")
        P("     uninformative over much of the draw range.")
    P("")

    P(RULE)
    P("H3  WHICH RATE DOES THE DAMAGE  (hide one at a time, 1 kcal/mol, others exact)")
    P(RULE)
    S_meas = dict(mu=0.050745, k_kill=4.089134, a=3.370324, b=0.317206)   # RESULTS_rateneed.txt
    P(f"  {'rate':>8s} {'|S| measured':>13s} {'sd (orders)':>12s} {'within x10':>11s}")
    per = {}
    for nm in NAMES:
        s = summarise(*trial_set(1.0, N_TRIALS // 2, 200 + hash(nm) % 100, names=(nm,)))
        per[nm] = s["sd"]
        P(f"  {nm:>8s} {S_meas[nm]:13.3f} {s['sd']:12.3f} {s['w10']:11.4f}")
    order_S = [k for k, _ in sorted(S_meas.items(), key=lambda x: -x[1])]
    order_sd = [k for k, _ in sorted(per.items(), key=lambda x: -x[1])]
    P(f"  ranking by |S|:  {order_S}")
    P(f"  ranking by sd :  {order_sd}")
    P(f"  H3 {'PASS -- same order' if order_S == order_sd else 'FAIL -- rankings differ'}")
    P("")

    P(RULE)
    P("H5  IS INDEPENDENCE GENEROUS?  (single shared bias on all four rates)")
    P(RULE)
    P(f"  {'kcal/mol':>9s} {'independent sd':>15s} {'correlated sd':>15s} {'ratio':>8s}")
    for kc in (0.5, 1.0, 2.0):
        sc = summarise(*trial_set(kc, N_TRIALS // 2, 300 + int(kc * 10), correlated=True))
        si = res[kc]["sd"]
        P(f"  {kc:9.1f} {si:15.3f} {sc['sd']:15.3f} {sc['sd']/si:8.3f}")
    P("  A ratio above 1 means correlated force-field error is WORSE than independent draws, so")
    P("  the independent numbers above are the optimistic case.")
    P("")

    P(RULE)
    P("H6  THE DELIVERABLE -- how often does hiding the rates still give a usable answer?")
    P(RULE)
    P(f"  {'kcal/mol':>9s} {'within x2':>11s} {'within x10':>12s} {'within x100':>13s}")
    for kc in (0.5, 1.0, 2.0):
        s = res[kc]
        P(f"  {kc:9.1f} {s['w2']:11.4f} {s['w10']:12.4f} {s['w100']:13.4f}")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
