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

Arms A, B and C project each draw onto the exact constraint manifold by solving the four
observable equations for four rates. Arm D does no solving at all.

MODELLING CHOICES, RECORDED RATHER THAN BURIED.
  * The Newton solve starts from the TRUE rate vector, so it finds the manifold branch nearest the
    truth. A more distant branch would give MORE spread, so this choice is conservative -- it is
    generous to the proposal being tested.
  * A manifold sample is not the conditional posterior under the chemistry prior; it lacks the
    Jacobian volume factor, and it depends on which rates are solved and which are drawn. D8
    repeats the whole thing on a different solve-set for exactly this reason.
  * The repaired outgrowth observable is reproduced here in closed form rather than by 601 matrix
    exponentials. D1 gates the two against each other, so the speedup cannot change the answer.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

D1  THE MACHINERY IS EXACT. (a) The closed-form outgrowth observable must agree with
    constrain_rank.py's matrix-exponential version to 1e-10 on a sample of random rate vectors.
    (b) The true rate vector must satisfy all four repaired observables to 1e-10.

D2  SOLVER HONESTY. Report the number of Newton failures, and verify that every point ACCEPTED as
    on-manifold reproduces the four observables to better than 1e-8 relative. Failures are
    reported, never silently dropped -- R4 reported 173 of 400 and this must do the same.

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

D8  THE PARAMETRISATION IS NOT THE RESULT. Repeat every arm on a second, differently chosen
    solve-set. Predeclared: if the arm B spread moves by more than 30% between solve-sets, the
    number is a property of my manifold parametrisation rather than of the biology, and must be
    reported as such rather than as a measurement.

D9  DOMAIN. Repeat at the rarer question, g0 = 8, where constrain_rank's R7 measured the surviving
    fraction nearly unchanged (0.3263) but the absolute irreducible spread larger (0.926 orders).
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import numpy as np
from scipy.optimize import fsolve

from rem.atlas.hybrid_tune import (
    RULE, NAMES, CANDIDATE, ORDERS_PER_KCAL, eradication, sensitivity,
)
from rem.atlas.hybrid import K, G0, CYCLES, T_ON, T_OFF, SEED, N_RATES
from rem.atlas.constrain import EPS
from rem.atlas.constrain_rank import aggregates_from_log, jacobian, split

G0_DEEP = 8
N_MANIFOLD = 600
SIGMA = EPS * ORDERS_PER_KCAL
LAG_GRID = np.linspace(0.0, 60.0, 601)      # identical to constrain_rank's repaired definition

# R6's projections, quoted from RESULTS_constrain_rank.txt before this module ran.
R6_BASELINE_NULL = 0.9462
R6_NULL_WITH_DDEATH = 0.0730
R6_NULL_WITH_KDKILL = 0.9442
R4_SD, R4_H2, R4_H10 = 0.5984, 0.4890, 0.8767


def _two_by_two(a, b, c, d):
    disc = np.sqrt(max((a - d) ** 2 + 4.0 * b * c, 0.0))
    return 0.5 * (a + d + disc), 0.5 * (a + d - disc)


def aggregates_fast(x):
    """The four repaired observables in closed form. Same lag rule as constrain_rank: the first
    point of a 601-point grid on [0, 60] at which the population regains its pre-drug size."""
    r = {nm: CANDIDATE[nm] * 10.0 ** x[k] for k, nm in enumerate(NAMES)}
    # drug off
    a, b = r["mu"] - r["a_off"], r["b_off"]
    c, d = r["a_off"], -(r["b_off"] + r["d_death"])
    l1, l2 = _two_by_two(a, b, c, d)
    doubling = np.log(2.0) / l1 if l1 > 1e-12 else np.inf
    # drug on
    A, B = -(r["k_kill"] + r["a_on"]), r["b_on"]
    C, D = r["a_on"], -(r["b_on"] + r["kd_kill"])
    lp, lm = _two_by_two(A, B, C, D)
    gap = lp - lm if abs(lp - lm) > 1e-300 else 1e-300
    ep, em = np.exp(lp * T_ON), np.exp(lm * T_ON)
    vG = (ep * (A - lm) - em * (A - lp)) / gap
    vD = C * (ep - em) / gap
    total_on = max(vG + vD, 1e-300)
    logkill = np.log10(total_on)
    plateau = np.log10(max(abs((A - lm + C) / gap), 1e-300))
    # repaired outgrowth: propagate the ACTUAL (G, D) mixture through the off-phase mean field
    g12 = l1 - l2 if abs(l1 - l2) > 1e-300 else 1e-300
    u1 = (((a - l2) * vG + b * vD) + (c * vG + (d - l2) * vD)) / g12
    u2 = 1.0 * (vG + vD) - u1
    tot = u1 * np.exp(l1 * LAG_GRID) + u2 * np.exp(l2 * LAG_GRID)
    hit = np.where(tot >= 1.0)[0]
    lag = float(LAG_GRID[hit[0]]) if len(hit) else 60.0
    return np.array([float(doubling), float(logkill), float(plateau), lag])


def solve_manifold(x_free, free_idx, solve_idx, obs_true, x_hold=None, hold_idx=()):
    """Project one draw onto the exact constraint manifold by solving for `solve_idx`."""
    def resid(xs):
        x = np.zeros(N_RATES)
        x[list(free_idx)] = x_free
        for k, v in zip(hold_idx, x_hold or ()):
            x[k] = v
        x[list(solve_idx)] = xs
        return aggregates_fast(x) - obs_true

    xs, info, ier, _ = fsolve(resid, np.zeros(len(solve_idx)), full_output=True, xtol=1e-12)
    if ier != 1:
        return None, None
    x = np.zeros(N_RATES)
    x[list(free_idx)] = x_free
    for k, v in zip(hold_idx, x_hold or ()):
        x[k] = v
    x[list(solve_idx)] = xs
    rel = float(np.abs(aggregates_fast(x) - obs_true).max() / max(np.abs(obs_true).max(), 1e-300))
    return x, rel


def arm_stats(vals):
    v = np.array(vals)
    if len(v) < 10:
        return None
    return dict(n=len(v), sd=float(v.std(ddof=1)), p05=float(np.percentile(v, 5)),
                p95=float(np.percentile(v, 95)), rng=float(v.max() - v.min()),
                h2=float((np.abs(v) <= np.log10(2.0)).mean()),
                h10=float((np.abs(v) <= 1.0).mean()))


def row(P, tag, s):
    if s is None:
        P(f"  {tag:>34}{'too few points':>50}")
    else:
        P(f"  {tag:>34}{s['n']:>7}{s['sd']:>10.4f}{s['p05']:>10.4f}{s['p95']:>10.4f}"
          f"{s['rng']:>10.4f}{s['h2']:>11.4f}{s['h10']:>12.4f}")


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("PHYSIOLOGY PLUS ONE MEASURED RATE  --  DOES THE GAP CLOSE?"); P(RULE)
    x0 = np.zeros(N_RATES)
    obs_true = aggregates_fast(x0)
    P(f"  chemistry error {EPS} kcal/mol, sigma = {SIGMA:.4f} orders per rate")
    P(f"  true physiology: doubling {obs_true[0]:.4f} h, log-kill {obs_true[1]:+.4f},"
      f" plateau {obs_true[2]:+.4f}, outgrowth {obs_true[3]:.4f} h")

    # ---- D1 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("D1  THE MACHINERY IS EXACT"); P(RULE)
    rg = np.random.default_rng(5)
    worst = 0.0
    for _ in range(12):
        xr = rg.normal(0.0, 0.5, N_RATES)
        worst = max(worst, float(np.abs(aggregates_fast(xr)
                                       - aggregates_from_log(xr, repaired=True)).max()))
    P(f"  (a) closed form vs matrix-exponential outgrowth, 12 random rate vectors: {worst:.2e}"
      f"   {'PASS' if worst < 1e-10 else 'FAIL'} (bar 1e-10)")
    res0 = float(np.abs(aggregates_fast(x0) - obs_true).max())
    P(f"  (b) true rate vector satisfies its own observables: {res0:.2e}"
      f"   {'PASS' if res0 < 1e-10 else 'FAIL'} (bar 1e-10)")

    # ---- pick solve-sets ----------------------------------------------------------------------
    J = jacobian(lambda z: aggregates_fast(z), x0, 0.02)
    ID = {nm: k for k, nm in enumerate(NAMES)}
    pool = [ID[n] for n in ("mu", "a_off", "b_off", "a_on", "b_on", "k_kill")]
    ranked = sorted(itertools.combinations(pool, 4),
                    key=lambda cb: -np.linalg.svd(J[:, list(cb)], compute_uv=False).min())
    P(f"\n  solve-set chosen by best-conditioned 4x4 Jacobian block, from the six rates that are")
    P(f"  neither d_death nor kd_kill so that both can be held in turn:")
    for lbl, cb in (("primary", ranked[0]), ("D8 alternative", ranked[1])):
        sv = np.linalg.svd(J[:, list(cb)], compute_uv=False)
        P(f"    {lbl:>16}: {[NAMES[k] for k in cb]}  smallest singular value {sv.min():.4e}")

    kw = dict(K=K, g0=G0, cycles=CYCLES, t_on=T_ON, t_off=T_OFF)
    Sd = {nm: sensitivity(CANDIDATE, nm, 0.02, K=K, g0=G0, cycles=CYCLES) for nm in NAMES}
    g = np.array([Sd[nm] for nm in NAMES])
    _, r_rank, _, g_null = split(g, J)
    P(f"\n  repaired Jacobian rank {r_rank}, ||g|| {np.linalg.norm(g):.4f},"
      f" ||g_null|| {np.linalg.norm(g_null):.4f}"
      f"  (constrain_rank R5 recorded {R6_BASELINE_NULL:.4f})")

    results = {}
    for set_lbl, solve_idx in (("primary", ranked[0]), ("alternative", ranked[1])):
        for g0v in (G0, G0_DEEP):
            y_true = eradication(CANDIDATE, K=K, g0=g0v, cycles=CYCLES, t_on=T_ON, t_off=T_OFF)
            ly = np.log10(y_true)
            drawn = [k for k in range(N_RATES) if k not in solve_idx]
            rng = np.random.default_rng(SEED + 31)
            Zd = rng.standard_normal((N_MANIFOLD, N_RATES))
            arms = {"A physiology alone": [], "B physiology + d_death": [],
                    "C physiology + kd_kill": [], "D d_death alone, no physiology": []}
            fails = {k: 0 for k in arms}
            worst_rel = 0.0
            for t in range(N_MANIFOLD):
                base = {k: SIGMA * Zd[t, k] for k in drawn}
                for arm, hold in (("A physiology alone", None),
                                  ("B physiology + d_death", ID["d_death"]),
                                  ("C physiology + kd_kill", ID["kd_kill"])):
                    if hold is None:
                        x, rel = solve_manifold([base[k] for k in drawn], drawn,
                                                solve_idx, obs_true)
                    else:
                        # The held rate is one of the drawn ones; pin it at its true value and
                        # solve over the rest. Every arm sees the same draws for the rates it
                        # does not hold, so the arms are paired.
                        free2 = [k for k in drawn if k != hold]
                        x, rel = solve_manifold([base[k] for k in free2], free2, solve_idx,
                                                obs_true, x_hold=(0.0,), hold_idx=(hold,))
                    if x is None:
                        fails[arm] += 1
                        continue
                    worst_rel = max(worst_rel, rel)
                    r = {nm: CANDIDATE[nm] * 10.0 ** x[k] for k, nm in enumerate(NAMES)}
                    arms[arm].append(np.log10(max(eradication(r, K=K, g0=g0v, cycles=CYCLES,
                                                              t_on=T_ON, t_off=T_OFF), 1e-300)) - ly)
                # arm D: no physiology, d_death held, everything else drawn
                xD = SIGMA * Zd[t].copy()
                xD[ID["d_death"]] = 0.0
                rD = {nm: CANDIDATE[nm] * 10.0 ** xD[k] for k, nm in enumerate(NAMES)}
                arms["D d_death alone, no physiology"].append(
                    np.log10(max(eradication(rD, K=K, g0=g0v, cycles=CYCLES,
                                             t_on=T_ON, t_off=T_OFF), 1e-300)) - ly)
            results[(set_lbl, g0v)] = ({k: arm_stats(v) for k, v in arms.items()},
                                       fails, worst_rel, y_true)

    # ---- D2 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("D2  SOLVER HONESTY"); P(RULE)
    wr = 0.0
    for (sl, gv), (st, fails, worst_rel, _) in results.items():
        wr = max(wr, worst_rel)
        P(f"  solve-set {sl}, g0={gv}: failures " +
          ", ".join(f"{k.split()[0]} {v}/{N_MANIFOLD}" for k, v in fails.items() if "D " not in k))
    P(f"  worst observable residual among ACCEPTED manifold points: {wr:.2e}"
      f"   {'PASS' if wr < 1e-8 else 'FAIL'} (bar 1e-8)")

    # ---- D3, D4, D6, D7 -------------------------------------------------------------------------
    for (sl, gv) in [("primary", G0), ("alternative", G0), ("primary", G0_DEEP),
                     ("alternative", G0_DEEP)]:
        st, fails, _, y_true = results[(sl, gv)]
        P("\n" + RULE)
        P(f"THE FOUR ARMS  --  solve-set {sl}, g0 = {gv}, Y_true = {y_true:.6e}")
        P(RULE)
        P(f"  {'arm':>34}{'n':>7}{'sd':>10}{'p05':>10}{'p95':>10}{'range':>10}"
          f"{'within x2':>11}{'within x10':>12}")
        for k in ("A physiology alone", "B physiology + d_death",
                  "C physiology + kd_kill", "D d_death alone, no physiology"):
            row(P, k, st[k])

    stp = results[("primary", G0)][0]
    A, B, C, D = (stp["A physiology alone"], stp["B physiology + d_death"],
                  stp["C physiology + kd_kill"], stp["D d_death alone, no physiology"])

    P("\n" + RULE); P("D3  IT REPRODUCES WHAT IT EXTENDS"); P(RULE)
    dv = abs(A["sd"] - R4_SD) / R4_SD
    P(f"  arm A sd {A['sd']:.4f} against constrain_rank R4's {R4_SD:.4f}: relative {dv:.4f}"
      f"   {'PASS' if dv <= 0.30 else 'FAIL'} (bar 30%)")
    P(f"  arm A within x2 {A['h2']:.4f} (R4 {R4_H2:.4f}), within x10 {A['h10']:.4f} (R4 {R4_H10:.4f})")

    P("\n" + RULE); P("D4  THE DELIVERABLE"); P(RULE)
    P(f"  physiology alone      : sd {A['sd']:.4f} orders, within x2 {A['h2']:.4f},"
      f" within x10 {A['h10']:.4f}")
    P(f"  physiology + d_death  : sd {B['sd']:.4f} orders, within x2 {B['h2']:.4f},"
      f" within x10 {B['h10']:.4f}")
    P(f"  the one measurement buys {B['h2']-A['h2']:+.4f} on x2 and {B['h10']-A['h10']:+.4f} on x10,")
    P(f"  and shrinks the spread by a factor of {A['sd']/max(B['sd'],1e-12):.2f}")

    P("\n" + RULE); P("D5  THE PREDICTION IS TESTED"); P(RULE)
    pred = SIGMA * R6_NULL_WITH_DDEATH
    dv5 = abs(B["sd"] - pred) / pred
    P(f"  R6 projected free component {R6_NULL_WITH_DDEATH:.4f}, i.e. spread"
      f" {SIGMA:.4f} * {R6_NULL_WITH_DDEATH:.4f} = {pred:.4f} orders")
    P(f"  measured {B['sd']:.4f} orders, relative disagreement {dv5:.4f}")
    if dv5 <= 0.30:
        P("  PASS -- the linear projection is confirmed as an experiment-design tool,")
        P("  and R6's table can be used to plan which measurement to buy.")
    elif B["sd"] > 2 * pred:
        P("  FAIL -- measured spread exceeds twice the prediction. The projection OVERSTATES")
        P("  what one measurement buys, and R6's table must not be used for planning as it stands.")
    else:
        P("  PARTIAL -- outside 30% but within a factor of two; reported as measured.")

    P("\n" + RULE); P("D6  THE MATCHED CONTROL  (holding a rate R6 called worthless)"); P(RULE)
    P(f"  R6 projected d_death worth a 0.8732 reduction and kd_kill worth 0.0021 -- 400-fold.")
    P(f"  measured: d_death sd {B['sd']:.4f}, kd_kill sd {C['sd']:.4f},"
      f" ratio {C['sd']/max(B['sd'],1e-12):.2f}x")
    P(f"  {'PASS -- the ranking is informative' if C['sd'] >= 3*B['sd'] else 'FAIL -- R6 REFUTED: holding a rate it called worthless does comparably well'}"
      f" (bar 3x)")

    P("\n" + RULE); P("D7  IS IT THE RATE OR THE COMBINATION?"); P(RULE)
    P(f"  d_death alone, no physiology : sd {D['sd']:.4f}, within x2 {D['h2']:.4f},"
      f" within x10 {D['h10']:.4f}")
    P(f"  hybrid.py's greedy m=1       : within x2 0.1050, within x10 0.4817")
    P(f"  physiology + d_death         : within x2 {B['h2']:.4f}, within x10 {B['h10']:.4f}")

    P("\n" + RULE); P("D8  THE PARAMETRISATION IS NOT THE RESULT"); P(RULE)
    Balt = results[("alternative", G0)][0]["B physiology + d_death"]
    dv8 = abs(Balt["sd"] - B["sd"]) / max(B["sd"], 1e-12)
    P(f"  arm B sd: primary solve-set {B['sd']:.4f}, alternative {Balt['sd']:.4f},"
      f" relative {dv8:.4f}")
    P(f"  {'PASS -- the number is not an artefact of the parametrisation' if dv8 <= 0.30 else 'FAIL -- the result depends on how the manifold is parametrised and must be reported as such'}"
      f" (bar 30%)")

    P("\n" + RULE); P("D9  DOMAIN  --  the rarer question"); P(RULE)
    sd8 = results[("primary", G0_DEEP)][0]
    P(f"  {'':>34}{'sd':>10}{'within x2':>12}{'within x10':>12}")
    for k in ("A physiology alone", "B physiology + d_death"):
        s6, s8 = stp[k], sd8[k]
        P(f"  {k+' , g0=6':>34}{s6['sd']:>10.4f}{s6['h2']:>12.4f}{s6['h10']:>12.4f}")
        P(f"  {k+' , g0=8':>34}{s8['sd']:>10.4f}{s8['h2']:>12.4f}{s8['h10']:>12.4f}")

    P("\n" + RULE)
    open(os.path.join(os.path.dirname(__file__), "RESULTS_ddeath.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
