"""How many rate directions does measured physiology actually pin, and what does it leave free?

WHY THIS EXISTS. constrain.py answered the question by rejection sampling and hit two limits that
are recorded in RESULTS_constrain.txt rather than hidden:

  (i)  Its gate C7 -- the idealised limit of PERFECT instruments, which is the sharpest form of
       the question -- could not be answered. Acceptance falls like (tolerance)^4, so tightening
       by 10x left 3 accepted draws out of 400,000 and by 100x left none. C7 is UNANSWERED, not
       passed, and no amount of extra sampling fixes an exponent.
  (ii) One of its four constraints turned out to carry no information. As defined there,
       regrowth lag = ln(1/total_on)/lambda_off, while log-kill already fixes total_on and
       doubling time already fixes lambda_off. Lag was algebraically determined by the other two,
       so that test had THREE independent constraints while claiming four. The error was mine and
       it was unfair to the proposal being tested, since a real outgrowth assay depends on the
       post-drug G/D composition, which the simplification discarded. R5 repairs it.

Both limits dissolve in linear algebra. Write the observables and the answer as functions of the
LOG rates. Then

    J = d(observables)/d(log10 k)      a q x n matrix
    g = d(log10 Y)/d(log10 k)          the answer's gradient, already measured

Perfect physiology pins exactly the row space of J and leaves its null space free. Decomposing
g = g_row + g_null, the part of the answer that survives perfect measurement of every aggregate is
g_null, and with chemistry error sigma on the free directions the irreducible uncertainty in the
answer is sigma * ||g_null||. No sampling, no tolerance, no acceptance rate -- and it is the exact
linear answer to "could better instruments fix this?".

The same decomposition turns the negative into a design tool: for any candidate NEW measurement,
appending its gradient as a row of J and recomputing ||g_null|| says exactly what that measurement
would buy. R6 uses it to compare one more aggregate assay against one more directly measured rate,
in the same currency.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

R1  THE JACOBIAN IS A JACOBIAN. Every entry is a central difference in log space; halving the step
    must change it by less than 1%, worst case over all entries, or what is reported is
    discretisation and not a derivative.

R2  HOW MANY INDEPENDENT CONSTRAINTS ARE THERE REALLY? Report the FULL singular-value spectrum of
    J, not just a rank with a threshold, so the cut is visible rather than asserted. Predeclared
    readings: a spectrum with a gap of many orders means the nominal count of measurements
    overstates the real one, and the specific dependency is identified by inspecting the null
    vector of J^T; a spectrum with no gap means all four are independent and the C7 defect above
    was mine alone.

R3  THE DELIVERABLE. ||g_null|| / ||g||: the fraction of the answer's sensitivity that perfect
    measurement of every aggregate cannot touch, and sigma * ||g_null|| in orders. Predeclared
    readings: below 0.1 means physiology essentially determines the answer and the route is open;
    above 0.5 means the aggregates pin directions the answer barely depends on; between is
    reported as measured.

R4  THE LINEAR PREDICTION IS CHECKED AGAINST AN EXACT MANIFOLD SAMPLE. Draw chemistry errors,
    project each draw onto the exact constraint manifold by solving the observable equations, and
    measure the spread of the answer over those exactly-physiology-matched rate vectors. This is
    C7's unanswerable gate, answered by construction instead of by rejection. The measured spread
    must agree with sigma * ||g_null|| to within 30%, or the linear decomposition does not
    describe the nonlinear manifold and R3 must be read as a local statement only.

R5  THE LAG DEFECT, REPAIRED. Redefine the outgrowth observable from the actual post-drug
    composition, which is what a regrowth curve reflects, and recompute the spectrum. Predeclared:
    if the rank rises, constrain.py's number was unfair to the proposal and BOTH are reported; if
    it does not rise, the dependency is structural and not an artefact of my simplification.

R6  WHAT SHOULD BE MEASURED NEXT? For each candidate additional observable -- further aggregate
    assays, and directly measured single rates -- report the reduction in ||g_null|| from adding
    it. Reported, not gated. This is the constructive form of whatever R3 concludes, and it puts
    an aggregate assay and a direct rate measurement in the same currency for the first time.

R7  DOMAIN. Repeat at the rarer question. rateneed's N6 and hybrid's H11 both showed the
    requirement tightens with tail depth; if ||g_null||/||g|| grows too, then physiology becomes
    less sufficient exactly where the question gets interesting.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.optimize import fsolve

from rem.atlas.hybrid_tune import (
    RULE, NAMES, CANDIDATE, ORDERS_PER_KCAL, eradication, sensitivity,
)
from rem.atlas.hybrid import K, G0, CYCLES, T_ON, T_OFF, SEED, N_RATES
from rem.atlas.constrain import _aggregates, EPS

G0_DEEP = 8
OBS_NAMES = ("doubling", "logkill", "plateau", "lag")
N_MANIFOLD = 400


def aggregates_from_log(x, repaired=False):
    """Observables as a function of log10 rates. `repaired` replaces the lag observable with one
    that depends on the post-drug COMPOSITION, which is what an outgrowth assay actually sees."""
    r = {nm: CANDIDATE[nm] * 10.0 ** x[k] for k, nm in enumerate(NAMES)}
    d, lk, pl, lag = _aggregates(*(r[nm] for nm in
        ("mu", "k_kill", "a_off", "a_on", "b_off", "b_on", "d_death", "kd_kill")))
    if not repaired:
        return np.array([float(d), float(lk), float(pl), float(lag)])
    # Repaired outgrowth: propagate the ACTUAL post-drug (G, D) mixture through the off-phase
    # mean field and read the time to recover the pre-drug population. A culture that emerges
    # mostly dormant regrows differently from one that emerges mostly growing, at identical
    # total count -- that is the information the first definition discarded.
    A = -(r["k_kill"] + r["a_on"]); B = r["b_on"]
    C = r["a_on"]; D = -(r["b_on"] + r["kd_kill"])
    Mon = np.array([[A, B], [C, D]])
    from scipy.linalg import expm as _e
    v = _e(Mon * T_ON) @ np.array([1.0, 0.0])
    Moff = np.array([[r["mu"] - r["a_off"], r["b_off"]],
                     [r["a_off"], -(r["b_off"] + r["d_death"])]])
    ts = np.linspace(0.0, 60.0, 601)
    tot = np.array([(_e(Moff * t) @ v).sum() for t in ts])
    hit = np.where(tot >= 1.0)[0]
    lag_r = float(ts[hit[0]]) if len(hit) else 60.0
    return np.array([float(d), float(lk), float(pl), lag_r])


def jacobian(f, x0, h):
    q = len(f(x0))
    J = np.zeros((q, len(x0)))
    for k in range(len(x0)):
        xp = x0.copy(); xp[k] += h
        xm = x0.copy(); xm[k] -= h
        J[:, k] = (f(xp) - f(xm)) / (2 * h)
    return J


def split(g, J, rcond=1e-8):
    U, s, Vt = np.linalg.svd(J)
    r = int((s > rcond * s.max()).sum()) if s.size and s.max() > 0 else 0
    Vr = Vt[:r].T
    g_row = Vr @ (Vr.T @ g) if r else np.zeros_like(g)
    return s, r, g_row, g - g_row


def main():
    out = []

    def P(t=""):
        print(t, flush=True)
        out.append(t)

    P(RULE); P("HOW MANY RATE DIRECTIONS DOES MEASURED PHYSIOLOGY PIN?"); P(RULE)
    x0 = np.zeros(N_RATES)
    sigma = EPS * ORDERS_PER_KCAL
    kw = dict(K=K, g0=G0, cycles=CYCLES)
    g = np.array([sensitivity(CANDIDATE, nm, 0.02, **kw) for nm in NAMES])
    P(f"  answer gradient g = d log10 Y / d log10 k, ||g|| = {np.linalg.norm(g):.4f}")
    for nm, v in zip(NAMES, g):
        P(f"    {nm:>9} {v:+.4f}")
    P(f"  chemistry error sigma = {sigma:.4f} orders per rate ({EPS} kcal/mol)")

    # ---- R1 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("R1  THE JACOBIAN IS A JACOBIAN"); P(RULE)
    J1 = jacobian(aggregates_from_log, x0, 0.02)
    J2 = jacobian(aggregates_from_log, x0, 0.01)
    rel = float(np.abs(J1 - J2).max() / max(np.abs(J2).max(), 1e-300))
    P(f"  worst entry change on halving the step: {rel:.2e}"
      f"   {'PASS' if rel < 0.01 else 'FAIL'} (bar 1%)")

    # ---- R2 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("R2  HOW MANY INDEPENDENT CONSTRAINTS ARE THERE REALLY?"); P(RULE)
    s, r, g_row, g_null = split(g, J1)
    P(f"  J is {J1.shape[0]} observables x {J1.shape[1]} rates. Singular values:")
    for i, sv in enumerate(s):
        P(f"    s{i+1} = {sv:.6e}   (ratio to s1: {sv/s[0]:.3e})")
    P(f"  numerical rank at rcond 1e-8: {r} of {J1.shape[0]} nominal measurements")
    if r < J1.shape[0]:
        U, _, _ = np.linalg.svd(J1)
        dep = U[:, r:]
        P(f"  the dependency, as a combination of the observables (left null vector):")
        for j in range(dep.shape[1]):
            P("    " + "  ".join(f"{OBS_NAMES[i]} {dep[i, j]:+.4f}" for i in range(len(OBS_NAMES))))
        P("  READING: the nominal count of measurements OVERSTATES the real one.")
    else:
        P("  READING: all nominal measurements are independent.")

    # ---- R3 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("R3  THE DELIVERABLE  --  what perfect physiology cannot reach"); P(RULE)
    frac = float(np.linalg.norm(g_null) / np.linalg.norm(g))
    P(f"  ||g||       = {np.linalg.norm(g):.4f}   (unconstrained)")
    P(f"  ||g_row||   = {np.linalg.norm(g_row):.4f}   (pinned by perfect physiology)")
    P(f"  ||g_null||  = {np.linalg.norm(g_null):.4f}   (free no matter how good the instruments)")
    P(f"  fraction of the answer's sensitivity that survives perfect physiology: {frac:.4f}")
    P(f"  irreducible answer uncertainty at {EPS} kcal/mol: sigma*||g_null|| ="
      f" {sigma*np.linalg.norm(g_null):.4f} orders")
    P(f"  against the unconstrained     : sigma*||g||      ="
      f" {sigma*np.linalg.norm(g):.4f} orders")
    P("  the free direction, by rate:")
    for nm, v in zip(NAMES, g_null):
        P(f"    {nm:>9} {v:+.4f}")
    P(f"  READING: {'route OPEN, physiology essentially determines the answer' if frac < 0.1 else ('the aggregates pin directions the answer barely depends on' if frac > 0.5 else 'partial -- reported as measured')}")

    # ---- R4 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("R4  THE EXACT MANIFOLD SAMPLE  (constrain.py's unanswerable C7, answered)")
    P(RULE)
    y_true = eradication(CANDIDATE, **kw)
    ly = np.log10(y_true)
    obs_true = aggregates_from_log(x0)
    # Solve for the r rates that the constrained directions load on most heavily -- those are the
    # ones the observables actually determine, and the best conditioned to invert for.
    _, _, Vt = np.linalg.svd(J1)
    solve_for = list(np.argsort(-np.abs(Vt[:r]).sum(axis=0))[:r])
    free = [k for k in range(N_RATES) if k not in solve_for]
    P(f"  solving for {[NAMES[k] for k in solve_for]} to match the observables exactly,")
    P(f"  with {[NAMES[k] for k in free]} drawn from chemistry at {EPS} kcal/mol.")
    rng = np.random.default_rng(SEED + 3)
    vals, nfail = [], 0
    for _ in range(N_MANIFOLD):
        x = np.zeros(N_RATES)
        x[free] = rng.normal(0.0, sigma, len(free))

        def eqs(z):
            xx = x.copy(); xx[solve_for] = z
            return (aggregates_from_log(xx) - obs_true)[:r]

        z, info, ok, _msg = fsolve(eqs, np.zeros(r), full_output=True)
        if ok != 1 or np.abs(eqs(z)).max() > 1e-8:
            nfail += 1
            continue
        xx = x.copy(); xx[solve_for] = z
        rr = {nm: CANDIDATE[nm] * 10.0 ** xx[k] for k, nm in enumerate(NAMES)}
        vals.append(np.log10(max(eradication(rr, t_on=T_ON, t_off=T_OFF, **kw), 1e-300)) - ly)
    v = np.array(vals)
    P(f"  {len(v)} exact-manifold points ({nfail} solver failures, reported not hidden)")
    if len(v) > 20:
        pred = sigma * np.linalg.norm(g_null)
        P(f"  measured sd of log10(Y/Y_true) on the manifold : {v.std(ddof=1):.4f} orders")
        P(f"  linear prediction sigma*||g_null||             : {pred:.4f} orders")
        rel4 = abs(v.std(ddof=1) - pred) / pred
        P(f"  relative disagreement {rel4:.4f}   {'PASS' if rel4 <= 0.30 else 'FAIL -- R3 is a local statement only'} (bar 30%)")
        P(f"  p05 {np.percentile(v,5):+.4f}, p95 {np.percentile(v,95):+.4f},"
          f" full range {v.max()-v.min():.4f} orders")
        P(f"  within x2 {float((np.abs(v)<=np.log10(2)).mean()):.4f},"
          f" within x10 {float((np.abs(v)<=1.0).mean()):.4f}")
        P("  These are cells with EXACTLY the true physiology -- no tolerance at all.")

    # ---- R5 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("R5  THE LAG DEFECT, REPAIRED"); P(RULE)
    Jr = jacobian(lambda z: aggregates_from_log(z, repaired=True), x0, 0.02)
    sr, rr_, grow_r, gnull_r = split(g, Jr)
    P("  outgrowth redefined from the post-drug composition rather than the total count.")
    P(f"  singular values: " + ", ".join(f"{x:.4e}" for x in sr))
    P(f"  rank {rr_} (was {r} with the original definition)")
    fr = float(np.linalg.norm(gnull_r) / np.linalg.norm(g))
    P(f"  fraction surviving perfect physiology: {fr:.4f}  (was {frac:.4f})")
    P(f"  irreducible uncertainty: {sigma*np.linalg.norm(gnull_r):.4f} orders"
      f"  (was {sigma*np.linalg.norm(g_null):.4f})")
    P(f"  READING: {'the repair recovers a real constraint; BOTH numbers are reported' if rr_ > r else 'the dependency is structural, not an artefact of my simplification'}")

    # ---- R6 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("R6  WHAT SHOULD BE MEASURED NEXT?"); P(RULE)
    Jbest = Jr if rr_ > r else J1
    base_null = float(np.linalg.norm(split(g, Jbest)[3]))
    cands = {}
    # a second point on the time-kill curve
    for mult, lbl in ((2.0, "log-kill at 2x the drug window"), (0.5, "log-kill at half the window")):
        def f(z, m=mult):
            rr2 = {nm: CANDIDATE[nm] * 10.0 ** z[k] for k, nm in enumerate(NAMES)}
            from scipy.linalg import expm as _e
            Mon = np.array([[-(rr2["k_kill"] + rr2["a_on"]), rr2["b_on"]],
                            [rr2["a_on"], -(rr2["b_on"] + rr2["kd_kill"])]])
            return np.array([np.log10(max((_e(Mon * T_ON * m) @ np.array([1.0, 0.0])).sum(), 1e-300))])
        cands[lbl] = jacobian(f, x0, 0.02)
    # dormant fraction at the end of the drug window
    def f_frac(z):
        rr2 = {nm: CANDIDATE[nm] * 10.0 ** z[k] for k, nm in enumerate(NAMES)}
        from scipy.linalg import expm as _e
        Mon = np.array([[-(rr2["k_kill"] + rr2["a_on"]), rr2["b_on"]],
                        [rr2["a_on"], -(rr2["b_on"] + rr2["kd_kill"])]])
        v2 = _e(Mon * T_ON) @ np.array([1.0, 0.0])
        return np.array([np.log10(max(v2[1] / max(v2.sum(), 1e-300), 1e-300))])
    cands["dormant fraction after the drug window"] = jacobian(f_frac, x0, 0.02)
    # each single rate, measured directly
    for k, nm in enumerate(NAMES):
        row = np.zeros((1, N_RATES)); row[0, k] = 1.0
        cands[f"direct measurement of {nm}"] = row
    P(f"  baseline ||g_null|| with the repaired aggregate set: {base_null:.4f}")
    P(f"  {'candidate additional measurement':>42}{'new ||g_null||':>16}{'reduction':>12}")
    ranked = []
    for lbl, row in cands.items():
        _, _, _, gn = split(g, np.vstack([Jbest, row]))
        ranked.append((float(np.linalg.norm(gn)), lbl))
    for val, lbl in sorted(ranked):
        P(f"  {lbl:>42}{val:>16.4f}{base_null-val:>12.4f}")

    # ---- R7 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("R7  DOMAIN  --  the rarer question"); P(RULE)
    kwd = dict(K=K, g0=G0_DEEP, cycles=CYCLES)
    gd = np.array([sensitivity(CANDIDATE, nm, 0.02, **kwd) for nm in NAMES])
    _, _, _, gnd = split(gd, Jbest)
    fd = float(np.linalg.norm(gnd) / np.linalg.norm(gd))
    P(f"  g0 = {G0_DEEP}: ||g|| = {np.linalg.norm(gd):.4f}, ||g_null|| = {np.linalg.norm(gnd):.4f}")
    P(f"  fraction surviving perfect physiology: {fd:.4f}  (against {float(np.linalg.norm(split(g,Jbest)[3])/np.linalg.norm(g)):.4f} at g0 = {G0})")
    P(f"  irreducible uncertainty {sigma*np.linalg.norm(gnd):.4f} orders"
      f"  (against {sigma*base_null:.4f})")

    P("\n" + RULE)
    P("R3 and R4 together answer the question rejection sampling could not: not 'how good are")
    P("our instruments' but 'what is left when the instruments are perfect'. R6 says what to")
    P("measure next, with aggregate assays and direct rate measurements in the same currency.")
    P(RULE)

    open(os.path.join(os.path.dirname(__file__), "RESULTS_constrain_rank.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
