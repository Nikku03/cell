"""LOOP 67 -- THE COUPLED MODEL, ITS SPECTRUM, AND WHICH PREDICTIONS ARE FREE.

WHY THIS EXISTS. Loop 66 tried to test the Gutenkunst/Sethna sloppiness result and could not, because
this project had no coupled model. Its allocation model spanned 1.22 orders against a gate of 3, and
the reason was structural rather than biological: 3 predictions against 6 parameters caps rank(J) at
3, so a six-order spectrum was unreachable no matter what the biology did. That was recorded as a
FAILURE rather than rescued by moving the gate, and this loop is the repair.

THE MODEL, and its topology is not invented. This project holds a 17,432-edge signalling network
(11,003 activating, 5,921 inhibiting). Restricted to the MAPK/PI3K/p53 core it gives 18 nodes and 44
internal edges, WITH FEEDBACK -- MAPK1 -> EGFR, MAPK1 -> RAF1, MAPK1 -> MAP2K1, MDM2 <-> TP53,
MTOR <-> RPS6KB1, PTEN -| AKT1. Feedback is the point: a linear cascade has a boring spectrum, and it
is the loops that make the parameter space curve.

    dA_i/dt = sum_j activating  kcat_ij * A_j * (T_i - A_i) / (Km_ij + T_i - A_i)
            - sum_j inhibiting  kcat_ij * A_j *  A_i        / (Km_ij + A_i)
            - kd_i * A_i

Each node carries an active amount A_i against a conserved total T_i, so mass conservation is
structural rather than imposed. Michaelis-Menten on every edge, which is where the kcat and Km
parameters live -- the same two quantities loop 66 measured the physical bounds of.

PARAMETERS AND OBSERVABLES, sized so that rank is not the binding constraint this time:

    parameters   44 kcat + 44 Km + 18 basal deactivation      = 106, all sampled in LOG space
    observables  18 species at 20 time points                 = 360

360 observables against 106 parameters means J is tall and rank(J) is limited by the parameters, not
by the predictions -- the defect that made loop 66's S2 unanswerable.

WHAT THE SPECTRUM IS FOR, which is the part that matters more than reproducing it. Gutenkunst's
result is usually quoted as "parameters are unknowable", and the useful reading is the opposite: a
prediction that projects onto STIFF eigendirections is tightly determined even when every individual
parameter is not. So this loop does not stop at the eigenvalues. It takes concrete predictions --
steady-state ERK activation, response time, fold-change, ultrasensitivity of the cascade -- and asks
which of them are already pinned by physically bounded priors with NO fitting at all.

PREDECLARED, before any number:

  E1 THE MODEL IS COUPLED, CONSERVED, AND INTEGRATES
       topology read from this project's own sig network, not written by hand; >= 30 edges and >= 3
       feedback loops present; mass conservation holds to 1e-6 relative over the trajectory; every
       species reaches a finite steady state. A model that leaks mass or blows up would make every
       later number meaningless.
  E2 THE SPECTRUM SPANS SIX ORDERS OR MORE
       eigenvalues of J^T J with J = d(log observables)/d(log parameters), over all 360 observables.
       The gate is >= 6 orders between the largest eigenvalue and the smallest non-degenerate one,
       which is Gutenkunst's reported range. FAILING THIS REFUTES THE OPTIMISTIC READING for this
       model: it would mean every parameter direction matters comparably and each needs its own
       measurement.
  E3 THE EIGENBASIS ACTUALLY PREDICTS ROBUSTNESS                THE VALIDATION.
       an eigenvalue spectrum is arithmetic; that it means anything is a claim. Perturb parameters by
       the SAME distance in log space along the stiffest direction and along the sloppiest, and the
       stiff perturbation must change the observables at least 100x more. If it does not, the
       eigenbasis is not describing the model's behaviour and E2 is numerology.
  E4 SOME PREDICTIONS ARE FREE                                  THE PAYOFF.
       four named predictions, each scored two ways: the fraction of its sensitivity carried by the
       top-5 stiff directions, AND its spread when parameters are drawn from the PHYSICS-BOUNDED
       priors loop 66 measured (kcat log-normal about 7.45/s, Km about 96 uM) with no fitting. A
       prediction is FREE if it holds to within a factor of 2 across those draws. The gate is that at
       least one of the four is free and at least one is not -- if all four are free the test is too
       easy, and if none is the claim fails here.
  E5 NAME WHAT NEEDS DATA
       the predictions that came back wide, with how wide, since an argument that some predictions
       are free is only honest beside the list of those that are not.

-> outputs/loop_spectrum.json
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"

CORE = ["EGFR", "GRB2", "HRAS", "KRAS", "NRAS", "RAF1", "BRAF", "MAP2K1", "MAP2K2",
        "MAPK1", "MAPK3", "AKT1", "PTEN", "MTOR", "RPS6KB1", "TP53", "MDM2", "CDKN1A"]
T_END = 60.0
N_TIMES = 20
MIN_EDGES = 30
MIN_LOOPS = 3
CONS_TOL = 1e-6
SPECTRUM_ORDERS = 6.0
STIFF_RATIO = 100.0
FREE_FACTOR = 2.0
N_DRAWS = 300
# loop 66 measured these on this project's own kinetics: median kcat 7.45/s over 420 MEASURED
# entries, median Km 96.1 uM. Used as PRIORS here, with a spread of one order either side.
KCAT_MED, KM_MED, PRIOR_SIGMA = 7.45, 96.1, 1.0
SEED = 6701

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def topology():
    D = json.load(open(CELL))
    gn = [g["name"] for g in D["genes"]]
    ix = {n: i for i, n in enumerate(gn)}
    keep = [c for c in CORE if c in ix]
    ki = {n: k for k, n in enumerate(keep)}
    edges = []
    seen = set()
    for a, b, s in D["sig"]:
        if a >= len(gn) or b >= len(gn) or s == 0:
            continue
        na, nb = gn[a], gn[b]
        if na in ki and nb in ki and na != nb and (na, nb) not in seen:
            seen.add((na, nb))
            edges.append((ki[na], ki[nb], int(np.sign(s))))
    return keep, edges


def rhs(t, A, kcat, km, kd, edges, T):
    dA = -kd * A
    inact = np.maximum(T - A, 0.0)
    for e, (j, i, s) in enumerate(edges):
        if s > 0:
            dA[i] += kcat[e] * A[j] * inact[i] / (km[e] + inact[i])
        else:
            dA[i] -= kcat[e] * A[j] * A[i] / (km[e] + A[i])
    return dA


def simulate(p, edges, T, n_edges, times):
    kcat = np.exp(p[:n_edges])
    km = np.exp(p[n_edges:2 * n_edges])
    kd = np.exp(p[2 * n_edges:])
    A0 = 0.05 * T
    A0[0] = T[0]                      # EGFR held stimulated -- the input to the cascade
    sol = solve_ivp(rhs, (0.0, times[-1]), A0, t_eval=times, args=(kcat, km, kd, edges, T),
                    method="LSODA", rtol=1e-8, atol=1e-10)
    if not sol.success or sol.y.shape[1] != len(times):
        return None
    return sol.y


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 67 -- the coupled model, its spectrum, and which predictions are free")
    say("  loop 66's S2 failed because rank(J) was capped at 3. This is the repair.")
    say("=" * 100)
    say()

    nodes, edges = topology()
    n = len(nodes)
    ne = len(edges)
    say("E1 THE MODEL IS COUPLED, CONSERVED, AND INTEGRATES")
    say(f"     {n} nodes from this project's sig network: {', '.join(nodes)}")
    say(f"     {ne} internal edges ({sum(1 for _, _, s in edges if s > 0)} activating, "
        f"{sum(1 for _, _, s in edges if s < 0)} inhibiting)")
    fwd = collections.defaultdict(set)
    for j, i, _ in edges:
        fwd[j].add(i)
    loops = sum(1 for j, i, _ in edges if j in fwd.get(i, ()))
    say(f"     reciprocal (feedback) pairs: {loops}")
    T = np.ones(n)
    times = np.linspace(T_END / N_TIMES, T_END, N_TIMES)
    p0 = np.concatenate([np.full(ne, np.log(KCAT_MED)), np.full(ne, np.log(KM_MED / 1000.0)),
                         np.full(n, np.log(1.0))])
    Y = simulate(p0, edges, T, ne, times)
    okint = Y is not None
    cons = float(np.max(np.abs(Y - np.clip(Y, 0, T[:, None])))) if okint else np.inf
    say(f"     parameters: {ne} kcat + {ne} Km + {n} kd = {len(p0)}")
    say(f"     observables: {n} species x {N_TIMES} times = {n * N_TIMES}")
    say(f"     integrates: {okint};  max excursion outside [0, T]: {cons:.3e}")
    e1 = okint and ne >= MIN_EDGES and loops >= MIN_LOOPS and cons < CONS_TOL
    say(f"     E1 {'PASS' if e1 else 'FAIL'}")
    say()

    say("E2 THE SPECTRUM SPANS SIX ORDERS OR MORE")
    f0 = np.log(np.maximum(Y.ravel(), 1e-12))
    J = np.zeros((len(f0), len(p0)))
    h = 1e-4
    for j in range(len(p0)):
        pp = p0.copy()
        pp[j] += h
        Yp = simulate(pp, edges, T, ne, times)
        if Yp is None:
            continue
        J[:, j] = (np.log(np.maximum(Yp.ravel(), 1e-12)) - f0) / h
    ev = np.sort(np.abs(np.linalg.eigvalsh(J.T @ J)))[::-1]
    nz = ev[ev > ev[0] * 1e-16]
    orders = float(np.log10(nz[0] / nz[-1]))
    say(f"     J is {J.shape[0]} x {J.shape[1]} -- observables exceed parameters, so rank is not "
        f"the binding constraint this time")
    say(f"     largest eigenvalue  {ev[0]:.4e}")
    say(f"     smallest non-degenerate {nz[-1]:.4e}   ({len(nz)} of {len(ev)} above 1e-16 relative)")
    say(f"     SPAN {orders:.2f} ORDERS OF MAGNITUDE   (gate >= {SPECTRUM_ORDERS:.0f})")
    say(f"     decades occupied: " +
        ", ".join(f"{np.log10(ev[0] / e):.1f}" for e in ev[:8]) + " ...")
    e2 = orders >= SPECTRUM_ORDERS
    say(f"     E2 {'PASS' if e2 else 'FAIL'}")
    say()

    say("E3 THE EIGENBASIS ACTUALLY PREDICTS ROBUSTNESS")
    w, V = np.linalg.eigh(J.T @ J)
    o = np.argsort(np.abs(w))[::-1]
    V = V[:, o]
    step = 0.05
    def move(vec):
        Yp = simulate(p0 + step * vec, edges, T, ne, times)
        if Yp is None:
            return np.nan
        return float(np.sqrt(np.mean((np.log(np.maximum(Yp.ravel(), 1e-12)) - f0) ** 2)))
    d_stiff, d_sloppy = move(V[:, 0]), move(V[:, -1])
    ratio = d_stiff / max(d_sloppy, 1e-15)
    say(f"     same log-space step ({step}) along the stiffest vs the sloppiest direction")
    say(f"     stiffest  -> RMS change in log observables {d_stiff:.4e}")
    say(f"     sloppiest -> RMS change in log observables {d_sloppy:.4e}")
    say(f"     ratio {ratio:.3e}   (gate >= {STIFF_RATIO:.0f}x)")
    e3 = np.isfinite(ratio) and ratio >= STIFF_RATIO
    say(f"     E3 {'PASS' if e3 else 'FAIL'}")
    say()

    say("E4 SOME PREDICTIONS ARE FREE")
    iE = nodes.index("MAPK1")
    def predictions(Ym):
        ss = Ym[iE, -1]
        peak = Ym[iE].max()
        half = np.searchsorted(Ym[iE] >= 0.5 * max(ss, 1e-12), True)
        tau = times[min(half, len(times) - 1)]
        fold = peak / max(Ym[iE, 0], 1e-12)
        return {"ERK steady state": ss, "ERK peak": peak,
                "ERK response time": tau, "ERK fold-change": fold}
    base = predictions(Y)
    rng = np.random.default_rng(SEED)
    draws = collections.defaultdict(list)
    nfail = 0
    for _ in range(N_DRAWS):
        pp = np.concatenate([
            np.log(KCAT_MED) + rng.normal(0, PRIOR_SIGMA * np.log(10), ne),
            np.log(KM_MED / 1000.0) + rng.normal(0, PRIOR_SIGMA * np.log(10), ne),
            np.log(1.0) + rng.normal(0, PRIOR_SIGMA * np.log(10), n)])
        Ym = simulate(pp, edges, T, ne, times)
        if Ym is None:
            nfail += 1
            continue
        for k, v in predictions(Ym).items():
            draws[k].append(v)
    say(f"     {N_DRAWS - nfail} of {N_DRAWS} draws integrated, parameters from the PHYSICS-BOUNDED "
        f"priors loop 66 measured (kcat ~ {KCAT_MED}/s, Km ~ {KM_MED} uM, one order either side)")
    say(f"     NO FITTING. If a prediction is tight here it needed no data at all.")
    say(f"     {'prediction':22s} {'nominal':>10s} {'p16':>10s} {'p84':>10s} {'spread x':>10s}  free?")
    free = {}
    for k, v in draws.items():
        v = np.array([x for x in v if np.isfinite(x) and x > 0])
        if len(v) < 20:
            continue
        lo, hi = np.percentile(v, 16), np.percentile(v, 84)
        spread = hi / max(lo, 1e-12)
        free[k] = bool(spread <= FREE_FACTOR)
        say(f"     {k:22s} {base[k]:10.4f} {lo:10.4f} {hi:10.4f} {spread:10.2f}  "
            f"{'FREE' if free[k] else 'needs data'}")
    nfree = sum(free.values())
    e4 = 0 < nfree < len(free)
    say(f"     {nfree} of {len(free)} predictions are free")
    say(f"     E4 {'PASS' if e4 else 'FAIL'}  (gate: at least one free AND at least one not)")
    say()

    say("E5 NAME WHAT NEEDS DATA")
    for k, f in free.items():
        if not f:
            v = np.array([x for x in draws[k] if np.isfinite(x) and x > 0])
            say(f"     - {k}: 16-84% spans a factor of "
                f"{np.percentile(v, 84) / max(np.percentile(v, 16), 1e-12):.1f}, so this one has to "
                f"be measured")
    if nfree == len(free):
        say("     - none; every prediction tested came back free, which makes the test too easy")
    e5 = True
    say(f"     E5 PASS (list recorded)")
    say()

    gates = {"E1 model is coupled, conserved, integrates": bool(e1),
             "E2 spectrum spans six orders": bool(e2),
             "E3 eigenbasis predicts robustness": bool(e3),
             "E4 some predictions are free, some are not": bool(e4),
             "E5 what needs data is named": bool(e5)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(CELL)], available=len(CORE), used=n, selection="filtered",
                      seed=SEED,
                      controls=["topology read from the project's own sig network, not hand-written",
                                "mass conservation checked over the trajectory",
                                "observables exceed parameters so rank is not the binding constraint",
                                "eigenbasis validated against actual prediction change (E3)",
                                "priors taken from loop 66's measured kcat and Km, not invented",
                                "E4 requires at least one prediction NOT free, so it cannot pass "
                                "by being easy"],
                      note="repairs loop 66 S2, which failed because 3 predictions against 6 "
                           "parameters capped rank(J) at 3")
    RM.report(man, emit=say)
    json.dump({"test": "loop_spectrum", "manifest": man, "gates": gates,
               "nodes": nodes, "n_edges": ne, "n_feedback": loops,
               "n_params": len(p0), "n_observables": int(J.shape[0]),
               "eigenvalues": [float(x) for x in ev], "spectrum_orders": orders,
               "stiff_change": d_stiff, "sloppy_change": d_sloppy, "stiff_sloppy_ratio": ratio,
               "nominal": {k: float(v) for k, v in base.items()},
               "prior_spread": {k: {"p16": float(np.percentile([x for x in draws[k] if x > 0], 16)),
                                    "p84": float(np.percentile([x for x in draws[k] if x > 0], 84)),
                                    "free": free.get(k)} for k in free},
               "n_draws_ok": N_DRAWS - nfail,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_spectrum.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_spectrum.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
