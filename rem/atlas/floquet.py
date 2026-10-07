"""Build item 3: the phase-resolved (Floquet) periodic solver of spec section 3.

WHAT IT COMPUTES. A generator that switches between configurations over a cycle of period T.
The cycle propagator A is assembled phase by phase; the periodic state is the fixed point of A,
extracted by a LINEAR SOLVE (never an eigenvector -- see F-eigen below); the cycle is then
stepped again to record the distribution at every phase.

THE PHYSICAL CASE, spec section 3.2. One gene, birth = k (transcription), death = mu * n
(dilution/decay), so tau = 1/mu. Transcription switches 1x / 2x per half cycle (DNA
replication). The cycle-averaged birth rate is 1.5k, so the mean is 1.5k/mu; setting mu = 1
(tau = 1) and k = 10 makes it exactly 15.0. That is arithmetic, not a fit.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN. Thresholds are the spec's, not anything measured here.
=================================================================================================

Every gate is on the TAIL quantity P(n >= 29) unless stated, reported as a RELATIVE error, and
scored on the WORST case over the swept points, never the median.

G2.1 / T06   T/tau -> 0 reproduces the CONSTANT-rate stationary answer to < 1% IN THE TAIL.
             Evaluated at the smallest swept point, T/tau = 0.01; also reported at T/tau = 0.1
             where the spec's own table says the ratio is 1.00x.
             DISCRIMINATION (predeclared, so the gate is not vacuous): the identical comparison
             at T/tau = 10 must MISS the 1% bar by more than two orders. If it does not, the
             bar is not testing anything.

G2.2 / T07   T/tau -> inf (evaluated at T/tau = 100) reproduces the ADIABATIC AVERAGE to < 5%
             IN THE TAIL. The adiabatic average is the DURATION-WEIGHTED MEAN OF THE TWO
             STATIONARY DISTRIBUTIONS, 0.5*Poisson(10) + 0.5*Poisson(20) -- NOT the stationary
             distribution of the mean rate, Poisson(15). The two are printed side by side to
             show the distinction is not cosmetic.
             DISCRIMINATION: the identical comparison at T/tau = 0.1 must MISS the 5% bar.

G2.3 / T08   Mean preserved to < 0.01% (relative) at EVERY swept T. Mean printed to 4 decimals
             at every T, as the spec table demands.

G2.4         No NaN and no inf anywhere -- periodic state, every recorded phase, and every
             derived statistic -- for T/tau in [0.01, 1000]. At least 12 swept values; 17 used.

F-monotone   The tail ratio against the CONSTANT model must climb MONOTONICALLY with T/tau and
             SATURATE. Scored as three clauses, all predeclared:
               (a) ratio[i+1] >= ratio[i] - 1e-9 for every consecutive pair of the 17-point
                   sweep (the 1e-9 is double-precision slack on an O(1..20) quantity, declared
                   a priori, not fitted);
               (b) NO INTERIOR MAXIMUM: argmax over the sweep is the largest T;
               (c) SATURATION: ratio at T/tau = 1000 is within 5% below the adiabatic ratio and
                   does not exceed it (slack 1e-9). Approach from below with no overshoot is
                   what makes the slow-cycle limit a computable worst-case bound.

F-control    MANDATORY NEGATIVE CONTROL. Both phase rates set EQUAL (both 15, no switching).
             Then the periodic solver MUST return the constant-rate stationary distribution at
             every phase, and the tail ratio must be 1.000 at EVERY swept T.
             BARS: |ratio - 1| < 1e-6 at every T (primary, spec's wording); and worst relative
             error of the cycle-averaged distribution against solver.stationary() over states
             with reference probability > 1e-10, bar < 1e-5.
             The two bars are set from an A PRIORI double-precision argument, not measurement:
             the dense expm/solve route carries absolute error ~1e-16 times the largest
             component (~0.1), i.e. ~1e-17 absolute, so a reference probability of 1e-10 has an
             a priori relative noise floor near 1e-7 and P(n>=29) ~ 8.6e-4 near 1e-13. Both
             bars sit above their floors and far below any real artefact.
             WHAT IT CATCHES: any tail inflation manufactured by the solver itself rather than
             by the cell cycle -- a sign error in L.T, expm/matrix_power drift, the clipping in
             max(solve(B,b),0) biasing the tail upward, boundary reflection at the truncation,
             or a wrong row replaced in B. If a no-switching cycle already reports a ratio of
             1.05, then the 19x in the table is measuring the solver, not DNA replication.
             WHAT IT CANNOT CATCH: phase-weighting errors, because with equal rates the
             distribution is time-independent and every weighting gives the same answer. That
             hole is covered by G2.3 (an unequal phase weighting moves the mean off 15) and by
             F-swap below.

F-swap       SECOND CONTROL, for the hole F-control leaves. Running the cycle in the opposite
             phase order [2k, k] instead of [k, 2k] is a time translation, so the
             CYCLE-AVERAGED distribution must be identical. BAR: worst relative difference
             < 1e-8 over states with probability > 1e-10 (same a priori floor argument).
             WHAT IT CATCHES: off-by-one phase bookkeeping, recording p after the step instead
             of before, and duration/weight misalignment between phases.

F-eigen      NEGATIVE CONTROL ON THE FORBIDDEN ROUTE. The eigenvector extraction the spec
             forbids is implemented and RUN, once per swept T, exactly as a naive user would
             write it: eigenvector of A for the eigenvalue nearest 1, clip negatives, normalise.
             Reported: NaN count, sign of the raw vector, and worst relative tail error against
             the linear-solve answer. Scored PASS if the route is demonstrably unsafe somewhere
             in the sweep (NaN, or a tail error above 1e-3). If it silently agrees everywhere,
             that is the finding and the gate is reported FAIL -- a prohibition nobody can
             reproduce is not a rule.

F-exact      INDEPENDENT-ROUTE CROSS-CHECK (not in the spec; added because the linear algebra
             needs a reference that is not linear algebra). For birth = lambda(t), death = mu*n,
             the CME maps Poisson(m) to Poisson(m') exactly, so the phase-resolved solution is
             EXACTLY Poisson(m(t)) with dm/dt = lambda(t) - mu*m, and the periodic m(t) is a
             closed form. This gives an analytic reference for the mean, the Fano factor and
             (by quadrature over the cycle) the tail, with no matrices anywhere.
             BARS: worst relative error over the 5 spec-table rows < 1e-3 for the tail and
             < 1e-3 for the Fano factor, at the sub-step count used for the table. Set a priori
             from the quadrature order: the recorded phase average is a uniform-grid periodic
             trapezoid, second order in 1/n_sub, and n_sub = 1000 per phase puts it near 1e-5.

F-period     The stepped cycle must return to its own starting point. BAR: max |p_end - p_0| /
             max(p_0) < 1e-10. Catches a propagator that does not match the one used to build A.

COST         Spec section 3: one gene, 12 phases, 91 states -> 5.0 ms. Measured as the best of
             several runs of the whole pipeline (build, expm per phase, cycle product, solve,
             step and record). Reported with no bar attached beyond the spec's own number.

=================================================================================================
SPEC SECTION 3.2 TABLE TO REPRODUCE (T/tau, mean, Fano, P(n>=29), ratio vs CONSTANT)
      0.1  15.0000  1.0003  8.6265e-04   1.00x
      1.0  15.0000  1.0339  1.0547e-03   1.23x
      3.0  15.0000  1.2553  2.5740e-03   2.99x
     10.0  15.0000  2.0094  9.8847e-03  11.48x
    100.0  15.0000  2.6042  1.6477e-02  19.14x
=================================================================================================
POST-RUN AMENDMENT, written after the first execution and kept separate from the predeclaration
above so the record of what was declared BEFORE running stays intact. Nothing above was edited.

A1  F-swap's bar was MIS-SET BY ME, and it is not being moved. The clause asked for a worst
    RELATIVE difference below 1e-8 over states with probability > 1e-10, while the same
    paragraph's own noise argument puts the a priori floor for a 1e-10 reference probability at
    ~1e-7. A bar an order BELOW its own floor cannot pass on perfect evidence, so by project
    rule 5 it is reported as declared (it fails, at 2.3e-08) and then marked VOID: it decides
    nothing. The measurement that backs this is printed with the gate -- the worst relative
    difference stops moving once the floor drops past 1e-8 (2.311e-08 at floors 1e-8, 1e-10 and
    1e-12 alike) because it is one fixed absolute wobble of ~3.5e-16 divided by an ever smaller
    p, which is the signature of double-precision noise and not of an asymmetry. The
    ACHIEVABLE version of the same question -- does the phase order move P(n>=29), the quantity
    the whole table is built on -- is reported beside it as a post-hoc diagnostic and is
    labelled as post-hoc wherever it appears.

A2  COST is reported for the required implementation pattern (12 separate expm calls, no
    caching), best of several warmed runs, and additionally at one BLAS thread, because on this
    4-core machine the default thread pool makes 91x91 expm calls ~5x SLOWER through
    oversubscription. Both numbers are printed; the gate is scored on the faster one.
"""
from __future__ import annotations

import math
import os
import time
from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad

try:
    from threadpoolctl import threadpool_limits
    _HAVE_TPC = True
except ImportError:                      # cost section degrades to default threads only
    _HAVE_TPC = False

from rem.atlas.solver import (
    birth_death_arrays,
    poisson_reference,
    stationary,
    truncation_cap,
)

# ---- case constants, all from the spec or from arithmetic on it, none from a measurement ----
MU = 1.0            # spec 3.2: tau = 1/mu, so mu = 1 sets tau = 1 and T/tau = T
K = 10.0            # ARITHMETIC, not a fit: cycle-mean birth rate is 1.5k, mean = 1.5k/mu = 15
N_STATES = 91       # spec section 3 cost line: 91 states, i.e. n = 0..90
THRESH = 29         # spec 3.2 tail column: P(n >= 29)
N_SUB = 1000        # sub-steps per phase for the recorded phase average (see F-exact)

SPEC_TABLE = {
    0.1: (15.0000, 1.0003, 8.6265e-04, 1.00),
    1.0: (15.0000, 1.0339, 1.0547e-03, 1.23),
    3.0: (15.0000, 1.2553, 2.5740e-03, 2.99),
    10.0: (15.0000, 2.0094, 9.8847e-03, 11.48),
    100.0: (15.0000, 2.6042, 1.6477e-02, 19.14),
}
SWEEP = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0,
         10.0, 20.0, 30.0, 50.0, 100.0, 300.0, 1000.0]


# -------------------------------------------------------------------------------------
# generators and statistics
# -------------------------------------------------------------------------------------

def generator(birth: Callable[[int], float], death: Callable[[int], float], N: int) -> np.ndarray:
    """Dense generator L from the same off-diagonal triplets solver.py builds."""
    rows, cols, vals, _ = birth_death_arrays(birth, death, N)
    L = np.zeros((N, N))
    for i, j, v in zip(rows, cols, vals):
        L[i, j] += v
        L[i, i] -= v
    return L


def const_generator(rate: float, N: int = N_STATES, mu: float = MU) -> np.ndarray:
    return generator(lambda n: rate, lambda n, m=mu: m * n, N)


def tail(p: np.ndarray, thresh: int = THRESH) -> float:
    return float(np.asarray(p)[thresh:].sum())


def moments(p: np.ndarray) -> Tuple[float, float]:
    n = np.arange(len(p))
    m = float((p * n).sum())
    v = float((p * n * n).sum() - m * m)
    return m, v


def fano(p: np.ndarray) -> float:
    m, v = moments(p)
    return v / m


# -------------------------------------------------------------------------------------
# the Floquet solver -- the required implementation pattern, verbatim
# -------------------------------------------------------------------------------------

def floquet_cycle(L_phases: Sequence[np.ndarray], durations: Sequence[float],
                  n_sub: int = N_SUB) -> dict:
    """Periodic state and the phase-resolved distributions over one cycle.

    The fixed point is taken by LINEAR SOLVE, never by eigen-decomposition: eigen-solvers return
    an arbitrary sign, and for small T every eigenvalue of A clusters near 1 so the selection is
    ambiguous. eigen_route() below runs that forbidden path once, as the negative control.
    """
    N = L_phases[0].shape[0]
    S = [expm(L.T * (d / n_sub)) for L, d in zip(L_phases, durations)]
    A = np.eye(N)
    for Sk in S:
        A = np.linalg.matrix_power(Sk, n_sub) @ A

    B = A - np.eye(N)
    B[0, :] = 1.0
    b = np.zeros(N)
    b[0] = 1.0
    p = np.maximum(np.linalg.solve(B, b), 0.0)
    p = p / p.sum()

    # step through the cycle recording p at each phase (every sub-step is a phase sample)
    recs: List[np.ndarray] = []
    wts: List[float] = []
    times: List[float] = []
    q = p.copy()
    t = 0.0
    for Sk, d in zip(S, durations):
        dt = d / n_sub
        for _ in range(n_sub):
            recs.append(q)
            wts.append(dt)
            times.append(t)
            q = Sk @ q
            t += dt
    R = np.array(recs)
    w = np.array(wts)
    # uniform grid over a periodic cycle: this weighted average IS the composite trapezoid rule
    avg = (w[:, None] * R).sum(axis=0) / w.sum()
    neg = float(-np.minimum(avg, 0.0).sum())
    avg = np.maximum(avg, 0.0)
    avg = avg / avg.sum()
    return {
        "p0": p,
        "A": A,
        "records": R,
        "weights": w,
        "times": np.array(times),
        "avg": avg,
        "clipped_mass": neg,
        "period_resid": float(np.max(np.abs(q - p)) / np.max(p)),
        "closure_err": float(np.max(np.abs(R.sum(axis=1) - 1.0))),
    }


def eigen_route(A: np.ndarray) -> dict:
    """THE FORBIDDEN ROUTE, run so the prohibition is checkable. Written naively on purpose."""
    w, V = np.linalg.eig(A)
    k = int(np.argmin(np.abs(w - 1.0)))
    v = np.real(V[:, k])
    raw_negative = bool(v.sum() < 0)
    p = np.maximum(v, 0.0)
    s = p.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        p = p / s
    return {
        "p": p,
        "eigval": complex(w[k]),
        "clip_sum": float(s),
        "raw_mostly_negative": raw_negative,
        "nan": bool(not np.all(np.isfinite(p))),
        "gap": float(np.sort(np.abs(w - 1.0))[1]),
    }


# -------------------------------------------------------------------------------------
# the physical case
# -------------------------------------------------------------------------------------

def gene_case(T_over_tau: float, n_sub: int = N_SUB, rates: Tuple[float, float] = (K, 2 * K),
              N: int = N_STATES, mu: float = MU) -> dict:
    T = T_over_tau / mu                      # tau = 1/mu
    durations = [T / 2.0, T / 2.0]           # spec 3.2: 1x / 2x per HALF cycle
    L_phases = [const_generator(r, N, mu) for r in rates]
    out = floquet_cycle(L_phases, durations, n_sub)
    out["T_over_tau"] = T_over_tau
    m, v = moments(out["avg"])
    out["mean"] = m
    out["var"] = v
    out["fano"] = v / m
    out["tail"] = tail(out["avg"])
    return out


# -------------------------------------------------------------------------------------
# F-exact: the analytic route, no matrices
# -------------------------------------------------------------------------------------

def periodic_m(rates: Sequence[float], durations: Sequence[float], mu: float) -> List[float]:
    """Start-of-phase values of the periodic solution of dm/dt = lambda(t) - mu m."""
    D = 1.0
    c = 0.0
    for lam, d in zip(rates, durations):
        e = math.exp(-mu * d)
        c = lam / mu + (c - lam / mu) * e
        D *= e
    m0 = c / (1.0 - D)
    ms = [m0]
    for lam, d in zip(rates, durations):
        e = math.exp(-mu * d)
        ms.append(lam / mu + (ms[-1] - lam / mu) * e)
    return ms


def exact_stats(rates: Sequence[float], durations: Sequence[float], mu: float,
                N: int = N_STATES, thresh: int = THRESH) -> dict:
    """Exact cycle-averaged mean, Fano and tail from the closed-form m(t). No linear algebra."""
    ms = periodic_m(rates, durations, mu)
    Tc = float(sum(durations))
    I1 = 0.0
    I2 = 0.0
    tail_int = 0.0
    for j, (lam, d) in enumerate(zip(rates, durations)):
        mu_j = lam / mu
        c = ms[j] - mu_j
        e = math.exp(-mu * d)
        I1 += mu_j * d + c * (1.0 - e) / mu
        I2 += (mu_j ** 2) * d + 2.0 * mu_j * c * (1.0 - e) / mu + (c ** 2) * (1.0 - e * e) / (2 * mu)

        def f(t, mu_j=mu_j, c=c):
            return float(poisson_reference(mu_j + c * math.exp(-mu * t), N)[thresh:].sum())

        edges = [0.0]
        x = 0.25 / mu
        while x < d:
            edges.append(x)
            x *= 2.0
        edges.append(d)
        for a, bb in zip(edges[:-1], edges[1:]):
            tail_int += quad(f, a, bb, limit=200, epsabs=1e-16, epsrel=1e-11)[0]
    mean = I1 / Tc
    m2 = I2 / Tc
    var = mean + (m2 - mean * mean)          # E[Var(n|t)] + Var(E[n|t]), Poisson at every t
    return {"mean": mean, "fano": var / mean, "tail": tail_int / Tc, "m_start": ms}


# -------------------------------------------------------------------------------------
# verify
# -------------------------------------------------------------------------------------

def _row(gates, name, expected, measured, ok, void=False, note=""):
    v = "VOID" if void else ("PASS" if ok else "FAIL")
    gates.append({"name": name, "expected": expected, "measured": measured,
                  "verdict": v, "note": note})


def _wre(got, ref, floor):
    """Worst RELATIVE error over states where the reference exceeds `floor`."""
    m = ref > floor
    return float(np.max(np.abs(got[m] - ref[m]) / ref[m]))


def verify() -> dict:
    gates: List[dict] = []
    bar = "=" * 100

    # ---- references ---------------------------------------------------------------
    r_lo, c_lo, v_lo, _ = birth_death_arrays(lambda n: K, lambda n: MU * n, N_STATES)
    r_hi, c_hi, v_hi, _ = birth_death_arrays(lambda n: 2 * K, lambda n: MU * n, N_STATES)
    r_cn, c_cn, v_cn, _ = birth_death_arrays(lambda n: 1.5 * K, lambda n: MU * n, N_STATES)
    p_lo = stationary(r_lo, c_lo, v_lo, N_STATES, 0)
    p_hi = stationary(r_hi, c_hi, v_hi, N_STATES, 0)
    p_const = stationary(r_cn, c_cn, v_cn, N_STATES, 0)
    p_adia = 0.5 * p_lo + 0.5 * p_hi                     # duration-weighted, durations equal
    tail_const = tail(p_const)
    tail_adia = tail(p_adia)

    print(bar)
    print("SETUP.  mu = 1 (tau = 1), k = 10 -> rates 10 / 20 per half cycle, 91 states, P(n>=29)")
    print(bar)
    cap_needed = truncation_cap(THRESH, math.sqrt(2 * K))
    print("  truncation: solver.truncation_cap(T=29, sigma=sqrt(20)) = " + str(cap_needed)
          + " states needed; using " + str(N_STATES))
    print("  CONSTANT model  = stationary(birth 15, death n)   mean "
          + "{:.6f}".format(moments(p_const)[0]) + "  Fano " + "{:.6f}".format(fano(p_const))
          + "  P(n>=29) " + "{:.6e}".format(tail_const))
    print("  ADIABATIC avg   = 0.5*stat(10) + 0.5*stat(20)     mean "
          + "{:.6f}".format(moments(p_adia)[0]) + "  Fano " + "{:.6f}".format(fano(p_adia))
          + "  P(n>=29) " + "{:.6e}".format(tail_adia))
    print("  NOT the same object: the adiabatic average has Fano "
          + "{:.4f}".format(fano(p_adia)) + " against " + "{:.4f}".format(fano(p_const))
          + " for the stationary law of the MEAN rate; its tail is "
          + "{:.2f}".format(tail_adia / tail_const) + "x larger.")
    print("  adiabatic tail ratio (the T/tau -> inf bound) = "
          + "{:.4f}".format(tail_adia / tail_const) + "x")

    # ---- spec 3.2 table -----------------------------------------------------------
    print("\n" + bar)
    print("SPEC 3.2 TABLE, MEASURED BESIDE THE SPEC (n_sub = " + str(N_SUB) + " per phase)")
    print(bar)
    hdr = ("  {:>9s} {:>9s} {:>9s} {:>9s} {:>9s} {:>12s} {:>12s} {:>8s} {:>8s}"
           .format("T/tau", "mean", "spec", "Fano", "spec", "P(n>=29)", "spec", "ratio", "spec"))
    print(hdr)
    table = {}
    for Tt in sorted(SPEC_TABLE):
        r = gene_case(Tt)
        table[Tt] = r
        sm, sf, st, sr = SPEC_TABLE[Tt]
        line = ("  {:>9.2f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>12.4e} {:>12.4e} "
                "{:>7.2f}x {:>7.2f}x").format(Tt, r["mean"], sm, r["fano"], sf,
                                              r["tail"], st, r["tail"] / tail_const, sr)
        print(line)
    print("  relative difference against the spec's own numbers:")
    for Tt in sorted(SPEC_TABLE):
        r = table[Tt]
        sm, sf, st, sr = SPEC_TABLE[Tt]
        dl = ("    T/tau {:>7.2f}   mean {:+.2e}   Fano {:+.2e}   tail {:+.2e}"
              .format(Tt, r["mean"] / sm - 1.0, r["fano"] / sf - 1.0, r["tail"] / st - 1.0))
        print(dl)

    print("\n  THE SAME TABLE AT n_sub = 80 SUB-STEPS PER PHASE (160 phase samples per cycle):")
    print("  " + "{:>9s} {:>9s} {:>9s} {:>12s} {:>12s} {:>10s} {:>10s}".format(
        "T/tau", "Fano", "spec", "P(n>=29)", "spec", "relF", "relT"))
    worst80 = 0.0
    for Tt in sorted(SPEC_TABLE):
        r80 = gene_case(Tt, n_sub=80)
        sm, sf, st, sr = SPEC_TABLE[Tt]
        worst80 = max(worst80, abs(r80["fano"] / sf - 1.0), abs(r80["tail"] / st - 1.0))
        print("  " + "{:>9.2f} {:>9.4f} {:>9.4f} {:>12.4e} {:>12.4e} {:>+10.1e} {:>+10.1e}".format(
            Tt, r80["fano"], sf, r80["tail"], st, r80["fano"] / sf - 1.0, r80["tail"] / st - 1.0))
    print("  worst relative deviation from the spec table at n_sub=80: "
          + "{:.1e}".format(worst80) + "  -- every column, every row, 5 significant digits.")
    print("  The spec's section 3.2 table is therefore a 160-sample-per-cycle QUADRATURE of the")
    print("  phase average, not the converged cycle average. See the n_sub convergence study and")
    print("  the analytic route below for the converged numbers.")

    # ---- full sweep ---------------------------------------------------------------
    print("\n" + bar)
    print("FULL SWEEP, " + str(len(SWEEP)) + " values of T/tau in [0.01, 1000]  (G2.3, G2.4, F-monotone)")
    print(bar)
    print("  {:>9s} {:>10s} {:>10s} {:>13s} {:>9s} {:>11s} {:>11s}".format(
        "T/tau", "mean", "Fano", "P(n>=29)", "ratio", "period res", "closure"))
    sweep = []
    for Tt in SWEEP:
        r = gene_case(Tt)
        sweep.append(r)
        ln = "  {:>9.2f} {:>10.4f} {:>10.4f} {:>13.4e} {:>8.3f}x {:>11.2e} {:>11.2e}".format(
            Tt, r["mean"], r["fano"], r["tail"], r["tail"] / tail_const,
            r["period_resid"], r["closure_err"])
        print(ln)
    ratios = np.array([r["tail"] / tail_const for r in sweep])
    means = np.array([r["mean"] for r in sweep])

    # G2.3
    mean_err = float(np.max(np.abs(means - 15.0) / 15.0))
    _row(gates, "G2.3/T08", "mean within 0.01% of 15.0 at every T",
         "worst {:.2e} rel ({:.4f} vs 15.0000)".format(mean_err, means[int(np.argmax(np.abs(means - 15.0)))]),
         mean_err < 1e-4)

    # G2.4
    finite = all(np.all(np.isfinite(r["records"])) and np.all(np.isfinite(r["p0"]))
                 and np.isfinite(r["mean"]) and np.isfinite(r["fano"]) and np.isfinite(r["tail"])
                 for r in sweep)
    nbad = sum(int(np.sum(~np.isfinite(r["records"]))) for r in sweep)
    _row(gates, "G2.4", "0 non-finite over " + str(len(SWEEP)) + " T values",
         str(nbad) + " non-finite entries in " + str(sum(r["records"].size for r in sweep))
         + " recorded values", finite and nbad == 0)

    # F-period
    pres = float(max(r["period_resid"] for r in sweep))
    _row(gates, "F-period", "cycle closes, resid < 1e-10", "worst {:.2e}".format(pres), pres < 1e-10)

    # G2.1 with its discrimination check
    i001 = SWEEP.index(0.01)
    i01 = SWEEP.index(0.1)
    i10 = SWEEP.index(10.0)
    e_small = abs(sweep[i001]["tail"] / tail_const - 1.0)
    e_01 = abs(sweep[i01]["tail"] / tail_const - 1.0)
    e_disc = abs(sweep[i10]["tail"] / tail_const - 1.0)
    print("\n  G2.1/T06  fast-cycle limit against the CONSTANT model, in the tail")
    print("    T/tau=0.01 : rel err {:.3e}   T/tau=0.1 : rel err {:.3e}".format(e_small, e_01))
    print("    DISCRIMINATION at T/tau=10 : rel err {:.3e}  ({:.0f}x the bar)".format(
        e_disc, e_disc / 0.01))
    _row(gates, "G2.1/T06", "tail within 1% of constant model as T/tau->0",
         "{:.3e} rel at T/tau=0.01 ({:.3e} at 0.1)".format(e_small, e_01), e_small < 0.01)
    _row(gates, "G2.1-disc", "same test at T/tau=10 must MISS 1% by >2 orders",
         "{:.3e} rel = {:.0f}x the bar".format(e_disc, e_disc / 0.01), e_disc > 1.0)

    # G2.2 with its discrimination check
    i100 = SWEEP.index(100.0)
    e_ad = abs(sweep[i100]["tail"] / tail_adia - 1.0)
    e_ad_disc = abs(sweep[i01]["tail"] / tail_adia - 1.0)
    print("\n  G2.2/T07  slow-cycle limit against the ADIABATIC AVERAGE, in the tail")
    print("    T/tau=100  : measured {:.6e}  adiabatic {:.6e}  rel err {:.3e}".format(
        sweep[i100]["tail"], tail_adia, e_ad))
    print("    for contrast, the stationary law of the MEAN rate would give rel err {:.3e}"
          .format(abs(sweep[i100]["tail"] / tail_const - 1.0)))
    print("    DISCRIMINATION at T/tau=0.1 : rel err {:.3e} against the adiabatic average"
          .format(e_ad_disc))
    _row(gates, "G2.2/T07", "tail within 5% of adiabatic average at T/tau=100",
         "{:.3e} rel ({:.4e} vs {:.4e})".format(e_ad, sweep[i100]["tail"], tail_adia), e_ad < 0.05)
    _row(gates, "G2.2-disc", "same test at T/tau=0.1 must MISS the 5% bar",
         "{:.3e} rel".format(e_ad_disc), e_ad_disc > 0.05)

    # F-monotone
    diffs = np.diff(ratios)
    mono = bool(np.all(diffs >= -1e-9))
    no_interior_max = bool(int(np.argmax(ratios)) == len(ratios) - 1)
    r_adia = tail_adia / tail_const
    r_last = ratios[-1]
    sat = bool(r_last >= 0.95 * r_adia and r_last <= r_adia * (1.0 + 1e-9))
    print("\n  F-monotone  ratio sweep, increments and saturation")
    print("    increments (ratio[i+1]-ratio[i]): "
          + " ".join("{:+.2e}".format(d) for d in diffs))
    print("    argmax at index {:d} of {:d} (largest T = {:.0f})".format(
        int(np.argmax(ratios)), len(ratios) - 1, SWEEP[int(np.argmax(ratios))]))
    print("    ratio(T/tau=1000) = {:.4f}x   adiabatic bound = {:.4f}x   fraction {:.5f}".format(
        r_last, r_adia, r_last / r_adia))
    _row(gates, "F-monotone", "strictly climbing, no interior max, saturates <= adiabatic",
         "monotone={} argmax_at_end={} ratio(1000)/adiabatic={:.5f}".format(mono, no_interior_max,
                                                                           r_last / r_adia),
         mono and no_interior_max and sat)

    # ---- F-control ----------------------------------------------------------------
    print("\n" + bar)
    print("F-control  NEGATIVE CONTROL: both phase rates EQUAL (15 / 15), no switching")
    print(bar)
    ctrl_ratio_err = 0.0
    ctrl_wre = 0.0
    print("  {:>9s} {:>13s} {:>10s} {:>12s} {:>12s}".format(
        "T/tau", "P(n>=29)", "ratio", "|ratio-1|", "worst rel"))
    for Tt in SWEEP:
        rc = gene_case(Tt, rates=(1.5 * K, 1.5 * K))
        rr = rc["tail"] / tail_const
        w = _wre(rc["avg"], p_const, 1e-10)
        ctrl_ratio_err = max(ctrl_ratio_err, abs(rr - 1.0))
        ctrl_wre = max(ctrl_wre, w)
        print("  {:>9.2f} {:>13.6e} {:>9.5f}x {:>12.2e} {:>12.2e}".format(
            Tt, rc["tail"], rr, abs(rr - 1.0), w))
    _row(gates, "F-control", "tail ratio = 1.000 at every T, |ratio-1| < 1e-6",
         "worst |ratio-1| = {:.2e}".format(ctrl_ratio_err), ctrl_ratio_err < 1e-6)
    _row(gates, "F-control-p", "worst rel err vs stationary() over p>1e-10 < 1e-5",
         "worst {:.2e}".format(ctrl_wre), ctrl_wre < 1e-5)

    # ---- F-swap -------------------------------------------------------------------
    print("\n" + bar)
    print("F-swap  SECOND CONTROL: phase order [2k, k] must give the same cycle average")
    print(bar)
    swap_worst = 0.0
    swap_tail_worst = 0.0
    swap_abs_worst = 0.0
    for Tt in (0.1, 1.0, 10.0, 100.0):
        ra = gene_case(Tt)
        rb = gene_case(Tt, rates=(2 * K, K))
        a, b = ra["avg"], rb["avg"]
        w = _wre(b, a, 1e-10)
        tw = abs(rb["tail"] / ra["tail"] - 1.0)
        aw = float(np.max(np.abs(a - b)))
        swap_worst = max(swap_worst, w)
        swap_tail_worst = max(swap_tail_worst, tw)
        swap_abs_worst = max(swap_abs_worst, aw)
        print("  T/tau {:>7.2f}   worst rel over p>1e-10 {:.3e}   max ABS diff {:.2e}   "
              "P(n>=29) rel diff {:.2e}".format(Tt, w, aw, tw))
    print("  IS THAT AN ASYMMETRY OR THE PRECISION FLOOR? worst relative difference at T/tau=0.1")
    print("  as the comparison floor is lowered (if it were an asymmetry it would stay put; if it")
    print("  is one fixed absolute wobble divided by a smaller p, it grows and then stops):")
    aa = gene_case(0.1)["avg"]
    bb = gene_case(0.1, rates=(2 * K, K))["avg"]
    for fl in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        m = aa > fl
        deep = int(np.flatnonzero(m)[-1])
        print("    floor {:>7.0e}  worst rel {:.3e}  deepest state n={:d} at p={:.2e}".format(
            fl, _wre(bb, aa, fl), deep, aa[deep]))
    _row(gates, "F-swap", "phase-order swap identical, worst rel < 1e-8 over p>1e-10",
         "worst {:.2e} (FAILS as declared)".format(swap_worst), swap_worst < 1e-8, void=True,
         note=("bar sits an order below its own a priori noise floor (~1e-7 at p=1e-10): "
               "max ABS difference is {:.1e}, so the ratio is noise/p, not an asymmetry"
               ).format(swap_abs_worst))
    _row(gates, "F-swap-tail", "POST-HOC repair, achievable form: phase order must not move "
         "P(n>=29); bar 1e-9",
         "worst {:.2e}".format(swap_tail_worst), swap_tail_worst < 1e-9)

    # ---- F-exact ------------------------------------------------------------------
    print("\n" + bar)
    print("F-exact  INDEPENDENT ANALYTIC ROUTE: p(t) is exactly Poisson(m(t)), no matrices")
    print(bar)
    print("  {:>9s} {:>12s} {:>12s} {:>10s} {:>12s} {:>12s} {:>10s}".format(
        "T/tau", "Fano solve", "Fano exact", "rel", "tail solve", "tail exact", "rel"))
    ex_fano_err = 0.0
    ex_tail_err = 0.0
    for Tt in sorted(SPEC_TABLE):
        T = Tt / MU
        ex = exact_stats([K, 2 * K], [T / 2, T / 2], MU)
        r = table[Tt]
        ef = abs(r["fano"] / ex["fano"] - 1.0)
        et = abs(r["tail"] / ex["tail"] - 1.0)
        ex_fano_err = max(ex_fano_err, ef)
        ex_tail_err = max(ex_tail_err, et)
        print("  {:>9.2f} {:>12.6f} {:>12.6f} {:>10.2e} {:>12.6e} {:>12.6e} {:>10.2e}".format(
            Tt, r["fano"], ex["fano"], ef, r["tail"], ex["tail"], et))
    _row(gates, "F-exact-F", "Fano matches analytic route to < 1e-3 rel",
         "worst {:.2e}".format(ex_fano_err), ex_fano_err < 1e-3)
    _row(gates, "F-exact-T", "tail matches analytic route to < 1e-3 rel",
         "worst {:.2e}".format(ex_tail_err), ex_tail_err < 1e-3)

    # phase-resolved check against Poisson(m(t)) at a handful of phases
    print("\n  phase-resolved distributions against Poisson(m(t)), T/tau = 10, 12 phases:")
    T = 10.0 / MU
    ex = exact_stats([K, 2 * K], [T / 2, T / 2], MU)
    r12 = gene_case(10.0, n_sub=6)          # 6 sub-steps x 2 phases = 12 recorded phases
    worst_phase = 0.0
    for idx in range(12):
        t = r12["times"][idx]
        j = 0 if t < T / 2 else 1
        lam = [K, 2 * K][j]
        t0 = t - j * (T / 2)
        m_t = lam / MU + (ex["m_start"][j] - lam / MU) * math.exp(-MU * t0)
        ref = poisson_reference(m_t, N_STATES)
        w = _wre(r12["records"][idx], ref, 1e-10)
        worst_phase = max(worst_phase, w)
        print("    phase {:>2d}  t = {:>7.3f}  m(t) = {:>8.5f}  worst rel err {:.3e}".format(
            idx, t, m_t, w))
    _row(gates, "F-exact-phase", "every phase equals Poisson(m(t)), worst rel < 1e-5 over p>1e-10",
         "worst {:.2e}".format(worst_phase), worst_phase < 1e-5)

    # n_sub convergence -- where do the spec's Fano digits come from?
    print("\n  n_sub convergence of the recorded phase average at T/tau = 100")
    print("    (spec quotes Fano 2.6042; the analytic cycle average is the reference)")
    exs = exact_stats([K, 2 * K], [50.0, 50.0], MU)
    for ns in (6, 12, 40, 80, 160, 320, 1000, 4000):
        rr = gene_case(100.0, n_sub=ns)
        print("    n_sub {:>5d} per phase  Fano {:.6f}  (analytic {:.6f}, rel {:+.2e})"
              " tail {:.6e}".format(ns, rr["fano"], exs["fano"], rr["fano"] / exs["fano"] - 1.0,
                                    rr["tail"]))

    # ---- F-eigen ------------------------------------------------------------------
    print("\n" + bar)
    print("F-eigen  THE FORBIDDEN ROUTE, run once per T so the prohibition is checkable")
    print(bar)
    print("  {:>9s} {:>12s} {:>11s} {:>10s} {:>8s} {:>12s}".format(
        "T/tau", "eigval", "2nd |w-1|", "clip sum", "NaN", "tail rel err"))
    n_nan = 0
    eig_worst = 0.0
    for Tt, r in zip(SWEEP, sweep):
        e = eigen_route(r["A"])
        n_nan += int(e["nan"])
        if e["nan"]:
            terr = float("nan")
        else:
            terr = abs(tail(e["p"]) / tail(r["p0"]) - 1.0)
            eig_worst = max(eig_worst, terr)
        print("  {:>9.2f} {:>12.8f} {:>11.2e} {:>10.2e} {:>8s} {:>12.3e}".format(
            Tt, e["eigval"].real, e["gap"], e["clip_sum"], str(e["nan"]), terr))
    unsafe = (n_nan > 0) or (eig_worst > 1e-3)
    print("  {:d} of {:d} sweep points returned ALL-NaN: the eigenvector came back with the sign"
          .format(n_nan, len(SWEEP)))
    print("  flipped, so max(v,0) is the all-zero vector and v/v.sum() is 0/0. The eigenvalue is")
    print("  1.00000000 every time and the second eigenvalue sits {:.1e} away at T/tau=0.01, so"
          .format(min(eigen_route(sweep[0]["A"])["gap"], 1.0)))
    print("  the failure is NOT bad eigenvalue selection -- it is the arbitrary sign, exactly as")
    print("  the spec says. Where the sign happens to come out positive the vector is correct to")
    print("  {:.1e} in the tail, which is what makes the route so dangerous: it is right about"
          .format(eig_worst))
    print("  half the time and silently unusable the other half.")
    _row(gates, "F-eigen", "forbidden route must be demonstrably unsafe somewhere",
         "{:d}/{:d} sweep points ALL-NaN; worst tail rel err {:.1e} on the {:d} that survived"
         .format(n_nan, len(SWEEP), eig_worst, len(SWEEP) - n_nan), unsafe)

    # ---- cost ---------------------------------------------------------------------
    print("\n" + bar)
    print("COST  one gene, 12 phases, 91 states   (spec: 5.0 ms)")
    print(bar)
    Tc = 10.0
    rate12 = [K if j < 6 else 2 * K for j in range(12)]
    L12 = [const_generator(r) for r in rate12]
    d12 = [Tc / 12.0] * 12
    M12 = [L.T * d for L, d in zip(L12, d12)]

    def best_ms(fn, warm=3, rep=10):
        for _ in range(warm):
            fn()
        out = float("inf")
        for _ in range(rep):
            t0 = time.perf_counter()
            fn()
            out = min(out, time.perf_counter() - t0)
        return out * 1e3

    def cached_pipeline():
        """Not the required pattern: reuses expm across identical generators."""
        seen = {}
        S = []
        for L, d in zip(L12, d12):
            key = (float(L[0, 1]), float(d))
            if key not in seen:
                seen[key] = expm(L.T * d)
            S.append(seen[key])
        A = np.eye(N_STATES)
        for Sk in S:
            A = Sk @ A
        B = A - np.eye(N_STATES)
        B[0, :] = 1.0
        bb = np.zeros(N_STATES)
        bb[0] = 1.0
        p = np.maximum(np.linalg.solve(B, bb), 0.0)
        p = p / p.sum()
        q = p
        for Sk in S:
            q = Sk @ q
        return p

    t_default = best_ms(lambda: floquet_cycle(L12, d12, n_sub=1))
    t1 = t1_expm = t1_one = t1_solve = t1_cached = float("nan")
    if _HAVE_TPC:
        with threadpool_limits(limits=1):
            t1 = best_ms(lambda: floquet_cycle(L12, d12, n_sub=1))
            t1_expm = best_ms(lambda: [expm(M) for M in M12])
            t1_one = best_ms(lambda: expm(M12[0]), rep=20)
            t1_solve = best_ms(lambda: np.linalg.solve(np.eye(N_STATES) + M12[0],
                                                       np.ones(N_STATES)), rep=20)
            t1_cached = best_ms(cached_pipeline)
    print("  REQUIRED PATTERN (12 separate expm calls), best of 10 warmed runs:")
    print("    default BLAS threads ({:d} cores)  : {:>8.2f} ms".format(os.cpu_count() or 0,
                                                                      t_default))
    print(("    1 BLAS thread                   : {:>8.2f} ms   <- FASTER: 91x91 is too small "
           "to").format(t1))
    print("                                                    thread, the pool oversubscribes")
    print("  breakdown at 1 BLAS thread (the stable setting; the threaded timings scatter by 2x")
    print("  run to run, so a breakdown taken there does not add up):")
    print("    12 expm(91x91) calls            : {:>8.2f} ms   ({:.0f}% of the pipeline)".format(
        t1_expm, 100.0 * t1_expm / t1))
    print("    one expm(91x91)                 : {:>8.3f} ms".format(t1_one))
    print("    one 91x91 dense solve           : {:>8.3f} ms".format(t1_solve))
    print("    cycle product + stepping (rest) : {:>8.2f} ms".format(t1 - t1_expm))
    print(("    cached expm (2 distinct only)   : {:>8.2f} ms   <- NOT the required pattern, "
           "but").format(t1_cached))
    print("                                                    the only way under the 5 ms claim")
    fastest = min(t_default, t1)
    print("  spec claim                        : {:>8.2f} ms".format(5.0))
    print("  measured / spec (fastest of the two thread settings): {:.1f}x".format(fastest / 5.0))
    _row(gates, "COST", "5.0 ms for 1 gene, 12 phases, 91 states",
         "{:.1f} ms (best of 10, 1 BLAS thread; {:.1f} ms at default threads)".format(t1, t_default),
         fastest <= 5.0,
         note="12 expm calls alone are {:.1f} ms, already {:.1f}x the whole spec budget".format(
             t1_expm, t1_expm / 5.0))

    # ---- summary ------------------------------------------------------------------
    print("\n" + bar)
    print("GATE SUMMARY   EXPECTED (from the spec) | MEASURED | VERDICT")
    print(bar)
    for g in gates:
        print("  {:<14s} {:<66s} {:<54s} {}".format(
            g["name"], g["expected"][:66], g["measured"][:54], g["verdict"]))
        if g.get("note"):
            print("  {:<14s} ^ {}".format("", g["note"]))
    nf = sum(1 for g in gates if g["verdict"] == "FAIL")
    nv = sum(1 for g in gates if g["verdict"] == "VOID")
    print("\n  {:d} PASS, {:d} FAIL, {:d} VOID, of {:d} gates".format(
        len(gates) - nf - nv, nf, nv, len(gates)))
    return {"gates": gates, "sweep": sweep, "table": table}


if __name__ == "__main__":
    verify()
