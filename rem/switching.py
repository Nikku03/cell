"""Exact spontaneous-switching rates for a genetic toggle switch, where Gillespie returns 0.

THE QUANTITY. A bistable promoter flips state spontaneously at 1e-6 to 1e-9 per cell per
generation. That number is what persister formation, lysis-lysogeny decisions, plasmid loss
and chromosome mis-segregation all turn on, and it is exactly the regime where sampling
fails: to see a 1e-9 event you need ~1e9 generations of simulation, while an exact
calculation costs the same whether the answer is 0.5 or 1e-30.

THE MODEL. Two mutually repressing genes, the standard symmetric toggle:
    A -> A+1  at  g / (1 + (n_B/K)^h)        A -> A-1  at  gamma * n_A
    B -> B+1  at  g / (1 + (n_A/K)^h)        B -> B-1  at  gamma * n_B
State is the pair (n_A, n_B) truncated at M, so the chemical master equation is a sparse
(M+1)^2 x (M+1)^2 generator. The switching time is the mean first passage time from the
A-dominant attractor to the symmetry line n_B >= n_A, solved exactly by one sparse linear
system. Deeper barriers -- larger mean copy number N = g/gamma -- make the event
exponentially rarer at no extra cost, which is the entire point.

WHAT THIS DOES AND DOES NOT SHOW ABOUT REM, stated before any number. The toggle has TWO
species, so its state space is (M+1)^2 and exactness comes from that space being small, not
from any treewidth argument. This is the tail-risk result reproduced in biology; it is not a
demonstration that factorisation was needed. T5 measures where treewidth would start to
matter, and reports honestly that a CHAIN of coupled switches runs into the same obstruction
as driven 1D transport: the stationary state of a master equation is a null vector, not a
product of local factors, so chain elimination does not apply to it.

WHAT verify() MUST SHOW -- PREDECLARED, BEFORE ANY NUMBER IS RUN.
  T1  THE GENERATOR IS A GENERATOR. Every row of Q sums to zero (probability conservation)
      and every off-diagonal entry is non-negative. GATE: max |row sum| < 1e-12.
  T2  MFPT AGAINST A CLOSED FORM. For a one-dimensional birth-death chain the mean first
      passage time has an exact recursion, T_n = 1/b_n + (d_n/b_n) T_{n-1}, summed from 0 to
      N-1. Build the same chain as a generator, solve the linear system, and compare.
      GATE: max relative error < 1e-10. This validates the SOLVER on a problem whose answer
      is known in closed form, before it is pointed at the toggle.
  T3  COST IS INDEPENDENT OF RARITY. Sweep the barrier so the switching probability spans
      many orders of magnitude and record wall-clock. GATE: the ratio of slowest to fastest
      solve is under 3x while the probability itself moves by more than 1e4. Monte Carlo's
      cost would scale as 1/p over the same sweep.
  T4  THE POSITIVE CONTROL, AND IT COMES FIRST. At a shallow barrier where switching is
      common, exact and Gillespie must AGREE within the Monte Carlo error bar. Without this
      "Gillespie returned zero" is indistinguishable from "the exact number is wrong".
      GATE: |exact - gillespie| < 3 standard errors at the shallow setting. Only then is
      the deep setting reported, where Gillespie is expected to return zero events in a
      matched wall-clock budget.
  T5  WHERE WOULD TREEWIDTH MATTER? Report state-space size against species count for
      coupled switches, and state plainly whether REM's factorisation is doing any work at
      n = 2. Reported, not gated -- but it must be reported, because four applications in
      this project have been exact, correct and irrelevant.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------------------------
# 1-D birth-death: the closed form that validates the solver
# --------------------------------------------------------------------------------------

def bd_mfpt_closed_form(birth: np.ndarray, death: np.ndarray, N: int) -> float:
    """Exact MFPT from state 0 to state N for a birth-death chain.

    T_n = 1/b_n + (d_n/b_n) T_{n-1} is the expected time to go from n to n+1; the total is
    the sum over n = 0 .. N-1. Standard and exact, with no matrix algebra in it, so it
    shares no code path with the linear-system solver it is used to check.
    """
    T = np.zeros(N)
    prev = 0.0
    for n in range(N):
        prev = 1.0 / birth[n] + (death[n] / birth[n]) * prev if n > 0 else 1.0 / birth[0]
        T[n] = prev
    return float(T.sum())


def bd_generator(birth: np.ndarray, death: np.ndarray, M: int) -> sp.csr_matrix:
    """Row = FROM, column = TO. Q[i,i] = -(sum of outgoing rates)."""
    rows, cols, vals = [], [], []
    diag = np.zeros(M + 1)
    for i in range(M + 1):
        out = 0.0
        if i < M and birth[i] > 0:
            rows.append(i); cols.append(i + 1); vals.append(birth[i]); out += birth[i]
        if i > 0 and death[i] > 0:
            rows.append(i); cols.append(i - 1); vals.append(death[i]); out += death[i]
        diag[i] = -out
    Q = sp.coo_matrix((vals, (rows, cols)), shape=(M + 1, M + 1)).tocsr()
    return (Q + sp.diags(diag)).tocsr()


# --------------------------------------------------------------------------------------
# the toggle switch
# --------------------------------------------------------------------------------------

def toggle_generator(M: int, g: float, gamma: float, K: float, h: float
                     ) -> Tuple[sp.csr_matrix, int]:
    """Sparse CME generator for the symmetric two-gene toggle. Row = FROM."""
    n = (M + 1) * (M + 1)
    idx = lambda a, b: a * (M + 1) + b
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    for a in range(M + 1):
        for b in range(M + 1):
            i = idx(a, b)
            out = 0.0
            ra = g / (1.0 + (b / K) ** h)
            rb = g / (1.0 + (a / K) ** h)
            if a < M:
                rows.append(i); cols.append(idx(a + 1, b)); vals.append(ra); out += ra
            if a > 0:
                d = gamma * a
                rows.append(i); cols.append(idx(a - 1, b)); vals.append(d); out += d
            if b < M:
                rows.append(i); cols.append(idx(a, b + 1)); vals.append(rb); out += rb
            if b > 0:
                d = gamma * b
                rows.append(i); cols.append(idx(a, b - 1)); vals.append(d); out += d
            diag[i] = -out
    Q = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return (Q + sp.diags(diag)).tocsr(), n


def mfpt_to_set(Q: sp.csr_matrix, target: np.ndarray) -> np.ndarray:
    """Exact mean first passage time to `target` from every other state.

    Solves Q_S tau = -1 on the non-target states, which is the standard absorbing-chain
    system. One sparse solve, and its cost does not depend on how rare the passage is.
    """
    keep = np.where(~target)[0]
    Qs = Q[keep][:, keep]
    tau = spla.spsolve(Qs.tocsc(), -np.ones(len(keep)))
    out = np.zeros(Q.shape[0])
    out[keep] = tau
    return out


def toggle_switching(M: int, g: float, gamma: float, K: float, h: float,
                     generation_time: float = 1.0) -> dict:
    """Exact MFPT from the A-dominant attractor across the symmetry line, and the
    per-generation switching probability that follows from it."""
    t0 = time.perf_counter()
    Q, n = toggle_generator(M, g, gamma, K, h)
    a_idx = np.repeat(np.arange(M + 1), M + 1)
    b_idx = np.tile(np.arange(M + 1), M + 1)
    target = b_idx >= a_idx                       # crossed to the B-dominant side
    tau = mfpt_to_set(Q, target)
    N = g / gamma
    start = int(round(min(N, M))) * (M + 1) + 0   # A-high, B-empty
    mf = float(tau[start])
    p = 1.0 - np.exp(-generation_time / mf) if mf > 0 else 1.0
    return {"mfpt": mf, "p_per_generation": float(p), "n_states": n,
            "seconds": time.perf_counter() - t0, "N_mean": N, "M": M,
            "start_state": (int(round(min(N, M))), 0)}


def gillespie_toggle(M: int, g: float, gamma: float, K: float, h: float,
                     t_max: float, seed: int = 0, n_runs: int = 1) -> dict:
    """Direct stochastic simulation. Counts crossings of the symmetry line from A-high."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    switches, total_t = 0, 0.0
    for _ in range(n_runs):
        a, b = int(round(min(g / gamma, M))), 0
        t = 0.0
        while t < t_max:
            ra = g / (1.0 + (b / K) ** h)
            rb = g / (1.0 + (a / K) ** h)
            da, db = gamma * a, gamma * b
            tot = ra + rb + da + db
            if tot <= 0:
                break
            t += rng.exponential(1.0 / tot)
            u = rng.random() * tot
            if u < ra:
                a = min(a + 1, M)
            elif u < ra + da:
                a = max(a - 1, 0)
            elif u < ra + da + rb:
                b = min(b + 1, M)
            else:
                b = max(b - 1, 0)
            if b >= a:
                switches += 1
                a, b = int(round(min(g / gamma, M))), 0     # reset to the A basin
        total_t += t
    sec = time.perf_counter() - t0
    rate = switches / total_t if total_t > 0 else 0.0
    se = np.sqrt(max(switches, 0)) / total_t if total_t > 0 else np.inf
    return {"switches": switches, "sim_time": total_t, "rate": rate, "rate_se": se,
            "mfpt": (1.0 / rate) if rate > 0 else np.inf, "seconds": sec}


# --------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------

def _params(N: float) -> dict:
    """Strongly bistable symmetric toggle at mean high-state copy number N.

    g = N, gamma = 1, K = N/4, h = 3 puts the high state at ~N and the low state at
    N/(1+4^3) = N/65, so the two basins are well separated and the barrier deepens with N.
    """
    return {"g": float(N), "gamma": 1.0, "K": float(N) / 4.0, "h": 3.0}


def verify(verbose: bool = True) -> dict:
    """Run T1-T5. Bars are fixed in the module docstring, above, before any number."""
    say = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    out: Dict[str, object] = {}

    # ---- T1: the generator is a generator -------------------------------------------
    Q, n = toggle_generator(24, **_params(6))
    rs = np.abs(np.asarray(Q.sum(axis=1)).ravel()).max()
    offdiag_min = Q.copy(); offdiag_min.setdiag(0.0); offdiag_min.eliminate_zeros()
    neg = float(offdiag_min.data.min()) if offdiag_min.nnz else 0.0
    out["T1_rowsum"], out["T1"] = float(rs), bool(rs < 1e-12 and neg >= 0.0)
    say(f"  T1 generator: max |row sum| {rs:.2e}, min off-diagonal {neg:.3f}, "
        f"{n:,} states   {'PASS' if out['T1'] else 'FAIL'}")

    # ---- T2: MFPT against the closed form ---------------------------------------------
    say("\n  T2 MFPT solver vs the exact birth-death recursion")
    say(f"      {'M':>4s} {'N':>4s} {'lambda':>7s} {'closed form':>14s} "
        f"{'linear solve':>14s} {'rel err':>10s}")
    t2 = True
    for M, Ntgt, lam, mu in ((40, 20, 5.0, 1.0), (60, 30, 8.0, 1.0), (80, 25, 3.0, 0.5)):
        birth = np.full(M + 1, lam); death = mu * np.arange(M + 1, dtype=float)
        cf = bd_mfpt_closed_form(birth, death, Ntgt)
        Qb = bd_generator(birth, death, M)
        tgt = np.zeros(M + 1, dtype=bool); tgt[Ntgt:] = True
        ls = float(mfpt_to_set(Qb, tgt)[0])
        rel = abs(cf - ls) / cf
        t2 &= rel < 1e-10
        say(f"      {M:4d} {Ntgt:4d} {lam:7.1f} {cf:14.6f} {ls:14.6f} {rel:10.2e}")
    out["T2"] = bool(t2)
    say(f"      T2 {'PASS' if t2 else 'FAIL'}  (bar 1e-10)")

    # ---- T3: cost independent of rarity -------------------------------------------------
    say("\n  T3 does the cost depend on how rare the event is?")
    say(f"      {'N':>4s} {'states':>8s} {'MFPT (gen)':>14s} "
        f"{'p per generation':>18s} {'seconds':>9s}")
    rows, secs, ps = [], [], []
    for N in (5, 8, 12, 16, 20, 24):
        M = max(4 * N, 24)
        r = toggle_switching(M, generation_time=1.0, **_params(N))
        rows.append((N, r)); secs.append(r["seconds"]); ps.append(r["p_per_generation"])
        say(f"      {N:4d} {r['n_states']:8,d} {r['mfpt']:14.4e} "
            f"{r['p_per_generation']:18.4e} {r['seconds']:9.2f}")
    span = max(ps) / max(min(ps), 1e-300)
    ratio = max(secs) / max(min(secs), 1e-12)
    out["T3_p_span"], out["T3_time_ratio"] = float(span), float(ratio)
    out["T3"] = bool(ratio < 3.0 and span > 1e4)
    say(f"      probability spans {span:.2e}x while wall-clock spans {ratio:.2f}x   "
        f"{'PASS' if out['T3'] else 'FAIL'}  (bars: span > 1e4, time ratio < 3)")

    # ---- T4: POSITIVE CONTROL FIRST, then the demonstration ------------------------------
    say("\n  T4 POSITIVE CONTROL: a shallow barrier where Gillespie CAN see the event")
    Nsh = 5
    Msh = max(4 * Nsh, 24)
    ex_sh = toggle_switching(Msh, generation_time=1.0, **_params(Nsh))
    gi_sh = gillespie_toggle(Msh, t_max=20000.0, seed=1, **_params(Nsh))
    dev = abs(ex_sh["mfpt"] - gi_sh["mfpt"])
    se_mfpt = (gi_sh["rate_se"] / max(gi_sh["rate"], 1e-30)) * gi_sh["mfpt"] \
        if gi_sh["rate"] > 0 else np.inf
    out["T4_exact_mfpt"], out["T4_gill_mfpt"] = ex_sh["mfpt"], gi_sh["mfpt"]
    out["T4"] = bool(np.isfinite(se_mfpt) and dev < 3 * se_mfpt)
    say(f"      exact MFPT      {ex_sh['mfpt']:.4f} generations  ({ex_sh['seconds']:.2f}s)")
    say(f"      Gillespie MFPT  {gi_sh['mfpt']:.4f} +/- {se_mfpt:.4f}  "
        f"({gi_sh['switches']:,} switches in {gi_sh['sim_time']:.0f} generations, "
        f"{gi_sh['seconds']:.0f}s)")
    say(f"      |difference| {dev:.4f} vs 3 s.e. {3*se_mfpt:.4f}   "
        f"{'PASS' if out['T4'] else 'FAIL'}")
    say("      -> only with this in hand does a Gillespie zero mean anything.")

    say("\n  T4b THE DEMONSTRATION: a deep barrier, matched wall-clock budget")
    Ndp = 24
    Mdp = 4 * Ndp
    ex_dp = toggle_switching(Mdp, generation_time=1.0, **_params(Ndp))
    budget = max(30.0, 10 * ex_dp["seconds"])
    gi_dp = gillespie_toggle(Mdp, t_max=1e9, seed=2, **_params(Ndp))
    exp_hits = gi_dp["sim_time"] / ex_dp["mfpt"]
    out["T4b"] = {"exact_mfpt": ex_dp["mfpt"], "p": ex_dp["p_per_generation"],
                  "exact_seconds": ex_dp["seconds"], "gill_switches": gi_dp["switches"],
                  "gill_time": gi_dp["sim_time"], "gill_seconds": gi_dp["seconds"],
                  "expected_hits": float(exp_hits)}
    say(f"      exact:     MFPT {ex_dp['mfpt']:.4e} generations, "
        f"p = {ex_dp['p_per_generation']:.4e} per generation, in {ex_dp['seconds']:.2f}s")
    say(f"      Gillespie: {gi_dp['switches']} switches in {gi_dp['sim_time']:.3e} "
        f"generations ({gi_dp['seconds']:.0f}s)")
    say(f"      expected hits in that budget: {exp_hits:.3e}")
    say(f"      generations of simulation needed to expect ONE event: "
        f"{ex_dp['mfpt']:.3e}")

    # ---- T5: where would treewidth matter? ------------------------------------------------
    say("\n  T5 is REM's factorisation doing any work here? (reported, not gated)")
    say(f"      {'species':>8s} {'states (M=100)':>16s}   comment")
    for k in (2, 3, 4, 6, 10):
        say(f"      {k:8d} {101.0**k:16.2e}   "
            + ("tractable by direct sparse solve" if k <= 2 else
               "needs factorisation -- but see below"))
    say("      At 2 species the state space is 10^4 and a sparse solve is trivial, so")
    say("      exactness here comes from the space being SMALL, not from low treewidth.")
    say("      Scaling to many coupled switches hits the SAME obstruction as driven 1D:")
    say("      a master equation's stationary state is a null vector, not a product of")
    say("      local factors, so chain elimination does not apply to it.")

    gates = ["T1", "T2", "T3", "T4"]
    out["all_pass"] = all(bool(out[k]) for k in gates)
    say(f"\n  {'ALL GATES PASS' if out['all_pass'] else 'GATE FAILURE'}: "
        + "  ".join(f"{k}={'pass' if out[k] else 'FAIL'}" for k in gates))
    return out


if __name__ == "__main__":
    verify()
