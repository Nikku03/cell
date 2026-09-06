"""LOOP 149 -- A MARKOV CHAIN ON THE UBIQUITIN CHAIN: CAN A MULTISTEP PROCESS AMPLIFY 1.5x INTO 20.3x?

WHAT LOOP 148 LEFT AND WHY A MARKOV CHAIN IS THE RIGHT TOOL FOR IT. Three stages of the destruction
machine were measured and the smallest binds: the 26S has 57x-170x of throughput headroom, the
per-substrate rate envelope spans 10.7x at p99, and the specificity subunits -- the F-box proteins
and APC/C activators that pick the substrate -- have a median cycle fold-range of 1.51x, BELOW their
abundance-matched null. The pulse equation needs 20.29x. So if b(t) were set by how much receptor is
present, the pulse could not be built, and loop 148 said so.

That is an AMPLIFICATION question, and it is the one thing in this arc a Markov chain answers
exactly. A Markov chain cannot say which proteins are pulsed -- nine mechanisms have been eliminated
on that question and none of them was a modelling failure -- and this loop does not touch it.

THE CHAIN. Ubiquitin is not added once. It is added one at a time to build a K48 chain, DUBs remove
it one at a time, and the proteasome engages only past a threshold length. That is a birth-death
process on chain length i = 0..n with an absorbing state, and the per-molecule loss rate b is
exactly the reciprocal of the mean first passage time from 0 to absorption:

    up-rate    lambda = k_u * R      (E3-catalysed, proportional to ACTIVE RECEPTOR R)
    down-rate  mu     = k_d          (DUB-catalysed)
    absorbing  at i = n, then engagement at k_p
    b = 1 / ( MFPT(0 -> n) + 1/k_p )

With rho = mu/lambda the MFPT has a closed form,

    T(0->n) = [ n - rho*(1 - rho^n)/(1 - rho) ] / ( lambda * (1 - rho) )        (rho != 1)
    T(0->n) = n(n+1) / (2*lambda)                                               (rho == 1)

WHY THIS COULD CLOSE THE GAP. For rho >> 1 -- a resting state where DUBs outrun the E3 -- the sum is
dominated by its last term and T ~ rho^(n-1)/lambda, so with lambda proportional to R,

    b  ~  R^n

An n-step chain turns an r-fold change in receptor into an r^n-fold change in degradation rate. The
whole gap loop 148 measured is 20.29/1.51, and if the exponent is n rather than 1 that gap closes at
a chain length the literature already says is real. For rho << 1 the DUBs are irrelevant, T = n/lambda
exactly, and the exponent collapses to 1 with no amplification at all. So the mechanism makes a hard
falsifiable prediction about the RESTING regime rather than fitting anything.

WHAT WAS ALREADY CHECKED AT THE CONSOLE BEFORE THIS MODULE WAS WRITTEN, said plainly so M0 is not
dressed up as a discovery: the closed form was verified against the exact recursion t_i = 1/lambda +
rho*t_(i-1), and the two agree to EXACTLY zero over n=1..12 and rho=0.05..20. A third arm, solving
the absorbing generator by linear algebra, disagreed by up to 9e-3 -- and that is the LINEAR SOLVE
being wrong, not the formula: cond(A) reaches 9.95e15 at n=12, rho=20, which is float64's limit. The
exponent table was also computed there and runs 1.00 at rho=0.1 to 7.98 at rho=100 for n=8. M0 and
M1 below re-run both as regressions. Everything from M2 down is new.

PREDECLARED:

  M0 DOES THE ARITHMETIC CHECK THE ARITHMETIC?                       THE REGRESSION.
       closed form against the exact recursion over the full grid, and against the linear-algebra
       solve of the absorbing generator wherever cond(A) < 1e12. Gate: agreement to 1e-10 against
       the recursion, 1e-6 against linear algebra in the well-conditioned region, and the two known
       limits exact -- T = n/lambda at rho = 0, T = 1/lambda at n = 1 for every rho.

  M1 IS THE EXPONENT REALLY n?                                       THE MECHANISM CHECK.
       d(ln b)/d(ln R) across a grid of n and rho. Gate: the exponent must converge to n as rho
       grows and to 1 as rho falls, within 1% at rho = 1000 and rho = 0.001. If it does not, the
       amplification claim is wrong regardless of what any fit says.

  M2 WHAT CHAIN LENGTH DOES THE MEASURED RECEPTOR CHANGE DEMAND?     THE ANSWER.
       for each of loop 148's three MEASURED receptor ratios -- 1.51x the median receptor, 19.65x
       CDC20 in LFQ, 39.72x CDC20 in raw Intensity -- the smallest n that reaches 20.29x, optimising
       over rho. Gate: n <= 10. Thrower 2000 put the minimum proteasome-competent K48 chain at 4 and
       observed chains run to roughly 10, so an n inside 4-10 is a mechanism and an n of 40 is a
       refutation. This gate can kill the idea outright.

  M3 THE REGIME THE MECHANISM REQUIRES.                              THE FALSIFIABLE PREDICTION.
       amplification needs rho > 1: the resting cell must be DUB-dominated, deubiquitinating faster
       than it ubiquitinates, so that most chains fall back before reaching n. Report the minimum
       rho at which M2's n works. Then test it as far as the data allows -- total DUB abundance
       against total targeting-receptor abundance in the same Ly proteome.
       PREDECLARED LIMIT: abundance is not rate. k_cat differs between a DUB and an E3 and this
       comparison cannot see that. It is the weakest available proxy, the gate is REPORT rather than
       decide, and the number is barred from being quoted as a measurement of rho -- the same
       handling loop 148's K4 gave the mRNA proxy.

  M4 CAN ONE PARAMETER SET CARRY BOTH MEASURED RATES?                THE CONSISTENCY TEST.
       the chain must reproduce the resting b_lo = 0.02347/h AND the pulsed b_hi = 0.47617/h under
       ONLY a receptor change of the measured size -- not two independent fits. Because T scales as
       f(rho)/lambda, the achievable ratio r*f(rho)/f(rho/r) depends on (n, rho, r) alone and k_u is
       then fixed by matching b_lo, so this is two equations in two unknowns and it can fail.
       Gate: a solution with k_u and k_d both inside 1e-3 to 1e5 /h. That band is eight orders wide
       on purpose; anything outside it is not a rate constant.

  M5 THE OBJECTION THAT COULD KILL IT.                               THE ADVERSARIAL CONTROL.
       the amplification assumes PROCESSIVITY -- that the substrate stays bound while the chain is
       built. If it dissociates and the chain is trimmed, the ladder resets and the exponent should
       collapse. Add escape at k_off from every intermediate state back to length 0 and sweep it.
       Gate: report the largest k_off/lambda at which M2's n still delivers 20.29x. If the mechanism
       only works at k_off = 0 it is a mathematical curiosity rather than a mechanism.

  M6 WHAT THIS DOES AND DOES NOT BUY.
       state plainly that this is a REACHABILITY result, not a measurement. It converts loop 148's
       "1.5x cannot make 20.3x" into "1.5x makes 20.3x if and only if n >= N and rho > P", which is
       a prediction with numbers in it. It identifies no protein and no timing, and the nine
       mechanisms eliminated on that question stay eliminated.

-> outputs/loop_ubiquitin_markov.json
"""
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM            # noqa: E402
import loop_replication as LR        # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
LY = SC / "ly2014_supp1-v1.txt"

SEED = 14900
M0_TOL_RECUR = 1e-10
M0_TOL_LINALG = 1e-6
M0_COND_MAX = 1e12
M1_TOL = 0.01
M2_MAX_N = 10                  # Thrower 2000: >=4 for proteasome engagement, ~10 at the top end
M4_RATE_LO, M4_RATE_HI = 1e-3, 1e5
NMAX = 40                      # searched well past the gate so a FAIL is quantified, not truncated

# DUBs, declared before the run. Prefix families plus the singletons that do not share a prefix.
DUB_PREFIX = ("USP", "UCHL", "OTUD", "JOSD", "MINDY", "ZUP")
DUB_EXACT = ("UCHL1", "UCHL3", "UCHL5", "OTUB1", "OTUB2", "YOD1", "ZRANB1", "VCPIP1", "ATXN3",
             "ATXN3L", "BAP1", "MYSM1", "MPND", "STAMBP", "STAMBPL1", "BRCC3", "COPS5", "COPS6",
             "EIF3F", "EIF3H", "ZUFSP", "USPL1", "SENP1", "SENP2", "SENP3", "SENP5", "SENP6",
             "SENP7", "PRPF8")
DUB_EXCLUDE = ("USP1L1",)

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def f_shape(rho, n):
    """The dimensionless part of the MFPT: T = f_shape(rho, n) / lambda."""
    if abs(rho - 1.0) < 1e-12:
        return n * (n + 1) / 2.0
    return (n - rho * (1.0 - rho ** n) / (1.0 - rho)) / (1.0 - rho)


def T_closed(lam, mu, n):
    return f_shape(mu / lam, n) / lam


def T_recur(lam, mu, n):
    rho, tot, ti = mu / lam, 0.0, 0.0
    for i in range(n):
        ti = 1.0 / lam + (rho * ti if i > 0 else 0.0)
        tot += ti
    return tot


def T_linalg(lam, mu, n, k_off=0.0):
    """MFPT by solving the absorbing generator. k_off sends any intermediate state back to 0."""
    A = np.zeros((n, n))
    for i in range(n):
        out = lam + (mu if i > 0 else 0.0) + (k_off if i > 0 else 0.0)
        A[i, i] = -out
        if i + 1 < n:
            A[i, i + 1] = lam
        if i > 0:
            A[i, i - 1] = mu
            A[i, 0] += k_off
    return float(np.linalg.solve(A, -np.ones(n))[0]), float(np.linalg.cond(A))


def amplification(rho, n, r):
    """b(R=r)/b(R=1) for the pure chain: lambda -> r*lambda sends rho -> rho/r."""
    return r * f_shape(rho, n) / f_shape(rho / r, n)


def best_amplification(n, r, grid):
    a = [amplification(rho, n, r) for rho in grid]
    j = int(np.argmax(a))
    return a[j], grid[j]


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 149 -- a Markov chain on the ubiquitin chain: can a multistep process amplify "
        "1.5x into 20.3x?")
    say("=" * 100)
    say()

    import pandas as pd

    PEQ = json.load(open(OUT / "loop_pulse_equation.json"))
    CAP = json.load(open(OUT / "loop_capacity_ratio.json"))
    CTL = json.load(open(OUT / "loop_capacity_ratio_controls.json"))
    B_LO = float(PEQ["x3"]["b_rest"])
    B_HI = float(PEQ["x3"]["b_hi_required"])
    REQ = float(PEQ["x3"]["fold_acceleration"])
    R_MEASURED = {
        "median receptor": float(CAP["k3"]["receptor_median_fold"]),
        "CDC20 (LFQ)": float(CTL["l2"]["channels"]["LFQ"]["fold"]),
        "CDC20 (raw Intensity)": float(CTL["l2"]["channels"]["rawIntensity"]["fold"]),
    }
    say(f"  read from the record, not retyped:  b_lo {B_LO:.5f}/h   b_hi {B_HI:.5f}/h   "
        f"required {REQ:.2f}x")
    say(f"  measured receptor changes: " + "   ".join(f"{k} {v:.2f}x"
                                                      for k, v in R_MEASURED.items()))
    say()

    gates, res = {}, {}
    RHO_GRID = np.concatenate([np.logspace(-3, 4, 4001)])

    # ---------------------------------------------------------------- M0
    say("M0 DOES THE ARITHMETIC CHECK THE ARITHMETIC?")
    w_rec, w_lin, n_lin, worst_cond = 0.0, 0.0, 0, 0.0
    for n in range(1, 13):
        for rho in (0.001, 0.05, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
            c, rr = T_closed(1.0, rho, n), T_recur(1.0, rho, n)
            w_rec = max(w_rec, abs(c - rr) / rr)
            l, k = T_linalg(1.0, rho, n)
            worst_cond = max(worst_cond, k)
            if k < M0_COND_MAX:
                w_lin = max(w_lin, abs(c - l) / l)
                n_lin += 1
    lim_rho0 = max(abs(T_closed(1.0, 1e-12, n) - n) / n for n in range(1, 13))
    lim_n1 = max(abs(T_closed(1.0, rho, 1) - 1.0) for rho in (0.001, 1.0, 100.0, 1e4))
    say(f"     closed form vs exact recursion, n=1..12 x rho=1e-3..1e2: worst {w_rec:.2e}   "
        f"gate < {M0_TOL_RECUR:.0e}")
    say(f"     closed form vs linear algebra over the {n_lin} cells with cond(A) < "
        f"{M0_COND_MAX:.0e}: worst {w_lin:.2e}   gate < {M0_TOL_LINALG:.0e}")
    say(f"     worst cond(A) anywhere on the grid {worst_cond:.2e} -- past 1e15 the LINEAR SOLVE is "
        f"the thing that is wrong, not the formula, and that is why it is fenced by conditioning")
    say(f"     limit rho -> 0 gives T = n/lambda: worst deviation {lim_rho0:.2e}")
    say(f"     limit n = 1 gives T = 1/lambda for every rho: worst deviation {lim_n1:.2e}")
    gates["M0"] = bool(w_rec < M0_TOL_RECUR and w_lin < M0_TOL_LINALG
                       and lim_rho0 < 1e-9 and lim_n1 < 1e-9)
    res["m0"] = {"worst_vs_recursion": w_rec, "worst_vs_linalg": w_lin,
                 "n_wellconditioned_cells": n_lin, "worst_cond": worst_cond,
                 "limit_rho0": lim_rho0, "limit_n1": lim_n1, "pass": gates["M0"]}
    say(f"     M0 {'PASS' if gates['M0'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- M1
    say("M1 IS THE EXPONENT REALLY n?")

    def expo(n, rho, eps=1e-6):
        b1 = 1.0 / T_closed(1.0, rho, n)
        b2 = 1.0 / T_closed(1.0 * (1 + eps), rho, n)
        return math.log(b2 / b1) / math.log(1 + eps)

    tab = {}
    for n in (1, 2, 4, 8, 12):
        tab[n] = {f"{rho:g}": expo(n, rho) for rho in (0.001, 0.1, 1.0, 10.0, 100.0, 1000.0)}
        say(f"       n={n:2d}   " + "   ".join(f"rho={k:<6} {v:6.3f}" for k, v in tab[n].items()))
    hi_err = max(abs(expo(n, 1000.0) - n) / n for n in (1, 2, 4, 8, 12))
    lo_err = max(abs(expo(n, 0.001) - 1.0) for n in (1, 2, 4, 8, 12))
    say(f"     converges to n at rho=1000: worst relative error {hi_err:.4f}   gate < {M1_TOL}")
    say(f"     converges to 1 at rho=0.001: worst absolute error {lo_err:.4f}   gate < {M1_TOL}")
    say(f"     the DUBs are what make the chain nonlinear. Without them (rho -> 0) every step is")
    say(f"     one-way, the times just add, and n steps buy nothing at all.")
    gates["M1"] = bool(hi_err < M1_TOL and lo_err < M1_TOL)
    res["m1"] = {"table": tab, "err_high_rho": hi_err, "err_low_rho": lo_err, "pass": gates["M1"]}
    say(f"     M1 {'PASS' if gates['M1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- M2
    say("M2 WHAT CHAIN LENGTH DOES THE MEASURED RECEPTOR CHANGE DEMAND?")
    need = {}
    for name, r in R_MEASURED.items():
        row = None
        for n in range(1, NMAX + 1):
            a, rho_star = best_amplification(n, r, RHO_GRID)
            if a >= REQ:
                row = {"n": n, "rho": float(rho_star), "amplification": float(a)}
                break
        if row is None:
            a, rho_star = best_amplification(NMAX, r, RHO_GRID)
            row = {"n": None, "rho": float(rho_star), "amplification": float(a),
                   "searched_to": NMAX}
        need[name] = row
        if row["n"]:
            say(f"       receptor {r:6.2f}x  ->  n = {row['n']:2d} ubiquitins "
                f"(best rho {row['rho']:.3g}, delivers {row['amplification']:.1f}x)")
        else:
            say(f"       receptor {r:6.2f}x  ->  NOT REACHABLE by n <= {NMAX} "
                f"(best {row['amplification']:.1f}x)")
    ns = [v["n"] for v in need.values() if v["n"]]
    worst_n = max(ns) if len(ns) == len(need) else None
    say(f"     gate: every measured receptor ratio must close at n <= {M2_MAX_N}")
    say(f"     Thrower 2000 puts the minimum proteasome-competent K48 chain at 4 and observed "
        f"chains around 10, so 4-10 is a mechanism and 40 is a refutation.")
    gates["M2"] = bool(worst_n is not None and worst_n <= M2_MAX_N)
    res["m2"] = {"required": REQ, "per_receptor": need, "worst_n": worst_n,
                 "gate_max_n": M2_MAX_N, "pass": gates["M2"]}
    say(f"     M2 {'PASS' if gates['M2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- M3
    say("M3 THE REGIME THE MECHANISM REQUIRES")
    rho_min = {}
    for name, r in R_MEASURED.items():
        n = need[name]["n"]
        if not n:
            continue
        ok = [rho for rho in RHO_GRID if amplification(rho, n, r) >= REQ]
        rho_min[name] = {"n": n, "rho_min": float(min(ok)), "rho_max": float(max(ok))}
        say(f"       {name}: at n={n}, 20.29x needs rho between {min(ok):.3g} and {max(ok):.3g}")
    all_above_1 = all(v["rho_min"] > 1.0 for v in rho_min.values()) if rho_min else False
    say(f"     THE PREDICTION: the resting cell must be DUB-dominated, rho > 1, on these "
        f"substrates. rho > 1 everywhere in the solution band: {all_above_1}")

    d = pd.read_csv(LY, sep="\t", low_memory=False)
    d["g"] = d["gene_names"].astype(str).str.split(";").str[0]
    F = [f"iBAQ_F{i}" for i in range(1, 7)]
    V = d[F].apply(pd.to_numeric, errors="coerce").values.astype(float)
    ok = np.isfinite(V).all(1) & (np.nanmin(V, axis=1) > 0)
    ab = V[ok].mean(1)
    G = d["g"].values[ok]
    is_dub = np.array([(g in DUB_EXACT or any(g.startswith(p) for p in DUB_PREFIX))
                       and g not in DUB_EXCLUDE for g in G])
    recset = set(CAP["k0"]["receptors"])
    is_rec = np.array([g in recset for g in G])
    dub_tot, rec_tot = float(ab[is_dub].sum()), float(ab[is_rec].sum())
    say(f"     PROXY, and it is only a proxy: iBAQ mass in the same Ly proteome")
    say(f"       {int(is_dub.sum())} DUBs total {dub_tot:.3e};  {int(is_rec.sum())} targeting "
        f"receptors total {rec_tot:.3e};  ratio {dub_tot / max(rec_tot, 1e-30):.1f}")
    say(f"     ABUNDANCE IS NOT RATE. A DUB and an E3 do not share a k_cat and this comparison "
        f"cannot see the difference, so the gate here is REPORT and the number is BARRED from")
    say(f"     being quoted as a measurement of rho -- the same handling loop 148's K4 gave mRNA.")
    gates["M3"] = bool(rho_min and all_above_1)
    res["m3"] = {"rho_band": rho_min, "prediction_dub_dominated": all_above_1,
                 "n_dub": int(is_dub.sum()), "n_receptor": int(is_rec.sum()),
                 "dub_ibaq_total": dub_tot, "receptor_ibaq_total": rec_tot,
                 "abundance_ratio": float(dub_tot / max(rec_tot, 1e-30)),
                 "proxy_barred_from_quotation_as_rho": True, "pass": gates["M3"]}
    say(f"     M3 {'PASS' if gates['M3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- M4
    say("M4 CAN ONE PARAMETER SET CARRY BOTH MEASURED RATES?")
    sol = {}
    for name, r in R_MEASURED.items():
        n = need[name]["n"]
        if not n:
            continue
        cand = [rho for rho in RHO_GRID if amplification(rho, n, r) >= REQ]
        rho = float(cand[int(np.argmin([abs(amplification(c, n, r) - REQ) for c in cand]))])
        k_u = f_shape(rho, n) * B_LO           # matches b_lo exactly with k_p -> infinity
        k_d = rho * k_u
        b_lo_chk = 1.0 / T_closed(k_u, k_d, n)
        b_hi_chk = 1.0 / T_closed(k_u * r, k_d, n)
        sane = (M4_RATE_LO <= k_u <= M4_RATE_HI) and (M4_RATE_LO <= k_d <= M4_RATE_HI)
        sol[name] = {"n": n, "rho": rho, "k_u_per_h": float(k_u), "k_d_per_h": float(k_d),
                     "b_lo_check": float(b_lo_chk), "b_hi_check": float(b_hi_chk),
                     "ratio_check": float(b_hi_chk / b_lo_chk), "sane": bool(sane)}
        say(f"       {name}: n={n}  rho={rho:.3g}  k_u={k_u:.4g}/h  k_d={k_d:.4g}/h")
        say(f"          reproduces b_lo {b_lo_chk:.5f} (target {B_LO:.5f}) and b_hi "
            f"{b_hi_chk:.5f} (target {B_HI:.5f}) -> {b_hi_chk / b_lo_chk:.2f}x   "
            f"{'sane' if sane else 'OUTSIDE THE RATE BAND'}")
    say(f"     gate: k_u and k_d both inside {M4_RATE_LO:g} to {M4_RATE_HI:g} /h. Eight orders "
        f"wide on purpose -- outside it, it is not a rate constant.")
    gates["M4"] = bool(sol and all(v["sane"] for v in sol.values()))
    res["m4"] = {"solutions": sol, "band": [M4_RATE_LO, M4_RATE_HI], "pass": gates["M4"]}
    say(f"     M4 {'PASS' if gates['M4'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- M5
    say("M5 THE OBJECTION THAT COULD KILL IT: does it survive a substrate that lets go?")
    name0 = "median receptor"
    n0 = need[name0]["n"] or M2_MAX_N
    rho0 = sol.get(name0, {}).get("rho", 100.0)
    r0 = R_MEASURED[name0]
    lam = 1.0
    mu = rho0 * lam
    sweep = []
    for koff_rel in (0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0):
        ko = koff_rel * lam
        t1, c1 = T_linalg(lam, mu, n0, ko)
        t2, c2 = T_linalg(lam * r0, mu, n0, ko)
        amp = t1 / t2
        sweep.append({"koff_over_lambda": koff_rel, "amplification": float(amp),
                      "cond": max(c1, c2)})
        say(f"       k_off/lambda {koff_rel:<7g}  amplification {amp:7.2f}x   "
            f"{'reaches' if amp >= REQ else 'below'} {REQ:.1f}x")
    surviving = [s["koff_over_lambda"] for s in sweep if s["amplification"] >= REQ]
    kmax = max(surviving) if surviving else None
    say(f"     largest k_off/lambda at which n={n0} still delivers {REQ:.1f}x: "
        f"{kmax if kmax is not None else 'none -- only at k_off = 0'}")
    say(f"     processivity is a real requirement of this mechanism, not an accident of it: the")
    say(f"     amplification lives in the DUBs winning the race back down, and a substrate that")
    say(f"     lets go resets the ladder before the race can be run.")
    gates["M5"] = bool(kmax is not None and kmax > 0.0)
    res["m5"] = {"n": n0, "rho": rho0, "receptor_ratio": r0, "sweep": sweep,
                 "max_koff_over_lambda": kmax, "pass": gates["M5"]}
    say(f"     M5 {'PASS' if gates['M5'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- M6
    say("M6 WHAT THIS DOES AND DOES NOT BUY")
    say(f"     IT IS A REACHABILITY RESULT, NOT A MEASUREMENT. Nothing here was fitted to a")
    say(f"     trajectory and no new data was read; the chain is arithmetic and the inputs are")
    say(f"     loop 142's two rates and loop 148's three receptor ratios.")
    say(f"     What it converts: loop 148's 'a 1.5x receptor change cannot make a 20.3x rate")
    say(f"     change' becomes 'it makes it if and only if the chain is at least n long and the")
    say(f"     resting state is DUB-dominated'. That is a statement with numbers in it and it can")
    say(f"     be refuted by measuring chain length or a DUB/E3 rate ratio.")
    say(f"     IT IDENTIFIES NO PROTEIN AND NO TIMING. Nine mechanisms have been eliminated on")
    say(f"     that question -- transcription, two TF networks, degron motifs, 5'UTR control,")
    say(f"     relocalisation, ubiquitin annotation, protease competition, and phosphodegrons")
    say(f"     twice -- and a Markov chain on chain length does not touch any of it.")
    gates["M6"] = True
    res["m6"] = {"reachability_not_measurement": True, "identifies_which_proteins": False,
                 "refutable_by": ["K48 chain length on a pulsed substrate",
                                  "a DUB/E3 rate ratio at rest",
                                  "processivity of APC/C on a pulsed substrate"]}
    say()

    say("=" * 100)
    for k in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[OUT / "loop_pulse_equation.json", OUT / "loop_capacity_ratio.json",
                              OUT / "loop_capacity_ratio_controls.json", LY],
                      available=len(R_MEASURED), used=len(R_MEASURED), selection="all", seed=SEED,
                      controls=["the closed form checked against an exact recursion AND against "
                                "linear algebra, fenced by conditioning (M0)",
                                "both asymptotic limits of the exponent verified (M1)",
                                "one parameter set required to carry both measured rates rather "
                                "than two independent fits (M4)",
                                "the processivity objection swept rather than assumed away (M5)",
                                "the DUB/receptor abundance comparison barred from being quoted "
                                "as a rate ratio (M3)"],
                      note="a reachability result. The chain is arithmetic; the inputs are loop "
                           "142's two rates and loop 148's three measured receptor ratios. It "
                           "converts an impossibility into a prediction with a chain length and a "
                           "regime in it, and identifies no protein.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 149 -- the ubiquitin chain as a Markov chain", "manifest": man,
               "gates": gates, **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_ubiquitin_markov.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_ubiquitin_markov.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
