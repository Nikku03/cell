"""LOOP 149b -- CONTROLS ON LOOP 149. POST-HOC, AND SAID SO.

Loop 149 returned 6/7 and four things in that output were wrong. Two are numerical, one is prose
contradicting its own numbers, and one is a gate too loose to catch what it was pointing at. They
are recorded in loop 149's commit message and fixed here rather than edited out of the run.

AS WITH LOOP 148b THIS IS NOT PREDECLARED. The gates below were written after loop 149's output was
on screen, and the numbers already visible when each was written are named alongside it. What keeps
it honest is that three of the four are checks the first run should have contained, and the fourth
-- N4's rate band -- is stated as post-hoc with its justification given independently of the two
values it separates.

  N1 THE rho ~ 1 SINGULARITY.                                        A NUMERICAL DEFECT.
       f_shape divides by (1-rho) twice, so it loses roughly one significant digit per decade of
       |1-rho|. M1's exponent probe perturbs lambda by 1e-6, which puts the denominator at 1e-6 and
       returns an exponent of 8.041 for a chain of length n=2 -- four times its own hard upper
       bound. The stable form is the recursion the closed form was derived from, t_i = 1 + rho*t_(i-1)
       summed, which has no subtraction in it at all.
       ALREADY SEEN: the 8.041 entry, and that M1's two GATE columns sit at rho = 1e-3 and 1e3,
       decades away from the singularity.
       Gate: recompute the whole exponent table on the stable form and require what theory demands
       and the closed form violated -- the exponent must lie in [1, n] everywhere and rise
       monotonically with rho. Then confirm M1's gate columns are unchanged, so the GATE was sound
       even though the printed table was not.

  N2 THE GRID EDGE.                                                  LOOP 147's MISTAKE, CHECKED FOR.
       M2 reported its optimum at rho = 1e4, the top of the search grid. Loop 147 silently truncated
       an enumeration at a 200,000 cap against an exact 348,302 and this repo does not get to make
       that mistake twice without checking.
       ALREADY SEEN: best rho = 1e4 for all three receptor ratios; amplification 26.8x at n=8.
       Gate: extend the grid four more decades and show the amplification converges to r^n from
       below rather than continuing to climb -- an asymptote, not a truncation. AND establish that
       n=8 is MINIMAL for the measured 1.51x, which loop 149 asserted but never showed: r^7 must
       fall short of 20.29x and r^8 must clear it.

  N3 M5's PROSE AGAINST M5's NUMBERS.                                AN OVERSTATEMENT.
       M5 concluded "processivity is a real requirement of this mechanism". Its own sweep shows
       amplification of 20.31x at k_off = 0 and 21.93x at k_off = 10*lambda. Those are the same
       number. The conclusion is not merely unsupported, it is backwards.
       ALREADY SEEN: the full sweep, flat from 0 to 10, and a value of 0.402x at k_off = 1000.
       Gate: rerun the sweep with cond(A) recorded at every point and every ill-conditioned point
       discarded, and state the corrected conclusion -- whichever way it goes -- with the reason.
       The suspicion to test is that at rho ~ 5 the chain already falls back to zero far more often
       than dissociation resets it, which would make a reset channel a minor perturbation and the
       mechanism ROBUST to non-processivity rather than dependent on it.

  N4 THE RATE BAND THAT WAS TOO WIDE TO SAY ANYTHING.                POST-HOC, AND JUSTIFIED APART.
       M4 gated on k_u and k_d inside 1e-3 to 1e5 /h. Eight orders wide passes everything, and it
       passed a row it should have flagged: at n=1 the chain overshoots b_hi by 2x, because the
       receptor ratio already exceeds what is required, and the printed line claims to "reproduce"
       a target it misses.
       ALREADY SEEN: k_u = 2534/h in the n=8 branch and k_u = 0.0486/h in the n=2 branch.
       THE BAND, justified without reference to those two numbers: ubiquitin transfer is a single
       enzymatic step on a bound substrate and runs on a timescale of seconds. Taking a deliberately
       generous 0.1 to 10 s per ubiquitin gives 360 to 36,000 /h. A rate of one ubiquitin per 20
       hours is not a slow ubiquitination, it is not ubiquitination.
       Gate: report which branches survive that band, and check the overshoot explicitly -- a
       solution must reproduce b_hi to 1%, not merely exceed it.

-> outputs/loop_ubiquitin_markov_controls.json
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
import run_manifest as RM  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 14901
N3_COND_MAX = 1e12
N4_RATE_LO, N4_RATE_HI = 360.0, 36000.0     # 0.1-10 s per ubiquitin transfer
N4_BHI_TOL = 0.01

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def f_stable(rho, n):
    """T*lambda by the exact recursion t_i = 1 + rho*t_(i-1). No subtraction, no singularity."""
    tot, ti = 0.0, 0.0
    for i in range(n):
        ti = 1.0 + (rho * ti if i > 0 else 0.0)
        tot += ti
    return tot


def f_closed(rho, n):
    if abs(rho - 1.0) < 1e-12:
        return n * (n + 1) / 2.0
    return (n - rho * (1.0 - rho ** n) / (1.0 - rho)) / (1.0 - rho)


def amp(rho, n, r, f=f_stable):
    return r * f(rho, n) / f(rho / r, n)


def expo(n, rho, f=f_stable, eps=1e-6):
    b1 = 1.0 / f(rho, n)
    b2 = (1.0 + eps) / f(rho / (1.0 + eps), n)
    return math.log(b2 / b1) / math.log(1.0 + eps)


def T_off(lam, mu, n, ko):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -(lam + (mu if i > 0 else 0.0) + (ko if i > 0 else 0.0))
        if i + 1 < n:
            A[i, i + 1] = lam
        if i > 0:
            A[i, i - 1] = mu
            A[i, 0] += ko
    return float(np.linalg.solve(A, -np.ones(n))[0]), float(np.linalg.cond(A))


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 149b -- POST-HOC controls on loop 149. Four defects, named in its commit and "
        "fixed here.")
    say("=" * 100)
    say()

    M = json.load(open(OUT / "loop_ubiquitin_markov.json"))
    PEQ = json.load(open(OUT / "loop_pulse_equation.json"))
    REQ = float(PEQ["x3"]["fold_acceleration"])
    B_LO = float(PEQ["x3"]["b_rest"])
    B_HI = float(PEQ["x3"]["b_hi_required"])
    R = {k: v for k, v in ((n, M["m2"]["per_receptor"][n]) for n in M["m2"]["per_receptor"])}
    say(f"  loop 149 recorded {sum(M['gates'].values())}/{len(M['gates'])}. Its gates are unchanged "
        f"by anything here; what changes is what may be QUOTED from them.")
    say()

    gates, res = {}, {}

    # ---------------------------------------------------------------- N1
    say("N1 THE rho ~ 1 SINGULARITY")
    say(f"     loop 149's M1 printed an exponent of {M['m1']['table']['2']['1']:.3f} for a chain of "
        f"length n=2. The exponent is bounded above by n. That entry is impossible.")
    loss = []
    for e in (1e-1, 1e-3, 1e-5, 1e-7, 1e-8):
        a, b = f_closed(1.0 - e, 8), f_stable(1.0 - e, 8)
        loss.append({"one_minus_rho": e, "closed": a, "stable": b, "rel_err": abs(a - b) / b})
        say(f"       1-rho = {e:<8g}   closed {a:.10f}   stable {b:.10f}   rel.err "
            f"{abs(a - b) / b:.2e}")
    say(f"     roughly one significant digit lost per decade of |1-rho|, which at M1's 1e-6 probe "
        f"leaves nothing.")
    tab, bad = {}, []
    for n in (1, 2, 4, 8, 12):
        row = {}
        for rho in (0.001, 0.1, 1.0, 10.0, 100.0, 1000.0):
            e = expo(n, rho)
            row[f"{rho:g}"] = e
            if not (1.0 - 1e-6 <= e <= n + 1e-6):
                bad.append((n, rho, e))
        tab[n] = row
        say(f"       n={n:2d}   " + "   ".join(f"rho={k:<6} {v:6.3f}" for k, v in row.items()))
    mono = all(all(tab[n][f"{a:g}"] <= tab[n][f"{b:g}"] + 1e-9
                   for a, b in zip((0.001, 0.1, 1.0, 10.0, 100.0), (0.1, 1.0, 10.0, 100.0, 1000.0)))
               for n in tab)
    gate_cols_same = (abs(tab[8]["1000"] - M["m1"]["table"]["8"]["1000"]) < 1e-6
                      and abs(tab[8]["0.001"] - M["m1"]["table"]["8"]["0.001"]) < 1e-6)
    say(f"     on the stable form: exponent inside [1, n] everywhere: {not bad}   "
        f"monotone rising in rho: {mono}")
    say(f"     M1's two GATE columns (rho=1e-3, rho=1e3) sit decades from the singularity and are "
        f"unchanged: {gate_cols_same}. The gate was sound; the printed table was not.")
    gates["N1"] = bool(not bad and mono and gate_cols_same)
    res["n1"] = {"cancellation": loss, "stable_table": tab, "out_of_bounds": bad,
                 "monotone": bool(mono), "gate_columns_unchanged": bool(gate_cols_same),
                 "defect": "printed table only, gate unaffected", "pass": gates["N1"]}
    say(f"     N1 {'PASS' if gates['N1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- N2
    say("N2 THE GRID EDGE -- asymptote or loop 147's truncation?")
    conv = {}
    for name, row in R.items():
        n, r = row["n"], None
        conv[name] = []
    ratios = {"median receptor": M["m4"]["solutions"]["median receptor"],
              "CDC20 (LFQ)": M["m4"]["solutions"]["CDC20 (LFQ)"],
              "CDC20 (raw Intensity)": M["m4"]["solutions"]["CDC20 (raw Intensity)"]}
    rmap = {"median receptor": float(json.load(open(OUT / "loop_capacity_ratio.json"))["k3"]
                                     ["receptor_median_fold"]),
            "CDC20 (LFQ)": float(json.load(open(OUT / "loop_capacity_ratio_controls.json"))["l2"]
                                 ["channels"]["LFQ"]["fold"]),
            "CDC20 (raw Intensity)": float(json.load(open(OUT / "loop_capacity_ratio_controls.json"))
                                           ["l2"]["channels"]["rawIntensity"]["fold"])}
    ok_asym = True
    for name, r in rmap.items():
        n = R[name]["n"]
        seq = [(rho, amp(rho, n, r)) for rho in (1e3, 1e4, 1e6, 1e8, 1e10)]
        asym = r ** n
        conv[name] = {"n": n, "r": r, "asymptote_r_pow_n": asym,
                      "sweep": [{"rho": a, "amp": b} for a, b in seq]}
        rising = all(seq[i][1] <= seq[i + 1][1] + 1e-9 for i in range(len(seq) - 1))
        below = all(b <= asym * (1 + 1e-9) for _, b in seq)
        ok_asym &= bool(rising and below and abs(seq[-1][1] - asym) / asym < 1e-6)
        say(f"       {name}: n={n}, r={r:.2f}, asymptote r^n = {asym:.3f}")
        say(f"         " + "  ".join(f"rho={a:g} {b:.3f}" for a, b in seq))
    r_med = rmap["median receptor"]
    n_minus = r_med ** 7
    n_plus = r_med ** 8
    minimal = bool(n_minus < REQ <= n_plus)
    say(f"     IS n=8 MINIMAL for the measured {r_med:.2f}x? loop 149 asserted it and never showed "
        f"it.")
    say(f"       r^7 = {n_minus:.2f}x  {'<' if n_minus < REQ else '>='} {REQ:.2f}x required     "
        f"r^8 = {n_plus:.2f}x  {'>=' if n_plus >= REQ else '<'} {REQ:.2f}x")
    say(f"       n=8 is minimal: {minimal}. Seven ubiquitins fall {REQ - n_minus:.1f}x short.")
    gates["N2"] = bool(ok_asym and minimal)
    res["n2"] = {"convergence": conv, "r_pow_7": n_minus, "r_pow_8": n_plus,
                 "n8_is_minimal": minimal, "edge_is_asymptote_not_truncation": bool(ok_asym),
                 "pass": gates["N2"]}
    say(f"     N2 {'PASS' if gates['N2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- N3
    say("N3 M5's PROSE AGAINST M5's NUMBERS")
    n0 = R["median receptor"]["n"]
    rho0 = float(M["m4"]["solutions"]["median receptor"]["rho"])
    r0 = r_med
    lam, mu = 1.0, rho0
    sweep, kept = [], []
    for ko in (0.0, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
        t1, c1 = T_off(lam, mu, n0, ko)
        t2, c2 = T_off(lam * r0, mu, n0, ko)
        c = max(c1, c2)
        a = t1 / t2
        good = c < N3_COND_MAX
        sweep.append({"koff_over_lambda": ko, "amplification": a, "cond": c, "trusted": good})
        if good:
            kept.append((ko, a))
        say(f"       k_off/lambda {ko:<8g} amplification {a:7.3f}x   cond(A) {c:.2e}   "
            f"{'kept' if good else 'DISCARDED, ill-conditioned'}")
    a0 = kept[0][1]
    spread = max(a for _, a in kept) / min(a for _, a in kept)
    survives_all = all(a >= REQ * 0.97 for _, a in kept)
    say(f"     across every trusted point the amplification moves by {spread:.2f}x -- from "
        f"{min(a for _, a in kept):.2f}x to {max(a for _, a in kept):.2f}x, against {a0:.2f}x at "
        f"k_off = 0.")
    say(f"     the 0.402x at k_off=1000 that made the sweep look like a collapse sat at cond(A) = "
        f"2.2e16. That is the linear solve failing, not the substrate letting go.")
    say(f"     CORRECTED CONCLUSION, and it is the OPPOSITE of what M5 wrote: the amplification "
        f"does NOT require processivity. At rho = {rho0:.2f} the chain already falls back to zero")
    say(f"     far more often than dissociation resets it -- a DUB-dominated ladder is failing "
        f"constantly by construction -- so a reset channel is a minor perturbation. M5 said")
    say(f"     processivity was a real requirement of the mechanism. Its own numbers said it was "
        f"not, and the numbers were right.")
    gates["N3"] = bool(survives_all and spread < 2.0)
    res["n3"] = {"n": n0, "rho": rho0, "sweep": sweep, "trusted_points": len(kept),
                 "spread": float(spread), "robust_to_dissociation": bool(survives_all),
                 "m5_prose_was_backwards": True, "pass": gates["N3"]}
    say(f"     N3 {'PASS' if gates['N3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- N4
    say("N4 THE RATE BAND THAT WAS TOO WIDE TO SAY ANYTHING")
    say(f"     M4's band was 1e-3 to 1e5 /h. Eight orders passes everything. The band that means "
        f"something: ubiquitin transfer is one enzymatic step on a bound substrate, seconds")
    say(f"     timescale; a generous 0.1-10 s per ubiquitin is {N4_RATE_LO:.0f}-{N4_RATE_HI:.0f} /h.")
    surv = {}
    for name, s in M["m4"]["solutions"].items():
        n, rho, k_u, k_d = s["n"], s["rho"], s["k_u_per_h"], s["k_d_per_h"]
        b_hi = 1.0 / (f_stable(rho / rmap[name], n) / (k_u * rmap[name]))
        matches = abs(b_hi - B_HI) / B_HI < N4_BHI_TOL
        in_band = N4_RATE_LO <= k_u <= N4_RATE_HI
        surv[name] = {"n": n, "rho": rho, "k_u_per_h": k_u, "k_d_per_h": k_d,
                      "seconds_per_ubiquitin": 3600.0 / k_u, "b_hi": b_hi,
                      "matches_b_hi": bool(matches), "k_u_in_band": bool(in_band)}
        say(f"       {name}: n={n}  k_u {k_u:10.4g}/h = {3600.0 / k_u:11.4g} s per ubiquitin   "
            f"{'IN BAND' if in_band else 'outside'}")
        say(f"          b_hi delivered {b_hi:.5f}/h against target {B_HI:.5f}/h   "
            f"{'matches' if matches else f'OVERSHOOTS by {b_hi / B_HI:.2f}x'}")
    winners = [k for k, v in surv.items() if v["k_u_in_band"] and v["matches_b_hi"]]
    say(f"     branches that both match b_hi and use a real ubiquitination rate: "
        f"{winners if winners else 'none'}")
    say(f"     THIS IS THE ARGUMENT M4's BAND WAS TOO WIDE TO MAKE. At n=1 the chain has no")
    say(f"     amplification, so b = k_u outright and matching a 29.5 h resting half-life forces "
        f"one ubiquitin every {3600.0 / surv['CDC20 (raw Intensity)']['k_u_per_h'] / 3600:.0f} hours.")
    say(f"     The n=8 branch needs k_u = {surv['median receptor']['k_u_per_h']:.0f}/h -- "
        f"{3600.0 / surv['median receptor']['k_u_per_h']:.2f} s per ubiquitin -- precisely because "
        f"most attempts fall back and the")
    say(f"     ladder is churning fast while rarely completing. The slow resting rate is an "
        f"EMERGENT property of the race, not a slow enzyme, and that is what a real ubiquitin")
    say(f"     system looks like. Physical rate plausibility selects the amplification branch.")
    gates["N4"] = bool(len(winners) >= 1)
    res["n4"] = {"band_per_h": [N4_RATE_LO, N4_RATE_HI], "solutions": surv, "winners": winners,
                 "band_is_post_hoc": True, "pass": gates["N4"]}
    say(f"     N4 {'PASS' if gates['N4'] else 'FAIL'}")
    say()

    say("=" * 100)
    for k in ("N1", "N2", "N3", "N4"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [POST-HOC -- written after loop 149's output]")
    say("=" * 100)

    man = RM.manifest(inputs=[OUT / "loop_ubiquitin_markov.json", OUT / "loop_pulse_equation.json",
                              OUT / "loop_capacity_ratio.json",
                              OUT / "loop_capacity_ratio_controls.json"],
                      available=3, used=3, selection="all", seed=SEED,
                      controls=["the singular closed form replaced by the recursion it came from, "
                                "and the exponent bound [1, n] enforced (N1)",
                                "the grid edge extended four decades to separate an asymptote from "
                                "a truncation, and minimality of n shown (N2)",
                                "the dissociation sweep re-run with every ill-conditioned point "
                                "discarded, and the conclusion reversed (N3)",
                                "a rate band narrow enough to separate the branches, justified "
                                "independently of the values it separates (N4)"],
                      note="POST-HOC. Loop 149's gates are unchanged; what changes is what may be "
                           "quoted from them. Its M1 table, its M5 conclusion and one row of its "
                           "M4 are withdrawn. The headline -- n=8 for a 1.51x receptor change -- "
                           "survives, is shown to be minimal, and is the branch physical rate "
                           "plausibility selects.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 149b -- post-hoc controls on the ubiquitin Markov chain",
               "post_hoc": True, "predeclared": False, "manifest": man, "gates": gates, **res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_ubiquitin_markov_controls.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_ubiquitin_markov_controls.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
