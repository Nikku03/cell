"""LOOP 151b -- CONTROLS ON LOOP 151. POST-HOC, AND SAID SO.

Loop 151's T0 -- its own CAPABILITY gate -- failed, and by its own predeclaration that blocks
reading everything beneath it. It failed at 1.32e-02 against a 1e-8 tolerance, comparing an exact
recursion against numpy's solve of a system whose cond(A) reaches 2.06e15 at n=12, rho=20.

THAT IS THE FOURTH TIME IN THIS ARC. Loop 149b's N1 fixed it in the exponent table. Loop 150b's S1
fixed it in the continued-fraction check and settled that one in exact rationals. And then I wrote
it again, in a capability gate, two loops later.

IT ALSO ESCAPED THE FIX. gate_guard.verdict, added at loop 150b, makes a SENTENCE take its gate as
an argument, so no prose can contradict a gate. Loop 151 used it throughout and it worked -- T3 and
T4 printed their negative verdicts correctly. But T0's failure was not a sentence contradicting a
gate. It was a TOLERANCE that could not be met by the instrument it was applied to, and no amount
of conditioning the prose on the gate catches that. The two are different classes and I fixed one
and kept committing the other. U1 says so rather than adding a second helper and declaring victory.

NOT PREDECLARED. Written after loop 151's output. Numbers already on screen are named per gate.

  U1 SETTLE T0(b) THE WAY THE ARC ALREADY KNOWS HOW.
       ALREADY SEEN: 1.32e-02 against a 1e-8 gate, and that T0(a) and T0(c) both passed.
       Gate: the identity E[up-transitions] = lambda*T must hold in EXACT RATIONAL ARITHMETIC --
       zero disagreements, no floating point -- and the float arm must agree wherever cond(A) is
       trustworthy. If it holds exactly, T0's FAIL was the gate and T1-T5 become readable. If it
       does not, loop 151's cost is not a cost and the whole loop is withdrawn.

  U2 RE-READ THE ANSWER WITH T0 SETTLED.
       ALREADY SEEN: the full T5 frontier, n=8 at 1449% E1 utilisation and 6.83x translation,
       n=9 and n=10 affordable.
       Gate: report the window, and state whether loop 149's n = 8 survives its own price. This is
       a NARROWING of the prediction, not a contradiction: 8 was the shortest ladder that REACHES
       20.29x, and the cost says the shortest AFFORDABLE one is longer.

  U3 THE HINGE THE ANSWER TURNS ON.                                  THE CONTROL THAT MATTERS MOST.
       loop 151 gated on the dilution-free load because only true proteolysis costs ubiquitin, and
       said so before any gate. That choice is worth 7.4x and it decides the answer.
       ALREADY SEEN: dilution-free 3.94e6/h against gross 2.93e7/h, and that the gross figure was
       carried through every line but never gated on.
       Gate: recompute the whole frontier on BOTH loads and report whether the affordable window
       survives either way. If the window exists on one load and not the other, then the answer to
       "can chains stay reversible out to 8" depends on whether Schwanhausser's half-lives are
       division-corrected, and that must be stated as the hinge rather than buried in a caveat.

-> outputs/loop_reversibility_price_controls.json
"""
import json
import os
import sys
import time
import warnings
from fractions import Fraction as F
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM   # noqa: E402
import gate_guard as GG     # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 15101
COND_MAX = 1e10
NMAX = 30
BIOCHEM_NMAX = 10

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def emit(s):
    say(s)


def f_shape(rho, n, one=1.0, zero=0.0):
    tot, ti = zero, zero
    for i in range(n):
        ti = one + (rho * ti if i > 0 else zero)
        tot += ti
    return tot


def upsteps_exact(rho, n):
    lam, mu = F(1), rho
    A = [[F(0)] * n for _ in range(n)]
    b = [F(0)] * n
    for i in range(n):
        tot = lam + (mu if i > 0 else F(0))
        A[i][i] = F(1)
        b[i] = lam / tot
        if i + 1 < n:
            A[i][i + 1] = -lam / tot
        if i > 0:
            A[i][i - 1] = -mu / tot
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        b[c], b[p] = b[p], b[c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            if f:
                for k in range(c, n):
                    A[r][k] -= f * A[c][k]
                b[r] -= f * b[c]
    x = [F(0)] * n
    for r in range(n - 1, -1, -1):
        x[r] = (b[r] - sum(A[r][k] * x[k] for k in range(r + 1, n))) / A[r][r]
    return x[0]


def upsteps_float(rho, n):
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        tot = 1.0 + (rho if i > 0 else 0.0)
        A[i, i] = 1.0
        b[i] = 1.0 / tot
        if i + 1 < n:
            A[i, i + 1] = -1.0 / tot
        if i > 0:
            A[i, i - 1] = -rho / tot
    return float(np.linalg.solve(A, b)[0]), float(np.linalg.cond(A))


def amp(rho, n, r):
    return r * f_shape(rho, n) / f_shape(rho / r, n)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 151b -- POST-HOC controls on loop 151. The capability gate failed, and it failed "
        "the same way for the fourth time.")
    say("=" * 100)
    say()

    P = json.load(open(OUT / "loop_reversibility_price.json"))
    PEQ = json.load(open(OUT / "loop_pulse_equation.json"))
    PRO = json.load(open(OUT / "loop_proteostasis.json"))
    CAP = json.load(open(OUT / "loop_capacity_ratio.json"))
    SIG = json.load(open(OUT / "loop_signalling_cost.json"))
    REQ = float(PEQ["x3"]["fold_acceleration"])
    R_MED = float(CAP["k3"]["receptor_median_fold"])
    LOAD = {"dilution-free (loop 151 gated on this)":
            float(PRO["p2"]["load_without_dilution_term"]),
            "gross (carried but never gated)": float(PRO["p2"]["load_molecules_per_h"])}
    ATP_T = float(SIG["y2"]["translation_atp_h"])
    E1 = float(P["t0"]["uba1_copies"])
    E1_CAP = E1 * 5.0                    # the generous end loop 151 gated on
    gates, res = {}, {}
    say(f"  loop 151 recorded {sum(P['gates'].values())}/{len(P['gates'])} with T0 FAILING at "
        f"{P['t0']['worst_upsteps']:.2e} against a 1e-08 gate.")
    say()

    # ---------------------------------------------------------------- U1
    say("U1 SETTLE T0(b) THE WAY THE ARC ALREADY KNOWS HOW")
    bad, tested = 0, 0
    for rho in (F(1, 10), F(1), F(491, 100), F(20), F(100)):
        for n in (2, 4, 8, 12, 16):
            tested += 1
            if f_shape(rho, n, one=F(1), zero=F(0)) != upsteps_exact(rho, n):
                bad += 1
    say(f"     EXACT RATIONAL: f_shape vs a Gaussian solve for expected up-transitions, {tested} "
        f"settings (rho 0.1..100, n 2..16): {bad} disagreements")
    fl, kept = 0.0, 0
    for rho in (0.1, 1.0, 4.91, 20.0, 100.0):
        for n in (2, 4, 8, 12, 16):
            v, c = upsteps_float(rho, n)
            if c < COND_MAX:
                fl = max(fl, abs(f_shape(rho, n) - v) / v)
                kept += 1
    say(f"     FLOAT arm fenced at cond(A) < {COND_MAX:.0e}: worst {fl:.2e} over {kept} cells")
    say(f"     the 1.32e-02 sits at n=12, rho=20 where cond(A) = 2.06e15. The identity was never "
        f"in question; the instrument was.")
    ok = bad == 0 and fl < 1e-6
    GG.verdict(ok,
               "the identity is EXACT. T0's FAIL was the gate, not the arithmetic, and T1-T5 are "
               "readable.",
               "the identity does not hold exactly, so loop 151's cost is not a cost and the loop "
               "is withdrawn.", emit=emit)
    say()
    say(f"     AND IT ESCAPED THE FIX. gate_guard.verdict, added at loop 150b, conditions a "
        f"SENTENCE on its gate, and in loop 151 it worked -- T3 and T4 printed their negatives")
    say(f"     correctly. T0's failure was a different animal: a TOLERANCE the instrument could "
        f"not meet. Conditioning prose on gates does nothing about that, and the honest record is")
    say(f"     that I fixed one class and went on committing the other. No second helper is added "
        f"here; naming the class is worth more than a wrapper I would forget to call.")
    gates["U1"] = bool(ok)
    res["u1"] = {"exact_settings": tested, "exact_disagreements": bad, "float_worst": fl,
                 "float_cells_kept": kept, "t0_reported": P["t0"]["worst_upsteps"],
                 "instance_number_in_arc": 4,
                 "prior_instances": ["149b N1", "150b S1", "150b S1 (transform arm)"],
                 "escaped_the_verdict_helper": True,
                 "class": "tolerance incompatible with the instrument, NOT prose contradicting a "
                          "gate", "pass": gates["U1"]}
    say(f"     U1 {'PASS' if gates['U1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- U2
    say("U2 RE-READ THE ANSWER WITH T0 SETTLED")
    t5 = {r["n"]: r for r in P["t5"]["frontier"] if r.get("reachable")}
    say(f"     n=8 -- loop 149's answer, the shortest ladder that REACHES {REQ:.2f}x:")
    say(f"       {t5[8]['transfers']:,.0f} transfers per degradation, E1 at "
        f"{t5[8]['e1_utilisation']:.0%} of capacity, ATP at "
        f"{t5[8]['atp_vs_translation']:.2f}x translation")
    say(f"     n=9  {t5[9]['transfers']:,.0f} transfers, E1 {t5[9]['e1_utilisation']:.0%}, "
        f"ATP {t5[9]['atp_vs_translation']:.3f}x")
    say(f"     n=10 {t5[10]['transfers']:,.0f} transfers, E1 {t5[10]['e1_utilisation']:.0%}, "
        f"ATP {t5[10]['atp_vs_translation']:.3f}x")
    say(f"     ONE ubiquitin between n=8 and n=9 drops the cost "
        f"{t5[8]['transfers'] / t5[9]['transfers']:.0f}-fold, because the minimum rho that still "
        f"delivers {REQ:.1f}x falls from {t5[8]['rho']:.2f} to {t5[9]['rho']:.2f} and the cost "
        f"goes as rho^(n-1).")
    win = P["t5"]["affordable_within_biochemical_bound"]
    GG.verdict(bool(win) and 8 not in win,
               f"n = 8 REACHES the amplification and cannot PAY for it. The window is n = {win}: "
               f"still inside the observed K48 range, but pinned to two values instead of one. "
               f"That is a narrowing of loop 149's prediction, not a contradiction of it -- 8 was "
               f"the shortest ladder that reaches 20.29x, and the shortest AFFORDABLE ladder is "
               f"one longer.",
               f"the cost does not separate n = 8 from its neighbours, so loop 149's answer stands "
               f"unchanged at 8.", emit=emit)
    gates["U2"] = bool(win)
    res["u2"] = {"n8": t5[8], "n9": t5[9], "n10": t5[10], "window": win,
                 "cost_drop_8_to_9": t5[8]["transfers"] / t5[9]["transfers"],
                 "narrowing_not_contradiction": True, "pass": gates["U2"]}
    say(f"     U2 {'PASS' if gates['U2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- U3
    say("U3 THE HINGE THE ANSWER TURNS ON")
    say(f"     loop 151 gated on the dilution-free load and carried the gross figure without ever "
        f"gating on it. That choice is worth {LOAD['gross (carried but never gated)'] / LOAD['dilution-free (loop 151 gated on this)']:.1f}x.")
    grid = np.logspace(-3, 6, 9001)
    out = {}
    for lname, lval in LOAD.items():
        rows, aff = [], []
        for n in range(2, NMAX + 1):
            ok_rho = [x for x in grid if amp(x, n, R_MED) >= REQ]
            if not ok_rho:
                continue
            rho = float(min(ok_rho))
            f = f_shape(rho, n)
            e1u = (lval * f / 3600.0) / E1_CAP
            at = lval * f * 2.0 / ATP_T
            good = bool(e1u <= 1.0 and at < 1.0)
            rows.append({"n": n, "rho": rho, "e1_utilisation": e1u, "atp_vs_translation": at,
                         "affordable": good})
            if good:
                aff.append(n)
        out[lname] = {"rows": rows, "affordable": aff,
                      "within_bound": [n for n in aff if n <= BIOCHEM_NMAX]}
        say(f"     {lname}: load {lval:,.0f} /h")
        say(f"       affordable at any n <= {NMAX}: {aff if aff else 'NONE'}")
        say(f"       affordable AND n <= {BIOCHEM_NMAX}: "
            f"{out[lname]['within_bound'] if out[lname]['within_bound'] else 'NONE'}")
        if rows:
            best = min(rows, key=lambda r: r["e1_utilisation"])
            say(f"       cheapest configuration anywhere: n={best['n']}, E1 at "
                f"{best['e1_utilisation']:.0%} of capacity")
    a = out["dilution-free (loop 151 gated on this)"]["within_bound"]
    b = out["gross (carried but never gated)"]["within_bound"]
    b_txt = str(b) if b else f"EMPTY -- no ladder of any length up to {NMAX} is affordable"
    GG.verdict(bool(a) == bool(b),
               "the window is the same under both loads, so the answer does not turn on the "
               "dilution question.",
               f"THE ANSWER TURNS ENTIRELY ON THIS. On the dilution-free load the window is "
               f"n = {a}; on the gross load it is {b_txt}. So 'can chains stay reversible out to "
               f"8' reduces to 'are Schwanhausser's half-lives division-corrected'. If the gross "
               f"figure is true proteolysis, the amplification mechanism is dead at every chain "
               f"length and CDC20 is the only branch left. That is the hinge, and it belongs in "
               f"the headline rather than in a caveat.", emit=emit)
    gates["U3"] = bool(a) == bool(b)
    res["u3"] = {"by_load": out, "window_dilution_free": a, "window_gross": b,
                 "answer_depends_on_load_choice": bool(a) != bool(b),
                 "hinge": "whether Schwanhausser protein half-lives are division-corrected",
                 "pass": gates["U3"]}
    say(f"     U3 {'PASS' if gates['U3'] else 'FAIL'}")
    say()

    say("=" * 100)
    for k in ("U1", "U2", "U3"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [POST-HOC -- written after loop 151's output]")
    say("=" * 100)

    man = RM.manifest(inputs=[OUT / "loop_reversibility_price.json",
                              OUT / "loop_proteostasis.json", OUT / "loop_pulse_equation.json",
                              OUT / "loop_capacity_ratio.json",
                              OUT / "loop_signalling_cost.json"],
                      available=NMAX - 1, used=NMAX - 1, selection="all", seed=SEED,
                      controls=["the identity settled in exact rational arithmetic, with the "
                                "float arm fenced by conditioning (U1)",
                                "the frontier recomputed on BOTH loads rather than the one the "
                                "loop chose (U3)",
                                "every conclusion emitted through gate_guard.verdict"],
                      note="POST-HOC. T0's FAIL was a tolerance the instrument could not meet, "
                           "the fourth instance of that defect in this arc and one the verdict "
                           "helper does not catch, because it conditions prose on gates and this "
                           "was not prose. With T0 settled the answer reads: n = 8 reaches the "
                           "amplification and cannot pay for it, the affordable window is n = "
                           "9-10 -- and that window exists only on the dilution-free load, which "
                           "makes the whole answer turn on whether Schwanhausser's half-lives are "
                           "division-corrected.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 151b -- post-hoc controls on the price of reversibility",
               "post_hoc": True, "predeclared": False, "manifest": man, "gates": gates, **res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_reversibility_price_controls.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_reversibility_price_controls.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
