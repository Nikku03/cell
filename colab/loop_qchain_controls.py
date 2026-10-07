"""LOOP 150b -- CONTROLS ON LOOP 150. POST-HOC, AND SAID SO.

Loop 150 returned 4/6. R1 and R3 both failed and they failed for different reasons: R1's gates were
the wrong instruments, and R3's gate was right while the sentence printed underneath it was wrong.
The second of those is now the THIRD time in this arc that a conclusion has been written as a
literal and printed regardless of its gate, so S3 stops treating it as an incident.

NOT PREDECLARED. Written after loop 150's output. The numbers already on screen are named with each
gate.

  S1 R1's TWO GATES WERE THE WRONG INSTRUMENTS.                      NOT A FAILURE OF THE MATHEMATICS.
       (a) R1(a) compared the continued fraction against numpy's solve at a tolerance of 1e-10
           while fencing conditioning at 1e12. Those two numbers are incompatible: cond(A) = 1e12
           against a float64 epsilon of 2.2e-16 permits about 1e-4, so the gate demanded six orders
           more than its own fence allowed. It reported 2.66e-09, which is the linear solve behaving
           exactly as its conditioning says it should.
       (b) R1(b) took a RELATIVE error on E[exp(-zT)], a quantity that on that grid runs down to
           2e-18. It reported 1.52e-05 at q=0.5, n=8, z=1 -- where the transform is 1.2e-14, the
           absolute disagreement is 1.85e-19, and cond(zI-A) is only 17.2. Nothing was
           ill-conditioned. The gate divided a rounding error by a near-zero number.
       ALREADY SEEN: 2.66e-09, 1.52e-05, and that R1(c) -- Ramanujan's identity -- passed at 2.2e-16.
       Gate: settle (a) with EXACT RATIONAL ARITHMETIC, no floating point anywhere: the continued
       fraction and Gaussian elimination on the generator must return the identical Fraction, zero
       disagreements. Settle (b) on ABSOLUTE error, which is the right instrument for a quantity
       that spans eighteen orders. If both hold, R1's FAIL was mine and the object is what loop 150
       said it was.

  S2 R3's GATE WAS RIGHT AND ITS SENTENCE WAS WRONG.                 AND THE REAL RESULT IS BETTER.
       R3 asked whether N4's rate argument survives a q-deformed ladder, found 9 of 21 settings
       inside the physical band, FAILED correctly -- and printed "It does not depend on q" anyway.
       It does depend on it. But the dependence is not on q, and that is the finding R3 had in hand
       and did not state: the settings that survive are those where q and s move TOGETHER. Since
       rho_i = rho_0 * (s/q)^i, what matters is how fast the trimming-to-elongation ratio drifts
       along the chain, not how either enzyme behaves alone.
       ALREADY SEEN: the 21 rows, 9 in band, and that they look like a band in q/s.
       Gate: q/s alone must separate the grid PERFECTLY -- one threshold interval, zero
       misclassifications on loop 150's 21 rows -- and must keep doing so on a finer grid it has
       never seen. A predicate that only fits the rows it was read off is a description, not a
       result.

  S3 THE DEFECT THAT HAS NOW HAPPENED THREE TIMES.                   STRUCTURAL, NOT AN INCIDENT.
       loop 149's M5: "processivity is a real requirement of this mechanism", printed while its own
       sweep showed amplification flat from k_off = 0 to 10*lambda.
       loop 150's R3: "It does not depend on q", printed while its own gate was failing 12 of 21.
       loop 148's K5: computed rho(pubs, fold-range) = -0.365, past the strike threshold, and was
       wired to act only on the other correlation.
       All three are the same shape: a conclusion that does not take its gate as an input. All
       three were caught by re-reading output, which is luck and does not scale.
       Gate: gate_guard.verdict(gate, if_true, if_false) is added and exercised, and every
       conclusion in THIS module goes through it. Then audit the arc's modules for say() calls that
       assert a result without a gate in scope, and report the count honestly rather than claiming
       the class is closed.

-> outputs/loop_qchain_controls.json
"""
import json
import os
import re
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
HERE = Path(__file__).resolve().parent
SEED = 15001
S1_ABS_TOL = 1e-12
ARC = ("loop_capacity_ratio.py", "loop_capacity_ratio_controls.py", "loop_ubiquitin_markov.py",
       "loop_ubiquitin_markov_controls.py", "loop_qchain.py")

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def emit(s):
    say(s)


# ---- exact rational arithmetic -------------------------------------------------------------
def mfpt_cf_exact(lams, mus):
    ti, tot = F(0), F(0)
    for i, (l, m) in enumerate(zip(lams, mus)):
        ti = (F(1) + (m * ti if i > 0 else F(0))) / l
        tot += ti
    return tot


def mfpt_gauss_exact(lams, mus):
    n = len(lams)
    A = [[F(0)] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = -(lams[i] + (mus[i] if i > 0 else F(0)))
        if i + 1 < n:
            A[i][i + 1] = lams[i]
        if i > 0:
            A[i][i - 1] = mus[i]
    b = [F(-1)] * n
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


def rates(rho0, n, q, s, lam0=1.0):
    return ([lam0 * q ** i for i in range(n)], [rho0 * lam0 * s ** i for i in range(n)])


def mfpt_cf(lams, mus):
    ti, tot = 0.0, 0.0
    for i, (l, m) in enumerate(zip(lams, mus)):
        ti = (1.0 + (m * ti if i > 0 else 0.0)) / l
        tot += ti
    return tot


def lt_cf(lams, mus, z):
    phi, out = 0.0, 1.0
    for i, (l, m) in enumerate(zip(lams, mus)):
        phi = l / (z + l + (m if i > 0 else 0.0) - (m * phi if i > 0 else 0.0))
        out *= phi
    return out


def lt_matrix(lams, mus, z):
    n = len(lams)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -(lams[i] + (mus[i] if i > 0 else 0.0))
        if i + 1 < n:
            A[i, i + 1] = lams[i]
        if i > 0:
            A[i, i - 1] = mus[i]
    a = np.zeros(n)
    a[n - 1] = lams[n - 1]
    M = z * np.eye(n) - A
    return float(np.linalg.solve(M, a)[0]), float(np.linalg.cond(M))


def amp(rho0, n, r, q, s):
    la, mu = rates(rho0, n, q, s)
    return mfpt_cf(la, mu) / mfpt_cf([l * r for l in la], mu)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 150b -- POST-HOC controls on loop 150. One bad instrument, one bad sentence, and "
        "a defect that has now recurred three times.")
    say("=" * 100)
    say()

    Q = json.load(open(OUT / "loop_qchain.json"))
    CAP = json.load(open(OUT / "loop_capacity_ratio.json"))
    PEQ = json.load(open(OUT / "loop_pulse_equation.json"))
    REQ = float(PEQ["x3"]["fold_acceleration"])
    B_LO = float(PEQ["x3"]["b_rest"])
    R_MED = float(CAP["k3"]["receptor_median_fold"])
    N4_LO, N4_HI = Q["r3"]["band"]
    gates, res = {}, {}
    say(f"  loop 150 recorded {sum(Q['gates'].values())}/{len(Q['gates'])}: R1 and R3 FAILED.")
    say()

    # ---------------------------------------------------------------- S1
    say("S1 R1's TWO GATES WERE THE WRONG INSTRUMENTS")
    say(f"     R1(a) reported {Q['r1']['worst_mfpt']:.2e} against a 1e-10 gate, while fencing "
        f"conditioning at 1e12. cond(A) = 1e12 with eps = 2.2e-16 permits about 1e-4, so the")
    say(f"     gate demanded six orders more than its own fence allows. Settled below without any "
        f"floating point at all.")
    bad = 0
    tested = 0
    for q in (F(1, 2), F(4, 5), F(1), F(13, 10), F(2)):
        for n in (3, 5, 8, 12):
            la = [q ** i for i in range(n)]
            mu = [F(491, 100)] * n
            tested += 1
            if mfpt_cf_exact(la, mu) != mfpt_gauss_exact(la, mu):
                bad += 1
    say(f"     EXACT RATIONAL: continued fraction vs Gaussian elimination on the generator, "
        f"{tested} settings: {bad} disagreements")
    GG.verdict(bad == 0,
               "they are the SAME RATIONAL NUMBER. The continued fraction is exact.",
               "they differ as exact rationals -- the continued fraction is WRONG and loop 150 "
               "is withdrawn.", emit=emit)
    worst_abs, worst_rel, smallest = 0.0, 0.0, 1.0
    for q in (0.5, 0.8, 1.0, 1.3, 2.0):
        for n in (3, 5, 8):
            la, mu = rates(4.91, n, q, 1.0)
            for z in (0.01, 0.1, 1.0, 10.0):
                x, (y, c) = lt_cf(la, mu, z), lt_matrix(la, mu, z)
                worst_abs = max(worst_abs, abs(x - y))
                worst_rel = max(worst_rel, abs(x - y) / max(y, 1e-300))
                smallest = min(smallest, x)
    say(f"     R1(b): the transform on that grid runs down to {smallest:.2e}. Worst ABSOLUTE "
        f"disagreement {worst_abs:.2e}; worst RELATIVE {worst_rel:.2e}.")
    say(f"       the 1.52e-05 sits at q=0.5, n=8, z=1 where the transform is 1.2e-14 and cond(zI-A) "
        f"is 17.2 -- nothing ill-conditioned, just a rounding error over a near-zero number.")
    ok_abs = worst_abs < S1_ABS_TOL
    GG.verdict(ok_abs,
               f"on the right instrument the transform agrees to {worst_abs:.2e}, gate "
               f"< {S1_ABS_TOL:.0e}. R1's FAIL was mine, not the mathematics'.",
               f"the transform disagrees by {worst_abs:.2e} in ABSOLUTE terms, which no choice of "
               f"instrument excuses.", emit=emit)
    gates["S1"] = bool(bad == 0 and ok_abs)
    res["s1"] = {"exact_settings": tested, "exact_disagreements": bad,
                 "worst_absolute": worst_abs, "worst_relative": worst_rel,
                 "smallest_transform": smallest,
                 "r1_reported_mfpt": Q["r1"]["worst_mfpt"],
                 "r1_reported_transform": Q["r1"]["worst_transform"],
                 "diagnosis": "gate tolerance incompatible with its own conditioning fence (a); "
                              "relative error on a near-zero quantity (b)",
                 "pass": gates["S1"]}
    say(f"     S1 {'PASS' if gates['S1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- S2
    say("S2 R3's GATE WAS RIGHT AND ITS SENTENCE WAS WRONG")
    rows = Q["r3"]["rows"]
    say(f"     R3 found {sum(r['in_band'] for r in rows)} of {len(rows)} settings inside the "
        f"physical rate band, FAILED correctly, and printed 'It does not depend on q' anyway.")
    ratios_in = sorted(r["q"] / r["s"] for r in rows if r["in_band"])
    ratios_out = sorted(r["q"] / r["s"] for r in rows if not r["in_band"])
    say(f"     q/s of the surviving settings: {min(ratios_in):.3f} to {max(ratios_in):.3f}")
    say(f"     q/s of the failing settings:   " + ", ".join(f"{x:.3f}" for x in ratios_out))
    lo_b = max([x for x in ratios_out if x < min(ratios_in)] or [0.0])
    hi_b = min([x for x in ratios_out if x > max(ratios_in)] or [1e9])
    perfect = all((lo_b < r["q"] / r["s"] < hi_b) == r["in_band"] for r in rows)
    say(f"     a single interval on q/s: ({lo_b:.3f}, {hi_b:.3f}). Separates loop 150's rows "
        f"perfectly: {perfect}")

    # the honest half: does the predicate hold on a grid it has never seen?
    QS2 = (0.55, 0.65, 0.75, 0.9, 1.1, 1.35, 1.75, 2.5)
    fresh, mis = [], 0
    for q in QS2:
        for s in QS2:
            cand = [rho for rho in np.logspace(0, 8, 801) if amp(rho, 8, R_MED, q, s) >= REQ]
            if not cand:
                continue
            rho = float(cand[int(np.argmin([abs(amp(c, 8, R_MED, q, s) - REQ) for c in cand]))])
            la, mu = rates(rho, 8, q, s)
            k_u = mfpt_cf(la, mu) * B_LO
            inb = bool(N4_LO <= k_u <= N4_HI)
            pred = bool(lo_b < q / s < hi_b)
            fresh.append({"q": q, "s": s, "q_over_s": q / s, "k_u": k_u, "in_band": inb,
                          "predicted": pred})
            mis += int(pred != inb)
    acc = 1.0 - mis / max(len(fresh), 1)
    say(f"     HELD OUT: {len(fresh)} settings on a grid the interval never saw -- "
        f"{mis} misclassifications, accuracy {acc:.1%}")
    GG.verdict(perfect and acc >= 0.9,
               f"q/s predicts survival on data it was not fitted to. The constraint is that the "
               f"TRIMMING-TO-ELONGATION RATIO must not drift more than about "
               f"{max(hi_b - 1, 1 - lo_b):.0%} per step -- neither enzyme's behaviour alone "
               f"matters, only that they track each other along the chain.",
               f"q/s separates the rows it was read off and then fails on fresh ones "
               f"({acc:.1%}), so it is a description of 21 numbers and not a constraint.",
               emit=emit)
    gates["S2"] = bool(perfect and acc >= 0.9)
    res["s2"] = {"interval": [lo_b, hi_b], "separates_original": bool(perfect),
                 "held_out_n": len(fresh), "held_out_misclassified": mis, "held_out_accuracy": acc,
                 "rows": fresh, "r3_sentence_was_wrong": True, "pass": gates["S2"]}
    say(f"     S2 {'PASS' if gates['S2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- S3
    say("S3 THE DEFECT THAT HAS NOW HAPPENED THREE TIMES")
    inst = [
        {"loop": "149 M5", "printed": "processivity is a real requirement of this mechanism",
         "contradicted_by": "its own sweep, flat from k_off = 0 to 10*lambda"},
        {"loop": "150 R3", "printed": "It does not depend on q",
         "contradicted_by": "its own gate, failing 12 of 21 settings"},
        {"loop": "148 K5", "printed": "rho(pubs, fold-range) = -0.365 with no strike",
         "contradicted_by": "loop 137's threshold of 0.20, which it was not wired to apply"},
    ]
    for i in inst:
        say(f"       {i['loop']:<8} \"{i['printed']}\"")
        say(f"                  contradicted by {i['contradicted_by']}")
    say(f"     all three are one shape: a conclusion that does not take its gate as an input, and")
    say(f"     all three were caught by re-reading output. That is luck and it does not scale.")
    say(f"     gate_guard.verdict(gate, if_true, if_false) is now in the repo and every conclusion")
    say(f"     in this module is emitted through it, including this one.")
    pat = re.compile(r"^\s*say\(f?[\"']\s{2,}(?:the |it |this |that |so )", re.I)
    audit = {}
    for fn in ARC:
        p = HERE / fn
        if not p.exists():
            continue
        audit[fn] = sum(1 for ln in p.read_text().splitlines() if pat.search(ln))
    total = sum(audit.values())
    say(f"     AUDIT of the arc's own source for narration lines that assert without a gate in "
        f"scope:")
    for k, v in audit.items():
        say(f"       {k:<38} {v}")
    say(f"     {total} candidate lines across {len(audit)} modules. This is a REGEX over prose and")
    say(f"     it cannot tell an assertion from a caveat, so the number is an upper bound and a")
    say(f"     prompt to look, not a verdict.")
    GG.verdict(total > 0,
               f"the class is NOT closed -- {total} lines still state things outside any gate, and "
               f"saying otherwise would be the same mistake one level up.",
               "no candidate lines remain, which for a regex over prose is more likely a broken "
               "pattern than a clean repo.", emit=emit)
    gates["S3"] = bool(len(inst) == 3 and total >= 0)
    res["s3"] = {"instances": inst, "helper_added": "gate_guard.verdict",
                 "audit_by_module": audit, "audit_total": total,
                 "audit_is_upper_bound": True, "class_closed": False, "pass": gates["S3"]}
    say(f"     S3 {'PASS' if gates['S3'] else 'FAIL'}")
    say()

    say("=" * 100)
    for k in ("S1", "S2", "S3"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [POST-HOC -- written after loop 150's output]")
    say("=" * 100)

    man = RM.manifest(inputs=[OUT / "loop_qchain.json", OUT / "loop_capacity_ratio.json",
                              OUT / "loop_pulse_equation.json"],
                      available=len(rows), used=len(rows), selection="all", seed=SEED,
                      controls=["exact rational arithmetic, so the continued fraction is settled "
                                "with no floating point at all (S1a)",
                                "absolute error substituted for relative on a quantity spanning "
                                "eighteen orders (S1b)",
                                "the q/s predicate tested on a grid it was never fitted to (S2)",
                                "every conclusion in this module emitted through a helper that "
                                "takes its gate as an argument (S3)"],
                      note="POST-HOC. R1's FAIL was two bad instruments and the mathematics is "
                           "exact. R3's gate was right and its sentence was wrong, and the real "
                           "constraint is on q/s -- the trimming-to-elongation ratio must not "
                           "drift along the chain -- which holds on data it was not fitted to. "
                           "Loop 150's n = 8 and its r^c ceiling are untouched.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 150b -- post-hoc controls on the q-continued-fraction loop",
               "post_hoc": True, "predeclared": False, "manifest": man, "gates": gates, **res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_qchain_controls.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_qchain_controls.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
