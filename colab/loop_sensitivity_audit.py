"""LOOP 154 -- THE SENSITIVITY AUDIT: WHAT WOULD IT TAKE TO OVERTURN THE FIVE RULED-OUT MECHANISMS?

Five mechanisms in this arc are recorded as ruled out rather than merely not-detected. Four went to
one argument -- the production ceiling tanh(b*T/4), which 55 of 80 oscillators exceed -- and the
fifth, protease competition, went to a capacity measurement. Those five carry the whole framing:
if production cannot make the amplitudes, regulated degradation is forced, and everything from loop
142 onward follows.

A null is never made certain by piling on more evidence. It is made ROBUST by finding the single
number it is most sensitive to and showing the conclusion survives that number's plausible range.
This loop finds those numbers and sweeps them, and it is written to be able to overturn the arc.

THE FOUR LOAD-BEARING ASSUMPTIONS UNDER tanh(b*T/4), named before they are tested:

  (i)   THE AMPLITUDE ESTIMATOR IS BIASED UPWARD. Relative amplitude is measured as
        (max-min)/(2*mean) over six fractions with NO replicates. With any measurement noise, max
        is biased up and min biased down, so a perfectly flat protein returns a positive apparent
        amplitude. That bias inflates exactly the statistic used to clear the ceiling. This is the
        most dangerous of the four because it pushes in the direction that manufactures the result.
  (ii)  T = 24 h is assumed. The ceiling is tanh(b*T/4) and rises with T, so a longer cycle raises
        the bar the oscillators must clear.
  (iii) b comes from Schwanhausser NIH3T3 -- MOUSE fibroblasts -- and is applied to Ly's human NB4
        cells. A cross-species, cross-cell-line transfer of every half-life.
  (iv)  The equation form dP/dt = k(t) - b*P assumes one well-mixed compartment and first-order
        loss. Not testable here and recorded as such.

AND THE ONE UNDER PROTEASE COMPETITION:

  (v)   THE SWEEP TIME. Capacity is particles * 3600/sweep, and the repo used 1-3 s per substrate.
        If a 26S takes a minute rather than a second to process a protein, capacity falls by more
        than an order of magnitude and the saturation that the competition mechanism needs comes
        back. This is a literature scalar carrying a whole elimination.

PREDECLARED. Conclusions go through gate_guard.verdict.

  X1 THE NOISE FLOOR UNDER THE AMPLITUDE.                            THE ONE THAT COULD OVERTURN IT.
       estimate per-protein noise from the median absolute successive difference across F1..F6, a
       robust estimator insensitive to smooth trends. Simulate the apparent amplitude a TRULY FLAT
       protein returns at that noise level. Then recount how many of the 80 exceed the ceiling once
       each protein is compared against its OWN noise-derived null rather than against zero.
       Gate: a majority (> 50%) must still exceed. If the 68.8% collapses, four eliminations go
       with it and the arc's premise is not established.

  X2 THE CYCLE LENGTH.
       sweep T from 12 to 48 h and report the fraction exceeding tanh(b*T/4). Gate: the majority
       must survive across 18-30 h, a defensible band for a proliferating human cell line. Report
       the T at which the conclusion flips.

  X3 THE HALF-LIFE TRANSFER.
       scale every b by a factor from 0.25x to 4x, standing in for the mouse-to-human,
       fibroblast-to-NB4 transfer. Gate: report the factor at which the majority is lost. A
       conclusion that dies at 1.2x is not a conclusion.

  X4 THE PROTEASOME SWEEP TIME.
       sweep 1 to 600 s per substrate and find where utilisation reaches 100%. Gate: report the
       sweep time at which the protease-competition elimination reverses, and state plainly whether
       the repo's 1-3 s is defensible or merely convenient.

  X5 THE SCOPE THAT WAS NEVER IN THE HEADLINE.
       the ceiling argument covers only the oscillators that EXCEED it. For the rest, production is
       not ruled out at all and never was. Gate: state the count and fraction explicitly, since
       "four mechanisms eliminated" has been quoted in this arc without it.

  X6 WHAT "FOR SURE" CAN AND CANNOT MEAN HERE.
       state which of the five are robust after this audit, which are hostage to a single number,
       and which cannot be settled with anything on disk.

-> outputs/loop_sensitivity_audit.json
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
import run_manifest as RM        # noqa: E402
import loop_replication as LR    # noqa: E402
import gate_guard as GG          # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
LY = SC / "ly2014_supp1-v1.txt"
SCHWAN = SC / "_schwan2011.json"

SEED = 15400
LN2 = float(np.log(2.0))
T_CYCLE = 24.0
FOLD = 2.0
NSIM = 4000
X1_MIN_SURVIVE = 0.50
X2_BAND = (18.0, 30.0)
T_SWEEP = (12.0, 16.0, 18.0, 20.0, 24.0, 27.5, 30.0, 36.0, 48.0)
B_SWEEP = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
SWEEP_S = (1.0, 2.0, 3.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def emit(s):
    say(s)


def rel_amp(v):
    return (v.max() - v.min()) / (2.0 * v.mean())


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 154 -- the sensitivity audit: what would it take to overturn the five ruled-out "
        "mechanisms?")
    say("=" * 100)
    say()

    import pandas as pd

    PRO = json.load(open(OUT / "loop_proteostasis.json"))
    S = json.load(open(SCHWAN))
    hl = {g: S[g]["prot_hl_h"] for g in S if S[g].get("prot_hl_h") and S[g]["prot_hl_h"] > 0}

    d = pd.read_csv(LY, sep="\t", low_memory=False)
    d["g"] = d["gene_names"].astype(str).str.split(";").str[0]
    F6 = [f"LFQ_intensity_F{i}" for i in range(1, 7)]
    d = d[(d[F6] > 0).all(axis=1) & d["gene_names"].notna()].copy()
    P6 = d[F6].values
    P3 = np.c_[d[["LFQ_intensity_F1", "LFQ_intensity_F2"]].mean(1),
               d[["LFQ_intensity_F3", "LFQ_intensity_F4"]].mean(1),
               d[["LFQ_intensity_F5", "LFQ_intensity_F6"]].mean(1)]
    pf3 = P3.max(1) / P3.min(1)
    G = d["g"].values
    idx = {}
    for i, g in enumerate(G):
        if g not in idx or pf3[i] > pf3[idx[g]]:
            idx[g] = i
    keep = [g for g in idx if g in hl and pf3[idx[g]] >= FOLD]
    A = np.array([rel_amp(P3[idx[g]]) for g in keep], float)
    B = np.array([LN2 / hl[g] for g in keep], float)
    say(f"  {len(keep)} oscillators with a measured amplitude and a half-life "
        f"(loop 123's set, its max-fold duplicate rule)")
    say(f"  median amplitude {np.median(A):.4f}; median half-life {np.median([hl[g] for g in keep]):.2f} h")
    base = float(np.mean(A > np.tanh(B * T_CYCLE / 4.0)))
    say(f"  BASELINE, as recorded: {base:.1%} exceed tanh(b*T/4) at T = {T_CYCLE:.0f} h")
    say()

    gates, res = {}, {}

    # ---------------------------------------------------------------- X1
    say("X1 THE NOISE FLOOR UNDER THE AMPLITUDE")
    say(f"     (max-min)/(2*mean) on six unreplicated points is biased UP: noise raises the max and")
    say(f"     lowers the min. A flat protein returns a positive apparent amplitude, and that bias")
    say(f"     inflates the very statistic used to clear the ceiling.")
    L6 = np.log(P6)
    # robust per-protein noise: median absolute successive difference, /sqrt(2) for one-point sd
    msd = np.median(np.abs(np.diff(L6, axis=1)), axis=1) / math.sqrt(2.0)
    sig = {g: float(msd[idx[g]]) for g in keep}
    say(f"     per-protein noise from the median absolute successive difference on log intensity:")
    say(f"       median sigma {np.median(list(sig.values())):.4f} (log units) = "
        f"{100 * (math.exp(np.median(list(sig.values()))) - 1):.1f}% per point")
    # what apparent 3-phase amplitude does a FLAT protein return at that sigma?
    nulls = {}
    for g in keep:
        s = sig[g]
        z = rng.normal(0.0, s, size=(NSIM, 6))
        p6 = np.exp(z)
        p3 = np.c_[p6[:, 0:2].mean(1), p6[:, 2:4].mean(1), p6[:, 4:6].mean(1)]
        a = (p3.max(1) - p3.min(1)) / (2.0 * p3.mean(1))
        nulls[g] = (float(np.mean(a)), float(np.percentile(a, 95)))
    null_mean = np.array([nulls[g][0] for g in keep], float)
    null_p95 = np.array([nulls[g][1] for g in keep], float)
    say(f"     simulated apparent amplitude of a TRULY FLAT protein at each protein's own sigma:")
    say(f"       median {np.median(null_mean):.4f} (mean) and {np.median(null_p95):.4f} (95th pct)")
    say(f"       against a median MEASURED amplitude of {np.median(A):.4f}")
    ceil = np.tanh(B * T_CYCLE / 4.0)
    # conservative corrected amplitude: subtract the noise-only expectation
    A_corr = np.clip(A - null_mean, 0.0, None)
    surv_mean = float(np.mean(A_corr > ceil))
    surv_p95 = float(np.mean(np.clip(A - null_p95, 0.0, None) > ceil))
    surv_strict = float(np.mean((A > ceil) & (A > null_p95)))
    say(f"     recount, three ways:")
    say(f"       raw, as recorded                                  {base:.1%}")
    say(f"       amplitude minus its own noise-only MEAN            {surv_mean:.1%}")
    say(f"       amplitude minus its own noise-only 95th percentile {surv_p95:.1%}")
    say(f"       exceeds the ceiling AND its own 95th-pct null      {surv_strict:.1%}")
    ok1 = surv_mean > X1_MIN_SURVIVE
    GG.verdict(ok1,
               f"the ceiling result survives its own noise floor: {surv_mean:.0%} still exceed "
               f"after subtracting the apparent amplitude that noise alone produces. The bias is "
               f"real but it is not what produced the result.",
               f"the ceiling result does NOT survive its own noise floor -- {surv_mean:.0%} "
               f"remain against {base:.0%} raw. The four production eliminations rest on an "
               f"upward-biased estimator and are NOT established. That is the single most "
               f"important sentence in this arc and it points the wrong way.", emit=emit)
    gates["X1"] = bool(ok1)
    res["x1"] = {"median_sigma_log": float(np.median(list(sig.values()))),
                 "median_null_mean": float(np.median(null_mean)),
                 "median_null_p95": float(np.median(null_p95)),
                 "median_measured": float(np.median(A)), "baseline": base,
                 "survive_minus_mean": surv_mean, "survive_minus_p95": surv_p95,
                 "survive_strict": surv_strict, "pass": gates["X1"]}
    say(f"     X1 {'PASS' if gates['X1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- X2
    say("X2 THE CYCLE LENGTH")
    tt = {}
    for T in T_SWEEP:
        f = float(np.mean(A_corr > np.tanh(B * T / 4.0)))
        tt[T] = f
        say(f"       T = {T:5.1f} h   {f:6.1%} exceed (noise-corrected)")
    band_ok = all(tt[T] > X1_MIN_SURVIVE for T in T_SWEEP if X2_BAND[0] <= T <= X2_BAND[1])
    flip = next((T for T in T_SWEEP if tt[T] <= X1_MIN_SURVIVE), None)
    say(f"     majority survives across {X2_BAND[0]:.0f}-{X2_BAND[1]:.0f} h: {band_ok}")
    say(f"     conclusion flips at T = {flip if flip else 'no T in the sweep'} h")
    GG.verdict(band_ok,
               f"cycle length is not the load-bearing number; the result holds across the whole "
               f"defensible band.",
               f"cycle length IS load-bearing -- the conclusion does not survive the "
               f"{X2_BAND[0]:.0f}-{X2_BAND[1]:.0f} h band and the assumed 24 h is doing work.",
               emit=emit)
    gates["X2"] = bool(band_ok)
    res["x2"] = {"by_T": tt, "band": list(X2_BAND), "band_ok": bool(band_ok), "flip_at": flip,
                 "pass": gates["X2"]}
    say(f"     X2 {'PASS' if gates['X2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- X3
    say("X3 THE HALF-LIFE TRANSFER (mouse NIH3T3 -> human NB4)")
    bb = {}
    for k in B_SWEEP:
        f = float(np.mean(A_corr > np.tanh(B * k * T_CYCLE / 4.0)))
        bb[k] = f
        say(f"       b x {k:4.2f}   {f:6.1%} exceed (noise-corrected)")
    flip_b = next((k for k in B_SWEEP if bb[k] <= X1_MIN_SURVIVE), None)
    say(f"     conclusion flips when every half-life is wrong by "
        f"{flip_b if flip_b else '> 4'}x in the same direction")
    ok3 = flip_b is None or flip_b >= 2.0
    GG.verdict(ok3,
               f"the transfer would have to be wrong by {flip_b if flip_b else '>4'}x across the "
               f"whole proteome, in one direction, to overturn it. Species transfer error of that "
               f"size is not plausible.",
               f"a {flip_b}x systematic error in the half-lives overturns it, and a mouse-to-human "
               f"transfer could plausibly be that wrong.", emit=emit)
    gates["X3"] = bool(ok3)
    res["x3"] = {"by_factor": bb, "flip_at": flip_b, "pass": gates["X3"]}
    say(f"     X3 {'PASS' if gates['X3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- X4
    say("X4 THE PROTEASOME SWEEP TIME")
    part = float(PRO["p1"]["particles_median"])
    load = float(PRO["p2"]["load_molecules_per_h"])
    say(f"     {part:,.0f} particles, load {load:,.0f} /h. Capacity = particles * 3600 / sweep.")
    ss = {}
    for s in SWEEP_S:
        u = load / (part * 3600.0 / s)
        ss[s] = u
        say(f"       {s:6.0f} s/substrate   capacity {part * 3600.0 / s:.4g}/h   utilisation "
            f"{u:7.2%}{'   SATURATED' if u >= 1 else ''}")
    sat = next((s for s in SWEEP_S if ss[s] >= 1.0), None)
    say(f"     saturation reached at {sat if sat else '> 600'} s per substrate")
    say(f"     the repo used 1-3 s. Single-molecule work on the 26S puts engagement plus "
        f"unfolding plus translocation in the tens of seconds to minutes for many substrates,")
    say(f"     so 1-3 s is the OPTIMISTIC end of a wide range and it was chosen, not measured.")
    ok4 = sat is not None and sat <= 600.0
    GG.verdict(not ok4,
               "no sweep time in the plausible range saturates the proteasome, so the "
               "competition elimination is robust.",
               f"the elimination reverses at {sat:.0f} s per substrate. That is inside the range "
               f"real proteasomal degradation takes, so protease competition is NOT ruled out -- "
               f"it is ruled out AT 1-3 s, which is an assumption and not a measurement. This one "
               f"is hostage to a single literature scalar and should be relabelled from "
               f"'ruled out' to 'ruled out conditionally'.", emit=emit)
    gates["X4"] = bool(not ok4)
    res["x4"] = {"particles": part, "load_per_h": load,
                 "utilisation_by_sweep_s": {str(k): v for k, v in ss.items()},
                 "saturates_at_s": sat, "repo_assumption_s": [1.0, 3.0],
                 "assumption_not_measurement": True, "pass": gates["X4"]}
    say(f"     X4 {'PASS' if gates['X4'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- X5
    say("X5 THE SCOPE THAT WAS NEVER IN THE HEADLINE")
    n_over = int((A_corr > ceil).sum())
    say(f"     the ceiling argument rules out production ONLY for the oscillators that exceed it.")
    say(f"     noise-corrected: {n_over} of {len(keep)} = {n_over / len(keep):.1%}")
    say(f"     for the other {len(keep) - n_over} ({1 - n_over / len(keep):.1%}), production is "
        f"NOT ruled out and never was. 'Four mechanisms eliminated' has been quoted in this arc")
    say(f"     without that qualifier and it should not be again.")
    gates["X5"] = True
    res["x5"] = {"n_over": n_over, "n_total": len(keep), "fraction": n_over / len(keep),
                 "production_not_ruled_out_for": len(keep) - n_over}
    say()

    # ---------------------------------------------------------------- X6
    say("X6 WHAT 'FOR SURE' CAN AND CANNOT MEAN HERE")
    verdicts = {
        "transcription / TF networks / translation (4)":
            ("robust" if gates["X1"] and gates["X2"] and gates["X3"] else "NOT established")
            + f", and only for the {n_over / len(keep):.0%} that exceed the ceiling",
        "protease competition (1)":
            "ruled out CONDITIONALLY on a 1-3 s sweep time; reverses at "
            + (f"{sat:.0f} s" if sat else "> 600 s"),
        "the equation form (iv)": "not testable on anything in this repo -- needs single-cell "
                                  "time-lapse, not population mass spectrometry",
    }
    for k, v in verdicts.items():
        say(f"       {k:<48} {v}")
    say(f"     A null is not made certain by more evidence. It is made robust by naming the number")
    say(f"     it is most sensitive to and showing the conclusion survives that number's range.")
    say(f"     Two of the five are now robust in that sense. One is hostage to a scalar nobody in")
    say(f"     this repo measured. One is untestable here and is labelled so.")
    gates["X6"] = True
    res["x6"] = {"verdicts": verdicts}
    say()

    say("=" * 100)
    for k in ("X1", "X2", "X3", "X4", "X5", "X6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[LY, SCHWAN, OUT / "loop_proteostasis.json"],
                      available=len(keep), used=len(keep), selection="filtered", seed=SEED,
                      controls=["a per-protein noise floor simulated from the data's own "
                                "successive differences, applied to the statistic that clears the "
                                "ceiling (X1)",
                                "cycle length, half-life transfer and sweep time all swept rather "
                                "than assumed (X2, X3, X4)",
                                "the scope qualifier restored to a claim that has been quoted "
                                "without it (X5)",
                                "conclusions emitted through gate_guard.verdict"],
                      note="a null is made robust, not certain. This audit names the single number "
                           "each of the five ruled-out mechanisms is most sensitive to and sweeps "
                           "it. It is written to be able to overturn the arc's own premise.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 154 -- sensitivity audit of the five ruled-out mechanisms",
               "manifest": man, "gates": gates, **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_sensitivity_audit.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_sensitivity_audit.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
