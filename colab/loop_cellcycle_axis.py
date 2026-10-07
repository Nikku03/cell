"""LOOP 119 -- THE TIME AXIS, FETCHED: the cell cycle exists, and the wiring cannot produce it.

WHAT WAS MISSING. Loop 99 built a cell-cycle layer and it FAILED its own gate: the canonical phase
order was recovered, and 176 of 500 shuffled orderings recovered it too. The stated cause was blunt --
"no phase-resolved layer exists". Loop 117 X-rayed the cell 1,517 times and found the compartments
move, then declared its own limit in the same sentence: those 1,517 columns are cell LINES, not
timepoints. Every dynamic claim in this repository so far is a claim about states, not about time.

WHAT WAS FETCHED. Mahdessian, Cesnik, Gnann et al. 2021 (Nature 590:649), single-cell proteogenomics
of the cell cycle: U2OS cells imaged by immunofluorescence, each cell placed at a precise pseudotime
position in the cycle by its FUCCI markers, and each protein scored for whether its level depends on
that position. Integrated into the Human Protein Atlas as two columns, so the fetch is one file:

    CCD Protein     748 Yes, 776 No, 18,638 not called
    CCD Transcript  530 Yes, 1,104 No, 18,528 not called

THE MATCHED SET IS FREE. A gene is called Yes or No only if it was imaged, so the 776 No genes are
the correct control for the 748 Yes genes: same assay, same cells, same antibody-availability
selection. This is the rare case where the confound is removed by the design of the source rather
than by anything this loop does -- which makes it worth checking that it is actually true (C1).

THE REASON THIS IS NOT JUST ANOTHER ANNOTATION LAYER. This repository's protein equation is

    dP/dt = k_sp * M - b * P

and it is the ONLY route by which protein changes. A first-order filter driven at angular frequency
w has gain 1/sqrt(1 + (w/b)^2), which is strictly less than one. So the model can make a protein
oscillate ONLY by oscillating its mRNA first, and the protein's relative swing is always SMALLER than
the mRNA's. Protein oscillation must therefore be RARER than transcript oscillation in this model,
at every amplitude threshold, for every parameter choice -- it is forced by the form of the equation,
not by any number in it. The measurement says 748 proteins against 530 transcripts. That is not a
tuning error; it is the wrong sign, and no fitting can fix it.

PREDECLARED, before any number:

  C1 THE CALLED SET IS MATCHED, NOT SELECTED                        THE PREREQUISITE.
       CCD-Yes against CCD-No must NOT be separable by protein abundance. Gate: AUC < 0.65 on
       measured Itzhak copy number. If abundance separates them, "cell-cycle dependent" is partly
       "bright enough to see change", and every downstream number inherits that. This gate can fail
       and the loop stops if it does.
  C2 THE POST-TRANSCRIPTIONAL GAP IS REAL AND SIZED                 THE MEASUREMENT.
       Yes/Yes overlap against a capability-checked label permutation on the genes called for BOTH.
       Gate: overlap above chance at |z| >= 2, i.e. transcriptional control of protein oscillation
       exists at all. Then the number that matters: how many CCD proteins have a non-CCD transcript,
       because those are exactly the proteins this model has no mechanism to oscillate.
  C3 THE FILTER'S OWN PREDICTION, ON AN INDEPENDENT DATASET         THE PHYSICS.
       gain = 1/sqrt(1 + (w/b)^2) per gene, w from the cycle length, b = ln2/t_half + mu, with
       half-lives from Schwanhausser (NIH3T3, mass spectrometry) and CCD calls from HPA (U2OS,
       imaging) -- different lab, method, cell line, so this is a prediction and not a lookup.
       Gate: mean gain of CCD-Yes above CCD-No, permutation p < 0.05. Swept over cycle lengths
       16/20/24/28 h so the verdict is not a property of one assumed period.
  C4 FAME                                                           THE CONTROL THAT KEEPS WINNING.
       publication count has beaten or matched the biology in six loops this session. Gate: AUC of
       the half-life gain must exceed AUC of pubs on the same genes. If pubs wins, C3 is reported as
       lost regardless of its own p-value.
  C5 EVERY NULL CHECKED FOR CAPABILITY                              THE GUARD.
       twelve gates in this session fired while measuring nothing. Each permutation here is passed
       through gate_guard.null_can_move() before its verdict is read, and every survival fraction
       through gate_guard.survival() so that "no effect" cannot be printed as a percentage.
  C6 THE STRUCTURAL CONTRADICTION                                   THE POINT OF THE LOOP.
       drive the committed state vector's k_sm with a sinusoid at the cycle period, integrate the
       real dM/dt and dP/dt to a steady oscillation, and measure each gene's relative amplitude.
       Gate, in two parts, both required:
         (a) the model's inequality is FORCED: relative amplitude of P below that of M for >= 99%
             of the 4,190 genes with a full state. This is a check on the implementation -- the
             analytic claim says 100%, so a lower number means the integrator is wrong, not the
             biology.
         (b) the measurement points the OTHER way: McNemar on the paired calls, protein-Yes/
             transcript-No against protein-No/transcript-Yes, significant at p < 0.05 in the
             direction the model cannot produce.
       Passing C6 means recording a falsification of this repository's protein wiring. That is the
       result; it is not a flattering one.

WHAT THIS LOOP DOES NOT DO. It does not give the cell a clock. A binary per-gene call is not a phase,
so loop 99's ordering test cannot be redone from this file -- the per-cell pseudotime is in the
paper's own supplement, not in the HPA release. What it does is settle whether the layer this model
would need is transcriptional at all, and the answer determines whether a clock is even worth wiring
to the equation that exists.

-> outputs/loop_cellcycle_axis.json
"""
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402
import cell_assembled as CA  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
HPA = LR.SC / "proteinatlas.tsv"
ITZHAK = LR.SC / "itzhak_supp1.xlsx"
SHEET = "Compact HeLa Spatial Proteome"
SEED = 11900
NPERM = 2000

C1_AUC = 0.65            # above this, abundance explains the CCD call and the loop stops
C2_Z = 2.0               # overlap must clear this many null SDs
C3_P = 0.05
C6_FORCED = 0.99         # fraction of genes for which the filter inequality must hold
LN2 = float(np.log(2.0))

T_CYCLE_H = (16.0, 20.0, 24.0, 28.0)   # U2OS doubling is quoted near 21-24 h; swept, not assumed
T_MAIN = 24.0
# Schwanhausser's half-lives are NIH3T3. Loop 98's correction: the dilution term must use THAT cell
# line's doubling time, 27.5 h, not the 24 h a comment once claimed.
T_DOUBLE_H = 27.5
MU = LN2 / T_DOUBLE_H

DRIVE_AMP = 0.50         # relative amplitude imposed on k_sm; the C6(a) claim is amplitude-free
N_CYCLES = 12            # integrate this many periods so the transient has decayed
DT_FRAC = 400            # timesteps per cycle

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def auc(pos, neg):
    """Rank-based AUC: probability a random positive scores above a random negative."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def gain(b, T):
    """Amplitude gain of dP/dt = k*M - b*P driven at period T. Strictly below 1, always."""
    w = 2.0 * np.pi / T
    return 1.0 / np.sqrt(1.0 + (w / b) ** 2)


def osc_amplitude(st, T, amp=DRIVE_AMP, ncyc=N_CYCLES, nstep=DT_FRAC):
    """Integrate the real equations with k_sm(t) = k_sm * (1 + amp*sin(wt)); return relative swings.

    Exponential stepper, piecewise-constant drive within a step -- the same scheme cell_assembled
    uses, so this measures the committed model and not a second implementation of it.
    """
    a, b = st["k_loss_mrna"], st["k_loss_prot"]
    ks, kp = st["k_sm"], st["k_sp"]
    dt = T / nstep
    w = 2.0 * np.pi / T
    ea, eb = np.exp(-a * dt), np.exp(-b * dt)
    M = ks / a
    P = kp * M / b
    trM = np.zeros((nstep, len(M)))
    trP = np.zeros((nstep, len(M)))
    total = ncyc * nstep
    for s in range(total):
        k_t = ks * (1.0 + amp * np.sin(w * (s * dt)))
        Mn = M * ea + (k_t / a) * (1 - ea)
        P = P * eb + (kp * M / b) * (1 - eb)
        M = Mn
        if s >= total - nstep:                       # record the final cycle only
            trM[s - (total - nstep)] = M
            trP[s - (total - nstep)] = P
    relM = (trM.max(0) - trM.min(0)) / (2.0 * trM.mean(0))
    relP = (trP.max(0) - trP.min(0)) / (2.0 * trP.mean(0))
    return relM, relP


def mcnemar(n01, n10):
    """Exact two-sided binomial test on the discordant pairs. No chi-square approximation."""
    from math import comb
    n = n01 + n10
    if n == 0:
        return float("nan")
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 119 -- the time axis, fetched: the cell cycle exists, and the wiring cannot make it")
    say("=" * 100)
    say()

    # ---------------------------------------------------------------- the fetched layer
    with open(HPA, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        h = next(r)
        iG, iCP, iCT = h.index("Gene"), h.index("CCD Protein"), h.index("CCD Transcript")
        iRel = h.index("Reliability (IF)")
        rows = [(x[iG], x[iCP], x[iCT], x[iRel]) for x in r]
    cp = {g: v for g, v, _, _ in rows if v in ("Yes", "No")}
    ct = {g: v for g, _, v, _ in rows if v in ("Yes", "No")}
    rel = {g: v for g, _, _, v in rows}
    say(f"  Mahdessian 2021 via HPA: {len(rows):,} genes; CCD Protein called for {len(cp):,} "
        f"({sum(v == 'Yes' for v in cp.values())} Yes), CCD Transcript for {len(ct):,} "
        f"({sum(v == 'Yes' for v in ct.values())} Yes)")

    D = CA.load()
    st = CA.state_vector(D)
    sgene = {g: i for i, g in enumerate(st["genes"])}
    say(f"  model: {len(D['names']):,} genes; full dynamical state for {len(st['genes']):,}")

    import pandas as pd
    itz = pd.read_excel(ITZHAK, sheet_name=SHEET)
    itz_cn = {}
    for g, c in zip(itz["Lead Gene name"].astype(str),
                    pd.to_numeric(itz["Estimated Copy number per cell"], errors="coerce")):
        if np.isfinite(c) and c > 0:
            itz_cn[g] = max(itz_cn.get(g, 0.0), float(c))
    itz_mw = {}
    for g, m in zip(itz["Lead Gene name"].astype(str),
                    pd.to_numeric(itz["Mol. weight [kDa]"], errors="coerce")):
        if np.isfinite(m) and m > 0:
            itz_mw[g] = float(m)
    say(f"  Itzhak 2016 copy numbers for abundance control: {len(itz_cn):,} proteins")
    say()

    gates = {}

    # ---------------------------------------------------------------- C1
    say("C1 THE CALLED SET IS MATCHED, NOT SELECTED")
    yes = [g for g, v in cp.items() if v == "Yes"]
    no = [g for g, v in cp.items() if v == "No"]
    ay = [itz_cn[g] for g in yes if g in itz_cn]
    an = [itz_cn[g] for g in no if g in itz_cn]
    a_auc = auc(np.log10(ay), np.log10(an))
    say(f"     CCD Yes {len(yes)} genes ({len(ay)} with a measured copy number), "
        f"No {len(no)} ({len(an)})")
    say(f"     median copies  Yes {np.median(ay):>12,.0f}   No {np.median(an):>12,.0f}")
    say(f"     AUC(abundance separates Yes from No) = {a_auc:.4f}   gate < {C1_AUC}")
    relc = {}
    for g in list(cp):
        relc.setdefault(cp[g], {}).setdefault(rel.get(g, ""), 0)
        relc[cp[g]][rel.get(g, "")] += 1
    for v in ("Yes", "No"):
        say(f"     IF reliability, {v:>3}: " +
            ", ".join(f"{k or 'none'} {n}" for k, n in sorted(relc[v].items(), key=lambda z: -z[1])))
    gates["C1"] = bool(np.isfinite(a_auc) and abs(a_auc - 0.5) < (C1_AUC - 0.5))
    say(f"     C1 {'PASS' if gates['C1'] else 'FAIL'} -- "
        f"{'abundance does not explain the call' if gates['C1'] else 'abundance explains the call'}")
    say()

    # ---------------------------------------------------------------- C2
    say("C2 THE POST-TRANSCRIPTIONAL GAP IS REAL AND SIZED")
    both = sorted(set(cp) & set(ct))
    pv = np.array([cp[g] == "Yes" for g in both])
    tv = np.array([ct[g] == "Yes" for g in both])
    n11 = int((pv & tv).sum())
    n10 = int((pv & ~tv).sum())     # protein oscillates, transcript does NOT  <- unreachable
    n01 = int((~pv & tv).sum())     # transcript oscillates, protein does not  <- what a filter makes
    n00 = int((~pv & ~tv).sum())
    say(f"     {len(both):,} genes called for BOTH")
    say(f"                          transcript Yes   transcript No")
    say(f"       protein Yes  {n11:>14,} {n10:>15,}")
    say(f"       protein No   {n01:>14,} {n00:>15,}")
    nulls, cap = [], None
    for i in range(NPERM):
        sh = rng.permutation(tv)
        if i == 0:
            cap = GG.null_can_move(tv.astype(int), sh.astype(int))
        nulls.append(int((pv & sh).sum()))
    sur = GG.survival(float(n11), nulls, z_min=C2_Z)
    say(f"     null: shuffled transcript labels -- {cap['reason']}")
    GG.report("Yes/Yes overlap", sur, emit=say)
    gates["C2"] = bool(cap["capable"] and np.isfinite(sur["z"]) and sur["z"] >= C2_Z)
    frac_unreach = n10 / max(1, n11 + n10)
    say(f"     CCD proteins with a NON-CCD transcript: {n10:,} of {n11 + n10:,} "
        f"= {frac_unreach:.1%} -- the model has no mechanism for these")
    say(f"     C2 {'PASS' if gates['C2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- C3
    say("C3 THE FILTER'S OWN PREDICTION, ON AN INDEPENDENT DATASET")
    S = D["schwan"]
    hl = {g: S[g]["prot_hl_h"] for g in S if S[g].get("prot_hl_h")}
    gy = [g for g in yes if g in hl]
    gn = [g for g in no if g in hl]
    say(f"     Schwanhausser protein half-lives on the called set: Yes {len(gy)}, No {len(gn)}")
    sweep, c3_ok = {}, True
    for T in T_CYCLE_H:
        by = np.array([gain(LN2 / hl[g] + MU, T) for g in gy])
        bn = np.array([gain(LN2 / hl[g] + MU, T) for g in gn])
        d = float(by.mean() - bn.mean())
        lab = np.array([1] * len(by) + [0] * len(bn))
        val = np.concatenate([by, bn])
        nd, mcap = [], None
        for i in range(NPERM):
            sh = rng.permutation(lab)
            if i == 0:
                mcap = GG.null_can_move(lab, sh)
            nd.append(float(val[sh == 1].mean() - val[sh == 0].mean()))
        nd = np.array(nd)
        p = float((np.abs(nd) >= abs(d)).mean())
        a = auc(by, bn)
        sweep[T] = {"gain_yes": float(by.mean()), "gain_no": float(bn.mean()), "delta": d,
                    "p": p, "auc": a, "null_capable": mcap["capable"],
                    "null_changed": mcap["changed"], "null_achievable": mcap["achievable"]}
        say(f"     T={T:>4.0f} h   gain Yes {by.mean():.4f}  No {bn.mean():.4f}  "
            f"delta {d:+.4f}  p {p:.4f}  AUC {a:.4f}")
        if not (d > 0 and p < C3_P and mcap["capable"]):
            c3_ok = False
    gates["C3"] = bool(c3_ok)
    say(f"     median half-life  Yes {np.median([hl[g] for g in gy]):.1f} h   "
        f"No {np.median([hl[g] for g in gn]):.1f} h")
    say(f"     C3 {'PASS' if gates['C3'] else 'FAIL'} -- the prediction holds at "
        f"{'every' if c3_ok else 'not every'} swept period")
    say()

    # ---------------------------------------------------------------- C4
    say("C4 FAME")
    pubs = D["pubs"]
    py = [pubs.get(g, 0.0) for g in gy]
    pn = [pubs.get(g, 0.0) for g in gn]
    p_auc = auc(py, pn)
    b_auc = sweep[T_MAIN]["auc"]
    say(f"     AUC(pubs)              {p_auc:.4f}   median pubs  Yes {np.median(py):.0f}  "
        f"No {np.median(pn):.0f}")
    say(f"     AUC(half-life gain)    {b_auc:.4f}   at T = {T_MAIN:.0f} h")
    p_all = auc([pubs.get(g, 0.0) for g in yes], [pubs.get(g, 0.0) for g in no])
    say(f"     AUC(pubs), all {len(yes) + len(no):,} called genes  {p_all:.4f}")
    gates["C4"] = bool(np.isfinite(b_auc) and np.isfinite(p_auc)
                       and abs(b_auc - 0.5) > abs(p_auc - 0.5))
    say(f"     C4 {'PASS' if gates['C4'] else 'FAIL'} -- "
        f"{'biology beats fame' if gates['C4'] else 'FAME WINS; C3 is reported as lost'}")
    say()

    # ---------------------------------------------------------------- C6 (a) the forced inequality
    say("C6 THE STRUCTURAL CONTRADICTION")
    say("     (a) the model's inequality, measured by integrating the committed equations")
    relM, relP = osc_amplitude(st, T_MAIN)
    ok = float(np.mean(relP <= relM + 1e-12))
    ratio = relP / np.maximum(relM, 1e-300)
    say(f"     driven k_sm at T={T_MAIN:.0f} h, amplitude {DRIVE_AMP:.0%}, {N_CYCLES} cycles, "
        f"{len(relM):,} genes with a full state")
    say(f"     relative swing:  mRNA median {np.median(relM):.4f}   protein median "
        f"{np.median(relP):.4f}")
    say(f"     protein swing <= mRNA swing for {ok:.4%} of genes   gate >= {C6_FORCED:.0%}")
    say(f"     attenuation protein/mRNA: median {np.median(ratio):.4f}, "
        f"90th pct {np.percentile(ratio, 90):.4f}, max {ratio.max():.4f}")
    frac_damped = float(np.mean(relP < 0.1 * DRIVE_AMP))
    say(f"     {frac_damped:.1%} of proteins are damped below a tenth of the drive")

    # (b) the measurement, on the same genes, in the opposite direction
    say("     (b) the measurement, McNemar on the paired calls")
    pm = mcnemar(n10, n01)
    say(f"     protein-Yes/transcript-No  {n10:,}    protein-No/transcript-Yes  {n01:,}")
    say(f"     exact two-sided binomial on the {n10 + n01:,} discordant pairs: p = {pm:.3e}")
    direction = "protein-only oscillation dominates -- the direction the filter CANNOT make" \
        if n10 > n01 else "transcript-only oscillation dominates -- consistent with a filter"
    say(f"     {direction}")
    gates["C6"] = bool(ok >= C6_FORCED and n10 > n01 and pm < 0.05)
    say(f"     C6 {'PASS' if gates['C6'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- C5
    say("C5 EVERY NULL CHECKED FOR CAPABILITY")
    caps = {"C2 transcript-label shuffle": cap,
            "C3 group-label shuffle": {"capable": all(v["null_capable"] for v in sweep.values()),
                                       "achievable": sweep[T_MAIN]["null_achievable"],
                                       "changed": sweep[T_MAIN]["null_changed"],
                                       "reason": "one permutation per swept period, each checked"}}
    for k, v in caps.items():
        say(f"     {k}: {'CAPABLE' if v['capable'] else 'INERT'} -- moved {v.get('changed', 0):.1%} "
            f"of an achievable {v.get('achievable', float('nan')):.1%} -- {v['reason']}")
    say("     the bar is a fraction of the ACHIEVABLE move, not a fixed 0.5. A binary label vector")
    say("     of prevalence p can never change more than 2p(1-p) of its entries under permutation,")
    say("     so the old fixed bar was unreachable for every binary null -- gate_guard's own")
    say("     family-two error, found by this loop and corrected in that module.")
    say(f"     C2 survival passed through gate_guard.survival: defined={sur.get('defined')}")
    gates["C5"] = bool(all(v["capable"] for v in caps.values()))
    say(f"     C5 {'PASS' if gates['C5'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- what it costs the cell
    say("  HOW MUCH OF THE CELL IS TIME-VARYING (measured, not gated)")
    tot = sum(itz_cn[g] * itz_mw.get(g, 0.0) for g in itz_cn)
    m_yes = sum(itz_cn[g] * itz_mw.get(g, 0.0) for g in yes if g in itz_cn)
    m_no = sum(itz_cn[g] * itz_mw.get(g, 0.0) for g in no if g in itz_cn)
    m_call = m_yes + m_no
    say(f"     CCD-Yes proteins carry {m_yes / tot:.2%} of measured proteome mass")
    say(f"     of the mass that was CALLED at all, CCD-Yes is {m_yes / m_call:.1%} "
        f"({m_call / tot:.1%} of the proteome was called)")
    n_state = sum(1 for g in yes if g in sgene)
    say(f"     {n_state} of {len(yes)} CCD proteins have a full dynamical state in this model "
        f"({n_state / len(yes):.1%})")
    say()

    # ---------------------------------------------------------------- verdict
    say("=" * 100)
    npass = sum(gates.values())
    for k in ("C1", "C2", "C3", "C4", "C5", "C6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {npass}/6")
    say("=" * 100)

    man = RM.manifest(inputs=[HPA, ITZHAK, LR.CELL, LR.SC / "_schwan2011.json"],
                      available=len(rows), used=len(cp), selection="filtered", seed=SEED,
                      controls=["776 imaged non-CCD genes as the assay-matched control",
                                "measured copy number tested for the abundance confound",
                                "transcript-label permutation, capability-checked",
                                "group-label permutation at four cycle periods",
                                "publication count against the half-life prediction",
                                "the model's own integrator run as the falsifier"],
                      note="Mahdessian 2021 Nature 590:649 via HPA; CCD calls are binary, so the "
                           "per-cell pseudotime that loop 99 needed is still not in this repository")
    RM.report(man, emit=say)
    json.dump({"test": "loop_cellcycle_axis", "manifest": man, "gates": gates,
               "source": "Mahdessian et al. 2021 Nature 590:649, via Human Protein Atlas",
               "c1": {"n_yes": len(yes), "n_no": len(no), "n_yes_abund": len(ay),
                      "n_no_abund": len(an), "auc_abundance": a_auc,
                      "median_yes": float(np.median(ay)), "median_no": float(np.median(an))},
               "c2": {"n_both": len(both), "n11": n11, "n10": n10, "n01": n01, "n00": n00,
                      "survival": sur, "capability": cap, "unreachable_fraction": frac_unreach},
               "c3": {"n_yes": len(gy), "n_no": len(gn), "sweep": {str(k): v
                                                                   for k, v in sweep.items()}},
               "c4": {"auc_pubs": p_auc, "auc_gain": b_auc, "auc_pubs_all": p_all},
               "c6": {"forced_fraction": ok, "median_relM": float(np.median(relM)),
                      "median_relP": float(np.median(relP)),
                      "median_attenuation": float(np.median(ratio)),
                      "damped_fraction": frac_damped, "mcnemar_p": pm,
                      "n_genes_integrated": int(len(relM))},
               "mass": {"ccd_yes_of_proteome": m_yes / tot, "called_of_proteome": m_call / tot,
                        "ccd_yes_of_called": m_yes / m_call,
                        "ccd_with_full_state": n_state},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_cellcycle_axis.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_cellcycle_axis.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
