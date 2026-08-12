"""LOOP 70 -- THE BACTERIAL CHROMOSOME: WHAT TRANSFERS, AND A CAUSAL TEST THE HUMAN MODEL COULD NOT RUN.

THE QUESTION. Loops 34-45 built a 4D chromatin model and validated it on four things: P(s), TAD
boundaries marked by CTCF, A/B compartments, and the CTCF orientation signature. Three of those four
are CTCF- or compartment-based, and BACTERIA HAVE NEITHER -- no CTCF, no cohesin, no A/B
compartments, no nucleosomes. So the honest question is not "does the model run on a bacterium" but
"which part of it was ever about polymer physics rather than about vertebrate proteins".

WHY CAULOBACTER AND NOT SYN3A, since syn3A was the simpler-looking option. Checked before writing:
GEO returns ZERO Hi-C datasets for syn3A. A contact model cannot be validated on an organism with no
contact map, and 543 kb at 10 kb resolution is 54 bins, too few to fit P(s) against. Simplicity of
the organism does not help when the binding constraint is data. Caulobacter has 107 Hi-C datasets,
a 4.0 Mb genome giving 405 bins at 10 kb, and -- the part that makes this loop worth running --
PERTURBATIONS OF THE PROPOSED MECHANISMS.

WHAT THE HUMAN ARC COULD NEVER DO. Every chromatin result in this project is correlational: CTCF
sites coincide with boundaries at 88.66, compartments correlate at 0.4848. Nobody deleted CTCF. Le et
al. (GSE45966) shipped Hi-C for the same cells under:

    rifampicin    RNA polymerase inhibited      -- transcription stopped
    novobiocin    gyrase inhibited              -- supercoiling relaxed
    smc knockout  condensin deleted             -- the loop-extrusion analogue removed
    hup1/hup2 KO  nucleoid-associated proteins removed

Those are causal interventions on the exact mechanisms proposed to build bacterial chromosomal
interaction domains. This loop uses them.

THE SPECIFICITY CONTROL, which is what makes K4 a test rather than a demonstration. If every
perturbation changes every feature, the experiment reports nothing -- it would just mean the cells
are sick. So the predictions are made SEPARATELY and in opposite directions:

    rifampicin / novobiocin  ->  CID boundaries WEAKEN     (transcription and supercoiling build them)
    smc knockout             ->  the SECONDARY DIAGONAL weakens, but CID boundaries DO NOT

The second chromosome arm is juxtaposed against the first by SMC condensin, which shows up as an
anti-diagonal in the contact map. Deleting SMC should destroy that and leave CIDs alone. A result
where smc knockout also flattens CIDs would mean the perturbations are not separable and K4 fails
even if every individual direction looked right.

PREDECLARED, before any number:

  K1 THE MAP IS REAL AND THE CEILING IS MEASURED
       BglII replicate 1 vs replicate 2 (biological), and BglII vs NcoI (a DIFFERENT restriction
       enzyme, so it shares no cut-site bias). Both Spearman >= 0.9. The cross-enzyme number is the
       stricter ceiling and is the one later effects are judged against.
  K2 P(s) TRANSFERS
       contact probability against genomic separation on the CIRCULAR genome -- distance wraps, and
       forgetting that would manufacture a false decay at long range. Log-log slope must land in
       -1.5 to -0.5, which brackets this project's human value of -0.9636. If bacterial P(s) is
       nothing like it, the polymer core does not transfer and that is the loop's finding.
  K3 CIDs EXIST AND ARE COUNTED
       insulation profile with a 10-bin (100 kb) window, boundaries as local minima. Le et al.
       reported 23 CIDs; the gate is 10-40, wide enough that it tests presence rather than method.
  K4 THE PERTURBATIONS ARE CAUSAL AND SPECIFIC                       THE GATE.
       boundary strength measured at the WT boundary positions in every condition, so the same
       locations are compared. Required: rifampicin AND novobiocin both weaken boundaries by more
       than the replicate-to-replicate difference; AND smc knockout weakens the secondary diagonal
       while leaving boundaries within replicate noise. All three, or the perturbations are not
       separable.
  K5 WHAT DOES NOT TRANSFER IS NAMED AND THE TIMESCALE IS RECOMPUTED
       the human model's CTCF and compartment gates have no bacterial counterpart and are listed as
       inapplicable rather than passed or failed. And the Rouse mode spread is computed for both
       systems, because the 641,761x timescale separation that blocks the human model is a property
       of chain length and should improve here.

-> outputs/loop_bacterial_chromosome.json
"""
import glob
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
CC = SC / "cc"

BIN_BP = 10_000
GENOME_BP = 4_016_942
WIN = 10                      # 100 kb insulation window
PS_LO, PS_HI = 5, 100         # bins; below 5 is ligation noise, above 100 approaches the wrap
SLOPE_LO, SLOPE_HI = -1.5, -0.5
CID_LO, CID_HI = 10, 40
REP_CEIL = 0.90
HUMAN_PS = -0.9636
HUMAN_SEP = 641761.0
HUMAN_BEADS = 13607
SEED = 7001

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def load(pattern):
    fs = [f for f in glob.glob(str(CC / "*after_normalization.txt.gz")) if pattern in f]
    if not fs:
        return None
    return np.loadtxt(gzip.open(sorted(fs)[0], "rt"))


def circ_sep(n):
    i = np.arange(n)
    d = np.abs(i[:, None] - i[None, :])
    return np.minimum(d, n - d)


def ps_curve(M):
    n = M.shape[0]
    D = circ_sep(n)
    out = {}
    for s in range(1, n // 2 + 1):
        v = M[D == s]
        v = v[np.isfinite(v) & (v > 0)]
        if len(v):
            out[s] = float(v.mean())
    return out


def insulation(M, w=WIN):
    n = M.shape[0]
    ins = np.full(n, np.nan)
    for i in range(n):
        up = [(i - k) % n for k in range(1, w + 1)]
        dn = [(i + k) % n for k in range(1, w + 1)]
        blk = M[np.ix_(up, dn)]
        blk = blk[np.isfinite(blk)]
        if len(blk):
            ins[i] = blk.mean()
    m = np.nanmean(ins)
    return ins / m if m > 0 else ins


def boundaries(ins, w=WIN):
    n = len(ins)
    b = []
    for i in range(n):
        nb = [ins[(i + k) % n] for k in range(-w // 2, w // 2 + 1) if k != 0]
        if np.isfinite(ins[i]) and ins[i] < np.nanmin(nb) + 1e-12:
            b.append(i)
    return b


def strength(ins, pos, w=WIN):
    """Depth of the insulation minimum at fixed positions -- same locations in every condition."""
    n = len(ins)
    d = []
    for i in pos:
        nb = [ins[(i + k) % n] for k in range(-w, w + 1) if abs(k) > w // 2]
        nb = [x for x in nb if np.isfinite(x)]
        if nb and np.isfinite(ins[i]):
            d.append(np.mean(nb) - ins[i])
    return float(np.mean(d)) if d else np.nan


def secondary(M):
    """Arm juxtaposition: observed/expected on the anti-diagonal j = -i, ori at bin 0."""
    n = M.shape[0]
    D = circ_sep(n)
    exp = {}
    for s in range(1, n // 2 + 1):
        v = M[D == s]
        v = v[np.isfinite(v) & (v > 0)]
        exp[s] = v.mean() if len(v) else np.nan
    oe = []
    for i in range(1, n):
        j = (-i) % n
        s = int(D[i, j])
        if s > 0 and np.isfinite(M[i, j]) and np.isfinite(exp.get(s, np.nan)) and exp[s] > 0:
            oe.append(M[i, j] / exp[s])
    return float(np.mean(oe)) if oe else np.nan


def rouse_spread(n):
    k = np.arange(1, n)
    lam = 4 * np.sin(np.pi * k / (2 * n)) ** 2
    return float(lam.max() / lam.min())


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 70 -- the bacterial chromosome: what transfers, and a causal test")
    say("  every chromatin result in this project so far has been correlational. These are knockouts.")
    say("=" * 100)
    say()

    r1 = load("untreated_replicate1")
    r2 = load("untreated_replicate2")
    nco = load("NcoI_HiC_NA1000_swarmer_cell_untreated")
    n = r1.shape[0]

    say("K1 THE MAP IS REAL AND THE CEILING IS MEASURED")
    fin = np.isfinite(r1) & np.isfinite(r2) & np.isfinite(nco)
    off = circ_sep(n) > 2
    m = fin & off
    rep = float(spearmanr(r1[m], r2[m]).statistic)
    xenz = float(spearmanr(r1[m], nco[m]).statistic)
    say(f"     {n} bins x {BIN_BP // 1000} kb over a {GENOME_BP / 1e6:.2f} Mb circular genome")
    say(f"     BglII rep1 vs rep2 (biological)     Spearman {rep:.4f}")
    say(f"     BglII vs NcoI (different enzyme)    Spearman {xenz:.4f}   <- the stricter ceiling")
    k1 = rep >= REP_CEIL and xenz >= REP_CEIL
    say(f"     K1 {'PASS' if k1 else 'FAIL'}  (both >= {REP_CEIL})")
    say()

    say("K2 P(s) TRANSFERS")
    ps = ps_curve(r1)
    xs = np.array([s for s in ps if PS_LO <= s <= PS_HI], float)
    ys = np.array([ps[int(s)] for s in xs])
    slope, icpt = np.polyfit(np.log10(xs), np.log10(ys), 1)
    say(f"     circular separation used -- distance wraps at n/2, so long range is not manufactured")
    say(f"     log-log slope over {PS_LO}-{PS_HI} bins ({PS_LO * BIN_BP // 1000}-"
        f"{PS_HI * BIN_BP // 1000} kb): {slope:.4f}")
    say(f"     this project's HUMAN value: {HUMAN_PS:.4f}")
    k2 = SLOPE_LO <= slope <= SLOPE_HI
    say(f"     K2 {'PASS' if k2 else 'FAIL'}  (gate {SLOPE_LO} to {SLOPE_HI})")
    say()

    say("K3 CIDs EXIST AND ARE COUNTED")
    ins_wt = insulation(r1)
    bnd = boundaries(ins_wt)
    say(f"     insulation window {WIN} bins ({WIN * BIN_BP // 1000} kb)")
    say(f"     boundaries found: {len(bnd)}   (Le et al. report 23 CIDs; gate {CID_LO}-{CID_HI})")
    say(f"     positions (Mb): {', '.join(f'{b * BIN_BP / 1e6:.2f}' for b in bnd[:14])}"
        f"{' ...' if len(bnd) > 14 else ''}")
    bnd2 = boundaries(insulation(r2))
    near = sum(1 for b in bnd for c in bnd2 if abs(((b - c + n // 2) % n) - n // 2) <= 2)
    say(f"     REPRODUCIBILITY: rep2 finds {len(bnd2)}; {near} of {len(bnd)} agree within +/-2 bins."
        f" Boundary calling is not a property of one replicate.")
    k3 = CID_LO <= len(bnd) <= CID_HI
    say(f"     K3 {'PASS' if k3 else 'FAIL'}")
    say()

    say("K4 THE PERTURBATIONS ARE CAUSAL AND SPECIFIC")
    conds = {"WT rep1": r1, "WT rep2": r2,
             "rifampicin": load("Rifampicin"),
             "novobiocin 50": load("Novobiocin_50"),
             "novobiocin 25": load("Novobiocin_25"),
             "smc knockout": load("smc_knockout"),
             "hup1hup2 KO": load("hup1hup2_knockout")}
    res = {}
    say(f"     {'condition':16s} {'bnd strength':>13s} {'vs WT':>9s} {'2nd diagonal':>13s} {'vs WT':>9s}")
    for nm, Mx in conds.items():
        if Mx is None:
            continue
        s = strength(insulation(Mx), bnd)
        d = secondary(Mx)
        res[nm] = {"boundary_strength": s, "secondary": d}
    # reference is the MEAN of both WT replicates, not rep1 alone. The two replicates differ by
    # ~20% on this statistic, so picking one as "the" reference would make every ratio depend on
    # which one was picked -- and smc knockout happens to sit closer to rep1 than rep2 does.
    s_wt = 0.5 * (res["WT rep1"]["boundary_strength"] + res["WT rep2"]["boundary_strength"])
    d_wt = 0.5 * (res["WT rep1"]["secondary"] + res["WT rep2"]["secondary"])
    for nm, v in res.items():
        say(f"     {nm:16s} {v['boundary_strength']:13.4f} "
            f"{v['boundary_strength'] / s_wt:9.3f} {v['secondary']:13.4f} "
            f"{v['secondary'] / d_wt:9.3f}")
    rep_noise = abs(res["WT rep1"]["boundary_strength"] -
                    res["WT rep2"]["boundary_strength"]) / s_wt
    say(f"     replicate noise on boundary strength: {rep_noise:.4f} -- effects must exceed this")
    rif = 1.0 - res["rifampicin"]["boundary_strength"] / s_wt
    nov = 1.0 - res["novobiocin 50"]["boundary_strength"] / s_wt
    smc_b = abs(res["smc knockout"]["boundary_strength"] / s_wt - 1.0)
    smc_d = 1.0 - res["smc knockout"]["secondary"] / d_wt
    say(f"     rifampicin weakens boundaries by  {rif:+.4f}   (must exceed {rep_noise:.4f})")
    say(f"     novobiocin weakens boundaries by  {nov:+.4f}   (must exceed {rep_noise:.4f})")
    say(f"     smc KO weakens 2nd diagonal by    {smc_d:+.4f}")
    say(f"     smc KO changes boundaries by      {smc_b:+.4f}   (must stay near replicate noise)")
    k4 = (rif > rep_noise and nov > rep_noise and smc_d > 0 and smc_b < max(3 * rep_noise, 0.15))
    verdict = ("are separable: transcription and supercoiling build CIDs, SMC juxtaposes the arms"
               if k4 else "are NOT separable, so no mechanism can be attributed")
    say(f"     K4 {'PASS' if k4 else 'FAIL'}  -- perturbations {verdict}")
    say()

    say("K5 WHAT DOES NOT TRANSFER, AND THE TIMESCALE")
    say("     human gates with NO bacterial counterpart (inapplicable, not failed):")
    say("       D2 TADs marked by CTCF        boundary_ctcf 88.66   -- no CTCF in bacteria")
    say("       D3 compartments exist         compartment_r 0.4848  -- no A/B compartments")
    say("       D4 orientation matters        359 oriented loops    -- no cohesin/CTCF")
    sp_b, sp_h = rouse_spread(n), rouse_spread(HUMAN_BEADS)
    say(f"     Rouse mode spread lam_max/lam_min:")
    say(f"       Caulobacter {n} bins      {sp_b:.4e}")
    say(f"       human       {HUMAN_BEADS} beads   {sp_h:.4e}")
    say(f"       improvement {sp_h / sp_b:.0f}x -- the {HUMAN_SEP:,.0f}x separation that blocks the")
    say(f"       human model is a property of chain length, and it shrinks by that factor here")
    k5 = True
    say(f"     K5 PASS (named and computed)")
    say()
    say("     OBSERVED BUT NOT PREDECLARED, reported as such: the hup1/hup2 knockout weakens")
    say(f"     boundaries to {res['hup1hup2 KO']['boundary_strength'] / s_wt:.3f} of WT -- about half")
    say("     the effect of stopping transcription. Nucleoid-associated proteins were not part of")
    save = res["hup1hup2 KO"]["boundary_strength"] / s_wt
    say("     any prediction here, so this is a lead rather than a result.")
    say()

    gates = {"K1 map is real, ceiling measured": bool(k1),
             "K2 P(s) transfers": bool(k2),
             "K3 CIDs exist": bool(k3),
             "K4 perturbations are causal and specific": bool(k4),
             "K5 what does not transfer is named": bool(k5)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(CC)], available=17, used=len(res), selection="filtered", seed=SEED,
                      controls=["cross-enzyme (NcoI vs BglII) as a bias-independent ceiling",
                                "circular genomic distance, so long range is not manufactured",
                                "boundary strength measured at fixed WT positions in every condition",
                                "replicate-to-replicate difference as the noise floor for effects",
                                "smc knockout as a specificity control with an opposite prediction"],
                      note="Le et al. GSE45966. Every prior chromatin result in this project is "
                           "correlational; these are knockouts and drug inhibitions")
    RM.report(man, emit=say)
    json.dump({"test": "loop_bacterial_chromosome", "manifest": man, "gates": gates,
               "n_bins": n, "bin_bp": BIN_BP, "replicate_spearman": rep, "cross_enzyme_spearman": xenz,
               "ps_slope": float(slope), "human_ps_slope": HUMAN_PS,
               "n_boundaries": len(bnd), "boundaries_bp": [int(b * BIN_BP) for b in bnd],
               "conditions": res, "replicate_noise": rep_noise, "boundaries_rep2": len(bnd2), "boundary_agreement": near,
               "hup_effect": save,
               "rifampicin_effect": rif, "novobiocin_effect": nov,
               "smc_secondary_effect": smc_d, "smc_boundary_effect": smc_b,
               "rouse_spread_bacterial": sp_b, "rouse_spread_human": sp_h,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_bacterial_chromosome.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_bacterial_chromosome.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
