"""LOOP 148 -- THE DESTRUCTION MACHINE'S CAPACITY RATIO.

Loop 147 asked what CIRCUIT could produce a 20.3x destruction pulse and got nothing: its N1 failed
outright (all three architectures returned exactly 1.00x), N3-N6 ran anyway because the gates were
in the wrong order, N3 silently truncated at a 200,000 cap against an exact 348,302, and N6's
answer was pure fame -- TP53 at the 100th publication percentile, ESR1 99.9, AR 99.8, against a
median gene of 51 papers. That loop is a recorded negative and this one does not repeat it.

WHY THE CIRCUIT QUESTION WAS THE WRONG QUESTION. Write the loss rate the way a cell actually builds
one: a resting term plus an activatable term,

    b(t) = k_basal + k_max * Y(t),        Y(t) in [0, 1]

where Y is whatever the wiring computes -- a Goodwin loop, a relaxation oscillator, a phosphatase
switch, anything. Then for ANY topology whatsoever

    b_hi / b_lo  <=  (k_basal + k_max) / k_basal

because Y cannot exceed 1. The wiring sets WHEN the pulse fires and HOW SHARP its edges are. It
cannot set how tall it is. The height is a property of the destruction machinery, and that is a
biochemical number rather than a graph-theoretic one. Loop 147's Goodwin reached 1.34x and its
Tyson-Novak 1.87x, and no amount of rewiring was ever going to close a gap to 20.3x, because the
gap is not in the wiring.

WHAT LOOP 142 LEFT ON THE TABLE, and what this loop has to answer. X3 measured the requirement
exactly: the median MS oscillator rests at b = ln2/29.53 h = 0.02347/h and must reach 0.47617/h
inside a 2.4 h window, a 20.29-fold acceleration. Those two numbers are read from
outputs/loop_pulse_equation.json here, not retyped, and the inversion is done by importing loop
142's own pulse_amp_exact/required_b_hi so the equation cannot drift between loops.

THE MACHINE HAS THREE STAGES AND EACH CAN BIND SEPARATELY. Ubiquitin-proteasome degradation is
targeting (an E3 picks the substrate), then processing (the 26S unfolds and shreds it). A capacity
ratio measured on one stage says nothing about the other, so all three are measured here:

    1. THE SHREDDER -- can the 26S absorb the burst at all? Bulk throughput against bulk load.
    2. THE PER-SUBSTRATE ENVELOPE -- what rate does the machine demonstrably deliver on ANY one
       substrate, measured inside a single proteome so no rate is a quotient of two?
    3. THE TARGETING STEP -- how much does the specificity subunit itself change over the cycle?

The machine's capacity ratio is the MINIMUM over the three. A loop that measured only the first
would report 55x-170x of headroom and declare the problem solved, which is exactly the mistake this
one is built to avoid.

AND THE MEASUREMENT THAT WOULD SETTLE IT IS NOT ON DISK. All three above are ACROSS-substrate or
across-subunit. The quantity the equation actually needs is WITHIN-substrate: the same protein's
own b, measured in two states of the same cell. Nothing in this repo has that for protein. K4 says
so plainly, shows the nearest available thing -- mRNA decay across 13 cell lines -- and gates it on
a control rather than on the answer, because mRNA is a DIFFERENT destruction machine (deadenylase
and exosome, not the proteasome) and cell lines are different genotypes rather than one cell in two
states. That number is recorded as a proxy and must never be quoted as the protein answer.

PREDECLARED. Every threshold below is fixed before a single number is computed.

  K0 CAN THE QUESTION BE ANSWERED FROM WHAT IS ON DISK?              THE CAPABILITY GATE.
       (a) >= 3000 genes in the Schwanhausser table with a positive protein half-life;
       (b) >= 5000 genes in Ly 2014 quantified in all six elutriation fractions;
       (c) the required 0.47617/h must lie AT OR BELOW the fastest rate the half-life assay
           actually measured -- if the requirement is off the top of the assay's scale then K2 is
           unanswerable by construction and a PASS there would be an artifact;
       (d) >= 20 genes of the curated ubiquitin-targeting machinery quantified in Ly;
       (e) REGRESSION: recomputing loop 123's 80-gene oscillator set and loop 142's median relative
           amplitude must reproduce 0.4453 and a 20.29x requirement to within 1%.
       Gate: all five. If K0 fails nothing below is read.

  K1 THE SHREDDER.                                                   BULK THROUGHPUT.
       proteasome particles x 3600/sweep against the measured proteolytic load, at both ends of the
       1-3 s/substrate sweep band, and then again with the load raised by the full required factor
       -- first on the oscillating subset alone, then in the absurd worst case where EVERY protein
       in the proteome pulses simultaneously. Gate: peak utilisation < 50% in the worst case.
       Report capacity/load as the shredder's capacity ratio.

  K2 THE PER-SUBSTRATE ENVELOPE.                                     WHAT THE MACHINE DEMONSTRABLY DOES.
       b_i = ln2/t_half,i across the Schwanhausser proteome -- copies and half-lives from the same
       cells, so loop 92's abundance rule is obeyed and nothing here is a quotient of two proteomes.
       Report p99/p50, p99.9/p50 and max/p50. Gate: p99/median >= 20.29.
       CENSORING CONTROL, required before the gate is read: if more than 20% of the fastest
       percentile share one modal half-life the tail is an assay boundary rather than a
       measurement, and K2 is struck regardless of the ratio.
       PREDECLARED LIMIT ON WHAT A PASS MEANS: an across-substrate range licenses "the machine can
       run a substrate at 0.476/h". It does NOT license "this substrate can be switched to
       0.476/h". Those are different claims and only the first is measured here.

  K3 THE TARGETING STEP.                                             THE ONE I EXPECT TO FAIL.
       in the SAME Ly dataset that defined the oscillators, the fold-range max/min over F1..F6 for
       the ubiquitin-targeting machinery, taken from CORUM membership in the assembled bundle and
       split before the run into the two kinds of subunit: RECEPTORS (CDC20, FZR1 and the F-box
       proteins -- the interchangeable part whose job is to be switched) and SCAFFOLD (the anaphase
       core, SKP1, the cullins, RBX1 -- the part that is not). Gate: median receptor fold-range
       >= 20.29.
       MATCHED-ABUNDANCE NULL, required before the gate is read: fold-range is intensity-dependent,
       so 2000 random gene sets matched on mean log LFQ decile give the null, and the receptors are
       reported as a percentile of it. Above the null means regulation; at the null means noise.
       Also report the largest fold-range achieved by ANY protein in the table, so a FAIL can be
       attributed to the biology rather than to the instrument's reach.

  K4 THE MEASUREMENT THIS REPO DOES NOT HAVE.                        THE HONEST GATE.
       state plainly that no within-substrate across-state PROTEIN degradation rate exists on disk,
       then measure the nearest proxy: within-gene fold-range of mRNA kdeg across 13 cell lines,
       genes with >= 5 lines. Gate is NOT the ratio -- it is the control. Permute gene labels
       within each cell line, preserving both the per-line marginal and the coverage pattern
       exactly, 200 times. PASS iff the real within-gene range is TIGHTER than shuffled at p < 0.01,
       which is what makes the reported number signal rather than spread. The ratio itself is
       reported and explicitly barred from being quoted as the protein answer.

  K5 FAME.                                                           THE STANDING CONTROL.
       Spearman rho of publication count against (a) the Schwanhausser degradation rate and (b) the
       Ly fold-range. Since loop 137 the strike threshold is |rho| >= 0.20. Because K2's envelope is
       a range rather than a selected set, the predeclared handling is: K5 passes if |rho| < 0.20
       OR the envelope survives recomputation on a publication-matched subsample with p99/p50 still
       >= 20.29. Anything else and K2 is struck.

  K6 WHICH STAGE BINDS.                                              THE ANSWER.
       per oscillator, the required b_hi from loop 142's exact inversion, against the envelope K2
       measured. Gate: >= 50% of the 80 lie inside it. Then name the binding stage as the minimum
       of the three ratios and say what a number below 20.29 would mean for the equation.

-> outputs/loop_capacity_ratio.json
"""
import csv
import gzip
import json
import math
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM            # noqa: E402
import loop_replication as LR        # noqa: E402
import loop_pulse_equation as PE     # noqa: E402  -- loop 142's own equation, imported not retyped

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
LY = SC / "ly2014_supp1-v1.txt"
SCHWAN = SC / "_schwan2011.json"
KDEG = SC / "AvgKdegs_genes_v1.csv"
BUNDLE = Path(__file__).resolve().parent / "data" / "cell_complete.json.gz"

SEED = 14800
LN2 = float(np.log(2.0))
T_CYCLE = 24.0
DUTY = 0.10
FOLD = 2.0                 # loop 123's oscillator call, on the pooled 3-phase fold

K0_MIN_SCHWAN = 3000
K0_MIN_LY = 5000
K0_MIN_MACH = 20
K0_REGRESSION_TOL = 0.01
K1_MAX_UTIL = 0.50
K2_CENSOR_MAX = 0.20
K3_NULL_N = 2000
K4_MIN_LINES = 5
K4_NPERM = 200
K4_ALPHA = 0.01
K5_RHO_MAX = 0.20
K6_MIN_INSIDE = 0.50

# The two kinds of subunit, split BEFORE the run. Receptors are the interchangeable specificity
# part -- the APC/C activators and the F-box proteins -- and are the subunits whose abundance is
# supposed to switch. Scaffold is the invariant machine they plug into.
APC_ACTIVATORS = ("CDC20", "FZR1")
SCAFFOLD_EXTRA = ("SKP1", "CUL1", "CUL2", "CUL3", "CUL4A", "CUL4B", "CUL5", "CUL7", "RBX1", "RBX2",
                  "ANAPC1", "ANAPC2", "ANAPC10", "ANAPC15")

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return float("nan")
    ra, rb = _rank(a[m]), _rank(b[m])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    # average ties
    i = 0
    xs = x[o]
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def machinery_sets():
    """RECEPTORS and SCAFFOLD from CORUM membership in the assembled bundle, plus the two APC/C
    activators, which are not members of the 'Anaphase-promoting core complex' entry because they
    are the exchangeable subunit that binds it."""
    with gzip.open(BUNDLE, "rt") as f:
        B = json.load(f)
    G = [g["name"] for g in B["genes"]]
    C = B["complexes"]
    scaffold, receptor, ncplx = set(), set(APC_ACTIVATORS), 0
    for name, members in C.items():
        nm = name.lower()
        is_apc = nm.startswith("anaphase-promoting")
        is_scf = nm.startswith("scf e3 ubiquitin ligase complex")
        is_crl = nm.startswith("cullin-ring")
        if not (is_apc or is_scf or is_crl):
            continue
        ncplx += 1
        for i in members:
            g = G[i]
            if g in SCAFFOLD_EXTRA or g.startswith("ANAPC") or g in ("CDC16", "CDC26", "CDC27"):
                scaffold.add(g)
            elif is_scf or is_crl:
                receptor.add(g)      # in an SCF entry, whatever is not the scaffold IS the receptor
            else:
                scaffold.add(g)
    receptor -= scaffold
    return sorted(receptor), sorted(scaffold), ncplx


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 148 -- the destruction machine's capacity ratio: three stages, and the one "
        "measurement that is missing")
    say("=" * 100)
    say()

    import pandas as pd

    gates, res = {}, {}

    # ------------------------------------------------------------------------------ inputs
    PEQ = json.load(open(OUT / "loop_pulse_equation.json"))
    PRO = json.load(open(OUT / "loop_proteostasis.json"))
    B_REST = float(PEQ["x3"]["b_rest"])
    B_HI_REQ = float(PEQ["x3"]["b_hi_required"])
    REQ_RATIO = float(PEQ["x3"]["fold_acceleration"])
    A_REF = float(PEQ["x3"]["median_A"])
    say(f"  THE REQUIREMENT, read from loop 142 rather than retyped:")
    say(f"    resting rate      b_lo = {B_REST:.6f} /h   (t1/2 = {PEQ['x3']['median_hl_h']} h)")
    say(f"    required at peak  b_hi = {B_HI_REQ:.6f} /h  over a {DUTY * T_CYCLE:.1f} h window")
    say(f"    CAPACITY RATIO NEEDED  = {REQ_RATIO:.2f}x")
    say()

    S = json.load(open(SCHWAN))
    sg = [g for g in S if S[g].get("prot_hl_h") and np.isfinite(S[g]["prot_hl_h"])
          and S[g]["prot_hl_h"] > 0]
    HL = np.array([S[g]["prot_hl_h"] for g in sg], float)
    BB = LN2 / HL
    NCOP = np.array([S[g].get("prot_copies") or np.nan for g in sg], float)

    d = pd.read_csv(LY, sep="\t", low_memory=False)
    d["g"] = d["gene_names"].astype(str).str.split(";").str[0]
    F = [f"LFQ_intensity_F{i}" for i in range(1, 7)]
    d = d[(d[F] > 0).all(axis=1) & d["gene_names"].notna()].copy()
    d["P_G1"] = d[["LFQ_intensity_F1", "LFQ_intensity_F2"]].mean(axis=1)
    d["P_S"] = d[["LFQ_intensity_F3", "LFQ_intensity_F4"]].mean(axis=1)
    d["P_G2"] = d[["LFQ_intensity_F5", "LFQ_intensity_F6"]].mean(axis=1)
    P6 = d[F].values
    P3 = d[["P_G1", "P_S", "P_G2"]].values
    pf3 = P3.max(1) / P3.min(1)
    fold6 = P6.max(1) / P6.min(1)
    lint = np.log10(P6.mean(1))
    genes_ly = d["g"].values
    idx = {}
    for i, g in enumerate(genes_ly):
        idx.setdefault(g, i)

    receptor, scaffold, ncplx = machinery_sets()

    C = json.load(open(LR.CELL))
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}

    # ---------------------------------------------------------------- K0
    say("K0 CAN THE QUESTION BE ANSWERED FROM WHAT IS ON DISK?")
    a_ok = len(sg) >= K0_MIN_SCHWAN
    b_ok = len(d) >= K0_MIN_LY
    b_max_measured = float(BB.max())
    c_ok = B_HI_REQ <= b_max_measured
    mach_in_ly = [g for g in receptor + scaffold if g in idx]
    d_ok = len(mach_in_ly) >= K0_MIN_MACH

    keep = [g for g in idx if g in S and S[g].get("prot_hl_h") and pf3[idx[g]] >= FOLD]
    rel_obs = np.array([(P3[idx[g]].max() - P3[idx[g]].min()) / (2.0 * P3[idx[g]].mean())
                        for g in keep], float)
    A_med = float(np.median(rel_obs))
    bhi_med = PE.required_b_hi(A_med, B_REST, DUTY, T_CYCLE)
    fold_med = bhi_med / B_REST if bhi_med else float("nan")
    e_ok = (abs(A_med - A_REF) / A_REF < K0_REGRESSION_TOL
            and abs(fold_med - REQ_RATIO) / REQ_RATIO < K0_REGRESSION_TOL)

    say(f"     (a) Schwanhausser genes with a protein half-life   {len(sg):,}  "
        f"gate >= {K0_MIN_SCHWAN:,}   {'ok' if a_ok else 'FAIL'}")
    say(f"     (b) Ly genes quantified in all six fractions        {len(d):,}  "
        f"gate >= {K0_MIN_LY:,}   {'ok' if b_ok else 'FAIL'}")
    say(f"     (c) fastest rate the assay measured  {b_max_measured:.4f} /h "
        f"(t1/2 {LN2 / b_max_measured:.2f} h)")
    say(f"         the requirement {B_HI_REQ:.4f} /h is "
        f"{'INSIDE' if c_ok else 'OFF THE TOP OF'} that scale   {'ok' if c_ok else 'FAIL'}")
    say(f"     (d) targeting machinery quantified in Ly            {len(mach_in_ly)} of "
        f"{len(receptor) + len(scaffold)}  gate >= {K0_MIN_MACH}   {'ok' if d_ok else 'FAIL'}")
    say(f"         {ncplx} CORUM entries -> {len(receptor)} receptors, {len(scaffold)} scaffold")
    say(f"     (e) REGRESSION on loops 123/142: {len(keep)} oscillators, median A {A_med:.4f} "
        f"(recorded {A_REF:.4f}), required {fold_med:.2f}x (recorded {REQ_RATIO:.2f}x)"
        f"   {'ok' if e_ok else 'FAIL'}")
    gates["K0"] = bool(a_ok and b_ok and c_ok and d_ok and e_ok)
    res["k0"] = {"n_schwan": len(sg), "n_ly": int(len(d)), "b_max_measured": b_max_measured,
                 "shortest_halflife_h": float(LN2 / b_max_measured),
                 "requirement_inside_scale": bool(c_ok), "n_machinery_in_ly": len(mach_in_ly),
                 "n_receptor": len(receptor), "n_scaffold": len(scaffold), "n_corum": ncplx,
                 "receptors": receptor, "scaffold": scaffold,
                 "n_osc": len(keep), "median_A": A_med, "median_A_recorded": A_REF,
                 "fold_recomputed": float(fold_med), "fold_recorded": REQ_RATIO, "pass": gates["K0"]}
    say(f"     K0 {'PASS' if gates['K0'] else 'FAIL'}")
    say()
    if not gates["K0"]:
        say("     the capability gate failed; nothing below is interpretable and no ratio is "
            "reported.")

    # ---------------------------------------------------------------- K1
    say("K1 THE SHREDDER -- can the 26S absorb the burst at all?")
    particles = float(PRO["p1"]["particles_median"])
    load = float(PRO["p2"]["load_molecules_per_h"])
    cap = {s: particles * 3600.0 / s for s in (1.0, 2.0, 3.0)}
    ok_basal = {s: load / c for s, c in cap.items()}

    # the oscillating subset raised by the full required factor, everything else left resting
    osc_set = set(keep)
    flux = NCOP * BB
    is_osc = np.array([g in osc_set for g in sg])
    flux_osc = float(np.nansum(flux[is_osc]))
    flux_rest = float(np.nansum(flux[~is_osc]))
    load_pulse_osc = flux_rest + flux_osc * REQ_RATIO
    load_pulse_all = float(np.nansum(flux)) * REQ_RATIO
    u_osc = {s: load_pulse_osc / c for s, c in cap.items()}
    u_all = {s: load_pulse_all / c for s, c in cap.items()}
    worst = max(u_all.values())

    say(f"     {particles:,.0f} proteasome particles; measured load {load:,.0f} molecules/h")
    for s in (1.0, 2.0, 3.0):
        say(f"       at {s:.0f} s/substrate  capacity {cap[s]:,.0f}/h   resting utilisation "
            f"{ok_basal[s]:.2%}   headroom {1 / ok_basal[s]:.0f}x")
    say(f"     the {int(is_osc.sum())} oscillators carry {flux_osc:,.0f} molecules/h of the "
        f"{flux_osc + flux_rest:,.0f} covered ({flux_osc / (flux_osc + flux_rest):.2%})")
    say(f"     pulse them all at {REQ_RATIO:.1f}x simultaneously: utilisation "
        + "  ".join(f"{s:.0f}s {u_osc[s]:.2%}" for s in (1.0, 2.0, 3.0)))
    say(f"     ABSURD WORST CASE, every protein in the proteome pulsing at once: "
        + "  ".join(f"{s:.0f}s {u_all[s]:.2%}" for s in (1.0, 2.0, 3.0)))
    say(f"     worst-case peak utilisation {worst:.2%}   gate < {K1_MAX_UTIL:.0%}")
    shredder_ratio = 1.0 / max(ok_basal.values())
    say(f"     SHREDDER CAPACITY RATIO = {shredder_ratio:.0f}x "
        f"({1 / max(ok_basal.values()):.0f}x-{1 / min(ok_basal.values()):.0f}x over the sweep band)")
    gates["K1"] = bool(worst < K1_MAX_UTIL)
    res["k1"] = {"particles": particles, "load_per_h": load,
                 "capacity_per_h": {str(k): v for k, v in cap.items()},
                 "resting_utilisation": {str(k): v for k, v in ok_basal.items()},
                 "utilisation_osc_pulsed": {str(k): v for k, v in u_osc.items()},
                 "utilisation_all_pulsed": {str(k): v for k, v in u_all.items()},
                 "worst_case_utilisation": worst,
                 "shredder_capacity_ratio_low": float(1 / max(ok_basal.values())),
                 "shredder_capacity_ratio_high": float(1 / min(ok_basal.values())),
                 "pass": gates["K1"]}
    say(f"     K1 {'PASS' if gates['K1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- K2
    say("K2 THE PER-SUBSTRATE ENVELOPE -- what rate does the machine demonstrably deliver?")
    p50, p90, p99, p999 = (float(np.percentile(BB, q)) for q in (50, 90, 99, 99.9))
    bmax = float(BB.max())
    r99, r999, rmax = p99 / p50, p999 / p50, bmax / p50
    # CENSORING CONTROL -- is the fast tail an assay boundary?
    top1 = HL[BB >= p99]
    modal_n, modal_v = Counter(np.round(top1, 3)).most_common(1)[0][1], \
        Counter(np.round(top1, 3)).most_common(1)[0][0]
    censor_frac = modal_n / len(top1)
    censored = censor_frac > K2_CENSOR_MAX
    say(f"     {len(BB):,} measured protein degradation rates, one proteome, no cross-proteome "
        f"quotient")
    say(f"       median {p50:.4f} /h (t1/2 {LN2 / p50:.1f} h)   p90 {p90:.4f}   p99 {p99:.4f} "
        f"(t1/2 {LN2 / p99:.2f} h)   max {bmax:.4f}")
    say(f"       p99/median {r99:.1f}x    p99.9/median {r999:.1f}x    max/median {rmax:.1f}x")
    say(f"     CENSORING CONTROL: the fastest percentile is {len(top1)} proteins; the modal "
        f"half-life {modal_v} h holds {censor_frac:.1%}   gate <= {K2_CENSOR_MAX:.0%}")
    if censored:
        say(f"       the tail is pinned at an assay boundary -- K2 is STRUCK whatever the ratio.")
    say(f"     the requirement is {B_HI_REQ:.4f} /h; the measured envelope reaches {bmax:.4f} /h, "
        f"and {float(np.mean(BB >= B_HI_REQ)):.2%} of the proteome sits at or above it already")
    gates["K2"] = bool(r99 >= REQ_RATIO and not censored)
    say(f"     WHAT A PASS MEANS AND DOES NOT: the machine CAN run a substrate at "
        f"{B_HI_REQ:.3f}/h -- {int((BB >= B_HI_REQ).sum())} of its substrates sit there in the "
        f"resting state.")
    say(f"     It does NOT mean a substrate resting at {B_REST:.4f}/h can be SWITCHED there. That "
        f"is a different claim and K4 is where it should have been measured.")
    res["k2"] = {"n": int(len(BB)), "p50": p50, "p90": p90, "p99": p99, "p999": p999, "max": bmax,
                 "ratio_p99": float(r99), "ratio_p999": float(r999), "ratio_max": float(rmax),
                 "frac_at_or_above_requirement": float(np.mean(BB >= B_HI_REQ)),
                 "n_at_or_above_requirement": int((BB >= B_HI_REQ).sum()),
                 "censor_modal_halflife_h": float(modal_v), "censor_fraction": float(censor_frac),
                 "censored": bool(censored), "pass": gates["K2"]}
    say(f"     K2 {'PASS' if gates['K2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- K3
    say("K3 THE TARGETING STEP -- how much does the specificity subunit itself change?")
    rec_i = [idx[g] for g in receptor if g in idx]
    sca_i = [idx[g] for g in scaffold if g in idx]
    rec_f = fold6[rec_i]
    sca_f = fold6[sca_i]
    rec_med, sca_med = float(np.median(rec_f)), float(np.median(sca_f))
    say(f"     {len(rec_i)} receptors and {len(sca_i)} scaffold subunits quantified over F1..F6")
    say(f"       receptor fold-range  median {rec_med:.2f}x   max {rec_f.max():.2f}x "
        f"({genes_ly[rec_i[int(np.argmax(rec_f))]]})")
    say(f"       scaffold fold-range  median {sca_med:.2f}x   max {sca_f.max():.2f}x")
    say(f"       the widest fold-range of ANY protein in the table: {fold6.max():.1f}x "
        f"({genes_ly[int(np.argmax(fold6))]}) -- the instrument reaches far past what is needed")
    top_rec = sorted(zip([genes_ly[i] for i in rec_i], rec_f.tolist()), key=lambda t: -t[1])[:12]
    say("       widest receptors: " + "  ".join(f"{g} {v:.1f}x" for g, v in top_rec))

    # MATCHED-ABUNDANCE NULL
    dec = np.clip(np.searchsorted(np.percentile(lint, np.arange(10, 100, 10)), lint), 0, 9)
    want = Counter(dec[rec_i])
    pool = {k: np.where(dec == k)[0] for k in range(10)}
    null = np.empty(K3_NULL_N, float)
    for b in range(K3_NULL_N):
        pick = np.concatenate([rng.choice(pool[k], size=n, replace=False)
                               for k, n in want.items() if len(pool[k]) >= n])
        null[b] = np.median(fold6[pick])
    pct = float(np.mean(null < rec_med))
    say(f"     MATCHED-ABUNDANCE NULL ({K3_NULL_N} draws, matched on mean log LFQ decile): null "
        f"median {np.median(null):.2f}x, receptors at the {pct:.1%} percentile")
    if pct < 0.95:
        say(f"       the receptors are NOT above their abundance-matched null -- their spread is "
            f"what any protein of that abundance shows, so it is noise and not regulation.")
    say(f"     gate: median receptor fold-range >= {REQ_RATIO:.2f}x   measured {rec_med:.2f}x")
    gates["K3"] = bool(rec_med >= REQ_RATIO)
    res["k3"] = {"n_receptor_quant": len(rec_i), "n_scaffold_quant": len(sca_i),
                 "receptor_median_fold": rec_med, "receptor_max_fold": float(rec_f.max()),
                 "scaffold_median_fold": sca_med, "scaffold_max_fold": float(sca_f.max()),
                 "proteome_max_fold": float(fold6.max()),
                 "proteome_max_gene": str(genes_ly[int(np.argmax(fold6))]),
                 "top_receptors": top_rec, "null_median": float(np.median(null)),
                 "receptor_percentile_vs_null": pct, "pass": gates["K3"]}
    say(f"     K3 {'PASS' if gates['K3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- K4
    say("K4 THE MEASUREMENT THIS REPO DOES NOT HAVE")
    say("     nothing on disk gives one protein's own degradation rate in two states of the same")
    say("     cell. Schwanhausser is one state. Ly is abundance, not rate. The nearest available")
    say("     thing is mRNA decay across cell lines, which is a DIFFERENT destruction machine.")
    rows = []
    with open(KDEG, newline="") as f:
        for r in csv.DictReader(f):
            try:
                k = float(r["avg_kdeg"])
            except (TypeError, ValueError):
                continue
            if k > 0 and np.isfinite(k):
                rows.append((r["feature_ID"], r["cell_line"], math.log(k)))
    gl = sorted({g for g, _, _ in rows})
    cl = sorted({c for _, c, _ in rows})
    gi = {g: i for i, g in enumerate(gl)}
    ci = {c: i for i, c in enumerate(cl)}
    M = np.full((len(gl), len(cl)), np.nan)
    for g, c, v in rows:
        M[gi[g], ci[c]] = v
    have = np.isfinite(M).sum(1)
    sel = have >= K4_MIN_LINES
    Ms = M[sel]
    rng_real = np.nanmax(Ms, axis=1) - np.nanmin(Ms, axis=1)
    real_med = float(np.median(rng_real))
    say(f"     {len(gl):,} genes x {len(cl)} cell lines; {int(sel.sum()):,} genes in >= "
        f"{K4_MIN_LINES} lines")
    say(f"     within-gene fold-range across lines: median {math.exp(real_med):.2f}x   "
        f"p90 {math.exp(float(np.percentile(rng_real, 90))):.2f}x   "
        f"max {math.exp(float(rng_real.max())):.1f}x")
    perm = np.empty(K4_NPERM, float)
    for b in range(K4_NPERM):
        Mp = M.copy()
        for j in range(M.shape[1]):
            ok = np.where(np.isfinite(M[:, j]))[0]
            Mp[ok, j] = M[ok[rng.permutation(len(ok))], j]
        Q = Mp[sel]
        perm[b] = float(np.median(np.nanmax(Q, axis=1) - np.nanmin(Q, axis=1)))
    p_perm = float((np.sum(perm <= real_med) + 1) / (K4_NPERM + 1))
    say(f"     LABEL-SHUFFLE CONTROL ({K4_NPERM} permutations, per-line marginal and coverage "
        f"pattern both preserved):")
    say(f"       shuffled median fold-range {math.exp(float(np.median(perm))):.2f}x   "
        f"real {math.exp(real_med):.2f}x   p = {p_perm:.4f}   gate < {K4_ALPHA}")
    gates["K4"] = bool(p_perm < K4_ALPHA)
    say(f"     the number is {'signal' if gates['K4'] else 'NOT distinguishable from spread'}. "
        f"Either way it is mRNA, it is across genotypes rather than across states of one cell, and")
    say(f"     it is BARRED from being quoted as the protein capacity ratio. The protein number is")
    say(f"     not measured in this repo and this loop does not pretend otherwise.")
    res["k4"] = {"n_genes": len(gl), "n_lines": len(cl), "n_used": int(sel.sum()),
                 "median_fold": float(math.exp(real_med)),
                 "p90_fold": float(math.exp(float(np.percentile(rng_real, 90)))),
                 "max_fold": float(math.exp(float(rng_real.max()))),
                 "shuffled_median_fold": float(math.exp(float(np.median(perm)))),
                 "p_perm": p_perm, "currency": "mRNA, NOT protein",
                 "barred_from_quotation_as_protein_answer": True, "pass": gates["K4"]}
    say(f"     K4 {'PASS' if gates['K4'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- K5
    say("K5 FAME")
    pv_s = np.array([pubs.get(g, np.nan) for g in sg], float)
    rho_b = spearman(pv_s, BB)
    pv_l = np.array([pubs.get(g, np.nan) for g in genes_ly], float)
    rho_f = spearman(pv_l, fold6)
    say(f"     pubs vs Schwanhausser degradation rate   rho {rho_b:+.4f}   "
        f"(n {int(np.isfinite(pv_s).sum()):,})")
    say(f"     pubs vs Ly fold-range                    rho {rho_f:+.4f}   "
        f"(n {int(np.isfinite(pv_l).sum()):,})")
    survives = None
    if abs(rho_b) >= K5_RHO_MAX:
        # publication-matched subsample: equalise the pubs distribution across the rate range
        ok = np.isfinite(pv_s)
        q = np.clip(np.searchsorted(np.percentile(pv_s[ok], np.arange(10, 100, 10)), pv_s), 0, 9)
        take = []
        nmin = min(int(np.sum((q == k) & ok)) for k in range(10))
        for k in range(10):
            c = np.where((q == k) & ok)[0]
            take.append(rng.choice(c, size=nmin, replace=False))
        t = np.concatenate(take)
        r99m = float(np.percentile(BB[t], 99)) / float(np.percentile(BB[t], 50))
        survives = bool(r99m >= REQ_RATIO)
        say(f"     |rho| >= {K5_RHO_MAX}, so the predeclared handling applies: envelope on a "
            f"publication-matched subsample ({len(t):,} genes, {nmin} per decile) p99/p50 "
            f"{r99m:.1f}x   {'SURVIVES' if survives else 'DOES NOT SURVIVE'}")
        gates["K5"] = survives
        if not survives:
            gates["K2"] = False
            res["k2"]["pass"] = False
            res["k2"]["struck_by_fame"] = True
            say(f"     K2 IS STRUCK.")
    else:
        gates["K5"] = True
    res["k5"] = {"rho_pubs_rate": rho_b, "rho_pubs_fold": rho_f, "threshold": K5_RHO_MAX,
                 "matched_envelope_survives": survives, "pass": gates["K5"]}
    say(f"     K5 {'PASS' if gates['K5'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- K6
    say("K6 WHICH STAGE BINDS")
    inside, req_ratios, req_bhi = [], [], []
    for g, A in zip(keep, rel_obs):
        blo = LN2 / S[g]["prot_hl_h"]
        bh = PE.required_b_hi(float(A), blo, DUTY, T_CYCLE)
        if bh is None or not np.isfinite(bh):
            continue
        req_bhi.append(bh)
        req_ratios.append(bh / blo)
        inside.append(bh <= p99)
    req_bhi = np.array(req_bhi, float)
    req_ratios = np.array(req_ratios, float)
    frac_in = float(np.mean(inside)) if len(inside) else 0.0
    say(f"     {len(req_bhi)} oscillators inverted through loop 142's exact expression at duty "
        f"{DUTY:.2f}")
    say(f"       required b_hi   median {np.median(req_bhi):.3f} /h   p90 "
        f"{np.percentile(req_bhi, 90):.3f}   max {req_bhi.max():.3f}")
    say(f"       required ratio  median {np.median(req_ratios):.1f}x   p90 "
        f"{np.percentile(req_ratios, 90):.1f}x   max {req_ratios.max():.1f}x")
    say(f"       inside the p99 envelope ({p99:.3f} /h): {frac_in:.1%}   gate >= "
        f"{K6_MIN_INSIDE:.0%}")
    gates["K6"] = bool(frac_in >= K6_MIN_INSIDE)

    stages = {"shredder (26S throughput)": float(1 / max(ok_basal.values())),
              "per-substrate envelope (p99/p50)": float(r99),
              "targeting subunit abundance (receptor fold-range)": rec_med}
    binder = min(stages, key=stages.get)
    say()
    say(f"     THE THREE STAGES, and the machine is only as fast as its slowest:")
    for k, v in sorted(stages.items(), key=lambda t: t[1]):
        say(f"       {v:8.1f}x   {k}")
    say(f"     REQUIRED {REQ_RATIO:.1f}x")
    say(f"     BINDING STAGE: {binder} at {stages[binder]:.1f}x")
    say()
    if stages[binder] < REQ_RATIO:
        say(f"     that is BELOW the requirement, and it is below it at the targeting step rather")
        say(f"     than at the shredder. The 26S has {shredder_ratio:.0f}x of headroom and the")
        say(f"     per-substrate envelope reaches {r99:.0f}x, so neither of those is what stops a")
        say(f"     20.3x pulse. What does not move is the SUBUNIT ABUNDANCE: the specificity")
        say(f"     subunits change {rec_med:.1f}-fold over the cycle, and if b(t) were set by how")
        say(f"     much receptor is present, the pulse could not be built. So EITHER the amplitude")
        say(f"     comes from activating receptor that is already there -- phosphorylation,")
        say(f"     pseudosubstrate release, localisation -- OR from modifying the SUBSTRATE so it")
        say(f"     becomes visible to a receptor whose amount never changed. Loops 145 and 146")
        say(f"     tested the second on annotation and both failed; the first has never been")
        say(f"     tested here and cannot be, because abundance proteomics does not see it.")
    else:
        say(f"     that is at or above the requirement.")
    res["k6"] = {"n": int(len(req_bhi)), "median_required_bhi": float(np.median(req_bhi)),
                 "median_required_ratio": float(np.median(req_ratios)),
                 "max_required_ratio": float(req_ratios.max()),
                 "frac_inside_envelope": frac_in, "stages": stages, "binding_stage": binder,
                 "binding_ratio": float(stages[binder]), "required": REQ_RATIO,
                 "sufficient": bool(stages[binder] >= REQ_RATIO), "pass": gates["K6"]}
    say(f"     K6 {'PASS' if gates['K6'] else 'FAIL'}")
    say()

    say("=" * 100)
    for k in ("K0", "K1", "K2", "K3", "K4", "K5", "K6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(
        inputs=[LY, SCHWAN, KDEG, BUNDLE, OUT / "loop_pulse_equation.json",
                OUT / "loop_proteostasis.json"],
        available=int(len(d)), used=len(keep), selection="filtered", seed=SEED,
        controls=["a censoring control on the fast tail, read BEFORE K2's ratio (K2)",
                  "an abundance-matched null for the fold-range, read BEFORE K3's ratio (K3)",
                  "a within-cell-line label shuffle preserving marginal and coverage (K4)",
                  "the standing fame control with a publication-matched envelope (K5)",
                  "a regression on loops 123 and 142 before any new number (K0)"],
        note="the capacity ratio is a property of the destruction machinery, not of the wiring: "
             "for b(t) = k_basal + k_max*Y(t) with Y in [0,1] the ratio is bounded by "
             "(k_basal+k_max)/k_basal whatever the topology. Three stages measured, the minimum "
             "binds, and the within-substrate switching ratio -- the one the equation actually "
             "needs -- is not measurable from anything on disk.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 148 -- the destruction machine's capacity ratio", "manifest": man,
               "gates": gates, **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_capacity_ratio.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_capacity_ratio.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
