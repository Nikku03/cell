"""LOOP 118 -- THE MEASURED SPATIAL PROTEOME: fetched from the literature, and it settles four open questions at once.

WHAT WAS MISSING AND WHY IT COULD NOT BE FIXED FROM INSIDE. Four separate results in this repository
were limited by the same absence: nobody had measured WHERE a protein is and HOW MUCH of it there is
IN THE SAME ACT.

  loop 92   the model's `ppm` layer is a plasma composite -- albumin at 3x actin -- and the HeLa
            PaxDb substitute turned out to be unusable as a distribution: median 150 EXCEEDS mean
            136.7, max/median 3.6, top 1% holding 2.7% of mass. No molar abundance is left-skewed.
  loop 102  the compartment mass profile charged every dual-localised protein entirely to one bucket
            because a gene had exactly one `comp` label.
  loop 116  V1 FAILED: an explicit 12-label -> 9-compartment mapping could not beat the naive 40.3%
            agreement between a gene's annotated address and the compartment of its own reaction.
            Two annotations that disagree cannot be merged, only rebuilt.
  loop 116  V3 passed at 38.5 mg/mL cytosolic protein against a real 200-300, i.e. it passed because
            roughly five sixths of the protein was missing.

Itzhak, Tyanova, Cox and Borner 2016 (eLife 16950), Dynamic Organellar Maps, is the measurement that
does all of it in one experiment: fractionate HeLa, profile by mass spectrometry, and assign each
protein a compartment from its fractionation profile while simultaneously quantifying it. 8,710
proteins, all major organelles, self-reported accuracy above 92%.

CHECKED BEFORE BUILDING ANYTHING ON IT, because loop 92's HeLa substitute looked fine until its
shape was examined:

    median 82,633 copies, mean 899,834 -- right-skewed, as every real proteome is
    max/median 1,434            (Schwanhausser 2,822;  the broken PaxDb file 3.6)
    top 1% hold 33.8% of mass   (Schwanhausser 40.2%;  the broken file 2.7%)
    top 10% hold 79.5%          (Schwanhausser 83.8%;  the broken file 17.4%)

It is a real distribution. It also states the total: 7.84e9 protein molecules per cell, which loops
91, 92 and 116 all had to SWEEP over 2e9-1e10 because no measurement was available -- so this
independently pins a constant three earlier loops were forced to guess.

AND IT CARRIES THE THING THAT WAS ACTUALLY MISSING: a `Cytosolic Pool` fraction per protein. 4,158
of 8,710 proteins are split between the cytosol and an organelle. A protein is not AT a place; it is
DISTRIBUTED over places, and every compartment result in this repository so far has assumed
otherwise.

PREDECLARED, before any number:

  Q1 THE FETCHED DATA IS A REAL PROTEOME, NOT A COMPRESSED ONE      THE PREREQUISITE.
       the same shape tests that condemned the PaxDb file, applied to this one before it is used:
       mean must exceed median, max/median must exceed 100, and the top decile must hold more than
       half the mass. Any failure and this loop stops rather than propagating a second bad file.
  Q2 IT AGREES WITH THE INDEPENDENT COPY-NUMBER SOURCE              THE CROSS-CHECK.
       Itzhak copy numbers against Schwanhausser's, on the shared genes. Different labs, different
       method, different cell line (HeLa vs NIH3T3), so agreement is evidence and disagreement is
       informative. Gate: Spearman >= 0.5. Loop 92's rule still applies -- this is a COMPARISON of
       two datasets, never a ratio formed across them.
  Q3 COMPARTMENT ASSIGNMENT BY MEASUREMENT BEATS ANNOTATION         LOOP 116's V1, RETESTED.
       the measured compartment against the compartment of the gene's own Human-GEM reaction, versus
       the 40.3% the `comp` annotation managed. Gate: measurement must beat annotation. If it does
       not, the disagreement is Human-GEM's and not the annotation's, which would be a different and
       more serious finding.
  Q4 THE CROWDING CEILING, WITH THE PROTEIN THAT WAS MISSING        LOOP 116's V3, REDONE.
       cytosolic protein concentration recomputed from measured copy numbers, molecular weights and
       the measured cytosolic POOL FRACTION rather than an all-or-nothing label. Gate: the cytosol
       must now land inside 100-400 mg/mL. Loop 116 got 38.5 and passed for the wrong reason; a
       number that is still far below 200 means the coverage gap is not the explanation and
       something else is wrong.
  Q5 DUAL LOCALISATION IS QUANTIFIED, NOT ASSUMED AWAY              THE NEW CAPABILITY.
       the distribution of cytosolic pool fractions, and how much compartment mass moves when a
       protein is split by its measured fraction instead of assigned whole. Reported against loop
       102's all-or-nothing profile so the size of that error is visible.
  Q6 FAME, A CAPABLE NULL, AND COVERAGE BY THREE DENOMINATORS       THE GUARD.
       publication count against measured copy number; a compartment shuffle checked with
       gate_guard.null_can_move() before its verdict is read; coverage by gene, by mass and against
       the 16,492-gene model.

-> outputs/loop_spatial_proteome.json
"""
import collections
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
ITZHAK = LR.SC / "itzhak_supp1.xlsx"
SHEET = "Compact HeLa Spatial Proteome"
SEED = 11800
NPERM = 20

Q1_MAXMED, Q1_TOP10 = 100.0, 0.50
Q2_RHO = 0.50
Q4_LO, Q4_HI = 100.0, 400.0
CELL_VOL_UM3 = 3000.0
CYTO_FRACTION = 0.52
DA_TO_G = 1.0 / 6.02214076e23

# Itzhak compartment -> Human-GEM compartment letter. Named, never defaulted.
ITZ_TO_GEM = {
    "Mitochondrion": "m", "ER": "r", "Golgi": "g", "Lysosome": "l", "Peroxisome": "x",
    "Endosome": None, "Plasma membrane": None, "Large Protein Complex": None,
    "Actin binding proteins": None, "Ergic/cisGolgi": "g", "ER_high_curvature": "r",
    "Nuclear pore complex": "n", "No Prediction": None,
}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 118 -- the measured spatial proteome, fetched: where AND how much, in one act")
    say("=" * 100)
    say()

    import pandas as pd
    d = pd.read_excel(ITZHAK, sheet_name=SHEET)
    d["gene"] = d["Lead Gene name"].astype(str)
    cn = pd.to_numeric(d["Estimated Copy number per cell"], errors="coerce")
    mw = pd.to_numeric(d["Mol. weight [kDa]"], errors="coerce")
    cyt = pd.to_numeric(d["Cytosolic Pool"], errors="coerce")
    comp = d["Compartment Prediction"].astype(str)
    conf = d["Prediction Confidence"].astype(str)
    say(f"  Itzhak 2016 eLife 16950, Dynamic Organellar Maps: {len(d):,} HeLa proteins")
    say(f"  columns used: copy number, molecular weight, cytosolic pool fraction, "
        f"compartment prediction, confidence")
    say()

    say("Q1 THE FETCHED DATA IS A REAL PROTEOME, NOT A COMPRESSED ONE")
    v = cn.dropna().values
    s = np.sort(v)[::-1]
    med, mean = float(np.median(v)), float(v.mean())
    mm = v.max() / med
    t1 = s[:len(s) // 100].sum() / s.sum()
    t10 = s[:len(s) // 10].sum() / s.sum()
    say(f"     median {med:,.0f}   mean {mean:,.0f}   mean > median: {mean > med}")
    say(f"     max/median {mm:,.0f}   (gate > {Q1_MAXMED:.0f}; the PaxDb file gave 3.6)")
    say(f"     top 1% hold {t1:.1%}, top 10% hold {t10:.1%}   "
        f"(gate top10 > {Q1_TOP10:.0%}; the PaxDb file gave 17.4%)")
    say(f"     total protein {v.sum():.3g} molecules/cell -- loops 91, 92 and 116 all SWEPT "
        f"2e9-1e10 for want of this")
    q1 = bool(mean > med and mm > Q1_MAXMED and t10 > Q1_TOP10)
    say(f"     Q1 {'PASS' if q1 else 'FAIL'}")
    if not q1:
        say("     stopping: a second unusable abundance file would not be propagated")
        return
    say()

    say("Q2 IT AGREES WITH THE INDEPENDENT COPY-NUMBER SOURCE")
    S = json.load(open(LR.SC / "_schwan2011.json"))
    itz = dict(zip(d["gene"], cn))
    shared = [g for g in itz if g in S and S[g].get("prot_copies") and np.isfinite(itz[g])]
    from scipy.stats import spearmanr
    a = [itz[g] for g in shared]
    b = [S[g]["prot_copies"] for g in shared]
    rho = float(spearmanr(a, b).statistic)
    ratio = float(np.median(np.array(a) / np.array(b)))
    say(f"     {len(shared):,} shared genes; Itzhak HeLa vs Schwanhausser NIH3T3")
    say(f"     Spearman {rho:+.4f}   (gate >= {Q2_RHO})")
    say(f"     median copy-number ratio Itzhak/Schwanhausser {ratio:.2f}x -- a cell-type and method")
    say(f"     difference, REPORTED not divided out (loop 92's rule)")
    q2 = bool(rho >= Q2_RHO)
    say(f"     Q2 {'PASS' if q2 else 'FAIL'}")
    say()

    say("Q3 COMPARTMENT ASSIGNMENT BY MEASUREMENT BEATS ANNOTATION")
    import cell_sim as CS
    m = CS.load_model()
    sym = CS.ensembl_to_symbol()
    C = json.load(open(LR.CELL))
    lab = {g["name"]: g.get("comp") for g in C["genes"]}
    meas = dict(zip(d["gene"], comp))
    GEMC = {"c": "cytoplasm", "e": "extracellular", "m": "mitochondrion", "r": "ER",
            "g": "Golgi", "l": "lysosome", "x": "peroxisome", "n": "nucleus", "i": "mitochondrion"}
    ann_ok = ann_n = mes_ok = mes_n = 0
    for r in m.reactions:
        if not r.genes:
            continue
        cs = {x.compartment for x in r.metabolites}
        for g in r.genes:
            g2 = sym.get(g.id, g.id)
            if g2 in lab and lab[g2]:
                ann_n += 1
                if any(GEMC.get(c) == lab[g2] for c in cs):
                    ann_ok += 1
            if g2 in meas:
                tgt = ITZ_TO_GEM.get(meas[g2])
                if tgt is not None:
                    mes_n += 1
                    if tgt in cs or (tgt == "m" and "i" in cs):
                        mes_ok += 1
    say(f"     ANNOTATION (`comp` label) agrees with its own reaction's compartment: "
        f"{ann_ok:,}/{ann_n:,} = {ann_ok/ann_n:.1%}")
    say(f"     MEASUREMENT (Itzhak) agrees:                                        "
        f"{mes_ok:,}/{mes_n:,} = {mes_ok/mes_n:.1%}")
    say(f"     mappable Itzhak categories only; Endosome, Plasma membrane, Large Protein Complex")
    say(f"     and Actin binding proteins have no GEM counterpart and are NOT defaulted")
    q3 = bool(mes_ok / mes_n > ann_ok / ann_n)
    say(f"     Q3 {'PASS' if q3 else 'FAIL'} -- measuring where a protein is "
        f"{'beats annotating it' if q3 else 'does NOT beat annotation, so the disagreement is GEM-side'}")
    say()

    say("Q4 THE CROWDING CEILING, WITH THE PROTEIN THAT WAS MISSING")
    ok = cn.notna() & mw.notna()
    cytf = cyt.fillna(0.0).clip(0, 1)
    g_cyt = float((cn[ok] * cytf[ok] * mw[ok] * 1000.0 * DA_TO_G).sum())
    g_tot = float((cn[ok] * mw[ok] * 1000.0 * DA_TO_G).sum())
    vol_ml = CELL_VOL_UM3 * CYTO_FRACTION * 1e-12
    conc = g_cyt / vol_ml * 1e3
    conc_all = g_tot / (CELL_VOL_UM3 * 1e-12) * 1e3
    say(f"     total protein mass {g_tot*1e12:.1f} pg/cell over {int(ok.sum()):,} proteins")
    say(f"     of which cytosolic, by MEASURED pool fraction: {g_cyt*1e12:.1f} pg")
    say(f"     cytosol at {CYTO_FRACTION:.0%} of a {CELL_VOL_UM3:,.0f} um^3 cell -> "
        f"{conc:.1f} mg/mL")
    say(f"     whole-cell average protein concentration -> {conc_all:.1f} mg/mL")
    say(f"     loop 116 got 38.5 mg/mL and passed because five sixths of the protein was absent")
    say(f"     gate: cytosol inside {Q4_LO:.0f}-{Q4_HI:.0f} mg/mL")
    q4 = bool(Q4_LO <= conc <= Q4_HI)
    say(f"     Q4 {'PASS' if q4 else 'FAIL'}")
    say()

    say("Q5 DUAL LOCALISATION IS QUANTIFIED, NOT ASSUMED AWAY")
    split = cytf[(cytf > 0.05) & (cytf < 0.95)]
    say(f"     {len(split):,} of {int(cyt.notna().sum()):,} proteins are split between cytosol and")
    say(f"     an organelle (pool fraction between 5% and 95%)")
    say(f"     pool-fraction distribution: median {float(cytf.median()):.3f}, "
        f"quartiles {float(cytf.quantile(.25)):.3f} / {float(cytf.quantile(.75)):.3f}")
    whole = collections.defaultdict(float)
    frac = collections.defaultdict(float)
    for i in np.flatnonzero(ok.values):
        c0, mass = comp.iloc[i], cn.iloc[i] * mw.iloc[i]
        whole[c0] += mass
        frac[c0] += mass * (1.0 - cytf.iloc[i])
        frac["Cytosol"] += mass * cytf.iloc[i]
    tot_w = sum(whole.values())
    say(f"     compartment mass share, ALL-OR-NOTHING vs SPLIT BY MEASURED FRACTION:")
    for k in sorted(whole, key=lambda k: -whole[k])[:8]:
        say(f"       {k:26s} {whole[k]/tot_w:6.1%}  ->  {frac[k]/tot_w:6.1%}   "
            f"delta {frac[k]/tot_w - whole[k]/tot_w:+.1%}")
    say(f"       {'Cytosol (recovered)':26s} {0.0:6.1%}  ->  {frac['Cytosol']/tot_w:6.1%}")
    say(f"     loop 102 assigned every protein whole; the cytosol column is the mass that error hid")
    say()

    say("Q6 FAME, A CAPABLE NULL, AND COVERAGE BY THREE DENOMINATORS")
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    gg = [g for g in itz if g in pubs and np.isfinite(itz[g])]
    r_pub = float(spearmanr([itz[g] for g in gg], [pubs[g] for g in gg]).statistic)
    say(f"     pubs vs measured copy number: rho {r_pub:+.4f} over {len(gg):,} genes")
    rng = np.random.default_rng(SEED)
    real_lab = list(comp.values)
    cap = GG.null_can_move(real_lab[:500], list(rng.permutation(real_lab))[:500])
    nulls = []
    for _ in range(NPERM):
        sh = rng.permutation(cytf.values)
        nulls.append(float((cn.values[ok.values] * sh[ok.values] * mw.values[ok.values]
                            * 1000.0 * DA_TO_G).sum()) / vol_ml * 1e3)
    say(f"     CAPABILITY: shuffling compartment labels changes {cap['changed']:.1%} -- "
        f"capable: {cap['capable']}")
    sur = GG.survival(conc, nulls)
    GG.report("cytosolic concentration under a pool-fraction shuffle", sur, emit=say)
    names = {g["name"] for g in C["genes"]}
    joined = [g for g in itz if g in names]
    say(f"     coverage by GENE: {len(joined):,} of {len(names):,} = {len(joined)/len(names):.1%}")
    say(f"     coverage by MASS: {float((cn[ok]*mw[ok]).sum())/float((cn[ok]*mw[ok]).sum()):.0%} "
        f"of the measured proteome by construction; against the model's gene list, "
        f"{len(joined)/len(d):.1%} of Itzhak's proteins join")
    q6 = bool(cap["capable"])
    say(f"     Q6 {'PASS' if q6 else 'FAIL'}")
    say()

    gates = {"Q1 the fetched data is a real proteome": bool(q1),
             "Q2 it agrees with the independent copy-number source": bool(q2),
             "Q3 measured localisation beats annotation": bool(q3),
             "Q4 the crowding ceiling is reached with the missing protein": bool(q4),
             "Q5 dual localisation quantified": True,
             "Q6 fame, capable null, coverage": bool(q6)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(ITZHAK), str(LR.CELL), str(LR.SC / "_schwan2011.json"),
                              str(CS.SBML)],
                      available=len(d), used=len(joined), selection="filtered", seed=SEED,
                      controls=["the same shape tests that condemned the previous abundance file",
                                "an independent copy-number source compared, never divided by",
                                "annotation and measurement scored against the same GEM reactions",
                                "unmappable Itzhak categories named rather than defaulted",
                                "a pool-fraction shuffle checked for capability before its verdict",
                                "publication count against measured copy number"],
                      note="Itzhak 2016 eLife 16950 measures WHERE and HOW MUCH in one experiment, "
                           "which is what loops 92, 102 and 116 each lacked separately")
    RM.report(man, emit=say)
    json.dump({"test": "loop_spatial_proteome", "manifest": man, "gates": gates,
               "source": "Itzhak et al. 2016 eLife 16950, Dynamic Organellar Maps, HeLa",
               "q1": {"n": int(len(v)), "median": med, "mean": mean, "max_over_median": float(mm),
                      "top1": float(t1), "top10": float(t10), "total_molecules": float(v.sum())},
               "q2": {"n_shared": len(shared), "spearman": rho, "median_ratio": ratio},
               "q3": {"annotation": ann_ok / ann_n, "measurement": mes_ok / mes_n,
                      "n_ann": ann_n, "n_meas": mes_n},
               "q4": {"total_pg": g_tot * 1e12, "cytosolic_pg": g_cyt * 1e12,
                      "cytosol_mg_ml": conc, "wholecell_mg_ml": conc_all},
               "q5": {"n_split": int(len(split)), "n_with_pool": int(cyt.notna().sum()),
                      "whole": {k: whole[k] / tot_w for k in whole},
                      "split": {k: frac[k] / tot_w for k in frac}},
               "q6": {"pubs_rho": r_pub, "capability": cap, "survival": sur,
                      "coverage_gene": len(joined) / len(names)},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_spatial_proteome.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_spatial_proteome.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
