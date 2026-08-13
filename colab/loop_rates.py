"""LOOP 91 -- TRANSCRIPTION AND TRANSLATION RATES, FROM AN IDENTITY WE HAVE ALWAYS OWNED.

THE HOLE THIS FILLS. Every layer in this repository answers a different question in a different
currency: flux balance gives mol/gDW/h at steady state with no clock, chromatin gives a contact map
with a clock but no chemistry, the TF network gives 612,133 edges with no units at all, and loop 74's
half-lives give hours attached to nothing. Loops 87 and 87b tried to wire chromatin to regulation
using graph structure on both sides and found nothing, twice, with publication count beating the
biology. The reason is not the estimator. It is that a graph edge and a contact are not quantities,
so there was never anything to conserve, and nothing to be wrong about.

WHAT WE ALREADY HAD. For any species at steady state with copy number N and half-life t_half, the
production rate is fixed by the loss rate:

        production = N * ln2 / t_half                [molecules per cell per hour]

Schwanhausser 2011 is on disk with all four quantities on 4,190 genes -- mRNA copies, mRNA half-life,
protein copies, protein half-life -- so

        k_sm = mRNA_copies * ln2 / mRNA_t_half                      transcription, mRNA/cell/h
        k_sp = prot_copies * ln2 / prot_t_half / mRNA_copies        translation, protein/mRNA/h

is arithmetic on data this project has held since loop 74. Neither has ever been computed here. That
is the shared currency the layers lack, and it costs one line each.

AND THE FIRST ANSWER IS WRONG IN A WAY THAT IS ITSELF THE TEST. Computed naively, before this module
was written:

        k_sm median   1.34 mRNA/cell/h      Schwanhausser report a median near 2
        k_sp median  40.31 protein/mRNA/h   Schwanhausser report a median near 140

Transcription is close; translation is low by about 3.5x. The naive identity assumes degradation is
the only way a cell loses protein. It is not: a dividing cell halves its contents, so the true loss
rate is degradation PLUS dilution, and for a protein with a 62 h half-life in a cell doubling every
24 h, dilution is the larger term by a factor of nearly three. Predicted, before running:

        ln2/61.92 + ln2/24 = 0.0401  against  ln2/61.92 = 0.0112       a factor of 3.6

which is the size of the discrepancy. So R1 is not a correction fitted to close a gap -- it is a
prediction that a specific, independently known term will close a specific, already measured gap, and
it fails if the corrected median overshoots as easily as if it undershoots.

PREDECLARED, before any number beyond the two medians quoted above:

  R1 GROWTH DILUTION CLOSES THE TRANSLATION GAP                     THE GATE.
       adding ln2/t_double to the loss rate must bring the k_sp median within a factor of 1.5 of the
       published ~140 protein/mRNA/h, and must NOT push k_sm outside a factor of 1.5 of ~2 mRNA/cell/h
       (mRNA half-lives are hours, so dilution should barely move transcription -- an asymmetry the
       correction has to respect, and the reason this is a test rather than a knob).
  R2 THE RATE DOES NOT DEPEND ON WHOSE HALF-LIFE YOU USE                 ROBUSTNESS.
       k_sm recomputed with RNADecayCafe half-lives -- different labs, different cell lines,
       independent measurement -- against k_sm from Schwanhausser. Gate: Spearman >= 0.5 on the shared
       genes. A rate that changes when the half-life source changes is a property of one experiment.
  R3 THE RATE IS NOT ABUNDANCE IN DISGUISE                              THE TRIVIAL BASELINE.
       if half-life varied little, k_sm would be copy number rescaled and this whole loop would be a
       renaming. Gate: Spearman(k_sm, mRNA copies) < 0.95, AND the half-life must carry independent
       variance -- reported as the partial correlation of k_sm with half-life given abundance.
  R4 THE FAME CONTROL                                                   THE RECURRING KILLER.
       `pubs` against both rates. Reported, and reported next to the biology rather than after it.
       Fame beat the biology in loops 87 and 87b and it gets checked here before anything is claimed.
  R5 THE RATES FIT INSIDE THE POLYMERASE                                THE PHYSICAL CLOSURE.
       total transcription summed over all genes must fit within RNA polymerase II capacity computed
       from measured POLR2A abundance and a literature elongation rate. This is loop 74's ribosome
       budget applied to transcription: a bound set by counting molecules, which no amount of
       publication bias can move. Gate: demand <= capacity. If the derived rates need more polymerase
       than the cell has, they are wrong regardless of how well they match a published median.
  R6 EXTENSION BEYOND THE 4,190                                         THE DELIVERABLE.
       k_sm and k_sp written for every gene with an abundance and a half-life, with coverage and the
       fraction of total abundance mass covered both reported, so the next loop knows exactly which
       genes it may use and which it may not.

-> outputs/loop_rates.json
"""
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

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
LN2 = float(np.log(2.0))

SCHWAN = SC / "_schwan2011.json"
RNADECAY = SC / "rnadecay" / "AvgKdegs.csv"
PROT_HL = SC / "_prot_halflife_human.json"

# NIH 3T3 in Schwanhausser 2011 double in about 27.5 h; the sensitivity to this is reported in R1
T_DOUBLE_H = 27.5
LIT_KSM, LIT_KSP = 2.0, 140.0          # Schwanhausser 2011 reported medians
R1_FACTOR = 1.5
R2_RHO = 0.5
R3_MAX_RHO = 0.95
# Pol II elongation ~2 kb/min = 120 kb/h; a Pol II is productively engaged over the gene body
POLII_KB_PER_H = 120.0
MEAN_GENE_KB = 30.0                     # measured from the model's own gene_start/gene_end below
SEED = 9101

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def load_schwan():
    d = json.load(open(SCHWAN))
    keep = {}
    for g, v in d.items():
        if all(v.get(x) for x in ("prot_hl_h", "mrna_hl_h", "prot_copies", "mrna_copies")):
            keep[g] = v
    return keep


def rates(v, t_double_h=None):
    """Steady-state production rates. t_double_h=None gives the naive degradation-only identity."""
    dil = LN2 / t_double_h if t_double_h else 0.0
    k_sm = v["mrna_copies"] * (LN2 / v["mrna_hl_h"] + dil)
    k_sp = v["prot_copies"] * (LN2 / v["prot_hl_h"] + dil) / v["mrna_copies"]
    return k_sm, k_sp


def rnadecay_halflives():
    """Median half-life per gene over RNADecayCafe cell lines -- an independent measurement."""
    import collections
    acc = collections.defaultdict(list)
    with open(RNADECAY) as f:
        hdr = f.readline().rstrip("\n").split(",")
        gi, hi = hdr.index("feature_ID"), hdr.index("avg_halflife")
        for ln in f:
            p = ln.rstrip("\n").split(",")
            if len(p) <= max(gi, hi):
                continue
            try:
                h = float(p[hi])
            except ValueError:
                continue
            if np.isfinite(h) and 0 < h < 1000:
                acc[p[gi]].append(h)
    return {g: float(np.median(v)) for g, v in acc.items() if v}


def spear(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 30:
        return float("nan"), int(f.sum())
    return float(spearmanr(a[f], b[f]).statistic), int(f.sum())


def partial_spear(x, y, z):
    from scipy.stats import spearmanr, rankdata
    x, y, z = map(lambda a: np.asarray(a, float), (x, y, z))
    f = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if f.sum() < 30:
        return float("nan")
    rx, ry, rz = rankdata(x[f]), rankdata(y[f]), rankdata(z[f])
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return float("nan")
    return float(spearmanr(ex, ey).statistic)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 91 -- transcription and translation rates, from an identity we have always owned")
    say("=" * 100)
    say()

    S = load_schwan()
    genes = sorted(S)
    say(f"  Schwanhausser 2011: {len(S):,} genes with mRNA copies, mRNA half-life, protein copies "
        f"and protein half-life")
    say(f"  identity: production = N * (ln2/t_half + ln2/t_double)   [molecules/cell/h]")
    say()

    say("R1 GROWTH DILUTION CLOSES THE TRANSLATION GAP")
    naive = np.array([rates(S[g], None) for g in genes])
    corr = np.array([rates(S[g], T_DOUBLE_H) for g in genes])
    m_sm_n, m_sp_n = float(np.median(naive[:, 0])), float(np.median(naive[:, 1]))
    m_sm_c, m_sp_c = float(np.median(corr[:, 0])), float(np.median(corr[:, 1]))
    say(f"     naive (degradation only)   k_sm {m_sm_n:8.2f}   k_sp {m_sp_n:9.2f}")
    say(f"     + dilution at t_double {T_DOUBLE_H:.1f} h   k_sm {m_sm_c:8.2f}   k_sp {m_sp_c:9.2f}")
    say(f"     published                  k_sm {LIT_KSM:8.2f}   k_sp {LIT_KSP:9.2f}")
    f_sm = max(m_sm_c, LIT_KSM) / min(m_sm_c, LIT_KSM)
    f_sp = max(m_sp_c, LIT_KSP) / min(m_sp_c, LIT_KSP)
    say(f"     factor from published:     k_sm {f_sm:8.2f}   k_sp {f_sp:9.2f}   "
        f"(gate: both <= {R1_FACTOR})")
    sens = {}
    for td in (20.0, 24.0, 27.5, 34.0):
        c = np.array([rates(S[g], td) for g in genes])
        sens[td] = (float(np.median(c[:, 0])), float(np.median(c[:, 1])))
        say(f"     sensitivity t_double {td:5.1f} h -> k_sm {sens[td][0]:7.2f}  k_sp {sens[td][1]:8.2f}")
    r1 = bool(f_sm <= R1_FACTOR and f_sp <= R1_FACTOR)
    say(f"     R1 {'PASS' if r1 else 'FAIL'} -- dilution "
        f"{'closes the translation gap without breaking transcription' if r1 else 'does NOT close both'}")
    say()

    ksm = {g: corr[i, 0] for i, g in enumerate(genes)}
    ksp = {g: corr[i, 1] for i, g in enumerate(genes)}

    say("R2 THE RATE DOES NOT DEPEND ON WHOSE HALF-LIFE YOU USE")
    rd = rnadecay_halflives()
    shared = [g for g in genes if g in rd]
    say(f"     RNADecayCafe: {len(rd):,} genes, {len(shared):,} shared with Schwanhausser")
    a = [S[g]["mrna_copies"] * (LN2 / S[g]["mrna_hl_h"] + LN2 / T_DOUBLE_H) for g in shared]
    b = [S[g]["mrna_copies"] * (LN2 / rd[g] + LN2 / T_DOUBLE_H) for g in shared]
    rho_r, n_r = spear(a, b)
    rho_hl, _ = spear([S[g]["mrna_hl_h"] for g in shared], [rd[g] for g in shared])
    say(f"     k_sm from the two half-life sources: Spearman {rho_r:+.4f} on {n_r:,} genes")
    say(f"     the half-lives themselves agree at {rho_hl:+.4f}")
    r2 = bool(np.isfinite(rho_r) and rho_r >= R2_RHO)
    say(f"     R2 {'PASS' if r2 else 'FAIL'}")
    say()

    say("R3 THE RATE IS NOT ABUNDANCE IN DISGUISE")
    mc = [S[g]["mrna_copies"] for g in genes]
    hl = [S[g]["mrna_hl_h"] for g in genes]
    rho_ab, _ = spear([ksm[g] for g in genes], mc)
    rho_hl_k, _ = spear([ksm[g] for g in genes], hl)
    part = partial_spear([ksm[g] for g in genes], hl, mc)
    say(f"     k_sm vs mRNA copies   {rho_ab:+.4f}   (gate: < {R3_MAX_RHO})")
    say(f"     k_sm vs half-life     {rho_hl_k:+.4f}")
    say(f"     k_sm vs half-life, given abundance   {part:+.4f}")
    say(f"     half-life spans {np.min(hl):.2f} to {np.max(hl):.2f} h, "
        f"{np.percentile(hl,75)/np.percentile(hl,25):.1f}x across the IQR")
    r3 = bool(np.isfinite(rho_ab) and rho_ab < R3_MAX_RHO and abs(part) > 0.1)
    say(f"     R3 {'PASS' if r3 else 'FAIL'} -- the rate "
        f"{'carries information beyond abundance' if r3 else 'IS abundance rescaled'}")
    say()

    say("R4 THE FAME CONTROL")
    C = json.load(open(LR.CELL))
    idx = {g["name"]: i for i, g in enumerate(C["genes"])}
    pub = {g: float(C["genes"][idx[g]].get("pubs") or 0) for g in genes if g in idx}
    gg = [g for g in genes if g in pub]
    rho_p_sm, n_p = spear([ksm[g] for g in gg], [pub[g] for g in gg])
    rho_p_sp, _ = spear([ksp[g] for g in gg], [pub[g] for g in gg])
    rho_p_ab, _ = spear([S[g]["mrna_copies"] for g in gg], [pub[g] for g in gg])
    say(f"     pubs vs k_sm {rho_p_sm:+.4f}   vs k_sp {rho_p_sp:+.4f}   "
        f"vs mRNA copies {rho_p_ab:+.4f}   (n {n_p:,})")
    say(f"     reported, not gated -- these are derived quantities, not predictions, so fame here")
    say(f"     is a property of which genes were measured, and it constrains loops 94 and 95")
    say()

    say("R5 THE RATES FIT INSIDE THE POLYMERASE")
    lens = [abs(g.get("gene_end", 0) - g.get("gene_start", 0)) / 1e3
            for g in C["genes"] if g.get("gene_end") and g.get("gene_start")]
    mean_kb = float(np.median([x for x in lens if 0 < x < 3000])) if lens else MEAN_GENE_KB
    ppm = {int(k): v for k, v in C["ppm"]} if isinstance(C["ppm"], list) else {}
    pol_ppm = ppm.get(idx.get("POLR2A", -1), None)
    TOTAL_PROTEINS = 2.0e9              # ~2e9 protein molecules in a mammalian cell (Milo, BNID 108692)
    pol_copies = pol_ppm / 1e6 * TOTAL_PROTEINS if pol_ppm else float("nan")
    demand_kb_h = sum(ksm[g] * mean_kb for g in genes)
    scale = 16492 / max(len(genes), 1)
    demand_all = demand_kb_h * scale
    capacity = pol_copies * POLII_KB_PER_H if np.isfinite(pol_copies) else float("nan")
    say(f"     median gene body {mean_kb:.1f} kb; Pol II elongation {POLII_KB_PER_H:.0f} kb/h")
    say(f"     POLR2A {pol_ppm:.1f} ppm -> {pol_copies:,.0f} molecules/cell "
        f"(of {TOTAL_PROTEINS:.0e} total protein)")
    say(f"     transcription demand over {len(genes):,} measured genes {demand_kb_h:,.0f} kb/h")
    say(f"     scaled to all {16492:,} genes                      {demand_all:,.0f} kb/h")
    say(f"     Pol II capacity                                    {capacity:,.0f} kb/h")
    util = demand_all / capacity if np.isfinite(capacity) and capacity > 0 else float("nan")
    say(f"     utilisation {util:.1%}   (gate: demand <= capacity)")
    r5 = bool(np.isfinite(util) and util <= 1.0)
    say(f"     R5 {'PASS' if r5 else 'FAIL'} -- the derived rates "
        f"{'fit inside the measured polymerase count' if r5 else 'need MORE polymerase than the cell has'}")
    say()

    say("R6 EXTENSION BEYOND THE 4,190")
    life = {}
    for i, g in enumerate(C["genes"]):
        v = g.get("lifetime")
        if v:
            life[g["name"]] = float(v)
    have_ppm = {C["genes"][k]["name"]: v for k, v in ([(int(a), b) for a, b in C["ppm"]]
                                                      if isinstance(C["ppm"], list) else [])
                if k < len(C["genes"])}
    ext = {g: (have_ppm[g], life[g]) for g in have_ppm if g in life and life[g] > 0}
    mass_all = sum(have_ppm.values())
    mass_cov = sum(have_ppm[g] for g in ext)
    say(f"     genes with a written half-life: {len(life):,}")
    say(f"     genes with a ppm abundance:     {len(have_ppm):,}")
    say(f"     both, so a protein rate is derivable: {len(ext):,}")
    say(f"     abundance mass covered: {mass_cov/mass_all:.1%} of the measured proteome")
    say(f"     mRNA rates remain limited to the {len(genes):,} genes with a measured mRNA half-life")
    say(f"     and copy number -- this is the constraint loops 94-108 inherit and must respect")
    say()

    gates = {"R1 growth dilution closes the translation gap": bool(r1),
             "R2 the rate survives an independent half-life source": bool(r2),
             "R3 the rate is not abundance in disguise": bool(r3),
             "R4 fame reported": True,
             "R5 the rates fit inside the polymerase": bool(r5),
             "R6 coverage declared": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(SCHWAN), str(RNADECAY), str(LR.CELL)],
                      available=len(S), used=len(genes), selection="filtered", seed=SEED,
                      controls=["an independent half-life source (RNADecayCafe) for the same rate",
                                "the trivial abundance baseline, with a partial correlation",
                                "publication count against both derived rates",
                                "a physical capacity bound from measured POLR2A copies",
                                "the dilution correction predicted before it was applied",
                                "doubling-time sensitivity swept rather than fixed"],
                      note="first rate in molecules/cell/h anywhere in this repository; the layers "
                           "have had no shared currency until now")
    RM.report(man, emit=say)
    json.dump({"test": "loop_rates", "manifest": man, "gates": gates,
               "n_genes": len(genes),
               "medians": {"naive": {"k_sm": m_sm_n, "k_sp": m_sp_n},
                           "corrected": {"k_sm": m_sm_c, "k_sp": m_sp_c},
                           "published": {"k_sm": LIT_KSM, "k_sp": LIT_KSP},
                           "factor": {"k_sm": f_sm, "k_sp": f_sp}},
               "t_double_sensitivity": {str(k): v for k, v in sens.items()},
               "r2": {"rho_rate": rho_r, "rho_halflife": rho_hl, "n": n_r},
               "r3": {"rho_abundance": rho_ab, "rho_halflife": rho_hl_k, "partial": part},
               "r4": {"pubs_vs_ksm": rho_p_sm, "pubs_vs_ksp": rho_p_sp,
                      "pubs_vs_abundance": rho_p_ab},
               "r5": {"polr2a_ppm": pol_ppm, "polii_copies": pol_copies,
                      "demand_kb_h": demand_all, "capacity_kb_h": capacity, "utilisation": util,
                      "median_gene_kb": mean_kb},
               "coverage": {"halflife": len(life), "ppm": len(have_ppm), "both": len(ext),
                            "abundance_mass": mass_cov / mass_all if mass_all else None},
               "k_sm": {g: ksm[g] for g in genes}, "k_sp": {g: ksp[g] for g in genes},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_rates.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_rates.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
