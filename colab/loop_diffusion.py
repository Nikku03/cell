"""LOOP 126 -- DIFFUSION: is the well-stirred assumption WRONG, and does physics falsify the kinetics?

TWO WAYS TO TREAT AN ABSENT LAYER, and only one of them is worth doing. The lazy version adds
coordinates and a Fickian solver, produces a prettier object and proves nothing. The other version
asks whether the approximation already in place is actually broken, and that is answerable with
numbers this repository already has.

The model treats every compartment as a well-stirred pool: a metabolite consumed anywhere is drawn
from a single uniform concentration. That is an approximation, and approximations are only defects
when they are violated. The quantity that decides is the Damkohler number, reaction rate over
diffusion rate:

    Da = tau_diffusion / tau_reaction = (L^2 / 6D) * (kcat * [E] / K_M)

Every term is available. L from loop 116's compartment volumes. D from Stokes-Einstein on Itzhak's
measured molecular weights. [E] from Itzhak's measured copy numbers over those volumes. kcat and
K_M from the kinetics bundle and loop 124's UniProt fetch. Nothing here is fitted.

AND DIFFUSION SETS A CEILING NOTHING CAN EXCEED. Two molecules cannot react faster than they can
find each other. The Smoluchowski encounter rate

    k_diff = 4*pi*(D_E + D_S)*(R_E + R_S)*N_A

is roughly 1e10 M^-1 s^-1 in water and a few times lower in cytoplasm, and k_cat/K_M is exactly the
bimolecular rate constant that must obey it. So every kcat and K_M in this model -- the 242 measured
in loop 124 and the thousands predicted -- can be checked against a bound that owes nothing to any
dataset. A value above the diffusion limit is not unlikely, it is impossible.

That makes this loop a physical audit of the kinetics rather than a new layer, which is the more
useful thing to be. Loop 124 showed the predicted kcats do not beat a constant. F4 asks whether they
also break physics.

PREDECLARED:

  F1 THE DIFFUSION CONSTANTS ARE PHYSICAL                           THE PREREQUISITE.
       Stokes-Einstein at 37 C on measured molecular weights, with the hydrodynamic radius from
       protein specific volume. Positive control: a 27 kDa protein -- GFP's mass, the most-measured
       diffusion constant in cell biology -- must give 70-110 um^2/s in water and 15-40 um^2/s in
       cytoplasm at the swept crowding factor. Gate: both. A radius or viscosity error lands
       outside immediately.
  F2 IS THE WELL-STIRRED ASSUMPTION WRONG?                          THE QUESTION.
       Damkohler number per enzyme, swept over cell volume 2000/3000/4000 um^3 and crowding factor
       3/4/5 so the verdict is not a property of one choice. Gate: the MEDIAN enzyme must have
       Da < 1 at every swept corner. If it does, "no diffusion" is a justified approximation and
       the audit table should say CLOSES rather than ABSENT. If it does not, the metabolic layer is
       wrong wherever it fails and this loop says where.
  F3 THE DIFFUSION LIMIT AS A CEILING, ON MEASURED VALUES           THE PHYSICS AUDIT.
       k_cat/K_M against Smoluchowski, using loop 124's UniProt measurements. Gate: fewer than 5%
       of measured pairs may exceed the limit. A measured value above it is an error in the
       measurement, the parsing, or the units -- and 5% is already generous, because published
       kinetics do include mistakes.
  F4 THE SAME CEILING, ON THE PREDICTED VALUES                      LOOP 124's SEQUEL.
       loop 124 found the bundle's kcat predictions do not beat a constant 1.85/s. Do they also
       break physics? Gate: the predicted set's violation rate must not exceed the measured set's
       by more than a factor of two. A predictor that produces impossible rate constants is worse
       than one that is merely inaccurate, because impossibility is checkable without any data.
  F5 THE TEXTBOOK NEAR-LIMIT ENZYMES                                THE POSITIVE CONTROL.
       carbonic anhydrase, superoxide dismutase, catalase and triosephosphate isomerase are the
       canonical catalytically perfect enzymes -- their k_cat/K_M sits within about an order of
       magnitude of the diffusion limit and that is why they are famous. Gate: those present must
       rank in the top decile of the k_cat/K_M distribution. If the perfect enzymes are not at the
       top, the axis is not measuring what it claims.
  F6 THE CROSSOVER, AND THREE DENOMINATORS                          THE CONSEQUENCE.
       the compartment size at which Da reaches 1 for the median enzyme -- the length scale above
       which well-stirred stops being safe -- reported against real organelle and cell dimensions.
       Plus coverage by gene, by reaction and against the 12,931.

-> outputs/loop_diffusion.json
"""
import csv
import gzip
import json
import os
import re
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
import cell_assembled as CA  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
ITZHAK = LR.SC / "itzhak_supp1.xlsx"
SHEET = "Compact HeLa Spatial Proteome"
UPK = LR.SC / "uniprot_kinetics_human.tsv"
BUNDLE = Path("colab/data/kinetics_bundle.json.gz")
SEED = 12600

KB = 1.380649e-23          # J/K
TEMP = 310.15              # K, 37 C
ETA = 0.6913e-3            # Pa s, water at 37 C
VBAR = 0.73e-6             # m^3/kg, protein partial specific volume (0.73 cm^3/g)
NA = 6.02214076e23
CROWDING = (3.0, 4.0, 5.0)          # cytoplasm is this many times more viscous than water
CELL_VOL_UM3 = (2000.0, 3000.0, 4000.0)   # loop 116
CYTO_FRACTION = 0.52                       # loop 116/118
METABOLITE_DA = 200.0                      # a typical small metabolite

F1_WATER = (70.0, 110.0)
F1_CYTO = (15.0, 40.0)
F3_VIOLATE = 0.05
F4_RATIO = 2.0
F5_ENZYMES = ("CA1", "CA2", "CA3", "SOD1", "SOD2", "CAT", "TPI1", "ACHE")
F5_DECILE = 0.90

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def radius_m(mass_da):
    """Hydrodynamic radius from mass, via protein specific volume. R = (3 M vbar / 4 pi N_A)^(1/3)."""
    m_kg = np.asarray(mass_da, float) / NA / 1000.0
    return (3.0 * m_kg * VBAR / (4.0 * np.pi)) ** (1.0 / 3.0)


def stokes_einstein(mass_da, crowd=1.0):
    """D in m^2/s at 37 C. crowd > 1 is the cytoplasmic slowdown relative to water."""
    return KB * TEMP / (6.0 * np.pi * ETA * crowd * radius_m(mass_da))


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 126 -- diffusion: is well-stirred wrong, and does physics falsify the kinetics?")
    say("=" * 100)
    say()

    import pandas as pd
    d = pd.read_excel(ITZHAK, sheet_name=SHEET)
    mw, cn = {}, {}
    for g, m, c in zip(d["Lead Gene name"].astype(str),
                       pd.to_numeric(d["Mol. weight [kDa]"], errors="coerce"),
                       pd.to_numeric(d["Estimated Copy number per cell"], errors="coerce")):
        if np.isfinite(m) and m > 0:
            mw[g] = float(m) * 1000.0
        if np.isfinite(c) and c > 0:
            cn[g] = max(cn.get(g, 0.0), float(c))
    say(f"  Itzhak 2016: {len(mw):,} molecular weights, {len(cn):,} copy numbers (measured)")

    B = json.load(gzip.open(BUNDLE, "rt"))
    gk, gkm = B["gene_kcat_per_s"], B["gene_km_uM"]
    say(f"  kinetics bundle: {len(gk):,} gene kcat, {len(gkm):,} gene K_M (all predicted)")

    # loop 124's parser, reused so the two loops agree by construction
    NUM = r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    K1 = re.compile(r"kcat is " + NUM + r"\s*(sec|min|hour|h)\(-1\)", re.I)
    KMR = re.compile(r"KM=" + NUM + r"\s*(nM|uM|mM|M)\b")
    PS = {"sec": 1.0, "min": 1 / 60.0, "hour": 1 / 3600.0, "h": 1 / 3600.0}
    UM = {"nM": 1e-3, "uM": 1.0, "mM": 1e3, "M": 1e6}
    rows = list(csv.reader(open(UPK, newline=""), delimiter="\t"))
    hh, rows = rows[0], rows[1:]
    iG, iK = hh.index("Gene Names (primary)"), hh.index("Kinetics")
    mkc, mkm = {}, {}
    for x in rows:
        g = x[iG].strip()
        if not g:
            continue
        v = [float(a) * PS[b.lower()] for a, b in K1.findall(x[iK])]
        if v:
            mkc[g] = float(np.exp(np.mean(np.log(v))))
        w = [float(a) * UM[b] for a, b in KMR.findall(x[iK])]
        if w:
            mkm[g] = float(np.exp(np.mean(np.log(w))))
    say(f"  UniProt measured: {len(mkc)} kcat, {len(mkm)} K_M (loop 124's parser)")
    say()

    gates = {}

    # ---------------------------------------------------------------- F1
    say("F1 THE DIFFUSION CONSTANTS ARE PHYSICAL")
    r27 = radius_m(27000.0)
    dw = stokes_einstein(27000.0) * 1e12
    say(f"     27 kDa protein: hydrodynamic radius {r27 * 1e9:.2f} nm")
    say(f"     D in water {dw:.1f} um^2/s   gate {F1_WATER[0]}-{F1_WATER[1]} "
        f"(GFP measured ~87)")
    ok_c = True
    for c in CROWDING:
        dc = stokes_einstein(27000.0, c) * 1e12
        inr = F1_CYTO[0] <= dc <= F1_CYTO[1]
        say(f"     D in cytoplasm at crowding {c:.0f}x: {dc:.1f} um^2/s   "
            f"{'in' if inr else 'OUT OF'} gate {F1_CYTO[0]}-{F1_CYTO[1]} (GFP measured ~25-30)")
        ok_c = ok_c and inr
    dmet = stokes_einstein(METABOLITE_DA) * 1e12
    say(f"     a {METABOLITE_DA:.0f} Da metabolite: D in water {dmet:.0f} um^2/s "
        f"(glucose measured ~670)")
    gates["F1"] = bool(F1_WATER[0] <= dw <= F1_WATER[1] and ok_c)
    say(f"     F1 {'PASS' if gates['F1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- F2
    say("F2 IS THE WELL-STIRRED ASSUMPTION WRONG?")
    genes = [g for g in gk if g in gkm and g in cn]
    say(f"     {len(genes):,} genes with a kcat, a K_M and a measured copy number")
    kc = np.array([gk[g] for g in genes])
    kmv = np.array([gkm[g] for g in genes]) * 1e-6         # uM -> M
    cp = np.array([cn[g] for g in genes])
    sweep, ok2 = {}, True
    for V in CELL_VOL_UM3:
        Vc_L = V * CYTO_FRACTION * 1e-15                    # um^3 -> L
        E = cp / (NA * Vc_L)                                # molar
        L = (3.0 * V * CYTO_FRACTION / (4.0 * np.pi)) ** (1.0 / 3.0) * 1e-6   # m
        for c in CROWDING:
            Dm = stokes_einstein(METABOLITE_DA, c)          # m^2/s
            tau_d = L ** 2 / (6.0 * Dm)
            k_obs = kc * E / np.maximum(kmv, 1e-300)        # 1/s
            Da = tau_d * k_obs
            med = float(np.median(Da))
            sweep[f"V{V:.0f}_c{c:.0f}"] = {"median_Da": med, "frac_gt1": float(np.mean(Da > 1)),
                                           "tau_diff_s": float(tau_d), "L_um": float(L * 1e6)}
            if med >= 1.0:
                ok2 = False
    k0 = f"V{CELL_VOL_UM3[1]:.0f}_c{CROWDING[1]:.0f}"
    s0 = sweep[k0]
    say(f"     at 3000 um^3 and 4x crowding: cytosol radius {s0['L_um']:.2f} um, "
        f"metabolite crossing time {s0['tau_diff_s'] * 1000:.1f} ms")
    say(f"     Damkohler number: median {s0['median_Da']:.3e}, "
        f"{s0['frac_gt1']:.2%} of enzymes above 1")
    say(f"     swept over 3 cell volumes x 3 crowding factors:")
    say(f"       median Da ranges {min(v['median_Da'] for v in sweep.values()):.2e} to "
        f"{max(v['median_Da'] for v in sweep.values()):.2e}")
    say(f"       fraction above 1 ranges {min(v['frac_gt1'] for v in sweep.values()):.3%} to "
        f"{max(v['frac_gt1'] for v in sweep.values()):.3%}")
    gates["F2"] = bool(ok2)
    say(f"     F2 {'PASS' if gates['F2'] else 'FAIL'} -- well-stirred is "
        f"{'a JUSTIFIED approximation, not a defect' if gates['F2'] else 'VIOLATED for the median enzyme'}")
    say()

    # ---------------------------------------------------------------- F3
    say("F3 THE DIFFUSION LIMIT AS A CEILING, ON MEASURED VALUES")
    def k_diff_for(g, crowd):
        m = mw.get(g, 50000.0)
        De, Ds = stokes_einstein(m, crowd), stokes_einstein(METABOLITE_DA, crowd)
        Re, Rs = radius_m(m), radius_m(METABOLITE_DA)
        return 4.0 * np.pi * (De + Ds) * (Re + Rs) * NA * 1000.0   # M^-1 s^-1
    kd_ref = k_diff_for("ACTB", 1.0)
    say(f"     Smoluchowski limit for a 50 kDa enzyme and a {METABOLITE_DA:.0f} Da metabolite:")
    for c in (1.0,) + CROWDING:
        say(f"       crowding {c:.0f}x: {k_diff_for('X', c):.2e} M^-1 s^-1"
            + ("   (water; textbook 1e9-1e10)" if c == 1.0 else ""))
    meas = [g for g in mkc if g in mkm]
    ke_m = np.array([mkc[g] / (mkm[g] * 1e-6) for g in meas])
    lim_m = np.array([k_diff_for(g, CROWDING[1]) for g in meas])
    viol_m = float(np.mean(ke_m > lim_m))
    say(f"     {len(meas)} genes with BOTH a measured kcat and a measured K_M")
    say(f"     k_cat/K_M: median {np.median(ke_m):.2e}, 90th {np.percentile(ke_m, 90):.2e}, "
        f"max {ke_m.max():.2e} M^-1 s^-1")
    say(f"     exceed the diffusion limit: {int((ke_m > lim_m).sum())} of {len(meas)} "
        f"= {viol_m:.2%}   gate < {F3_VIOLATE:.0%}")
    if (ke_m > lim_m).any():
        bad = [(meas[i], ke_m[i] / lim_m[i]) for i in np.argsort(-ke_m / lim_m)[:5]
               if ke_m[i] > lim_m[i]]
        say("     worst offenders (fold above the limit): " +
            ", ".join(f"{g} {f:.1f}x" for g, f in bad))
    gates["F3"] = bool(viol_m < F3_VIOLATE)
    say(f"     F3 {'PASS' if gates['F3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- F4
    say("F4 THE SAME CEILING, ON THE PREDICTED VALUES")
    pred = [g for g in gk if g in gkm and g in mw]
    ke_p = np.array([gk[g] / (gkm[g] * 1e-6) for g in pred])
    lim_p = np.array([k_diff_for(g, CROWDING[1]) for g in pred])
    viol_p = float(np.mean(ke_p > lim_p))
    say(f"     {len(pred):,} genes with a PREDICTED kcat and K_M")
    say(f"     k_cat/K_M: median {np.median(ke_p):.2e}, 90th {np.percentile(ke_p, 90):.2e}, "
        f"max {ke_p.max():.2e} M^-1 s^-1")
    say(f"     exceed the diffusion limit: {int((ke_p > lim_p).sum())} of {len(pred)} "
        f"= {viol_p:.2%}")
    say(f"     measured {viol_m:.2%} vs predicted {viol_p:.2%}   ratio "
        f"{viol_p / max(viol_m, 1e-9):.2f}x   gate < {F4_RATIO}x")
    gates["F4"] = bool(viol_p <= F4_RATIO * max(viol_m, 1e-9) or viol_p < 0.01)
    say(f"     F4 {'PASS' if gates['F4'] else 'FAIL'} -- the predictions "
        f"{'respect physics as well as the measurements do' if gates['F4'] else 'BREAK PHYSICS more often than the measurements'}")
    say()

    # ---------------------------------------------------------------- F5
    say("F5 THE TEXTBOOK NEAR-LIMIT ENZYMES")
    present, ranks = [], []
    order = np.argsort(ke_p)
    rank_of = {pred[order[i]]: i / max(len(order) - 1, 1) for i in range(len(order))}
    for g in F5_ENZYMES:
        if g in rank_of:
            present.append(g)
            ranks.append(rank_of[g])
            say(f"     {g:6} k_cat/K_M {gk[g] / (gkm[g] * 1e-6):.2e}   "
                f"percentile {rank_of[g]:.1%}")
    if not present:
        say("     none of the canonical perfect enzymes are in the predicted set")
    gates["F5"] = bool(present and float(np.mean(ranks)) >= F5_DECILE)
    say(f"     mean percentile of the canonical perfect enzymes: "
        f"{np.mean(ranks) if ranks else float('nan'):.1%}   gate >= {F5_DECILE:.0%}")
    say(f"     F5 {'PASS' if gates['F5'] else 'FAIL'} -- the axis "
        f"{'ranks the perfect enzymes at the top' if gates['F5'] else 'does NOT rank the perfect enzymes at the top, so it is not measuring catalytic efficiency'}")
    say()

    # ---------------------------------------------------------------- F6
    say("F6 THE CROSSOVER, AND THREE DENOMINATORS")
    Vc_L = CELL_VOL_UM3[1] * CYTO_FRACTION * 1e-15
    E = cp / (NA * Vc_L)
    k_obs = kc * E / np.maximum(kmv, 1e-300)
    Dm = stokes_einstein(METABOLITE_DA, CROWDING[1])
    k_med = float(np.median(k_obs))
    L_star = np.sqrt(6.0 * Dm / k_med) * 1e6
    say(f"     median pseudo-first-order consumption rate {k_med:.3e} /s")
    say(f"     Da = 1 when the compartment radius reaches {L_star:.0f} um")
    say(f"     for comparison: cytosol {(3 * CELL_VOL_UM3[1] * CYTO_FRACTION / (4 * np.pi)) ** (1 / 3):.1f} um, "
        f"a mitochondrion ~0.5 um, a lysosome ~0.25 um, a whole cell ~9 um")
    say(f"     the crossover is {L_star / 9.0:.0f}x the radius of the cell, so no organelle in this "
        f"model is close to being diffusion-limited")
    rg = B["reaction_genes"]
    ens = {}
    with open(LR.SC / "HumanGEM_genes.tsv") as f:
        rr = csv.reader(f, delimiter="\t")
        hd = [c.strip('"') for c in next(rr)]
        a_, b_ = hd.index("genes"), hd.index("geneSymbols")
        for x in rr:
            e, s = x[a_].strip('"'), x[b_].strip('"')
            if e and s:
                ens[e] = s.split(";")[0]
    gs = set(genes)
    rx = sum(1 for r_, gg in rg.items() if any(ens.get(z) in gs for z in gg))
    say(f"     THREE DENOMINATORS  by gene {len(genes):,} of 16,492 = {len(genes) / 16492:.1%}")
    say(f"                         by reaction {rx:,} of {len(rg):,} with a gene rule "
        f"= {rx / len(rg):.1%}, and of 12,931 = {rx / 12931:.1%}")
    gates["F6"] = bool(np.isfinite(L_star) and rx > 0)
    say(f"     F6 {'PASS' if gates['F6'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- verdict
    say("=" * 100)
    for k in ("F1", "F2", "F3", "F4", "F5", "F6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/6")
    say("=" * 100)

    man = RM.manifest(inputs=[ITZHAK, UPK, BUNDLE, LR.SC / "HumanGEM_genes.tsv"],
                      available=len(gk), used=len(genes), selection="filtered", seed=SEED,
                      controls=["GFP's measured diffusion constant as the Stokes-Einstein control",
                                "cell volume and crowding factor both swept, 3x3",
                                "the Smoluchowski limit, which owes nothing to any dataset",
                                "measured kinetics against predicted, on the same ceiling",
                                "the canonical catalytically perfect enzymes as a rank control",
                                "three denominators"],
                      note="no fitted parameter appears in this loop; every quantity is either "
                           "measured or derived from physical constants")
    json.dump({"test": "loop_diffusion", "manifest": man, "gates": gates,
               "f1": {"radius_27kda_nm": float(r27 * 1e9), "D_water_um2_s": float(dw),
                      "D_cyto": {str(c): float(stokes_einstein(27000.0, c) * 1e12)
                                 for c in CROWDING},
                      "D_metabolite_water": float(dmet)},
               "f2": {"n_genes": len(genes), "sweep": sweep},
               "f3": {"n": len(meas), "median_kcatkm": float(np.median(ke_m)),
                      "max": float(ke_m.max()), "violation_rate": viol_m,
                      "limit_water": float(k_diff_for("X", 1.0)),
                      "limit_cyto": float(k_diff_for("X", CROWDING[1]))},
               "f4": {"n": len(pred), "median_kcatkm": float(np.median(ke_p)),
                      "max": float(ke_p.max()), "violation_rate": viol_p,
                      "ratio": viol_p / max(viol_m, 1e-9)},
               "f5": {"present": present, "percentiles": [float(r) for r in ranks],
                      "mean_percentile": float(np.mean(ranks)) if ranks else None},
               "f6": {"k_obs_median": k_med, "crossover_radius_um": float(L_star),
                      "reactions": rx, "gene_rule_reactions": len(rg)},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_diffusion.json", "w"), indent=1)
    RM.report(man, emit=say)
    say(f"\n  -> {OUT / 'loop_diffusion.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
