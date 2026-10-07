"""LOOP 116 -- GIVE THE CELL GEOMETRY: one compartment system, real volumes, and concentrations that must be possible.

WHAT IS LACKING, MEASURED RATHER THAN ASSERTED. The assembled cell has compartments and transport
and they work -- Human-GEM resolves 9 compartments and 4,742 of its 12,931 reactions (36.7%) move
material across a membrane, 714 of them carrying flux at the optimum, which is 31.0% of all active
reactions. What it does not have is GEOMETRY. Transport moves material between POOLS. A pool has no
volume, so it has no concentration, so nothing in this repository has ever been able to ask whether
a molecule count is physically possible.

AND THERE ARE TWO COMPARTMENT SYSTEMS THAT DISAGREE. The gene `comp` field has 12 buckets. Human-GEM
independently assigns every metabolite one of 9. Asking whether a gene's label matches the
compartment of the reaction it catalyses gives 9,356 of 23,225 gene-reaction pairs, or 40.3%. Three
fifths of the time the protein's annotated address and the address of its own chemistry disagree.
Some of that is legitimate -- a membrane protein acts between two compartments, and "plasma
membrane" and "cytoskeleton" have no GEM counterpart -- but a spatial model cannot be built on two
annotations that contradict each other, and picking one silently would hide the problem.

WHY GEOMETRY IS WORTH ADDING AND NOT DECORATION. A volume converts a count into a concentration, and
a concentration can be WRONG in a way a count cannot. Mammalian cytoplasm holds roughly 200-300 mg
of protein per mL and cannot hold much more; that is a hard physical ceiling set by macromolecular
crowding, and it is the same kind of bound as loop 92's ribosome budget and loop 103's proteasome
capacity -- countable, unfitted, and able to fail. Once volumes exist, the 4,190-gene state vector
stops being a list of numbers and becomes a set of concentrations that either fit inside the cell or
do not.

PREDECLARED, before any number:

  V1 ONE COMPARTMENT SYSTEM, WITH WHAT CANNOT BE MAPPED DECLARED     THE RECONCILIATION.
       an explicit mapping from the 12 gene labels to the 9 GEM compartments, the agreement rate
       recomputed under it, and every label that has no counterpart named. Gate: agreement must
       IMPROVE over the naive 40.3%, and the unmappable labels must be listed rather than dropped
       into a default bucket. If a mapping cannot beat 40.3% the two systems are not reconcilable
       and this loop says so instead of forcing them.
  V2 VOLUMES COME FROM LITERATURE RANGES AND ARE SWEPT, NOT FITTED   THE HONEST PARAMETER.
       organelle volume fractions for a mammalian cell, each as a RANGE, and cell volume swept over
       2,000-4,000 um^3. Nothing here is tuned to make V3 pass; the sweep is reported so the reader
       sees how much of V3's verdict is the constant.
  V3 THE PROTEIN CONCENTRATIONS ARE PHYSICALLY POSSIBLE              THE GATE.
       total protein mass per compartment volume against the macromolecular crowding ceiling of
       ~200-300 mg/mL. Gate: the cytosol must land INSIDE that ceiling at the central volume, and
       must not exceed it anywhere in the swept range by more than 2x. A cell whose cytosol computes
       to 3 g/mL is not a cell, and this is the first test in the repository that could say so.
  V4 TRANSPORT ACQUIRES A DIRECTION                                  THE NEW CAPABILITY.
       for the transport reactions carrying flux, the compartment volumes now let a flux be turned
       into a concentration change per hour. Report the distribution, and report how many reactions
       would need a concentration change faster than the compartment can physically buffer.
       Reported, not gated -- this is the first time the quantity has existed.
  V5 A NULL THAT CAN FIRE, AND THE FAME CONTROL                      THE GUARD.
       shuffle genes between compartments, keeping compartment sizes fixed, and recompute the
       concentration profile. gate_guard.null_can_move() confirms the shuffle moves the statistic
       BEFORE its verdict is read. Publication count against per-compartment protein mass reported
       alongside, since loop 102 found compartment mass correlates with study intensity at +0.4056.
  V6 COVERAGE DECLARED BY ALL THREE DENOMINATORS
       genes, proteome mass, and reactions -- because a compartment assignment covering a quarter of
       genes can cover most of the mass, and quoting only one is how coverage gets overstated.

-> outputs/loop_geometry.json
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
SEED = 11600
AVOGADRO = 6.02214076e23
DA_TO_G = 1.0 / AVOGADRO
MEAN_RES_DA = 110.0

# Mammalian organelle volume fractions, as RANGES from the cell-biology literature. Swept, never
# fitted. The point of a range is that V3's verdict has to survive all of it.
VOL_FRACTION = {
    "c": (0.45, 0.60),    # cytosol
    "n": (0.06, 0.12),    # nucleus
    "m": (0.08, 0.22),    # mitochondrion
    "r": (0.08, 0.15),    # endoplasmic reticulum
    "g": (0.02, 0.04),    # Golgi
    "l": (0.005, 0.015),  # lysosome
    "x": (0.002, 0.008),  # peroxisome
}
CELL_VOL_UM3 = (2000.0, 3000.0, 4000.0)
CROWDING_MG_ML = (200.0, 300.0)     # protein, mammalian cytoplasm
V3_TOLERANCE = 2.0

# The reconciliation. Every gene label gets a GEM compartment or an explicit reason it cannot.
LABEL_TO_GEM = {
    "nucleus": "n", "cytoplasm": "c", "mitochondrion": "m", "ER": "r", "Golgi": "g",
    "lysosome": "l", "peroxisome": "x", "extracellular": "e",
    # these have no GEM counterpart and are NOT silently mapped
    "plasma membrane": None, "membrane": None, "cytoskeleton": None, "endosome": None,
}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def protein_lengths():
    import gzip
    L, nm, c = {}, None, 0
    with gzip.open(LR.SC / "human_proteome.fasta.gz", "rt") as f:
        for ln in f:
            if ln.startswith(">"):
                if nm and c:
                    L[nm] = max(L.get(nm, 0), c)
                c, nm = 0, None
                for p in ln.split():
                    if p.startswith("GN="):
                        nm = p[3:]
                        break
            else:
                c += len(ln.strip())
    if nm and c:
        L[nm] = max(L.get(nm, 0), c)
    return L


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 116 -- give the cell geometry: one compartment system, real volumes, "
        "possible concentrations")
    say("=" * 100)
    say()

    C = json.load(open(LR.CELL))
    names = [g["name"] for g in C["genes"]]
    lab = {g["name"]: g.get("comp") for g in C["genes"]}
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    S = json.load(open(LR.SC / "_schwan2011.json"))
    plen = protein_lengths()
    import cell_sim as CS
    m = CS.load_model()
    sym = CS.ensembl_to_symbol()
    say(f"  {len(names):,} genes, {len(m.reactions):,} reactions, "
        f"{len({x.compartment for x in m.metabolites})} GEM compartments")
    say()

    say("V1 ONE COMPARTMENT SYSTEM, WITH WHAT CANNOT BE MAPPED DECLARED")
    naive = agree = tot = 0
    GEMC = {"c": "cytoplasm", "e": "extracellular", "m": "mitochondrion", "r": "ER",
            "g": "Golgi", "l": "lysosome", "x": "peroxisome", "n": "nucleus", "i": "mitochondrion"}
    unmappable = collections.Counter()
    for r in m.reactions:
        if not r.genes:
            continue
        cs = {x.compartment for x in r.metabolites}
        for g in r.genes:
            s = sym.get(g.id, g.id)
            if s not in lab or not lab[s]:
                continue
            tot += 1
            if any(GEMC.get(c) == lab[s] for c in cs):
                naive += 1
            tgt = LABEL_TO_GEM.get(lab[s], "MISSING")
            if tgt is None:
                unmappable[lab[s]] += 1
                continue
            if tgt in cs or (tgt == "m" and "i" in cs):
                agree += 1
    say(f"     naive label-vs-GEM agreement      {naive:,}/{tot:,} = {naive/tot:.1%}")
    say(f"     under the explicit mapping        {agree:,}/{tot:,} = {agree/tot:.1%}")
    say(f"     gene-reaction pairs whose label has NO GEM counterpart: "
        f"{sum(unmappable.values()):,}")
    for k, v in unmappable.most_common():
        say(f"       {k:18s} {v:6,}  -- named, not defaulted")
    say(f"     of the reconcilable pairs alone: "
        f"{agree/(tot-sum(unmappable.values())):.1%}")
    v1 = agree / tot > naive / tot
    say(f"     V1 {'PASS' if v1 else 'FAIL'} -- the mapping "
        f"{'improves on the naive match' if v1 else 'does NOT improve, so the systems are not reconcilable'}")
    say()

    say("V2 VOLUMES FROM LITERATURE RANGES, SWEPT, NOT FITTED")
    for k, (lo, hi) in VOL_FRACTION.items():
        say(f"     {GEMC[k]:15s} ({k})  {lo:.3f} - {hi:.3f} of cell volume")
    say(f"     cell volume swept over {CELL_VOL_UM3} um^3")
    say(f"     nothing here is tuned to make V3 pass; the sweep is the honesty")
    say()

    say("V3 THE PROTEIN CONCENTRATIONS ARE PHYSICALLY POSSIBLE")
    # assign each measured protein to a compartment via the explicit mapping
    bycomp = collections.defaultdict(float)
    assigned = 0
    for g, v in S.items():
        if not v.get("prot_copies") or g not in plen or g not in lab or not lab[g]:
            continue
        tgt = LABEL_TO_GEM.get(lab[g])
        if tgt is None or tgt == "e":
            continue
        bycomp[tgt] += v["prot_copies"] * plen[g] * MEAN_RES_DA * DA_TO_G
        assigned += 1
    say(f"     {assigned:,} measured proteins assigned to an intracellular GEM compartment")
    say(f"     total assigned protein mass {sum(bycomp.values())*1e12:.1f} pg")
    rows, worst = [], 0.0
    for cv in CELL_VOL_UM3:
        for end, frac_idx in (("low-volume", 0), ("high-volume", 1)):
            conc = {}
            for k, g_mass in bycomp.items():
                if k not in VOL_FRACTION:
                    continue
                vol_ml = cv * VOL_FRACTION[k][frac_idx] * 1e-12    # um^3 -> mL
                conc[k] = g_mass / vol_ml * 1e3                     # mg/mL
            rows.append({"cell_vol_um3": cv, "end": end,
                         **{f"conc_{k}": conc.get(k) for k in conc}})
            cyt = conc.get("c", float("nan"))
            worst = max(worst, cyt)
            say(f"     cell {cv:,.0f} um^3, {end:11s} cytosol {cyt:7.1f} mg/mL   "
                + "  ".join(f"{GEMC[k][:4]} {conc[k]:6.1f}" for k in sorted(conc) if k != "c"))
    lo, hi = CROWDING_MG_ML
    central = [r for r in rows if r["cell_vol_um3"] == 3000.0]
    cyt_central = float(np.mean([r["conc_c"] for r in central]))
    say(f"     crowding ceiling {lo:.0f}-{hi:.0f} mg/mL protein")
    say(f"     cytosol at the central volume: {cyt_central:.1f} mg/mL")
    say(f"     worst across the whole sweep:  {worst:.1f} mg/mL "
        f"({worst/hi:.2f}x the ceiling; gate allows {V3_TOLERANCE:.0f}x)")
    v3 = bool(cyt_central <= hi and worst <= hi * V3_TOLERANCE)
    say(f"     V3 {'PASS' if v3 else 'FAIL'} -- the assembled proteome "
        f"{'fits inside a cell' if v3 else 'does NOT fit inside a cell'}")
    say()

    say("V4 TRANSPORT ACQUIRES A DIRECTION")
    CS.set_medium(m)
    sol = m.optimize()
    tr = [r for r in m.reactions if len({x.compartment for x in r.metabolites}) > 1]
    act = [r for r in tr if abs(sol.fluxes[r.id]) > 1e-9]
    GDW_PER_CELL = 3000e-15 * 1.1 * 0.20
    rates = []
    for r in act:
        cs = {x.compartment for x in r.metabolites} & set(VOL_FRACTION)
        if not cs:
            continue
        k = min(cs, key=lambda c: VOL_FRACTION[c][0])
        vol_l = 3000.0 * np.mean(VOL_FRACTION[k]) * 1e-15          # um^3 -> L
        mol_h = abs(sol.fluxes[r.id]) * GDW_PER_CELL * 1e-3        # mmol/gDW/h -> mol/cell/h
        rates.append(mol_h / vol_l)                                 # M/h
    rates = np.array(rates)
    say(f"     {len(act):,} transport reactions carry flux, {len(rates):,} into a volumed compartment")
    say(f"     implied concentration change: median {np.median(rates):.3g} M/h, "
        f"90th {np.percentile(rates,90):.3g}, max {rates.max():.3g}")
    say(f"     above 1 M/h (implausible for any metabolite pool): "
        f"{int((rates>1).sum()):,} ({(rates>1).mean():.1%})")
    say(f"     this quantity has not existed before in this repository -- reported, not gated")
    say()

    say("V5 A NULL THAT CAN FIRE, AND THE FAME CONTROL")
    rng = np.random.default_rng(SEED)
    gs = [g for g in S if S[g].get("prot_copies") and g in plen and lab.get(g)
          and LABEL_TO_GEM.get(lab[g]) not in (None, "e")]
    real_lab = [LABEL_TO_GEM[lab[g]] for g in gs]
    nulls = []
    for _ in range(20):
        sh = list(rng.permutation(real_lab))
        bc = collections.defaultdict(float)
        for g, t in zip(gs, sh):
            bc[t] += S[g]["prot_copies"] * plen[g] * MEAN_RES_DA * DA_TO_G
        vol_ml = 3000.0 * np.mean(VOL_FRACTION["c"]) * 1e-12
        nulls.append(bc.get("c", 0.0) / vol_ml * 1e3)
    cap = GG.null_can_move(real_lab, list(rng.permutation(real_lab)))
    say(f"     CAPABILITY: shuffling compartment labels changes {cap['changed']:.1%} of "
        f"assignments -- capable: {cap['capable']}")
    real_c = bycomp["c"] / (3000.0 * np.mean(VOL_FRACTION["c"]) * 1e-12) * 1e3
    s = GG.survival(real_c, nulls)
    GG.report("cytosolic protein concentration under a compartment shuffle", s, emit=say)
    from scipy.stats import spearmanr
    mass_by = collections.defaultdict(float)
    pub_by = collections.defaultdict(float)
    for g in gs:
        t = LABEL_TO_GEM[lab[g]]
        mass_by[t] += S[g]["prot_copies"] * plen[g]
        pub_by[t] += pubs.get(g, 0.0)
    ks = sorted(mass_by)
    r_p = float(spearmanr([mass_by[k] for k in ks], [pub_by[k] for k in ks]).statistic)
    say(f"     per-compartment protein mass vs publication mass: rho {r_p:+.4f} over {len(ks)} "
        f"compartments (loop 102 found +0.4056 on its own 12-bucket version)")
    v5 = bool(cap["capable"])
    say(f"     V5 {'PASS' if v5 else 'FAIL'}")
    say()

    say("V6 COVERAGE DECLARED BY ALL THREE DENOMINATORS")
    tot_mass = sum(S[g]["prot_copies"] * plen[g] for g in S
                   if S[g].get("prot_copies") and g in plen)
    cov_mass = sum(S[g]["prot_copies"] * plen[g] for g in gs)
    n_rxn_comp = len([r for r in m.reactions if r.metabolites])
    say(f"     by GENE           {assigned:,} of {len(names):,} = {assigned/len(names):.1%}")
    say(f"     by PROTEOME MASS  {cov_mass/tot_mass:.1%} of measured mass")
    say(f"     by REACTION       every one of {n_rxn_comp:,} reactions is compartment-resolved "
        f"in GEM already")
    say(f"     the gene layer is the sparse one; the chemistry layer was never missing compartments")
    say()

    gates = {"V1 one compartment system, unmappable labels declared": bool(v1),
             "V2 volumes from literature ranges, swept": True,
             "V3 the protein concentrations are physically possible": bool(v3),
             "V4 transport acquires a direction": True,
             "V5 the null is capable and fame reported": bool(v5),
             "V6 coverage by all three denominators": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(LR.CELL), str(LR.SC / "_schwan2011.json"),
                              str(LR.SC / "human_proteome.fasta.gz"), str(CS.SBML)],
                      available=len(names), used=assigned, selection="filtered", seed=SEED,
                      controls=["organelle volumes as literature RANGES, swept, never fitted",
                                "a crowding ceiling that the assembled proteome could exceed",
                                "unmappable compartment labels named rather than defaulted",
                                "a compartment shuffle checked for capability before its verdict",
                                "publication mass per compartment reported next to protein mass",
                                "coverage by gene, by mass and by reaction, all three"],
                      note="the cell had compartments and transport but no volumes, so no "
                           "concentration, so nothing could be physically impossible")
    RM.report(man, emit=say)
    json.dump({"test": "loop_geometry", "manifest": man, "gates": gates,
               "v1": {"naive_agreement": naive / tot, "mapped_agreement": agree / tot,
                      "unmappable": dict(unmappable), "n_pairs": tot},
               "v2": {"vol_fraction": {k: list(v) for k, v in VOL_FRACTION.items()},
                      "cell_vol_um3": list(CELL_VOL_UM3)},
               "v3": {"rows": rows, "cytosol_central_mg_ml": cyt_central,
                      "worst_mg_ml": worst, "ceiling": list(CROWDING_MG_ML),
                      "n_assigned": assigned},
               "v4": {"n_active_transport": len(act), "n_volumed": len(rates),
                      "median_M_per_h": float(np.median(rates)),
                      "p90": float(np.percentile(rates, 90)), "max": float(rates.max()),
                      "frac_above_1M_h": float((rates > 1).mean())},
               "v5": {"capability": cap, "survival": s, "pubs_rho": r_p},
               "v6": {"by_gene": assigned / len(names), "by_mass": cov_mass / tot_mass},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_geometry.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_geometry.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
