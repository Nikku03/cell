"""Clade-aware cross-validation splits for the multi-organism essentiality predictor.

THE GATE: every same-genus organism is held out together. Without this,
in-sample MCC numbers from cross-organism CV are inflated because the
training set contains close relatives of the held-out organism, which
defeats the entire point of leave-one-out evaluation.

Mapping policy:
  - Genus-level holdout. All Pseudomonas variants are one unit, all
    Ralstonia variants are one unit, etc.
  - Within a genus, all strain variants and all data sources are LOCKED
    TOGETHER (cannot end up in different folds).
  - Per user directive: "lock the 3 E. coli, 3 S. aureus, the
    Mycoplasma/M. mycoides cluster together." We treat the Mycoplasmoid
    minimal genomes (syn3A + M. genitalium + M. pneumoniae) as one
    clade unit because they're functionally near-equivalent and were
    constructed from the same lineage.

Input  : memory_bank/data/multiorg_essentiality/labels.csv (59 organism keys)
Output : memory_bank/data/multiorg_essentiality/clade_splits.csv
           one row per organism with assigned clade + fold

For BERIL Fitness Browser orgIds, we use a hardcoded lookup derived from:
  kbaseincubator/BERIL-research-observatory/projects/functional_dark_matter/
  data/extended_tractable_organisms.tsv

Run:
  python scripts/build_clade_splits.py
  # optional: --n-folds 5  --seed 0
"""
from __future__ import annotations
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
LABELS_CSV  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "labels.csv"
OUT_CSV     = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "clade_splits.csv"
INV_CSV     = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "clade_inventory.csv"


# Manual genus-level mapping for non-BERIL organism keys.
ORGANISM_TO_CLADE_MANUAL: dict[str, str] = {
    # ---- Mycoplasmoid minimal cells (LOCK TOGETHER per user directive) ----
    "syn3a":    "mycoplasmoid_minimal",
    "mgen":     "mycoplasmoid_minimal",
    "mgen_deg": "mycoplasmoid_minimal",
    "mpne":     "mycoplasmoid_minimal",
    "mpne_deg": "mycoplasmoid_minimal",
    # M. mycoides DELIBERATELY excluded from data (label-leakage for syn3A)
    # If it's added later, route here.

    # ---- Mycobacterium tuberculosis (separate from Mycoplasma!) ----
    "mtub":     "mycobacterium",
    "mtub_deg": "mycobacterium",

    # ---- Streptococcus pneumoniae (lock all data sources together) ----
    "spneT4":   "streptococcus",
    "spne19F":  "streptococcus",
    "spne_deg": "streptococcus",
    "ssan":     "streptococcus",   # S. sanguinis -- same genus

    # ---- Escherichia coli (4 source variants LOCKED) ----
    "ecoli_keio":          "escherichia",
    "ecoli_goodall":       "escherichia",
    "ecoli_gerdes":        "escherichia",
    "ecoli_BW25113_tradis":"escherichia",

    # ---- Bacillus subtilis (2 source variants) ----
    "bsub":            "bacillus",
    "bsub_kobayashi":  "bacillus",

    # ---- Staphylococcus aureus (LOCKED across all variants) ----
    "saur_studies":           "staphylococcus",
    "saur_studies_n":         "staphylococcus",
    "saur_NCTC8325_biotradis":"staphylococcus",
    "saur_n315":              "staphylococcus",
    "saur_nctc":              "staphylococcus",
    "saur_usa300":            "staphylococcus",

    # ---- Pseudomonas aeruginosa (3 datasets) ----
    "paer_1030": "pseudomonas",
    "paer_1036": "pseudomonas",
    "paer_1051": "pseudomonas",

    # ---- Vibrio cholerae ----
    "vcho_n16961": "vibrio",
    "vcho_c6706":  "vibrio",

    # ---- Haemophilus influenzae ----
    "hinf": "haemophilus",
    # ---- Helicobacter pylori ----
    "hpyl": "helicobacter",
    # ---- Acinetobacter baumannii ----
    "abau_1043": "acinetobacter",
    "abau_1044": "acinetobacter",
    # ---- Francisella tularensis ----
    "ftul": "francisella",
    # ---- Caulobacter crescentus ----
    "ccre": "caulobacter",
    # ---- Porphyromonas gingivalis ----
    "pgin": "porphyromonas",
    # ---- Bacteroides thetaiotaomicron ----
    "bthe": "bacteroides",

    # ---- Burkholderia (genus-wide, includes cenocepacia + thailandensis + BERIL Burk376) ----
    "bcen_J2315": "burkholderia",
    "bcen_K56_2": "burkholderia",
    "btha":       "burkholderia",

    # ---- Ralstonia eutropha (our 2 m-jahn variants) ----
    "reut":      "ralstonia",
    "reut_full": "ralstonia",

    # ---- Aeromonas (new genus, single source) ----
    "aeromonas": "aeromonas",
}


# BERIL Fitness Browser orgIds -> genus.
# Source: kbaseincubator/BERIL-research-observatory's extended_tractable_organisms.tsv
# These are environmental / soil / water bacteria with extensive Tn-seq.
BERIL_ORG_TO_CLADE: dict[str, str] = {
    # ---- Pseudomonas (genus-wide -- 9 strains across multiple species) ----
    "Putida":               "pseudomonas",   # P. putida KT2440
    "pseudo1_N1B4":         "pseudomonas",   # P. fluorescens FW300-N1B4
    "pseudo3_N2E3":         "pseudomonas",   # P. fluorescens FW300-N2E3
    "pseudo5_N2C3_1":       "pseudomonas",   # P. fluorescens FW300-N2C3
    "pseudo6_N2E2":         "pseudomonas",   # P. fluorescens FW300-N2E2
    "pseudo13_GW456_L13":   "pseudomonas",   # P. simiae
    "SyringaeB728a":        "pseudomonas",   # P. syringae B728a
    "SyringaeB728a_mexBdelta": "pseudomonas",  # same with knockout
    "WCS417":               "pseudomonas",   # P. fluorescens WCS417
    "psRCH2":               "pseudomonas",   # P. stutzeri RCH2

    # ---- Ralstonia / Cupriavidus (lock with our reut entries) ----
    "RalstoniaBSBF1503":    "ralstonia",
    "RalstoniaUW163":       "ralstonia",
    "RalstoniaGMI1000":     "ralstonia",
    "RalstoniaPSI07":       "ralstonia",
    "Cup4G11":              "ralstonia",     # Cupriavidus 4G11 -- same family
    # Note: BERIL also has a Cupriavidus separate. Group with Ralstonia (Burkholderiaceae).

    # ---- Shewanella ----
    "MR1":   "shewanella",   # S. oneidensis MR-1
    "ANA3":  "shewanella",   # S. oneidensis ANA-3
    "SB2B":  "shewanella",   # S. amazonensis SB2B

    # ---- Dickeya ----
    "Dda3937":   "dickeya",   # D. dadantii 3937
    "Ddia6719":  "dickeya",   # D. dianthicola 6719
    "DdiaME23":  "dickeya",   # D. dianthicola ME23

    # ---- Burkholderia (lock with our bcen entries) ----
    "Burk376":   "burkholderia",   # Burkholderia phytofirmans PsJN

    # ---- Klebsiella ----
    "Koxy":      "klebsiella",     # K. oxytoca

    # ---- Sinorhizobium ----
    "Smeli":     "sinorhizobium",  # S. meliloti

    # ---- E. coli BERIL (lock with our other E. coli) ----
    "Keio":      "escherichia",    # E. coli BW25113 Keio

    # ---- Acidovorax ----
    "acidovorax_3H11": "acidovorax",

    # ---- Azotobacter ----
    "azobra":    "azotobacter",    # A. vinelandii

    # ---- Marinobacter ----
    "Marino":    "marinobacter",

    # ---- Phaeobacter ----
    "Phaeo":     "phaeobacter",

    # ---- Dinoroseobacter ----
    "Dino":      "dinoroseobacter",

    # ---- Dyella ----
    "Dyella79":  "dyella",

    # ---- Pseudovibrio ----
    "PV4":       "pseudovibrio",

    # ---- Pedobacter ----
    "Pedo557":   "pedobacter",

    # ---- "BFirm" -- Burkholderia thailandensis E264 in BERIL nomenclature ----
    "BFirm":     "burkholderia",

    # ---- Herbaspirillum ----
    "HerbieS":   "herbaspirillum",

    # ---- BERIL ADP1 strains (adp1 triple essentiality dataset) ----
    # adp1_triple feeds rows where strain is the BERIL orgId; route by genus.
    # All ADP1 strains are environmental gram-neg bacteria already covered above.

    # ---- Newly mapped from the 210k Drive run ----
    "Btheta":  "bacteroides",       # Bacteroides thetaiotaomicron VPI-5482
    "Caulo":   "caulobacter",       # Caulobacter crescentus NA1000
    "DvH":     "desulfovibrio",     # Desulfovibrio vulgaris Hildenborough
    "Kang":    "kangiella",         # Kangiella aquimarina
    "Magneto": "magnetospirillum",  # Magnetospirillum magneticum AMB-1
    "SynE":    "synechococcus",     # Synechococcus elongatus PCC 7942 (cyanobacterium = bacteria)

    # Uncertain identity -- treat as singleton clades to be safe.
    # If you identify them later (FB website has the species name) move
    # them into a proper genus row above.
    "Cola":  "beril_cola_unknown",
    "Korea": "beril_korea_unknown",
    "Miya":  "beril_miya_unknown",
    "PS":    "beril_PS_unknown",
    "Ponti": "beril_ponti_unknown",

    # ---- ARCHAEA -- explicitly flagged for exclusion from bacterial training ----
    # User directive: "search only bacterial for now". Methanococcus is in DEG/FB
    # but it's archaeal. Keep the clade label visible so anyone can filter on
    # 'archaea_' prefix to drop these ~3239 rows before training.
    "Methanococcus_JJ": "archaea_methanococcus",
    "Methanococcus_S2": "archaea_methanococcus",
}


def is_bacterial_clade(clade: str) -> bool:
    """Filter helper for downstream consumers.

    User directive: "search only bacterial for now". Anything with an
    'archaea_' prefix is held out from training. Unmapped clades are
    treated as bacterial by default (no false negatives -- they show
    up as warnings in the run output and the user adds them explicitly).
    """
    return not clade.startswith('archaea_')


def organism_to_clade(org: str) -> str:
    """Resolve a labels.csv 'organism' field to a clade ID."""
    if org in ORGANISM_TO_CLADE_MANUAL:
        return ORGANISM_TO_CLADE_MANUAL[org]
    if org.startswith("beril_"):
        org_id = org[len("beril_"):]
        if org_id in BERIL_ORG_TO_CLADE:
            return BERIL_ORG_TO_CLADE[org_id]
        # Heuristic fallback for unrecognized BERIL keys
        if "Ralstonia" in org_id:                    return "ralstonia"
        if "pseudo" in org_id.lower() or "Putida" in org_id or "Syring" in org_id or "WCS" in org_id:
            return "pseudomonas"
        if org_id.lower().startswith("ddi") or "Dickeya" in org_id or org_id.startswith("Dda"):
            return "dickeya"
        if "MR" in org_id or "ANA" in org_id or "SB2" in org_id:
            return "shewanella"
        if "Burk" in org_id or "BFirm" in org_id:
            return "burkholderia"
        return f"beril_unknown:{org_id}"
    return f"unmapped:{org}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed",    type=int, default=0)
    args = p.parse_args()

    # ---- Count per-organism rows ----
    org_counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    with open(LABELS_CSV) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            org = r["organism"]
            ess = int(r["essential"])
            n, ne = org_counts[org]
            org_counts[org] = (n + 1, ne + ess)
    print(f"Loaded {sum(n for n,_ in org_counts.values())} labels "
          f"across {len(org_counts)} organism keys")

    # ---- Assign clades ----
    org_to_clade = {org: organism_to_clade(org) for org in org_counts}
    # Sanity: list any unmapped
    unmapped = sorted({c for c in org_to_clade.values() if c.startswith("unmapped:") or c.startswith("beril_unknown:")})
    if unmapped:
        print(f"\n*** WARNING: {len(unmapped)} organisms unmapped, add to MANUAL or BERIL dicts:")
        for u in unmapped:
            print(f"    {u}")

    # ---- Aggregate to clade-level ----
    clade_rows: dict[str, dict] = defaultdict(lambda: {"n_genes": 0, "n_essential": 0, "organisms": []})
    for org, (n, ne) in org_counts.items():
        c = org_to_clade[org]
        clade_rows[c]["n_genes"]     += n
        clade_rows[c]["n_essential"] += ne
        clade_rows[c]["organisms"].append(org)

    # ---- Stratified K-fold by clade ----
    # Sort by n_genes descending; assign each clade to the fold with the
    # currently-smallest total. Simple greedy = good balance.
    clades_sorted = sorted(clade_rows.items(), key=lambda kv: -kv[1]["n_genes"])
    fold_load = [0] * args.n_folds
    fold_of_clade: dict[str, int] = {}
    for clade, info in clades_sorted:
        f = fold_load.index(min(fold_load))
        fold_of_clade[clade] = f
        fold_load[f] += info["n_genes"]

    # ---- Write per-organism splits ----
    rows_out = []
    for org in sorted(org_counts):
        c = org_to_clade[org]
        n, ne = org_counts[org]
        rows_out.append({
            "organism":     org,
            "clade":        c,
            "fold":         fold_of_clade[c],
            "n_genes":      n,
            "n_essential":  ne,
            "ess_rate":     round(ne / max(n, 1), 3),
        })
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"\nwrote {OUT_CSV} ({len(rows_out)} organisms)")

    # ---- Write per-clade summary + fold assignment ----
    inv_rows = []
    for clade, info in sorted(clade_rows.items(), key=lambda kv: -kv[1]["n_genes"]):
        inv_rows.append({
            "clade":       clade,
            "fold":        fold_of_clade[clade],
            "n_organisms": len(info["organisms"]),
            "organisms":   ";".join(sorted(info["organisms"])),
            "n_genes":     info["n_genes"],
            "n_essential": info["n_essential"],
            "ess_rate":    round(info["n_essential"] / max(info["n_genes"], 1), 3),
        })
    with open(INV_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inv_rows[0].keys()))
        w.writeheader()
        for r in inv_rows:
            w.writerow(r)
    print(f"wrote {INV_CSV}\n")

    # ---- Headline summary ----
    print(f"=== Clade inventory ({len(clade_rows)} unique clades) ===")
    print(f"{'clade':<25s}  {'fold':>4s}  {'#orgs':>5s}  {'#genes':>7s}  {'#ess':>6s}  {'rate':>5s}")
    print("-" * 75)
    for r in inv_rows:
        print(f"  {r['clade']:<23s}  {r['fold']:>4d}  {r['n_organisms']:>5d}  "
              f"{r['n_genes']:>7d}  {r['n_essential']:>6d}  {r['ess_rate']:>5.2f}")

    print(f"\n=== Fold balance (n_genes per fold) ===")
    for i, load in enumerate(fold_load):
        clades_in = [c for c, f in fold_of_clade.items() if f == i]
        print(f"  fold {i}: {load:>7d} genes  ({len(clades_in)} clades): {', '.join(sorted(clades_in))}")

    # ---- Bacteria vs Archaea breakdown ----
    bact_genes = sum(info['n_genes'] for c, info in clade_rows.items()
                       if is_bacterial_clade(c))
    arch_genes = sum(info['n_genes'] for c, info in clade_rows.items()
                       if not is_bacterial_clade(c))
    bact_ess = sum(info['n_essential'] for c, info in clade_rows.items()
                     if is_bacterial_clade(c))
    arch_ess = sum(info['n_essential'] for c, info in clade_rows.items()
                     if not is_bacterial_clade(c))
    print(f"\n=== Bacteria vs Archaea (user directive: bacteria only) ===")
    print(f"  bacterial clades:  {sum(1 for c in clade_rows if is_bacterial_clade(c)):>3d}  "
          f"({bact_genes:>7d} genes, {bact_ess:>6d} essentials)")
    print(f"  archaeal clades:   {sum(1 for c in clade_rows if not is_bacterial_clade(c)):>3d}  "
          f"({arch_genes:>7d} genes, {arch_ess:>6d} essentials)")
    print(f"  --> downstream training should filter on is_bacterial_clade()")

    return 0


if __name__ == "__main__":
    sys.exit(main())
