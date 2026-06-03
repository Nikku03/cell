"""Per-gene orthology / no-backup feature builder, FOLD-AWARE.

Per project plan Part 3.1, BUT with the leakage fix:

  family_frac_essential computed globally is leaky: if you compute it
  using ALL 48 BERIL organisms, and then train cross-fold, the held-out
  fold's organisms contributed to the feature value. Any LOO-CV MCC
  computed against that feature is inflated.

  Fix: compute family_frac_essential PER FOLD. For fold k, exclude
  every BERIL organism that's in fold k from the numerator AND
  denominator, then write a separate column. Downstream code picks the
  column matching the held-out fold.

Features written (one row per BERIL gene):

  n_paralogs_in_genome           : OG co-members in same genome minus self
  family_size_total              : OG members across all organisms
  family_n_organisms             : phylogenetic breadth
  is_orphan                      : (n_paralogs == 0) AND (only 1 org has family)
  family_frac_essential_global   : leaky version -- for diagnostics ONLY
  family_frac_essential_fold0..4 : leakage-free per fold

For each row, the LEGITIMATE feature value is the column matching the
row's own fold:
    row in fold k  ->  use family_frac_essential_fold{k}
because that's the version where the row's clade was excluded from the
prior.

Inputs:
  - BERIL/projects/essential_genome/data/all_ortholog_groups.csv
  - BERIL/projects/essential_genome/data/essential_families.tsv
  - memory_bank/data/multiorg_essentiality/clade_splits.csv

Output:
  memory_bank/data/multiorg_essentiality/orthology_features.csv

CLI:
  python scripts/build_orthology_features.py \
      --beril-dir /content/drive/MyDrive/cell_count_dynamics/multiorg/repos/kbaseincubator__BERIL-research-observatory \
      --splits-csv memory_bank/data/multiorg_essentiality/clade_splits.csv
"""
from __future__ import annotations
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
DEFAULT_BERIL_DIR = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "_inventory_cache" / "BERIL"
OUT_CSV    = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "orthology_features.csv"
LABELS_CSV = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "labels.csv"
SPLITS_CSV = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "clade_splits.csv"


def load_ortholog_groups(og_csv: Path) -> tuple[dict, dict, dict]:
    """Returns:
      gene_to_og:  {(orgId, locusId): OG_id}
      og_members:  {OG_id: [(orgId, locusId), ...]}
      og_to_orgs:  {OG_id: set(orgId, ...)}
    """
    gene_to_og: dict[tuple[str, str], str] = {}
    og_members: dict[str, list[tuple[str, str]]] = defaultdict(list)
    og_to_orgs: dict[str, set[str]] = defaultdict(set)

    with open(og_csv) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            og  = r["OG_id"]
            org = r["orgId"]
            loc = r["locusId"]
            gene_to_og[(org, loc)] = og
            og_members[og].append((org, loc))
            og_to_orgs[og].add(org)
    return gene_to_og, og_members, og_to_orgs


def load_family_essentials(fam_tsv: Path) -> tuple[dict[str, float], dict[str, set[str]]]:
    """Returns
        og_frac_global:  {OG_id: global frac_essential} (LEAKY -- for diagnostics)
        og_essential_orgs: {OG_id: set of essential orgIds}
                          (used to compute fold-aware fracs)
    """
    og_frac_global: dict[str, float] = {}
    og_essential_orgs: dict[str, set[str]] = {}
    with open(fam_tsv) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            og = r.get("OG_id")
            if not og:
                continue
            try:
                og_frac_global[og] = float(r.get("frac_essential", "nan"))
            except (ValueError, TypeError):
                og_frac_global[og] = float("nan")
            ess_str = r.get("essential_organisms", "") or ""
            og_essential_orgs[og] = set(s.strip() for s in ess_str.split(";") if s.strip())
    return og_frac_global, og_essential_orgs


def load_splits(splits_csv: Path) -> tuple[int, dict[int, set[str]]]:
    """Returns (n_folds, {fold_idx: set of BERIL orgIds in that fold}).

    Strips the 'beril_' prefix to get the raw BERIL orgId for matching.
    Non-BERIL organisms are silently ignored (no OG mapping to update).
    """
    fold_to_beril_orgs: dict[int, set[str]] = {}
    with open(splits_csv) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            org    = r["organism"]
            fold   = int(r["fold"])
            if not org.startswith("beril_"):
                continue
            orgid = org[len("beril_"):]
            fold_to_beril_orgs.setdefault(fold, set()).add(orgid)
    n_folds = max(fold_to_beril_orgs.keys()) + 1 if fold_to_beril_orgs else 0
    return n_folds, fold_to_beril_orgs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--beril-dir", type=Path, default=DEFAULT_BERIL_DIR,
                   help="Path to the BERIL clone directory")
    p.add_argument("--splits-csv", type=Path, default=SPLITS_CSV,
                   help="Clade fold splits CSV (required for fold-aware fracs)")
    p.add_argument("--out", type=Path, default=OUT_CSV)
    args = p.parse_args()

    og_csv  = args.beril_dir / "projects" / "essential_genome" / "data" / "all_ortholog_groups.csv"
    fam_tsv = args.beril_dir / "projects" / "essential_genome" / "data" / "essential_families.tsv"
    if not og_csv.exists():
        print(f"ERROR: {og_csv} not found. Pass --beril-dir pointing at the "
              f"cloned BERIL repo (e.g. on Drive in Colab).", file=sys.stderr)
        return 1
    if not fam_tsv.exists():
        print(f"WARNING: {fam_tsv} not found; family_frac_essential will be "
              f"NaN for all rows.", file=sys.stderr)

    print(f"Loading {og_csv} ...")
    gene_to_og, og_members, og_to_orgs = load_ortholog_groups(og_csv)
    print(f"  {len(gene_to_og)} genes in {len(og_members)} ortholog groups "
          f"across {len({org for og_set in og_to_orgs.values() for org in og_set})} organisms")

    og_frac_global: dict[str, float] = {}
    og_ess_orgs:    dict[str, set[str]] = {}
    if fam_tsv.exists():
        print(f"Loading {fam_tsv} ...")
        og_frac_global, og_ess_orgs = load_family_essentials(fam_tsv)
        print(f"  family-essentiality calls for {len(og_frac_global)} OGs")

    # ---- Load clade splits + compute fold-aware family fracs ----
    if not args.splits_csv.exists():
        print(f"ERROR: {args.splits_csv} not found. Run "
              f"scripts/build_clade_splits.py first.", file=sys.stderr)
        return 1
    print(f"Loading {args.splits_csv} for fold-aware computation ...")
    n_folds, fold_to_orgs = load_splits(args.splits_csv)
    print(f"  {n_folds} folds across {sum(len(v) for v in fold_to_orgs.values())} BERIL orgs")

    # Precompute fold-aware family fracs: og_fold_frac[og][k] = excluded-fold value
    print(f"Computing fold-aware family_frac_essential ...")
    og_fold_frac: dict[str, list[float]] = {}
    for og, orgs_with_og in og_to_orgs.items():
        ess_orgs_for_og = og_ess_orgs.get(og, set())
        per_fold = []
        for k in range(n_folds):
            excluded = fold_to_orgs.get(k, set())
            orgs_remaining = orgs_with_og - excluded
            ess_remaining  = ess_orgs_for_og - excluded
            denom = len(orgs_remaining)
            per_fold.append(len(ess_remaining) / denom if denom > 0 else float("nan"))
        og_fold_frac[og] = per_fold

    # ---- Per-gene paralog count: count co-genome members of same OG ----
    print("Computing per-gene features ...")
    paralogs_per_gene: dict[tuple[str, str], int] = {}
    for og, members in og_members.items():
        # Group by orgId
        by_org: dict[str, list[str]] = defaultdict(list)
        for org, loc in members:
            by_org[org].append(loc)
        for org, locs in by_org.items():
            n_in_genome = len(locs)
            for loc in locs:
                paralogs_per_gene[(org, loc)] = n_in_genome - 1  # exclude self

    # ---- Per-organism -> fold lookup for the "use this column" annotation ----
    org_to_fold: dict[str, int] = {}
    for k, orgs in fold_to_orgs.items():
        for o in orgs:
            org_to_fold[o] = k

    # ---- Assemble feature rows ----
    rows = []
    for (org, loc), og in gene_to_og.items():
        n_paralogs       = paralogs_per_gene[(org, loc)]
        family_size      = len(og_members[og])
        family_n_orgs    = len(og_to_orgs[og])
        is_orphan        = 1 if (n_paralogs == 0 and family_n_orgs <= 1) else 0
        ff_global        = og_frac_global.get(og, float("nan"))
        ff_folds         = og_fold_frac.get(og, [float("nan")] * n_folds)
        own_fold         = org_to_fold.get(org, -1)
        # The legitimate (leak-free) feature value for this row.
        if own_fold >= 0 and own_fold < len(ff_folds):
            ff_leakfree = ff_folds[own_fold]
        else:
            ff_leakfree = float("nan")
        row = {
            "organism":              f"beril_{org}",
            "locus_tag":             loc,
            "og_id":                 og,
            "own_fold":              own_fold,
            "n_paralogs_in_genome":  n_paralogs,
            "family_size_total":     family_size,
            "family_n_organisms":    family_n_orgs,
            "is_orphan":             is_orphan,
            "family_frac_essential_global":    ff_global,
            "family_frac_essential_leakfree":  ff_leakfree,
        }
        for k in range(n_folds):
            row[f"family_frac_essential_fold{k}"] = ff_folds[k]
        rows.append(row)
    print(f"  built {len(rows)} per-gene feature rows")

    # ---- Stats ----
    n_orphans      = sum(r["is_orphan"] for r in rows)
    n_no_paralog   = sum(1 for r in rows if r["n_paralogs_in_genome"] == 0)
    n_singletons   = sum(1 for r in rows if r["family_size_total"] == 1)
    print(f"\n  orphans (no genome paralog AND no other org has family): {n_orphans} "
          f"({100*n_orphans/len(rows):.1f}%)")
    print(f"  genes with zero paralogs in genome (broader 'no-backup'): {n_no_paralog} "
          f"({100*n_no_paralog/len(rows):.1f}%)")
    print(f"  family singletons (one gene total across all 33 orgs):    {n_singletons} "
          f"({100*n_singletons/len(rows):.1f}%)")

    # ---- Write CSV ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["organism","locus_tag","og_id","own_fold",
                   "n_paralogs_in_genome","family_size_total","family_n_organisms",
                   "is_orphan",
                   "family_frac_essential_global",
                   "family_frac_essential_leakfree"]
    fieldnames += [f"family_frac_essential_fold{k}" for k in range(n_folds)]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.out}")

    # ---- Diagnostic: how much does the leakage fix shift the feature? ----
    print(f"\n=== leakage delta ===")
    deltas = []
    for r in rows:
        g = r["family_frac_essential_global"]
        lf = r["family_frac_essential_leakfree"]
        if isinstance(g, float) and isinstance(lf, float) and g == g and lf == lf:
            deltas.append(abs(g - lf))
    if deltas:
        import statistics
        print(f"  rows compared:        {len(deltas)}")
        print(f"  mean |global - leakfree|: {statistics.mean(deltas):.4f}")
        print(f"  max  |global - leakfree|: {max(deltas):.4f}")
        large_shift = sum(1 for d in deltas if d > 0.1)
        print(f"  rows shifted by >0.1:  {large_shift}  ({100*large_shift/len(deltas):.1f}%)")
        print(f"  (small shifts = global value was already mostly determined by other clades;")
        print(f"   large shifts = the row's clade was a big contributor and exclusion matters)")

    # ---- Quick join-rate check vs labels.csv ----
    if LABELS_CSV.exists():
        labeled_keys = set()
        with open(LABELS_CSV) as f:
            for r in csv.DictReader(f):
                labeled_keys.add((r["organism"], r["locus_tag"]))
        feature_keys = {(r["organism"], r["locus_tag"]) for r in rows}
        overlap = labeled_keys & feature_keys
        beril_labeled = sum(1 for k in labeled_keys if k[0].startswith("beril_"))
        print(f"\n  Coverage check vs labels.csv:")
        print(f"    labels.csv total:                {len(labeled_keys)}")
        print(f"    labels.csv BERIL rows:           {beril_labeled}")
        print(f"    feature rows joinable to labels: {len(overlap)}")
        print(f"    (non-BERIL labels.csv rows will need self-BLAST for n_paralogs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
