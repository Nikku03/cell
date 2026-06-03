"""Per-gene orthology / no-backup feature builder.

THE strongest independent feature per the project plan (Part 3.1):

  "Irreplaceability: no backup (no paralog / isozyme / alt-route).
   Strongest single mechanism; only feature that sees orphan essentials."

Computes from `BERIL/all_ortholog_groups.csv` (OG_id, orgId, locusId):

  Per-gene features:
    n_paralogs_in_genome    # other genes in same OG within same genome
    family_size_total       # total members of OG across all organisms
    family_n_organisms      # phylogenetic breadth (count of distinct orgIds)
    is_orphan               # 1 if no paralogs in genome AND family <=1 org
    family_frac_essential   # from essential_families.tsv -- cross-org prior

  Outputs:
    memory_bank/data/multiorg_essentiality/orthology_features.csv

Coverage: BERIL covers ~33 environmental bacteria. Non-BERIL organisms
(syn3A, M.gen, M.pne, M.tub, E. coli/B. subtilis from non-FB sources)
get NaN for these features and would need a separate self-BLAST pass to
compute their own paralog counts.

Run:
  # Pointing at the Drive-cloned BERIL repo (Colab path)
  python scripts/build_orthology_features.py \
      --beril-dir /content/drive/MyDrive/cell_count_dynamics/multiorg/repos/kbaseincubator__BERIL-research-observatory

  # Or omit --beril-dir to use the default REPO-relative path.
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


def load_family_essentials(fam_tsv: Path) -> dict[str, float]:
    """Returns {OG_id: frac_essential} from essential_families.tsv."""
    out: dict[str, float] = {}
    with open(fam_tsv) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            og = r.get("OG_id")
            if not og:
                continue
            try:
                out[og] = float(r.get("frac_essential", "nan"))
            except (ValueError, TypeError):
                continue
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--beril-dir", type=Path, default=DEFAULT_BERIL_DIR,
                   help="Path to the BERIL clone directory")
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

    fam_ess: dict[str, float] = {}
    if fam_tsv.exists():
        print(f"Loading {fam_tsv} ...")
        fam_ess = load_family_essentials(fam_tsv)
        print(f"  family-essentiality calls for {len(fam_ess)} OGs")

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

    # ---- Assemble feature rows ----
    rows = []
    for (org, loc), og in gene_to_og.items():
        n_paralogs       = paralogs_per_gene[(org, loc)]
        family_size      = len(og_members[og])
        family_n_orgs    = len(og_to_orgs[og])
        is_orphan        = 1 if (n_paralogs == 0 and family_n_orgs <= 1) else 0
        ff_ess           = fam_ess.get(og, '')
        rows.append({
            "organism":              f"beril_{org}",
            "locus_tag":             loc,
            "og_id":                 og,
            "n_paralogs_in_genome":  n_paralogs,
            "family_size_total":     family_size,
            "family_n_organisms":    family_n_orgs,
            "is_orphan":             is_orphan,
            "family_frac_essential": ff_ess,
        })
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
    fieldnames = ["organism","locus_tag","og_id",
                   "n_paralogs_in_genome","family_size_total","family_n_organisms",
                   "is_orphan","family_frac_essential"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.out}")

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
