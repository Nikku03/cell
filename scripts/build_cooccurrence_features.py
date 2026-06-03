"""Per-gene OG co-occurrence features (the synteny-without-GFFs feature).

Per project plan Part 3.1 (synteny / co-occurrence):
  "What's before/after this gene, and which genes are gained/lost
   together across organisms. This is your generalization lever --
   the pattern holds for organisms you've never labeled."

The strict "before/after" angle needs gene coordinates (GFFs), which
BERIL doesn't carry. The "gained/lost together" angle (phylogenetic
profile co-occurrence) is fully computable from
all_ortholog_groups.csv. That's what this script does.

For each OG_X, we compute its phylogenetic profile = set of organisms
that have X. Two OGs that share organism sets are "co-occurring" --
they show up in / disappear from genomes together, suggesting
functional coupling (operon-mates, pathway partners, or co-evolving
modules).

Per-gene features (one row per BERIL gene):
  cooccur_max_jaccard         : largest Jaccard similarity with any other OG
  cooccur_n_neighbors_50      : count of OGs with Jaccard >= 0.5
  cooccur_n_neighbors_80      : count of OGs with Jaccard >= 0.8
  cooccur_essential_neighbor_frac : frac of >=0.5-Jaccard OG neighbors that
                                    have family_frac_essential >= 0.5
                                    (= "are my co-occurring buddies essential?")

These features are NOT leaky -- they use only OG presence/absence
across organisms, not essentiality labels. No fold-aware version needed
(unlike family_frac_essential).

Inputs:
  - BERIL/projects/essential_genome/data/all_ortholog_groups.csv
  - BERIL/projects/essential_genome/data/essential_families.tsv

Output:
  memory_bank/data/multiorg_essentiality/cooccurrence_features.csv

Run:
  python scripts/build_cooccurrence_features.py \\
      --beril-dir /content/drive/MyDrive/cell_count_dynamics/multiorg/repos/kbaseincubator__BERIL-research-observatory
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BERIL_DIR = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "_inventory_cache" / "BERIL"
OUT_CSV = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "cooccurrence_features.csv"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--beril-dir", type=Path, default=DEFAULT_BERIL_DIR)
    p.add_argument("--out",       type=Path, default=OUT_CSV)
    p.add_argument("--min-orgs",  type=int,   default=3,
                   help="Skip OGs present in fewer than this many organisms "
                        "(noise floor; default 3)")
    p.add_argument("--ess-threshold", type=float, default=0.5,
                   help="An OG counts as 'essential' for neighbor scoring if "
                        "family_frac_essential >= this (default 0.5)")
    args = p.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("ERROR: pip install numpy", file=sys.stderr)
        return 1

    og_csv  = args.beril_dir / "projects" / "essential_genome" / "data" / "all_ortholog_groups.csv"
    fam_tsv = args.beril_dir / "projects" / "essential_genome" / "data" / "essential_families.tsv"
    if not og_csv.exists():
        print(f"ERROR: {og_csv} not found.", file=sys.stderr); return 1

    # ---- Load OG -> set of orgs ----
    print(f"Loading {og_csv} ...")
    og_to_orgs: dict[str, set[str]] = defaultdict(set)
    gene_to_og: dict[tuple[str, str], str] = {}
    with open(og_csv) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            og  = r["OG_id"]
            org = r["orgId"]
            loc = r["locusId"]
            og_to_orgs[og].add(org)
            gene_to_og[(org, loc)] = og
    n_genes = len(gene_to_og)
    n_orgs  = len({o for s in og_to_orgs.values() for o in s})
    print(f"  {n_genes} genes, {len(og_to_orgs)} OGs, {n_orgs} organisms")

    # ---- Load family fracs (for essential-neighbor weighting) ----
    og_frac: dict[str, float] = {}
    if fam_tsv.exists():
        with open(fam_tsv) as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for r in rdr:
                og = r.get("OG_id")
                if not og: continue
                try:
                    og_frac[og] = float(r.get("frac_essential", "nan"))
                except (ValueError, TypeError):
                    pass
    print(f"  family_frac_essential available for {len(og_frac)} OGs")

    # ---- Build a (n_OGs x n_orgs) presence matrix in numpy bit-packed ----
    # Use one bit per (OG, org) cell; aggregate set ops via numpy bitwise.
    og_list   = sorted(og_to_orgs.keys())
    og_to_idx = {og: i for i, og in enumerate(og_list)}
    org_list  = sorted({o for s in og_to_orgs.values() for o in s})
    org_to_idx = {org: i for i, org in enumerate(org_list)}
    n_OG  = len(og_list)
    n_org = len(org_list)
    print(f"\n  Building presence matrix: {n_OG} OGs x {n_org} orgs")

    # n_org <= 48 typically -- fits in uint64
    assert n_org <= 64, f"too many orgs ({n_org}) for uint64 bitmap; use np.packbits instead"
    presence = np.zeros(n_OG, dtype=np.uint64)
    for og, orgs in og_to_orgs.items():
        bits = 0
        for org in orgs:
            bits |= (1 << org_to_idx[org])
        presence[og_to_idx[og]] = bits

    # Helper: popcount via numpy
    def popcount(x: np.ndarray) -> np.ndarray:
        x = x - ((x >> 1) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + ((x >> 2) & np.uint64(0x3333333333333333))
        x = (x + (x >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return ((x * np.uint64(0x0101010101010101)) >> 56)

    og_size = popcount(presence)  # int per OG

    # ---- Filter to OGs with >= min_orgs members; index keeps same global og_to_idx ----
    qualifying = og_size >= args.min_orgs
    qual_indices = np.nonzero(qualifying)[0]
    print(f"  OGs with >= {args.min_orgs} orgs: {len(qual_indices)}  "
          f"(median size: {int(np.median(og_size[qual_indices]))})")

    # ---- Compute co-occurrence features per OG ----
    # For each OG i in qualifying set, we want max/count Jaccard with OG j (i != j)
    # Jaccard(i,j) = |i ∩ j| / |i ∪ j| = popcount(i & j) / (size_i + size_j - popcount(i & j))
    print(f"\n  Computing per-OG co-occurrence features (this is the slow part) ...")
    t0 = time.time()

    n_qual = len(qual_indices)
    qual_bits = presence[qual_indices]
    qual_size = og_size[qual_indices]
    qual_ess_flag = np.array([
        1 if og_frac.get(og_list[i], 0.0) >= args.ess_threshold else 0
        for i in qual_indices
    ], dtype=np.int32)

    per_og_max_jacc   = np.zeros(n_qual, dtype=np.float32)
    per_og_n_jacc_50  = np.zeros(n_qual, dtype=np.int32)
    per_og_n_jacc_80  = np.zeros(n_qual, dtype=np.int32)
    per_og_ess_neigh  = np.zeros(n_qual, dtype=np.float32)

    # Process in chunks to keep memory reasonable
    chunk = 1000
    for start in range(0, n_qual, chunk):
        end = min(start + chunk, n_qual)
        # Pairwise intersection sizes (chunk x n_qual)
        # presence[start:end] is (chunk, ) uint64; broadcast vs qual_bits (n_qual, )
        a = qual_bits[start:end][:, None]              # (chunk, 1)
        b = qual_bits[None, :]                         # (1, n_qual)
        inter = popcount(a & b).astype(np.float32)     # (chunk, n_qual)
        size_a = qual_size[start:end][:, None].astype(np.float32)
        size_b = qual_size[None, :].astype(np.float32)
        union = size_a + size_b - inter
        # Avoid div-by-zero
        jacc = np.where(union > 0, inter / union, 0.0).astype(np.float32)
        # Zero self-similarity
        for i in range(end - start):
            jacc[i, start + i] = 0.0
        per_og_max_jacc[start:end]  = jacc.max(axis=1)
        per_og_n_jacc_50[start:end] = (jacc >= 0.5).sum(axis=1)
        per_og_n_jacc_80[start:end] = (jacc >= 0.8).sum(axis=1)
        # Essential-neighbor fraction (Jaccard>=0.5 mask)
        neigh_mask = (jacc >= 0.5)
        n_neigh    = neigh_mask.sum(axis=1).astype(np.float32)
        n_ess_n    = (neigh_mask & qual_ess_flag[None, :].astype(bool)).sum(axis=1).astype(np.float32)
        per_og_ess_neigh[start:end] = np.where(n_neigh > 0, n_ess_n / n_neigh, 0.0)

        if (start // chunk) % 5 == 0:
            print(f"    progress: {end}/{n_qual}  wall={time.time()-t0:.1f}s")
    print(f"  done in {time.time()-t0:.1f}s")

    # ---- Per-OG -> per-gene ----
    qual_og_lookup = {qual_indices[i]: i for i in range(n_qual)}
    rows_out = []
    n_skipped = 0
    for (org, loc), og in gene_to_og.items():
        gi = og_to_idx[og]
        if gi not in qual_og_lookup:
            n_skipped += 1
            continue
        qi = qual_og_lookup[gi]
        rows_out.append({
            "organism":              f"beril_{org}",
            "locus_tag":             loc,
            "og_id":                 og,
            "cooccur_max_jaccard":          round(float(per_og_max_jacc[qi]), 4),
            "cooccur_n_neighbors_50":       int(per_og_n_jacc_50[qi]),
            "cooccur_n_neighbors_80":       int(per_og_n_jacc_80[qi]),
            "cooccur_essential_neighbor_frac": round(float(per_og_ess_neigh[qi]), 4),
        })
    print(f"\n  wrote {len(rows_out)} per-gene rows  "
           f"(skipped {n_skipped} genes from <{args.min_orgs}-org OGs)")

    # ---- Write CSV ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["organism", "locus_tag", "og_id",
                   "cooccur_max_jaccard", "cooccur_n_neighbors_50",
                   "cooccur_n_neighbors_80", "cooccur_essential_neighbor_frac"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"\nwrote {args.out}")

    # ---- Quick distribution check ----
    import statistics
    max_j_values = [r["cooccur_max_jaccard"] for r in rows_out]
    n50_values   = [r["cooccur_n_neighbors_50"] for r in rows_out]
    print(f"\n=== feature distributions (sanity) ===")
    print(f"  cooccur_max_jaccard:    median={statistics.median(max_j_values):.3f}  "
          f"mean={statistics.mean(max_j_values):.3f}  max={max(max_j_values):.3f}")
    print(f"  cooccur_n_neighbors_50: median={statistics.median(n50_values)}  "
          f"mean={statistics.mean(n50_values):.1f}  max={max(n50_values)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
