"""Synteny / gene-neighborhood features per gene.

Fix 4 of the error-mode plan: operonic / synteny-based co-essentiality.

Per gene, records the OG identity of its immediate upstream and
downstream neighbors on the chromosome (heuristically, via
locus-tag sort within the organism). train_loo_xgboost.py then
joins these against the per-fold family_frac table to produce:

  - prev_family_frac        : mean essentiality of upstream
                                neighbor's OG (LOO across folds)
  - next_family_frac        : same for downstream neighbor
  - max_neighbor_family_frac: max of the two

This captures operon-like co-essentiality without needing an
operon predictor: if your conserved neighbors come from
families that tend to be essential, you likely are too.

Genome-order heuristic:
  Sort locus_tags within an organism by (alpha-prefix, numeric
  tail). Correct for:
    b0001, b0002       (E. coli K12 b-numbers)
    Rv0001, Rv0002     (M. tb H37Rv)
    PP_0001, PP_0002   (P. putida KT2440)
    SAOUHSC_00001..    (S. aureus NCTC8325)
    16329, 14824, ..   (BERIL internal numeric IDs - sortable)
  Approximate for everything else; cross-chromosome edges in
  multi-chromosome organisms (Burkholderia, Vibrio) appear as
  ~1 spurious neighbor at the chromosome boundary per pair.

Coverage: every gene that appears in orthology_features.csv
(~145K BERIL rows + whatever else has OG assignments).
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ORTH_CSV  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "orthology_features.csv"
OUT_CSV   = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "synteny_features.csv"

_KEY_RE = re.compile(r"^([^\d]*)(\d+)(.*)$")


def _sort_key(locus):
    s = str(locus)
    m = _KEY_RE.match(s)
    if m:
        return (m.group(1), int(m.group(2)), m.group(3))
    return (s, 0, "")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--orth-csv", type=Path, default=ORTH_CSV)
    p.add_argument("--out",      type=Path, default=OUT_CSV)
    args = p.parse_args()

    try:
        import pandas as pd
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    orth = pd.read_csv(args.orth_csv)
    needed = {"organism", "locus_tag", "og_id"}
    missing = needed - set(orth.columns)
    if missing:
        print(f"ERROR: orthology_features missing cols {missing}", file=sys.stderr)
        return 1
    print(f"Loaded {len(orth):,} rows  ({orth.organism.nunique()} organisms)")

    parts = []
    print("\nPer-organism gene-order sort:")
    for org, g in orth.groupby("organism", sort=True):
        g = g.assign(_k=g["locus_tag"].apply(_sort_key)) \
              .sort_values("_k") \
              .drop(columns="_k") \
              .reset_index(drop=True)
        out = g[["organism", "locus_tag", "og_id"]].copy()
        out["prev_og_id"]     = g["og_id"].shift(1)
        out["next_og_id"]     = g["og_id"].shift(-1)
        out["prev_locus_tag"] = g["locus_tag"].shift(1)
        out["next_locus_tag"] = g["locus_tag"].shift(-1)
        parts.append(out)
        print(f"  {org:<32s}  n={len(g):>5d}  "
              f"first={g.locus_tag.iloc[0]:<14s}  last={g.locus_tag.iloc[-1]}")

    syn = pd.concat(parts, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    syn.to_csv(args.out, index=False)
    print(f"\nwrote {len(syn):,} rows to {args.out}")
    print(f"  prev_og_id present: {syn.prev_og_id.notna().sum()/len(syn)*100:.1f}%  "
          f"(missing = first gene of each organism)")
    print(f"  next_og_id present: {syn.next_og_id.notna().sum()/len(syn)*100:.1f}%  "
          f"(missing = last gene of each organism)")

    # how informative is each neighbor's OG?
    n_known_prev = int((syn.prev_og_id.notna() &
                         (syn.prev_og_id != "ORPHAN") &
                         (syn.prev_og_id != syn.og_id)).sum())
    print(f"  prev_og_id non-orphan, distinct from self: "
          f"{n_known_prev:,} ({100*n_known_prev/len(syn):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
