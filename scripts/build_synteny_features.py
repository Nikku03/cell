"""Synteny / gene-neighborhood features per gene.

Fix 4 of the error-mode plan: operonic / synteny-based co-essentiality.

Per gene, records:
  - prev_og_id, next_og_id           : OG of upstream/downstream
                                          neighbor (used by training
                                          script to look up per-fold
                                          family_frac of the neighbor).
  - prev_locus_tag, next_locus_tag   : locus_tag of neighbor.
  - synteny_count_prev               : count, across all 59 organisms,
                                          of (this OG, prev OG)
                                          appearing adjacent (unordered
                                          pair). High = conserved
                                          synteny block = likely operon.
  - synteny_count_next               : same for downstream pair.
  - max_synteny_count                : max of those two.
  - prev_same_og, next_same_og       : 1 if neighbor is in same OG
                                          (tandem duplicate -- this
                                          copy may be redundant even
                                          when family is essential).

train_loo_xgboost.py joins these and adds per-fold:
  - prev_family_frac, next_family_frac, max_neighbor_family_frac

Three orthogonal operon-like signals:
  1. Neighbor essentiality via family_frac of neighbor's OG
  2. Neighbor conservation via synteny_count across all organisms
  3. Tandem duplicate adjacency via prev/next_same_og

Genome-order heuristic:
  Sort locus_tags within an organism by (alpha-prefix, numeric tail).
  Correct for b-numbers (E. coli), Rv-numbers (M.tb), PP_ (Putida),
  SAOUHSC_ (S. aureus), BERIL internal numeric IDs. Approximate for
  multi-chromosome organisms (1 spurious neighbor at each chromosome
  boundary).

Leak note on synteny_count:
  Computed across ALL organisms including the held-out test fold.
  Each test fold contributes ~1-3 instances to typical pair counts
  (out of 5-30); the count is structural (no labels), so this is
  trivial distribution leak, not label leak. Same status as
  family_n_organisms which is also computed globally.
"""
from __future__ import annotations
import argparse, re, sys
from collections import Counter
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
    p.add_argument("--orphan-marker", default="ORPHAN",
                   help="OG id value treated as 'no family' (pairs "
                        "involving this value are excluded from synteny "
                        "count). Default: ORPHAN")
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

    # Pass 1: sort each organism + accumulate global pair-occurrence counter
    sorted_parts: list[tuple[str, "pd.DataFrame"]] = []
    pair_count: Counter[tuple[str, str]] = Counter()

    print("\nPass 1: per-organism sort + global pair counting")
    for org, g in orth.groupby("organism", sort=True):
        g = (g.assign(_k=g["locus_tag"].apply(_sort_key))
               .sort_values("_k")
               .drop(columns="_k")
               .reset_index(drop=True))
        sorted_parts.append((org, g))

        ogs = g["og_id"].tolist()
        n_pairs_org = 0
        for i in range(len(ogs) - 1):
            a, b = ogs[i], ogs[i + 1]
            if (pd.notna(a) and pd.notna(b) and
                a != args.orphan_marker and b != args.orphan_marker):
                key = (str(a), str(b)) if str(a) < str(b) else (str(b), str(a))
                pair_count[key] += 1
                n_pairs_org += 1
        print(f"  {org:<32s}  n={len(g):>5d}  contributed {n_pairs_org:>4d} OG pairs")

    print(f"\nGlobal pair stats: {len(pair_count):,} unique adjacent OG pairs")
    if pair_count:
        cs = sorted(pair_count.values(), reverse=True)
        n_singleton = sum(1 for c in cs if c == 1)
        print(f"  count distribution: max={cs[0]}  90th={cs[len(cs)//10]}  "
              f"median={cs[len(cs)//2]}  min={cs[-1]}")
        print(f"  singleton pairs (count==1): {n_singleton:,} "
              f"({100*n_singleton/len(pair_count):.1f}%)")
        print(f"  top 5 most-conserved adjacent OG pairs:")
        for pair, n in pair_count.most_common(5):
            print(f"    {pair[0]:<15s} <-> {pair[1]:<15s}  appears in {n} organisms")

    # Pass 2: emit per-gene synteny features
    print("\nPass 2: emit per-gene synteny features")
    parts = []
    orphan = args.orphan_marker
    for org, g in sorted_parts:
        out = g[["organism", "locus_tag", "og_id"]].copy()
        out["prev_og_id"]     = g["og_id"].shift(1)
        out["next_og_id"]     = g["og_id"].shift(-1)
        out["prev_locus_tag"] = g["locus_tag"].shift(1)
        out["next_locus_tag"] = g["locus_tag"].shift(-1)
        out["prev_same_og"]   = ((out["prev_og_id"] == out["og_id"]) &
                                  out["og_id"].notna()).astype("Int64")
        out["next_same_og"]   = ((out["next_og_id"] == out["og_id"]) &
                                  out["og_id"].notna()).astype("Int64")

        def _pair_n(neighbor_og, self_og):
            if (pd.isna(neighbor_og) or pd.isna(self_og) or
                neighbor_og == orphan or self_og == orphan):
                return 0
            a, b = str(neighbor_og), str(self_og)
            key = (a, b) if a < b else (b, a)
            return pair_count.get(key, 0)

        out["synteny_count_prev"] = [
            _pair_n(prev, sog) for prev, sog in zip(out["prev_og_id"], out["og_id"])
        ]
        out["synteny_count_next"] = [
            _pair_n(nxt, sog) for nxt, sog in zip(out["next_og_id"], out["og_id"])
        ]
        out["max_synteny_count"] = out[
            ["synteny_count_prev", "synteny_count_next"]].max(axis=1)
        parts.append(out)

    syn = pd.concat(parts, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    syn.to_csv(args.out, index=False)
    print(f"\nwrote {len(syn):,} rows to {args.out}")

    print(f"\n=== feature distributions ===")
    print(f"  prev_og_id present:        {syn.prev_og_id.notna().sum()/len(syn)*100:.1f}%")
    print(f"  next_og_id present:        {syn.next_og_id.notna().sum()/len(syn)*100:.1f}%")
    print(f"  prev_same_og == 1:         {(syn.prev_same_og == 1).sum():,}  "
          f"({100*(syn.prev_same_og==1).sum()/len(syn):.2f}%)")
    print(f"  next_same_og == 1:         {(syn.next_same_og == 1).sum():,}  "
          f"({100*(syn.next_same_og==1).sum()/len(syn):.2f}%)")
    print(f"  synteny_count_prev:        median={syn.synteny_count_prev.median():.0f}  "
          f"mean={syn.synteny_count_prev.mean():.2f}  "
          f"max={syn.synteny_count_prev.max()}")
    print(f"  synteny_count_next:        median={syn.synteny_count_next.median():.0f}  "
          f"mean={syn.synteny_count_next.mean():.2f}  "
          f"max={syn.synteny_count_next.max()}")
    print(f"  max_synteny_count:         median={syn.max_synteny_count.median():.0f}  "
          f"mean={syn.max_synteny_count.mean():.2f}  "
          f"90th={syn.max_synteny_count.quantile(0.9):.0f}  "
          f"max={syn.max_synteny_count.max()}")
    print(f"  rows w/ max_synteny>=10 (highly conserved): "
          f"{(syn.max_synteny_count >= 10).sum():,}  "
          f"({100*(syn.max_synteny_count>=10).sum()/len(syn):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
