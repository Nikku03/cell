"""OG-level breakdown of FP/FN errors.

Why this script vs analyze_wrong_gene_functions.py:
  - labels.csv carries gene_name only for non-BERIL sources
    (syn3A, supplementaries). BERIL rows have no gene_name.
  - So we can't categorize BERIL FPs/FNs by gene-name regex.
  - Instead, join (organism, locus_tag) -> OG_id via BERIL's
    all_ortholog_groups.csv, then look at essential_families.tsv
    for OG-level metadata (frac_essential, n_total, plus any
    representative gene_name or annotation column the TSV carries).

Outputs:
  reliability_og_breakdown.json
    - top_FP_OGs:  OGs ranked by FP count, with frac_essential and any
                     annotation columns from essential_families.tsv
    - top_FN_OGs:  same for FN
    - top_FP_universal_OGs:  FPs restricted to OGs in 31+ orgs (the
                                 broadly-conserved-but-overcalled set)
    - fp_fn_by_og_size:        OGs binned by org breadth

Run:
  python scripts/analyze_wrong_ogs.py --beril-dir <path-to-BERIL>
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality"
PRED_CSV  = DATA_DIR / "loo_predictions.csv"
OUT_JSON  = DATA_DIR / "reliability_og_breakdown.json"
DEFAULT_BERIL = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "_inventory_cache" / "BERIL"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-csv",  type=Path, default=PRED_CSV)
    p.add_argument("--beril-dir", type=Path, default=DEFAULT_BERIL,
                   help="Root of the BERIL repo clone")
    p.add_argument("--out",       type=Path, default=OUT_JSON)
    p.add_argument("--top-n",     type=int,   default=30)
    args = p.parse_args()

    try:
        import pandas as pd
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    og_csv  = args.beril_dir / "projects" / "essential_genome" / "data" / "all_ortholog_groups.csv"
    fam_tsv = args.beril_dir / "projects" / "essential_genome" / "data" / "essential_families.tsv"
    for f in (og_csv, fam_tsv, args.pred_csv):
        if not f.exists():
            print(f"ERROR: missing {f}", file=sys.stderr); return 1

    # ---- (org, locus) -> OG_id ----
    print(f"Loading {og_csv} ...")
    gene_to_og: dict[tuple[str,str], str] = {}
    with open(og_csv) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            gene_to_og[(r["orgId"], r["locusId"])] = r["OG_id"]
    print(f"  {len(gene_to_og)} (org, locus) -> OG mappings")

    # ---- OG metadata ----
    print(f"Loading {fam_tsv} ...")
    og_meta: dict[str, dict] = {}
    fam_cols: list[str] = []
    with open(fam_tsv) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        fam_cols = list(rdr.fieldnames or [])
        for r in rdr:
            og = r.get("OG_id")
            if og: og_meta[og] = r
    print(f"  essential_families.tsv columns: {fam_cols}")
    print(f"  {len(og_meta)} OGs with metadata")

    # Detect annotation column heuristically
    annot_cols = [c for c in fam_cols
                  if c.lower() in ("gene_name","gene","name","symbol",
                                     "description","desc","product",
                                     "function","annotation","representative",
                                     "rep_gene","rep_locus")]
    print(f"  annotation column(s) detected: {annot_cols or '(none)'}")

    # ---- Load predictions ----
    pred = pd.read_csv(args.pred_csv)
    print(f"  predictions: {len(pred)} rows")

    # Map predictions org name (beril_<orgId>) back to BERIL orgId
    def to_orgid(o: str) -> str:
        return o[6:] if isinstance(o, str) and o.startswith("beril_") else o
    pred["orgId"] = pred.organism.apply(to_orgid)
    pred["og_id"] = [gene_to_og.get((o, l)) for o, l in
                      zip(pred.orgId, pred.locus_tag)]
    n_mapped = int(pred.og_id.notna().sum())
    print(f"  mapped to OG: {n_mapped} ({100*n_mapped/len(pred):.1f}%)")
    pred = pred[pred.og_id.notna()].reset_index(drop=True)

    pred["error_class"] = pred.apply(
        lambda r: "TP" if (r.y_pred==1 and r.y_true==1) else
                  "FN" if (r.y_pred==0 and r.y_true==1) else
                  "FP" if (r.y_pred==1 and r.y_true==0) else "TN", axis=1)

    # ---- Top FP / FN OGs ----
    def og_summary(sub: "pd.DataFrame") -> list[dict]:
        g = sub.groupby("og_id").size().reset_index(name="n_wrong")
        g = g.sort_values("n_wrong", ascending=False).head(args.top_n)
        rows = []
        for _, r in g.iterrows():
            og = r.og_id
            meta = og_meta.get(og, {})
            entry = {"og_id": og, "n_wrong": int(r.n_wrong)}
            for c in fam_cols:
                if c == "OG_id": continue
                entry[c] = meta.get(c)
            rows.append(entry)
        return rows

    fp_df = pred[pred.error_class=="FP"]
    fn_df = pred[pred.error_class=="FN"]
    print(f"\n=== Top {args.top_n} OGs by FP count ===")
    top_fp = og_summary(fp_df)
    for r in top_fp[:20]:
        annot = " | ".join(f"{c}={r.get(c)}" for c in annot_cols) if annot_cols else ""
        print(f"  {r['og_id']:<14s}  n_FP={r['n_wrong']:>5d}  "
              f"frac_ess={r.get('frac_essential') or '?':>5}  {annot[:80]}")

    print(f"\n=== Top {args.top_n} OGs by FN count ===")
    top_fn = og_summary(fn_df)
    for r in top_fn[:20]:
        annot = " | ".join(f"{c}={r.get(c)}" for c in annot_cols) if annot_cols else ""
        print(f"  {r['og_id']:<14s}  n_FN={r['n_wrong']:>5d}  "
              f"frac_ess={r.get('frac_essential') or '?':>5}  {annot[:80]}")

    # ---- FPs restricted to broadly-conserved OGs (universal band) ----
    fp_universal = fp_df[fp_df.family_n_organisms >= 31]
    print(f"\n=== Top FP OGs restricted to 31+ orgs (universal band) ===")
    print(f"  total FPs in this band: {len(fp_universal)}")
    top_fp_uni = og_summary(fp_universal)
    for r in top_fp_uni[:20]:
        annot = " | ".join(f"{c}={r.get(c)}" for c in annot_cols) if annot_cols else ""
        print(f"  {r['og_id']:<14s}  n_FP={r['n_wrong']:>5d}  "
              f"frac_ess={r.get('frac_essential') or '?':>5}  {annot[:80]}")

    # ---- Summarise: how many FPs are explained by the top-K OGs? ----
    cumulative = 0
    total_fp = len(fp_df)
    fp_by_og = fp_df.groupby("og_id").size().sort_values(ascending=False)
    print(f"\n=== FP concentration by OG ===")
    print(f"  total FP: {total_fp}  unique OGs with FPs: {len(fp_by_og)}")
    for K in [10, 50, 100, 500, 1000]:
        if K > len(fp_by_og): break
        cov = int(fp_by_og.head(K).sum())
        print(f"  top {K:>5d} OGs cover {cov:>6d} FPs  ({100*cov/total_fp:.1f}%)")

    total_fn = len(fn_df)
    fn_by_og = fn_df.groupby("og_id").size().sort_values(ascending=False)
    print(f"\n=== FN concentration by OG ===")
    print(f"  total FN: {total_fn}  unique OGs with FNs: {len(fn_by_og)}")
    for K in [10, 50, 100, 500, 1000]:
        if K > len(fn_by_og): break
        cov = int(fn_by_og.head(K).sum())
        print(f"  top {K:>5d} OGs cover {cov:>6d} FNs  ({100*cov/total_fn:.1f}%)")

    # ---- Save ----
    out = {
        "essential_families_columns": fam_cols,
        "annotation_columns_detected": annot_cols,
        "n_predictions_mapped_to_og": n_mapped,
        "top_FP_OGs":            top_fp,
        "top_FN_OGs":            top_fn,
        "top_FP_universal_OGs":  top_fp_uni,
        "fp_total": int(total_fp),
        "fn_total": int(total_fn),
        "fp_unique_ogs": int(len(fp_by_og)),
        "fn_unique_ogs": int(len(fn_by_og)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
