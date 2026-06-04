"""Predict per-organism essentiality-model MCC from organism-level features.

Motivated by the per-clade evaluation finding: per-organism MCC
spans 0.48 .. 0.67. That spread is not explained by clade size
alone (corr = +0.28). This script asks: from the organism's own
characteristics -- before we ever run the essentiality predictor
on it -- can we predict how well that predictor will work?

If yes, we can attach calibrated confidence at deployment time:
  "For this new organism, expected MCC = 0.62 +/- 0.05" rather
  than the macro mean +/- macro std.

Pipeline:
  1. Load per_organism_loo.csv (target = per-org MCC measured)
  2. Build per-organism features:
       - clade_n_orgs                  : how many labelled relatives
       - n_genes                        : organism gene count
       - frac_orphan                    : fraction of genes in singleton OGs
       - mean / median family_n_organisms across this org's genes
       - frac_in_large_OGs              : fraction of genes whose OG has
                                            >= 30 organisms (deeply
                                            conserved core)
       - mean / median n_paralogs_in_genome
       - mean / std family_frac (LOO, leakage-free over training orgs)
       - cooc_mean_n_neighbors_80       : mean co-occurrence breadth
  3. LOO-organism regression (XGBoost or Ridge): predict MCC
  4. Report R^2, predicted-vs-actual table, and the feature
     importance ranking that explains predictability.

Caveat: 46 organisms is small for ML. R^2 of 0.3-0.5 would be a
real signal; anything higher would be suspicious overfit. We use
LOO-org CV to bound that risk.

Output:
  per_org_predictability.csv     : organism, actual_MCC, predicted_MCC, residual, features...
  organism_predictability.json   : R^2, feature importances, regression config
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--per-org",  type=Path,
                   default=DATA_DIR / "per_organism_loo.csv")
    p.add_argument("--out-csv",  type=Path,
                   default=DATA_DIR / "per_org_predictability.csv")
    p.add_argument("--out-json", type=Path,
                   default=DATA_DIR / "organism_predictability.json")
    p.add_argument("--method", choices=["xgb","ridge"], default="ridge",
                   help="ridge is more conservative on n=46 (default); "
                        "xgb can find nonlinearities if they exist")
    args = p.parse_args()

    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    D = args.data_dir
    po = pd.read_csv(args.per_org)
    print(f"Targets: {len(po)} organisms with measured MCC "
          f"(mean {po.MCC.mean():.4f} +/- {po.MCC.std(ddof=0):.4f})")

    # ---- build per-organism features ----
    print("Building per-organism features ...")
    orth = pd.read_csv(D / "orthology_features.csv")
    print(f"  orth: {len(orth):,} rows")

    feats = []
    cooc_path = D / "cooccurrence_features.csv"
    cooc = pd.read_csv(cooc_path) if cooc_path.exists() else None
    if cooc is not None:
        print(f"  cooc: {len(cooc):,} rows")

    fold_cols = [c for c in orth.columns if c.startswith("family_frac_essential_fold")]
    # use mean across folds as a generic family_frac proxy at the org level
    orth["_ff_mean"] = orth[fold_cols].mean(axis=1) if fold_cols else float("nan")

    # compute OG breadth per OG
    og_breadth = orth.groupby("og_id")["family_n_organisms"].first() \
                       if "family_n_organisms" in orth.columns else None

    for _, row in po.iterrows():
        org = row["organism"]
        g = orth[orth.organism == org]
        if len(g) == 0:
            print(f"  WARN: {org} has 0 orth rows, skipping")
            continue
        rec = {
            "organism":             org,
            "clade":                row["clade"],
            "clade_n_orgs":         int(row["clade_n_orgs"]),
            "actual_MCC":           float(row["MCC"]),
            "n_genes":              int(len(g)),
            "frac_orphan":          float((g["is_orphan"] == 1).mean()) if "is_orphan" in g else float("nan"),
            "mean_family_n_orgs":   float(g["family_n_organisms"].mean()),
            "median_family_n_orgs": float(g["family_n_organisms"].median()),
            "frac_large_OGs":       float((g["family_n_organisms"] >= 30).mean()),
            "mean_n_paralogs":      float(g["n_paralogs_in_genome"].mean()),
            "median_n_paralogs":    float(g["n_paralogs_in_genome"].median()),
            "mean_family_frac":     float(g["_ff_mean"].mean()),
            "std_family_frac":      float(g["_ff_mean"].std()),
            "log10_n_genes":        float(np.log10(max(len(g), 1))),
            "log2_clade_n_orgs":    float(np.log2(int(row["clade_n_orgs"]))),
        }
        if cooc is not None:
            cg = cooc[cooc.organism == org]
            if len(cg) > 0:
                for c in ["cooccur_n_neighbors_80", "cooccur_max_jaccard"]:
                    if c in cg.columns:
                        rec[f"mean_{c}"] = float(cg[c].mean())
        feats.append(rec)

    fdf = pd.DataFrame(feats)
    print(f"  built features for {len(fdf)} organisms")

    feat_cols = [c for c in fdf.columns
                  if c not in ("organism","clade","actual_MCC")]
    print(f"  features used ({len(feat_cols)}): {feat_cols}")
    print(f"\n=== feature -> MCC simple correlations (Pearson) ===")
    correlations = []
    for c in feat_cols:
        s = fdf[c]
        if s.std() == 0 or s.isna().all(): continue
        corr = float(s.corr(fdf.actual_MCC))
        correlations.append((c, corr))
    correlations.sort(key=lambda kv: -abs(kv[1]))
    for c, corr in correlations:
        print(f"  {c:<32s} r = {corr:+.3f}")

    # ---- LOO-organism regression ----
    print(f"\nLOO-organism regression (method = {args.method}) ...")
    X = fdf[feat_cols].fillna(fdf[feat_cols].median(numeric_only=True)).values
    y = fdf["actual_MCC"].values
    preds = np.zeros_like(y)
    import warnings; warnings.filterwarnings("ignore")
    for i in range(len(fdf)):
        mask = np.ones(len(fdf), dtype=bool); mask[i] = False
        if args.method == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(X[mask])
            mdl = Ridge(alpha=2.0).fit(sc.transform(X[mask]), y[mask])
            preds[i] = mdl.predict(sc.transform(X[[i]]))[0]
        else:
            import xgboost as xgb
            mdl = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, min_child_weight=2,
                                     verbosity=0, random_state=0)
            mdl.fit(X[mask], y[mask])
            preds[i] = float(mdl.predict(X[[i]])[0])

    # ---- metrics ----
    fdf["predicted_MCC"] = preds
    fdf["residual"]      = fdf["actual_MCC"] - fdf["predicted_MCC"]
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2     = 1 - ss_res / ss_tot if ss_tot else float("nan")
    mae    = float(np.abs(fdf.residual).mean())
    rmse   = float(np.sqrt((fdf.residual ** 2).mean()))

    print(f"\n=== LOO-organism regression metrics ===")
    print(f"  R^2  = {r2:+.3f}   (1.0 = perfect; 0.0 = no skill vs mean)")
    print(f"  MAE  = {mae:.4f}   ({mae/fdf.actual_MCC.std(ddof=0):.2f} x sigma of actual)")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  range actual:    {y.min():+.3f}  ..  {y.max():+.3f}")
    print(f"  range predicted: {preds.min():+.3f}  ..  {preds.max():+.3f}")

    # ---- per-org table sorted by residual (where do we under/over-predict?) ----
    print(f"\n=== organisms where predictor is most WRONG (residual = actual - pred) ===")
    fdf_sorted = fdf.sort_values("residual")
    print(f"  most over-predicted (predictor said high, actual low):")
    for _, r in fdf_sorted.head(5).iterrows():
        print(f"    {r.organism:<30s} actual={r.actual_MCC:+.3f}  pred={r.predicted_MCC:+.3f}  "
              f"residual={r.residual:+.3f}  clade={r.clade}")
    print(f"  most under-predicted (predictor said low, actual high):")
    for _, r in fdf_sorted.tail(5).iterrows():
        print(f"    {r.organism:<30s} actual={r.actual_MCC:+.3f}  pred={r.predicted_MCC:+.3f}  "
              f"residual={r.residual:+.3f}  clade={r.clade}")

    # ---- save ----
    fdf.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}")

    summary = {
        "n_organisms":   len(fdf),
        "method":        args.method,
        "r2_loo_org":    r2,
        "mae":           mae,
        "rmse":          rmse,
        "feature_correlations": [
            {"feature": c, "pearson_r": r} for c, r in correlations
        ],
        "features_used": feat_cols,
        "actual_mean":   float(y.mean()),
        "actual_std":    float(np.std(y)),
        "predicted_mean": float(preds.mean()),
        "predicted_std": float(np.std(preds)),
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out_json}")

    if r2 > 0.30:
        print(f"\nVerdict: R^2 = {r2:+.2f} -- predictability features carry real signal.")
        print(f"  Can attach per-organism confidence at deployment.")
    elif r2 > 0.10:
        print(f"\nVerdict: R^2 = {r2:+.2f} -- weak but non-zero. Predictability features "
              f"capture some variance but not most.")
    else:
        print(f"\nVerdict: R^2 = {r2:+.2f} -- no predictive signal. Per-organism MCC "
              f"variance is essentially unexplained by org-level features we computed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
