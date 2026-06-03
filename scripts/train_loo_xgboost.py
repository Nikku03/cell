"""Baseline LOO-by-clade XGBoost training on the multi-organism inventory.

Implements step #3 of the locked architecture: XGBoost on engineered
features, clade-aware leave-one-out CV.

Features used (all from orthology_features.csv):
  - n_paralogs_in_genome
  - family_size_total
  - family_n_organisms
  - is_orphan
  - family_frac_essential_fold{k}  <- fold-aware, k = held-out fold

For each LOO iteration:
  - Test = rows in fold k
  - Train = rows in all other folds
  - Feature `family_frac_essential` value comes from the column
    family_frac_essential_fold{k} (= leakage-free for fold k as test set).
    Both train AND test rows use the same column so the feature
    distribution is consistent across the split.

Non-BERIL labels (rows without orthology features) are dropped from
this baseline -- they have no features at all. ~30k rows excluded.
Once self-BLAST is computed for those, they can be added.

Archaeal clades are filtered out per the user directive
(is_bacterial_clade in build_clade_splits.py).

Outputs:
  memory_bank/data/multiorg_essentiality/loo_baseline_results.json
    {fold_metrics: [...], per_clade_metrics: [...],
     mean_mcc, std_mcc, n_train_total, n_test_total, features_used}

Run:
  python scripts/train_loo_xgboost.py
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ORTH_CSV   = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "orthology_features.csv"
LABELS_CSV = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "labels.csv"
SPLITS_CSV = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "clade_splits.csv"
OUT_JSON   = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "loo_baseline_results.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--orth-csv",     type=Path, default=ORTH_CSV)
    p.add_argument("--cooccur-csv",  type=Path,
                   default=REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "cooccurrence_features.csv",
                   help="Optional co-occurrence features CSV (skip if missing)")
    p.add_argument("--reg-csv",      type=Path,
                   default=REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "regulator_features.csv",
                   help="Optional regulator/signaling/transporter features CSV (skip if missing)")
    p.add_argument("--fba-csv",      type=Path,
                   default=REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality" / "fba_features.csv",
                   help="Optional FBA single-gene-KO features CSV (skip if missing)")
    p.add_argument("--labels-csv",   type=Path, default=LABELS_CSV)
    p.add_argument("--splits-csv",   type=Path, default=SPLITS_CSV)
    p.add_argument("--out",          type=Path, default=OUT_JSON)
    p.add_argument("--n-estimators",   type=int,   default=500)
    p.add_argument("--max-depth",      type=int,   default=4)
    p.add_argument("--learning-rate",  type=float, default=0.05)
    p.add_argument("--seed",           type=int,   default=0)
    args = p.parse_args()

    try:
        import pandas as pd
        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import matthews_corrcoef
    except ImportError as e:
        print(f"ERROR: {e}. Install: pip install pandas xgboost scikit-learn",
              file=sys.stderr)
        return 1

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_clade_splits import is_bacterial_clade

    # ---- Load ----
    print("Loading labels, splits, features ...")
    labels = pd.read_csv(args.labels_csv)
    splits = pd.read_csv(args.splits_csv)
    orth   = pd.read_csv(args.orth_csv)
    print(f"  labels:        {len(labels):>7d}")
    print(f"  splits:        {len(splits):>7d} organisms")
    print(f"  orthology:     {len(orth):>7d} rows")

    # ---- Join: labels x splits (clade+fold) ----
    j = labels.merge(splits[["organism","clade","fold"]],
                       on="organism", how="left")
    # Drop unmapped (shouldn't exist after step #1)
    j = j[j.clade.notna()]
    # Drop archaea
    j["bact"] = j.clade.apply(is_bacterial_clade)
    n_total = len(j)
    n_arch  = int((~j.bact).sum())
    j = j[j.bact].drop(columns="bact")
    print(f"  after archaea filter: {len(j):>7d} (dropped {n_arch} archaeal)")

    # ---- Join with orthology features ----
    feat_cols_base = ["n_paralogs_in_genome", "family_size_total",
                       "family_n_organisms", "is_orphan"]
    fold_cols = [c for c in orth.columns
                  if c.startswith("family_frac_essential_fold")]
    n_folds = len(fold_cols)
    print(f"  features:      {feat_cols_base}  + {n_folds} fold-aware columns")

    keep_cols = ["organism","locus_tag"] + feat_cols_base + fold_cols
    orth = orth[keep_cols]
    j = j.merge(orth, on=["organism","locus_tag"], how="left")

    # ---- Optional: join co-occurrence features ----
    cooccur_cols: list[str] = []
    if args.cooccur_csv.exists():
        cooccur = pd.read_csv(args.cooccur_csv)
        cooccur_cols = [c for c in cooccur.columns if c.startswith("cooccur_")]
        print(f"  co-occurrence features ({len(cooccur)} rows): {cooccur_cols}")
        j = j.merge(cooccur[["organism","locus_tag"] + cooccur_cols],
                     on=["organism","locus_tag"], how="left")
    else:
        print(f"  (no co-occurrence features at {args.cooccur_csv})")

    # ---- Optional: join regulator / signaling / transporter flags ----
    reg_cols: list[str] = []
    if args.reg_csv.exists():
        reg = pd.read_csv(args.reg_csv)
        reg_cols = [c for c in reg.columns
                    if c.startswith("is_") and c != "is_orphan"]
        print(f"  regulator features ({len(reg)} rows): {reg_cols}")
        j = j.merge(reg[["organism","locus_tag"] + reg_cols],
                     on=["organism","locus_tag"], how="left")
    else:
        print(f"  (no regulator features at {args.reg_csv})")

    # ---- Optional: join FBA single-gene-KO features ----
    fba_cols: list[str] = []
    if args.fba_csv.exists():
        fba = pd.read_csv(args.fba_csv)
        fba_cols = [c for c in fba.columns if c.startswith("fba_")]
        print(f"  FBA features ({len(fba)} rows, "
              f"{fba.organism.nunique()} organisms): {fba_cols}")
        j = j.merge(fba[["organism","locus_tag"] + fba_cols],
                     on=["organism","locus_tag"], how="left")
    else:
        print(f"  (no FBA features at {args.fba_csv})")

    has_feat = j.n_paralogs_in_genome.notna()
    n_feat = int(has_feat.sum())
    n_no   = len(j) - n_feat
    print(f"  rows with orthology features: {n_feat:>7d}")
    print(f"  rows without (non-BERIL):     {n_no:>7d}  (DROPPED for this baseline)")
    j = j[has_feat].reset_index(drop=True)

    print(f"\n  TRAINING SET: {len(j)} rows, "
           f"{int(j.essential.sum())} essentials ({j.essential.mean()*100:.1f}%)")
    print(f"  unique clades: {j.clade.nunique()}  unique folds: {j.fold.nunique()}")

    # ---- LOO over folds ----
    fold_results = []
    all_per_clade: list[dict] = []
    for k in range(n_folds):
        t0 = time.time()
        train = j[j.fold != k].copy()
        test  = j[j.fold == k].copy()
        if len(test) == 0:
            print(f"  fold {k}: empty test set, skip")
            continue

        # Feature columns: fixed base + the fold-k specific frac column + cooccur.
        # Rename the fold-specific column to a stable name so the model
        # treats it as one feature regardless of which fold is held out.
        ff_col = f"family_frac_essential_fold{k}"
        feat_cols = feat_cols_base + [ff_col] + cooccur_cols + reg_cols + fba_cols
        X_train = train[feat_cols].rename(columns={ff_col: "family_frac_essential"})
        y_train = train["essential"].astype(int).values
        X_test  = test[feat_cols].rename(columns={ff_col: "family_frac_essential"})
        y_test  = test["essential"].astype(int).values

        # Class imbalance weight
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        spw = n_neg / max(n_pos, 1)

        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=2,
            scale_pos_weight=spw,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0,
            random_state=args.seed,
        )
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        prob = clf.predict_proba(X_test)[:, 1]
        pred = (prob > 0.5).astype(int)

        mcc       = float(matthews_corrcoef(y_test, pred))
        tp        = int(((pred==1)&(y_test==1)).sum())
        fp        = int(((pred==1)&(y_test==0)).sum())
        tn        = int(((pred==0)&(y_test==0)).sum())
        fn        = int(((pred==0)&(y_test==1)).sum())
        precision = tp / max(tp+fp, 1)
        recall    = tp / max(tp+fn, 1)
        f1        = 2*precision*recall / max(precision+recall, 1e-12)

        held_out_clades = sorted(test.clade.unique().tolist())
        print(f"  fold {k}: MCC={mcc:+.3f}  P={precision:.3f}  R={recall:.3f}  "
              f"F1={f1:.3f}  (tp={tp} fp={fp} tn={tn} fn={fn})  "
              f"clades={','.join(held_out_clades)}  wall={time.time()-t0:.1f}s")

        fold_results.append({
            "fold": k, "MCC": mcc, "precision": precision, "recall": recall,
            "F1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_train": len(train), "n_test": len(test),
            "held_out_clades": held_out_clades,
        })

        # ---- per-held-out-clade ----
        test_with_pred = test.assign(pred=pred, prob=prob)
        for clade in held_out_clades:
            sub = test_with_pred[test_with_pred.clade == clade]
            if len(sub) == 0 or sub.essential.nunique() < 2:
                # MCC undefined if class is uniform in this clade's test set
                clade_mcc = None
            else:
                clade_mcc = float(matthews_corrcoef(
                    sub.essential.astype(int), sub.pred.astype(int)))
            all_per_clade.append({
                "fold": k, "clade": clade,
                "n": len(sub),
                "n_essential": int(sub.essential.sum()),
                "MCC": clade_mcc,
            })

    # ---- Headline ----
    fold_mccs = [f["MCC"] for f in fold_results]
    headline = {
        "mean_MCC":       float(sum(fold_mccs)/len(fold_mccs)) if fold_mccs else None,
        "std_MCC":        float(_std(fold_mccs)) if fold_mccs else None,
        "n_folds":        len(fold_results),
        "n_total_test":   sum(f["n_test"] for f in fold_results),
        "n_total_train":  sum(f["n_train"] for f in fold_results) // max(len(fold_results),1),
        "features_used":  feat_cols_base + ["family_frac_essential (fold-aware)"],
        "fold_metrics":   fold_results,
        "per_clade_metrics": all_per_clade,
        "config": {
            "n_estimators":   args.n_estimators,
            "max_depth":      args.max_depth,
            "learning_rate":  args.learning_rate,
            "seed":           args.seed,
        },
    }

    print(f"\n=== HEADLINE (LOO-by-clade across {len(fold_results)} folds) ===")
    print(f"  mean MCC: {headline['mean_MCC']:+.4f} +/- {headline['std_MCC']:.4f}")
    print(f"  total test rows across folds: {headline['n_total_test']}")

    print(f"\n=== Per-clade MCC (sorted) ===")
    pc = [r for r in all_per_clade if r["MCC"] is not None]
    pc.sort(key=lambda r: r["MCC"])
    print(f"  WORST 10:")
    for r in pc[:10]:
        print(f"    {r['clade']:<25s}  fold={r['fold']}  n={r['n']:>5d}  "
              f"ess={r['n_essential']:>4d}  MCC={r['MCC']:+.3f}")
    print(f"  BEST 10:")
    for r in pc[-10:]:
        print(f"    {r['clade']:<25s}  fold={r['fold']}  n={r['n']:>5d}  "
              f"ess={r['n_essential']:>4d}  MCC={r['MCC']:+.3f}")

    # ---- Save ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(headline, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


def _std(xs):
    n = len(xs)
    if n < 2: return 0.0
    m = sum(xs) / n
    return (sum((x-m)**2 for x in xs) / (n-1)) ** 0.5


if __name__ == "__main__":
    sys.exit(main())
