"""Train a Tier-1 XGBoost essentiality predictor.

Stacks four feature groups on the 455 Breuer-labeled Syn3A CDS:

  * ESM-2 650M protein embeddings     (1280 dims)
  * ESMFold v1 structural descriptors (9 dims)
  * MACE-OFF BDE-derived kcat stats   (7 dims, 111/455 rows with real values)
  * v15 composed-detector outputs     (essential, confidence,
                                       time_to_failure_s, failure_mode
                                       one-hot)

and evaluates three configurations via stratified 5-fold CV:

  * ``tier1``    — ESM-2 + ESMFold + MACE only
  * ``v15``      — v15 detector outputs only (the existing baseline
                   lifted into XGBoost to pick up interactions)
  * ``union``    — Tier-1 ⊕ v15

Target: ``Essential`` or ``Quasiessential`` in the Breuer 2019 labels
maps to 1; ``Nonessential`` maps to 0. The v15 composed detector on
the same labels produces MCC = 0.5372 (measured fact
``mcc_against_breuer_v15_round2_priors``); we ask whether Tier-1
features lift the CV MCC above that.

Usage::

    python scripts/train_tier1_xgboost.py \\
        --out outputs/tier1_xgboost.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "cell_sim/features/cache"
V15_PRED_CSV = REPO_ROOT / (
    "outputs/predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4"
    "_composed_all455_v15_round2_priors.csv"
)

SEED = 42
N_FOLDS = 5
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=5,
    reg_lambda=1.0,
    objective="binary:logistic",
    tree_method="hist",
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=4,
)


# ------------------------------------------------------------------
# feature join
# ------------------------------------------------------------------
def _load_v15_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("locus_tag")
    # Binary prediction + confidence + time-to-failure + failure mode.
    df = df[["essential", "confidence", "time_to_failure_s", "failure_mode"]]
    # One-hot failure mode (5 categories in the v15 run).
    mode_dummies = pd.get_dummies(
        df["failure_mode"], prefix="v15_mode", dtype=float,
    )
    df = df.drop(columns=["failure_mode"]).join(mode_dummies)
    df.columns = [f"v15_{c}" if not c.startswith("v15_") else c
                  for c in df.columns]
    return df


def _load_tier1(cache_dir: Path) -> pd.DataFrame:
    esm = pd.read_parquet(cache_dir / "esm2_650M.parquet")
    ef = pd.read_parquet(cache_dir / "esmfold_v1.parquet")
    mace = pd.read_parquet(cache_dir / "mace_off_kcat.parquet")
    # esmfold_disorder_fraction is broken in 0.1.0 (pre-unit-fix parquet)
    # — it's 1.0 for every row. Drop it so XGBoost doesn't waste a
    # tree split on a constant column.
    ef = ef.drop(columns=["esmfold_disorder_fraction"], errors="ignore")
    joined = esm.join(ef, how="outer").join(mace, how="outer")
    return joined


def _build_label_vector(v15_df: pd.DataFrame, raw_v15: pd.DataFrame
                        ) -> pd.Series:
    # raw_v15 has the original true_class column we use for the label.
    y = (
        raw_v15.set_index("locus_tag")["true_class"]
        .reindex(v15_df.index)
        .isin(["Essential", "Quasiessential"])
        .astype(int)
    )
    return y


# ------------------------------------------------------------------
# training / evaluation
# ------------------------------------------------------------------
def _cv_mcc(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> dict:
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed,
    )
    fold_mccs: list[float] = []
    total_tp = total_fp = total_tn = total_fn = 0
    per_fold: list[dict] = []
    oof_proba = np.full(len(X), np.nan, dtype=float)
    feat_imp_sum = np.zeros(X.shape[1], dtype=float)

    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        # Scale-pos-weight balances the positive class during training.
        spw = float((y_tr == 0).sum()) / max(1, (y_tr == 1).sum())
        clf = XGBClassifier(scale_pos_weight=spw, **params)
        clf.fit(X_tr, y_tr, verbose=False)
        proba = clf.predict_proba(X_te)[:, 1]
        oof_proba[te] = proba
        y_hat = (proba >= 0.5).astype(int)
        mcc = float(matthews_corrcoef(y_te, y_hat))
        tn, fp, fn, tp = confusion_matrix(
            y_te, y_hat, labels=[0, 1],
        ).ravel()
        fold_mccs.append(mcc)
        total_tp += int(tp); total_fp += int(fp)
        total_tn += int(tn); total_fn += int(fn)
        per_fold.append({
            "fold": fold_idx,
            "mcc": mcc,
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
        })
        feat_imp_sum += clf.feature_importances_

    # Aggregate pooled confusion matrix — the MCC on the pooled counts
    # is a more honest single headline than the mean-of-fold MCCs.
    pooled_mcc = _mcc_from_confusion(
        total_tp, total_fp, total_tn, total_fn,
    )

    feat_imp_mean = feat_imp_sum / n_folds
    top_idx = np.argsort(feat_imp_mean)[::-1][:15]
    top_features = [
        (X.columns[i], float(feat_imp_mean[i])) for i in top_idx
    ]

    return {
        "fold_mcc_mean": float(np.mean(fold_mccs)),
        "fold_mcc_std": float(np.std(fold_mccs, ddof=1)),
        "pooled_mcc": pooled_mcc,
        "pooled_confusion": {
            "tp": total_tp, "fp": total_fp,
            "tn": total_tn, "fn": total_fn,
        },
        "per_fold": per_fold,
        "top_features": top_features,
        "oof_proba": oof_proba,
    }


def _mcc_from_confusion(tp: int, fp: int, tn: int, fn: int) -> float:
    num = tp * tn - fp * fn
    den_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den_sq <= 0:
        return 0.0
    return float(num / np.sqrt(den_sq))


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "outputs/tier1_xgboost.json",
    )
    ap.add_argument(
        "--save-oof", action="store_true",
        help="also write predictions_tier1_xgboost.csv",
    )
    args = ap.parse_args()

    t_load = time.perf_counter()
    raw_v15 = pd.read_csv(V15_PRED_CSV)
    v15_feat = _load_v15_features(V15_PRED_CSV)
    tier1 = _load_tier1(CACHE_DIR)

    # Align on the 455 v15-scored genes (which is all Breuer-labeled CDS).
    common = v15_feat.index.intersection(tier1.index)
    assert len(common) == 455, f"expected 455 common loci, got {len(common)}"
    v15_feat = v15_feat.loc[common]
    tier1 = tier1.loc[common]
    y = _build_label_vector(v15_feat, raw_v15)
    assert y.notna().all(), "some labels missing"
    load_wall = time.perf_counter() - t_load
    print(f"loaded features in {load_wall:.1f}s: "
          f"tier1={tier1.shape}  v15={v15_feat.shape}  y={y.shape}  "
          f"pos_rate={y.mean():.3f}")

    configs = {
        "tier1": tier1,
        "v15": v15_feat,
        "union": tier1.join(v15_feat),
    }

    results: dict = {}
    for name, X in configs.items():
        print(f"\n=== config: {name}  (X.shape={X.shape}) ===")
        t = time.perf_counter()
        res = _cv_mcc(X, y, XGB_PARAMS)
        res["wall_s"] = time.perf_counter() - t
        res["n_features"] = X.shape[1]
        print(f"  fold MCC: {res['fold_mcc_mean']:.4f} "
              f"± {res['fold_mcc_std']:.4f}  "
              f"pooled MCC: {res['pooled_mcc']:.4f}")
        print(f"  pooled confusion: {res['pooled_confusion']}")
        print("  top features:")
        for name_f, imp in res["top_features"][:10]:
            print(f"    {name_f:40s}  imp={imp:.4f}")
        # Pop the numpy oof_proba before JSON dump; save separately
        # as a CSV column later if --save-oof is set.
        oof = res.pop("oof_proba")
        if args.save_oof and name == "union":
            pred_df = pd.DataFrame({
                "locus_tag": common,
                "true_class_binary": y.values,
                "tier1_xgb_proba": oof,
                "tier1_xgb_pred": (oof >= 0.5).astype(int),
            })
            pred_csv = args.out.with_name(
                "predictions_tier1_xgboost_union.csv",
            )
            pred_df.to_csv(pred_csv, index=False)
            print(f"  wrote {pred_csv}")
        results[name] = res

    # Add the v15 baseline (the raw binary calls in the CSV).
    y_v15 = raw_v15.set_index("locus_tag").loc[common, "essential"]
    mcc_v15 = float(matthews_corrcoef(y, y_v15))
    tn, fp, fn, tp = confusion_matrix(y, y_v15, labels=[0, 1]).ravel()
    results["_v15_reference"] = {
        "mcc": mcc_v15,
        "confusion": {"tp": int(tp), "fp": int(fp),
                      "tn": int(tn), "fn": int(fn)},
        "note": "v15 composed detector direct calls (no CV; whole set)",
    }
    print(f"\nreference: v15 composed detector MCC = {mcc_v15:.4f}  "
          f"(tp={tp} fp={fp} tn={tn} fn={fn})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
