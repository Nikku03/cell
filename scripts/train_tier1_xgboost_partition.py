"""Tier-1 XGBoost restricted to the v15-silent partition.

The naive stack (see ``train_tier1_xgboost.py`` and the measured fact
``mcc_tier1_xgboost_naive_stack``) is falsified: feeding ESM-2's 1280
dims into XGBoost alongside v15's outputs overfits and drops MCC from
0.537 to 0.443. v15's 290 positive predictions have 3 FPs — basically
perfect — so any refinement has to avoid corrupting that regime.

The partition approach asks a narrower question: "among the 168 genes
v15 calls NONESSENTIAL, can Tier-1 features distinguish the 96 true
essentials (current FNs) from the 72 true nonessentials (current TNs)?"
If yes, we flip those v15-nonessential genes that the partition model
rates positive. The v15-positive regime is left untouched, so the
3 FPs stay 3 FPs and the precision floor is preserved.

Headline to measure::

    new_mcc = MCC(
        tp = v15_tp + partition_recovered_tps,
        fp = v15_fp + partition_introduced_fps,
        tn = v15_tn - partition_introduced_fps,
        fn = v15_fn - partition_recovered_tps,
    )

Because we only flip v15-negatives, partition-recovered-tps come out of
v15_fn and partition-introduced-fps come out of v15_tn.

Usage::

    python scripts/train_tier1_xgboost_partition.py \\
        --out outputs/tier1_xgboost_partition.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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


def _load_tier1(cache_dir: Path) -> pd.DataFrame:
    esm = pd.read_parquet(cache_dir / "esm2_650M.parquet")
    ef = pd.read_parquet(cache_dir / "esmfold_v1.parquet").drop(
        columns=["esmfold_disorder_fraction"], errors="ignore",
    )
    mace = pd.read_parquet(cache_dir / "mace_off_kcat.parquet")
    return esm.join(ef, how="outer").join(mace, how="outer")


def _mcc_from_confusion(tp: int, fp: int, tn: int, fn: int) -> float:
    num = tp * tn - fp * fn
    den_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den_sq <= 0:
        return 0.0
    return float(num / np.sqrt(den_sq))


# ------------------------------------------------------------------
# per-threshold fold CV with PCA-reduced ESM-2
# ------------------------------------------------------------------
def _partition_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_pca: int,
    threshold: float,
    xgb_params: dict,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> dict:
    """5-fold stratified CV on the v15-silent partition.

    Projects the 1280 ESM-2 dims through PCA fit per-fold on the
    training split (no leakage of test-set rows into the PCA basis).
    Non-ESM columns (ESMFold, MACE) are passed through unchanged.
    """
    esm_cols = [c for c in X.columns if c.startswith("esm2_650M_")]
    other_cols = [c for c in X.columns if c not in esm_cols]

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed,
    )
    oof_proba = np.full(len(X), np.nan, dtype=float)
    fold_mccs: list[float] = []

    for tr, te in skf.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        # Fit PCA on ESM-2 dims using ONLY the training fold.
        # ESMFold + MACE are low-dim already so we pass them through.
        pca = PCA(n_components=min(n_pca, X_tr.shape[0] - 1), random_state=seed)
        tr_esm = pca.fit_transform(X_tr[esm_cols].fillna(0.0))
        te_esm = pca.transform(X_te[esm_cols].fillna(0.0))
        tr_other = X_tr[other_cols].values
        te_other = X_te[other_cols].values
        X_tr_proj = np.concatenate([tr_esm, tr_other], axis=1)
        X_te_proj = np.concatenate([te_esm, te_other], axis=1)

        spw = float((y_tr == 0).sum()) / max(1, (y_tr == 1).sum())
        clf = XGBClassifier(scale_pos_weight=spw, **xgb_params)
        clf.fit(X_tr_proj, y_tr, verbose=False)
        proba = clf.predict_proba(X_te_proj)[:, 1]
        oof_proba[te] = proba
        y_hat = (proba >= threshold).astype(int)
        fold_mccs.append(
            float(matthews_corrcoef(y_te, y_hat))
        )

    y_hat_all = (oof_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, y_hat_all, labels=[0, 1],
    ).ravel()
    pooled = _mcc_from_confusion(int(tp), int(fp), int(tn), int(fn))
    return {
        "fold_mcc_mean": float(np.mean(fold_mccs)),
        "fold_mcc_std": float(np.std(fold_mccs, ddof=1)),
        "partition_pooled_mcc": pooled,
        "partition_confusion": {
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        },
        "oof_proba": oof_proba,
    }


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "outputs/tier1_xgboost_partition.json",
    )
    ap.add_argument(
        "--n-pca", type=int, default=32,
        help="ESM-2 PCA components (default 32, vs 1280 raw dims)",
    )
    args = ap.parse_args()

    # --- load ---
    t = time.perf_counter()
    raw_v15 = pd.read_csv(V15_PRED_CSV).set_index("locus_tag")
    tier1 = _load_tier1(CACHE_DIR)
    common = tier1.index.intersection(raw_v15.index)
    assert len(common) == 455, f"expected 455 loci, got {len(common)}"
    raw_v15 = raw_v15.loc[common]
    tier1 = tier1.loc[common]

    # Binary label: Essential or Quasi -> 1
    y_full = raw_v15["true_class"].isin(
        ["Essential", "Quasiessential"],
    ).astype(int)
    y_full.index = raw_v15.index

    # v15 baseline on the full set
    v15_pred = raw_v15["essential"].astype(int)
    v15_conf = confusion_matrix(y_full, v15_pred, labels=[0, 1])
    v15_tn, v15_fp, v15_fn, v15_tp = v15_conf.ravel()
    mcc_v15 = _mcc_from_confusion(
        int(v15_tp), int(v15_fp), int(v15_tn), int(v15_fn),
    )
    print(f"v15 baseline: MCC={mcc_v15:.4f}  "
          f"tp={v15_tp} fp={v15_fp} tn={v15_tn} fn={v15_fn}")

    # --- restrict to v15-negative partition ---
    neg_idx = v15_pred[v15_pred == 0].index
    X_part = tier1.loc[neg_idx]
    y_part = y_full.loc[neg_idx]
    print(f"\npartition (v15-negative) size: {len(neg_idx)}  "
          f"positives (current FNs)={int(y_part.sum())}  "
          f"negatives (current TNs)={int((y_part == 0).sum())}  "
          f"dims (raw)={X_part.shape[1]}  "
          f"load {time.perf_counter() - t:.1f}s")

    # --- sweep thresholds ---
    xgb_params = dict(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        reg_lambda=2.0, objective="binary:logistic",
        tree_method="hist", eval_metric="logloss",
        random_state=SEED, n_jobs=4,
    )

    scan: list[dict] = []
    best = None
    for thr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        res = _partition_cv(X_part, y_part, args.n_pca, thr, xgb_params)
        # Translate partition confusion -> global MCC
        recovered_tp = res["partition_confusion"]["tp"]
        introduced_fp = res["partition_confusion"]["fp"]
        new_tp = int(v15_tp) + recovered_tp
        new_fp = int(v15_fp) + introduced_fp
        new_tn = int(v15_tn) - introduced_fp
        new_fn = int(v15_fn) - recovered_tp
        global_mcc = _mcc_from_confusion(new_tp, new_fp, new_tn, new_fn)
        delta = global_mcc - mcc_v15
        row = {
            "threshold": thr,
            "partition_pooled_mcc": res["partition_pooled_mcc"],
            "partition_confusion": res["partition_confusion"],
            "global_tp": new_tp, "global_fp": new_fp,
            "global_tn": new_tn, "global_fn": new_fn,
            "global_mcc": global_mcc,
            "delta_over_v15": delta,
        }
        scan.append(row)
        print(f"thr={thr:.2f}  partition_mcc={res['partition_pooled_mcc']:.3f}  "
              f"recovered_tp={recovered_tp}  introduced_fp={introduced_fp}  "
              f"global_mcc={global_mcc:.4f}  delta={delta:+.4f}")
        if best is None or global_mcc > best["global_mcc"]:
            best = row

    out = {
        "v15_baseline": {
            "mcc": mcc_v15,
            "confusion": {
                "tp": int(v15_tp), "fp": int(v15_fp),
                "tn": int(v15_tn), "fn": int(v15_fn),
            },
        },
        "n_pca": args.n_pca,
        "xgb_params": xgb_params,
        "partition_size": int(len(neg_idx)),
        "partition_positives": int(y_part.sum()),
        "partition_negatives": int((y_part == 0).sum()),
        "threshold_scan": scan,
        "best": best,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nbest: thr={best['threshold']} global_mcc={best['global_mcc']:.4f}  "
          f"delta={best['delta_over_v15']:+.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
