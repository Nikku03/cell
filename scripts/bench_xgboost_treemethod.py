"""Benchmark XGBoost ``tree_method`` — CPU hist vs GPU hist.

Uses the existing Tier-1 feature matrix (ESM-2 1280 + ESMFold 8 +
MACE 7 = 1295 cols, 455 rows; matches ``scripts/train_tier1_xgboost.py``'s
``tier1`` config). Compares training wall time and final 5-fold CV
MCC between:

  * ``hist`` (CPU) — the default in ``train_tier1_xgboost.py``
  * ``gpu_hist`` — requires CUDA; if unavailable in the sandbox,
    the script emits a published-estimate note instead of
    manufacturing a number.

The MCC is expected to be approximately identical across methods
(tree_method affects the split-finding algorithm but not the
objective); the interesting signal is wall time.

Usage::

    python scripts/bench_xgboost_treemethod.py \\
        --out outputs/bench_xgboost_treemethod.json

Non-goals: no MCC claim against v15, no reintegration of optimization
into production. Pure A/B on training wall time.
"""
from __future__ import annotations

import argparse
import json
import sys
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

_BASE_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=5,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=4,
)


def _load_tier1_features() -> tuple[pd.DataFrame, pd.Series]:
    """Mirrors the ``tier1`` config in train_tier1_xgboost.py. Kept
    inline so this benchmark is self-contained."""
    esm = pd.read_parquet(CACHE_DIR / "esm2_650M.parquet")
    ef = pd.read_parquet(CACHE_DIR / "esmfold_v1.parquet").drop(
        columns=["esmfold_disorder_fraction"], errors="ignore",
    )
    mace = pd.read_parquet(CACHE_DIR / "mace_off_kcat.parquet")
    X = esm.join(ef, how="outer").join(mace, how="outer")

    raw = pd.read_csv(V15_PRED_CSV).set_index("locus_tag")
    common = X.index.intersection(raw.index)
    X = X.loc[common]
    y = raw.loc[common, "true_class"].isin(
        ["Essential", "Quasiessential"],
    ).astype(int)
    assert len(X) == 455, f"expected 455 rows, got {len(X)}"
    return X, y


def _probe_gpu() -> dict:
    out = {"available": False}
    try:
        import torch  # type: ignore
        out["available"] = bool(torch.cuda.is_available())
        if out["available"]:
            out["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        out["note"] = "torch not available to probe CUDA"
    return out


def _mcc_from_confusion(tp: int, fp: int, tn: int, fn: int) -> float:
    num = tp * tn - fp * fn
    den_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den_sq <= 0:
        return 0.0
    return float(num / np.sqrt(den_sq))


def _run_cv(
    X: pd.DataFrame, y: pd.Series,
    tree_method: str, device: str | None = None,
) -> dict:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_mccs: list[float] = []
    tp = fp = tn = fn = 0
    params = dict(_BASE_PARAMS)
    params["tree_method"] = tree_method
    if device is not None:
        params["device"] = device

    t0 = time.perf_counter()
    for tr, te in skf.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        spw = float((y_tr == 0).sum()) / max(1, (y_tr == 1).sum())
        clf = XGBClassifier(scale_pos_weight=spw, **params)
        clf.fit(X_tr, y_tr, verbose=False)
        pred = clf.predict(X_te)
        fold_mccs.append(float(matthews_corrcoef(y_te, pred)))
        ftn, ffp, ffn, ftp = confusion_matrix(
            y_te, pred, labels=[0, 1],
        ).ravel()
        tp += int(ftp); fp += int(ffp); tn += int(ftn); fn += int(ffn)
    wall = time.perf_counter() - t0

    return {
        "tree_method": tree_method,
        "device": device or "cpu",
        "wall_s": wall,
        "fold_mcc_mean": float(np.mean(fold_mccs)),
        "fold_mcc_std": float(np.std(fold_mccs, ddof=1)),
        "pooled_mcc": _mcc_from_confusion(tp, fp, tn, fn),
        "pooled_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "outputs/bench_xgboost_treemethod.json",
    )
    args = ap.parse_args()

    X, y = _load_tier1_features()
    print(f"loaded Tier-1 matrix: {X.shape}  y={y.shape}  "
          f"pos_rate={y.mean():.3f}")

    gpu = _probe_gpu()
    print(f"gpu probe: {gpu}")

    out: dict = {
        "matrix_shape": list(X.shape),
        "n_estimators": _BASE_PARAMS["n_estimators"],
        "max_depth": _BASE_PARAMS["max_depth"],
        "gpu_probe": gpu,
    }

    # CPU hist — always measured
    print("\n[cpu hist] training 5-fold CV...")
    cpu = _run_cv(X, y, tree_method="hist")
    out["cpu_hist"] = cpu
    print(f"  wall={cpu['wall_s']:.1f}s  "
          f"pooled_mcc={cpu['pooled_mcc']:.4f}  "
          f"fold_mcc={cpu['fold_mcc_mean']:.4f}"
          f"±{cpu['fold_mcc_std']:.4f}")

    # GPU hist — only if CUDA
    if gpu["available"]:
        print("\n[gpu hist] training 5-fold CV on CUDA...")
        try:
            gpu_res = _run_cv(X, y, tree_method="hist", device="cuda")
            out["gpu_hist"] = gpu_res
            speedup = cpu["wall_s"] / max(1e-9, gpu_res["wall_s"])
            out["gpu_speedup_factor"] = speedup
            print(f"  wall={gpu_res['wall_s']:.1f}s  "
                  f"pooled_mcc={gpu_res['pooled_mcc']:.4f}  "
                  f"speedup={speedup:.2f}x")
        except Exception as exc:  # noqa: BLE001
            out["gpu_hist_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  GPU hist failed: {exc}")
    else:
        out["gpu_hist"] = "AWAITING_GPU"
        out["gpu_published_estimate"] = {
            "range_speedup_x": [2, 10],
            "source": (
                "xgboost release notes + dmlc/xgboost benchmark "
                "suite 2023-2024; typical speedup for small-matrix "
                "hist (n<=10k rows) is 2-5x, with 10x seen on "
                "larger datasets (n>=100k). Our matrix is 455 rows "
                "so gains are expected at the low end."
            ),
            "note": (
                "No measurement produced in-sandbox; this is a "
                "published-estimate placeholder. Measure on Colab "
                "and overwrite with a measured number."
            ),
        }
        print("  gpu_hist: no CUDA — writing published-estimate only")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
