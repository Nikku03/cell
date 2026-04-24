"""Aggregate v15 multi-seed replicates and compute MCC mean / stddev.

Reads the three v15 prediction CSVs (seed=42 + seed=1 + seed=2),
loads the Breuer labels, and emits a JSON summary suitable for
writing into a measured-fact replicate block.

Usage::

    python scripts/aggregate_v15_replicates.py \\
        --out outputs/v15_replicates_summary.json
"""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _confusion_mcc(pred_csv: Path) -> dict:
    df = pd.read_csv(pred_csv)
    y_true = df["true_class"].isin(["Essential", "Quasiessential"]).astype(int)
    y_pred = df["essential"].astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    num = tp * tn - fp * fn
    den_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = num / sqrt(den_sq) if den_sq > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "mcc": mcc,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "outputs/v15_replicates_summary.json",
    )
    args = p.parse_args()

    prefix = "outputs/predictions_parallel_s0.05_t0.5_seed"
    suffix = "_thr0.1_w4_composed_all455_v15_round2_priors.csv"
    replicates = {}
    for seed in (42, 1, 2):
        path = REPO_ROOT / f"{prefix}{seed}{suffix}"
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        replicates[f"seed_{seed}"] = _confusion_mcc(path)

    if len(replicates) < 2:
        print("need at least 2 replicates; abort")
        return 1

    mccs = [r["mcc"] for r in replicates.values()]
    n = len(mccs)
    mean_mcc = sum(mccs) / n
    var = sum((m - mean_mcc) ** 2 for m in mccs) / max(1, n - 1)
    std_mcc = var ** 0.5
    summary = {
        "n_seeds": n,
        "seeds_run": sorted(int(k.split("_")[1]) for k in replicates),
        "mean_mcc": round(mean_mcc, 4),
        "std_mcc": round(std_mcc, 4),
        "min_mcc": round(min(mccs), 4),
        "max_mcc": round(max(mccs), 4),
        "ci_95_low": round(mean_mcc - 1.96 * std_mcc / sqrt(n), 4),
        "ci_95_high": round(mean_mcc + 1.96 * std_mcc / sqrt(n), 4),
        "per_seed": replicates,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
