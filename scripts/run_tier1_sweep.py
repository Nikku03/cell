"""Measure Tier-1 feature detectors vs Breuer 2019.

Runs four comparisons on both the full 455-gene set and a
balanced n=40 panel:

  1. v10b priors-union (no ML)               — the rule-based baseline
  2. Tier1XgbDetector(feature_slice="esm2_only")
  3. Tier1XgbDetector(feature_slice="priors_only")
  4. Tier1XgbDetector(feature_slice="esm2_plus_priors")

XGBoost runs use 5-fold stratified CV with a fixed split seed.
Outputs a JSON summary under ``outputs/`` and prints a markdown-
formatted table the caller can paste into a PR or fact file.

Usage::

    python scripts/run_tier1_sweep.py \\
        --cache-dir cell_sim/features/cache \\
        --trajectory-csv outputs/predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4_composed_all455_v10a.csv \\
        --panel-seed 42 --n-panel 40 \\
        --out outputs/tier1_sweep_v11.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from cell_sim.layer6_essentiality.tier1_xgb_detector import (
    PriorFeatureSet,
    PriorsUnionDetector,
    Tier1XgbDetector,
    build_balanced_panel,
    build_feature_bundle,
    confusion,
    default_registry,
    load_breuer_labels,
    mcc,
)


def _format_row(name: str, d: dict) -> str:
    conf = d["confusion"]
    return (
        f"| {name:30s} | {d['mcc']:.3f} | {d['confusion']['tp']:>3} "
        f"| {d['confusion']['fp']:>3} | {d['confusion']['tn']:>3} "
        f"| {d['confusion']['fn']:>3} |"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path,
                   default=REPO_ROOT / "cell_sim/features/cache")
    p.add_argument("--breuer", type=Path,
                   default=REPO_ROOT
                   / "memory_bank/data/syn3a_essentiality_breuer2019.csv")
    p.add_argument("--trajectory-csv", type=Path,
                   default=REPO_ROOT
                   / "outputs/predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4_composed_all455_v10a.csv",
                   help="v10a/v10b sweep output used to derive the "
                        "traj_flag prior. Optional; traj is zero if "
                        "missing.")
    p.add_argument("--panel-seed", type=int, default=42)
    p.add_argument("--n-panel", type=int, default=40)
    p.add_argument("--cv-split-seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "outputs/tier1_sweep_v11.json")
    args = p.parse_args()

    t0 = time.perf_counter()

    registry = default_registry(args.cache_dir)
    priors = PriorFeatureSet(trajectory_csv=args.trajectory_csv)
    labels = load_breuer_labels(args.breuer, quasi_as_positive=True)
    strict = load_breuer_labels(args.breuer, quasi_as_positive=False)

    result = {
        "config": {
            "cache_dir": str(args.cache_dir),
            "breuer": str(args.breuer),
            "trajectory_csv": str(args.trajectory_csv),
            "panel_seed": args.panel_seed,
            "n_panel": args.n_panel,
            "cv_split_seed": args.cv_split_seed,
            "n_splits": args.n_splits,
            "registered_sources": registry.list_sources(),
            "cached_sources": {
                n: registry.is_cached(n) for n in registry.list_sources()
            },
        },
        "full_455": None,
        "balanced_panel": None,
    }

    # ----- FULL 455 -----
    keep_full = sorted(set(labels) & set(registry.load("esm2_650M").index))
    bundle_full = build_feature_bundle(keep_full, registry, priors)
    y_full = np.array([labels[t] for t in keep_full], dtype=int)

    pu = PriorsUnionDetector(prior_set=priors)
    pu_pred = pu.predict(keep_full)
    pu_mcc = mcc(y_full, pu_pred)
    pu_conf = confusion(y_full, pu_pred)

    xgb_results = {}
    for slice_name in ("esm2_only", "priors_only", "esm2_plus_priors",
                        "stacked"):
        det = Tier1XgbDetector(feature_slice=slice_name)
        xgb_results[slice_name] = det.cv_score(
            bundle_full, y_full,
            n_splits=args.n_splits,
            split_seed=args.cv_split_seed,
        )

    result["full_455"] = {
        "n_genes": len(keep_full),
        "n_pos": int(y_full.sum()),
        "n_neg": int((y_full == 0).sum()),
        "label_scheme": "binary; Quasiessential -> positive",
        "priors_union": {
            "mcc": pu_mcc,
            "confusion": pu_conf,
        },
        "xgb": xgb_results,
    }

    # ----- BALANCED n=40 -----
    esm_idx = set(registry.load("esm2_650M").index)
    panel = build_balanced_panel(
        strict, args.n_panel, args.panel_seed, require_in=esm_idx,
    )
    bundle_panel = build_feature_bundle(panel, registry, priors)
    y_panel = np.array([strict[t] for t in panel], dtype=int)

    pu_pred_panel = pu.predict(panel)
    pu_mcc_panel = mcc(y_panel, pu_pred_panel)
    pu_conf_panel = confusion(y_panel, pu_pred_panel)

    xgb_panel = {}
    for slice_name in ("esm2_only", "priors_only", "esm2_plus_priors",
                        "stacked"):
        det = Tier1XgbDetector(feature_slice=slice_name)
        xgb_panel[slice_name] = det.cv_score(
            bundle_panel, y_panel,
            n_splits=args.n_splits,
            split_seed=args.cv_split_seed,
        )

    result["balanced_panel"] = {
        "n_panel": args.n_panel,
        "n_pos": int(y_panel.sum()),
        "n_neg": int((y_panel == 0).sum()),
        "panel_seed": args.panel_seed,
        "panel": panel,
        "label_scheme": "binary; strict Essential vs Nonessential",
        "priors_union": {
            "mcc": pu_mcc_panel,
            "confusion": pu_conf_panel,
        },
        "xgb": xgb_panel,
    }

    result["elapsed_s"] = time.perf_counter() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float))

    # Markdown summary.
    print()
    print(f"## Tier-1 sweep v11 — n={len(keep_full)} full / "
          f"n={args.n_panel} balanced")
    print()
    print(f"Elapsed: {result['elapsed_s']:.1f}s")
    print()
    for label, block in (("FULL 455 (quasi=positive)", result["full_455"]),
                          ("BALANCED-40 (strict)", result["balanced_panel"])):
        print(f"### {label}")
        print(f"n={block['n_pos'] + block['n_neg']}  "
              f"pos={block['n_pos']}  neg={block['n_neg']}")
        print()
        print("| detector                       | MCC   | TP  | FP  | TN  | FN  |")
        print("|--------------------------------|-------|-----|-----|-----|-----|")
        print(_format_row(
            "v10b priors-union (no ML)",
            {"mcc": block["priors_union"]["mcc"],
             "confusion": block["priors_union"]["confusion"]},
        ))
        for slice_name, xgb in block["xgb"].items():
            print(_format_row(
                f"Tier1XGB  {slice_name}",
                {"mcc": xgb["aggregated_mcc"],
                 "confusion": xgb["aggregated_confusion"]},
            ))
        print()

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
