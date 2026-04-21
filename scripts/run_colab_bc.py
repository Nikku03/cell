"""Colab orchestrator: runs the B (multi-seed replicates) + C (higher-scale)
sweeps queued in NEXT_SESSION.md and writes measured-MCC facts.

This script is designed for a Colab L4/A100 VM (8-24 vCPUs) but works
anywhere with >=4 CPU cores, Python 3.11+, the Luthey-Schulten data
staged, and the cell_sim_rust wheel installed. On the project's
sandbox (4 vCPUs) the full run takes ~45 min; on an A100 Colab
instance (12 vCPUs) it takes ~20 min.

Usage (from repo root)::

    python scripts/run_colab_bc.py --workers 12 --out-dir outputs \
        --configs b c \
        --seeds 42 1 2 3 4

Writes:
    outputs/predictions_*.csv            per-sweep predictions
    outputs/metrics_*.json               per-sweep metrics + config
    memory_bank/facts/measured/          summary facts (v*_replicates, v8)

This script does NOT commit or push - the notebook layer handles that.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_sweep_parallel.py"

# ---- B: multi-seed replicates ----
# Each entry is (short_name, detector_config_args). We hold the gene
# panel fixed across replicates by setting --panel-seed=42 and vary
# the simulator RNG via --seed. v2/v3 use scale/t_end variants.
B_CONFIGS: dict[str, list[str]] = {
    "v1_shortwindow_cal": [
        "--max-genes", "40", "--balanced",
        "--calibrate", "10", "--safety-factor", "2.5",
        "--scale", "0.05", "--t-end-s", "0.5",
        "--threshold", "0.05",
        "--use-rust",
    ],
    "v4_shortwindow_nonmetabolic": [
        "--max-genes", "40", "--balanced",
        "--calibrate", "10", "--safety-factor", "2.5",
        "--scale", "0.05", "--t-end-s", "0.5",
        "--threshold", "0.04",
        "--use-rust",
    ],
    "v5_per_rule": [
        "--max-genes", "40", "--balanced",
        "--scale", "0.05", "--t-end-s", "0.5",
        "--detector", "per-rule",
        "--use-rust",
    ],
    "v6a_ensemble_pool_confirm": [
        "--max-genes", "40", "--balanced",
        "--calibrate", "10", "--safety-factor", "2.5",
        "--scale", "0.05", "--t-end-s", "0.5",
        "--detector", "ensemble",
        "--ensemble-policy", "per_rule_with_pool_confirm",
        "--min-pool-dev", "0.02",
        "--use-rust",
    ],
    "v6b_ensemble_and_unique": [
        "--max-genes", "40", "--balanced",
        "--calibrate", "10", "--safety-factor", "2.5",
        "--scale", "0.05", "--t-end-s", "0.5",
        "--detector", "ensemble",
        "--ensemble-policy", "and",
        "--rule-necessity-only",
        "--use-rust",
    ],
}

# ---- C: the one unmeasured config we never got to run ----
C_CONFIG: list[str] = [
    "--max-genes", "40", "--balanced",
    "--calibrate", "10", "--safety-factor", "2.5",
    "--scale", "0.5", "--t-end-s", "1.0",
    "--detector", "ensemble",
    "--ensemble-policy", "per_rule_with_pool_confirm",
    "--min-pool-dev", "0.05",
    "--use-rust",
]


def run_sweep(extra_args: list[str], *, workers: int, seed: int,
              panel_seed: int, out_dir: Path, log_tag: str) -> dict:
    """Invoke run_sweep_parallel.py and return the parsed metrics JSON."""
    cmd = [
        sys.executable, str(SCRIPT),
        "--workers", str(workers),
        "--seed", str(seed),
        "--panel-seed", str(panel_seed),
        "--out-dir", str(out_dir),
        *extra_args,
    ]
    print(f"\n=== {log_tag} seed={seed} panel_seed={panel_seed} ===")
    print("  " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    print(f"  wall: {wall:.1f}s")
    if proc.returncode != 0:
        print("  STDERR tail:")
        print(proc.stderr[-1500:])
        raise RuntimeError(f"sweep failed: {log_tag} seed={seed}")
    # Parse MCC lines from stdout
    for line in proc.stdout.splitlines()[-10:]:
        if "MCC=" in line:
            print(f"  {line.strip()}")
    # Find the metrics JSON this sweep just wrote.
    # run_sweep_parallel.py tags its output files with all the flags;
    # we grab the most recent matching file.
    candidates = sorted(out_dir.glob("metrics_parallel_*.json"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError("no metrics JSON produced")
    latest = candidates[-1]
    with latest.open() as fh:
        payload = json.load(fh)
    try:
        payload["_source_file"] = str(latest.relative_to(REPO_ROOT))
    except ValueError:
        payload["_source_file"] = str(latest)
    payload["_wall_s"] = wall
    return payload


def summarise_replicates(name: str, payloads: list[dict]) -> dict:
    """Compute mean / std / min / max of MCC across seed replicates."""
    mccs = [p["mcc"] for p in payloads]
    tps = [p["tp"] for p in payloads]
    fps = [p["fp"] for p in payloads]
    tns = [p["tn"] for p in payloads]
    fns = [p["fn"] for p in payloads]
    return {
        "name": name,
        "n_replicates": len(payloads),
        "seeds": [p["config"].get("seed") for p in payloads],
        "mcc": {
            "mean": statistics.fmean(mccs),
            "std": statistics.stdev(mccs) if len(mccs) > 1 else 0.0,
            "min": min(mccs),
            "max": max(mccs),
            "values": mccs,
        },
        "confusion_mean": {
            "tp": statistics.fmean(tps),
            "fp": statistics.fmean(fps),
            "tn": statistics.fmean(tns),
            "fn": statistics.fmean(fns),
        },
        "per_replicate_files": [p["_source_file"] for p in payloads],
        "mean_wall_s": statistics.fmean(p["_wall_s"] for p in payloads),
    }


def write_replicates_fact(summary_all: dict, out_path: Path) -> None:
    """Write memory_bank/facts/measured/mcc_replicates_summary.json."""
    best_mean = max(
        summary_all.values(),
        key=lambda s: s["mcc"]["mean"],
    )
    payload = {
        "id": "mcc_replicates_summary",
        "claim": (
            "Multi-seed (5 replicate) MCC measurement for all Session 4-8 "
            "detector configs on a fixed balanced n=40 panel. Gene panel "
            "seed fixed at 42; simulator seeds {42, 1, 2, 3, 4}. "
            "Each config's MCC is reported as mean +/- std across the 5 "
            "replicates so the Session-to-Session comparisons in the "
            "REPORT history table can be read with error bars. Run on a "
            "Colab VM via notebooks/colab_bc_sweep.ipynb."
        ),
        "value": {
            "parameter": "mcc_replicates_summary",
            "n_panel": 40,
            "n_replicates_per_config": 5,
            "panel_seed": 42,
            "simulator_seeds": [42, 1, 2, 3, 4],
            "by_config": summary_all,
            "best_config": best_mean["name"],
            "best_mean_mcc": best_mean["mcc"]["mean"],
        },
        "source": "breuer_2019_elife",
        "source_detail": (
            "Reproducible via 'python scripts/run_colab_bc.py --configs "
            "b --seeds 42 1 2 3 4 --workers 12' on any machine with "
            "cell_sim_rust installed + Luthey-Schulten data staged."
        ),
        "context": {
            "entity": "syn3a_essentiality_predictor",
            "version": "replicates_session_10_colab",
        },
        "confidence": "measured",
        "caveats": [
            "All configs use scale=0.05 + t_end=0.5 s (the short-window "
            "regime). Session 9 (v7) already falsified Path A at scale=0.05 "
            "+ t_end=5.0 s on n=20; this summary does NOT re-measure that.",
            "Std across 5 seeds does not fully capture the gene-panel "
            "component of variance - that requires varying panel_seed, "
            "which is a larger design question deferred to Session 11+.",
            "Brief target MCC > 0.59 remains unreached. Error bars make "
            "the existing v0-v6 measurements credible but do not close "
            "the simulator-biology gap diagnosed across Sessions 4-9.",
        ],
        "dependencies": [
            "mcc_against_breuer_v0", "mcc_against_breuer_v1",
            "mcc_against_breuer_v2", "mcc_against_breuer_v3",
            "mcc_against_breuer_v4", "mcc_against_breuer_v5",
            "mcc_against_breuer_v6",
        ],
        "last_verified": "2026-04-21",
        "used_by": ["scripts/run_colab_bc.py",
                    "notebooks/colab_bc_sweep.ipynb"],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    try:
        rel = out_path.relative_to(REPO_ROOT)
    except ValueError:
        rel = out_path
    print(f"Wrote {rel}")


def write_v8_fact(payload: dict, out_path: Path) -> None:
    payload_mcc = payload["mcc"]
    fact = {
        "id": "mcc_against_breuer_v8",
        "claim": (
            "Ninth MCC measurement. The one scale * t_end corner never "
            "measured in Sessions 4-9: scale=0.5 (10x the v0 baseline), "
            "t_end=1.0 s. Ensemble per_rule_with_pool_confirm, n=40 "
            "balanced, calibration=10, seed=42 (simulator) / panel_seed=42. "
            f"Result: MCC = {payload_mcc:.3f}."
        ),
        "value": {
            "parameter": "mcc_binary_quasi_positive",
            "number": payload_mcc,
            "units": "dimensionless",
            "n_genes": payload["n"],
            "confusion": {
                "tp": payload["tp"], "fp": payload["fp"],
                "tn": payload["tn"], "fn": payload["fn"],
            },
            "config": payload["config"],
            "sweep_wall_s": payload.get("sweep_wall_s"),
        },
        "source": "breuer_2019_elife",
        "source_detail": (
            f"Predictions: {payload.get('_source_file')}."
        ),
        "context": {
            "entity": "syn3a_essentiality_predictor",
            "version": "v8_session_10_scale_0.5_t_1.0",
        },
        "confidence": "measured",
        "caveats": [
            "Higher scale reduces stochastic noise on pool deviations, "
            "which may tighten the FP/TP separation. Whether that "
            "actually lifts MCC above v4's 0.229 is what this "
            "measurement checks.",
            "scale=0.5 is the same scale the existing test_knockouts.py "
            "uses and is close to the biological cell.",
            "If MCC > 0.3 here, the path forward is running this config "
            "at full 458-CDS scope (~2 h wall on the same machine). If "
            "MCC <= 0.2, the architectural diagnosis from Session 9 "
            "also holds at higher scale - simulator biology is the "
            "bottleneck.",
        ],
        "dependencies": ["mcc_against_breuer_v4", "mcc_against_breuer_v7"],
        "last_verified": "2026-04-21",
        "used_by": ["scripts/run_colab_bc.py",
                    "notebooks/colab_bc_sweep.ipynb"],
    }
    out_path.write_text(json.dumps(fact, indent=2))
    try:
        rel = out_path.relative_to(REPO_ROOT)
    except ValueError:
        rel = out_path
    print(f"Wrote {rel}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["b", "c"],
                   choices=["b", "c"],
                   help="Which sweep blocks to run.")
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[42, 1, 2, 3, 4],
                   help="Simulator seeds for the B replicates.")
    p.add_argument("--panel-seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    summary_all: dict[str, dict] = {}

    if "b" in args.configs:
        print("\n### B: multi-seed replicates ###")
        for name, base_args in B_CONFIGS.items():
            payloads = []
            for seed in args.seeds:
                payload = run_sweep(
                    base_args, workers=args.workers,
                    seed=seed, panel_seed=args.panel_seed,
                    out_dir=out_dir, log_tag=f"B:{name}",
                )
                payloads.append(payload)
            summary_all[name] = summarise_replicates(name, payloads)
            mcc = summary_all[name]["mcc"]
            print(f"  -> {name}: MCC = {mcc['mean']:.3f} "
                  f"+/- {mcc['std']:.3f}  (min={mcc['min']:.3f}, "
                  f"max={mcc['max']:.3f})")
        write_replicates_fact(
            summary_all,
            REPO_ROOT / "memory_bank/facts/measured/"
                        "mcc_replicates_summary.json",
        )

    if "c" in args.configs:
        print("\n### C: higher-scale sweep (scale=0.5, t_end=1.0) ###")
        payload = run_sweep(
            C_CONFIG, workers=args.workers,
            seed=42, panel_seed=args.panel_seed,
            out_dir=out_dir, log_tag="C:v8",
        )
        print(f"  -> v8 MCC = {payload['mcc']:.3f}  "
              f"(TP={payload['tp']} FP={payload['fp']} "
              f"TN={payload['tn']} FN={payload['fn']})")
        write_v8_fact(
            payload,
            REPO_ROOT / "memory_bank/facts/measured/"
                        "mcc_against_breuer_v8.json",
        )

    print(f"\nTotal wall: {time.time()-t_total:.1f}s")
    print("Done. Invariant checker next:")
    subprocess.run(
        [sys.executable, "memory_bank/.invariants/check.py"],
        cwd=REPO_ROOT, check=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
