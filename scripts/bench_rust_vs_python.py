"""Benchmark the Gillespie backend speedup: Rust vs pure Python.

Runs 20 Syn3A gene knockouts end-to-end through the
``RealSimulator.run`` pipeline twice — once with the Rust backend
enabled, once forced to the Python ``FastEventSimulator``. Reports
per-gene mean / median / std wall time and the aggregate speedup
factor.

The measurement is the Gillespie hot-path only, not the full sweep
orchestration (no WT pre-compute, no detector, no CSV write). That
keeps the signal clean — every second in the measured block is the
state-trajectory kernel, which is what the Rust port targets.

Usage::

    python scripts/bench_rust_vs_python.py \\
        --out outputs/bench_rust_python.json

Non-goals: no integration change, no config tweak, no MCC claim. If
the Rust backend can't import (missing pyo3 binary), reports the
failure honestly and still emits the Python numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cell_sim"))


def _detect_hardware() -> dict:
    import multiprocessing as mp
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": mp.cpu_count(),
        "machine": platform.machine(),
    }
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        pass
    return info


def _is_rust_available() -> bool:
    try:
        from layer2_field.rust_dynamics import (  # noqa: F401
            RustBackedFastEventSimulator,
        )
        return True
    except ImportError:
        return False


def _pick_genes(n: int, seed: int) -> list[str]:
    """Pick N gene loci deterministically from the Breuer label set.

    Uses the label file so the same genes are selected across runs;
    shuffles once with the session seed for a non-adversarial sample.
    """
    from cell_sim.layer6_essentiality.labels import load_breuer2019_labels
    labels = load_breuer2019_labels()
    loci = sorted(labels.keys())
    import random
    rng = random.Random(seed)
    rng.shuffle(loci)
    return loci[:n]


def _run_one_gene(locus: str, use_rust: bool, t_end_s: float,
                  dt_s: float, scale: float, seed: int) -> float:
    """Run a single knockout, return wall time in seconds."""
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(
        scale_factor=scale, seed=seed,
        use_rust_backend=use_rust,
        enable_imb155_patches=False,
    ))
    t0 = time.perf_counter()
    sim.run([locus], t_end_s=t_end_s, sample_dt_s=dt_s)
    return time.perf_counter() - t0


def _bench_mode(
    genes: list[str], *, use_rust: bool, t_end_s: float,
    dt_s: float, scale: float, seed: int,
) -> dict:
    walls: list[float] = []
    label = "rust" if use_rust else "python"
    print(f"\n[{label}] benchmarking {len(genes)} genes "
          f"(scale={scale} t_end={t_end_s} dt={dt_s} seed={seed})")
    t_total = time.perf_counter()
    for i, lt in enumerate(genes, 1):
        w = _run_one_gene(lt, use_rust, t_end_s, dt_s, scale, seed)
        walls.append(w)
        print(f"  [{i:2d}/{len(genes)}] {lt}  wall={w:.3f}s")
    total = time.perf_counter() - t_total
    return {
        "mode": label,
        "n_genes": len(genes),
        "mean_s": statistics.fmean(walls),
        "median_s": statistics.median(walls),
        "stdev_s": (
            statistics.stdev(walls) if len(walls) > 1 else 0.0
        ),
        "min_s": min(walls),
        "max_s": max(walls),
        "total_s": total,
        "per_gene_s": walls,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "outputs/bench_rust_python.json",
    )
    ap.add_argument("--n-genes", type=int, default=20)
    ap.add_argument("--t-end-s", type=float, default=0.5)
    ap.add_argument("--dt-s", type=float, default=0.05)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hw = _detect_hardware()
    print(f"hardware: {hw.get('cpu_model','?')}  "
          f"cores={hw['cpu_count']}  python={hw['python']}")
    rust_ok = _is_rust_available()
    print(f"Rust backend available: {rust_ok}")

    genes = _pick_genes(args.n_genes, args.seed)
    print(f"picked {len(genes)} genes (seed={args.seed}): "
          f"{genes[:3]}...{genes[-3:]}")

    common = dict(
        t_end_s=args.t_end_s, dt_s=args.dt_s,
        scale=args.scale, seed=args.seed,
    )

    py_res = _bench_mode(genes, use_rust=False, **common)

    if rust_ok:
        rust_res = _bench_mode(genes, use_rust=True, **common)
        speedup = py_res["mean_s"] / max(1e-9, rust_res["mean_s"])
        print(f"\n== summary ==")
        print(f"python  mean={py_res['mean_s']:.3f}s  "
              f"median={py_res['median_s']:.3f}s  "
              f"stdev={py_res['stdev_s']:.3f}s")
        print(f"rust    mean={rust_res['mean_s']:.3f}s  "
              f"median={rust_res['median_s']:.3f}s  "
              f"stdev={rust_res['stdev_s']:.3f}s")
        print(f"speedup = {speedup:.2f}x")
    else:
        rust_res = None
        speedup = None
        print("\nRust backend not importable — only Python numbers reported.")

    out = {
        "config": {
            "n_genes": args.n_genes,
            "t_end_s": args.t_end_s,
            "dt_s": args.dt_s,
            "scale": args.scale,
            "seed": args.seed,
            "genes": genes,
        },
        "hardware": hw,
        "rust_available": rust_ok,
        "python": py_res,
        "rust": rust_res,
        "speedup_factor": speedup,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
