"""Benchmark feature-assembly speed — parquet load + join.

The Tier-1 predictor's feature pipeline does three things per run:
read the three parquets from disk, join them index-on-index, and
materialize a dense numpy matrix for XGBoost. At 455 rows × 1295
float32 columns the matrix is small (~2.4 MiB), but the I/O +
join pattern is exercised every time anyone retrains, and on a
multi-organism scale-up (~10k rows, many more columns) the wall
time starts to matter.

This benchmark compares:

  * **pandas + pyarrow** — the current stack
  * **polars** — rust-backed dataframe library with a native
    parquet reader

Both libraries get 10 iterations over the same synthetic feature
matrix (455 × 1295 float32, 5% NaN to mimic MACE coverage) so
small timing variations don't bias the comparison. The benchmark
rewrites the parquet once up-front; iterations read-and-join
fresh each time, matching the pattern in
``scripts/train_tier1_xgboost.py``.

Usage::

    python scripts/bench_feature_assembly.py \\
        --out outputs/bench_feature_assembly.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


# Matches the Tier-1 shape.
N_ROWS = 455
ESM_COLS = 1280
EF_COLS = 8
MACE_COLS = 7
NAN_FRACTION = 0.05
N_ITER = 10


def _make_synthetic_parquets(tmp: Path, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    loci = [f"SYN_{i:04d}" for i in range(N_ROWS)]

    def _mk(name: str, ncols: int, prefix: str, nan_rate: float) -> Path:
        arr = rng.standard_normal(
            (N_ROWS, ncols), dtype=np.float32,
        )
        mask = rng.random((N_ROWS, ncols)) < nan_rate
        arr[mask] = np.nan
        cols = [f"{prefix}_{i}" for i in range(ncols)]
        df = pd.DataFrame(arr, columns=cols, index=loci)
        df.index.name = "locus_tag"
        path = tmp / f"{name}.parquet"
        df.to_parquet(path)
        return path

    return {
        "esm": _mk("synth_esm2", ESM_COLS, "esm", 0.0),
        "ef":  _mk("synth_esmfold", EF_COLS, "ef", 0.0),
        # MACE-style: ~75% of rows are fully-NaN padded
        "mace": _mk("synth_mace", MACE_COLS, "mace", 0.75),
    }


def _bench_pandas(paths: dict) -> float:
    t = time.perf_counter()
    esm = pd.read_parquet(paths["esm"])
    ef = pd.read_parquet(paths["ef"])
    mace = pd.read_parquet(paths["mace"])
    X = esm.join(ef, how="outer").join(mace, how="outer")
    # Force materialization so we don't time a lazy plan
    _ = X.values.shape
    return time.perf_counter() - t


def _bench_polars(paths: dict) -> float:
    import polars as pl  # type: ignore

    t = time.perf_counter()
    esm = pl.read_parquet(paths["esm"])
    ef = pl.read_parquet(paths["ef"])
    mace = pl.read_parquet(paths["mace"])
    # polars reads index as a column ("locus_tag") — join on it.
    X = (
        esm.join(ef, on="locus_tag", how="full", coalesce=True)
           .join(mace, on="locus_tag", how="full", coalesce=True)
    )
    _ = X.to_numpy().shape
    return time.perf_counter() - t


def _bench_fastparquet(paths: dict) -> float | None:
    try:
        import fastparquet  # type: ignore  # noqa: F401
    except ImportError:
        return None
    t = time.perf_counter()
    esm = pd.read_parquet(paths["esm"], engine="fastparquet")
    ef = pd.read_parquet(paths["ef"], engine="fastparquet")
    mace = pd.read_parquet(paths["mace"], engine="fastparquet")
    X = esm.join(ef, how="outer").join(mace, how="outer")
    _ = X.values.shape
    return time.perf_counter() - t


def _sample(fn, paths: dict, n_iter: int, warmup: int = 2) -> dict:
    times: list[float] = []
    for _ in range(warmup):
        fn(paths)
    for _ in range(n_iter):
        times.append(fn(paths))
    return {
        "n_iter": n_iter,
        "mean_s": statistics.fmean(times),
        "median_s": statistics.median(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_s": min(times),
        "max_s": max(times),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "outputs/bench_feature_assembly.json",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-iter", type=int, default=N_ITER)
    args = ap.parse_args()

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        print(f"building synthetic feature parquets in {tmp}")
        paths = _make_synthetic_parquets(tmp, args.seed)
        for name, p in paths.items():
            print(f"  {name:4s}  {p}  {p.stat().st_size/1024:.1f} KiB")

        results: dict = {}

        print("\n[pandas + pyarrow]")
        pandas_res = _sample(_bench_pandas, paths, args.n_iter)
        print(f"  mean={pandas_res['mean_s']*1000:.2f} ms  "
              f"median={pandas_res['median_s']*1000:.2f} ms  "
              f"stdev={pandas_res['stdev_s']*1000:.2f} ms")
        results["pandas_pyarrow"] = pandas_res

        print("\n[polars]")
        try:
            polars_res = _sample(_bench_polars, paths, args.n_iter)
            print(f"  mean={polars_res['mean_s']*1000:.2f} ms  "
                  f"median={polars_res['median_s']*1000:.2f} ms  "
                  f"stdev={polars_res['stdev_s']*1000:.2f} ms")
            results["polars"] = polars_res
            speedup = pandas_res["mean_s"] / max(
                1e-9, polars_res["mean_s"],
            )
            results["polars_speedup_factor"] = speedup
            print(f"  speedup vs pandas: {speedup:.2f}x")
        except ImportError:
            results["polars"] = "not_installed"
            print("  polars not installed — skipping")
        except Exception as exc:  # noqa: BLE001
            results["polars_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  polars run failed: {exc}")

        print("\n[pandas + fastparquet]")
        fp_time = _bench_fastparquet(paths)
        if fp_time is None:
            print("  fastparquet not installed — skipping")
            results["pandas_fastparquet"] = "not_installed"
        else:
            results["pandas_fastparquet"] = {"single_run_s": fp_time}
            print(f"  single run: {fp_time*1000:.2f} ms")

    results["config"] = {
        "n_rows": N_ROWS,
        "esm_cols": ESM_COLS,
        "ef_cols": EF_COLS,
        "mace_cols": MACE_COLS,
        "nan_fraction": NAN_FRACTION,
        "n_iter": args.n_iter,
        "seed": args.seed,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
