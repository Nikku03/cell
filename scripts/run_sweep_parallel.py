"""Multi-process genome-wide essentiality sweep.

Each worker holds its own ``RealSimulator`` instance (the underlying
``FastEventSimulator`` is not thread-safe but is process-safe). The WT
baseline is computed once in worker 0 and broadcast to peers via a
shared on-disk pickle file to avoid recomputing.

Outputs the same ``predictions_*.csv`` and ``metrics_*.json`` shape as
``run_full_sweep_real.py``, so downstream tooling is identical.

Typical usage:

  # Full 458-CDS sweep on 4 workers (~25 min on this sandbox):
  python scripts/run_sweep_parallel.py --all --workers 4

  # Balanced subset:
  python scripts/run_sweep_parallel.py --max-genes 100 --balanced \
         --workers 4 --t-end-s 0.5 --scale 0.05 --threshold 0.05

  # With noise-floor calibration:
  python scripts/run_sweep_parallel.py --all --workers 4 --calibrate 10
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cell_sim"))

from cell_sim.layer0_genome.genome import Genome
from cell_sim.layer6_essentiality.harness import FailureMode, Trajectory
from cell_sim.layer6_essentiality.labels import (
    EssentialityClass, binary_labels, load_breuer2019_labels,
)
from cell_sim.layer6_essentiality.metrics import evaluate_binary


# ---- worker state ------------------------------------------------


_worker_sim = None
_worker_detector = None
_worker_cfg: dict = {}


def _worker_init(cfg_dict: dict, wt_pickle_path: str) -> None:
    """Per-process setup. Each worker builds its own RealSimulator and
    loads the shared WT trajectory from disk."""
    global _worker_sim, _worker_detector, _worker_cfg
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    from cell_sim.layer6_essentiality.short_window_detector import (
        ShortWindowDetector,
    )
    _worker_cfg = cfg_dict
    rs_cfg = RealSimulatorConfig(
        scale_factor=cfg_dict["scale"],
        seed=cfg_dict["seed"],
    )
    _worker_sim = RealSimulator(rs_cfg)
    with open(wt_pickle_path, "rb") as fh:
        wt = pickle.load(fh)
    _worker_detector = ShortWindowDetector(
        wt=wt,
        deviation_threshold=cfg_dict["threshold_payload"],
        fallback_threshold=cfg_dict["threshold"],
    )


def _worker_predict(item: tuple[str, str]) -> dict:
    from cell_sim.layer6_essentiality.harness import Prediction
    lt, gn = item
    t0 = time.time()
    ko = _worker_sim.run(
        [lt],
        t_end_s=_worker_cfg["t_end_s"],
        sample_dt_s=_worker_cfg["dt_s"],
    )
    mode, t_fail, conf, evidence = _worker_detector.detect(ko)
    wall = time.time() - t0
    pred = Prediction(
        locus_tag=lt, gene_name=gn,
        essential=mode != FailureMode.NONE,
        time_to_failure_s=t_fail,
        failure_mode=mode,
        confidence=conf,
    )
    return {**pred.as_row(), "evidence": evidence, "wall": wall}


# ---- driver ------------------------------------------------------


def _compute_wt(cfg_dict: dict) -> Trajectory:
    """Single-threaded WT baseline."""
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(
        scale_factor=cfg_dict["scale"], seed=cfg_dict["seed"],
    ))
    return sim.run([], t_end_s=cfg_dict["t_end_s"],
                   sample_dt_s=cfg_dict["dt_s"])


def _calibrate_thresholds(
    cfg_dict: dict, wt: Trajectory, labels: dict, k: int,
) -> tuple[dict[str, float], dict[str, float]]:
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    from cell_sim.layer6_essentiality.short_window_detector import (
        calibrate_noise_floor,
    )
    sim = RealSimulator(RealSimulatorConfig(
        scale_factor=cfg_dict["scale"], seed=cfg_dict["seed"],
    ))
    nons = [lt for lt, lab in labels.items()
            if lab.essentiality == EssentialityClass.NONESSENTIAL]
    rng = random.Random(cfg_dict["seed"] ^ 0xC0DE)
    rng.shuffle(nons)
    pick = nons[:k]
    print(f"[calibrate] running {len(pick)} non-essential KOs serially...")
    trajs = []
    for i, lt in enumerate(pick, 1):
        t0 = time.time()
        trajs.append(sim.run([lt], t_end_s=cfg_dict["t_end_s"],
                             sample_dt_s=cfg_dict["dt_s"]))
        print(f"[calibrate] {i}/{len(pick)} {lt} wall={time.time()-t0:.1f}s")
    floor = calibrate_noise_floor(wt, trajs)
    safety = cfg_dict["safety_factor"]
    floor_thr = cfg_dict["threshold"]
    thresholds = {
        p: max(floor[p] * safety, floor_thr) for p in floor
    }
    return thresholds, floor


def _select_genes(genome, labels, args):
    if args.reference_panel:
        return [
            ("JCVISYN3A_0445", "pgi"),
            ("JCVISYN3A_0779", "ptsG"),
            ("JCVISYN3A_0522", "ftsZ"),
            ("JCVISYN3A_0305", ""),
        ]
    eligible = [
        (g.locus_tag, g.gene_name) for g in genome.cds_genes()
        if g.locus_tag in labels
    ]
    if args.balanced and args.max_genes:
        per = args.max_genes // 2
        ess = [t for t in eligible
               if labels[t[0]].essentiality == EssentialityClass.ESSENTIAL]
        non = [t for t in eligible
               if labels[t[0]].essentiality == EssentialityClass.NONESSENTIAL]
        rng = random.Random(args.seed)
        rng.shuffle(ess); rng.shuffle(non)
        return ess[:per] + non[:per]
    if args.max_genes:
        rng = random.Random(args.seed)
        rng.shuffle(eligible)
        return eligible[:args.max_genes]
    return eligible


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--reference-panel", action="store_true")
    g.add_argument("--max-genes", type=int)
    g.add_argument("--all", action="store_true")
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--workers", type=int, default=mp.cpu_count())
    p.add_argument("--scale", type=float, default=0.05)
    p.add_argument("--t-end-s", type=float, default=0.5)
    p.add_argument("--dt-s", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--calibrate", type=int, default=0)
    p.add_argument("--safety-factor", type=float, default=2.0)
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_tag = f"_cal{args.calibrate}sf{args.safety_factor}" if args.calibrate else ""
    tag = (f"parallel_s{args.scale}_t{args.t_end_s}_seed{args.seed}"
           f"_thr{args.threshold}_w{args.workers}{cal_tag}")
    pred_csv = out_dir / f"predictions_{tag}.csv"
    metrics_json = out_dir / f"metrics_{tag}.json"

    genome = Genome.load()
    labels = load_breuer2019_labels()
    targets = _select_genes(genome, labels, args)
    print(f"Running on {len(targets)} genes with {args.workers} workers")
    print(f"  scale={args.scale} t_end={args.t_end_s} threshold={args.threshold}")
    print(f"  output: {pred_csv}")

    cfg_dict = {
        "scale": args.scale, "seed": args.seed,
        "t_end_s": args.t_end_s, "dt_s": args.dt_s,
        "threshold": args.threshold,
        "threshold_payload": args.threshold,  # becomes dict after calibrate
        "safety_factor": args.safety_factor,
    }

    t_setup = time.time()
    print("[wt] computing wild-type baseline...")
    wt = _compute_wt(cfg_dict)
    print(f"[wt] wall: {time.time()-t_setup:.1f}s")

    calibration_floor = None
    per_pool_thresholds = None
    if args.calibrate > 0:
        thresholds, calibration_floor = _calibrate_thresholds(
            cfg_dict, wt, labels, args.calibrate,
        )
        per_pool_thresholds = thresholds
        cfg_dict["threshold_payload"] = thresholds
        print(f"[calibrate] noise floor: " + ", ".join(
            f"{k}:{v:.3f}" for k, v in sorted(calibration_floor.items())
        ))
        print(f"[calibrate] thresholds:  " + ", ".join(
            f"{k}:{v:.3f}" for k, v in sorted(thresholds.items())
        ))

    # Persist WT for workers.
    wt_pickle = out_dir / f".wt_{tag}.pkl"
    with wt_pickle.open("wb") as fh:
        pickle.dump(wt, fh)

    # Fan out.
    rows: list[dict] = []
    t_sweep = time.time()
    ctx = mp.get_context("spawn")  # avoid CoW copying large memory
    with ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(cfg_dict, str(wt_pickle)),
    ) as pool:
        for i, row in enumerate(pool.imap_unordered(_worker_predict, targets), 1):
            lt = row["locus_tag"]
            true_class = labels[lt].essentiality.value if lt in labels else "?"
            print(f"[{i:3d}/{len(targets)}] {lt} wall={row['wall']:5.1f}s "
                  f"true={true_class:13s} pred={'ESS' if row['essential'] else 'non':3s} "
                  f"mode={row['failure_mode']:34s} conf={row['confidence']} "
                  f"{row['evidence']}")
            rows.append({**{k: v for k, v in row.items() if k != "wall"},
                         "true_class": true_class})

    wt_pickle.unlink(missing_ok=True)
    sweep_wall = time.time() - t_sweep
    print(f"\nSweep wall: {sweep_wall:.1f}s ({sweep_wall/max(1,len(targets)):.1f}s/gene effective)")

    # CSV
    with pred_csv.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["locus_tag","gene_name","essential","time_to_failure_s",
                        "failure_mode","confidence","evidence","true_class"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {pred_csv}")

    # MCC
    sub = {r["locus_tag"]: labels[r["locus_tag"]] for r in rows
           if r["locus_tag"] in labels}
    if sub:
        y_true = binary_labels(sub, quasi_as_positive=True)
        y_pred = {r["locus_tag"]: int(r["essential"]) for r in rows
                  if r["locus_tag"] in y_true}
        m = evaluate_binary(y_true, y_pred)
        payload = {"config": vars(args), **m.as_dict(),
                   "sweep_wall_s": sweep_wall}
        if calibration_floor is not None:
            payload["calibration_floor"] = calibration_floor
            payload["per_pool_thresholds"] = per_pool_thresholds
        with metrics_json.open("w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n=== MCC vs Breuer 2019 (binary, quasi=positive, n={m.n}) ===")
        print(f"  TP={m.tp} FP={m.fp} TN={m.tn} FN={m.fn}")
        print(f"  MCC={m.mcc:.3f}  precision={m.precision:.3f}  "
              f"recall={m.recall:.3f}  specificity={m.specificity:.3f}")
        print(f"  brief target: > 0.59")
    return 0


if __name__ == "__main__":
    sys.exit(main())
