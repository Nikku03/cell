"""Run the genome-wide essentiality sweep using the real simulator.

Produces a CSV at outputs/predictions_real_<scale>_<t_end>_<seed>.csv with
one row per gene plus a sidecar JSON of MCC vs Breuer 2019.

Single-process by default (the existing FastEventSimulator is not thread
safe). For large sweeps, fan out across processes by partitioning the
gene list and merging the CSVs after.

Examples:

  # 4-gene reference panel (~1 minute), default scale + t_end:
  python scripts/run_full_sweep_real.py --reference-panel

  # 50-gene subsample (mixed essential / nonessential per Breuer):
  python scripts/run_full_sweep_real.py --max-genes 50 --balanced

  # Full sweep (hours, single process):
  python scripts/run_full_sweep_real.py --all

Environment knobs:
  RS_SCALE       scale factor (default 0.05; matches RealSimulatorConfig)
  RS_T_END_S     bio time per run (default 0.5)
  RS_DT_S        sample dt (default 0.05)
  RS_SEED        RNG seed (default 42)
  RS_THRESHOLD   ShortWindowDetector threshold (default 0.10)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cell_sim"))

from cell_sim.layer0_genome.genome import Genome
from cell_sim.layer6_essentiality.harness import (
    FailureMode, Prediction, Trajectory,
)
from cell_sim.layer6_essentiality.labels import (
    EssentialityClass, binary_labels, load_breuer2019_labels,
)
from cell_sim.layer6_essentiality.metrics import evaluate_binary
from cell_sim.layer6_essentiality.real_simulator import (
    RealSimulator, RealSimulatorConfig,
)
from cell_sim.layer6_essentiality.short_window_detector import (
    ShortWindowDetector, calibrate_noise_floor,
)


REFERENCE_PANEL = [
    ("JCVISYN3A_0445", "pgi"),    # essential per Breuer
    ("JCVISYN3A_0779", "ptsG"),   # essential
    ("JCVISYN3A_0522", "ftsZ"),   # nonessential
    ("JCVISYN3A_0305", ""),       # nonessential
]


def _select_genes(
    genome: Genome,
    labels: dict,
    args: argparse.Namespace,
) -> list[tuple[str, str]]:
    if args.reference_panel:
        return REFERENCE_PANEL
    eligible = [
        (g.locus_tag, g.gene_name) for g in genome.cds_genes()
        if g.locus_tag in labels
    ]
    if args.balanced and args.max_genes:
        per_class = args.max_genes // 2
        ess = [t for t in eligible
               if labels[t[0]].essentiality == EssentialityClass.ESSENTIAL]
        non = [t for t in eligible
               if labels[t[0]].essentiality == EssentialityClass.NONESSENTIAL]
        rng = random.Random(args.seed)
        rng.shuffle(ess); rng.shuffle(non)
        return ess[:per_class] + non[:per_class]
    if args.max_genes:
        rng = random.Random(args.seed)
        rng.shuffle(eligible)
        return eligible[:args.max_genes]
    return eligible


def _predict(
    sim: RealSimulator,
    detector: ShortWindowDetector,
    locus_tag: str,
    gene_name: str,
    *,
    t_end_s: float,
    sample_dt_s: float,
) -> tuple[Prediction, str]:
    ko_traj = sim.run([locus_tag], t_end_s=t_end_s, sample_dt_s=sample_dt_s)
    mode, t_fail, conf, evidence = detector.detect(ko_traj)
    pred = Prediction(
        locus_tag=locus_tag,
        gene_name=gene_name,
        essential=mode != FailureMode.NONE,
        time_to_failure_s=t_fail,
        failure_mode=mode,
        confidence=conf,
    )
    return pred, evidence


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--reference-panel", action="store_true",
                   help="Run only the 4-gene Breuer reference panel.")
    g.add_argument("--max-genes", type=int,
                   help="Run on a random subset of N genes.")
    g.add_argument("--all", action="store_true",
                   help="Run on all 458 CDS that have Breuer labels.")
    p.add_argument("--balanced", action="store_true",
                   help="With --max-genes, split half essential / half non.")
    p.add_argument("--scale", type=float,
                   default=float(os.environ.get("RS_SCALE", "0.05")))
    p.add_argument("--t-end-s", type=float,
                   default=float(os.environ.get("RS_T_END_S", "0.5")))
    p.add_argument("--dt-s", type=float,
                   default=float(os.environ.get("RS_DT_S", "0.05")))
    p.add_argument("--seed", type=int,
                   default=int(os.environ.get("RS_SEED", "42")))
    p.add_argument("--threshold", type=float,
                   default=float(os.environ.get("RS_THRESHOLD", "0.10")))
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"real_s{args.scale}_t{args.t_end_s}_seed{args.seed}_thr{args.threshold}"
    pred_csv = out_dir / f"predictions_{tag}.csv"
    metrics_json = out_dir / f"metrics_{tag}.json"

    genome = Genome.load()
    labels = load_breuer2019_labels()
    targets = _select_genes(genome, labels, args)
    print(f"Running on {len(targets)} genes; output: {pred_csv}")

    cfg = RealSimulatorConfig(scale_factor=args.scale, seed=args.seed)
    sim = RealSimulator(cfg)

    print("[wt] running wild-type baseline...")
    t0 = time.time()
    wt = sim.run([], t_end_s=args.t_end_s, sample_dt_s=args.dt_s)
    print(f"[wt] {time.time()-t0:.1f}s, {len(wt.samples)} samples")

    detector = ShortWindowDetector(wt=wt, deviation_threshold=args.threshold)
    rows: list[dict] = []
    t_total = time.time()
    for i, (lt, gn) in enumerate(targets, 1):
        t0 = time.time()
        pred, evidence = _predict(
            sim, detector, lt, gn,
            t_end_s=args.t_end_s, sample_dt_s=args.dt_s,
        )
        wall = time.time() - t0
        true_class = labels[lt].essentiality.value if lt in labels else "?"
        print(f"[{i:3d}/{len(targets)}] {lt} {gn:8s} wall={wall:5.1f}s "
              f"true={true_class:13s} pred={'ESS' if pred.essential else 'non':3s} "
              f"mode={pred.failure_mode.value:34s} conf={pred.confidence:.3f} "
              f"{evidence}")
        rows.append({**pred.as_row(), "evidence": evidence,
                     "true_class": true_class})

    with pred_csv.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["locus_tag","gene_name","essential","time_to_failure_s",
                        "failure_mode","confidence","evidence","true_class"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {pred_csv}")

    # MCC
    sub = {r["locus_tag"]: labels[r["locus_tag"]] for r in rows
           if r["locus_tag"] in labels}
    if sub:
        y_true = binary_labels(sub, quasi_as_positive=True)
        y_pred = {r["locus_tag"]: int(r["essential"]) for r in rows
                  if r["locus_tag"] in y_true}
        m = evaluate_binary(y_true, y_pred)
        with metrics_json.open("w") as fh:
            json.dump({"config": vars(args), **m.as_dict()}, fh, indent=2)
        print(f"\n=== MCC vs Breuer 2019 (binary, quasi=positive, n={m.n}) ===")
        print(f"  TP={m.tp} FP={m.fp} TN={m.tn} FN={m.fn}")
        print(f"  MCC={m.mcc:.3f}  precision={m.precision:.3f}  "
              f"recall={m.recall:.3f}  specificity={m.specificity:.3f}")
        print(f"  brief target: > 0.59")
        print(f"\nWrote {metrics_json}")

    print(f"\nTotal wall time: {time.time()-t_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
