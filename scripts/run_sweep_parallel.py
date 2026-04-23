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


def _worker_init(cfg_dict: dict, wt_pickle_path: str,
                 gene_to_rules: dict | None = None) -> None:
    """Per-process setup. Each worker builds its own RealSimulator and
    loads the shared WT trajectory from disk."""
    global _worker_sim, _worker_detector, _worker_cfg
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    _worker_cfg = cfg_dict
    rs_cfg = RealSimulatorConfig(
        scale_factor=cfg_dict["scale"],
        seed=cfg_dict["seed"],
        use_rust_backend=cfg_dict.get("use_rust_backend", False),
        enable_metabolite_sinks=cfg_dict.get("enable_sinks", False),
        sink_k_per_s=cfg_dict.get("sink_k_per_s", 100.0),
        sink_tolerance=cfg_dict.get("sink_tolerance", 3.0),
    )
    _worker_sim = RealSimulator(rs_cfg)
    with open(wt_pickle_path, "rb") as fh:
        wt = pickle.load(fh)

    detector_kind = cfg_dict.get("detector", "short-window")
    if detector_kind == "redundancy-aware":
        from cell_sim.layer6_essentiality.redundancy_aware_detector import (
            RedundancyAwareDetector,
        )
        from cell_sim.layer6_essentiality.gene_rule_map import (
            build_metabolite_producers, build_rule_products,
        )
        assert gene_to_rules is not None, \
            "redundancy-aware detector requires gene_to_rules map"
        # Worker rebuilds producers/products map from its own RealSimulator.
        _worker_sim._ensure_setup()
        all_rules = list(_worker_sim._rev_rules or []) + \
                    list(_worker_sim._extra_rules or [])
        metabolite_producers = build_metabolite_producers(all_rules)
        rule_products = build_rule_products(all_rules)
        _worker_detector = RedundancyAwareDetector(
            wt=wt,
            gene_to_rules=gene_to_rules,
            metabolite_producers=metabolite_producers,
            rule_products=rule_products,
            min_wt_production=cfg_dict.get("redundancy_min_wt_production", 20),
            drop_threshold=cfg_dict.get("redundancy_drop_threshold", 0.30),
        )
    elif detector_kind == "per-rule":
        from cell_sim.layer6_essentiality.per_rule_detector import (
            PerRuleDetector,
        )
        assert gene_to_rules is not None, \
            "per-rule detector requires gene_to_rules map"
        _worker_detector = PerRuleDetector(
            wt=wt,
            gene_to_rules=gene_to_rules,
            min_wt_events=cfg_dict.get("min_wt_events", 20),
        )
    elif detector_kind == "ensemble":
        from cell_sim.layer6_essentiality.per_rule_detector import (
            PerRuleDetector,
        )
        from cell_sim.layer6_essentiality.short_window_detector import (
            ShortWindowDetector,
        )
        from cell_sim.layer6_essentiality.ensemble_detector import (
            EnsembleDetector, EnsemblePolicy,
        )
        assert gene_to_rules is not None, \
            "ensemble detector requires gene_to_rules map"
        pr = PerRuleDetector(
            wt=wt,
            gene_to_rules=gene_to_rules,
            min_wt_events=cfg_dict.get("min_wt_events", 20),
        )
        sw = ShortWindowDetector(
            wt=wt,
            deviation_threshold=cfg_dict["threshold_payload"],
            fallback_threshold=cfg_dict["threshold"],
        )
        _worker_detector = EnsembleDetector(
            per_rule=pr,
            short_window=sw,
            policy=EnsemblePolicy(cfg_dict.get("ensemble_policy",
                                               "per_rule_with_pool_confirm")),
            min_confidence=cfg_dict.get("min_confidence", 0.15),
            min_pool_dev=cfg_dict.get("min_pool_dev", 0.02),
        )
    elif detector_kind == "composed":
        # ComplexAssemblyDetector OR'd with PerRule. Structural signal
        # catches ribosomal / tRNA-synthetase / translation-factor
        # essentials that the metabolic detectors cannot see; the
        # trajectory detector handles metabolic essentials.
        from cell_sim.layer6_essentiality.per_rule_detector import (
            PerRuleDetector,
        )
        from cell_sim.layer6_essentiality.complex_assembly_detector import (
            ComplexAssemblyDetector,
        )
        from cell_sim.layer6_essentiality.composed_detector import (
            ComposedDetector,
        )
        assert gene_to_rules is not None, \
            "composed detector requires gene_to_rules map for its " \
            "trajectory sub-detector"
        pr = PerRuleDetector(
            wt=wt,
            gene_to_rules=gene_to_rules,
            min_wt_events=cfg_dict.get("min_wt_events", 20),
        )
        _worker_detector = ComposedDetector(
            structural=ComplexAssemblyDetector(),
            trajectory=pr,
        )
    else:
        from cell_sim.layer6_essentiality.short_window_detector import (
            ShortWindowDetector,
        )
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
    detector_kind = _worker_cfg.get("detector")
    if detector_kind in ("per-rule", "ensemble", "redundancy-aware",
                          "composed"):
        mode, t_fail, conf, evidence = _worker_detector.detect_for_gene(lt, ko)
    else:
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
        use_rust_backend=cfg_dict.get("use_rust_backend", False),
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
        use_rust_backend=cfg_dict.get("use_rust_backend", False),
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
    panel_seed = args.panel_seed if args.panel_seed is not None else args.seed
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
        rng = random.Random(panel_seed)
        rng.shuffle(ess); rng.shuffle(non)
        return ess[:per] + non[:per]
    if args.max_genes:
        rng = random.Random(panel_seed)
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
    p.add_argument("--panel-seed", type=int, default=None,
                   help="Separate seed for gene-panel selection. If "
                        "unset, defaults to --seed. Use this to vary "
                        "the simulator RNG across replicates while "
                        "holding the gene panel fixed.")
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--calibrate", type=int, default=0)
    p.add_argument("--safety-factor", type=float, default=2.0)
    p.add_argument("--use-rust", action="store_true",
                   help="Use the Rust-backed FastEventSimulator "
                        "(cell_sim_rust). ~2x speedup at scale=0.05.")
    p.add_argument("--detector",
                   choices=["short-window", "per-rule", "ensemble",
                            "redundancy-aware", "composed"],
                   default="short-window",
                   help="short-window: pool-deviation (v0-v4). "
                        "per-rule: per-rule event counts (v5). "
                        "ensemble: compose both detectors. "
                        "redundancy-aware (v9+): like per-rule but "
                        "abstains when silenced products are still "
                        "produced by alternate catalysis rules. "
                        "composed (v10+): OR the known-complex-subunit "
                        "structural signal with per-rule; catches "
                        "ribosomal / translation-machinery essentials "
                        "that metabolic detectors cannot see.")
    p.add_argument("--min-wt-events", type=int, default=20,
                   help="PerRuleDetector: minimum WT event count per "
                        "rule before it counts as 'should be firing'.")
    p.add_argument("--ensemble-policy",
                   choices=["and", "or_high_confidence",
                            "per_rule_with_pool_confirm"],
                   default="per_rule_with_pool_confirm")
    p.add_argument("--min-confidence", type=float, default=0.15)
    p.add_argument("--min-pool-dev", type=float, default=0.02)
    p.add_argument("--rule-necessity-only", action="store_true",
                   help="Restrict per-rule/ensemble detector to rules "
                        "with no alternate catalyser (addresses the v5 "
                        "false-positive mechanism).")
    p.add_argument("--redundancy-drop-threshold", type=float, default=0.30,
                   help="redundancy-aware only: KO trips only if a "
                        "product's total production drops below this "
                        "fraction of WT.")
    p.add_argument("--redundancy-min-wt-production", type=int, default=20,
                   help="redundancy-aware only: minimum WT total "
                        "production events for a product to count.")
    p.add_argument("--enable-sinks", action="store_true",
                   help="Add first-order metabolite sinks that drain "
                        "pools above tolerance*initial. Addresses v7 "
                        "transporter-KO blowups.")
    p.add_argument("--sink-k-per-s", type=float, default=100.0)
    p.add_argument("--sink-tolerance", type=float, default=3.0)
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_tag = f"_cal{args.calibrate}sf{args.safety_factor}" if args.calibrate else ""
    det_tag = f"_{args.detector}"
    if args.detector == "ensemble":
        det_tag += f"-{args.ensemble_policy}"
    if args.rule_necessity_only:
        det_tag += "_uniqueonly"
    tag = (f"parallel_s{args.scale}_t{args.t_end_s}_seed{args.seed}"
           f"_thr{args.threshold}_w{args.workers}{cal_tag}{det_tag}")
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
        "use_rust_backend": args.use_rust,
        "detector": args.detector,
        "min_wt_events": args.min_wt_events,
        "ensemble_policy": args.ensemble_policy,
        "min_confidence": args.min_confidence,
        "min_pool_dev": args.min_pool_dev,
        "redundancy_drop_threshold": args.redundancy_drop_threshold,
        "redundancy_min_wt_production": args.redundancy_min_wt_production,
        "enable_sinks": args.enable_sinks,
        "sink_k_per_s": args.sink_k_per_s,
        "sink_tolerance": args.sink_tolerance,
    }

    t_setup = time.time()
    print("[wt] computing wild-type baseline...")
    wt = _compute_wt(cfg_dict)
    print(f"[wt] wall: {time.time()-t_setup:.1f}s")

    calibration_floor = None
    per_pool_thresholds = None
    gene_to_rules = None
    gene_to_rules_summary = None
    needs_gene_map = args.detector in (
        "per-rule", "ensemble", "redundancy-aware", "composed",
    )
    if needs_gene_map:
        from cell_sim.layer6_essentiality.real_simulator import (
            RealSimulator, RealSimulatorConfig,
        )
        from cell_sim.layer6_essentiality.gene_rule_map import (
            summarise, unique_rules_per_gene,
        )
        print(f"[{args.detector}] building gene -> rules map...")
        setup_sim = RealSimulator(RealSimulatorConfig(
            scale_factor=cfg_dict["scale"], seed=cfg_dict["seed"],
            use_rust_backend=cfg_dict.get("use_rust_backend", False),
        ))
        gene_to_rules = setup_sim.build_gene_to_rules_map()
        if args.rule_necessity_only:
            before = len(gene_to_rules)
            gene_to_rules = unique_rules_per_gene(gene_to_rules)
            print(f"[{args.detector}] rule-necessity-only: {before} -> "
                  f"{len(gene_to_rules)} genes with uniquely-required rules")
        gene_to_rules_summary = summarise(gene_to_rules)
        print(f"[{args.detector}] {gene_to_rules_summary}")

    if args.detector == "ensemble" and args.calibrate > 0:
        thresholds, calibration_floor = _calibrate_thresholds(
            cfg_dict, wt, labels, args.calibrate,
        )
        per_pool_thresholds = thresholds
        cfg_dict["threshold_payload"] = thresholds
        print(f"[calibrate] ensemble per-pool thresholds: " + ", ".join(
            f"{k}:{v:.3f}" for k, v in sorted(thresholds.items())
        ))
    elif args.detector == "per-rule" and args.calibrate > 0:
        print("[per-rule] --calibrate ignored (no threshold tuning needed)")
    elif args.calibrate > 0:
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
        initargs=(cfg_dict, str(wt_pickle), gene_to_rules),
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
        if gene_to_rules_summary is not None:
            payload["gene_to_rules_summary"] = gene_to_rules_summary
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
