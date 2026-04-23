"""Multi-detector Syn3A essentiality sweep.

Extends the existing ``run_full_sweep_real.py`` pipeline by running
ALL three detectors (ShortWindow, PerRule, Ensemble) and the
RedundancyAware detector side-by-side on the same WT + KO
trajectories. Reports a per-detector MCC vs Breuer 2019, so we can
see which detector produces the best signal at current settings.

Rationale: the atom-engine integration made it obvious that
``PerRuleDetector`` on a well-defined gene->rule map is nearly
deterministic, while ``ShortWindowDetector`` on the real simulator
was flagging everything on a single noisy pool signal. Run them
both and let the numbers speak.

Usage::

    python scripts/run_multi_detector_sweep.py --max-genes 30 --balanced
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cell_sim.layer0_genome.genome import Genome
from cell_sim.layer6_essentiality.ensemble_detector import (
    EnsembleDetector,
    EnsemblePolicy,
)
from cell_sim.layer6_essentiality.harness import (
    FailureMode,
    Prediction,
    Trajectory,
)
from cell_sim.layer6_essentiality.labels import (
    EssentialityClass,
    binary_labels,
    load_breuer2019_labels,
)
from cell_sim.layer6_essentiality.metrics import evaluate_binary
from cell_sim.layer6_essentiality.per_rule_detector import PerRuleDetector
from cell_sim.layer6_essentiality.real_simulator import (
    RealSimulator,
    RealSimulatorConfig,
)
from cell_sim.layer6_essentiality.redundancy_aware_detector import (
    RedundancyAwareDetector,
)
from cell_sim.layer6_essentiality.short_window_detector import (
    ShortWindowDetector,
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-genes", type=int, default=30)
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--scale", type=float, default=0.05)
    p.add_argument("--t-end-s", type=float, default=0.5)
    p.add_argument("--dt-s", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sw-threshold", type=float, default=0.25,
                   help="ShortWindow deviation threshold; 0.10 was "
                        "too sensitive, 0.25+ suppresses the uniform "
                        "UNFOLDED_PROTEINS noise floor.")
    p.add_argument("--min-wt-events", type=int, default=10,
                   help="PerRule minimum wild-type event count.")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    genome = Genome.load()

    labels_full = load_breuer2019_labels()
    bin_labels = binary_labels(labels_full, quasi_as_positive=True)

    # Select balanced subset.
    all_cds = [g.locus_tag for g in genome.cds_genes() if g.locus_tag in bin_labels]
    essentials = [g for g in all_cds if bin_labels[g] == 1]
    nonessentials = [g for g in all_cds if bin_labels[g] == 0]
    if args.balanced and args.max_genes:
        import random
        rng = random.Random(args.seed)
        half = args.max_genes // 2
        picked = (rng.sample(essentials, min(half, len(essentials)))
                  + rng.sample(nonessentials, min(half, len(nonessentials))))
        rng.shuffle(picked)
        targets = picked[:args.max_genes]
    else:
        targets = all_cds[:args.max_genes]
    print(f"Targets: {len(targets)} genes "
          f"({sum(bin_labels[g] for g in targets)} essential, "
          f"{len(targets) - sum(bin_labels[g] for g in targets)} non-essential)")

    cfg = RealSimulatorConfig(scale_factor=args.scale, seed=args.seed)
    sim = RealSimulator(cfg)

    print("\n[wt] running wild-type baseline...")
    t0 = time.time()
    wt_traj = sim.run([], t_end_s=args.t_end_s, sample_dt_s=args.dt_s)
    print(f"[wt] {time.time() - t0:.1f} s, {len(wt_traj.samples)} samples")

    # Build gene -> rules map + metabolite production maps from the
    # same simulator so rule names line up with what the trajectories
    # actually emit.
    print("\n[setup] building gene->rules map...")
    gene_to_rules = sim.build_gene_to_rules_map()
    rule_coverage = sum(1 for g in targets if gene_to_rules.get(g))
    print(f"[setup] {rule_coverage}/{len(targets)} targets have "
          f"catalytic rules (PerRule can only score these)")

    from cell_sim.layer6_essentiality.gene_rule_map import (
        build_metabolite_producers,
        build_rule_products,
    )
    sim._ensure_setup()      # private but needed to access cached rules
    rules_all = list(sim._rev_rules or []) + list(sim._extra_rules or [])
    metabolite_producers = build_metabolite_producers(rules_all)
    rule_products = build_rule_products(rules_all)

    sw = ShortWindowDetector(
        wt=wt_traj, deviation_threshold=args.sw_threshold,
    )
    pr = PerRuleDetector(
        wt=wt_traj, gene_to_rules=gene_to_rules,
        min_wt_events=args.min_wt_events,
    )
    ens_and = EnsembleDetector(
        short_window=sw, per_rule=pr, policy=EnsemblePolicy.AND,
    )
    ens_pr_pool = EnsembleDetector(
        short_window=sw, per_rule=pr,
        policy=EnsemblePolicy.PER_RULE_WITH_POOL_CONFIRM,
    )
    ra = RedundancyAwareDetector(
        wt=wt_traj, gene_to_rules=gene_to_rules,
        metabolite_producers=metabolite_producers,
        rule_products=rule_products,
    )

    preds = {
        "short_window": {},
        "per_rule": {},
        "ensemble_and": {},
        "ensemble_per_rule_with_pool": {},
        "redundancy_aware": {},
    }

    print("\n[sweep] running knockouts...")
    t_sweep = time.time()
    for i, lt in enumerate(targets, 1):
        t0 = time.time()
        ko_traj = sim.run((lt,), t_end_s=args.t_end_s, sample_dt_s=args.dt_s)
        # Each detector returns (mode, ...). Essential iff mode != NONE.
        sw_out = sw.detect(ko_traj)
        pr_out = pr.detect_for_gene(lt, ko_traj)
        eand_out = ens_and.detect_for_gene(lt, ko_traj)
        eprp_out = ens_pr_pool.detect_for_gene(lt, ko_traj)
        ra_out = ra.detect_for_gene(lt, ko_traj)
        preds["short_window"][lt] = int(sw_out[0] != FailureMode.NONE)
        preds["per_rule"][lt] = int(pr_out[0] != FailureMode.NONE)
        preds["ensemble_and"][lt] = int(eand_out[0] != FailureMode.NONE)
        preds["ensemble_per_rule_with_pool"][lt] = int(
            eprp_out[0] != FailureMode.NONE
        )
        preds["redundancy_aware"][lt] = int(ra_out[0] != FailureMode.NONE)
        true = bin_labels.get(lt, "?")
        wall = time.time() - t0
        print(f"  [{i:2d}/{len(targets)}] {lt:20s} "
              f"true={true} sw={preds['short_window'][lt]} "
              f"pr={preds['per_rule'][lt]} "
              f"e_and={preds['ensemble_and'][lt]} "
              f"e_pr_pool={preds['ensemble_per_rule_with_pool'][lt]} "
              f"ra={preds['redundancy_aware'][lt]} "
              f"({wall:.1f} s)")
    print(f"[sweep] {time.time() - t_sweep:.1f} s for {len(targets)} genes")

    print("\n=== MCC per detector ===")
    metrics_out = {}
    for name, pred_dict in preds.items():
        y_true = {g: bin_labels[g] for g in pred_dict if g in bin_labels}
        y_pred = {g: pred_dict[g] for g in y_true}
        m = evaluate_binary(y_true, y_pred)
        metrics_out[name] = m.as_dict()
        print(f"  {name:30s} MCC={m.mcc:+.3f} acc={m.accuracy:.3f} "
              f"tp={m.tp} fp={m.fp} tn={m.tn} fn={m.fn} "
              f"n={m.n}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "config": vars(args),
            "n_targets": len(targets),
            "targets": targets,
            "rule_coverage": rule_coverage,
            "labels": {g: int(v) for g, v in
                       [(g, bin_labels.get(g, -1)) for g in targets]},
            "predictions": preds,
            "metrics": metrics_out,
        }, indent=2))
        print(f"\nsummary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
