"""Run the atom-engine essentiality sweep through existing Layer-6 detectors.

Wild-type run + one knockout run per bondable element-pair. Feeds the
resulting Trajectories into ``ShortWindowDetector``, ``PerRuleDetector``,
and the ``EnsembleDetector(OR)`` that combines them. Computes MCC against
the physics-defined ground truth from ``wt_essentiality_labels``.

This is the smart integration point: the atom engine now plugs into the
exact same detector API the Syn3A gene-knockout pipeline uses. Same
Trajectory/Sample format, same detector interface, same metrics module.
Only the simulator and the gene-to-rule map change.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cell_sim.atom_engine.essentiality_bridge import (
    AtomEngineSimConfig,
    AtomEngineSimulator,
    enumerate_pair_genes,
    wt_essentiality_labels,
)
from cell_sim.atom_engine.integrator import _rule_name_for_pair
from cell_sim.layer6_essentiality.ensemble_detector import (
    EnsembleDetector,
    EnsemblePolicy,
)
from cell_sim.layer6_essentiality.harness import FailureMode
from cell_sim.layer6_essentiality.metrics import evaluate_binary
from cell_sim.layer6_essentiality.per_rule_detector import PerRuleDetector
from cell_sim.layer6_essentiality.short_window_detector import (
    ShortWindowDetector,
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temperature", type=float, default=3000.0)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--radius", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-wt-events", type=int, default=5)
    p.add_argument("--deviation-threshold", type=float, default=0.10)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = AtomEngineSimConfig(
        temperature_K=args.temperature,
        steps=args.steps,
        radius_nm=args.radius,
        seed=args.seed,
    )
    sim = AtomEngineSimulator(cfg=cfg)
    genes = enumerate_pair_genes()
    print(f"{len(genes)} candidate pair-genes: "
          f"{[g for g, _ in genes]}")

    # Ground-truth labels from WT.
    print("\nrunning WT...")
    t0 = time.time()
    wt_traj = sim.run([])
    wt_wall = time.time() - t0
    wt_events = wt_traj.samples[-1].event_counts_by_rule or {}
    print(f"WT done in {wt_wall:.1f} s, "
          f"{wt_traj.samples[-1].t_s:.2f} (sim-time units), "
          f"events_by_rule={dict(wt_events)}")

    # Ground-truth essential labels (pair has >= min_wt_events in WT).
    def _pair_to_rule(pair):
        elems = list(pair)
        a = elems[0]
        b = elems[1] if len(elems) > 1 else elems[0]
        return _rule_name_for_pair(a, b)

    labels: dict[str, int] = {}
    for gene_id, pair in genes:
        rule = _pair_to_rule(pair)
        labels[gene_id] = int(wt_events.get(rule, 0) >= args.min_wt_events)
    positives = [g for g, v in labels.items() if v]
    negatives = [g for g, v in labels.items() if not v]
    print(f"\nground truth: {len(positives)} essential pairs {positives}, "
          f"{len(negatives)} non-essential {negatives}")

    # Gene-to-rule map for PerRuleDetector.
    gene_to_rules = {
        gene_id: {_pair_to_rule(pair)}
        for gene_id, pair in genes
    }

    # Build detectors.
    sw = ShortWindowDetector(
        wt=wt_traj,
        deviation_threshold=args.deviation_threshold,
        pools=tuple(sorted(wt_traj.samples[-1].pools.keys())),
    )
    pr = PerRuleDetector(wt=wt_traj, gene_to_rules=gene_to_rules,
                         min_wt_events=args.min_wt_events)
    ens = EnsembleDetector(
        short_window=sw, per_rule=pr,
        policy=EnsemblePolicy.OR_HIGH_CONFIDENCE,
    )

    preds_sw: dict[str, int] = {}
    preds_pr: dict[str, int] = {}
    preds_ens: dict[str, int] = {}

    print("\nrunning knockouts...")
    for i, (gene_id, _pair) in enumerate(genes):
        t0 = time.time()
        ko_traj = sim.run([gene_id])
        # ShortWindowDetector — pool deviation.
        sw_mode, _, sw_conf, sw_evidence = sw.detect(ko_traj)
        preds_sw[gene_id] = int(sw_mode != FailureMode.NONE)
        # PerRuleDetector — rule silencing.
        pr_mode, _, pr_conf, pr_evidence = pr.detect_for_gene(gene_id, ko_traj)
        preds_pr[gene_id] = int(pr_mode != FailureMode.NONE)
        # Ensemble — OR of the two.
        en_mode, _, en_conf, en_evidence = ens.detect_for_gene(
            gene_id, ko_traj
        )
        preds_ens[gene_id] = int(en_mode != FailureMode.NONE)
        ko_ev = ko_traj.samples[-1].event_counts_by_rule or {}
        wall = time.time() - t0
        print(f"  [{i+1}/{len(genes)}] {gene_id:10s} "
              f"label={labels[gene_id]} "
              f"sw={preds_sw[gene_id]} pr={preds_pr[gene_id]} "
              f"ens={preds_ens[gene_id]} "
              f"ko_events={dict(ko_ev)} ({wall:.1f} s)")

    # Compute MCC for each detector.
    m_sw = evaluate_binary(labels, preds_sw)
    m_pr = evaluate_binary(labels, preds_pr)
    m_ens = evaluate_binary(labels, preds_ens)
    print("\n=== MCC results ===")
    for name, m in [("ShortWindow", m_sw), ("PerRule", m_pr),
                    ("Ensemble(OR_HIGH_CONFIDENCE)", m_ens)]:
        print(f"  {name:14s}: MCC={m.mcc:+.3f} "
              f"acc={m.accuracy:.3f} "
              f"tp={m.tp} fp={m.fp} tn={m.tn} fn={m.fn}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "config": vars(args),
            "n_genes": len(genes),
            "positives": positives,
            "negatives": negatives,
            "wt_events": dict(wt_events),
            "labels": labels,
            "preds": {
                "short_window": preds_sw,
                "per_rule": preds_pr,
                "ensemble": preds_ens,
            },
            "metrics": {
                "short_window": m_sw.as_dict(),
                "per_rule": m_pr.as_dict(),
                "ensemble": m_ens.as_dict(),
            },
        }, indent=2))
        print(f"\nsummary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
