"""Train the next-event bond-formation predictor.

Pipeline:
  1. Run several short reactive MD trajectories to collect
     ``Snapshot`` objects (atom features + live-bond edges +
     within-horizon bond_formed labels).
  2. Split snapshots by trajectory into train/val.
  3. Evaluate the HeuristicBaseline.
  4. Train AtomGNN for a few epochs, report val AUC / accuracy.

Usage:
    python scripts/train_next_event.py [--trajectories 6] [--epochs 20]
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

from cell_sim.atom_engine.atom_soup import SoupSpec, build_soup
from cell_sim.atom_engine.element import Element
from cell_sim.atom_engine.force_field import ForceFieldConfig
from cell_sim.atom_engine.integrator import IntegratorConfig, SimState
from cell_sim.atom_engine.ml_dataset import TrajectoryCollector
from cell_sim.atom_engine.ml_model import (
    HeuristicBaseline,
    TrainConfig,
    evaluate,
    train_atom_gnn,
)


def _make_state(seed: int, temperature_K: float) -> tuple[SimState,
                                                           ForceFieldConfig,
                                                           IntegratorConfig]:
    atoms = build_soup(SoupSpec(
        composition={Element.H: 40, Element.C: 10, Element.N: 4, Element.O: 6},
        radius_nm=1.2, temperature_K=temperature_K, seed=seed,
    ))
    state = SimState(atoms=atoms, bonds=[])
    ff = ForceFieldConfig(
        lj_cutoff_nm=1.0, use_confinement=True, confinement_radius_nm=1.2,
        use_neighbor_list=True, reactive_sigma_scale=0.3,
    )
    ic = IntegratorConfig(
        dt_ps=0.001, target_temperature_K=temperature_K,
        dynamic_bonding=True, bond_form_distance_nm=0.2,
        neighbor_rebuild_every=10,
    )
    return state, ff, ic


def collect(n_trajectories: int,
            steps_per_traj: int,
            snapshot_every: int,
            horizon: int,
            progress=print) -> list:
    all_snaps = []
    for tid in range(n_trajectories):
        T = 1500.0 + 500.0 * (tid % 3)     # mix 1500/2000/2500 K
        state, ff, ic = _make_state(seed=100 + tid, temperature_K=T)
        progress(f"[traj {tid}] T={T:.0f} K, {steps_per_traj} steps")
        coll = TrajectoryCollector(snapshot_every=snapshot_every, horizon=horizon)
        coll.run(state, ff, ic, n_steps=steps_per_traj, trajectory_id=tid)
        all_snaps.extend(coll.snapshots)
        positive = sum(int(s.labels.sum()) for s in coll.snapshots)
        total = sum(int(s.labels.size) for s in coll.snapshots)
        progress(f"[traj {tid}] +{len(coll.snapshots)} snapshots, "
                 f"{positive}/{total} positive labels "
                 f"({positive / max(total, 1):.1%})")
    return all_snaps


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectories", type=int, default=6)
    p.add_argument("--steps", type=int, default=2000,
                   help="MD steps per trajectory.")
    p.add_argument("--snapshot-every", type=int, default=100)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    t0 = time.time()
    print(f"collecting data: {args.trajectories} trajectories x "
          f"{args.steps} steps")
    snaps = collect(args.trajectories, args.steps,
                    args.snapshot_every, args.horizon)
    print(f"collected {len(snaps)} snapshots in {time.time() - t0:.1f} s")

    # Split by snapshot_id (trajectory_id * 100000 + step).
    rng = np.random.default_rng(args.seed)
    snap_ids = sorted({s.snapshot_id // 100000 for s in snaps})
    rng.shuffle(snap_ids)
    n_val = max(1, int(round(args.val_fraction * len(snap_ids))))
    val_ids = set(snap_ids[:n_val])
    val = [s for s in snaps if (s.snapshot_id // 100000) in val_ids]
    train = [s for s in snaps if (s.snapshot_id // 100000) not in val_ids]
    print(f"train: {len(train)} snapshots, val: {len(val)} snapshots")

    # Baseline.
    baseline = HeuristicBaseline()
    bs = evaluate(val, baseline.predict_proba)
    print(f"BASELINE (heuristic): AUC={bs['auc']:.3f} "
          f"acc={bs['acc']:.3f} pos_rate={bs['positive_rate']:.3f}")

    # Train GNN.
    train_cfg = TrainConfig(epochs=args.epochs, hidden=args.hidden,
                            seed=args.seed)
    model, history = train_atom_gnn(train, val, train_cfg,
                                    progress=lambda m: print(f"[gnn] {m}"))

    # Final evaluation.
    import torch
    model.eval()

    def gnn_predict(snap):
        import torch as _t
        nf = _t.tensor(snap.node_features)
        e = _t.tensor(snap.edges.astype(np.int64))
        ef = _t.tensor(snap.edge_features)
        with _t.no_grad():
            logits = model(nf, e, ef)
            return _t.sigmoid(logits).numpy()

    gs = evaluate(val, gnn_predict)
    print(f"GNN (final): AUC={gs['auc']:.3f} acc={gs['acc']:.3f} "
          f"pos_rate={gs['positive_rate']:.3f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "n_trajectories": args.trajectories,
            "steps": args.steps,
            "snapshot_every": args.snapshot_every,
            "horizon": args.horizon,
            "train_snapshots": len(train),
            "val_snapshots": len(val),
            "baseline": bs,
            "gnn": gs,
            "training_history": history,
        }, indent=2))
        print(f"summary: {out_path}")

    print(f"total wall: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
