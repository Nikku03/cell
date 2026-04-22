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
from cell_sim.atom_engine.ml_dataset import (
    REACTION_CLASSES,
    TrajectoryCollector,
)
from cell_sim.atom_engine.ml_model import (
    HeuristicBaseline,
    TrainConfig,
    evaluate,
    train_atom_gnn,
    train_force_surrogate,
    train_reaction_classifier,
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


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Matthews correlation coefficient. Returns 0 if the denominator
    is zero (undefined), which matches the Breuer 2019 detector code
    elsewhere in the repo."""
    import math as _m
    num = tp * tn - fp * fn
    denom = _m.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return num / denom


def _cell_essentiality(train, val) -> dict:
    """Treat each snapshot as a 'cell state' and predict whether it
    belongs to the 'high-activity' class (reactive chemistry proxy for
    essentiality). Features: pooled per-atom embedding (summary stats)
    + fraction of atoms with valence remaining. Classifier: logistic
    regression via torch, trained from scratch.

    This is a PROOF OF CONCEPT, not a real Breuer-2019 MCC readout.
    The point is to show that the atom-engine ML outputs plug into a
    cell-level binary classification exactly the same way the earlier
    detector framework does.
    """
    import torch
    import torch.nn as nn

    def _summarise(snap):
        # Summary features per snapshot:
        # [mean speed, max speed, fraction_valence_free, edge_density,
        #  n_bonded_edges, n_proximity_edges]
        nf = snap.node_features
        speed = nf[:, 7]   # speed col (see extract_node_features)
        val_free = (nf[:, 6] > 0.0).mean()
        n_edges = snap.edges.shape[0]
        # bonded edges have edge_features[:, 3] == 1.0
        n_bonded = int((snap.edge_features[:, 3] == 1.0).sum())
        n_prox = n_edges - n_bonded
        return np.array([
            speed.mean(), speed.max(), val_free,
            n_edges / max(nf.shape[0], 1),
            n_bonded / max(nf.shape[0], 1),
            n_prox / max(nf.shape[0], 1),
        ], dtype=np.float32)

    def _label(snap):
        # "Essential" = above-median bond formation rate.
        return int(snap.labels.sum())

    X_train = np.stack([_summarise(s) for s in train])
    y_train_raw = np.array([_label(s) for s in train], dtype=np.float32)
    X_val = np.stack([_summarise(s) for s in val])
    y_val_raw = np.array([_label(s) for s in val], dtype=np.float32)

    # Binarise at median of training set.
    thresh = float(np.median(y_train_raw))
    y_train = (y_train_raw > thresh).astype(np.float32)
    y_val = (y_val_raw > thresh).astype(np.float32)

    # Simple logistic regression.
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 16), nn.ReLU(),
        nn.Linear(16, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    Xt = torch.tensor(X_train)
    yt = torch.tensor(y_train)
    for _ in range(100):
        opt.zero_grad()
        logits = model(Xt).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_pred = (torch.sigmoid(model(Xt).squeeze(-1)).numpy() > 0.5).astype(np.float32)
        Xv = torch.tensor(X_val)
        val_pred = (torch.sigmoid(model(Xv).squeeze(-1)).numpy() > 0.5).astype(np.float32)
    tp = int(((val_pred == 1) & (y_val == 1)).sum())
    tn = int(((val_pred == 0) & (y_val == 0)).sum())
    fp = int(((val_pred == 1) & (y_val == 0)).sum())
    fn = int(((val_pred == 0) & (y_val == 1)).sum())
    return {
        "train_acc": float((train_pred == y_train).mean()),
        "val_acc": float((val_pred == y_val).mean()),
        "val_mcc": _mcc(tp, tn, fp, fn),
        "val_positive_rate": float(y_val.mean()),
        "threshold_bond_count": thresh,
    }


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
    print(f"GNN next-event: AUC={gs['auc']:.3f} acc={gs['acc']:.3f} "
          f"pos_rate={gs['positive_rate']:.3f}")

    # --- Move 2: reaction-type predictor ---
    print("\n=== Move 2: reaction-type multiclass ===")
    rxn_model, rxn_hist = train_reaction_classifier(
        train, val, train_cfg,
        progress=lambda m: print(f"[rxn] {m}"),
    )
    # Summarize class distribution on val events
    from collections import Counter
    event_count = Counter()
    for s in val:
        for lbl in s.reaction_labels:
            if lbl != REACTION_CLASSES.index("none"):
                event_count[REACTION_CLASSES[int(lbl)]] += 1
    print(f"val event distribution: {dict(event_count)}")

    # --- Move 3: neural surrogate force field ---
    print("\n=== Move 3: force surrogate ===")
    train_pairs = [(s, s.forces_gt) for s in train if s.forces_gt is not None]
    val_pairs = [(s, s.forces_gt) for s in val if s.forces_gt is not None]
    force_model, force_hist = train_force_surrogate(
        train_pairs, val_pairs, train_cfg,
        progress=lambda m: print(f"[ff] {m}"),
    )

    # --- Move 4: cell-level essentiality readout (proof of concept) ---
    print("\n=== Move 4: cell-level essentiality readout ===")
    # Aggregate per-snapshot features into a single vector and classify
    # whether that SNAPSHOT shows "essential" reactive activity
    # (proxy: > threshold bond formations per atom in horizon window).
    cell_results = _cell_essentiality(train, val)
    print(f"cell essentiality proxy: train_acc={cell_results['train_acc']:.3f} "
          f"val_acc={cell_results['val_acc']:.3f} "
          f"val_mcc={cell_results['val_mcc']:.3f}")

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
            "move1_baseline": bs,
            "move1_gnn": gs,
            "move1_training_history": history,
            "move2_reaction_history": rxn_hist,
            "move2_event_distribution": dict(event_count),
            "move3_force_history": force_hist,
            "move4_cell_essentiality": cell_results,
        }, indent=2))
        print(f"summary: {out_path}")

    print(f"total wall: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
