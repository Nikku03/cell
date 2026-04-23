"""Level-up: 10x data for all four moves, fix Move 3, use the
force surrogate to drive real MD and benchmark.

Usage::

    python scripts/level_up_ml.py --trajectories 80 --steps 4000 \\
        --epochs 25 --out /tmp/level_up.json
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
import torch

from cell_sim.atom_engine.atom_soup import SoupSpec, build_soup
from cell_sim.atom_engine.element import Element
from cell_sim.atom_engine.force_field import ForceFieldConfig, compute_forces
from cell_sim.atom_engine.integrator import (
    IntegratorConfig,
    SimState,
    current_temperature_K,
    step,
)
from cell_sim.atom_engine.ml_dataset import REACTION_CLASSES, TrajectoryCollector
from cell_sim.atom_engine.ml_model import (
    HeuristicBaseline,
    TrainConfig,
    evaluate,
    train_atom_gnn,
    train_force_surrogate,
    train_force_surrogate_equivariant,
    train_reaction_classifier,
)
from cell_sim.atom_engine.neural_force_field import NeuralForceField


def _make_state(seed: int, temperature_K: float):
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


def _collect(n_traj: int, steps: int, every: int, horizon: int, progress):
    snaps = []
    for tid in range(n_traj):
        T = 1500.0 + 500.0 * (tid % 4)
        state, ff, ic = _make_state(seed=500 + tid, temperature_K=T)
        coll = TrajectoryCollector(snapshot_every=every, horizon=horizon)
        coll.run(state, ff, ic, n_steps=steps, trajectory_id=tid)
        snaps.extend(coll.snapshots)
        if (tid + 1) % 10 == 0:
            progress(f"  collected {tid+1}/{n_traj} trajectories "
                     f"({len(snaps)} snapshots)")
    return snaps


def _benchmark_surrogate_md(
    model,
    n_steps: int = 300,
    temperature_K: float = 2000.0,
    progress=print,
) -> dict:
    """Run the same short MD twice — once with the real compute_forces,
    once with NeuralForceField — and report wall-clock step/s, mean
    force error on the first step, and whether T stayed bounded."""
    # Native path
    state_a, ff, ic = _make_state(seed=999, temperature_K=temperature_K)
    t0 = time.time()
    forces = None
    for _ in range(n_steps):
        forces = step(state_a, ff, ic, forces)
    native_elapsed = time.time() - t0
    native_T = current_temperature_K(state_a.atoms)

    # Surrogate path — patch integrator's compute_forces.
    from cell_sim.atom_engine import integrator as _integ
    state_b, ff, ic = _make_state(seed=999, temperature_K=temperature_K)
    nff = NeuralForceField(model=model, max_force_kj_per_nm=5.0e4)
    original = _integ.compute_forces
    _integ.compute_forces = nff
    try:
        t0 = time.time()
        forces = None
        for k in range(n_steps):
            forces = step(state_b, ff, ic, forces)
        surrogate_elapsed = time.time() - t0
        surrogate_T = current_temperature_K(state_b.atoms)
    finally:
        _integ.compute_forces = original

    # Force-error comparison on a fresh common state.
    state_c, ff, ic = _make_state(seed=1234, temperature_K=temperature_K)
    pos = np.array([a.position for a in state_c.atoms], dtype=np.float64)
    f_native = original(state_c.atoms, state_c.bonds, 0.0, ff, pos=pos)
    f_surrogate = nff(state_c.atoms, state_c.bonds, 0.0, ff, pos=pos)
    force_mse = float(((f_native - f_surrogate) ** 2).mean())
    gt_mse = float((f_native ** 2).mean())

    return {
        "n_steps": n_steps,
        "native_wall_s": native_elapsed,
        "native_steps_per_s": n_steps / native_elapsed,
        "native_T_K": native_T,
        "surrogate_wall_s": surrogate_elapsed,
        "surrogate_steps_per_s": n_steps / surrogate_elapsed,
        "surrogate_T_K": surrogate_T,
        "speedup": native_elapsed / surrogate_elapsed,
        "force_mse_vs_native": force_mse,
        "ground_truth_mse": gt_mse,
        "force_relative_error": float(np.sqrt(force_mse) /
                                      max(np.sqrt(gt_mse), 1e-9)),
    }


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectories", type=int, default=80)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--snapshot-every", type=int, default=100)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--force-epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--md-benchmark-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    t_all = time.time()

    print(f"collecting {args.trajectories} trajectories x {args.steps} steps")
    t0 = time.time()
    snaps = _collect(args.trajectories, args.steps, args.snapshot_every,
                     args.horizon, progress=lambda m: print(m))
    print(f"collected {len(snaps)} snapshots in {time.time() - t0:.1f} s")

    rng = np.random.default_rng(args.seed)
    traj_ids = sorted({s.snapshot_id // 100000 for s in snaps})
    rng.shuffle(traj_ids)
    n_val = max(1, int(round(args.val_fraction * len(traj_ids))))
    val_ids = set(traj_ids[:n_val])
    train = [s for s in snaps if (s.snapshot_id // 100000) not in val_ids]
    val = [s for s in snaps if (s.snapshot_id // 100000) in val_ids]
    print(f"train: {len(train)} snapshots  val: {len(val)} snapshots")

    cfg = TrainConfig(epochs=args.epochs, hidden=args.hidden, seed=args.seed)

    # === Move 1 ===
    print("\n=== Move 1: next-event binary ===")
    baseline = HeuristicBaseline()
    bs = evaluate(val, baseline.predict_proba)
    print(f"baseline: AUC={bs['auc']:.3f} acc={bs['acc']:.3f} "
          f"pos_rate={bs['positive_rate']:.3f}")
    model1, hist1 = train_atom_gnn(train, val, cfg,
                                    progress=lambda m: print(f"[gnn1] {m}"))
    model1.eval()

    def gnn_pred(snap):
        import torch as _t
        with _t.no_grad():
            logits = model1(
                _t.tensor(snap.node_features),
                _t.tensor(snap.edges.astype(np.int64)),
                _t.tensor(snap.edge_features),
            )
            return _t.sigmoid(logits).numpy()

    gs = evaluate(val, gnn_pred)
    print(f"GNN: AUC={gs['auc']:.3f} acc={gs['acc']:.3f}")

    # === Move 2 ===
    print("\n=== Move 2: reaction-type multiclass ===")
    rxn_cfg = TrainConfig(epochs=args.epochs, hidden=args.hidden, seed=args.seed)
    _rxn, rxn_hist = train_reaction_classifier(
        train, val, rxn_cfg, progress=lambda m: print(f"[rxn] {m}"))

    # === Move 3: EQUIVARIANT FORCE SURROGATE ===
    print("\n=== Move 3: equivariant force surrogate ===")
    train_pairs = [(s, s.forces_gt) for s in train if s.forces_gt is not None]
    val_pairs = [(s, s.forces_gt) for s in val if s.forces_gt is not None]
    print(f"  pairs: train={len(train_pairs)} val={len(val_pairs)}")
    force_cfg = TrainConfig(epochs=args.force_epochs, hidden=args.hidden,
                            seed=args.seed, lr=1e-3)
    force_model, force_hist = train_force_surrogate_equivariant(
        train_pairs, val_pairs, force_cfg, force_clip=100.0,
        progress=lambda m: print(f"[ff] {m}"))

    # === Move 4 (kept from earlier) ===
    print("\n=== Move 4: cell-level essentiality proxy ===")
    cell = _cell_essentiality(train, val)
    print(f"  val_acc={cell['val_acc']:.3f} val_mcc={cell['val_mcc']:.3f}")

    # === Closing the loop: MD with neural surrogate ===
    print("\n=== Closing the loop: MD with surrogate force field ===")
    md_bench = _benchmark_surrogate_md(force_model,
                                        n_steps=args.md_benchmark_steps)
    print(f"  native:    {md_bench['native_steps_per_s']:.1f} steps/s "
          f"final T={md_bench['native_T_K']:.0f} K")
    print(f"  surrogate: {md_bench['surrogate_steps_per_s']:.1f} steps/s "
          f"final T={md_bench['surrogate_T_K']:.0f} K")
    print(f"  speedup: {md_bench['speedup']:.2f}x")
    print(f"  force relative error: "
          f"{md_bench['force_relative_error']:.3f} "
          f"(vs native on unseen snapshot)")

    print(f"\ntotal wall: {time.time() - t_all:.1f} s")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "trajectories": args.trajectories,
            "steps": args.steps,
            "train_snapshots": len(train),
            "val_snapshots": len(val),
            "move1_baseline": bs,
            "move1_gnn": gs,
            "move1_history": hist1,
            "move2_history": rxn_hist,
            "move3_history": force_hist,
            "move4_cell": cell,
            "md_benchmark": md_bench,
        }, indent=2))
        print(f"summary: {out}")
    return 0


# copy of the cell-essentiality proxy from train_next_event.py
def _mcc(tp, tn, fp, fn):
    import math as _m
    num = tp * tn - fp * fn
    denom = _m.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / denom if denom > 0 else 0.0


def _cell_essentiality(train, val):
    import torch
    import torch.nn as nn

    def _summarise(snap):
        nf = snap.node_features
        speed = nf[:, 7]
        val_free = (nf[:, 6] > 0.0).mean()
        n_edges = snap.edges.shape[0]
        n_bonded = int((snap.edge_features[:, 3] == 1.0).sum())
        n_prox = n_edges - n_bonded
        return np.array([
            speed.mean(), speed.max(), val_free,
            n_edges / max(nf.shape[0], 1),
            n_bonded / max(nf.shape[0], 1),
            n_prox / max(nf.shape[0], 1),
        ], dtype=np.float32)

    X_train = np.stack([_summarise(s) for s in train])
    y_train_raw = np.array([int(s.labels.sum()) for s in train], dtype=np.float32)
    X_val = np.stack([_summarise(s) for s in val])
    y_val_raw = np.array([int(s.labels.sum()) for s in val], dtype=np.float32)
    thresh = float(np.median(y_train_raw))
    y_train = (y_train_raw > thresh).astype(np.float32)
    y_val = (y_val_raw > thresh).astype(np.float32)
    model = nn.Sequential(nn.Linear(X_train.shape[1], 16), nn.ReLU(),
                          nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    Xt, yt = torch.tensor(X_train), torch.tensor(y_train)
    for _ in range(200):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(Xt).squeeze(-1), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_pred = (torch.sigmoid(model(Xt).squeeze(-1)).numpy() > 0.5)
        val_pred = (torch.sigmoid(model(torch.tensor(X_val)).squeeze(-1))
                    .numpy() > 0.5)
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


if __name__ == "__main__":
    sys.exit(main())
