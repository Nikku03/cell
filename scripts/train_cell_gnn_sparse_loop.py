"""Tiny CPU-tractable training driver for SparseCellGNN.

Generates a synthetic cell_physics dataset on the fly (small enough to fit in
seconds), trains for a few epochs with a configurable variant, evaluates on a
held-out trajectory, and appends a single JSONL row to a leaderboard file.

Used as the per-iteration unit of work for a 15-step improvement loop.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'cell_sim'))

from cell_sim.atom_engine.cell_physics import (
    CellPhysicsConfig, SpeciesEntry, initial_state, run, compute_forces,
    CellState,
)
from cell_sim.atom_engine.cell_gnn import (
    EdgeType, build_dynamic_edges,
)
from cell_sim.atom_engine import cell_gnn_sparse


LEADERBOARD = REPO / 'outputs' / 'cell_gnn_sparse_leaderboard.jsonl'


# --------------------------------------------------------------------------
def normalize_forces_inplace(traj_list: list[dict], stats: dict | None = None
                              ) -> dict:
    """Z-score forces across all frames in all trajectories. If stats is
    given, reuse mean/std (so val uses train statistics). Returns the stats
    dict used."""
    if stats is None:
        all_f = np.concatenate(
            [fr['forces'].reshape(-1, 3)
             for traj in traj_list for fr in traj['frames']], axis=0)
        mean = all_f.mean(axis=0)
        std = all_f.std(axis=0) + 1e-6
        stats = {'mean': mean.astype(np.float32),
                 'std': std.astype(np.float32)}
    for traj in traj_list:
        for fr in traj['frames']:
            fr['forces'] = ((fr['forces'] - stats['mean']) / stats['std']
                            ).astype(np.float32)
    return stats


def make_synth_dataset(n_traj: int = 4, n_steps: int = 200,
                        save_every: int = 10, n_species: int = 5,
                        seed: int = 0) -> list[dict]:
    """Generate small cell_physics trajectories. Each entry has positions
    (T, N, 3), species_id (N,), and per-frame forces (T, N, 3) as targets."""
    species = [SpeciesEntry(name=f's{i}', sigma_nm=2.0,
                              epsilon_kj_per_mol=0.5, mass_da=10000.0,
                              diffusion_nm2_per_ns=0.05)
                 for i in range(n_species)]
    # Build species lookup arrays for fast CellState construction
    sigma_lkp = np.array([s.sigma_nm for s in species], dtype=np.float64)
    eps_lkp = np.array([s.epsilon_kj_per_mol for s in species], dtype=np.float64)
    mass_lkp = np.array([s.mass_da for s in species], dtype=np.float64)
    diff_lkp = np.array([s.diffusion_nm2_per_ns for s in species], dtype=np.float64)

    out = []
    for t in range(n_traj):
        cfg = CellPhysicsConfig(cell_radius_nm=25.0, dt_ns=0.05,
                                 pair_rebuild_every=10, seed=seed + t)
        composition = {0: 120, 1: 60, 2: 60, 3: 40, 4: 20}
        st = initial_state(species, composition, cfg)
        res = run(st, cfg, species, n_steps=n_steps, save_every=save_every)
        positions = res['positions'].astype(np.float32)         # (T, N, 3)
        sid = res['species_id'].astype(np.int64)                # (T, N)
        T, N, _ = positions.shape
        # For each saved frame, reconstruct a CellState containing only the
        # alive particles, run compute_forces to get the analytic-FF force on
        # each, and store as the training target. This gives a deterministic
        # signal (unlike Brownian displacement, which is dominated by noise).
        frames = []
        for fi in range(T):
            alive = (sid[fi] >= 0) & (sid[fi] < n_species)
            sid_a = sid[fi][alive]
            pos_a = positions[fi][alive]
            cs = CellState(
                pos=pos_a.astype(np.float64),
                species_id=sid_a.astype(np.int64),
                mass=mass_lkp[sid_a],
                sigma=sigma_lkp[sid_a],
                epsilon=eps_lkp[sid_a],
                diffusion=diff_lkp[sid_a],
                persistent_id=np.arange(sid_a.shape[0], dtype=np.int64),
            )
            f = compute_forces(cs, cfg).astype(np.float32)
            frames.append({'pos': pos_a.astype(np.float32),
                           'species_id': sid_a, 'forces': f})
        out.append({'frames': frames})
    return out


# --------------------------------------------------------------------------
def make_batch(traj: dict, frame_idx: int, r_cut: float
                ) -> dict:
    fr = traj['frames'][frame_idx]
    pos = torch.from_numpy(fr['pos'])
    sid = torch.from_numpy(fr['species_id'])
    f = torch.from_numpy(fr['forces'])
    extras = torch.zeros(sid.shape[0], 4, dtype=torch.float32)
    edges_d = build_dynamic_edges(pos, sid, r_cut_spatial=r_cut)
    return {
        'pos': pos, 'species_id': sid, 'extras': extras,
        'forces_target': f, 'edges': edges_d['edges'],
        'r': edges_d['r'], 'edge_type': edges_d['edge_type'],
        'thickness': edges_d['thickness'],
    }


# --------------------------------------------------------------------------
def train_variant(variant: str, config: dict, dataset_train: list,
                   dataset_val: list, log_every: int = 5) -> dict:
    """Train a SparseCellGNN with the given config dict. Returns a metrics
    dict to append to the leaderboard."""
    torch.manual_seed(config.get('seed', 0))

    importlib.reload(cell_gnn_sparse)         # pick up code edits between runs
    SparseCellGNN = cell_gnn_sparse.SparseCellGNN
    compute_sparse_losses = cell_gnn_sparse.compute_sparse_losses

    model = SparseCellGNN(
        n_species=config['n_species'],
        hidden=config['hidden'],
        n_rounds=config['n_rounds'],
        n_reaction_classes=config['n_reaction_classes'],
        r_cut=config['r_cut'],
        n_extra_node_features=4,
        k_patterns=config['k_patterns'],
        top_k=config['top_k'],
    )
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = None
    if config.get('cosine_lr'):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=config['epochs'])

    t0 = time.time()
    train_losses = []
    val_metrics = []

    for ep in range(config['epochs']):
        model.train()
        ep_loss = 0.0
        n_steps = 0
        for traj in dataset_train:
            T = len(traj['frames'])
            for fi in range(T):
                batch = make_batch(traj, fi, config['r_cut'])
                if batch['edges'].shape[0] == 0:
                    continue
                out = model(species_id=batch['species_id'],
                             extra_feats=batch['extras'],
                             edges=batch['edges'],
                             r=batch['r'],
                             edge_type=batch['edge_type'],
                             pos=batch['pos'],
                             thickness=batch['thickness'])
                losses = compute_sparse_losses(
                    out, batch['forces_target'],
                    w_force=config.get('w_force', 1.0),
                    w_recon=config.get('w_recon', 0.1),
                    w_diversity=config.get('w_diversity', 0.1),
                    w_align=config.get('w_align', 0.3),
                )
                opt.zero_grad()
                losses['total'].backward()
                if config.get('grad_clip'):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config['grad_clip'])
                opt.step()
                ep_loss += float(losses['total'].item())
                n_steps += 1
        if scheduler is not None:
            scheduler.step()
        train_losses.append(ep_loss / max(n_steps, 1))

    # Eval
    model.eval()
    val_force_mse = 0.0
    val_n = 0
    pat_usage = torch.zeros(config['k_patterns'])
    pat_active_per_node = []
    raw_hiddens, sids_all = [], []
    with torch.no_grad():
        for traj in dataset_val:
            T = len(traj['frames'])
            for fi in range(T):
                batch = make_batch(traj, fi, config['r_cut'])
                if batch['edges'].shape[0] == 0:
                    continue
                out = model(species_id=batch['species_id'],
                             extra_feats=batch['extras'],
                             edges=batch['edges'],
                             r=batch['r'],
                             edge_type=batch['edge_type'],
                             pos=batch['pos'],
                             thickness=batch['thickness'])
                val_force_mse += float(F.mse_loss(out['forces'],
                                                    batch['forces_target']
                                                    ).item())
                val_n += 1
                pat_usage += (out['pattern_act'] > 0).float().sum(dim=0)
                pat_active_per_node.append(
                    (out['pattern_act'] > 0).float().sum(dim=-1).mean().item()
                )
                raw_hiddens.append(out['raw_hidden'])
                sids_all.append(batch['species_id'])
    val_force_mse /= max(val_n, 1)
    n_active_patterns = int((pat_usage > 0).sum().item())
    mean_active_per_node = float(np.mean(pat_active_per_node)) if pat_active_per_node else 0.0
    # Pattern-vs-species linear separability: train a 1-vs-rest logistic on
    # raw_hidden -> species_id over the val set. Simple proxy for "are
    # embeddings discriminative?"
    sep = float('nan')
    if raw_hiddens:
        H = torch.cat(raw_hiddens)
        S = torch.cat(sids_all)
        sep = quick_linear_acc(H, S)

    elapsed = time.time() - t0
    metrics = {
        'variant': variant,
        'config': config,
        'train_loss_final': train_losses[-1],
        'val_force_mse': val_force_mse,
        'pattern_coverage': n_active_patterns,
        'mean_active_per_node': mean_active_per_node,
        'species_linear_acc': sep,
        'elapsed_s': elapsed,
        'n_params': sum(p.numel() for p in model.parameters()),
    }
    return metrics, model


def quick_linear_acc(H: torch.Tensor, S: torch.Tensor) -> float:
    """Train a tiny linear classifier (closed form via torch lstsq on one-hot)
    and return train accuracy. Used purely as an embedding-quality proxy."""
    n, hd = H.shape
    K = int(S.max().item()) + 1
    if K < 2:
        return float('nan')
    Y = F.one_hot(S, num_classes=K).float()
    # Solve W minimising || H W - Y ||^2 with ridge
    A = H.T @ H + 1e-3 * torch.eye(hd)
    B = H.T @ Y
    W = torch.linalg.solve(A, B)
    pred = (H @ W).argmax(dim=-1)
    return float((pred == S).float().mean().item())


# --------------------------------------------------------------------------
def append_leaderboard(metrics: dict):
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    with LEADERBOARD.open('a') as f:
        f.write(json.dumps(metrics) + '\n')


def default_config() -> dict:
    return dict(
        n_species=5, hidden=32, n_rounds=2, n_reaction_classes=8,
        r_cut=8.0, k_patterns=64, top_k=4, lr=3e-3, epochs=4,
        w_force=1.0, w_recon=0.1, w_diversity=0.1, w_align=0.3,
        seed=0, cosine_lr=False, grad_clip=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True,
                    help='label for this leaderboard row')
    ap.add_argument('--config-overrides', default='{}',
                    help='JSON dict to merge into default_config()')
    ap.add_argument('--n-train', type=int, default=4)
    ap.add_argument('--n-val', type=int, default=2)
    ap.add_argument('--n-steps', type=int, default=120)
    args = ap.parse_args()

    overrides = json.loads(args.config_overrides)
    cfg = default_config()
    cfg.update(overrides)

    print(f'[V {args.variant}] config: {cfg}')
    print(f'[V {args.variant}] generating synth data...')
    train_set = make_synth_dataset(n_traj=args.n_train, n_steps=args.n_steps,
                                     save_every=15, seed=0)
    val_set   = make_synth_dataset(n_traj=args.n_val, n_steps=args.n_steps,
                                     save_every=15, seed=1000)

    print(f'[V {args.variant}] training...')
    metrics, model = train_variant(args.variant, cfg, train_set, val_set)

    print(f'[V {args.variant}] {metrics}')
    append_leaderboard(metrics)


if __name__ == '__main__':
    main()
