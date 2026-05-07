"""15-variant SparseCellGNNV2 improvement loop, full-freedom edition.

Each variant tests a single physics-aware or training-aware change against
a strong V0 baseline (V13's settings from the v1 loop). The final V15 is an
ensemble over the top-3 winners.

Logs to outputs/cell_gnn_sparse_v2_leaderboard.jsonl.
"""
from __future__ import annotations

import importlib
import json
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
from cell_sim.atom_engine.cell_gnn import build_dynamic_edges
from cell_sim.atom_engine import cell_gnn_sparse_v2

LEADERBOARD = REPO / 'outputs' / 'cell_gnn_sparse_v2_leaderboard.jsonl'


# --------------------------------------------------------------------------
def make_dataset(n_traj: int, n_steps: int, save_every: int,
                  seed: int, n_species: int = 5) -> list[dict]:
    species = [SpeciesEntry(name=f's{i}', sigma_nm=2.0,
                              epsilon_kj_per_mol=0.5, mass_da=10000.0,
                              diffusion_nm2_per_ns=0.05)
                 for i in range(n_species)]
    sigma = np.array([s.sigma_nm for s in species], dtype=np.float64)
    eps = np.array([s.epsilon_kj_per_mol for s in species], dtype=np.float64)
    mass = np.array([s.mass_da for s in species], dtype=np.float64)
    diff = np.array([s.diffusion_nm2_per_ns for s in species], dtype=np.float64)
    out = []
    for t in range(n_traj):
        # Vary cell radius per trajectory for richer training distribution
        radius = 22.0 + (t % 4) * 1.5
        cfg = CellPhysicsConfig(cell_radius_nm=radius, dt_ns=0.05,
                                 pair_rebuild_every=10, seed=seed + t)
        composition = {0: 130, 1: 70, 2: 60, 3: 30, 4: 20}
        st = initial_state(species, composition, cfg)
        res = run(st, cfg, species, n_steps=n_steps, save_every=save_every)
        positions = res['positions'].astype(np.float32)
        sid = res['species_id'].astype(np.int64)
        T = positions.shape[0]
        frames = []
        for fi in range(T):
            alive = (sid[fi] >= 0) & (sid[fi] < n_species)
            sa = sid[fi][alive]
            pa = positions[fi][alive]
            cs = CellState(pos=pa.astype(np.float64), species_id=sa,
                            mass=mass[sa], sigma=sigma[sa], epsilon=eps[sa],
                            diffusion=diff[sa],
                            persistent_id=np.arange(sa.shape[0],
                                                     dtype=np.int64))
            f = compute_forces(cs, cfg).astype(np.float32)
            frames.append({'pos': pa.astype(np.float32),
                           'species_id': sa, 'forces': f, 'cfg': cfg})
        out.append({'frames': frames})
    return out


def normalize_forces(traj_list, stats=None):
    if stats is None:
        all_f = np.concatenate(
            [fr['forces'].reshape(-1, 3) for traj in traj_list
             for fr in traj['frames']], axis=0)
        stats = {'mean': all_f.mean(axis=0).astype(np.float32),
                 'std': (all_f.std(axis=0) + 1e-6).astype(np.float32)}
    for traj in traj_list:
        for fr in traj['frames']:
            fr['forces'] = ((fr['forces'] - stats['mean']) / stats['std']
                            ).astype(np.float32)
    return stats


# --------------------------------------------------------------------------
def make_batch(traj, fi, r_cut):
    fr = traj['frames'][fi]
    pos = torch.from_numpy(fr['pos'])
    sid = torch.from_numpy(fr['species_id'])
    f = torch.from_numpy(fr['forces'])
    extras = torch.zeros(sid.shape[0], 4, dtype=torch.float32)
    edges_d = build_dynamic_edges(pos, sid, r_cut_spatial=r_cut)
    return {'pos': pos, 'species_id': sid, 'extras': extras,
            'forces_target': f, 'edges': edges_d['edges'],
            'r': edges_d['r'], 'edge_type': edges_d['edge_type'],
            'thickness': edges_d['thickness']}


# --------------------------------------------------------------------------
def train(model, train_set, val_set, opt, n_epochs, r_cut,
          loss_kwargs, scheduler=None, grad_clip=1.0):
    for ep in range(n_epochs):
        model.train()
        ep_loss = 0.0
        n = 0
        for traj in train_set:
            for fi in range(len(traj['frames'])):
                b = make_batch(traj, fi, r_cut)
                if b['edges'].shape[0] == 0:
                    continue
                # Reactivity target: 1 if |force_target| > 1.0 z-score (high
                # force particles), else 0
                target_react = (b['forces_target'].norm(dim=-1) > 1.0).float()
                out = model(species_id=b['species_id'],
                             extra_feats=b['extras'], edges=b['edges'],
                             r=b['r'], edge_type=b['edge_type'], pos=b['pos'],
                             thickness=b['thickness'])
                losses = cell_gnn_sparse_v2.compute_v2_losses(
                    out, b['forces_target'], target_species=b['species_id'],
                    target_reactivity=target_react, **loss_kwargs)
                opt.zero_grad()
                losses['total'].backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                ep_loss += float(losses['total'].item())
                n += 1
        if scheduler is not None:
            scheduler.step()
    # Eval
    return evaluate(model, val_set, r_cut)


def evaluate(model, val_set, r_cut, return_predictions=False):
    model.eval()
    val_force_mse = 0.0
    n = 0
    pat_usage = None
    pat_active_per_node = []
    react_correct = 0
    react_total = 0
    concept_correct = 0
    concept_total = 0
    preds_per_frame = []
    with torch.no_grad():
        for traj in val_set:
            for fi in range(len(traj['frames'])):
                b = make_batch(traj, fi, r_cut)
                if b['edges'].shape[0] == 0:
                    continue
                out = model(species_id=b['species_id'],
                             extra_feats=b['extras'], edges=b['edges'],
                             r=b['r'], edge_type=b['edge_type'], pos=b['pos'],
                             thickness=b['thickness'])
                val_force_mse += float(F.mse_loss(out['forces'],
                                                    b['forces_target']).item())
                n += 1
                act = (out['pattern_act'] > 0).float()
                if pat_usage is None:
                    pat_usage = act.sum(dim=0)
                else:
                    pat_usage = pat_usage + act.sum(dim=0)
                pat_active_per_node.append(act.sum(dim=-1).mean().item())
                target_react = (b['forces_target'].norm(dim=-1) > 1.0).float()
                react_pred = (out['reactivity_logit'] > 0).float()
                react_correct += (react_pred == target_react).sum().item()
                react_total += target_react.shape[0]
                concept_pred = out['species_from_pattern'].argmax(dim=-1)
                concept_correct += (concept_pred == b['species_id']).sum().item()
                concept_total += b['species_id'].shape[0]
                if return_predictions:
                    preds_per_frame.append(out['forces'].detach().clone())
    metrics = {
        'val_force_mse': val_force_mse / max(n, 1),
        'pattern_coverage': int((pat_usage > 0).sum().item()) if pat_usage is not None else 0,
        'mean_active_per_node': float(np.mean(pat_active_per_node)) if pat_active_per_node else 0.0,
        'reactivity_acc': react_correct / max(react_total, 1),
        'concept_acc': concept_correct / max(concept_total, 1),
    }
    if return_predictions:
        return metrics, preds_per_frame
    return metrics


# --------------------------------------------------------------------------
def base_model_kwargs():
    return dict(n_species=5, hidden=64, n_rounds=3, n_reaction_classes=8,
                r_cut=12.0, n_extra_node_features=4,
                k_patterns=128, top_k=8,
                hetero_msg=False, sinkhorn_patterns=False,
                pattern_dropout=0.0, iterative_refinement=False)


def base_loss_kwargs():
    return dict(w_force=1.0, w_recon=0.1, w_diversity=0.1, w_concept=0.0,
                w_reactivity=0.0, force_kind='mse', weight_by_magnitude=False)


# --------------------------------------------------------------------------
VARIANTS = [
    # V0 baseline: V13's settings carried over (no hetero msg, no extras)
    dict(name='V0_baseline_v2', model={}, loss={}, train={'epochs': 8}),

    # V1 — true hetero per-edge-type message MLPs
    dict(name='V1_hetero_msg', model={'hetero_msg': True}, loss={},
         train={'epochs': 8}),

    # V2 — Newton-symmetric edge force only (this is BUILT INTO v2 already
    # but the v1 baseline was asymmetric; re-confirm here against asymmetric
    # by training v2 with hidden+r_cut matching v1)
    dict(name='V2_huber_loss', model={}, loss={'force_kind': 'huber'},
         train={'epochs': 8}),

    # V3 — magnitude-weighted force loss (focus on close-contact particles)
    dict(name='V3_mag_weighted', model={}, loss={'weight_by_magnitude': True},
         train={'epochs': 8}),

    # V4 — concept supervision (predict species from pattern code)
    dict(name='V4_concept_sup', model={}, loss={'w_concept': 0.5},
         train={'epochs': 8}),

    # V5 — reactivity multi-task head
    dict(name='V5_reactivity_aux', model={}, loss={'w_reactivity': 0.3},
         train={'epochs': 8}),

    # V6 — Sinkhorn pattern uniformity
    dict(name='V6_sinkhorn', model={'sinkhorn_patterns': True}, loss={},
         train={'epochs': 8}),

    # V7 — pattern dropout
    dict(name='V7_pattern_dropout', model={'pattern_dropout': 0.1}, loss={},
         train={'epochs': 8}),

    # V8 — iterative refinement (two-pass force prediction)
    dict(name='V8_iter_refine', model={'iterative_refinement': True}, loss={},
         train={'epochs': 8}),

    # V9 — bigger model: hidden=96, n_rounds=4
    dict(name='V9_bigger_model',
         model={'hidden': 96, 'n_rounds': 4}, loss={}, train={'epochs': 8}),

    # V10 — more epochs (16)
    dict(name='V10_more_epochs', model={}, loss={}, train={'epochs': 16}),

    # V11 — combo of best-prior wins (hetero + huber + concept)
    dict(name='V11_combo_best',
         model={'hetero_msg': True},
         loss={'force_kind': 'huber', 'w_concept': 0.5},
         train={'epochs': 12}),

    # V12 — bigger combo: + Sinkhorn + reactivity aux
    dict(name='V12_super_combo',
         model={'hetero_msg': True, 'sinkhorn_patterns': True},
         loss={'force_kind': 'huber', 'w_concept': 0.5, 'w_reactivity': 0.3},
         train={'epochs': 12}),

    # V13 — V12 + iterative refinement + bigger
    dict(name='V13_kitchen_sink',
         model={'hetero_msg': True, 'sinkhorn_patterns': True,
                'iterative_refinement': True, 'hidden': 96, 'n_rounds': 4},
         loss={'force_kind': 'huber', 'w_concept': 0.5, 'w_reactivity': 0.3,
                'weight_by_magnitude': True},
         train={'epochs': 12}),

    # V14 — V13 minus iterative (was iter_refine helping?)
    dict(name='V14_kitchen_minus_iter',
         model={'hetero_msg': True, 'sinkhorn_patterns': True,
                'hidden': 96, 'n_rounds': 4},
         loss={'force_kind': 'huber', 'w_concept': 0.5, 'w_reactivity': 0.3,
                'weight_by_magnitude': True},
         train={'epochs': 16}),
]


# --------------------------------------------------------------------------
def main():
    LEADERBOARD.unlink(missing_ok=True)
    print('=' * 72)
    print('SparseCellGNNV2 — full-freedom 15-iteration loop')
    print('=' * 72)
    print('Generating dataset (8 train traj + 3 val traj, varied radii)...')
    train_set = make_dataset(n_traj=8, n_steps=120, save_every=15, seed=0)
    val_set   = make_dataset(n_traj=3, n_steps=120, save_every=15, seed=2000)
    stats = normalize_forces(train_set)
    normalize_forces(val_set, stats=stats)
    n_train = sum(len(t['frames']) for t in train_set)
    n_val = sum(len(t['frames']) for t in val_set)
    print(f'  train frames: {n_train}    val frames: {n_val}')

    rows = []
    saved_models = {}            # variant_name -> state_dict (for ensemble)

    def run_variant(name, model_kw, loss_kw, train_kw):
        importlib.reload(cell_gnn_sparse_v2)
        torch.manual_seed(0)
        m_kw = base_model_kwargs(); m_kw.update(model_kw)
        l_kw = base_loss_kwargs(); l_kw.update(loss_kw)
        t_kw = {'epochs': 8, 'lr': 3e-3}; t_kw.update(train_kw)
        model = cell_gnn_sparse_v2.SparseCellGNNV2(**m_kw)
        opt = torch.optim.Adam(model.parameters(), lr=t_kw['lr'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                            T_max=t_kw['epochs'])
        t0 = time.time()
        metrics = train(model, train_set, val_set, opt, t_kw['epochs'],
                         m_kw['r_cut'], loss_kwargs=l_kw, scheduler=sched)
        elapsed = time.time() - t0
        n_params = sum(p.numel() for p in model.parameters())
        rec = dict(variant=name, model=model_kw, loss=loss_kw, train=train_kw,
                    elapsed_s=elapsed, n_params=n_params, **metrics)
        with LEADERBOARD.open('a') as f:
            f.write(json.dumps(rec, default=str) + '\n')
        rows.append(rec)
        saved_models[name] = (model, m_kw)
        return rec

    for v in VARIANTS:
        print(f"\n--- {v['name']} ---")
        try:
            rec = run_variant(v['name'], v['model'], v['loss'], v['train'])
            print(f"  force_mse={rec['val_force_mse']:.4f}  "
                  f"cov={rec['pattern_coverage']}/128  "
                  f"react={rec['reactivity_acc']:.3f}  "
                  f"concept={rec['concept_acc']:.3f}  "
                  f"t={rec['elapsed_s']:.1f}s")
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()

    # V15 — ensemble of top 3 by val_force_mse
    print('\n--- V15_ensemble ---')
    sorted_rows = sorted(rows, key=lambda r: r['val_force_mse'])
    top3 = sorted_rows[:3]
    print(f"  averaging predictions from: {[r['variant'] for r in top3]}")
    # Run each top model in eval, accumulate predictions, average per frame
    importlib.reload(cell_gnn_sparse_v2)
    accum_mse = 0.0
    n_frames = 0
    with torch.no_grad():
        for traj in val_set:
            for fi in range(len(traj['frames'])):
                b = make_batch(traj, fi, 12.0)
                if b['edges'].shape[0] == 0:
                    continue
                preds = []
                for r_top in top3:
                    model, m_kw = saved_models[r_top['variant']]
                    model.eval()
                    out = model(species_id=b['species_id'],
                                 extra_feats=b['extras'], edges=b['edges'],
                                 r=b['r'], edge_type=b['edge_type'],
                                 pos=b['pos'], thickness=b['thickness'])
                    preds.append(out['forces'])
                avg = torch.stack(preds).mean(dim=0)
                accum_mse += float(F.mse_loss(avg, b['forces_target']).item())
                n_frames += 1
    ens_mse = accum_mse / max(n_frames, 1)
    rec_ens = dict(variant='V15_ensemble',
                    members=[r['variant'] for r in top3],
                    val_force_mse=ens_mse,
                    pattern_coverage=0, reactivity_acc=0.0, concept_acc=0.0,
                    elapsed_s=0.0, n_params=0,
                    model={}, loss={}, train={})
    with LEADERBOARD.open('a') as f:
        f.write(json.dumps(rec_ens, default=str) + '\n')
    rows.append(rec_ens)
    print(f"  ensemble force_mse={ens_mse:.4f}")

    # Final report
    print('\n' + '=' * 72)
    print('FINAL LEADERBOARD  (sorted by val_force_mse)')
    print('=' * 72)
    rows.sort(key=lambda r: r['val_force_mse'])
    print(f'{"rank":>4s}  {"variant":<22s}  {"force_mse":>10s}  '
          f'{"cov":>6s}  {"react":>6s}  {"concept":>7s}  {"params":>8s}  '
          f'{"t (s)":>7s}')
    print('-' * 78)
    for i, r in enumerate(rows):
        cov = f'{r["pattern_coverage"]}/128'
        print(f'{i+1:>4d}  {r["variant"]:<22s}  '
              f'{r["val_force_mse"]:>10.4f}  '
              f'{cov:>6s}  '
              f'{r.get("reactivity_acc",0):>6.3f}  '
              f'{r.get("concept_acc",0):>7.3f}  '
              f'{r.get("n_params",0):>8d}  '
              f'{r.get("elapsed_s",0):>7.1f}')
    best = rows[0]
    print(f'\nBEST: {best["variant"]}  '
          f'force_mse={best["val_force_mse"]:.4f}')

    # Save the best non-ensemble model
    non_ensemble = [r for r in rows if r['variant'] != 'V15_ensemble']
    best_real = sorted(non_ensemble, key=lambda r: r['val_force_mse'])[0]
    if best_real['variant'] in saved_models:
        model, m_kw = saved_models[best_real['variant']]
        ckpt_path = REPO / 'cell_sim' / 'atom_engine' / 'checkpoints' / 'sparse_cell_gnn_v2_best.pt'
        torch.save({
            'state_dict': model.state_dict(),
            'config': m_kw,
            'val_force_mse': best_real['val_force_mse'],
            'pattern_coverage': best_real['pattern_coverage'],
            'force_norm_stats': stats,
            'variant': best_real['variant'],
        }, ckpt_path)
        print(f'\nsaved best non-ensemble -> {ckpt_path}  '
              f'(variant={best_real["variant"]}, mse={best_real["val_force_mse"]:.4f})')


if __name__ == '__main__':
    main()
