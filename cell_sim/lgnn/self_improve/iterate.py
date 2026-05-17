"""M7 self-improvement loop: iterative weakness-targeted fine-tuning.

The deterministic noise floor (val_count ≈ 0.0716) bounds how low MSE
can go regardless of training compute. Within that bound, however,
M7's errors are not uniform — some species, timesteps, and replicates
are systematically harder than others. Self-improvement exploits this:

   1. Profile the model's predictions on val; find the worst-performing
      species/timestep/replicate buckets.
   2. Re-sample training pairs with a strong bias toward those weak
      regions (70 / 30 split: 70% biased, 30% uniform).
   3. Fine-tune briefly (~2,000 steps) from the current weights.
   4. Re-evaluate. Accept the new weights if val MSE improved; reject
      and keep the prior weights otherwise.
   5. Loop.

Practical outcome on M7.7 (deterministic, AUROC plateau at 0.64):
expect 2-5% reduction in per-species worst-case MSE per iteration for
2-3 rounds, then a plateau. The loop will detect the plateau and stop.

This is NOT a way to break the noise floor — that requires a
stochastic head (M8). It IS a way to reduce systematic bias in the
deterministic prediction.

Usage (script):
    python -m cell_sim.lgnn.self_improve.iterate \\
        --initial /content/drive/MyDrive/cell_count_dynamics/m7_7_best.pt \\
        --output  /content/drive/MyDrive/cell_count_dynamics/self_improve/ \\
        --n_iterations 5

Usage (programmatic):
    from cell_sim.lgnn.self_improve.iterate import self_improvement_loop
    final_path, history = self_improvement_loop(initial_path=..., ...)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cell_sim.lgnn.data.species_graph import load_species_graph
from cell_sim.lgnn.data.stoichiometric_matrix import (
    load_stoichiometric_matrix_for_pinn,
)
from cell_sim.lgnn.models.gnn_v7_hybrid import CellGNNv7Hybrid
from cell_sim.lgnn.training.train_m7 import (
    _build_combined_static_features,
    _m7_step_loss,
    categorise_row_indices,
)


# ============================================================
# Model loader (mirrors explorer_app, but bare for this module)
# ============================================================
def _override_paths_for_drive(cfg_d: dict) -> dict:
    """Patch the saved cfg's paths to point at current Drive locations.
    Old M7.* checkpoints reference /content/cell/.../input_data which
    may not exist in this session."""
    drive_input = '/content/drive/MyDrive/Minimal_Cell_4DWCM/extracted/Minimal_Cell_4DWCM/CODE/input_data'
    drive_analys = '/content/drive/MyDrive/Minimal_Cell_4DWCM/extracted/Minimal_Cell_4DWCM/CODE/analysis'
    drive_dyn = '/content/drive/MyDrive/cell_count_dynamics'
    updates = {
        'sbml_path_for_S':            f'{drive_input}/Syn3A_updated.xml',
        'spatial_parquet':            f'{drive_dyn}/syn3a_spatial_features.parquet',
        'proteomics_xlsx':            f'{drive_input}/initial_concentrations.xlsx',
        'kinetic_params_xlsx':        f'{drive_input}/kinetic_params_new.xlsx',
        'medium_xlsx':                f'{drive_analys}/GIP_Gibbs_Concs_Fluxes/Medium.xlsx',
        'protein_metabolites_xlsx':   f'{drive_input}/protein_metabolites.xlsx',
        'large_subunit_xlsx':         f'{drive_input}/LargeSubunit.xlsx',
        'gibbs_csv_path':             f'{drive_dyn}/gibbs.csv',
        'complex_formation_xlsx':     '',
        'use_complex_constraints':    False,
    }
    cfg_d.update(updates)
    return cfg_d


def load_m7_with_state(checkpoint_path: str, sg, row_names, device='cuda'):
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_d = _override_paths_for_drive(dict(obj['cfg']))
    S_raw, fi_raw, _, _ = load_stoichiometric_matrix_for_pinn(
        cfg_d['sbml_path_for_S'], row_names,
    )
    S    = (S_raw if isinstance(S_raw, torch.Tensor)
            else torch.from_numpy(S_raw)).to(device)
    fidx = (fi_raw if isinstance(fi_raw, torch.Tensor)
            else torch.from_numpy(fi_raw)).cpu()
    stat = _build_combined_static_features(
        row_names,
        use_role=cfg_d.get('use_role_features', True),
        use_spatial=cfg_d.get('use_spatial_features', False),
        use_proteomics=cfg_d.get('use_proteomics_features', False),
        use_kinetic_priors=cfg_d.get('use_kinetic_priors', False),
        use_complex_constraints=False,
        use_medium_features=cfg_d.get('use_medium_features', False),
        use_regulatory_features=cfg_d.get('use_regulatory_features', False),
        use_ribosome_subunit=cfg_d.get('use_ribosome_subunit', False),
        use_thermodynamics=cfg_d.get('use_thermodynamics', False),
        spatial_parquet=cfg_d.get('spatial_parquet'),
        proteomics_xlsx=cfg_d.get('proteomics_xlsx'),
        gene_class_csv=cfg_d.get('gene_class_csv'),
        kinetic_params_xlsx=cfg_d.get('kinetic_params_xlsx'),
        complex_formation_xlsx='',
        medium_xlsx=cfg_d.get('medium_xlsx'),
        protein_metabolites_xlsx=cfg_d.get('protein_metabolites_xlsx'),
        large_subunit_xlsx=cfg_d.get('large_subunit_xlsx'),
        gibbs_csv_path=cfg_d.get('gibbs_csv_path'),
        verbose=False,
    )
    m = CellGNNv7Hybrid(
        graph=sg, stoich_matrix=S, flux_indices=fidx,
        hidden=cfg_d.get('hidden', 64),
        n_layers=cfg_d.get('n_layers', 3),
        n_input_channels=cfg_d.get('n_input_channels', 2),
        static_features=stat,
        use_checkpoint=cfg_d.get('use_checkpoint', True),
        edge_chunk_size=cfg_d.get('edge_chunk_size'),
        cfc_tau_min=cfg_d.get('cfc_tau_min', 0.1),
        rate_clip=cfg_d.get('rate_clip', 6.0),
        use_residual=cfg_d.get('use_residual', True),
        multi_step_horizon=cfg_d.get('multi_step_horizon', 1),
    ).to(device)
    if cfg_d.get('use_bf16', True) and device == 'cuda':
        m = m.to(torch.bfloat16)
    m.load_state_dict(obj['state_dict'], strict=False)
    return m, cfg_d, fidx


# ============================================================
# Weakness analyzer
# ============================================================
def analyze_model_weaknesses(
    model,
    val_data: torch.Tensor,                  # (R_val, T, S)
    sigma: torch.Tensor,                     # (T, S)
    row_names: List[str],
    *,
    t_step_stride: int = 50,
    t_skip_initial: int = 10,
    top_k_species: int = 50,
    top_k_timesteps: int = 50,
) -> Dict[str, Any]:
    """Profile the model's prediction errors on val.

    Computes per-species and per-timestep mean-squared error on
    single-step predictions, and identifies the worst regions.
    Subsamples timesteps with stride to keep this fast (~10-30 sec).
    """
    model = model.eval().to(torch.float32)
    device = val_data.device
    R_val, T, S = val_data.shape
    n_timesteps_sampled = max(1, (T - t_skip_initial - 1) // t_step_stride)

    # Accumulate per-species squared error
    se_per_species = torch.zeros(S, device=device, dtype=torch.float32)
    se_per_timestep: List[float] = []
    timestep_positions: List[int] = []
    n_seen = 0

    print(f'  profiling: stride={t_step_stride}  '
          f'~{n_timesteps_sampled} timesteps sampled per replicate')
    t0 = time.time()
    for rep in range(R_val):
        for t in range(t_skip_initial, T - 1, t_step_stride):
            x_in = val_data[rep, t].unsqueeze(0).to(torch.float32)
            v_in = sigma[t].unsqueeze(0).to(torch.float32)
            with torch.no_grad():
                out = model(x_in, x_var=v_in)
            x_pred = (out[0] if isinstance(out, tuple) else out).squeeze(0)
            x_actual = val_data[rep, t + 1].to(torch.float32)
            sq = (x_pred - x_actual).pow(2)                 # (S,)
            se_per_species += sq
            se_per_timestep.append(float(sq.mean().item()))
            timestep_positions.append(t)
            n_seen += 1
    elapsed = time.time() - t0

    mse_per_species = (se_per_species / max(n_seen, 1)).cpu().numpy()
    mse_per_timestep = np.array(se_per_timestep)
    positions = np.array(timestep_positions)

    # Top-K weakest species
    worst_sp_idx = np.argsort(-mse_per_species)[:top_k_species].tolist()
    worst_sp_names = [row_names[i] for i in worst_sp_idx]
    worst_sp_mse = mse_per_species[worst_sp_idx].tolist()

    # Top-K weakest timesteps
    worst_t_idx = np.argsort(-mse_per_timestep)[:top_k_timesteps].tolist()
    worst_t_pos = positions[worst_t_idx].tolist()
    worst_t_mse = mse_per_timestep[worst_t_idx].tolist()

    # Per-prefix breakdown (count, flux, cum)
    count_mask, flux_mask, cum_mask = categorise_row_indices(row_names)
    mse_count = float(mse_per_species[count_mask.cpu().numpy()].mean())
    mse_flux  = float(mse_per_species[flux_mask.cpu().numpy()].mean()) if flux_mask.any() else 0.0
    mse_cum   = float(mse_per_species[cum_mask.cpu().numpy()].mean()) if cum_mask.any() else 0.0

    print(f'  profile done in {elapsed:.1f}s')
    print(f'  per-prefix MSE: count={mse_count:.4f}  flux={mse_flux:.4f}  cum={mse_cum:.4f}')
    print(f'  worst species (by MSE):')
    for n, m in zip(worst_sp_names[:8], worst_sp_mse[:8]):
        print(f'    {n[:35]:35s}  mse={m:.4f}')
    print(f'  worst timesteps (by MSE):  '
          f'{[(p, f"{m:.3f}") for p, m in zip(worst_t_pos[:5], worst_t_mse[:5])]}')

    return {
        'mse_per_species': mse_per_species,
        'mse_per_timestep': mse_per_timestep,
        'timestep_positions': positions,
        'worst_species_idx': worst_sp_idx,
        'worst_species_names': worst_sp_names,
        'worst_species_mse': worst_sp_mse,
        'worst_timesteps_idx': worst_t_pos,
        'worst_timesteps_mse': worst_t_mse,
        'val_mse_count': mse_count,
        'val_mse_flux':  mse_flux,
        'val_mse_cum':   mse_cum,
        'val_mse_total': float(mse_per_species.mean()),
        'n_sampled': n_seen,
    }


# ============================================================
# Biased fine-tuner
# ============================================================
def _build_weighted_t_idx(
    weakness: Dict[str, Any],
    train_data_T: int,
    batch_size: int,
    n_steps: int,
    bias_strength: float = 0.7,
    t_skip_initial: int = 10,
    device='cuda',
):
    """Generator: yield (rep_idx, t_idx) where t_idx is biased toward
    weak timesteps with probability `bias_strength`, uniform otherwise."""
    weak_pos = np.asarray(weakness['worst_timesteps_idx'], dtype=np.int64)
    weak_pos = weak_pos[(weak_pos >= t_skip_initial)
                        & (weak_pos < train_data_T - 1)]
    if len(weak_pos) == 0:
        weak_pos = np.array([t_skip_initial])
    n_weak = len(weak_pos)
    valid_t_lo = t_skip_initial
    valid_t_hi = train_data_T - 1
    for _ in range(n_steps):
        # Decide which samples in this batch are biased vs uniform
        mask = torch.rand(batch_size, device=device) < bias_strength
        # Biased: pick from weak positions, add small random jitter ±20
        biased_choices = np.random.choice(weak_pos, size=batch_size)
        jitter = np.random.randint(-20, 21, size=batch_size)
        biased_t = np.clip(biased_choices + jitter, valid_t_lo, valid_t_hi - 1)
        # Uniform fallback
        uniform_t = np.random.randint(valid_t_lo, valid_t_hi, size=batch_size)
        # Combine
        biased_t_t = torch.from_numpy(biased_t).to(device)
        uniform_t_t = torch.from_numpy(uniform_t).to(device)
        t_idx = torch.where(mask, biased_t_t, uniform_t_t)
        yield t_idx


def fine_tune_on_weaknesses(
    model,
    train_data: torch.Tensor,
    sigma: torch.Tensor,
    val_data: torch.Tensor,
    row_names: List[str],
    flux_indices: torch.Tensor,
    weakness: Dict[str, Any],
    *,
    n_steps: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-4,
    bias_strength: float = 0.7,
    t_skip_initial: int = 10,
    weight_count: float = 0.05,
    weight_flux: float = 1.0,
    weight_cumulative: float = 0.5,
    log_every: int = 100,
):
    """Lightweight fine-tune with weakness-biased sampling.

    Each batch:
      - 70% of samples drawn from t-positions where the model is
        currently worst (jittered ±20).
      - 30% drawn uniformly across the full valid range.
    This keeps the model from forgetting the easy regions while
    concentrating gradient on the hard ones.
    """
    device = train_data.device
    R_train, T, S = train_data.shape

    count_mask, flux_mask, cum_mask = categorise_row_indices(row_names)
    # Ensure all index tensors live on the same device as the data so
    # downstream index_select / boolean-mask ops in _m7_step_loss don't
    # raise "tensors on different devices" (load_m7_with_state returns
    # flux_indices on CPU by convention).
    flux_indices = flux_indices.to(device)
    count_mask = count_mask.to(device) if hasattr(count_mask, 'to') else count_mask
    flux_mask  = flux_mask.to(device)  if hasattr(flux_mask,  'to') else flux_mask
    cum_mask   = cum_mask.to(device)   if hasattr(cum_mask,   'to') else cum_mask

    model.train()
    # Keep encoder in fp32 for fine-tune stability
    model = model.to(torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t_gen = _build_weighted_t_idx(
        weakness, T, batch_size, n_steps,
        bias_strength=bias_strength,
        t_skip_initial=t_skip_initial,
        device=device,
    )
    ema_loss = None
    t0 = time.time()
    for step, t_idx in enumerate(t_gen):
        rep_idx = torch.randint(0, R_train, (batch_size,), device=device)
        x_in = train_data[rep_idx, t_idx].to(torch.float32)
        x_actual = train_data[rep_idx, t_idx + 1].to(torch.float32)
        v_in = sigma[t_idx].to(torch.float32)
        x_pred, v_log = model(x_in, x_var=v_in)
        target = x_actual
        L, _ = _m7_step_loss(
            x_pred, v_log, target, flux_indices,
            count_mask, cum_mask,
            weight_count, weight_flux, weight_cumulative,
        )
        opt.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        L_v = float(L.detach().item())
        ema_loss = L_v if ema_loss is None else 0.95 * ema_loss + 0.05 * L_v
        if (step + 1) % log_every == 0:
            rate = (step + 1) / (time.time() - t0)
            print(f'    step {step+1:>5d}/{n_steps}  loss={L_v:.4f}  '
                  f'ema={ema_loss:.4f}  {rate:.1f} steps/s')
    return model


# ============================================================
# Composite evaluator
# ============================================================
def compute_full_val_metrics(model, val_data, sigma, row_names,
                              t_skip_initial=10) -> Dict[str, float]:
    """Score model on val: single-step MSE, per-prefix breakdown,
    and 200-step rollout MSE. ~30 sec."""
    model = model.eval().to(torch.float32)
    R_val, T, S = val_data.shape
    count_mask, flux_mask, cum_mask = categorise_row_indices(row_names)
    # Move masks to val_data's device so boolean indexing works
    count_mask = count_mask.to(val_data.device) if hasattr(count_mask, 'to') else count_mask
    flux_mask  = flux_mask.to(val_data.device)  if hasattr(flux_mask,  'to') else flux_mask
    cum_mask   = cum_mask.to(val_data.device)   if hasattr(cum_mask,   'to') else cum_mask

    se_total = 0.0
    n_pairs = 0
    se_per_row = torch.zeros(S, device=val_data.device, dtype=torch.float32)
    # Single-step MSE at every 50 timesteps for speed
    for rep in range(R_val):
        for t in range(t_skip_initial, T - 1, 50):
            x_in = val_data[rep, t].unsqueeze(0).to(torch.float32)
            v_in = sigma[t].unsqueeze(0).to(torch.float32)
            with torch.no_grad():
                out = model(x_in, x_var=v_in)
            x_pred = (out[0] if isinstance(out, tuple) else out).squeeze(0)
            x_actual = val_data[rep, t + 1].to(torch.float32)
            sq = (x_pred - x_actual).pow(2)
            se_total += float(sq.mean().item())
            se_per_row += sq
            n_pairs += 1

    mse_per_row = se_per_row / max(n_pairs, 1)
    mse_count = float(mse_per_row[count_mask].mean().item())
    mse_flux  = float(mse_per_row[flux_mask].mean().item()) if flux_mask.any() else 0.0
    mse_cum   = float(mse_per_row[cum_mask].mean().item()) if cum_mask.any() else 0.0
    val_ss = se_total / max(n_pairs, 1)

    # 200-step rollout MSE from t=100
    x = val_data[0, 100].clone().to(torch.float32)
    diffs = []
    for s in range(200):
        v_in = sigma[100 + s].unsqueeze(0).to(torch.float32)
        with torch.no_grad():
            out = model(x.unsqueeze(0), x_var=v_in)
        x_pred = (out[0] if isinstance(out, tuple) else out).squeeze(0)
        target = val_data[0, 101 + s].to(torch.float32)
        diffs.append(float((x_pred - target).pow(2).mean().item()))
        x = x_pred
    rollout_mean = float(np.mean(diffs))

    return {
        'val_singlestep_mse': val_ss,
        'val_mse_count': mse_count,
        'val_mse_flux':  mse_flux,
        'val_mse_cum':   mse_cum,
        'val_rollout_200_mse': rollout_mean,
        'composite': val_ss + 0.3 * rollout_mean,  # used for accept/reject
    }


# ============================================================
# The self-improvement loop
# ============================================================
def self_improvement_loop(
    initial_path: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    sigma: torch.Tensor,
    sg,
    row_names: List[str],
    *,
    n_iterations: int = 5,
    fine_tune_steps: int = 2000,
    output_dir: str = './self_improve_output',
    early_stop_patience: int = 2,
    bias_strength: float = 0.7,
    device: str = 'cuda',
) -> Tuple[str, List[Dict[str, Any]]]:
    """Iteratively improve M7 by fine-tuning on its own weakest regions.

    Returns (final_checkpoint_path, history_list).
    """
    os.makedirs(output_dir, exist_ok=True)
    current_path = initial_path
    history: List[Dict[str, Any]] = []
    n_consecutive_rejects = 0

    # Establish baseline metrics
    print(f'{"=" * 60}\nSelf-improvement loop\n{"=" * 60}')
    print(f'Initial: {initial_path}')
    model0, cfg0, fidx0 = load_m7_with_state(initial_path, sg, row_names, device)
    baseline_metrics = compute_full_val_metrics(model0, val_data, sigma, row_names)
    print(f'baseline metrics:')
    for k, v in baseline_metrics.items():
        print(f'  {k}: {v:.4f}')
    best_composite = baseline_metrics['composite']
    history.append({
        'iter': 0, 'path': initial_path, 'accepted': True,
        'metrics': baseline_metrics, 'note': 'baseline',
    })
    del model0

    for it in range(1, n_iterations + 1):
        print(f'\n{"=" * 60}\nIteration {it}/{n_iterations}\n{"=" * 60}')
        # 1. Load + profile
        model, cfg, fidx = load_m7_with_state(current_path, sg, row_names, device)
        print('Step 1: profile weaknesses')
        weakness = analyze_model_weaknesses(
            model, val_data, sigma, row_names,
            t_step_stride=50,
        )
        with open(f'{output_dir}/weakness_iter{it}.json', 'w') as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray)
                            else (v.cpu().tolist() if isinstance(v, torch.Tensor)
                                   else v))
                       for k, v in weakness.items()},
                      f, indent=2, default=str)

        # 2. Fine-tune on weak regions
        print(f'Step 2: fine-tune {fine_tune_steps} steps with '
              f'{int(bias_strength * 100)}% bias toward weak regions')
        model = fine_tune_on_weaknesses(
            model, train_data, sigma, val_data, row_names, fidx, weakness,
            n_steps=fine_tune_steps, bias_strength=bias_strength,
        )

        # 3. Re-evaluate
        print('Step 3: evaluate fine-tuned model')
        new_metrics = compute_full_val_metrics(model, val_data, sigma, row_names)
        print(f'  new metrics:')
        for k, v in new_metrics.items():
            old = baseline_metrics[k] if it == 1 else history[-1]['metrics'][k]
            delta = v - old
            arrow = '↓' if delta < 0 else '↑'
            print(f'    {k}: {v:.4f}  ({arrow} {delta:+.4f})')

        # 4. Decide: accept if composite improved by >0.1%
        threshold = best_composite * 0.999
        accepted = new_metrics['composite'] < threshold
        new_path = f'{output_dir}/m7_self_iter{it}.pt'
        if accepted:
            torch.save({
                'state_dict': model.state_dict(),
                'cfg': cfg,
                'self_improve_metrics': new_metrics,
                'self_improve_iter': it,
                'self_improve_parent': current_path,
                'architecture': 'M7_hybrid',
                'reaction_ids': [],
                'flux_indices': fidx.cpu().tolist(),
            }, new_path)
            current_path = new_path
            best_composite = new_metrics['composite']
            n_consecutive_rejects = 0
            print(f'  ACCEPTED — saved to {new_path}')
        else:
            n_consecutive_rejects += 1
            print(f'  REJECTED — no significant improvement '
                  f'(consecutive rejects: {n_consecutive_rejects})')

        history.append({
            'iter': it, 'path': new_path if accepted else current_path,
            'accepted': accepted, 'metrics': new_metrics,
            'note': 'improved' if accepted else 'rejected',
        })

        with open(f'{output_dir}/history.json', 'w') as f:
            json.dump(history, f, indent=2, default=str)

        # Early stop
        if n_consecutive_rejects >= early_stop_patience:
            print(f'\nEarly stop: {n_consecutive_rejects} consecutive '
                  f'iterations without improvement.')
            break

        del model

    print(f'\n{"=" * 60}\nLoop complete.\n{"=" * 60}')
    print(f'Final checkpoint: {current_path}')
    print(f'Iterations: {len(history) - 1}')
    accepted = sum(1 for h in history if h['accepted']) - 1
    print(f'Accepted: {accepted}, Rejected: {len(history) - 1 - accepted}')
    if accepted > 0:
        delta = baseline_metrics['composite'] - best_composite
        pct = delta / baseline_metrics['composite'] * 100
        print(f'Total composite improvement: {delta:+.4f} ({pct:+.2f}%)')
    return current_path, history


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial', required=True,
                        help='Path to starting M7 checkpoint')
    parser.add_argument('--output', default='./self_improve_output',
                        help='Output directory for iter checkpoints + logs')
    parser.add_argument('--n_iterations', type=int, default=5)
    parser.add_argument('--fine_tune_steps', type=int, default=2000)
    parser.add_argument('--bias_strength', type=float, default=0.7)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--species_graph',
                        default='/content/drive/MyDrive/cell_count_dynamics/species_graph_v7.pt')
    parser.add_argument('--train_cache',
                        default='/content/drive/MyDrive/cell_count_dynamics/preload_train.pt')
    parser.add_argument('--val_cache',
                        default='/content/drive/MyDrive/cell_count_dynamics/preload_val.pt')
    parser.add_argument('--sigma',
                        default='/content/drive/MyDrive/cell_count_dynamics/sigma_train_reps_1to49.pt')
    args = parser.parse_args()

    print('Loading shared resources...')
    sg = load_species_graph(args.species_graph)
    row_names = list(sg.row_names)
    train_data = torch.load(args.train_cache, map_location=args.device,
                             weights_only=False)
    val_data = torch.load(args.val_cache, map_location=args.device,
                           weights_only=False)
    sigma = torch.load(args.sigma, map_location=args.device,
                        weights_only=False)
    print(f'  train: {tuple(train_data.shape)}  val: {tuple(val_data.shape)}  '
          f'sigma: {tuple(sigma.shape)}')

    final_path, history = self_improvement_loop(
        args.initial,
        train_data, val_data, sigma,
        sg, row_names,
        n_iterations=args.n_iterations,
        fine_tune_steps=args.fine_tune_steps,
        bias_strength=args.bias_strength,
        output_dir=args.output,
        device=args.device,
    )
    print(f'\nFinal model: {final_path}')


if __name__ == '__main__':
    main()
