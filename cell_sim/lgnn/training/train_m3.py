"""M3 training — Source-1-complete multi-task supervision.

Extends train_m2 with two changes:

  1.  Two-channel input: count + per-species cross-replicate std.
      Variance is precomputed once over the preloaded train tensor:
      sigma[t, s] = train_data[:, t, s].std(dim=0). The variance channel
      is signed-log1p-transformed for numerical compatibility with the
      count channel.

  2.  Multi-task loss with per-category weighting. The 8572 species
      decompose into:
        - F_* fluxes (175 rows): reaction rates — direct supervision
          target the simulator outputs every step. Weighted up since
          they're a tiny fraction of total rows.
        - PM_*, RPM_*, DM_* cumulative event counters (1400 rows):
          their delta IS the synthesis / membrane-translation /
          degradation rate. Weighted up for the same reason.
        - everything else: counts (G, R, P, D, singletons). Default
          weight 1.0.

The model architecture (CellGNNv3) is a drop-in extension of M2 with the
2-channel input projection. All other layers (CfC, axis-2 attention, edge
dropout, attention warmup) are unchanged.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.models.gnn_v3 import CellGNNv3, count_parameters
from cell_sim.lgnn.training.train_m1_axis2_fast import (
    DEFAULT_K_CURRICULUM,
    _gather_windows,
    _index_iterator,
    _torch_dtype,
    _warmup_ramp,
    preload_to_gpu,
)


@dataclass
class M3TrainConfig:
    n_species: int = 8572
    hidden: int = 64
    n_layers: int = 3
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-5
    train_replicates: tuple = tuple(range(1, 50))
    val_replicates: tuple = (50,)
    n_epochs: int = 4
    device: str = 'auto'
    seed: int = 42
    log_every: int = 50
    species_filter: Optional[List[str]] = None

    edge_dropout_p: float = 0.07
    lambda_dropout: float = 0.1
    lambda_dropout_warmup_steps: int = 2000
    lambda_dropout_ramp_steps: int = 2000
    lambda_attn: float = 1e-4
    lambda_attn_warmup_steps: int = 2000
    lambda_attn_ramp_steps: int = 2000

    k_curriculum: tuple = DEFAULT_K_CURRICULUM
    rollout_gamma: float = 0.95
    lambda_rollout: float = 1.0
    max_k: int = 4
    skip_rollout_at_k1: bool = True

    use_checkpoint: bool = False
    use_bf16: bool = True
    preload_dtype: str = 'bfloat16'
    edge_chunk_size: Optional[int] = None

    cfc_tau_min: float = 0.1

    use_compile: bool = False
    compile_mode: str = 'default'

    wall_clock_budget_s: float = 2 * 3600.0

    # ---- M3-specific ----
    # Per-category loss weights. Defaults equalise each category's
    # contribution to the total: a flux row gets ~40x the gradient of a
    # count row (175 vs 6997), a cumulative-counter row gets ~5x.
    weight_count:      float = 1.0
    weight_flux:       float = 40.0
    weight_cumulative: float = 5.0

    # Set to 1 to disable the variance channel and exactly recover M2's
    # architecture (useful for ablation).
    n_input_channels: int = 2


def _device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _k_for_epoch(curriculum: tuple, epoch: int) -> int:
    return int(curriculum[min(epoch, len(curriculum) - 1)])


def categorise_row_indices(row_names: Sequence[str]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partition row indices into (count, flux, cumulative) masks.

    Returns three torch BoolTensors of shape (S,), each True where the
    row belongs to that category. The three masks are mutually
    exclusive and together cover all S rows.
    """
    S = len(row_names)
    flux_mask = torch.zeros(S, dtype=torch.bool)
    cum_mask  = torch.zeros(S, dtype=torch.bool)
    for i, n in enumerate(row_names):
        if n.startswith('F_'):
            flux_mask[i] = True
        elif n.startswith(('PM_', 'RPM_', 'DM_')):
            cum_mask[i] = True
    count_mask = ~(flux_mask | cum_mask)
    return count_mask, flux_mask, cum_mask


def compute_variance_channel(train_data: torch.Tensor) -> torch.Tensor:
    """Per-species cross-replicate std, signed-log1p-transformed.

    train_data: (R, T, S) signed-log1p counts on device.
    Returns:    (T, S) std-of-signed-log1p, also signed-log1p, on the
                 same device + dtype.
    """
    # std along replicate axis. Result is always nonneg, so signed-log1p
    # collapses to plain log1p.
    sigma = train_data.float().std(dim=0)                  # (T, S)
    sigma = torch.log1p(sigma).to(train_data.dtype)
    return sigma


def _multi_task_mse(
    pred: torch.Tensor, target: torch.Tensor,
    count_mask: torch.Tensor, flux_mask: torch.Tensor, cum_mask: torch.Tensor,
    w_count: float, w_flux: float, w_cum: float,
) -> Tuple[torch.Tensor, dict]:
    """Per-category mean-squared-error with weighting.

    Returns (total_loss, breakdown_dict) where breakdown has the three
    per-category mean-MSE numbers for logging.
    """
    diff_sq = (pred - target).pow(2)                       # (B, S)
    # Mean over batch and over the category's rows.
    n_count = int(count_mask.sum()); n_flux = int(flux_mask.sum())
    n_cum   = int(cum_mask.sum())
    mse_count = diff_sq[:, count_mask].mean() if n_count > 0 else \
                  torch.zeros((), device=pred.device, dtype=pred.dtype)
    mse_flux  = diff_sq[:, flux_mask ].mean() if n_flux  > 0 else \
                  torch.zeros((), device=pred.device, dtype=pred.dtype)
    mse_cum   = diff_sq[:, cum_mask  ].mean() if n_cum   > 0 else \
                  torch.zeros((), device=pred.device, dtype=pred.dtype)
    total = w_count * mse_count + w_flux * mse_flux + w_cum * mse_cum
    return total, {
        'mse_count': mse_count.detach(),
        'mse_flux':  mse_flux.detach(),
        'mse_cum':   mse_cum.detach(),
    }


def train_m3(
    cfg: M3TrainConfig,
    lsdata_module,
    graph: SpeciesGraph,
    row_names: Sequence[str],
    checkpoint_path: Optional[Path] = None,
) -> dict:
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    pre_dtype = _torch_dtype(cfg.preload_dtype)

    # --- Preload corpus ---
    t0 = time.time()
    print(f'preloading {len(cfg.train_replicates)} train + '
          f'{len(cfg.val_replicates)} val replicates to {device} '
          f'as {cfg.preload_dtype}...')
    train_data = preload_to_gpu(cfg.train_replicates, lsdata_module,
                                  device, dtype=pre_dtype,
                                  species_filter=cfg.species_filter)
    val_data = preload_to_gpu(cfg.val_replicates, lsdata_module,
                                device, dtype=pre_dtype,
                                species_filter=cfg.species_filter)
    R, T, S = train_data.shape
    n_valid = T - cfg.max_k - 1
    print(f'  shape: train{tuple(train_data.shape)}  val{tuple(val_data.shape)}'
          f'   load wall {time.time()-t0:.1f}s')

    # --- Row categorisation ---
    count_mask, flux_mask, cum_mask = categorise_row_indices(row_names)
    count_mask = count_mask.to(device); flux_mask = flux_mask.to(device)
    cum_mask   = cum_mask  .to(device)
    print(f'rows: count={int(count_mask.sum())}  '
          f'flux={int(flux_mask.sum())}  '
          f'cumulative={int(cum_mask.sum())}')
    print(f'category weights: count={cfg.weight_count}  '
          f'flux={cfg.weight_flux}  '
          f'cumulative={cfg.weight_cumulative}')

    # --- Variance channel ---
    print('precomputing per-species cross-replicate std...')
    sigma = compute_variance_channel(train_data)         # (T, S)
    sigma_val = compute_variance_channel(
        torch.cat([train_data, val_data], dim=0)
    )
    # In practice for val we use the same train-derived sigma; val data
    # is only 1 replicate so it has zero replicate-variance.
    print(f'  sigma shape: {tuple(sigma.shape)}, '
          f'mean: {sigma.float().mean().item():.3f}, '
          f'max: {sigma.float().max().item():.3f}')

    # --- Model ---
    model = CellGNNv3(
        graph=graph, hidden=cfg.hidden, n_layers=cfg.n_layers,
        n_input_channels=cfg.n_input_channels,
        use_checkpoint=cfg.use_checkpoint,
        edge_chunk_size=cfg.edge_chunk_size,
        cfc_tau_min=cfg.cfc_tau_min,
    ).to(device)
    if cfg.use_bf16 and device.type == 'cuda':
        model = model.to(torch.bfloat16)
    model_dtype = next(model.parameters()).dtype
    n_params = count_parameters(model)
    print(f'M3: {n_params:,} parameters '
          f'(hidden={cfg.hidden}, n_layers={cfg.n_layers}, '
          f'channels={cfg.n_input_channels}, dtype={model_dtype})')

    if cfg.use_compile and device.type == 'cuda':
        try:
            model = torch.compile(model, mode=cfg.compile_mode)
        except Exception as e:
            print(f'  WARNING: torch.compile failed ({e}); continuing uncompiled')

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)

    history = {
        'train_total': [], 'train_mse_count': [], 'train_mse_flux': [],
        'train_mse_cum': [], 'train_rollout': [], 'train_attn_entropy': [],
        'val_singlestep_mse': [], 'val_mse_count': [], 'val_mse_flux': [],
        'val_mse_cum': [], 'val_rollout_mse_avg': [],
        'val_mse_per_step': [], 'k_per_epoch': [], 'samples_per_sec': [],
    }
    best_val = float('inf')
    train_t0 = time.time()
    global_step = 0

    for epoch in range(cfg.n_epochs):
        k_cur = _k_for_epoch(cfg.k_curriculum, epoch)
        history['k_per_epoch'].append(k_cur)
        eff_lambda_rollout = (0.0 if (k_cur == 1 and cfg.skip_rollout_at_k1)
                                else cfg.lambda_rollout)

        gen = torch.Generator(device=device).manual_seed(cfg.seed + epoch)
        model.train()
        run = {k: torch.zeros((), device=device, dtype=torch.float32)
                for k in ('total', 'mse_count', 'mse_flux', 'mse_cum',
                          'rollout', 'attn_entropy')}
        n_batches = 0
        t_epoch = time.time()
        n_seen = 0

        for rep_idx, t_idx in _index_iterator(
            R, n_valid, cfg.batch_size, gen, device,
        ):
            x_w = _gather_windows(train_data, rep_idx, t_idx, k_cur + 1
                                    ).to(model_dtype)
            x = x_w[:, 0, :]
            # Pull the variance channel for the batch's first timestep.
            v0 = sigma[t_idx].to(model_dtype)                  # (B, S)

            # 1. Multi-step rollout with variance from the rollout step's t
            x_pred = x
            attn_entropy = None
            L_rollout = torch.zeros((), device=device, dtype=model_dtype)
            L_full_ss = None
            mse_breakdown_step0 = None
            gamma_sum = 0.0
            for s in range(k_cur):
                # Variance channel at the current rollout time.
                v_cur = sigma[t_idx + s].to(model_dtype)
                if s == 0:
                    dx, ent = model(
                        x_pred, x_var=v_cur, return_attention_entropy=True,
                    )
                    attn_entropy = ent
                else:
                    dx = model(x_pred, x_var=v_cur)
                x_pred = x_pred + dx
                target = x_w[:, s + 1, :]
                step_loss, breakdown = _multi_task_mse(
                    x_pred, target, count_mask, flux_mask, cum_mask,
                    cfg.weight_count, cfg.weight_flux, cfg.weight_cumulative,
                )
                if s == 0:
                    L_full_ss = step_loss
                    mse_breakdown_step0 = breakdown
                w = cfg.rollout_gamma ** s
                L_rollout = L_rollout + w * step_loss
                gamma_sum += w
            L_rollout = L_rollout / max(gamma_sum, 1e-12)

            # 2. Warmup schedules + dropout pass
            cur_lambda_attn = _warmup_ramp(
                global_step, cfg.lambda_attn,
                cfg.lambda_attn_warmup_steps, cfg.lambda_attn_ramp_steps,
            )
            cur_lambda_dropout = _warmup_ramp(
                global_step, cfg.lambda_dropout,
                cfg.lambda_dropout_warmup_steps, cfg.lambda_dropout_ramp_steps,
            )

            if cur_lambda_dropout > 0.0:
                dx_drop = model(x, x_var=v0, edge_dropout_p=cfg.edge_dropout_p)
                L_drop_ss, _ = _multi_task_mse(
                    x + dx_drop, x_w[:, 1, :], count_mask, flux_mask, cum_mask,
                    cfg.weight_count, cfg.weight_flux, cfg.weight_cumulative,
                )
                drop_term = cur_lambda_dropout * L_drop_ss
            else:
                L_drop_ss = L_full_ss.detach()
                drop_term = torch.zeros((), device=device, dtype=L_full_ss.dtype)

            # 3. Combined loss
            if eff_lambda_rollout == 0.0:
                supervised = L_full_ss
            else:
                supervised = eff_lambda_rollout * L_rollout
            L = (supervised + drop_term + cur_lambda_attn * attn_entropy)

            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()
            global_step += 1

            run['total']        += L.detach().float()
            run['mse_count']    += mse_breakdown_step0['mse_count'].float()
            run['mse_flux']     += mse_breakdown_step0['mse_flux'].float()
            run['mse_cum']      += mse_breakdown_step0['mse_cum'].float()
            run['rollout']      += L_rollout.detach().float()
            run['attn_entropy'] += attn_entropy.detach().float()
            n_batches += 1
            n_seen += int(rep_idx.shape[0])

            if (n_batches) % cfg.log_every == 0:
                wall = time.time() - t_epoch
                vals = {k: float(v.item()) / n_batches for k, v in run.items()}
                print(f'  ep{epoch} k={k_cur} step{n_batches:>5d}'
                      f'  total={vals["total"]:.4f}'
                      f'  count={vals["mse_count"]:.4f}'
                      f'  flux={vals["mse_flux"]:.4f}'
                      f'  cum={vals["mse_cum"]:.4f}'
                      f'  rollout={vals["rollout"]:.4f}'
                      f'  λa={cur_lambda_attn:.1e}'
                      f'  {n_seen/wall:.0f} s/s')

            if (time.time() - train_t0) > cfg.wall_clock_budget_s:
                print(f'  WARNING: wall-clock budget '
                      f'{cfg.wall_clock_budget_s/3600:.2f}h exceeded')
                break

        epoch_wall = time.time() - t_epoch
        history['samples_per_sec'].append(n_seen / max(epoch_wall, 1e-6))
        for key in ('total', 'mse_count', 'mse_flux', 'mse_cum',
                     'rollout', 'attn_entropy'):
            history[f'train_{key}'].append(
                float(run[key].item()) / max(n_batches, 1)
            )

        # --- Validation ---
        model.eval()
        val_metrics = _evaluate(
            model, val_data, sigma, count_mask, flux_mask, cum_mask, cfg,
            model_dtype, device,
        )
        for k, v in val_metrics.items():
            history[f'val_{k}'].append(v)
        val_ss = val_metrics['singlestep_mse']
        print(f'ep{epoch} k={k_cur} val singlestep_mse={val_ss:.4f}  '
              f'count={val_metrics["mse_count"]:.4f}  '
              f'flux={val_metrics["mse_flux"]:.4f}  '
              f'cum={val_metrics["mse_cum"]:.4f}  '
              f'rollout_avg={val_metrics["rollout_mse_avg"]:.4f}  '
              f'wall={epoch_wall:.1f}s')

        if val_ss < best_val:
            best_val = val_ss
            if checkpoint_path is not None:
                payload = {
                    'state_dict': (model._orig_mod.state_dict()
                                    if hasattr(model, '_orig_mod')
                                    else model.state_dict()),
                    'cfg': cfg.__dict__,
                    'epoch': epoch,
                    'val_singlestep_mse': val_ss,
                    'val_rollout_mse_avg': val_metrics['rollout_mse_avg'],
                    'val_mse_per_step': val_metrics['mse_per_step'],
                    'k_at_save': k_cur,
                }
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, checkpoint_path)
                print(f'  -> saved best (val_ss={val_ss:.4f}) to {checkpoint_path}')

    return history


@torch.no_grad()
def _evaluate(model, val_data, sigma,
              count_mask, flux_mask, cum_mask,
              cfg, model_dtype, device):
    R_v, T_v, S = val_data.shape
    n_valid = T_v - cfg.max_k - 1
    bs = cfg.batch_size
    mse_per_step = [0.0] * cfg.max_k
    cnt_per_step = 0
    total_full_count = 0.0
    total_full_flux  = 0.0
    total_full_cum   = 0.0
    n_full = 0
    for t_start in range(0, n_valid, bs):
        t_idx = torch.arange(t_start, min(t_start + bs, n_valid), device=device)
        rep_idx = torch.zeros_like(t_idx)
        x_w = _gather_windows(val_data, rep_idx, t_idx, cfg.max_k + 1
                                ).to(model_dtype)
        x = x_w[:, 0, :]
        v_cur = sigma[t_idx].to(model_dtype)
        x_pred = x
        for s in range(cfg.max_k):
            v_s = sigma[t_idx + s].to(model_dtype)
            dx = model(x_pred, x_var=v_s)
            x_pred = x_pred + dx
            tgt = x_w[:, s + 1, :]
            mse_per_step[s] += float((x_pred - tgt).pow(2).mean().item())
            if s == 0:
                _, br = _multi_task_mse(
                    x_pred, tgt, count_mask, flux_mask, cum_mask,
                    cfg.weight_count, cfg.weight_flux, cfg.weight_cumulative,
                )
                total_full_count += float(br['mse_count'].item())
                total_full_flux  += float(br['mse_flux'] .item())
                total_full_cum   += float(br['mse_cum']  .item())
                n_full += 1
        cnt_per_step += 1
    mse_per_step = [m / max(cnt_per_step, 1) for m in mse_per_step]
    rollout_avg = sum(mse_per_step) / len(mse_per_step)
    return {
        'singlestep_mse':    mse_per_step[0],
        'mse_count':         total_full_count / max(n_full, 1),
        'mse_flux':          total_full_flux  / max(n_full, 1),
        'mse_cum':           total_full_cum   / max(n_full, 1),
        'rollout_mse_avg':   rollout_avg,
        'mse_per_step':      mse_per_step,
    }
