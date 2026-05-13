"""M6 training - Physics-patched baseline.

Implements roadmap Step 1: thermodynamic target shift.

Instead of comparing the rolled-out state x_pred (= x_w[:,0] + sum(dx_i)) to
the ground-truth state x_w[:,s+1], compare the model's predicted delta dx
directly to the true per-step delta:

    delta_target = x_w[:, s+1, :] - x_w[:, s, :]
    step_loss    = MSE(dx, delta_target)

Mathematically related to the residual form (M3) but the gradient flow is
cleaner: the model learns "rate of change" per step rather than absorbing
both the local rate and the accumulated drift correction. This is the
"thermodynamic flux predictor" rather than "state memorizer" formulation
from the roadmap.

Note: the rollout still uses x_pred = x_pred + dx for the model's input at
the next step (so it sees its own drift), but the LOSS targets the local
true delta. The model is asked to predict the right delta at each step
even from drifted state.

Combined with M6 model (gnn_v6.CellGNNv6) which implements roadmap Steps
2 (softplus aggregation) and 3 (RATE_LAW/MASS_BALANCE edge split).

Step 4 (n_nodes purge) deferred to M7.
Step 5 (adaptive ODE) deferred to M8.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.models.gnn_v6 import CellGNNv6, count_parameters
from cell_sim.lgnn.training.train_m1_axis2_fast import (
    DEFAULT_K_CURRICULUM,
    _gather_windows,
    _index_iterator,
    _torch_dtype,
    _warmup_ramp,
    preload_to_gpu,
)
from cell_sim.lgnn.training.train_m3 import (
    M3TrainConfig,
    _device,
    _k_for_epoch,
    categorise_row_indices,
    compute_variance_channel,
    _multi_task_mse,
)


@dataclass
class M6TrainConfig(M3TrainConfig):
    """Identical to M3 config. M6 model + delta-target training loop.

    n_input_channels intentionally fixed to 1 - M6 model does NOT take a
    variance channel (per Flaw 4: variance-as-input is a "cheat code"
    in a deterministic solver). Step 4 of the roadmap re-introduces
    proper noise handling via Neural SDE; for M6 we use count only.
    """
    n_input_channels: int = 1


def train_m6(
    cfg: M6TrainConfig,
    lsdata_module,
    graph: SpeciesGraph,
    row_names: Sequence[str],
    checkpoint_path: Optional[Path] = None,
) -> dict:
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    pre_dtype = _torch_dtype(cfg.preload_dtype)

    # --- Preload ---
    t0 = time.time()
    print(f'preloading {len(cfg.train_replicates)} train + '
          f'{len(cfg.val_replicates)} val replicates to {device}...')
    train_data = preload_to_gpu(cfg.train_replicates, lsdata_module, device,
                                  dtype=pre_dtype, species_filter=cfg.species_filter)
    val_data = preload_to_gpu(cfg.val_replicates, lsdata_module, device,
                                dtype=pre_dtype, species_filter=cfg.species_filter)
    R, T, S = train_data.shape
    n_valid = T - cfg.max_k - 1
    print(f'  shape: train{tuple(train_data.shape)}  val{tuple(val_data.shape)}'
          f'   load wall {time.time()-t0:.1f}s')

    count_mask, flux_mask, cum_mask = categorise_row_indices(row_names)
    count_mask = count_mask.to(device); flux_mask = flux_mask.to(device)
    cum_mask   = cum_mask  .to(device)
    print(f'rows: count={int(count_mask.sum())}  '
          f'flux={int(flux_mask.sum())}  cumulative={int(cum_mask.sum())}')

    # --- Model (M6: CfC + softplus aggregation, no variance channel) ---
    model = CellGNNv6(
        graph=graph, hidden=cfg.hidden, n_layers=cfg.n_layers,
        use_checkpoint=cfg.use_checkpoint,
        edge_chunk_size=cfg.edge_chunk_size,
        cfc_tau_min=cfg.cfc_tau_min,
    ).to(device)
    if cfg.use_bf16 and device.type == 'cuda':
        model = model.to(torch.bfloat16)
    model_dtype = next(model.parameters()).dtype
    n_params = count_parameters(model)
    print(f'M6: {n_params:,} parameters '
          f'(hidden={cfg.hidden}, n_layers={cfg.n_layers}, dtype={model_dtype})')
    print(f'graph: n_nodes={graph.n_nodes}, n_edges={graph.n_edges}, '
          f'unique edge kinds: {sorted(set(graph.edge_kind.tolist()))}')

    if cfg.use_compile and device.type == 'cuda':
        try:
            model = torch.compile(model, mode=cfg.compile_mode)
        except Exception as e:
            print(f'  WARNING: torch.compile failed ({e})')

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
                for k in ('total','mse_count','mse_flux','mse_cum',
                          'rollout','attn_entropy')}
        n_batches = 0
        t_epoch = time.time()
        n_seen = 0

        for rep_idx, t_idx in _index_iterator(R, n_valid, cfg.batch_size,
                                                gen, device):
            x_w = _gather_windows(train_data, rep_idx, t_idx, k_cur + 1
                                    ).to(model_dtype)
            x = x_w[:, 0, :]

            # === STEP 1: delta-target rollout ===
            x_pred = x
            attn_entropy = None
            L_rollout = torch.zeros((), device=device, dtype=model_dtype)
            L_full_ss = None
            mse_breakdown_step0 = None
            gamma_sum = 0.0
            for s in range(k_cur):
                if s == 0:
                    dx, ent = model(x_pred, return_attention_entropy=True)
                    attn_entropy = ent
                else:
                    dx = model(x_pred)

                # ✦ STEP 1 (THE CHANGE) ✦
                # Compare the model's predicted dx directly to the
                # TRUE delta at step s, not to the absolute next state.
                delta_target = x_w[:, s + 1, :] - x_w[:, s, :]
                step_loss, breakdown = _multi_task_mse(
                    dx, delta_target, count_mask, flux_mask, cum_mask,
                    cfg.weight_count, cfg.weight_flux, cfg.weight_cumulative,
                )
                if s == 0:
                    L_full_ss = step_loss
                    mse_breakdown_step0 = breakdown

                # The rollout state still advances by adding dx, so the
                # model sees its own drift at step s+1.
                x_pred = x_pred + dx

                w = cfg.rollout_gamma ** s
                L_rollout = L_rollout + w * step_loss
                gamma_sum += w
            L_rollout = L_rollout / max(gamma_sum, 1e-12)

            cur_lambda_attn = _warmup_ramp(
                global_step, cfg.lambda_attn,
                cfg.lambda_attn_warmup_steps, cfg.lambda_attn_ramp_steps,
            )
            cur_lambda_dropout = _warmup_ramp(
                global_step, cfg.lambda_dropout,
                cfg.lambda_dropout_warmup_steps, cfg.lambda_dropout_ramp_steps,
            )

            if cur_lambda_dropout > 0.0:
                dx_drop = model(x, edge_dropout_p=cfg.edge_dropout_p)
                delta_target_0 = x_w[:, 1, :] - x_w[:, 0, :]
                L_drop_ss, _ = _multi_task_mse(
                    dx_drop, delta_target_0,
                    count_mask, flux_mask, cum_mask,
                    cfg.weight_count, cfg.weight_flux, cfg.weight_cumulative,
                )
                drop_term = cur_lambda_dropout * L_drop_ss
            else:
                L_drop_ss = L_full_ss.detach()
                drop_term = torch.zeros((), device=device,
                                          dtype=L_full_ss.dtype)

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

            if n_batches % cfg.log_every == 0:
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
                print(f'  WARNING: wall-clock budget exceeded')
                break

        epoch_wall = time.time() - t_epoch
        history['samples_per_sec'].append(n_seen / max(epoch_wall, 1e-6))
        for key in ('total', 'mse_count', 'mse_flux', 'mse_cum',
                     'rollout', 'attn_entropy'):
            history[f'train_{key}'].append(
                float(run[key].item()) / max(n_batches, 1)
            )

        # --- Validation (delta-target form) ---
        model.eval()
        val_metrics = _evaluate_m6(model, val_data, count_mask, flux_mask,
                                     cum_mask, cfg, model_dtype, device)
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
                    'target_form': 'delta',          # marker
                    'aggregation': 'softplus',       # marker
                }
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, checkpoint_path)
                print(f'  -> saved best (val_ss={val_ss:.4f}) to {checkpoint_path}')

    return history


@torch.no_grad()
def _evaluate_m6(model, val_data, count_mask, flux_mask, cum_mask,
                  cfg, model_dtype, device):
    """Validation using delta-target form.

    val_singlestep_mse is the per-step MSE between predicted dx and true delta
    (not between rolled-out state and ground-truth state). For comparability
    with M3, also report the rolled-out state MSE (rollout_avg).
    """
    R_v, T_v, S = val_data.shape
    n_valid = T_v - cfg.max_k - 1
    bs = cfg.batch_size
    mse_per_step_delta = [0.0] * cfg.max_k       # delta-target MSE per step
    mse_per_step_state = [0.0] * cfg.max_k       # state MSE per step (for M3 comp)
    cnt = 0
    total_full_count = 0.0
    total_full_flux  = 0.0
    total_full_cum   = 0.0
    n_full = 0
    for t_start in range(0, n_valid, bs):
        t_idx = torch.arange(t_start, min(t_start + bs, n_valid), device=device)
        rep_idx = torch.zeros_like(t_idx)
        x_w = _gather_windows(val_data, rep_idx, t_idx, cfg.max_k + 1
                                ).to(model_dtype)
        x_pred = x_w[:, 0, :]
        for s in range(cfg.max_k):
            dx = model(x_pred)
            delta_target = x_w[:, s + 1, :] - x_w[:, s, :]
            mse_per_step_delta[s] += float((dx - delta_target).pow(2).mean().item())
            x_pred = x_pred + dx
            mse_per_step_state[s] += float((x_pred - x_w[:, s + 1, :]).pow(2).mean().item())
            if s == 0:
                _, br = _multi_task_mse(
                    dx, delta_target, count_mask, flux_mask, cum_mask,
                    cfg.weight_count, cfg.weight_flux, cfg.weight_cumulative,
                )
                total_full_count += float(br['mse_count'].item())
                total_full_flux  += float(br['mse_flux'] .item())
                total_full_cum   += float(br['mse_cum']  .item())
                n_full += 1
        cnt += 1
    mse_per_step_delta = [m / max(cnt, 1) for m in mse_per_step_delta]
    mse_per_step_state = [m / max(cnt, 1) for m in mse_per_step_state]
    rollout_avg = sum(mse_per_step_state) / len(mse_per_step_state)
    return {
        'singlestep_mse':    mse_per_step_state[0],   # comparable to M3
        'mse_count':         total_full_count / max(n_full, 1),
        'mse_flux':          total_full_flux  / max(n_full, 1),
        'mse_cum':           total_full_cum   / max(n_full, 1),
        'rollout_mse_avg':   rollout_avg,
        'mse_per_step':      mse_per_step_state,
        'mse_per_step_delta': mse_per_step_delta,
    }
