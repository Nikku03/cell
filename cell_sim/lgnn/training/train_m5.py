"""M5 - comprehensive static node features in a single retrain.

Combines all per-node static feature sources into one training run:
  - Gene-class features from syn3A.gb        (19 cols)
  - Spatial features from MinCell_*.lm       (12 cols, Step B output)
  - Per-species role one-hot indicators       (8 cols)
  - Promoter strength per Equation 20         (1 col)
  - Log initial count per species             (1 col)
Total: 41 static features per node.

Architecture: identical to M4 (CellGNNv4) - the model is agnostic to
F_static and just concatenates with the (count, variance) dynamic input.
Only the static_features tensor changes between M4 and M5.

Per the priority analysis, this single retrain captures categories 1+3+5
of the user's prioritized list (gene annotations, promoter strengths,
spatial) plus log_initial_count for free. Categories 2 (kinetic edge
features) and 4 (subsystem labels) are EDGE features and require a
separate retrain.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.data.static_node_features import (
    build_static_node_features, feature_summary, ALL_STATIC_COLS,
)
from cell_sim.lgnn.models.gnn_v4 import CellGNNv4, count_parameters
from cell_sim.lgnn.training.train_m1_axis2_fast import (
    _gather_windows, _index_iterator, _torch_dtype, _warmup_ramp,
    preload_to_gpu,
)
from cell_sim.lgnn.training.train_m3 import (
    M3TrainConfig, _device, _k_for_epoch, categorise_row_indices,
    compute_variance_channel, _multi_task_mse, _evaluate as _m3_evaluate,
)


@dataclass
class M5TrainConfig(M3TrainConfig):
    """M3 config + paths to extra static-feature sources."""
    gene_class_csv:    str = 'memory_bank/data/syn3a_gene_class_features.csv'
    spatial_parquet:   Optional[str] = None
    use_initial_state: bool = True   # if True, derive promoter + log_init


def train_m5(
    cfg: M5TrainConfig,
    lsdata_module,
    graph: SpeciesGraph,
    row_names: Sequence[str],
    checkpoint_path: Optional[Path] = None,
) -> dict:
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    pre_dtype = _torch_dtype(cfg.preload_dtype)

    # --- Build static node features ---
    print(f'Building static node features...')
    print(f'  gene_class_csv:  {cfg.gene_class_csv}')
    print(f'  spatial_parquet: {cfg.spatial_parquet}')
    print(f'  use_initial_state: {cfg.use_initial_state}')

    initial_state = None
    if cfg.use_initial_state:
        df0 = lsdata_module.load_replicate(1, time_start=0.0, time_end=0.0)
        s   = df0.iloc[:, 0].reindex(row_names)
        x0_real = s.values.astype(np.float32)
        # convert to signed-log1p (the format the builder expects)
        initial_state = np.sign(x0_real) * np.log1p(np.abs(x0_real))

    static_features = build_static_node_features(
        row_names           = row_names,
        gene_class_csv      = cfg.gene_class_csv,
        spatial_parquet     = cfg.spatial_parquet,
        initial_state_signed_log1p = initial_state,
        verbose             = True,
    )
    print(f'\nstatic features shape: {tuple(static_features.shape)}')
    summary = feature_summary(static_features)
    print(summary.to_string(index=False))

    # --- Preload corpus ---
    t0 = time.time()
    print(f'\npreloading {len(cfg.train_replicates)} train + '
          f'{len(cfg.val_replicates)} val replicates to {device}...')
    train_data = preload_to_gpu(cfg.train_replicates, lsdata_module, device,
                                  dtype=pre_dtype,
                                  species_filter=cfg.species_filter)
    val_data = preload_to_gpu(cfg.val_replicates, lsdata_module, device,
                                dtype=pre_dtype,
                                species_filter=cfg.species_filter)
    R, T, S = train_data.shape
    n_valid = T - cfg.max_k - 1
    print(f'  shape: train{tuple(train_data.shape)}  val{tuple(val_data.shape)}'
          f'   load wall {time.time()-t0:.1f}s')

    count_mask, flux_mask, cum_mask = categorise_row_indices(row_names)
    count_mask = count_mask.to(device); flux_mask = flux_mask.to(device)
    cum_mask   = cum_mask  .to(device)
    print(f'rows: count={int(count_mask.sum())}  '
          f'flux={int(flux_mask.sum())}  cumulative={int(cum_mask.sum())}')

    print('precomputing per-species cross-replicate std...')
    sigma = compute_variance_channel(train_data)

    # --- Model ---
    model = CellGNNv4(
        graph=graph, static_features=static_features,
        hidden=cfg.hidden, n_layers=cfg.n_layers,
        n_dynamic_channels=cfg.n_input_channels,
        use_checkpoint=cfg.use_checkpoint,
        edge_chunk_size=cfg.edge_chunk_size,
        cfc_tau_min=cfg.cfc_tau_min,
    ).to(device)
    if cfg.use_bf16 and device.type == 'cuda':
        model = model.to(torch.bfloat16)
    model_dtype = next(model.parameters()).dtype
    n_params = count_parameters(model)
    n_static = model.n_static_features
    print(f'M5: {n_params:,} parameters '
          f'(hidden={cfg.hidden}, n_layers={cfg.n_layers}, '
          f'dyn_channels={cfg.n_input_channels}, '
          f'static_features={n_static}, dtype={model_dtype})')

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
            v0 = sigma[t_idx].to(model_dtype)

            x_pred = x
            attn_entropy = None
            L_rollout = torch.zeros((), device=device, dtype=model_dtype)
            L_full_ss = None
            mse_breakdown_step0 = None
            gamma_sum = 0.0
            for s in range(k_cur):
                v_cur = sigma[t_idx + s].to(model_dtype)
                if s == 0:
                    dx, ent = model(x_pred, x_var=v_cur,
                                      return_attention_entropy=True)
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

            cur_lambda_attn = _warmup_ramp(
                global_step, cfg.lambda_attn,
                cfg.lambda_attn_warmup_steps, cfg.lambda_attn_ramp_steps,
            )
            cur_lambda_dropout = _warmup_ramp(
                global_step, cfg.lambda_dropout,
                cfg.lambda_dropout_warmup_steps, cfg.lambda_dropout_ramp_steps,
            )

            if cur_lambda_dropout > 0.0:
                dx_drop = model(x, x_var=v0,
                                  edge_dropout_p=cfg.edge_dropout_p)
                L_drop_ss, _ = _multi_task_mse(
                    x + dx_drop, x_w[:, 1, :],
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

        model.eval()
        val_metrics = _m3_evaluate(model, val_data, sigma,
                                     count_mask, flux_mask, cum_mask, cfg,
                                     model_dtype, device)
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
                    'static_feature_cols': ALL_STATIC_COLS,
                }
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, checkpoint_path)
                print(f'  -> saved best (val_ss={val_ss:.4f}) to {checkpoint_path}')

    return history
