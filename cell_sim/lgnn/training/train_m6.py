"""M6 training - the changes I actually agree with.

Honest revision after pushing back on the user's 5-step roadmap. M6 keeps
ONE step from the roadmap (Step 3, edge-kind split) and adds the actual
root-cause fix for the long-horizon drift problem we observed (exposure
bias, not stiffness).

Changes from M3:

  Step 3 (KEPT): RATE_LAW / MASS_BALANCE edge-kind split.
    - data side handled in species_graph.py via split_flux_coupling=True
    - N_EDGE_KINDS auto-bumps to 9, kind_embedding sized accordingly
    - stoichiometric coefficient as edge attribute on these edges

  NEW: Extended k-curriculum with equal-weight rollout.
    - M3 used (1, 2, 4) with gamma=0.95 -> step 0 dominates loss
    - M6 uses (1, 4, 16, 32) with gamma=1.0 -> all rollout steps weighted equally
    - Forces the model to learn dynamics that don't drift over many steps

  NEW: Scheduled sampling for exposure bias.
    - In M3 the model sees its own predictions for ≤4 future steps during
      training, but autoregressive eval rolls for 100s of steps. The
      model has no signal for "what to do when my prediction is off."
    - M6 mixes teacher-forced (use ground truth at step s) and free-running
      (use model's accumulated x_pred at step s) inputs, with the free-
      running probability ramping from 0 to 0.5 over training.
    - This is the classic Bengio 2015 fix for exposure bias.

  REVERTED from earlier M6 attempt:
    - delta target (Step 1) - mathematically equivalent to residual form
    - softplus aggregation (Step 2) - wrong layer for mass conservation,
      destabilizes high-degree nodes

  STILL TODO (separate Ms when justified):
    - PINN with hardwired stoichiometric matrix (the actual mass-conservation
      fix at the OUTPUT layer; user's "Flaw 8" - highest impact deferred)
    - Adaptive ODE solver (Step 5) - revisit after PINN is in place
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.models.gnn_v3 import CellGNNv3, count_parameters
from cell_sim.lgnn.training.train_m1_axis2_fast import (
    _gather_windows, _index_iterator, _torch_dtype, _warmup_ramp,
    preload_to_gpu,
)
from cell_sim.lgnn.training.train_m3 import (
    M3TrainConfig, _device, _k_for_epoch, categorise_row_indices,
    compute_variance_channel, _multi_task_mse, _evaluate as _m3_evaluate,
)


@dataclass
class M6TrainConfig(M3TrainConfig):
    """M3 config + extended k-curriculum + scheduled sampling + speed knobs."""
    # Override M3 defaults
    k_curriculum: tuple = (1, 4, 16, 32)       # was (1, 2, 4)
    rollout_gamma: float = 1.0                  # was 0.95
    max_k: int = 32                             # was 4
    n_epochs: int = 5                           # one per curriculum stage + 1 polish

    # Scheduled sampling (NEW)
    scheduled_sampling: bool = True
    p_ss_max: float = 0.5                       # max prob of free-running at step s>0

    # Speed knobs (validated by VRAM math: M6 fits in ~10 GB on 48 GB card)
    use_checkpoint: bool = False                # was True - we have VRAM, save 33%
    use_compile: bool = True                    # was False - 2-3x speedup via CUDA Graphs
    compile_mode: str = 'reduce-overhead'       # fuses k-step rollout kernels
    p_ss_warmup_steps: int = 2000               # ramp from 0 -> p_ss_max over this many steps

    # Step 3 of roadmap is on the data side (split_flux_coupling=True when
    # building the graph). Nothing to configure in the trainer.


def _maybe_load_or_preload(replicate_indices, lsdata_module, device, dtype,
                             species_filter, cache_path):
    """Load preloaded tensor from disk if available, else preload + save.

    On a cold Colab runtime the parquet preload can take ~30 min for 49
    replicates. Cached load is ~1-2 min (just a torch.load of a 6 GB
    bfloat16 tensor). Saves enormous wall-clock across retrains.
    """
    import time
    if cache_path is not None and Path(cache_path).exists():
        t0 = time.time()
        print(f'  loading cached preload from {cache_path}...')
        data = torch.load(cache_path, map_location=device, weights_only=False)
        # Verify shape matches expectation
        if data.dim() == 3 and data.shape[0] == len(replicate_indices):
            print(f'  ✓ loaded {tuple(data.shape)} in {time.time()-t0:.1f}s')
            return data.to(dtype) if data.dtype != dtype else data
        else:
            print(f'  cache shape mismatch ({tuple(data.shape)} vs expected '
                  f'first dim {len(replicate_indices)}); rebuilding')
    t0 = time.time()
    data = preload_to_gpu(replicate_indices, lsdata_module, device,
                            dtype=dtype, species_filter=species_filter)
    print(f'  preloaded fresh in {time.time()-t0:.1f}s')
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        torch.save(data.cpu(), cache_path)
        print(f'  cached to {cache_path} in {time.time()-t1:.1f}s')
    return data


def train_m6(
    cfg: M6TrainConfig,
    lsdata_module,
    graph: SpeciesGraph,
    row_names: Sequence[str],
    checkpoint_path: Optional[Path] = None,
    train_cache_path: Optional[Path] = None,
    val_cache_path: Optional[Path] = None,
) -> dict:
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    pre_dtype = _torch_dtype(cfg.preload_dtype)

    # --- Preload (with on-disk cache) ---
    print(f'preloading {len(cfg.train_replicates)} train + '
          f'{len(cfg.val_replicates)} val replicates to {device}...')
    train_data = _maybe_load_or_preload(
        cfg.train_replicates, lsdata_module, device, pre_dtype,
        cfg.species_filter, train_cache_path,
    )
    val_data = _maybe_load_or_preload(
        cfg.val_replicates, lsdata_module, device, pre_dtype,
        cfg.species_filter, val_cache_path,
    )
    R, T, S = train_data.shape
    n_valid = T - cfg.max_k - 1
    print(f'  train{tuple(train_data.shape)}  val{tuple(val_data.shape)}')

    count_mask, flux_mask, cum_mask = categorise_row_indices(row_names)
    count_mask = count_mask.to(device); flux_mask = flux_mask.to(device)
    cum_mask   = cum_mask  .to(device)
    print(f'rows: count={int(count_mask.sum())}  '
          f'flux={int(flux_mask.sum())}  cumulative={int(cum_mask.sum())}')

    # Sigma channel (count + variance, same as M3 — keeping the variance
    # input for M6 since removing it is a separate experiment.)
    print('precomputing per-species cross-replicate std...')
    sigma = compute_variance_channel(train_data)

    # --- Model: CellGNNv3 (2-channel input: count + variance) with N_EDGE_KINDS=9 ---
    from cell_sim.lgnn.models.gnn_v1_axis2 import N_EDGE_KINDS
    print(f'N_EDGE_KINDS (auto): {N_EDGE_KINDS}')
    assert N_EDGE_KINDS == 9, f'expected 9 edge kinds, got {N_EDGE_KINDS}'

    import collections
    ek_counts = collections.Counter(graph.edge_kind.tolist())
    print(f'graph edge-kind counts: {dict(sorted(ek_counts.items()))}')

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
    print(f'M6: {n_params:,} parameters '
          f'(hidden={cfg.hidden}, n_layers={cfg.n_layers}, '
          f'channels={cfg.n_input_channels}, dtype={model_dtype})')
    print(f'k_curriculum={cfg.k_curriculum}, rollout_gamma={cfg.rollout_gamma}, '
          f'scheduled_sampling={cfg.scheduled_sampling}, p_ss_max={cfg.p_ss_max}')

    # torch.compile for the k-step rollout (kernel fusion via CUDA Graphs)
    if cfg.use_compile and device.type == 'cuda':
        try:
            model = torch.compile(model, mode=cfg.compile_mode)
            print(f'  torch.compile enabled (mode={cfg.compile_mode}); '
                  f'first forward includes ~30-60s compilation overhead')
        except Exception as e:
            print(f'  WARNING: torch.compile failed ({e}); continuing uncompiled')

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)

    history = {
        'train_total': [], 'train_mse_count': [], 'train_mse_flux': [],
        'train_mse_cum': [], 'train_rollout': [], 'train_attn_entropy': [],
        'train_p_ss': [],
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

            # Current scheduled-sampling probability
            if cfg.scheduled_sampling and k_cur > 1:
                p_ss = min(global_step / max(cfg.p_ss_warmup_steps, 1),
                            1.0) * cfg.p_ss_max
            else:
                p_ss = 0.0

            x_pred = x
            attn_entropy = None
            L_rollout = torch.zeros((), device=device, dtype=model_dtype)
            L_full_ss = None
            mse_breakdown_step0 = None
            gamma_sum = 0.0
            for s in range(k_cur):
                v_cur = sigma[t_idx + s].to(model_dtype)

                # ✦ SCHEDULED SAMPLING ✦
                # At step s>0, with probability (1 - p_ss) replace the model's
                # accumulated x_pred with the GROUND-TRUTH state at time t+s.
                # This is teacher-forcing at step s. As p_ss ramps up, the
                # model spends more time on its own predictions (free-running).
                if s > 0 and p_ss < 1.0:
                    # Sample a Bernoulli per batch element. Vector form so
                    # each batch element can independently be teacher-forced.
                    use_tf = (torch.rand(x_pred.shape[0], 1, device=device)
                                > p_ss).to(x_pred.dtype)
                    x_input = use_tf * x_w[:, s, :] + (1 - use_tf) * x_pred
                else:
                    x_input = x_pred

                if s == 0:
                    dx, ent = model(x_input, x_var=v_cur,
                                      return_attention_entropy=True)
                    attn_entropy = ent
                else:
                    dx = model(x_input, x_var=v_cur)

                # RESIDUAL update (kept from M3)
                x_pred = x_input + dx
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
                      f'  p_ss={p_ss:.2f}'
                      f'  λa={cur_lambda_attn:.1e}'
                      f'  {n_seen/wall:.0f} s/s')

            if (time.time() - train_t0) > cfg.wall_clock_budget_s:
                print(f'  WARNING: wall-clock budget exceeded')
                break

        epoch_wall = time.time() - t_epoch
        history['samples_per_sec'].append(n_seen / max(epoch_wall, 1e-6))
        history['train_p_ss'].append(p_ss)
        for key in ('total', 'mse_count', 'mse_flux', 'mse_cum',
                     'rollout', 'attn_entropy'):
            history[f'train_{key}'].append(
                float(run[key].item()) / max(n_batches, 1)
            )

        # Validation: same as M3 evaluation (uses residual form internally)
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
              f'p_ss={p_ss:.2f}  '
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
                    'p_ss_at_save': p_ss,
                    'edge_kind_split': 'RATE_LAW_MASS_BALANCE',  # marker
                }
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, checkpoint_path)
                print(f'  -> saved best (val_ss={val_ss:.4f}) to {checkpoint_path}')

    return history
