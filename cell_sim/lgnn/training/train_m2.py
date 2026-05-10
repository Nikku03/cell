"""M2 Phase 1 training: CfC node dynamics on top of M1+axis2.

Reuses the M1+axis2 fast training infrastructure (preload-to-GPU,
bf16, λ_attn / λ_dropout warmup, k-curriculum rollout, sum-of-gammas
normalization, per-step val MSE breakdown) — the only change is which
model class gets instantiated. CfC dynamics are inside the layer; the
training loop's loss math is unchanged from M1+axis2.

Phase 2 (information bottleneck + perturbation augmentation) will land
in a separate file once Phase 1's PM headline confirms CfC is the
right lever for the multi-timescale dynamics ceiling.
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from cell_sim.lgnn.data.dataset import replicate_to_log1p_array
from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.models.gnn_v2 import CellGNNv2, count_parameters
from cell_sim.lgnn.training.train_m1_axis2_fast import (
    DEFAULT_K_CURRICULUM,
    _gather_windows,
    _index_iterator,
    _torch_dtype,
    _warmup_ramp,
    preload_to_gpu,
)


@dataclass
class M2TrainConfig:
    n_species: int = 8572
    hidden: int = 64
    n_layers: int = 3
    batch_size: int = 64                     # smaller default — CfC adds 3.4M params
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

    # CfC hyperparameter
    cfc_tau_min: float = 0.1                 # τ_min in units of Δt (=1s)

    wall_clock_budget_s: float = 2 * 3600.0


def _device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _k_for_epoch(curriculum: tuple, epoch: int) -> int:
    return int(curriculum[min(epoch, len(curriculum) - 1)])


def train_m2(
    cfg: M2TrainConfig,
    lsdata_module,
    graph: SpeciesGraph,
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
    nbytes = (train_data.nbytes + val_data.nbytes) / 1024**3
    print(f'  shape: train{tuple(train_data.shape)}  val{tuple(val_data.shape)}'
          f'   total {nbytes:.2f} GB on GPU   load wall {time.time()-t0:.1f}s')

    # --- Model (CfC) ---
    model = CellGNNv2(
        graph=graph,
        hidden=cfg.hidden,
        n_layers=cfg.n_layers,
        use_checkpoint=cfg.use_checkpoint,
        edge_chunk_size=cfg.edge_chunk_size,
        cfc_tau_min=cfg.cfc_tau_min,
    ).to(device)
    if cfg.use_bf16 and device.type == 'cuda':
        model = model.to(torch.bfloat16)
    model_dtype = next(model.parameters()).dtype

    print(f'M2 (CfC): {count_parameters(model):,} parameters'
          f'  (hidden={cfg.hidden}, n_layers={cfg.n_layers},'
          f'  τ_min={cfg.cfc_tau_min}, ckpt={cfg.use_checkpoint},'
          f'  dtype={model_dtype})')
    print(f'k-curriculum: {cfg.k_curriculum}   max_k: {cfg.max_k}')

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    history = {
        'train_total': [], 'train_full_ss': [], 'train_drop_ss': [],
        'train_rollout': [], 'train_attn_entropy': [],
        'val_singlestep_mse': [], 'val_rollout_mse_avg': [],
        'val_mse_per_step': [],
        'k_per_epoch': [], 'samples_per_sec': [],
        'lambda_attn_end_of_epoch': [],
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
                for k in ('total', 'full_ss', 'drop_ss',
                           'rollout', 'attn_entropy')}
        n_batches = 0
        t_epoch = time.time()
        n_seen = 0

        for rep_idx, t_idx in _index_iterator(
            R, n_valid, cfg.batch_size, gen, device,
        ):
            x_w = _gather_windows(
                train_data, rep_idx, t_idx, k_cur + 1,
            ).to(model_dtype)
            x = x_w[:, 0, :]

            # 1. Multi-step rollout
            x_pred = x
            attn_entropy = None
            L_rollout = torch.zeros((), device=device, dtype=model_dtype)
            L_full_ss = None
            gamma_sum = 0.0
            for s in range(k_cur):
                if s == 0:
                    dx, ent = model(
                        x_pred, return_attention_entropy=True,
                    )
                    attn_entropy = ent
                else:
                    dx = model(x_pred)
                x_pred = x_pred + dx
                target = x_w[:, s + 1, :]
                step_mse = mse(x_pred, target)
                if s == 0:
                    L_full_ss = step_mse
                w = cfg.rollout_gamma ** s
                L_rollout = L_rollout + w * step_mse
                gamma_sum += w
            L_rollout = L_rollout / max(gamma_sum, 1e-12)

            # 2. Warmup schedules + conditional dropout pass
            cur_lambda_attn = _warmup_ramp(
                global_step, cfg.lambda_attn,
                cfg.lambda_attn_warmup_steps,
                cfg.lambda_attn_ramp_steps,
            )
            cur_lambda_dropout = _warmup_ramp(
                global_step, cfg.lambda_dropout,
                cfg.lambda_dropout_warmup_steps,
                cfg.lambda_dropout_ramp_steps,
            )

            if cur_lambda_dropout > 0.0:
                dx_drop = model(x, edge_dropout_p=cfg.edge_dropout_p)
                L_drop_ss = mse(x + dx_drop, x_w[:, 1, :])
                drop_term = cur_lambda_dropout * L_drop_ss
            else:
                L_drop_ss = L_full_ss.detach()
                drop_term = torch.zeros((), device=device,
                                          dtype=L_full_ss.dtype)

            # 3. Combined loss
            if eff_lambda_rollout == 0.0:
                supervised = L_full_ss
            else:
                supervised = eff_lambda_rollout * L_rollout
            L = (supervised
                 + drop_term
                 + cur_lambda_attn * attn_entropy)

            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()
            global_step += 1

            run['total']        += L.detach().float()
            run['full_ss']      += L_full_ss.detach().float()
            run['drop_ss']      += L_drop_ss.detach().float()
            run['rollout']      += L_rollout.detach().float()
            run['attn_entropy'] += attn_entropy.detach().float()
            n_batches += 1
            n_seen += int(rep_idx.shape[0])

            if (n_batches) % cfg.log_every == 0:
                wall = time.time() - t_epoch
                vals = {k: float(v.item()) / n_batches for k, v in run.items()}
                print(f'  ep{epoch} k={k_cur} step{n_batches:>5d}'
                      f'  total={vals["total"]:.4f}'
                      f'  full_ss={vals["full_ss"]:.4f}'
                      f'  drop_ss={vals["drop_ss"]:.4f}'
                      f'  rollout={vals["rollout"]:.4f}'
                      f'  attn={vals["attn_entropy"]:.3f}'
                      f'  λa={cur_lambda_attn:.1e}'
                      f'  λd={cur_lambda_dropout:.1e}'
                      f'  {n_seen/wall:.0f} s/s')

            if (time.time() - train_t0) > cfg.wall_clock_budget_s:
                print(f'  WARNING: wall-clock budget '
                      f'{cfg.wall_clock_budget_s/3600:.2f}h exceeded')

        epoch_wall = time.time() - t_epoch
        history['samples_per_sec'].append(n_seen / max(epoch_wall, 1e-6))
        history['lambda_attn_end_of_epoch'].append(
            _warmup_ramp(global_step, cfg.lambda_attn,
                          cfg.lambda_attn_warmup_steps,
                          cfg.lambda_attn_ramp_steps)
        )
        for k, v in run.items():
            history.setdefault(f'train_{k}', []).append(
                float(v.item()) / max(n_batches, 1)
            )

        val = evaluate_m2(model, val_data, cfg, device, k_eval=k_cur)
        history['val_singlestep_mse'].append(val['mse_singlestep'])
        history['val_rollout_mse_avg'].append(val['mse_rollout_avg'])
        history['val_mse_per_step'].append(val['mse_per_step'])
        per_step_str = ' '.join(f'{m:.4f}' for m in val['mse_per_step'])
        print(f'  ep{epoch} done in {epoch_wall:.1f}s  '
              f'val_ss={val["mse_singlestep"]:.4f}  '
              f'val_per_step(k={k_cur})=[{per_step_str}]  '
              f'avg={val["mse_rollout_avg"]:.4f}  '
              f'{n_seen/epoch_wall:.0f} s/s')

        if val['mse_singlestep'] < best_val:
            best_val = val['mse_singlestep']
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'state_dict': model.state_dict(),
                    'cfg': cfg.__dict__,
                    'epoch': epoch,
                    'val_mse_singlestep': val['mse_singlestep'],
                    'val_mse_rollout_avg': val['mse_rollout_avg'],
                    'val_mse_per_step': val['mse_per_step'],
                    'k_at_save': k_cur,
                }, checkpoint_path)

    return {
        'history': history,
        'best_val_singlestep_mse': best_val,
        'final_model': model,
        'wall_clock_total_s': time.time() - train_t0,
    }


@torch.no_grad()
def evaluate_m2(
    model: nn.Module,
    val_data: torch.Tensor,
    cfg: M2TrainConfig,
    device: torch.device,
    k_eval: int = 4,
) -> dict:
    """Per-step rollout MSE on the preloaded val tensor. Same structure
    as evaluate_fast (M1+axis2)."""
    model.eval()
    model_dtype = next(model.parameters()).dtype
    R, T, S = val_data.shape
    n_valid = T - k_eval - 1
    sq_per_step = [torch.zeros((), device=device, dtype=torch.float32)
                    for _ in range(k_eval)]
    n_per_step = [0] * k_eval

    for rep in range(R):
        for s_start in range(0, n_valid, cfg.batch_size):
            s_end = min(s_start + cfg.batch_size, n_valid)
            t_idx = torch.arange(s_start, s_end, device=device)
            rep_idx = torch.full_like(t_idx, rep)
            x_w = _gather_windows(
                val_data, rep_idx, t_idx, k_eval + 1,
            ).to(model_dtype)

            x_pred = x_w[:, 0, :]
            for s in range(k_eval):
                dx = model(x_pred)
                x_pred = x_pred + dx
                err2 = (x_pred - x_w[:, s + 1, :]).pow(2).float()
                sq_per_step[s] += err2.sum()
                n_per_step[s]  += err2.numel()

    mse_per_step = [
        float(sq_per_step[s].item()) / max(n_per_step[s], 1)
        for s in range(k_eval)
    ]
    return {
        'mse_singlestep':  mse_per_step[0] if mse_per_step else 0.0,
        'mse_per_step':    mse_per_step,
        'mse_rollout_avg': sum(mse_per_step) / max(len(mse_per_step), 1),
    }
