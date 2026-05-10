"""M1 + Axis 2 training: multi-step rollout + counterfactual edge dropout.

Combined loss per training step:

    L = L_full_singlestep
      + λ_dropout · L_dropout_singlestep
      + λ_rollout · L_rollout(k_current)
      + λ_attn    · L_attention_entropy

where L_full_singlestep is captured from step s=0 of the rollout
(no extra forward pass) and L_dropout_singlestep is a separate forward
with `edge_dropout_p > 0`. So the per-batch cost is k_current + 1
forward passes, all in the autograd graph.

K-curriculum: epoch 0 → k=1, epoch 1 → k=2, epoch 2 → k=4, epoch ≥3 → k=10.
This warms up the model at single-step before rollout-stability is
demanded; without it, k=10 from epoch 0 OOM-bombs and learns nothing.

Memorization metrics (post-training, in `evaluate_memorization`):
  • within-replicate time-split MSE divergence (train t<3600 vs val t≥3600)
  • perturbation MSE ratio (input perturbed by ±5% on 20% of species)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cell_sim.lgnn.data.dataset import (
    CountsRolloutDataset, CountsTransitionDataset,
    replicate_to_log1p_array,
)
from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.models.gnn_v1_axis2 import (
    CellGNNv1Axis2, count_parameters,
)


# K-curriculum: schedule k by epoch. Last value applies to all later epochs.
DEFAULT_K_CURRICULUM = (1, 2, 4, 10)


@dataclass
class M1Axis2TrainConfig:
    n_species: int = 8572
    hidden: int = 64
    n_layers: int = 3
    batch_size: int = 128                # smaller for k=10 rollout memory
    lr: float = 3e-4
    weight_decay: float = 1e-5
    train_replicates: tuple = tuple(range(1, 50))
    val_replicates: tuple = (50,)
    n_epochs: int = 4
    device: str = 'auto'
    seed: int = 42
    log_every: int = 100
    species_filter: Optional[List[str]] = None
    # DataLoader workers — was hard-coded to 0, single-threaded data
    # prep on the main thread was the original 189 s/s bottleneck.
    # 4 workers + pin_memory hides per-batch host work behind GPU
    # compute. Use 0 for synchronous debugging.
    num_workers: int = 4

    # Axis 2: counterfactual edge dropout
    edge_dropout_p: float = 0.07
    lambda_dropout: float = 0.5

    # Axis 2: attention sparsity
    lambda_attn: float = 1e-3

    # Multi-step rollout
    k_curriculum: tuple = DEFAULT_K_CURRICULUM
    rollout_gamma: float = 0.95
    lambda_rollout: float = 1.0
    max_k: int = 10

    use_checkpoint: bool = False       # consistent with fast trainer; flip on
                                       #   only if memory pressure forces it
    use_bf16: bool = False             # mixed-precision autocast (~2× on Blackwell)

    # Wall-clock budget alarm (soft warning, doesn't kill training)
    wall_clock_budget_s: float = 6 * 3600.0


def _device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _k_for_epoch(curriculum: tuple, epoch: int) -> int:
    return int(curriculum[min(epoch, len(curriculum) - 1)])


def train_m1_axis2(cfg: M1Axis2TrainConfig,
                    lsdata_module,
                    graph: SpeciesGraph,
                    checkpoint_path: Optional[Path] = None) -> dict:
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    model = CellGNNv1Axis2(
        graph=graph,
        hidden=cfg.hidden,
        n_layers=cfg.n_layers,
        use_checkpoint=cfg.use_checkpoint,
    ).to(device)
    print(f'M1+axis2: {count_parameters(model):,} parameters'
          f'  (hidden={cfg.hidden}, n_layers={cfg.n_layers})')
    print(f'k-curriculum: {cfg.k_curriculum}')
    print(f'wall-clock budget: {cfg.wall_clock_budget_s/3600:.1f} h')

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    history = {
        'train_total': [], 'train_full_ss': [], 'train_drop_ss': [],
        'train_rollout': [], 'train_attn_entropy': [],
        'val_singlestep_mse': [], 'val_rollout_mse_avg': [],
        'val_mse_per_step': [],
        'k_per_epoch': [],
    }
    best_val = float('inf')
    train_t0 = time.time()

    for epoch in range(cfg.n_epochs):
        k_cur = _k_for_epoch(cfg.k_curriculum, epoch)
        history['k_per_epoch'].append(k_cur)
        ds = CountsRolloutDataset(
            cfg.train_replicates, lsdata_module,
            species_filter=cfg.species_filter,
            max_k=cfg.max_k,
            shuffle_within_replicate=True,
            seed=cfg.seed + epoch,
        )
        loader = DataLoader(
            ds, batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=(cfg.num_workers > 0 and torch.cuda.is_available()),
            persistent_workers=(cfg.num_workers > 0),
        )

        model.train()
        running = {'total': 0.0, 'full_ss': 0.0, 'drop_ss': 0.0,
                   'rollout': 0.0, 'attn': 0.0}
        n_batches = 0
        t0 = time.time()
        # bf16 autocast context. No GradScaler needed (bf16 has fp32-level
        # exponent range, no underflow risk like fp16). On Blackwell this
        # gives ~2× compute throughput AND halves activation memory, which
        # is what makes use_checkpoint=False viable at k=10 + B=256.
        if cfg.use_bf16 and device.type == 'cuda':
            autocast_ctx = lambda: torch.amp.autocast('cuda', dtype=torch.bfloat16)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext

        for step, (x, x_window) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            x_window = x_window.to(device, non_blocking=True)   # (B, max_k+1, S)

            # Truncate window to k_cur+1 timepoints
            x_w = x_window[:, : k_cur + 1, :]

            with autocast_ctx():
                # 1. Multi-step rollout on the full graph. Capture step-0
                # prediction for L_full_singlestep + attention entropy.
                # Normalize by sum-of-gammas so weights are a convex
                # combination (was /k_cur which broke γ-discounting).
                x_pred = x_w[:, 0, :]
                attn_entropy = None
                L_rollout = 0.0
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

                # 2. Single-step dropout pass on the full input
                dx_drop = model(x, edge_dropout_p=cfg.edge_dropout_p)
                L_drop_ss = mse(x + dx_drop, x_w[:, 1, :])

                # 3. Combined loss. L_full_ss is the s=0 component of
                # L_rollout — including both double-counts the single-
                # step gradient. Use L_rollout as the supervised term;
                # at k=1 this collapses to L_full_ss (so it's still
                # supervised, just no longer double-weighted).
                L = (cfg.lambda_rollout * L_rollout
                     + cfg.lambda_dropout * L_drop_ss
                     + cfg.lambda_attn    * attn_entropy)

            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()

            running['total']   += L.item()
            running['full_ss'] += L_full_ss.item()
            running['drop_ss'] += L_drop_ss.item()
            running['rollout'] += L_rollout.item()
            running['attn']    += attn_entropy.item()
            n_batches += 1

            if (step + 1) % cfg.log_every == 0:
                wall = time.time() - t0
                print(f'  ep{epoch} k={k_cur} step{step+1:>5d}'
                      f'  total={running["total"]/n_batches:.4f}'
                      f'  full_ss={running["full_ss"]/n_batches:.4f}'
                      f'  drop_ss={running["drop_ss"]/n_batches:.4f}'
                      f'  rollout={running["rollout"]/n_batches:.4f}'
                      f'  attn={running["attn"]/n_batches:.3f}'
                      f'  {(step+1)*cfg.batch_size/wall:.0f} s/s')

            # Soft wall-clock budget alarm — print warning, keep training
            if (time.time() - train_t0) > cfg.wall_clock_budget_s:
                print(f'  WARNING: wall-clock budget '
                      f'{cfg.wall_clock_budget_s/3600:.1f}h exceeded; '
                      f'consider optimizing _AttentionGNNLayer.forward')

        for key in running:
            running[key] /= max(n_batches, 1)
        history['train_total'].append(running['total'])
        history['train_full_ss'].append(running['full_ss'])
        history['train_drop_ss'].append(running['drop_ss'])
        history['train_rollout'].append(running['rollout'])
        history['train_attn_entropy'].append(running['attn'])

        val = evaluate(model, cfg, lsdata_module, graph, device,
                       k_eval=k_cur)
        history['val_singlestep_mse'].append(val['mse_singlestep'])
        history['val_rollout_mse_avg'].append(val['mse_rollout_avg'])
        history['val_mse_per_step'].append(val['mse_per_step'])
        per_step_str = ' '.join(f'{m:.4f}' for m in val['mse_per_step'])
        print(f'  ep{epoch}  val_mse_singlestep={val["mse_singlestep"]:.4f}'
              f'  val_mse_per_step(k={k_cur})=[{per_step_str}]'
              f'  avg={val["mse_rollout_avg"]:.4f}')

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
def evaluate(model: nn.Module,
             cfg: M1Axis2TrainConfig,
             lsdata_module,
             graph: SpeciesGraph,
             device: torch.device,
             k_eval: int = 10) -> dict:
    """val MSE on single-step + k-step rollout."""
    model.eval()
    ds = CountsRolloutDataset(
        cfg.val_replicates, lsdata_module,
        species_filter=cfg.species_filter,
        max_k=k_eval,
        shuffle_within_replicate=False,
        seed=cfg.seed,
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.num_workers > 0 and torch.cuda.is_available()),
    )

    # Per-step accumulators so we can report MSE at each rollout
    # position. The previously-named `mse_rollout` (average across all
    # positions) is preserved as `mse_rollout_avg` but isn't directly
    # cross-run-comparable for different k_eval values.
    sq_per_step = [0.0] * k_eval
    n_per_step = [0] * k_eval
    for x, x_window in loader:
        x = x.to(device, non_blocking=True)
        x_window = x_window.to(device, non_blocking=True)

        x_pred = x_window[:, 0, :]
        for s in range(k_eval):
            dx = model(x_pred)
            x_pred = x_pred + dx
            target = x_window[:, s + 1, :]
            err2 = (x_pred - target).pow(2)
            sq_per_step[s] += err2.sum().item()
            n_per_step[s]  += err2.numel()
    mse_per_step = [
        sq_per_step[s] / max(n_per_step[s], 1) for s in range(k_eval)
    ]
    return {
        'mse_singlestep':  mse_per_step[0] if mse_per_step else 0.0,
        'mse_per_step':    mse_per_step,
        'mse_rollout_avg': sum(mse_per_step) / max(len(mse_per_step), 1),
    }


# ---------------------------------------------------------------------------
# Memorization detectors (post-training diagnostics)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_memorization(model: nn.Module,
                           lsdata_module,
                           replicates: tuple,
                           device: torch.device,
                           split_time: int = 3600,
                           perturbation_eps: tuple = (0.01, 0.05, 0.1),
                           perturbation_frac: float = 0.20) -> dict:
    """Two metrics, both required to be co-passing:

    1. Within-replicate time-split MSE divergence. Predict next-step
       on t<split_time vs t>=split_time of the SAME replicate.
       Healthy ratio < 1.5; memorizing ratio > 3; constant-output ≈ 1.

    2. Perturbation MSE ratio. Apply ±ε multiplicative noise to a
       random `perturbation_frac` of input species, compute MSE of
       prediction vs unperturbed prediction. Constant-output gives
       ratio ≈ 0/0 (degenerate); healthy gives bounded ratio.

    Both metrics must be reported together.
    """
    model.eval()
    rng = np.random.default_rng(0)

    sq_train_t, n_train_t = 0.0, 0
    sq_val_t, n_val_t = 0.0, 0
    for rep_i in replicates:
        df = lsdata_module.load_replicate(rep_i)
        arr = replicate_to_log1p_array(df)               # (T, S)
        T, S = arr.shape
        n_train = min(split_time, T - 1)
        n_val   = T - 1 - n_train
        if n_val <= 0:
            continue

        # train half
        x = torch.from_numpy(arr[:n_train]).to(device)
        target = torch.from_numpy(arr[1:n_train + 1]).to(device)
        for s in range(0, x.shape[0], 128):
            xb = x[s:s + 128]
            tb = target[s:s + 128]
            pred = xb + model(xb)
            sq_train_t += (pred - tb).pow(2).sum().item()
            n_train_t  += pred.numel()

        # val half (within same replicate)
        x = torch.from_numpy(arr[n_train:T - 1]).to(device)
        target = torch.from_numpy(arr[n_train + 1:T]).to(device)
        for s in range(0, x.shape[0], 128):
            xb = x[s:s + 128]
            tb = target[s:s + 128]
            pred = xb + model(xb)
            sq_val_t += (pred - tb).pow(2).sum().item()
            n_val_t  += pred.numel()

    mse_train = sq_train_t / max(n_train_t, 1)
    mse_val   = sq_val_t   / max(n_val_t, 1)
    ratio_time = mse_val / max(mse_train, 1e-12)

    # Perturbation MSE ratio (use first replicate's last 200 timepoints)
    rep_i = replicates[0]
    df = lsdata_module.load_replicate(rep_i)
    arr = replicate_to_log1p_array(df)
    x_eval = torch.from_numpy(arr[-200:]).to(device)         # (200, S)
    pred_unpert = x_eval + model(x_eval)

    perturbation_results = {}
    for eps in perturbation_eps:
        # multiplicative noise on a random subset of species
        S = x_eval.shape[1]
        n_perturb = max(1, int(perturbation_frac * S))
        idx = rng.choice(S, size=n_perturb, replace=False)
        idx_t = torch.tensor(idx, device=device, dtype=torch.long)
        noise = (torch.rand(x_eval.shape[0], n_perturb, device=device)
                  * 2 * eps - eps)
        x_pert = x_eval.clone()
        x_pert[:, idx_t] = x_pert[:, idx_t] * (1 + noise)
        pred_pert = x_pert + model(x_pert)
        # Lipschitz-style ratio: ‖Δpred‖² / ‖Δinput‖². A model that
        # actually responds to input changes has ratio ~1 (linear-ish);
        # a constant-output model has ratio ~0; a chaotic model has
        # ratio >> 1. Was previously divided by pred_unpert.var(),
        # which a constant-output model passes (0/eps ≈ 0) — defeats
        # the memorization detector.
        delta_input  = (x_pert - x_eval).pow(2).mean().item()
        delta_output = (pred_pert - pred_unpert).pow(2).mean().item()
        perturbation_results[f'eps={eps:.2f}'] = (
            delta_output / max(delta_input, 1e-12)
        )

    return {
        'within_replicate_mse_ratio':       float(ratio_time),
        'within_replicate_train_mse':       float(mse_train),
        'within_replicate_val_mse':         float(mse_val),
        'perturbation_response_ratios':     perturbation_results,
    }
