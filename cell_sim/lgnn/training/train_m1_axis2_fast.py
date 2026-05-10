"""M1 + Axis 2 training, fast variant — preload-to-GPU + bf16 + no-checkpoint.

Drops the streaming-DataLoader path entirely. The full training corpus
(49 replicates × 7201 timesteps × 8572 species ≈ 6 GB at bf16) fits
comfortably in a Blackwell-class 96 GB GPU, so we load it once at
training start and index into one big resident tensor for every batch.

Key differences vs train_m1_axis2.py:

  - **No DataLoader, no IterableDataset, no num_workers question.**
    Per-epoch index iterator on GPU. Each batch is one fancy-index
    gather from the preloaded (R, T, S) tensor.

  - **bf16 by default.** Blackwell tensor cores want bf16; fp32 is the
    factor-of-2 you leave on the table. Autocast wraps the rollout +
    dropout forwards and loss compute. Backward + optimizer step in
    fp32 (standard).

  - **use_checkpoint=False by default.** With 96 GB on tap, the
    activation memory at B=256, k=4 is ~10 GB peak — no recompute
    needed. Halves wall-clock vs the checkpointed path.

  - **On-GPU loss accumulators.** No .item() per step (each is a
    forced CUDA sync). Accumulate as scalar tensors on device, sync
    once per log_every interval.

Memory: ~6 GB (train) + 124 MB (val) preloaded + ~10 GB activations +
~5 GB gradients + ~15 MB model = under 25 GB peak. 70+ GB headroom.

Expected throughput on RTX PRO 6000 Blackwell: ~5,000-10,000 samples/sec
(vs 186 s/s in the streaming variant). 4-epoch run: ~10-15 min.
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from cell_sim.lgnn.data.dataset import replicate_to_log1p_array
from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.models.gnn_v1_axis2 import (
    CellGNNv1Axis2, count_parameters,
)


DEFAULT_K_CURRICULUM = (1, 2, 4, 4)


@dataclass
class M1Axis2FastTrainConfig:
    n_species: int = 8572
    hidden: int = 64
    n_layers: int = 3
    batch_size: int = 256
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
    lambda_dropout: float = 0.1              # was 0.5 — too dominant pre-fit

    # Attention sparsity penalty + warmup. Applying λ_attn from step 0
    # collapses attention onto self-loops before the model knows which
    # edges matter (verified: every PM row produced identical weights
    # 0.821/0.142/0.037 on self/RPM/P_TC after the constant-λ run).
    # Schedule:
    #   step < warmup           : λ = 0   (attention learns shape unconstrained)
    #   warmup ≤ step < +ramp   : λ ramps linearly 0 → peak
    #   step ≥ warmup + ramp    : λ = peak
    lambda_attn: float = 1e-4                # peak; was 1e-3 (10× too high)
    lambda_attn_warmup_steps: int = 2000
    lambda_attn_ramp_steps: int = 2000

    k_curriculum: tuple = DEFAULT_K_CURRICULUM
    rollout_gamma: float = 0.95
    lambda_rollout: float = 1.0
    max_k: int = 4
    # When k_cur == 1, L_rollout collapses to L_full_singlestep (same
    # target, redundant gradient). Skip rollout for those epochs so
    # the supervised single-step loss isn't double-counted.
    skip_rollout_at_k1: bool = True

    use_checkpoint: bool = False             # Blackwell-class default
    use_bf16: bool = True
    preload_dtype: str = 'bfloat16'          # train_data dtype on GPU
    # When set, _AttentionGNNLayer processes edges in chunks of this
    # size with checkpointed forward per chunk. Cuts activation memory
    # ~5× per layer for ~30% backward compute. Required for B≥128
    # training at k=4 on Blackwell-class GPUs (without it the (B, E,
    # 2H+A+D) input concat tensor fills VRAM after a few rollout steps).
    edge_chunk_size: Optional[int] = None

    wall_clock_budget_s: float = 1 * 3600.0


def _device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _k_for_epoch(curriculum: tuple, epoch: int) -> int:
    return int(curriculum[min(epoch, len(curriculum) - 1)])


def _lambda_attn_at(global_step: int, peak: float,
                     warmup_steps: int, ramp_steps: int) -> float:
    """Warmup-then-ramp schedule for the attention entropy penalty.
    Returns 0 for global_step < warmup_steps, then ramps linearly to
    peak over the next ramp_steps, then stays at peak."""
    if global_step < warmup_steps:
        return 0.0
    if global_step < warmup_steps + ramp_steps:
        return peak * (global_step - warmup_steps) / max(ramp_steps, 1)
    return peak


def _torch_dtype(name: str) -> torch.dtype:
    return {
        'float32':  torch.float32, 'fp32': torch.float32,
        'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
        'float16':  torch.float16,  'fp16': torch.float16,
    }[name]


def preload_to_gpu(
    replicate_indices,
    lsdata_module,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    species_filter: Optional[List[str]] = None,
) -> torch.Tensor:
    """Load all `replicate_indices` into one (R, T, S) GPU tensor.

    Builds replicate-by-replicate so peak host memory is ~250 MB
    (one fp32 array) regardless of total corpus size. Each replicate
    is moved to GPU + cast in one .to() call, then the host array is
    discarded.
    """
    R = len(list(replicate_indices))
    big = None
    for i, rep_i in enumerate(replicate_indices):
        df = lsdata_module.load_replicate(rep_i, species=species_filter)
        arr = replicate_to_log1p_array(df)             # (T, S) fp32 cpu
        T, S = arr.shape
        if big is None:
            big = torch.empty(R, T, S, device=device, dtype=dtype)
        big[i] = torch.from_numpy(arr).to(device, dtype=dtype)
    return big


def _gather_windows(
    data: torch.Tensor,                 # (R, T, S) on device
    rep_idx: torch.Tensor,              # (B,) long on device
    t_idx:   torch.Tensor,              # (B,) long on device
    window_len: int,                    # max_k + 1
) -> torch.Tensor:
    """Return (B, window_len, S). All on device, no host transfer."""
    arange = torch.arange(window_len, device=data.device).unsqueeze(0)
    t_window = t_idx.unsqueeze(1) + arange                  # (B, window_len)
    rep_b = rep_idx.unsqueeze(1).expand_as(t_window)        # (B, window_len)
    return data[rep_b, t_window]                            # (B, window_len, S)


def _index_iterator(
    R: int, n_valid: int, batch_size: int,
    generator: torch.Generator, device: torch.device,
):
    """Yield (rep_idx, t_idx) tensors of length ≤ batch_size each
    (last batch may be partial). Permutes (R × n_valid) pairs once
    per epoch, drawn on GPU so no host roundtrip per step."""
    total = R * n_valid
    perm = torch.randperm(total, generator=generator, device=device)
    rep_idx_all = perm // n_valid
    t_idx_all   = perm %  n_valid
    for s in range(0, total, batch_size):
        e = min(s + batch_size, total)
        yield rep_idx_all[s:e], t_idx_all[s:e]


def train_m1_axis2_fast(
    cfg: M1Axis2FastTrainConfig,
    lsdata_module,
    graph: SpeciesGraph,
    checkpoint_path: Optional[Path] = None,
) -> dict:
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    pre_dtype = _torch_dtype(cfg.preload_dtype)

    # --- Preload corpus to GPU ---
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

    # --- Model ---
    model = CellGNNv1Axis2(
        graph=graph,
        hidden=cfg.hidden,
        n_layers=cfg.n_layers,
        use_checkpoint=cfg.use_checkpoint,
        edge_chunk_size=cfg.edge_chunk_size,
    ).to(device)
    # Native bf16: convert model params/buffers to bf16 so the entire
    # forward runs in bf16 without autocast's fp32↔bf16 cast overhead
    # at every Linear boundary. Was the dominant cost in the autocast
    # path — autocast_ctx wrapped autocast over a fp32 model and forced
    # cast-on-every-op. Native bf16 keeps activations, params, and data
    # in one dtype.
    if cfg.use_bf16 and device.type == 'cuda':
        model = model.to(torch.bfloat16)
    model_dtype = next(model.parameters()).dtype

    print(f'M1+axis2_fast: {count_parameters(model):,} parameters'
          f'  (hidden={cfg.hidden}, n_layers={cfg.n_layers},'
          f'  ckpt={cfg.use_checkpoint}, dtype={model_dtype})')
    print(f'k-curriculum: {cfg.k_curriculum}   max_k: {cfg.max_k}')

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    history = {
        'train_total': [], 'train_full_ss': [], 'train_drop_ss': [],
        'train_rollout': [], 'train_attn_entropy': [],
        'val_singlestep_mse': [], 'val_rollout_mse': [],
        'k_per_epoch': [], 'samples_per_sec': [],
        'lambda_attn_end_of_epoch': [],
    }
    best_val = float('inf')
    train_t0 = time.time()
    global_step = 0           # persists across epochs for λ_attn schedule

    for epoch in range(cfg.n_epochs):
        k_cur = _k_for_epoch(cfg.k_curriculum, epoch)
        history['k_per_epoch'].append(k_cur)
        # When k_cur == 1, L_rollout duplicates L_full_singlestep. Skip
        # rollout this epoch (set effective λ to 0) so the supervised
        # gradient isn't double-weighted while the model is still
        # learning single-step.
        eff_lambda_rollout = (0.0 if (k_cur == 1 and cfg.skip_rollout_at_k1)
                                else cfg.lambda_rollout)

        gen = torch.Generator(device=device).manual_seed(cfg.seed + epoch)

        model.train()
        # On-GPU running accumulators — sync only at log_every.
        # Keys MUST match the history schema below ('train_<key>').
        run = {k: torch.zeros((), device=device, dtype=torch.float32)
                for k in ('total', 'full_ss', 'drop_ss',
                           'rollout', 'attn_entropy')}
        n_batches = 0
        t_epoch = time.time()
        n_seen = 0

        for rep_idx, t_idx in _index_iterator(
            R, n_valid, cfg.batch_size, gen, device,
        ):
            # Single-window gather on GPU; cast to model dtype so the
            # forward stays in one dtype (bf16 if use_bf16 else fp32).
            x_w = _gather_windows(
                train_data, rep_idx, t_idx, k_cur + 1,
            ).to(model_dtype)
            x = x_w[:, 0, :]                                   # (B, S)

            # 1. Multi-step rollout (k_cur forwards on full graph).
            # Normalize by sum-of-gammas so weights form a proper
            # convex combination (else the dividing-by-k_cur made
            # rollout near-uniform regardless of γ — defeated the
            # whole point of discounting later steps).
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

            # 2. Single-step dropout pass
            dx_drop = model(x, edge_dropout_p=cfg.edge_dropout_p)
            L_drop_ss = mse(x + dx_drop, x_w[:, 1, :])

            # 3. Combined loss with λ_attn warmup schedule.
            # L_full_ss is the s=0 component of L_rollout (same target).
            # When skip_rollout_at_k1 is False at k=1, L_rollout collapses
            # to L_full_ss exactly. When k>1, L_full_ss is *redundant
            # weight* on s=0 since it's already in L_rollout's first
            # term. So we drop L_full_ss from the total loss; it's
            # tracked as a metric only.
            cur_lambda_attn = _lambda_attn_at(
                global_step, cfg.lambda_attn,
                cfg.lambda_attn_warmup_steps,
                cfg.lambda_attn_ramp_steps,
            )
            # When skip_rollout_at_k1 zeros out L_rollout at k=1, restore
            # L_full_ss as the supervised term so the model still trains
            # on single-step (otherwise the supervised gradient signal
            # is gone entirely at k=1).
            if eff_lambda_rollout == 0.0:
                supervised = L_full_ss
            else:
                supervised = eff_lambda_rollout * L_rollout
            L = (supervised
                 + cfg.lambda_dropout * L_drop_ss
                 + cur_lambda_attn    * attn_entropy)

            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()
            global_step += 1

            # On-GPU accumulation — no .item() in hot loop
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
                      f'  {n_seen/wall:.0f} s/s')

            if (time.time() - train_t0) > cfg.wall_clock_budget_s:
                print(f'  WARNING: wall-clock budget '
                      f'{cfg.wall_clock_budget_s/3600:.2f}h exceeded')

        epoch_wall = time.time() - t_epoch
        history['samples_per_sec'].append(n_seen / max(epoch_wall, 1e-6))
        history['lambda_attn_end_of_epoch'].append(
            _lambda_attn_at(global_step, cfg.lambda_attn,
                              cfg.lambda_attn_warmup_steps,
                              cfg.lambda_attn_ramp_steps)
        )
        for k, v in run.items():
            history.setdefault(f'train_{k}', []).append(
                float(v.item()) / max(n_batches, 1)
            )

        val = evaluate_fast(model, val_data, cfg, device, k_eval=k_cur)
        history['val_singlestep_mse'].append(val['mse_singlestep'])
        history['val_rollout_mse'].append(val['mse_rollout'])
        print(f'  ep{epoch} done in {epoch_wall:.1f}s  '
              f'val_ss={val["mse_singlestep"]:.4f}  '
              f'val_roll(k={k_cur})={val["mse_rollout"]:.4f}  '
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
                    'val_mse_rollout': val['mse_rollout'],
                    'k_at_save': k_cur,
                }, checkpoint_path)

    return {
        'history': history,
        'best_val_singlestep_mse': best_val,
        'final_model': model,
        'wall_clock_total_s': time.time() - train_t0,
    }


@torch.no_grad()
def evaluate_fast(
    model: nn.Module,
    val_data: torch.Tensor,                 # (R_val, T, S) on device
    cfg: M1Axis2FastTrainConfig,
    device: torch.device,
    k_eval: int = 4,
) -> dict:
    """Single-step + k-step rollout val MSE on the preloaded val tensor.
    No DataLoader; iterate windows on-GPU. Native dtype throughout —
    if model is bf16, inputs and accumulators stay in bf16."""
    model.eval()
    model_dtype = next(model.parameters()).dtype
    R, T, S = val_data.shape
    n_valid = T - k_eval - 1
    sq_ss = torch.zeros((), device=device, dtype=torch.float32)
    n_ss = 0
    sq_roll = torch.zeros((), device=device, dtype=torch.float32)
    n_roll = 0

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
                if s == 0:
                    sq_ss += err2.sum()
                    n_ss  += err2.numel()
                sq_roll += err2.sum()
                n_roll  += err2.numel()
    return {
        'mse_singlestep': float(sq_ss.item()) / max(n_ss, 1),
        'mse_rollout':    float(sq_roll.item()) / max(n_roll, 1),
    }
