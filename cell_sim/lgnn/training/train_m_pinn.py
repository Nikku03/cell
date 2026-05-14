"""M-PINN training: hardwired stoichiometry + flux-primary loss.

Implements Critiques 8 and 11 from
memory_bank/architecture/lgnn_critiques_and_roadmap.md:

  Critique 8 (PINN with hardwired Δx = S · v):
    The model produces a rate vector v at each F_* species; the
    stoichiometric matrix S then deterministically computes the
    species-count update. Mass and atom conservation are guaranteed
    by construction. Architecture lives in
    cell_sim/lgnn/models/gnn_pinn.py.

  Critique 11 (flux-primary loss, demoted count loss):
    Once Δx = S · v is hardwired, the count target is deterministically
    bound to the flux target. Supervising both equally double-dips the
    gradient. M-PINN uses:
      weight_flux       = 1.0       primary supervision on v vs F_* obs
      weight_cumulative = 0.5       secondary on PM/RPM/DM dynamics
      weight_count      = 0.05      regularization only (prevents negative
                                    predicted counts, doesn't drive learning)

  Also enables Critique 6 (RATE_LAW/MASS_BALANCE split) by using a v6
  graph built with split_flux_coupling=True.

  Critique 14 (RK4 train / dopri5 inference) does NOT apply here: M-PINN
  uses Forward Euler implicitly (Δx applied once per step = solver with
  Δt=1 and no internal integration). When the future M-ODE-PINN wraps the
  rate prediction in torchdiffeq, Critique 14 will apply.

Trains on (x_t, x_{t+1}, F_t) tuples from the simulator. The flux loss
directly compares the model's v_log to the observed F_* signed-log1p
values, which IS exactly what F_* represents in the data (period-averaged
flux). This is direct supervision of the physical rate.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.data.stoichiometric_matrix import (
    load_stoichiometric_matrix_for_pinn,
)
from cell_sim.lgnn.models.gnn_pinn import CellGNN_PINN, count_parameters
from cell_sim.lgnn.training.train_m1_axis2_fast import (
    DEFAULT_K_CURRICULUM,
    _gather_windows, _index_iterator, _torch_dtype, _warmup_ramp,
    preload_to_gpu,
)
from cell_sim.lgnn.training.train_m3 import (
    M3TrainConfig, _device, _k_for_epoch, categorise_row_indices,
)


@dataclass
class MPINNTrainConfig(M3TrainConfig):
    """M-PINN config: PINN architecture + demoted count loss + k-curriculum."""
    # PINN-specific
    # Path to the stoichiometric-matrix source. Accepts SBML XML or .lm.
    # SBML is strongly preferred — the .lm trajectory's reactionConstNames
    # attribute is unreliable (1527 entries vs. 3556 columns) and will yield
    # zero matched F_<rxn> reactions on Syn3A_updated.
    lm_path_for_S: str = ''
    # log-space v clamp. 6.0 -> |v_linear| < ~400, safe under bf16 expm1.
    # The PINNHead default tightened from 12 -> 6 in last session's audit;
    # keep the trainer override aligned so we don't reintroduce the risk.
    rate_clip: float = 6.0

    # Flux-primary loss weights (Critique 11)
    weight_flux:       float = 1.0
    weight_cumulative: float = 0.5
    weight_count:      float = 0.05               # regularization only

    # k-curriculum (smaller than M6 because PINN's hardwired conservation
    # should make long rollouts more stable; can crank up if it converges)
    k_curriculum: tuple = (1, 4, 8, 16)            # last stage may OOM at high
    rollout_gamma: float = 1.0                       # batch; reduce batch if so
    max_k: int = 16
    n_epochs: int = 4

    # Scheduled sampling — keep for stability even though PINN is more robust
    scheduled_sampling: bool = True
    p_ss_max: float = 0.5
    p_ss_warmup_steps: int = 2000

    # Speed defaults (memory-conservative after M6 OOM showed that
    # torch.compile + reduce-overhead leaks across k_curriculum changes):
    use_checkpoint: bool = True                      # activation checkpointing on
    use_compile: bool = False                        # disabled — memory leak
    compile_mode: str = 'reduce-overhead'

    n_input_channels: int = 1                        # PINN doesn't take variance


def _pinn_multi_task_loss(
    x_next_pred: torch.Tensor,        # (B, S) predicted state
    v_log_pred: torch.Tensor,         # (B, R) predicted log-rates
    x_next_target: torch.Tensor,      # (B, S) observed state
    flux_indices: torch.Tensor,       # (R,) which rows of x are F_*
    count_mask: torch.Tensor,         # (S,) bool: non-flux, non-cumulative
    cum_mask: torch.Tensor,           # (S,) bool: PM/RPM/DM cumulative
    w_count: float,
    w_flux: float,
    w_cum: float,
):
    """Flux-primary multi-task loss (Critique 11).

    The flux loss is DIRECT supervision of v_log against the F_* targets
    from the data — this is what F_* observations contain (period-
    averaged log-fluxes). No double-dipping through Δx = S · v.

    The count loss compares the PINN-derived next state to ground truth
    on count rows ONLY (weight 0.05 by default). This is regularization
    against runaway expm1 outputs, not the primary signal.

    The cumulative loss compares predicted PM/RPM/DM counters to ground
    truth — these represent rate counters that the PINN should also fit.
    """
    diff_sq_state = (x_next_pred - x_next_target).pow(2)        # (B, S)

    # Count loss: on count_mask rows (non-flux, non-cumulative)
    if int(count_mask.sum()) > 0:
        mse_count = diff_sq_state[:, count_mask].mean()
    else:
        mse_count = torch.zeros((), device=x_next_pred.device,
                                   dtype=x_next_pred.dtype)

    # Cumulative loss: on PM/RPM/DM rows
    if int(cum_mask.sum()) > 0:
        mse_cum = diff_sq_state[:, cum_mask].mean()
    else:
        mse_cum = torch.zeros((), device=x_next_pred.device,
                                 dtype=x_next_pred.dtype)

    # Flux loss: DIRECT comparison of v_log to F_* targets
    # The F_* observations live in x_next_target at flux_indices.
    flux_target_log = x_next_target.index_select(1, flux_indices)  # (B, R)
    mse_flux = (v_log_pred - flux_target_log).pow(2).mean()

    total = w_count * mse_count + w_flux * mse_flux + w_cum * mse_cum
    return total, {
        'mse_count': mse_count.detach(),
        'mse_flux':  mse_flux.detach(),
        'mse_cum':   mse_cum.detach(),
    }


def _maybe_load_or_preload_mpinn(replicate_indices, lsdata_module, device, dtype,
                                   species_filter, cache_path):
    """Same caching pattern as train_m6._maybe_load_or_preload (shared)."""
    import time
    if cache_path is not None and Path(cache_path).exists():
        t0 = time.time()
        print(f'  loading cached preload from {cache_path}...')
        data = torch.load(cache_path, map_location=device, weights_only=False)
        if data.dim() == 3 and data.shape[0] == len(replicate_indices):
            print(f'  ✓ loaded {tuple(data.shape)} in {time.time()-t0:.1f}s')
            return data.to(dtype) if data.dtype != dtype else data
        else:
            print(f'  cache shape mismatch; rebuilding')
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


def train_m_pinn(
    cfg: MPINNTrainConfig,
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

    if not cfg.lm_path_for_S:
        raise ValueError(
            'cfg.lm_path_for_S must be set. Prefer the SBML model '
            '(cell_sim/data/Minimal_Cell_ComplexFormation/input_data/'
            'Syn3A_updated.xml); .lm trajectories are accepted but '
            'unreliable for column mapping.'
        )

    # --- Load stoichiometric matrix + F_* index ---
    print(f'Loading stoichiometric matrix from {cfg.lm_path_for_S}...')
    S_pinn, flux_indices, reaction_ids, mapped_mask = (
        load_stoichiometric_matrix_for_pinn(cfg.lm_path_for_S, row_names)
    )
    print(f'S_pinn shape: {tuple(S_pinn.shape)}  '
          f'flux_indices: {flux_indices.shape[0]}')
    if flux_indices.numel() == 0:
        raise RuntimeError(
            'S_pinn has zero columns — no F_<rxn> species matched a reaction. '
            'This makes the flux loss NaN. Point cfg.lm_path_for_S at the '
            'SBML model (Syn3A_updated.xml) instead of the .lm trajectory.'
        )
    flux_indices = flux_indices.to(device)

    # --- Preload corpus (with on-disk cache; reuses M6's cache files) ---
    print(f'preloading {len(cfg.train_replicates)} train + '
          f'{len(cfg.val_replicates)} val replicates to {device}...')
    train_data = _maybe_load_or_preload_mpinn(
        cfg.train_replicates, lsdata_module, device, pre_dtype,
        cfg.species_filter, train_cache_path,
    )
    val_data = _maybe_load_or_preload_mpinn(
        cfg.val_replicates, lsdata_module, device, pre_dtype,
        cfg.species_filter, val_cache_path,
    )
    R, T, S = train_data.shape
    n_valid = T - cfg.max_k - 1
    print(f'  shape: train{tuple(train_data.shape)}  val{tuple(val_data.shape)}')

    count_mask, flux_mask, cum_mask = categorise_row_indices(row_names)
    count_mask = count_mask.to(device); flux_mask = flux_mask.to(device)
    cum_mask   = cum_mask  .to(device)
    print(f'rows: count={int(count_mask.sum())}  '
          f'flux={int(flux_mask.sum())}  cumulative={int(cum_mask.sum())}')

    # --- Model ---
    model = CellGNN_PINN(
        graph=graph,
        stoich_matrix=S_pinn,
        flux_indices=flux_indices.cpu(),
        hidden=cfg.hidden, n_layers=cfg.n_layers,
        use_checkpoint=cfg.use_checkpoint,
        edge_chunk_size=cfg.edge_chunk_size,
        cfc_tau_min=cfg.cfc_tau_min,
        rate_clip=cfg.rate_clip,
    ).to(device)
    if cfg.use_bf16 and device.type == 'cuda':
        model = model.to(torch.bfloat16)
    model_dtype = next(model.parameters()).dtype
    n_params = count_parameters(model)
    print(f'M-PINN: {n_params:,} parameters (hidden={cfg.hidden}, '
          f'n_layers={cfg.n_layers}, dtype={model_dtype})')
    print(f'k_curriculum={cfg.k_curriculum}, rollout_gamma={cfg.rollout_gamma}, '
          f'scheduled_sampling={cfg.scheduled_sampling}')
    print(f'loss weights: count={cfg.weight_count}, flux={cfg.weight_flux}, '
          f'cumulative={cfg.weight_cumulative}')

    if cfg.use_compile and device.type == 'cuda':
        try:
            model = torch.compile(model, mode=cfg.compile_mode)
            print(f'  torch.compile enabled (mode={cfg.compile_mode})')
        except Exception as e:
            print(f'  WARNING: torch.compile failed ({e})')

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
        p_ss = 0.0

        for rep_idx, t_idx in _index_iterator(R, n_valid, cfg.batch_size,
                                                gen, device):
            x_w = _gather_windows(train_data, rep_idx, t_idx, k_cur + 1
                                    ).to(model_dtype)
            x = x_w[:, 0, :]

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
                # Scheduled sampling: replace x_pred with ground truth at
                # step s with probability (1 - p_ss).
                if s > 0 and p_ss < 1.0:
                    use_tf = (torch.rand(x_pred.shape[0], 1, device=device)
                                > p_ss).to(x_pred.dtype)
                    x_input = use_tf * x_w[:, s, :] + (1 - use_tf) * x_pred
                else:
                    x_input = x_pred

                if s == 0:
                    x_next, v_log, ent = model(x_input,
                                                 return_attention_entropy=True)
                    attn_entropy = ent
                else:
                    x_next, v_log = model(x_input)

                x_pred = x_next                                 # PINN output IS x_next

                step_loss, breakdown = _pinn_multi_task_loss(
                    x_next, v_log, x_w[:, s + 1, :],
                    flux_indices, count_mask, cum_mask,
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

            if eff_lambda_rollout == 0.0:
                supervised = L_full_ss
            else:
                supervised = eff_lambda_rollout * L_rollout
            L = supervised + cur_lambda_attn * attn_entropy

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
                      f'  {n_seen/wall:.0f} s/s')

            if (time.time() - train_t0) > cfg.wall_clock_budget_s:
                print(f'  WARNING: wall-clock budget exceeded')
                break

        epoch_wall = time.time() - t_epoch
        history['samples_per_sec'].append(n_seen / max(epoch_wall, 1e-6))
        history['train_p_ss'].append(p_ss)
        for key in ('total','mse_count','mse_flux','mse_cum',
                     'rollout','attn_entropy'):
            history[f'train_{key}'].append(
                float(run[key].item()) / max(n_batches, 1)
            )

        # --- Validation ---
        model.eval()
        val_metrics = _evaluate_pinn(model, val_data, flux_indices,
                                       count_mask, cum_mask, cfg, model_dtype,
                                       device)
        for k, v in val_metrics.items():
            key = f'val_{k}'
            history.setdefault(key, []).append(v)
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
                    'reaction_ids': reaction_ids,
                    'flux_indices': flux_indices.cpu(),
                    'architecture': 'M_PINN',
                }
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, checkpoint_path)
                print(f'  -> saved best (val_ss={val_ss:.4f}) to {checkpoint_path}')

    return history


@torch.no_grad()
def _evaluate_pinn(model, val_data, flux_indices,
                    count_mask, cum_mask, cfg, model_dtype, device):
    R_v, T_v, S = val_data.shape
    n_valid = T_v - cfg.max_k - 1
    bs = cfg.batch_size
    mse_per_step_state = [0.0] * cfg.max_k
    mse_per_step_flux  = [0.0] * cfg.max_k
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
            x_next, v_log = model(x_pred)
            mse_per_step_state[s] += float(
                (x_next - x_w[:, s + 1, :]).pow(2).mean().item()
            )
            mse_per_step_flux[s] += float(
                (v_log - x_w[:, s + 1, :].index_select(1, flux_indices))
                .pow(2).mean().item()
            )
            x_pred = x_next
            if s == 0:
                # per-modality breakdown (delta in state)
                diff_sq = (x_next - x_w[:, s + 1, :]).pow(2)
                if int(count_mask.sum()) > 0:
                    total_full_count += float(diff_sq[:, count_mask].mean().item())
                if int(cum_mask.sum()) > 0:
                    total_full_cum += float(diff_sq[:, cum_mask].mean().item())
                total_full_flux += float(
                    (v_log - x_w[:, s + 1, :].index_select(1, flux_indices))
                    .pow(2).mean().item()
                )
                n_full += 1
        cnt += 1

    mse_per_step_state = [m / max(cnt, 1) for m in mse_per_step_state]
    mse_per_step_flux = [m / max(cnt, 1) for m in mse_per_step_flux]
    rollout_avg = sum(mse_per_step_state) / len(mse_per_step_state)
    return {
        'singlestep_mse':    mse_per_step_state[0],
        'mse_count':         total_full_count / max(n_full, 1),
        'mse_flux':          total_full_flux  / max(n_full, 1),
        'mse_cum':           total_full_cum   / max(n_full, 1),
        'rollout_mse_avg':   rollout_avg,
        'mse_per_step':      mse_per_step_state,
        'mse_per_step_flux': mse_per_step_flux,
    }
