"""Full-horizon autoregressive rollout evaluator for the M7 surrogate.

Distinct from the per-epoch validation in train_m7.py:
  * Per-epoch val rolls cfg.max_k (e.g. 16) steps from each timestep of
    the val replicate and averages step MSE - cheap, runs every epoch.
  * full_rollout_evaluation rolls ONE trajectory of length T_max=7200
    autoregressively from x_0, then computes per-species R^2 over the
    whole trajectory, per-category R^2, per-step rollout MSE series,
    and the mass-balance residual time series.

This is the headline metric for M7: did the model learn the dynamics,
or only the local Jacobian?

Implementation
--------------
- Forward is the SAME signature as the trainer's: model(x_log, x_var=sigma_t)
  returns (x_next_log, v_log). The PINN bridge already keeps state in
  signed-log1p space, so no extra conversion is needed.
- The variance channel sigma is precomputed from the training replicates
  (or supplied externally). At each rollout step t we feed sigma[t]. This
  matches how the trainer constructs the input — we are NOT cheating
  with future-leaking information, because sigma is a static cross-
  replicate statistic, not a per-trajectory observable.
- Rollout is chunked: we don't store all (T_max, S) predictions in
  memory. We accumulate Sum_t(pred), Sum_t(pred^2), Sum_t((pred-true)^2),
  Sum_t((true-mean_true)^2) and compute R^2 at the end. Mass-balance
  residual at each step is a single scalar so the (T_max,) series is
  kept in full.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from cell_sim.lgnn.training.train_m3 import categorise_row_indices


@dataclass
class RolloutResult:
    """Diagnostic bundle returned by full_rollout_evaluation."""
    T: int                                   # actual rollout length
    mean_r2:           float                 # mean over species with nonzero target variance
    median_r2:         float
    median_r2_top100:  float                 # R^2 over the 100 top-variance species
    r2_per_species:    np.ndarray            # (S,) per-species R^2 (NaN where target variance ~ 0)
    r2_by_category:    Dict[str, float]      # mean over species in each row category
    n_species_by_cat:  Dict[str, int]
    mse_per_step:      np.ndarray            # (T,) mean-square error at each step
    mass_balance_l2:   np.ndarray            # (T,) sqrt-mean-square drift of SBML species
    pred_mean_per_species:  np.ndarray       # (S,) mean over t of predicted state
    pred_std_per_species:   np.ndarray       # (S,) std over t of predicted state
    extras: Dict = field(default_factory=dict)


def _categorise_for_eval(
    row_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Same partition logic as train_m3.categorise_row_indices, returned
    as plain numpy bool masks keyed by category. Used for per-category
    R^2 reporting on top of the trainer's count / flux / cumulative
    split."""
    count_mask, flux_mask, cum_mask = categorise_row_indices(row_names)
    return {
        'count':      count_mask.cpu().numpy(),
        'flux':       flux_mask .cpu().numpy(),
        'cumulative': cum_mask  .cpu().numpy(),
    }


@torch.no_grad()
def full_rollout_evaluation(
    model: torch.nn.Module,
    val_data: torch.Tensor,                # (R_val, T, S) signed-log1p on device
    sigma: torch.Tensor,                   # (T, S) variance channel on device
    row_names: Sequence[str],
    sbml_mask: Optional[torch.Tensor] = None,    # (S,) bool, True on SBML-covered rows
    val_replicate_idx: int = 0,
    x0_step: int = 0,
    T_max: int = 7200,
    log_every: int = 500,
    feed_sigma: bool = True,
    eval_dtype: torch.dtype = torch.float32,
) -> RolloutResult:
    """Autoregressively roll the model from val_data[val_replicate_idx, x0_step]
    for T_max steps, then report per-species and per-category R^2 plus
    rollout MSE and mass-balance series.

    Args
    ----
    model           : M7 (or M-PINN / M3) module. Forward must accept
                       (x, x_var=...) and return (x_next, v_log[, ent]).
                       For models that return just dx (M3 residual form),
                       wrap them so this function still sees x_next.
    val_data        : (R_val, T, S) signed-log1p tensor on the device.
    sigma           : (T, S) variance channel matching val_data's timeline.
                       Only consulted if feed_sigma=True.
    row_names       : LGNN row names (used for per-category R^2).
    sbml_mask       : Optional (S,) bool. If None, we read it off
                       model.pinn_head.s_covered_mask when available;
                       otherwise the mass-balance series is left as zeros.
    val_replicate_idx, x0_step : where to start the rollout.
    T_max           : number of autoregressive steps. Caps automatically
                       at val_data.shape[1] - x0_step - 1.
    log_every       : print progress every N steps.
    feed_sigma      : if True (M3 / M7), pass sigma to the forward;
                       if False (M-PINN's 1-channel encoder), skip it.

    Returns
    -------
    RolloutResult.
    """
    device     = val_data.device
    # Force fp32 for the rollout. bf16 precision (~3 sig figs) compounds over
    # thousands of autoregressive steps, especially through the PINNHead's
    # signed_expm1 bridge for high-count species (proteins ~50k). Empirically
    # a bf16 model produces NaN after ~200-300 steps. fp32 keeps the rollout
    # honest at the cost of ~2x memory during eval. Restore at the end.
    original_dtype = next(model.parameters()).dtype
    if eval_dtype is not None and eval_dtype != original_dtype:
        model = model.to(eval_dtype)
    model_dtype = next(model.parameters()).dtype
    R_v, T_v, S = val_data.shape
    if val_replicate_idx >= R_v:
        raise ValueError(f'val_replicate_idx={val_replicate_idx} '
                         f'>= R_v={R_v}')
    T_max = min(T_max, T_v - x0_step - 1)
    if T_max <= 0:
        raise ValueError('rollout horizon is zero - check x0_step and T_v')

    # Build SBML mask, falling back to model.pinn_head.s_covered_mask
    # then to all-zeros (mass-balance series will be flat at 0).
    if sbml_mask is None:
        if hasattr(model, 'pinn_head'):
            sbml_mask = (model.pinn_head.s_covered_mask > 0).to(device)
        elif hasattr(model, 's_covered_mask'):
            sbml_mask = model.s_covered_mask.to(device).bool()
        else:
            sbml_mask = torch.zeros(S, device=device, dtype=torch.bool)
    else:
        sbml_mask = sbml_mask.to(device).bool()
    n_sbml = int(sbml_mask.sum())

    x_pred = val_data[val_replicate_idx, x0_step].to(model_dtype)             # (S,)
    x0     = x_pred.clone()

    # Accumulators in fp32 to avoid bf16 numerical noise dominating the R^2.
    # We compute R^2 across the rollout window t = x0_step+1 .. x0_step+T_max
    # against ground truth val_data[val_replicate_idx, t].
    sum_target          = torch.zeros(S, device=device, dtype=torch.float32)
    sum_target_sq       = torch.zeros(S, device=device, dtype=torch.float32)
    sum_pred            = torch.zeros(S, device=device, dtype=torch.float32)
    sum_pred_sq         = torch.zeros(S, device=device, dtype=torch.float32)
    sum_diff_sq         = torch.zeros(S, device=device, dtype=torch.float32)
    mse_per_step        = np.zeros(T_max, dtype=np.float64)
    mass_balance_series = np.zeros(T_max, dtype=np.float64)

    model.eval()
    for s in range(T_max):
        t_now  = x0_step + s
        t_next = t_now + 1

        if feed_sigma and sigma is not None:
            v_t = sigma[t_now].to(model_dtype)
        else:
            v_t = None

        # Run forward in a batch dim of 1
        x_in = x_pred.unsqueeze(0)
        v_in = v_t.unsqueeze(0) if v_t is not None else None
        out = model(x_in, x_var=v_in) if v_in is not None else model(x_in)
        if isinstance(out, tuple):
            x_next = out[0].squeeze(0)
        else:
            x_next = out.squeeze(0)

        target = val_data[val_replicate_idx, t_next].to(torch.float32)
        x_next_f32 = x_next.to(torch.float32)
        diff = x_next_f32 - target
        diff_sq = diff.pow(2)

        sum_target    += target
        sum_target_sq += target.pow(2)
        sum_pred      += x_next_f32
        sum_pred_sq   += x_next_f32.pow(2)
        sum_diff_sq   += diff_sq
        mse_per_step[s] = float(diff_sq.mean().item())

        if n_sbml > 0:
            drift = (x_next_f32 - x0.to(torch.float32))[sbml_mask]
            mass_balance_series[s] = float(drift.pow(2).mean().sqrt().item())

        x_pred = x_next

        if (s + 1) % log_every == 0:
            print(f'  rollout step {s+1}/{T_max}  '
                  f'mse_step={mse_per_step[s]:.4f}  '
                  f'sqrt_mb_drift={mass_balance_series[s]:.4f}')

    # ------------------------------------------------------------------
    # R^2 per species (fp32 numpy)
    # R^2 = 1 - SS_res / SS_tot
    # SS_res = sum_t (pred - target)^2  -> sum_diff_sq
    # SS_tot = sum_t (target - mean_target)^2 = sum_target_sq - N * mean^2
    # ------------------------------------------------------------------
    N = float(T_max)
    mean_target = (sum_target / N).cpu().numpy()
    var_target  = (sum_target_sq / N - (sum_target / N).pow(2)).cpu().numpy()
    var_target  = np.maximum(var_target, 0.0)              # numerical floor
    ss_tot      = (var_target * N).astype(np.float64)
    ss_res      = sum_diff_sq.cpu().numpy().astype(np.float64)
    # R^2 only well-defined where the target moves at all.
    has_var = var_target > 1e-9
    r2_per_species = np.full(S, np.nan, dtype=np.float64)
    r2_per_species[has_var] = 1.0 - ss_res[has_var] / np.maximum(ss_tot[has_var], 1e-12)

    valid = r2_per_species[~np.isnan(r2_per_species)]
    mean_r2   = float(valid.mean()) if valid.size > 0 else float('nan')
    median_r2 = float(np.median(valid)) if valid.size > 0 else float('nan')

    # Top-100 by target variance
    top100_idx = np.argsort(-var_target)[:100]
    top100_r2  = r2_per_species[top100_idx]
    top100_r2  = top100_r2[~np.isnan(top100_r2)]
    median_r2_top100 = (float(np.median(top100_r2))
                        if top100_r2.size > 0 else float('nan'))

    # Per-category R^2
    masks = _categorise_for_eval(row_names)
    r2_by_category: Dict[str, float] = {}
    n_by_category:  Dict[str, int] = {}
    for cat, m in masks.items():
        m_with_var = m & ~np.isnan(r2_per_species)
        n_by_category[cat] = int(m_with_var.sum())
        if m_with_var.any():
            r2_by_category[cat] = float(np.nanmean(r2_per_species[m_with_var]))
        else:
            r2_by_category[cat] = float('nan')

    pred_mean = (sum_pred / N).cpu().numpy()
    pred_var  = (sum_pred_sq / N - (sum_pred / N).pow(2)).cpu().numpy()
    pred_var  = np.maximum(pred_var, 0.0)
    pred_std  = np.sqrt(pred_var)

    return RolloutResult(
        T=int(T_max),
        mean_r2=mean_r2,
        median_r2=median_r2,
        median_r2_top100=median_r2_top100,
        r2_per_species=r2_per_species,
        r2_by_category=r2_by_category,
        n_species_by_cat=n_by_category,
        mse_per_step=mse_per_step,
        mass_balance_l2=mass_balance_series,
        pred_mean_per_species=pred_mean,
        pred_std_per_species=pred_std,
    )
