"""Knockout sweep: rank Syn3A genes by predicted impact of knockout.

This is the M7's "actual application." For every gene G_<locus>:
  1. Take an initial state x_0 from val data
  2. Set x_0[G_<locus>] = 0  (the knockout)
  3. Roll M7 forward for `rollout_steps` seconds
  4. Measure the divergence from the unperturbed prediction
  5. Genes with the largest divergence are predicted-essential

Compared to Breuer 2019's wet-lab essentiality labels, this gives an
MCC or AUROC score - the headline result for the screening use case.

What makes this useful:
  - The simulator takes ~4-8 hours per knockout. M7 takes ~1 sec/knockout.
  - 455 knockouts: simulator ~ 30 hours, M7 ~ 10 min.
  - Even moderate ranking quality (MCC > 0.3, AUROC > 0.65) is a useful
    pre-screen for which knockouts deserve full simulation.

This module does NOT require retraining M7. It just runs the existing
trained model on perturbed initial conditions.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


_G_LOCUS_RE = re.compile(r'^G_(\d{4})(?:_[A-Za-z0-9]+)?$')


def _gene_row_indices(row_names: Sequence[str]):
    """Return {locus_4d: [row_idx, ...]} for every G_<locus> row.

    Handles both bare 'G_0001' and the C1/C2 cell-cycle copies. All
    variants for the same locus group together so knockouts zero
    all of them.
    """
    out: Dict[str, List[int]] = {}
    for i, n in enumerate(row_names):
        m = _G_LOCUS_RE.match(n)
        if m is None:
            continue
        out.setdefault(m.group(1), []).append(i)
    return out


def _impact_score(
    x_baseline,
    x_perturbed,
    *,
    metric: str,
    sbml_mask=None,
    log_count_mask=None,
) -> float:
    """Compute a scalar impact score between two trajectories.

    Both inputs have shape (T, S) - state per timestep across all species.
    """
    import torch
    diff = (x_perturbed - x_baseline).float()
    if metric == 'l2_drift':
        # Mean ||x_pert - x_base||_2 over time
        return float(diff.pow(2).mean().sqrt().item())
    if metric == 'final_drift':
        # ||x_pert[T] - x_base[T]||_2 at end
        return float(diff[-1].pow(2).mean().sqrt().item())
    if metric == 'sbml_drift' and sbml_mask is not None:
        return float(diff[:, sbml_mask].pow(2).mean().sqrt().item())
    raise ValueError(f'unknown metric {metric!r}')


def run_knockout_sweep(
    model,
    val_data,                       # (R_val, T, S) tensor
    sigma,                          # (T, S) variance channel
    row_names: Sequence[str],
    *,
    x0_step: int = 0,
    rollout_steps: int = 50,
    knockout_loci: Optional[Sequence[str]] = None,
    impact_metric: str = 'l2_drift',
    eval_dtype=None,
    feed_sigma: bool = True,
    log_every: int = 50,
) -> Any:
    """Run a knockout for every gene locus, return a ranked DataFrame.

    Args
    ----
    model : trained M7 model (CellGNNv7Hybrid). Will be evaluated in eval mode.
    val_data : (R_val, T, S) tensor on the model's device.
    sigma : (T, S) variance channel tensor.
    row_names : species names matching val_data axis 2.
    x0_step : which timestep of val_data[0] to start from.
    rollout_steps : how many seconds to roll for each scenario. Short =
                    cheap and reliable; long = more comprehensive but
                    error compounds (we're still in M7's working range
                    at ~50 steps).
    knockout_loci : list of 4-digit locus strings to test. None = all genes.
    impact_metric : 'l2_drift', 'final_drift', or 'sbml_drift'.

    Returns pandas DataFrame:
      locus_4d, locus_tag, impact_score, n_gene_rows_zeroed
    sorted by impact_score descending. Higher score = predicted-essential.
    """
    import torch
    import pandas as pd

    if eval_dtype is None:
        eval_dtype = next(model.parameters()).dtype
    device = val_data.device

    gene_groups = _gene_row_indices(row_names)
    if knockout_loci is not None:
        gene_groups = {k: v for k, v in gene_groups.items()
                       if k in set(knockout_loci)}
    if not gene_groups:
        raise ValueError('no gene loci to sweep')

    sbml_mask = None
    if impact_metric == 'sbml_drift' and hasattr(model, 'pinn_head'):
        sbml_mask = (model.pinn_head.s_covered_mask > 0).to(device)

    # --- Baseline rollout (no knockout) ---
    def _rollout(x0_state):
        """Run a rollout from a given starting state. Returns (T, S) tensor."""
        x_pred = x0_state.clone()
        states = [x_pred.clone()]
        for s in range(rollout_steps):
            t_now = x0_step + s
            x_in = x_pred.unsqueeze(0).to(eval_dtype)
            v_in = (sigma[t_now].unsqueeze(0).to(eval_dtype)
                    if feed_sigma else None)
            with torch.no_grad():
                out = model(x_in, x_var=v_in) if v_in is not None else model(x_in)
            x_next = out[0].squeeze(0) if isinstance(out, tuple) else out.squeeze(0)
            x_pred = x_next
            states.append(x_pred.clone())
        return torch.stack(states, dim=0).float()        # (rollout_steps+1, S)

    print(f'running baseline rollout ({rollout_steps} steps from t={x0_step})...')
    t_b0 = time.time()
    x0_base = val_data[0, x0_step].to(eval_dtype)
    baseline = _rollout(x0_base)
    print(f'  baseline done in {time.time()-t_b0:.1f}s')

    # --- Per-locus knockout rollouts ---
    rows = []
    t_sweep_t0 = time.time()
    sorted_loci = sorted(gene_groups.keys())
    for i, locus in enumerate(sorted_loci):
        x0_pert = x0_base.clone()
        gene_rows = gene_groups[locus]
        for gr in gene_rows:
            x0_pert[gr] = 0.0
        perturbed = _rollout(x0_pert)
        score = _impact_score(
            baseline, perturbed,
            metric=impact_metric, sbml_mask=sbml_mask,
        )
        rows.append({
            'locus_4d': locus,
            'locus_tag': f'JCVISYN3A_{locus}',
            'impact_score': score,
            'n_gene_rows_zeroed': len(gene_rows),
        })
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t_sweep_t0
            rate = (i + 1) / elapsed
            eta = (len(sorted_loci) - i - 1) / rate
            print(f'  [{i+1}/{len(sorted_loci)}] {locus} '
                  f'score={score:.4f}  rate={rate:.1f}/s  eta={eta:.0f}s')

    sweep_time = time.time() - t_sweep_t0
    df = pd.DataFrame(rows).sort_values('impact_score', ascending=False).reset_index(drop=True)
    df.attrs['baseline_seconds'] = time.time() - t_b0
    df.attrs['sweep_seconds']    = sweep_time
    df.attrs['rollout_steps']    = rollout_steps
    df.attrs['impact_metric']    = impact_metric
    return df


def score_against_breuer(
    sweep_df,
    breuer_xlsx_path: Path,
):
    """Add Breuer 2019 essentiality labels and compute AUROC + MCC at
    several score thresholds.

    Returns the sweep_df augmented with a 'breuer_essential' column,
    plus an `evaluation` dict with AUROC, MCC at top-K, etc.
    """
    import pandas as pd
    from cell_sim.lgnn.data.static_node_features import load_breuer_labels

    breuer = load_breuer_labels(Path(breuer_xlsx_path))
    breuer_map = {row['locus_tag']: bool(row['essential'])
                  for _, row in breuer.iterrows()}

    df = sweep_df.copy()
    df['breuer_essential'] = df['locus_tag'].map(breuer_map)
    df['breuer_label_known'] = df['breuer_essential'].notna()
    labelled = df[df['breuer_label_known']].copy()
    labelled['breuer_essential'] = labelled['breuer_essential'].astype(bool)

    if labelled.empty:
        return df, {'note': 'no overlap with Breuer labels'}

    # AUROC (predicted-essential = high score)
    try:
        from sklearn.metrics import roc_auc_score, matthews_corrcoef
        y_true = labelled['breuer_essential'].values
        y_score = labelled['impact_score'].values
        auroc = float(roc_auc_score(y_true, y_score))
        # MCC at the top-K threshold where K = number of essentials in Breuer
        K = int(y_true.sum())
        sorted_idx = np.argsort(-y_score)
        y_pred = np.zeros_like(y_true)
        y_pred[sorted_idx[:K]] = True
        mcc = float(matthews_corrcoef(y_true, y_pred))
        n_total = int(len(y_true))
        n_essential = int(y_true.sum())
        n_top_correct = int((y_pred & y_true).sum())
        evaluation = {
            'auroc': auroc,
            'mcc_top_K': mcc,
            'K_threshold': K,
            'n_total_labelled': n_total,
            'n_essential_in_breuer': n_essential,
            'n_top_K_correctly_essential': n_top_correct,
            'precision_at_K': n_top_correct / max(K, 1),
        }
    except ImportError:
        evaluation = {'error': 'sklearn not installed; skipping AUROC'}
    return df, evaluation
