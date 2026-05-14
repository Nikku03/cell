"""M-Bio - dual-objective LGNN with bundled biology features.

Combines:
  - M4's static-node-features encoder (gene-class + spatial + role + extras
    + new proteomics columns from `initial_concentrations.xlsx`)
  - A per-gene essentiality classification head supervised on Breuer
    labels (also bundled in `initial_concentrations.xlsx`)
  - The same dynamics multi-task loss (count + flux + cumulative) as
    M3/M4, with k-curriculum rollout + scheduled sampling

Two-axis evaluation every epoch:
  - val_singlestep_mse: forecast quality (M3 baseline ~0.035)
  - val_essentiality_mcc: classifier MCC on held-out gene fold
    (M3 knockout-sweep baseline 0.126 — different metric but same scale)

Goal: simultaneously beat M3 on val_ss AND on essentiality MCC.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.data.static_node_features import (
    build_static_node_features, ALL_STATIC_COLS, feature_summary,
    load_breuer_labels,
)
from cell_sim.lgnn.data.breuer_alignment import (
    extract_gene_row_indices, build_breuer_label_tensor, gene_kfold_split,
)
from cell_sim.lgnn.models.gnn_v5_bio import CellGNN_Bio, count_parameters
from cell_sim.lgnn.training.train_m1_axis2_fast import (
    DEFAULT_K_CURRICULUM,
    _gather_windows, _index_iterator, _torch_dtype, _warmup_ramp,
    preload_to_gpu,
)
from cell_sim.lgnn.training.train_m3 import (
    M3TrainConfig, _device, _k_for_epoch, categorise_row_indices,
    compute_variance_channel, _multi_task_mse,
    _evaluate as _m3_evaluate,
)


@dataclass
class MBioTrainConfig(M3TrainConfig):
    """M-Bio config: M3 dynamics + static features + essentiality head."""
    gene_class_csv: str = 'memory_bank/data/syn3a_gene_class_features.csv'
    proteomics_xlsx: str = (
        'cell_sim/data/Minimal_Cell_ComplexFormation/input_data/'
        'initial_concentrations.xlsx'
    )
    spatial_parquet: Optional[str] = None
    # BCE loss weight. At convergence dynamics ~0.04 / BCE ~0.3, so 0.1 puts
    # them on roughly the same scale. Crank up if essentiality MCC isn't moving.
    weight_essentiality: float = 0.1
    essentiality_dropout: float = 0.10
    essentiality_n_folds: int = 5
    essentiality_fold: int = 0             # which fold's val set to hold out
    essentiality_seed: int = 0


def _binary_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """Return MCC, precision, recall, F1, accuracy from logits + binary labels."""
    p = (torch.sigmoid(logits) > threshold).cpu().numpy().astype(np.int64)
    y = (labels > 0.5).cpu().numpy().astype(np.int64)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    n = tp + fp + fn + tn
    accuracy = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    denom = float(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) ** 0.5
    mcc = (tp * tn - fp * fn) / max(denom, 1e-12)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'F1': f1, 'MCC': mcc}


@torch.no_grad()
def _evaluate_bio(
    model: nn.Module,
    val_data: torch.Tensor,
    sigma: torch.Tensor,
    count_mask: torch.Tensor,
    flux_mask: torch.Tensor,
    cum_mask: torch.Tensor,
    cfg: MBioTrainConfig,
    model_dtype,
    device,
    ess_labels: torch.Tensor,
    ess_train_mask: torch.Tensor,
    ess_val_mask: torch.Tensor,
) -> dict:
    """Run M3-style dynamics eval AND collect essentiality predictions."""
    base = _m3_evaluate(model, val_data, sigma, count_mask, flux_mask, cum_mask,
                          cfg, model_dtype, device)
    # Average essentiality logits across all val batches & timesteps
    R, T, S = val_data.shape
    n_take = min(64, T - 1)
    sample_t = torch.linspace(0, T - 2, n_take, dtype=torch.long, device=device)
    logits_acc = []
    for t in sample_t.tolist():
        x = val_data[:, t, :].to(model_dtype)
        v = sigma[t].to(model_dtype).unsqueeze(0).expand(R, -1)
        h, _ = model.encode(x, x_var=v)
        ess_logits = model.predict_essentiality(h)               # (R, n_genes)
        logits_acc.append(ess_logits.float().mean(dim=0))         # (n_genes,)
    avg_logits = torch.stack(logits_acc, dim=0).mean(dim=0)       # (n_genes,)

    # Train-fold metrics + Val-fold metrics (val is the "held out gene fold")
    if int(ess_train_mask.sum()) > 0:
        m_tr = _binary_classification_metrics(
            avg_logits[ess_train_mask], ess_labels[ess_train_mask])
    else:
        m_tr = {}
    if int(ess_val_mask.sum()) > 0:
        m_va = _binary_classification_metrics(
            avg_logits[ess_val_mask], ess_labels[ess_val_mask])
    else:
        m_va = {}

    base['ess_train_metrics'] = m_tr
    base['ess_val_metrics']   = m_va
    base['ess_avg_logits']    = avg_logits.cpu()
    return base


def train_m_bio(
    cfg: MBioTrainConfig,
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

    # --- Static node features (with proteomics) ---
    print(f'Building static node features...')
    print(f'  gene_class:  {cfg.gene_class_csv}')
    print(f'  proteomics:  {cfg.proteomics_xlsx}')
    print(f'  spatial:     {cfg.spatial_parquet or "(none)"}')
    static_features = build_static_node_features(
        row_names,
        gene_class_csv=cfg.gene_class_csv,
        spatial_parquet=cfg.spatial_parquet,
        initial_state_signed_log1p=None,
        proteomics_xlsx=cfg.proteomics_xlsx,
    )
    print(f'  shape: {tuple(static_features.shape)}')

    # --- Gene index + Breuer labels + fold split ---
    gene_idx, gene_loci = extract_gene_row_indices(row_names)
    print(f'\nG_<locus> rows in graph: {gene_idx.numel()}')
    breuer = load_breuer_labels(cfg.proteomics_xlsx)
    print(f'Breuer labels: {len(breuer)}  '
          f'(essential={int(breuer.essential.sum())}, '
          f'non={int((~breuer.essential).sum())})')
    folds = gene_kfold_split(gene_loci, breuer,
                               n_splits=cfg.essentiality_n_folds,
                               seed=cfg.essentiality_seed)
    train_loci, val_loci = folds[cfg.essentiality_fold]
    print(f'  fold {cfg.essentiality_fold}/{cfg.essentiality_n_folds}: '
          f'train={len(train_loci)}, val={len(val_loci)}')
    ess_labels, ess_tr_mask, ess_va_mask = build_breuer_label_tensor(
        gene_loci, breuer, train_loci=train_loci, val_loci=val_loci)
    ess_labels   = ess_labels  .to(device)
    ess_tr_mask  = ess_tr_mask .to(device)
    ess_va_mask  = ess_va_mask .to(device)
    n_ess_tr = int(ess_tr_mask.sum())
    n_ess_va = int(ess_va_mask.sum())
    print(f'  trainable gene labels: {n_ess_tr}  '
          f'(essential={int(ess_labels[ess_tr_mask].sum())})')
    print(f'  held-out gene labels:  {n_ess_va}  '
          f'(essential={int(ess_labels[ess_va_mask].sum())})')

    # --- Preload corpus (with on-disk cache) ---
    print(f'\npreloading {len(cfg.train_replicates)} train + '
          f'{len(cfg.val_replicates)} val replicates to {device}...')
    def _load_or_preload(reps, cache_path):
        if cache_path is not None and Path(cache_path).exists():
            t0 = time.time()
            data = torch.load(cache_path, map_location=device, weights_only=False)
            if data.dim() == 3 and data.shape[0] == len(reps):
                print(f'  loaded {tuple(data.shape)} from cache in '
                      f'{time.time()-t0:.1f}s')
                return data.to(pre_dtype) if data.dtype != pre_dtype else data
            print(f'  cache shape mismatch; rebuilding')
        t0 = time.time()
        data = preload_to_gpu(reps, lsdata_module, device,
                                 dtype=pre_dtype, species_filter=cfg.species_filter)
        print(f'  preloaded fresh in {time.time()-t0:.1f}s')
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(data.cpu(), cache_path)
        return data

    train_data = _load_or_preload(cfg.train_replicates, train_cache_path)
    val_data   = _load_or_preload(cfg.val_replicates,   val_cache_path)
    R, T, S = train_data.shape
    n_valid = T - cfg.max_k - 1
    print(f'  shape: train{tuple(train_data.shape)}  val{tuple(val_data.shape)}')

    count_mask, flux_mask, cum_mask = categorise_row_indices(row_names)
    count_mask = count_mask.to(device); flux_mask = flux_mask.to(device)
    cum_mask   = cum_mask  .to(device)
    print(f'rows: count={int(count_mask.sum())}  '
          f'flux={int(flux_mask.sum())}  '
          f'cumulative={int(cum_mask.sum())}')

    print('precomputing per-species cross-replicate std...')
    sigma = compute_variance_channel(train_data)

    # --- Model ---
    model = CellGNN_Bio(
        graph=graph,
        static_features=static_features,
        gene_indices=gene_idx,
        hidden=cfg.hidden, n_layers=cfg.n_layers,
        n_dynamic_channels=cfg.n_input_channels,
        use_checkpoint=cfg.use_checkpoint,
        edge_chunk_size=cfg.edge_chunk_size,
        cfc_tau_min=cfg.cfc_tau_min,
        essentiality_dropout=cfg.essentiality_dropout,
    ).to(device)
    if cfg.use_bf16 and device.type == 'cuda':
        model = model.to(torch.bfloat16)
    model_dtype = next(model.parameters()).dtype
    n_params = count_parameters(model)
    print(f'M-Bio: {n_params:,} parameters '
          f'(hidden={cfg.hidden}, n_layers={cfg.n_layers}, '
          f'n_genes={int(gene_idx.numel())}, dtype={model_dtype})')
    print(f'loss weights: count={cfg.weight_count}, flux={cfg.weight_flux}, '
          f'cumulative={cfg.weight_cumulative}, '
          f'essentiality={cfg.weight_essentiality}')
    print(f'k_curriculum={cfg.k_curriculum}, rollout_gamma={cfg.rollout_gamma}')

    if cfg.use_compile and device.type == 'cuda':
        try:
            model = torch.compile(model, mode=cfg.compile_mode)
            print(f'  torch.compile enabled (mode={cfg.compile_mode})')
        except Exception as e:
            print(f'  WARNING: torch.compile failed ({e})')

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)

    history: dict = {
        'train_total': [], 'train_mse_count': [], 'train_mse_flux': [],
        'train_mse_cum': [], 'train_rollout': [], 'train_ess_bce': [],
        'k_per_epoch': [], 'samples_per_sec': [],
    }
    best_combined = float('inf')
    train_t0 = time.time()
    global_step = 0

    # pre-cache labels broadcast for BCE (per-batch)
    ess_tr_idx = ess_tr_mask.nonzero(as_tuple=True)[0]
    ess_tr_lab = ess_labels[ess_tr_idx]    # (n_ess_tr,)

    for epoch in range(cfg.n_epochs):
        k_cur = _k_for_epoch(cfg.k_curriculum, epoch)
        history['k_per_epoch'].append(k_cur)
        eff_lambda_rollout = (0.0 if (k_cur == 1 and cfg.skip_rollout_at_k1)
                                else cfg.lambda_rollout)

        gen = torch.Generator(device=device).manual_seed(cfg.seed + epoch)
        model.train()
        run = {k: torch.zeros((), device=device, dtype=torch.float32)
                for k in ('total', 'mse_count', 'mse_flux', 'mse_cum',
                          'rollout', 'ess_bce')}
        n_batches = 0
        t_epoch = time.time()
        n_seen = 0

        for rep_idx, t_idx in _index_iterator(R, n_valid, cfg.batch_size,
                                                 gen, device):
            x_w = _gather_windows(train_data, rep_idx, t_idx, k_cur + 1
                                    ).to(model_dtype)
            x = x_w[:, 0, :]
            v0 = sigma[t_idx].to(model_dtype)

            # First step: encode once, get dynamics + essentiality
            h, _ = model.encode(x, x_var=v0)
            dx = model.predict_dynamics(h)
            x_pred = x + dx
            target = x_w[:, 1, :]
            L_full_ss, mse_breakdown_step0 = _multi_task_mse(
                x_pred, target, count_mask, flux_mask, cum_mask,
                cfg.weight_count, cfg.weight_flux, cfg.weight_cumulative,
            )
            L_rollout = L_full_ss
            gamma_sum = 1.0

            # Essentiality BCE on this batch
            ess_logits = model.predict_essentiality(h)            # (B, n_genes)
            ess_logits_tr = ess_logits.index_select(1, ess_tr_idx).float()
            target_b = ess_tr_lab.unsqueeze(0).expand_as(ess_logits_tr)
            L_ess = nn.functional.binary_cross_entropy_with_logits(
                ess_logits_tr, target_b)

            # Subsequent rollout steps (dynamics only)
            for s in range(1, k_cur):
                v_cur = sigma[t_idx + s].to(model_dtype)
                dx = model(x_pred, x_var=v_cur)
                x_pred = x_pred + dx
                target = x_w[:, s + 1, :]
                step_loss, _ = _multi_task_mse(
                    x_pred, target, count_mask, flux_mask, cum_mask,
                    cfg.weight_count, cfg.weight_flux, cfg.weight_cumulative,
                )
                w = cfg.rollout_gamma ** s
                L_rollout = L_rollout + w * step_loss
                gamma_sum += w
            L_rollout = L_rollout / max(gamma_sum, 1e-12)

            if eff_lambda_rollout == 0.0:
                supervised = L_full_ss
            else:
                supervised = eff_lambda_rollout * L_rollout
            L = supervised + cfg.weight_essentiality * L_ess

            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()
            global_step += 1

            run['total']     += L.detach().float()
            run['mse_count'] += mse_breakdown_step0['mse_count'].float()
            run['mse_flux']  += mse_breakdown_step0['mse_flux'].float()
            run['mse_cum']   += mse_breakdown_step0['mse_cum'].float()
            run['rollout']   += L_rollout.detach().float()
            run['ess_bce']   += L_ess.detach().float()
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
                      f'  ess={vals["ess_bce"]:.4f}'
                      f'  {n_seen/wall:.0f} s/s')

            if (time.time() - train_t0) > cfg.wall_clock_budget_s:
                print(f'  WARNING: wall-clock budget exceeded')
                break

        epoch_wall = time.time() - t_epoch
        history['samples_per_sec'].append(n_seen / max(epoch_wall, 1e-6))
        for key in ('total', 'mse_count', 'mse_flux', 'mse_cum',
                     'rollout', 'ess_bce'):
            history[f'train_{key}'].append(
                float(run[key].item()) / max(n_batches, 1)
            )

        # --- Evaluation ---
        model.eval()
        val_metrics = _evaluate_bio(
            model, val_data, sigma, count_mask, flux_mask, cum_mask,
            cfg, model_dtype, device,
            ess_labels=ess_labels,
            ess_train_mask=ess_tr_mask,
            ess_val_mask=ess_va_mask,
        )
        for k, v in val_metrics.items():
            history.setdefault(f'val_{k}', []).append(v)

        val_ss = val_metrics['singlestep_mse']
        m_va = val_metrics['ess_val_metrics']
        m_tr = val_metrics['ess_train_metrics']
        ess_mcc_va = m_va.get('MCC', float('nan'))
        ess_mcc_tr = m_tr.get('MCC', float('nan'))
        print(f'ep{epoch} k={k_cur}  '
              f'val_ss={val_ss:.4f}  '
              f'roll_avg={val_metrics["rollout_mse_avg"]:.4f}  '
              f'ess_mcc_train={ess_mcc_tr:.3f}  '
              f'ess_mcc_val={ess_mcc_va:.3f}  '
              f'(P={m_va.get("precision", 0):.2f} R={m_va.get("recall", 0):.2f})  '
              f'wall={epoch_wall:.1f}s')

        # Combined "best" metric: lower val_ss + (1 - MCC) — pick checkpoint
        # that beats both axes.
        combined = val_ss + (1.0 - max(ess_mcc_va, 0.0)) * 0.5
        if combined < best_combined:
            best_combined = combined
            if checkpoint_path is not None:
                payload = {
                    'state_dict': (model._orig_mod.state_dict()
                                    if hasattr(model, '_orig_mod')
                                    else model.state_dict()),
                    'cfg': cfg.__dict__,
                    'epoch': epoch,
                    'val_singlestep_mse': val_ss,
                    'val_ess_mcc': ess_mcc_va,
                    'val_ess_metrics': m_va,
                    'k_at_save': k_cur,
                    'gene_loci': gene_loci,
                    'static_feature_cols': ALL_STATIC_COLS,
                }
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, checkpoint_path)
                print(f'  -> saved best (val_ss={val_ss:.4f}, '
                      f'ess_mcc={ess_mcc_va:.3f}) to {checkpoint_path}')

    return history
