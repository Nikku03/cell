"""Gene-by-gene knockout sweep using a trained dynamics model.

The pattern: for each gene g in the test set, clamp its row(s) to zero
across an autoregressive rollout and measure how far the system
diverges from a wild-type rollout starting from the same state.
Compare the per-gene divergence ranking against an external
essentiality ground truth (e.g., Breuer 2019).

If the model has learned dynamics that route through gene products,
essential genes should produce LARGE divergence (knockout breaks the
system) and non-essential genes should produce SMALL divergence.
That ranking — compared by MCC / precision / recall against Breuer
calls — is the model's contribution as a hypothesis-generation tool.

The function below is batched: all knockouts are advanced in a
single forward pass per timestep using a per-row clamping mask, so
the cost is ~one rollout per timestep regardless of how many
knockouts are tested.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def knockout_sweep(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    species_to_test: Sequence[int],
    n_rollout_steps: int = 100,
    batch_size: Optional[int] = None,
    row_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Run wild-type + per-gene knockout rollouts; return divergence.

    Args
    ----
    model : trained dynamics model in eval() mode, on a device, taking
        (B, S) signed-log1p input and returning (B, S) signed-log1p deltas.
    initial_state : (S,) tensor at t=0 (signed-log1p space), on device.
    species_to_test : list of row indices, one per gene to knock out.
        Each entry is the row index that gets clamped to zero across
        all rollout steps for that knockout's trajectory.
    n_rollout_steps : rollout horizon (timesteps from t=0).
    batch_size : if set, run knockouts in chunks of this size. None =
        all in one batch (fastest if memory permits).
    row_names : optional list[str] of length S; if given, the returned
        DataFrame includes species names alongside indices.

    Returns
    -------
    DataFrame with columns:
        species_idx : int       -- the row index clamped for this KO
        species_name : str      -- (only if row_names given) row name
        divergence : float      -- MSE between KO final state and WT final
                                   state, averaged over all S rows
        divergence_excluding_ko : float  -- same MSE but excluding the
                                           clamped row itself (which is
                                           trivially divergent at 0)
    """
    device = initial_state.device
    S = int(initial_state.shape[0])
    species_to_test = [int(i) for i in species_to_test]
    n_ko = len(species_to_test)

    # --- Wild-type rollout (once) ---
    x_wt = initial_state.clone().unsqueeze(0)            # (1, S)
    for _ in range(n_rollout_steps):
        dx = model(x_wt)
        x_wt = x_wt + dx
    wt_final = x_wt.squeeze(0)                            # (S,)

    if batch_size is None:
        batch_size = n_ko

    rows = []
    for chunk_start in range(0, n_ko, batch_size):
        chunk_end = min(chunk_start + batch_size, n_ko)
        chunk_idxs = species_to_test[chunk_start:chunk_end]
        Bc = len(chunk_idxs)

        # Initialize KO batch and build clamp mask in one pass.
        # mask[b, s] = 1 if knocking out row s in trajectory b, else 0.
        ko_batch = initial_state.unsqueeze(0).expand(Bc, -1).clone()
        clamp_mask = torch.zeros(Bc, S, device=device, dtype=ko_batch.dtype)
        for b, spec_idx in enumerate(chunk_idxs):
            ko_batch[b, spec_idx] = 0.0
            clamp_mask[b, spec_idx] = 1.0
        keep_mask = 1.0 - clamp_mask                      # (Bc, S)

        # Rollout with per-step re-clamping.
        for _ in range(n_rollout_steps):
            dx = model(ko_batch)
            ko_batch = (ko_batch + dx) * keep_mask        # zero clamped rows
        ko_finals = ko_batch                              # (Bc, S)

        # Divergence metrics
        diff = ko_finals - wt_final.unsqueeze(0)          # (Bc, S)
        diff_sq = diff.pow(2)
        divergence = diff_sq.mean(dim=1)                  # (Bc,)
        # Exclude the clamped row itself from divergence (it's at 0 by
        # construction, which is trivially "different" from WT's nonzero).
        diff_sq_masked = diff_sq * keep_mask
        n_non_clamped = keep_mask.sum(dim=1).clamp(min=1)
        divergence_excl = diff_sq_masked.sum(dim=1) / n_non_clamped

        for b, spec_idx in enumerate(chunk_idxs):
            row = {
                'species_idx': spec_idx,
                'divergence': float(divergence[b].item()),
                'divergence_excluding_ko': float(divergence_excl[b].item()),
            }
            if row_names is not None:
                row['species_name'] = row_names[spec_idx]
            rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values('divergence_excluding_ko',
                            ascending=False).reset_index(drop=True)


def gene_row_mapping_from_loci(
    row_names: Sequence[str],
    knockout_prefix: str = 'G',
    locus_re: str = r'^([A-Za-z]+)_(\d{3,})(?:_.+)?$',
) -> Dict[str, int]:
    """Map JCVISYN3A_LLLL gene IDs → row index of the gene's
    representative row (default: G_LLLL — the gene row, which controls
    transcription downstream).

    `knockout_prefix` selects which row prefix represents the gene for
    knockout purposes. Defaults to 'G' (the gene itself); could be set
    to 'R' (knock out mRNA), 'P' (knock out cytoplasmic protein), etc.

    Returns: dict {'JCVISYN3A_LLLL': row_index}. Loci without the
    requested prefix in the row list are omitted from the mapping.
    """
    import re
    pat = re.compile(locus_re)
    out: Dict[str, int] = {}
    for i, name in enumerate(row_names):
        m = pat.match(name)
        if m is None:
            continue
        prefix, locus = m.group(1), m.group(2)
        if prefix != knockout_prefix:
            continue
        gene_id = f'JCVISYN3A_{locus}'
        if gene_id not in out:
            out[gene_id] = i
    return out


def evaluate_against_essentiality(
    divergence_df: pd.DataFrame,
    essentiality_df: pd.DataFrame,
    *,
    gene_id_col: str = 'locus_tag',
    essential_col: str = 'essential',
    gene_id_for_divergence: Optional[pd.Series] = None,
    top_k: int = 100,
    divergence_col: str = 'divergence_excluding_ko',
) -> dict:
    """Compare the model's top-divergence calls against an external
    essentiality ground truth.

    Args
    ----
    divergence_df : output of knockout_sweep(). Must be index-aligned
        with `gene_id_for_divergence` (same length, same order).
    essentiality_df : DataFrame with `gene_id_col` and `essential_col`.
        `essential` should be boolean or 0/1.
    gene_id_for_divergence : Series of gene IDs matching each row of
        `divergence_df`. If None, assumes `divergence_df` already has
        a 'gene_id' column.
    top_k : take the top-k by divergence as "predicted essential."
    divergence_col : which divergence metric to rank by.

    Returns
    -------
    dict with keys:
        n_genes_evaluated
        n_essential_true
        n_essential_predicted
        MCC, precision, recall, F1
        top_k_used
    """
    df = divergence_df.copy()
    if gene_id_for_divergence is not None:
        df['gene_id'] = list(gene_id_for_divergence)
    elif 'gene_id' not in df.columns:
        raise ValueError(
            'divergence_df has no gene_id column; pass gene_id_for_divergence'
        )

    ess = essentiality_df[[gene_id_col, essential_col]].rename(columns={
        gene_id_col: 'gene_id',
        essential_col: 'essential',
    })
    merged = df.merge(ess, on='gene_id', how='inner')
    merged['essential'] = merged['essential'].astype(bool)

    # Predict top-k by divergence as essential
    n_top = min(top_k, len(merged))
    rank = merged[divergence_col].rank(method='min', ascending=False)
    merged['predicted_essential'] = rank <= n_top

    y_true = merged['essential'].astype(int).values
    y_pred = merged['predicted_essential'].astype(int).values

    # MCC, precision, recall, F1 — compute directly to avoid sklearn dep
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    denom = float(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        'n_genes_evaluated':     len(merged),
        'n_essential_true':      int(y_true.sum()),
        'n_essential_predicted': int(y_pred.sum()),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'MCC':       mcc,
        'precision': precision,
        'recall':    recall,
        'F1':        f1,
        'top_k_used':            n_top,
    }
