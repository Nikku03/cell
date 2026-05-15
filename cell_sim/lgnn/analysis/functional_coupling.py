"""Functional coupling scanner: find pairs of species that change together.

The simulator that generated our trajectory data transcribes every gene
independently. So we shouldn't see operon-like correlated transcription
in this data. But genes can still appear coupled if they:
  - Compete for the same RNA polymerase pool
  - Share a degradation pathway (mRNA -> Degradosome)
  - Are regulated by the same protein-metabolite interaction
  - Have functionally-related dynamics for any biological reason

This module scans a trajectory for cross-species correlations and ranks
the most strongly coupled pairs. It's a starting point for "if I were
to design a wet-lab experiment, which gene pairs are worth looking at?"

NOT a proof of operonal structure - the simulator can't produce that
signal. Use it for hypothesis generation, not causal inference.

Methods supported:
  - 'pearson': linear correlation of (signed-log1p) counts over time
  - 'spearman': rank correlation (robust to monotone outliers)
  - 'hidden': cosine similarity of M7 encoder's hidden states for each
    species (requires a trained model; reveals what the model considers
    "similar")
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np


def find_coupled_species(
    trajectory: Any,
    row_names: Sequence[str],
    *,
    species_filter: Optional[Sequence[str]] = None,
    method: str = 'pearson',
    min_variance: float = 1e-4,
    top_n: int = 50,
):
    """Find top_n most-correlated pairs of species over the trajectory.

    Args
    ----
    trajectory : pd.DataFrame or numpy array of shape (n_species, n_timesteps)
    row_names : species names in trajectory order
    species_filter : if given, only consider these species (e.g., only 'R_*' rows)
    method : 'pearson' or 'spearman'
    min_variance : skip species whose row variance is below this (they're
                   essentially flat and would produce spurious "correlations")
    top_n : return this many top pairs

    Returns a pandas DataFrame with columns:
      species_a, species_b, correlation, var_a, var_b
    sorted descending by |correlation|.
    """
    import pandas as pd
    arr = trajectory.to_numpy() if hasattr(trajectory, 'to_numpy') else trajectory

    if species_filter is not None:
        keep_set = set(species_filter)
        keep_idx = [i for i, n in enumerate(row_names) if n in keep_set]
        if not keep_idx:
            raise ValueError(
                f'species_filter matched 0 rows. Sample names: '
                f'{list(row_names)[:5]}'
            )
        arr = arr[keep_idx]
        filtered_names = [row_names[i] for i in keep_idx]
    else:
        filtered_names = list(row_names)

    # Apply variance filter
    var_per_row = arr.var(axis=1)
    keep_mask = var_per_row > min_variance
    arr = arr[keep_mask]
    names = [n for n, k in zip(filtered_names, keep_mask) if k]
    if len(names) < 2:
        raise ValueError(
            f'fewer than 2 species pass min_variance={min_variance}; '
            f'lower the threshold or supply a different filter'
        )

    n = len(names)

    if method == 'pearson':
        # Use np.corrcoef for speed
        corr = np.corrcoef(arr)
    elif method == 'spearman':
        # Rank-transform each row then Pearson
        ranks = np.argsort(np.argsort(arr, axis=1), axis=1).astype(np.float64)
        corr = np.corrcoef(ranks)
    else:
        raise ValueError(f'unknown method {method!r}; use pearson or spearman')

    # Extract upper triangle (i < j), exclude diagonal
    iu = np.triu_indices(n, k=1)
    flat_i = iu[0]
    flat_j = iu[1]
    flat_c = corr[iu]
    # Sort by absolute correlation desc
    order = np.argsort(-np.abs(flat_c))[:top_n]
    rows = []
    for k in order:
        i, j = flat_i[k], flat_j[k]
        rows.append({
            'species_a': names[i],
            'species_b': names[j],
            'correlation': float(flat_c[k]),
            'var_a': float(arr[i].var()),
            'var_b': float(arr[j].var()),
        })
    return pd.DataFrame(rows)


def hidden_state_similarity(
    model,
    x_t: Any,                # (S,) or (1, S) tensor representing one timestep
    *,
    row_names: Sequence[str],
    species_filter: Optional[Sequence[str]] = None,
    top_n: int = 50,
):
    """Use a trained M7 encoder's hidden states to find species pairs the
    MODEL considers similar. Reveals biological groupings the encoder
    learned even without explicit supervision.

    Args
    ----
    model : trained M7 (CellGNNv7Hybrid or similar). Must have an encoder
            whose hidden state after `out_norm` is accessible.
    x_t   : a single-timestep state vector (any cell snapshot)

    Returns a DataFrame sorted by cosine similarity of hidden states.
    """
    import torch
    import pandas as pd
    model.eval()
    # Mirror the model's input-construction logic to avoid shape mismatch
    # if the model's n_input_channels or static-feature config is unusual.
    with torch.no_grad():
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        x_t = x_t.to(next(model.parameters()).dtype)
        # Dynamic channels
        if model.n_input_channels == 1:
            dyn = x_t.unsqueeze(-1)
        else:
            dyn = torch.stack([x_t, torch.zeros_like(x_t)], dim=-1)
        # Static channels (if model has them)
        if getattr(model, 'n_static', 0) > 0:
            static_b = (model.static_features
                        .unsqueeze(0)
                        .expand(x_t.shape[0], -1, -1)
                        .to(dyn.dtype))
            channels = torch.cat([dyn, static_b], dim=-1)
        else:
            channels = dyn
        h = model.input_proj(channels)
        for layer in model.layers:
            h, _ = layer(h, model.edge_index, model.edge_attr, model.edge_kind,
                          edge_mask=None, chunk_size=model.edge_chunk_size)
        h = model.out_norm(h)
    H = h.squeeze(0).float().cpu().numpy()    # (N, hidden)

    # Optional filter
    if species_filter is not None:
        keep_set = set(species_filter)
        idx = [i for i, n in enumerate(row_names) if n in keep_set]
        H = H[idx]
        names = [row_names[i] for i in idx]
    else:
        names = list(row_names)

    # Normalize for cosine similarity
    norms = np.linalg.norm(H, axis=1, keepdims=True) + 1e-12
    H_norm = H / norms
    sim = H_norm @ H_norm.T

    iu = np.triu_indices(len(names), k=1)
    flat_i = iu[0]; flat_j = iu[1]; flat_s = sim[iu]
    order = np.argsort(-flat_s)[:top_n]
    rows = []
    for k in order:
        i, j = flat_i[k], flat_j[k]
        rows.append({
            'species_a': names[i],
            'species_b': names[j],
            'cosine_similarity': float(flat_s[k]),
        })
    return pd.DataFrame(rows)
