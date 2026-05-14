"""Align Breuer essentiality labels with the LGNN's row order.

Two helpers:

  extract_gene_row_indices(row_names) -> (gene_indices, gene_loci)
    For each `G_<locus>` row in row_names, return its index and locus tag.
    Gene_indices is a torch.LongTensor; gene_loci is a list[str] of
    'JCVISYN3A_<locus>' parallel to it.

  build_breuer_label_tensor(gene_loci, breuer_df, train_loci=None,
                              val_loci=None) -> (labels, train_mask, val_mask)
    `labels`: float tensor of shape (n_genes,) — 1.0 for essential, 0.0
              for non-essential, NaN where the gene isn't in Breuer.
    `train_mask`, `val_mask`: bool tensors of shape (n_genes,) — which
              indices to use for training vs validation. The split is
              done on locus IDs (not row indices) so genes don't leak.
              If train_loci/val_loci are not provided, all labelled
              genes go in train_mask, val_mask is all-False.
"""
from __future__ import annotations
import re
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

_G_LOCUS_RE = re.compile(r'^G_(\d{4})$')


def extract_gene_row_indices(
    row_names: Sequence[str],
) -> Tuple[torch.Tensor, List[str]]:
    """Return (gene_indices, gene_loci) for every G_<locus> row, in order."""
    indices: List[int] = []
    loci: List[str] = []
    for i, name in enumerate(row_names):
        m = _G_LOCUS_RE.match(name)
        if m is None:
            continue
        indices.append(i)
        loci.append(f'JCVISYN3A_{m.group(1)}')
    return torch.tensor(indices, dtype=torch.long), loci


def build_breuer_label_tensor(
    gene_loci: Sequence[str],
    breuer_df: pd.DataFrame,
    train_loci: Optional[Set[str]] = None,
    val_loci: Optional[Set[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align Breuer labels to gene_loci, return (labels, train_mask, val_mask).

    breuer_df must have 'locus_tag' and 'essential' columns (output of
    load_breuer_labels()).
    """
    n = len(gene_loci)
    labels = torch.full((n,), float('nan'), dtype=torch.float32)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    by_locus = {row['locus_tag']: bool(row['essential'])
                  for _, row in breuer_df.iterrows()}
    for i, lt in enumerate(gene_loci):
        if lt in by_locus:
            labels[i] = 1.0 if by_locus[lt] else 0.0
            if val_loci is not None and lt in val_loci:
                val_mask[i] = True
            elif train_loci is not None:
                if lt in train_loci:
                    train_mask[i] = True
            else:
                train_mask[i] = True
    return labels, train_mask, val_mask


def gene_kfold_split(
    gene_loci: Sequence[str],
    breuer_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 0,
) -> List[Tuple[Set[str], Set[str]]]:
    """Stratified k-fold by essentiality on the labelled subset.

    Returns a list of (train_loci_set, val_loci_set) for each fold.
    Genes not in breuer_df are excluded from both sets entirely.
    """
    by_locus = {row['locus_tag']: bool(row['essential'])
                  for _, row in breuer_df.iterrows()}
    labelled = [lt for lt in gene_loci if lt in by_locus]
    rng = np.random.RandomState(seed)
    # Stratified shuffle
    ess  = sorted([lt for lt in labelled if by_locus[lt]])
    ness = sorted([lt for lt in labelled if not by_locus[lt]])
    rng.shuffle(ess); rng.shuffle(ness)
    folds: List[Tuple[Set[str], Set[str]]] = []
    for k in range(n_splits):
        val_ess  = ess [k::n_splits]
        val_ness = ness[k::n_splits]
        val = set(val_ess) | set(val_ness)
        train = set(labelled) - val
        folds.append((train, val))
    return folds
