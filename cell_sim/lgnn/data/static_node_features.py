"""Build per-node static feature tensor from syn3A.gb-derived gene-class
features (memory_bank/data/syn3a_gene_class_features.csv).

Each LGNN node belongs to either:
  (a) a 4-digit locus (G_LLLL_*, R_LLLL, P_LLLL, PM_LLLL, RP_LLLL_*,
      RPM_LLLL, D_LLLL, DM_LLLL) -> gets that locus's gene-class features
  (b) a cofactor / flux / metabolite (ribosomeP, RNAP, F_*, M_*, etc.)
      -> gets a zero feature vector

Returns: (N, 19) float tensor in the same row order as row_names.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

_LOCUS_RE = re.compile(r'_(\d{4})(?:_|$)')

STATIC_FEATURE_COLS = [
    # Keyword categories (14)
    'is_ribosomal', 'is_synthetase', 'is_trna', 'is_polymerase',
    'is_dna_machinery', 'is_translation', 'is_transcription',
    'is_membrane', 'is_atp_binding', 'is_kinase', 'is_phosphatase',
    'is_synthase', 'is_uncharacterized', 'is_metabolism',
    # Sequence summaries (3)
    'protein_length_aa_log', 'tm_helix_count', 'has_signal_pep',
    # Gene metadata (2)
    'has_gene_name', 'length_bp_log',
]


def build_static_node_features(
    row_names: Sequence[str],
    gene_class_csv: str | Path,
) -> torch.Tensor:
    """row_names: list of LGNN species names in node order.
    gene_class_csv: path to memory_bank/data/syn3a_gene_class_features.csv

    Returns: (N, 19) float tensor, in the same order as row_names.
    Species not associated with a gene_class locus get zero rows.
    """
    df = pd.read_csv(gene_class_csv)
    df['locus_4d'] = df['locus_tag'].str.replace('JCVISYN3A_', '', regex=False)

    # Derived columns: log-normalized length and protein_length
    df['protein_length_aa_log'] = np.log1p(
        df['protein_length_aa'].fillna(0).astype(float)
    )
    df['length_bp_log'] = np.log1p(df['length_bp'].astype(float))

    locus_to_features = df.set_index('locus_4d')[STATIC_FEATURE_COLS]

    N = len(row_names)
    F = len(STATIC_FEATURE_COLS)
    out = np.zeros((N, F), dtype=np.float32)
    matched = 0
    for i, name in enumerate(row_names):
        m = _LOCUS_RE.search(name)
        if m is None:
            continue
        locus = m.group(1)
        if locus in locus_to_features.index:
            out[i] = locus_to_features.loc[locus].values.astype(np.float32)
            matched += 1
    print(f'static node features: matched {matched}/{N} species to loci '
          f'({100*matched/N:.1f}%); '
          f'{N - matched} species (cofactors/fluxes/metabolites) get zero rows')
    return torch.from_numpy(out)


def feature_summary(features: torch.Tensor) -> pd.DataFrame:
    """Per-feature statistics for sanity checking."""
    arr = features.float().numpy()
    rows = []
    for i, name in enumerate(STATIC_FEATURE_COLS):
        col = arr[:, i]
        rows.append({
            'feature': name,
            'min':     col.min(),
            'max':     col.max(),
            'mean':    col.mean(),
            'frac_nonzero': (col != 0).mean(),
        })
    return pd.DataFrame(rows)
