"""Tier-3 addition: complex membership features from complex_formation.xlsx.

The Luthey-Schulten dataset lists ~24 multi-protein complexes (ribosomes,
RNA polymerase, etc.) with their constituent gene loci. We attach a
one-hot "is part of complex N" indicator to each species row that
belongs to a known complex.

The benefit: the encoder learns "these 50 species all belong to the
ribosome complex" - a strong categorical hint that they should be
predicted as a coordinated group, not 50 independent species.

Failure mode to avoid (M5 trap): if there are too many complex
indicators, the model can use them to row-identify. With ~24 complexes,
each per-row vector has at most 1 hot bit out of 24 - well below
shortcut-learning threshold.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

COMPLEX_FEATURE_PREFIX = 'in_complex'
_LOCUS_RE = re.compile(r'_(\d{4})(?:_|$)')


def build_complex_membership_features(
    xlsx_path: Path | str,
    row_names: Sequence[str],
    verbose: bool = True,
    max_complexes: int = 32,
) -> Optional[torch.Tensor]:
    """Load complex_formation.xlsx and return a (N, n_complexes) tensor.

    Each column j is 1.0 for rows whose locus is listed as part of
    complex j. Most rows will have zero hot bits (they're not in any
    known complex); rows in a complex have exactly one hot bit.

    Args
    ----
    xlsx_path : path to complex_formation.xlsx
    row_names : LGNN species names
    max_complexes : cap to limit feature dimensionality (and shortcut risk)
    """
    import pandas as pd

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        if verbose:
            print(f'  WARNING: complex_formation missing at {xlsx_path}; '
                  f'returning zero tensor')
        return torch.zeros(len(row_names), 1, dtype=torch.float32)

    try:
        df = None
        for sheet in (0, 'complex_formation', 'Sheet1', 'Complex Formation'):
            try:
                df = pd.read_excel(xlsx_path, sheet_name=sheet)
                break
            except Exception:
                continue
        if df is None or len(df) == 0:
            if verbose:
                print(f'  WARNING: complex_formation has no readable sheet; '
                      f'returning zero tensor')
            return torch.zeros(len(row_names), 1, dtype=torch.float32)
    except Exception as e:
        if verbose:
            print(f'  WARNING: failed to read {xlsx_path}: {e}')
        return torch.zeros(len(row_names), 1, dtype=torch.float32)

    # Try to identify complex-id and gene-locus columns
    complex_col = None
    gene_col = None
    for col in df.columns:
        cl = str(col).lower()
        if ('complex' in cl) and complex_col is None and 'name' in cl:
            complex_col = col
        if ('complex' in cl) and complex_col is None and 'id' in cl:
            complex_col = col
        if ('locus' in cl or 'gene' in cl) and gene_col is None:
            gene_col = col

    # Fallback: first two columns
    if complex_col is None:
        complex_col = df.columns[0]
    if gene_col is None:
        gene_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    if verbose:
        print(f'  complex_formation: complex_col={complex_col!r}  '
              f'gene_col={gene_col!r}')

    # Build complex -> [loci] map
    complex_loci = {}
    for _, r in df.iterrows():
        c_id = str(r[complex_col]).strip()
        g_str = str(r[gene_col])
        if not c_id or c_id == 'nan':
            continue
        # gene cell may contain comma/semicolon-separated list of locus tags
        loci = re.findall(r'(\d{4})', g_str)
        if not loci:
            continue
        complex_loci.setdefault(c_id, []).extend(loci)

    # Limit complex count
    sorted_complexes = sorted(complex_loci.keys())[:max_complexes]
    n_c = len(sorted_complexes)
    if n_c == 0:
        if verbose:
            print(f'  no complexes parsed; returning zero tensor')
        return torch.zeros(len(row_names), 1, dtype=torch.float32)

    # Build feature tensor: (N, n_complexes)
    N = len(row_names)
    out = np.zeros((N, n_c), dtype=np.float32)
    for j, c_name in enumerate(sorted_complexes):
        loci_set = set(complex_loci[c_name])
        for i, name in enumerate(row_names):
            m = _LOCUS_RE.search(name)
            if m is None:
                continue
            if m.group(1) in loci_set:
                out[i, j] = 1.0

    n_rows_in_any = int((out.sum(axis=1) > 0).sum())
    if verbose:
        print(f'  complex features: {n_c} complexes x {N} rows  '
              f'({n_rows_in_any} rows have at least one complex membership)')

    return torch.from_numpy(out)
