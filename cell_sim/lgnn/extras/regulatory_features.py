"""Tier-3 prep: regulatory protein-metabolite binding features.

From input_data/protein_metabolites.xlsx (+ frac.csv): which proteins
bind which metabolites without consuming them (regulatory interactions
outside the main metabolic SBML).

Used as static features:
  is_regulator        : 1 if this row is a protein that binds metabolites
  is_regulated_metab  : 1 if this row is a metabolite that's regulated
  n_binding_partners  : log1p of count of regulatory partners

3 columns total. Categorical/coarse-continuous, safely below shortcut
threshold even when combined with the other extras.

Default OFF in M7TrainConfig. Activate via use_regulatory_features=True.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

REGULATORY_FEATURE_COLS = [
    'is_regulator_protein',
    'is_regulated_metabolite',
    'log_n_binding_partners',
]
_LOCUS_RE = re.compile(r'_(\d{4})(?:_|$)')


def build_regulatory_features(
    xlsx_path: Path | str,
    row_names: Sequence[str],
    *,
    frac_csv_path: Optional[Path | str] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Parse protein_metabolites.xlsx and return a (N, 3) feature tensor.

    Robust to missing file (returns zeros) and missing or weird columns
    (best-effort heuristic match).
    """
    import pandas as pd

    N = len(row_names)
    out = np.zeros((N, 3), dtype=np.float32)
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        if verbose:
            print(f'  WARNING: protein_metabolites missing at {xlsx_path}')
        return torch.from_numpy(out)

    try:
        df = None
        for sheet in (0, 'protein_metabolites', 'Sheet1'):
            try:
                df = pd.read_excel(xlsx_path, sheet_name=sheet)
                break
            except Exception:
                continue
        if df is None or len(df) == 0:
            return torch.from_numpy(out)
    except Exception as e:
        if verbose:
            print(f'  WARNING: failed to read {xlsx_path}: {e}')
        return torch.from_numpy(out)

    # Heuristic column lookup
    prot_col = None
    metab_col = None
    for col in df.columns:
        cl = str(col).lower()
        if ('prot' in cl or 'locus' in cl or 'gene' in cl) and prot_col is None:
            prot_col = col
        if ('metab' in cl or 'compound' in cl) and metab_col is None:
            metab_col = col
    if prot_col is None and len(df.columns) >= 1:
        prot_col = df.columns[0]
    if metab_col is None and len(df.columns) >= 2:
        metab_col = df.columns[1]

    if verbose:
        print(f'  protein_metabolites: prot_col={prot_col!r}  metab_col={metab_col!r}')

    # Build per-row partner counts
    protein_partners: dict = defaultdict(int)
    regulated_metab_names: set = set()
    for _, r in df.iterrows():
        prot_str = str(r.get(prot_col, ''))
        metab_str = str(r.get(metab_col, ''))
        loci_in_protein = _LOCUS_RE.findall(prot_str)
        for locus in loci_in_protein:
            protein_partners[locus] += 1
        # Metabolite name match - exact + 'M_' prefix variants
        if metab_str and metab_str != 'nan':
            regulated_metab_names.add(metab_str.strip())
            regulated_metab_names.add(f'M_{metab_str.strip()}')

    n_prot_matched = 0
    n_metab_matched = 0
    for i, name in enumerate(row_names):
        # Is this row a regulator protein?
        m = _LOCUS_RE.search(name)
        if m is not None:
            locus = m.group(1)
            if name.startswith(('P_', 'PM_')) and locus in protein_partners:
                out[i, 0] = 1.0
                out[i, 2] = float(np.log1p(protein_partners[locus]))
                n_prot_matched += 1
        # Is this row a regulated metabolite?
        if name in regulated_metab_names:
            out[i, 1] = 1.0
            n_metab_matched += 1

    if verbose:
        print(f'  regulatory features: '
              f'{n_prot_matched} regulator proteins, '
              f'{n_metab_matched} regulated metabolites')
    return torch.from_numpy(out)
