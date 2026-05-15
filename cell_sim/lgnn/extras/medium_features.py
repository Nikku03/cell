"""Tier-3 addition: medium / environment boundary features.

The cell exchanges matter with the external medium. The simulator's
Medium.xlsx lists boundary metabolite concentrations (glucose, amino
acids, ions, etc.). For each transport reaction, the rate depends on
the external concentration of the substrate.

Simple integration: broadcast a small medium-summary vector to every
species row. The encoder can use the global medium context when
predicting transport-related dynamics.

Even simpler approach used here: produce a (N, k) tensor where each
row gets a copy of a few global medium-summary statistics (total
external carbon, total external nitrogen, etc.). The model learns to
pick up the medium signal where transport reactions are relevant
through attention.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

MEDIUM_FEATURE_PREFIX = 'medium'


def build_medium_features(
    xlsx_path: Path | str,
    row_names: Sequence[str],
    n_features: int = 4,
    verbose: bool = True,
) -> Optional[torch.Tensor]:
    """Load Medium.xlsx and produce a (N, n_features) broadcast tensor.

    Currently produces:
      [log_total_carbon, log_total_nitrogen, log_total_amino_acids,
       log_total_other_metabolites]

    These are aggregated summary stats. Future versions could expose
    individual metabolite concentrations - but with 8572 species, even
    that would not introduce per-row identity (every row sees the
    same medium values).
    """
    import pandas as pd

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        if verbose:
            print(f'  WARNING: Medium.xlsx missing at {xlsx_path}; '
                  f'returning zero tensor')
        return torch.zeros(len(row_names), n_features, dtype=torch.float32)

    try:
        df = None
        for sheet in (0, 'Medium', 'Sheet1'):
            try:
                df = pd.read_excel(xlsx_path, sheet_name=sheet)
                break
            except Exception:
                continue
        if df is None or len(df) == 0:
            if verbose:
                print(f'  WARNING: Medium.xlsx has no readable sheet')
            return torch.zeros(len(row_names), n_features, dtype=torch.float32)
    except Exception as e:
        if verbose:
            print(f'  WARNING: failed to read {xlsx_path}: {e}')
        return torch.zeros(len(row_names), n_features, dtype=torch.float32)

    # Heuristic: find a metabolite-name column and a concentration column
    name_col = None
    conc_col = None
    for col in df.columns:
        cl = str(col).lower()
        if ('name' in cl or 'metab' in cl or 'id' in cl) and name_col is None:
            name_col = col
        if ('conc' in cl or 'mm' in cl or 'molar' in cl) and conc_col is None:
            conc_col = col
    if name_col is None:
        name_col = df.columns[0]
    if conc_col is None:
        conc_col = df.columns[-1]

    # Aggregate: log of sum-by-category
    aa_metabolites = {'ala', 'arg', 'asn', 'asp', 'cys', 'gln', 'glu', 'gly',
                      'his', 'ile', 'leu', 'lys', 'met', 'phe', 'pro', 'ser',
                      'thr', 'trp', 'tyr', 'val'}
    carbon_metabolites = {'glc', 'pyr', 'lac', 'ace', 'glcn'}
    nitrogen_metabolites = {'nh4', 'no3', 'no2', 'urea'}

    sum_aa = sum_c = sum_n = sum_other = 0.0
    for _, r in df.iterrows():
        name = str(r[name_col]).lower()
        try:
            conc = float(r[conc_col])
        except (TypeError, ValueError):
            continue
        conc = max(conc, 0.0)
        if any(aa in name for aa in aa_metabolites):
            sum_aa += conc
        elif any(c in name for c in carbon_metabolites):
            sum_c += conc
        elif any(n in name for n in nitrogen_metabolites):
            sum_n += conc
        else:
            sum_other += conc

    global_vec = np.array([
        np.log1p(sum_c), np.log1p(sum_n),
        np.log1p(sum_aa), np.log1p(sum_other),
    ], dtype=np.float32)
    # Truncate or pad to n_features
    if global_vec.shape[0] >= n_features:
        global_vec = global_vec[:n_features]
    else:
        global_vec = np.pad(global_vec, (0, n_features - global_vec.shape[0]))

    # Broadcast to (N, n_features)
    N = len(row_names)
    out = np.tile(global_vec, (N, 1)).astype(np.float32)

    if verbose:
        print(f'  medium aggregates: C={sum_c:.2f}, N={sum_n:.2f}, '
              f'AA={sum_aa:.2f}, other={sum_other:.2f}  '
              f'(broadcast as {n_features} cols x {N} rows)')

    return torch.from_numpy(out)
