"""Tier-2 addition: per-reaction kinetic priors from kinetic_params.xlsx.

For each F_<reaction> row in the LGNN, look up the published k_cat
(and K_M if available) and attach them as static features. The PINN
rate head sees these as additional input channels and can use them
as priors on the magnitude of v_log it should predict.

Non-F_* rows get zero in these columns (kinetics are reaction-level).

Two columns:
  log_kcat  : log1p of k_cat in units of 1/s. log-scaled because
              k_cat spans many orders of magnitude (10^-3 to 10^6).
  log_km    : log1p of K_M in units of M (molar). Often missing;
              filled with median when unknown.

Robust to:
  - Different column-name conventions in the xlsx
  - Reactions missing from the xlsx (left as 0, neutral signal)
  - Loading failures (returns None with a warning; the trainer
    can disable the feature instead of crashing)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

KINETIC_PRIOR_COLS = ['log_kcat', 'log_km']


def build_kinetic_prior_features(
    xlsx_path: Path | str,
    row_names: Sequence[str],
    reaction_ids: Optional[Sequence[str]] = None,
    flux_indices: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> Optional[torch.Tensor]:
    """Load kinetic_params.xlsx and produce a (N, 2) feature tensor.

    Args
    ----
    xlsx_path : path to kinetic_params.xlsx or kinetic_params_new.xlsx
    row_names : LGNN species names (so we know N and which are F_*)
    reaction_ids : optional list of reaction names matching flux_indices.
                   When provided, each row[i] in flux_indices gets the
                   k_cat for reaction_ids[i]. When None, the loader
                   tries to match by F_<rxn> name suffix.
    flux_indices : LGNN row index of each F_* species; if None, derived
                   from row_names by prefix.

    Returns torch.FloatTensor of shape (N, 2). Zero outside F_* rows.
    Returns None if loading fails (caller should fall back).
    """
    import pandas as pd

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        if verbose:
            print(f'  WARNING: kinetic_params missing at {xlsx_path}; '
                  f'returning zero tensor')
        return torch.zeros(len(row_names), 2, dtype=torch.float32)

    try:
        # The Luthey-Schulten kinetic_params.xlsx has multiple sheets.
        # Try common ones in order.
        df = None
        for sheet in (0, 'Kinetic Parameters', 'Sheet1', 'kinetic_params'):
            try:
                df = pd.read_excel(xlsx_path, sheet_name=sheet)
                break
            except Exception:
                continue
        if df is None or len(df) == 0:
            if verbose:
                print(f'  WARNING: kinetic_params at {xlsx_path} '
                      f'has no readable sheet; returning zero tensor')
            return torch.zeros(len(row_names), 2, dtype=torch.float32)
    except Exception as e:
        if verbose:
            print(f'  WARNING: failed to read {xlsx_path}: {e}')
        return torch.zeros(len(row_names), 2, dtype=torch.float32)

    # Find the reaction-id column (heuristic)
    rxn_col = None
    for col in df.columns:
        cl = str(col).lower()
        if 'rxn' in cl or 'reaction' in cl or 'rid' in cl:
            rxn_col = col
            break
    if rxn_col is None:
        rxn_col = df.columns[0]   # last-resort: first column

    # Find kcat and Km columns
    kcat_col = None
    km_col = None
    for col in df.columns:
        cl = str(col).lower().replace(' ', '').replace('_', '')
        if 'kcat' in cl and kcat_col is None:
            kcat_col = col
        if ('km' == cl[:2] or 'k_m' in cl.lower()) and km_col is None:
            km_col = col

    if verbose:
        print(f'  kinetic_params: rxn_col={rxn_col!r}  kcat_col={kcat_col!r}  '
              f'km_col={km_col!r}')

    # Build reaction-id -> (kcat, km) map
    kin_map = {}
    for _, r in df.iterrows():
        rxn_id = str(r[rxn_col]).strip()
        if not rxn_id or rxn_id == 'nan':
            continue
        try:
            kcat_val = float(r[kcat_col]) if kcat_col is not None else 0.0
        except (TypeError, ValueError):
            kcat_val = 0.0
        try:
            km_val = float(r[km_col]) if km_col is not None else 0.0
        except (TypeError, ValueError):
            km_val = 0.0
        kin_map[rxn_id] = (kcat_val, km_val)

    # Derive flux indices if not provided
    if flux_indices is None:
        flux_indices = [i for i, n in enumerate(row_names) if n.startswith('F_')]

    # Build feature tensor
    N = len(row_names)
    out = np.zeros((N, 2), dtype=np.float32)
    n_matched = 0
    for i in flux_indices:
        name = row_names[i]
        # F_<rxn_id> -> rxn_id (try several variants)
        rxn_id = name[2:] if name.startswith('F_') else name
        candidates = [rxn_id, f'R_{rxn_id}', rxn_id[:-4] if rxn_id.endswith('_end') else rxn_id]
        kin = None
        for c in candidates:
            if c in kin_map:
                kin = kin_map[c]
                break
        if kin is not None:
            kcat, km = kin
            out[i, 0] = float(np.log1p(max(kcat, 0.0)))
            out[i, 1] = float(np.log1p(max(km, 0.0)))
            n_matched += 1

    if verbose:
        print(f'  kinetic priors matched: {n_matched}/{len(flux_indices)} F_* rows')

    return torch.from_numpy(out)
