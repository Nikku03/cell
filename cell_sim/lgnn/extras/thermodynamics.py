"""Tier-2 prep: thermodynamic (Gibbs ΔG) features per reaction.

From the eQuilibrator notebook output. The notebook itself produces a
table of (reaction_id, dG_prime_in_kJ_per_mol, K_eq) values - this
module expects that table as a CSV file. If unavailable, fall back
to zeros.

Used as static features on F_<reaction> rows:
  log_K_eq    : log(K_eq) where K_eq = exp(-ΔG°/RT) at T=310 K
  dG_sign     : sign of ΔG (+1 forward-favorable, -1 reverse-favorable, 0)

Effects when wired in (default OFF):
  - The PINN rate head receives a thermodynamic prior per reaction.
  - Forward-favorable reactions (dG_sign = +1) get a positive log_K_eq;
    the model learns to bias v_log positive for these.
  - Reverse-favorable reactions get the opposite signal.

For full thermodynamic *constraint* (not just prior), a separate loss
term would penalize predicted v_log against the K_eq. That's a future
addition; this module just supplies the features.

To produce the CSV input from eQuilibrator:
  cd analysis/GIP_Gibbs_Concs_Fluxes
  jupyter nbconvert --execute free_energy_comparison_eQuilibrator.ipynb
  # then export the final results dataframe to gibbs.csv with columns
  # 'reaction_id' and 'dG_prime_kJ_per_mol'.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

THERMO_FEATURE_COLS = ['log_k_eq', 'dG_sign']

# Physical constants
_R_GAS = 8.314e-3   # kJ/(mol·K)
_T_K   = 310.0      # body-temp-ish for mycoplasma


def build_thermodynamics_features(
    gibbs_csv_path: Path | str,
    row_names: Sequence[str],
    *,
    verbose: bool = True,
) -> torch.Tensor:
    """Parse a Gibbs CSV (reaction_id, dG_prime_kJ_per_mol) and return (N, 2).

    Non-F_* rows: zero. F_* rows: log(K_eq) and sign(dG) for the matching
    reaction.
    """
    import pandas as pd

    N = len(row_names)
    out = np.zeros((N, 2), dtype=np.float32)
    gibbs_csv_path = Path(gibbs_csv_path)
    if not gibbs_csv_path.exists():
        if verbose:
            print(f'  WARNING: gibbs CSV missing at {gibbs_csv_path}; '
                  f'extracted via eQuilibrator notebook is needed before '
                  f'this feature group can be activated')
        return torch.from_numpy(out)

    try:
        df = pd.read_csv(gibbs_csv_path)
    except Exception as e:
        if verbose:
            print(f'  WARNING: failed to read {gibbs_csv_path}: {e}')
        return torch.from_numpy(out)

    # Heuristic column lookup
    rxn_col = None
    dg_col = None
    for col in df.columns:
        cl = str(col).lower()
        if ('rxn' in cl or 'reaction' in cl) and rxn_col is None:
            rxn_col = col
        if ('dg' in cl or 'gibbs' in cl or 'kj' in cl) and dg_col is None:
            dg_col = col
    if rxn_col is None or dg_col is None:
        if verbose:
            print(f'  WARNING: Gibbs CSV must have reaction_id and dG columns')
        return torch.from_numpy(out)

    # Build reaction -> (log_keq, sign)
    # Clip log_K_eq to ±50 so that polymerization/aggregation reactions in
    # the SBML (which can sum to ΔG of -1000s of kJ/mol from multiple bond
    # formations) don't produce inf via exp() overflow. log_K=±50
    # corresponds to K_eq spanning ~22 orders of magnitude in either
    # direction - still expressive, but bounded.
    LOG_K_CLIP = 50.0
    rxn_data = {}
    n_clipped = 0
    for _, r in df.iterrows():
        rxn_id = str(r[rxn_col]).strip()
        try:
            dG = float(r[dg_col])
        except (TypeError, ValueError):
            continue
        # Compute log(K_eq) directly from -dG/RT to avoid the exp/log
        # roundtrip that overflows for extreme dG values.
        log_K_raw = -dG / (_R_GAS * _T_K)
        log_K = float(np.clip(log_K_raw, -LOG_K_CLIP, LOG_K_CLIP))
        if abs(log_K_raw) > LOG_K_CLIP:
            n_clipped += 1
        sign = float(np.sign(-dG))
        rxn_data[rxn_id] = (log_K, sign)
    if verbose and n_clipped > 0:
        print(f'  thermodynamics: clipped {n_clipped} extreme log_K values '
              f'to ±{LOG_K_CLIP} (likely polymerization-type aggregations)')

    # Match to F_* rows
    flux_indices = [i for i, n in enumerate(row_names) if n.startswith('F_')]
    n_matched = 0
    for i in flux_indices:
        name = row_names[i]
        rxn_id = name[2:]   # strip F_
        candidates = [rxn_id, f'R_{rxn_id}',
                      rxn_id[:-4] if rxn_id.endswith('_end') else rxn_id]
        for c in candidates:
            if c in rxn_data:
                log_K, sgn = rxn_data[c]
                out[i, 0] = log_K
                out[i, 1] = sgn
                n_matched += 1
                break

    if verbose:
        print(f'  thermodynamics features: {n_matched}/{len(flux_indices)} '
              f'F_* rows matched')
    return torch.from_numpy(out)
