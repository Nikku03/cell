"""Decompose species counts by subcellular location.

The trajectory data records aggregate counts ("there are 4,718 P_0612
molecules") but no information about where they are. The lm-derived
spatial parquet (from step B) gives us each species' average
distribution across cellular regions:
  - cytoplasm
  - membrane
  - DNA region
  - ribosomes
  - etc.

This module takes a count and breaks it down by location using the
spatial fingerprint. NOT exact at the per-molecule level - it's a
statistical decomposition that says "if P_0612 lives 78% at the
membrane on average, then approximately 0.78 × 4718 = 3680 of the
4718 molecules are membrane-bound."

Useful for:
  - Region-aware analysis ("how much ATP is in the cytoplasm vs DNA region")
  - Sanity checking ("did this membrane protein actually localize to the membrane")
  - Producing spatial heatmaps from non-spatial outputs
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


_LOCUS_RE = re.compile(r'_(\d{4})(?:_|$)')


def decompose_counts_by_location(
    counts: Dict[str, float],
    spatial_df,
    *,
    location_cols: Optional[Sequence[str]] = None,
):
    """Break aggregate species counts into per-location estimates.

    Args
    ----
    counts : {species_name: count_value}
    spatial_df : DataFrame indexed by locus (4-digit string) with per-locus
                 spatial fraction columns (e.g. 'protein_frac_in_cytoplasm').
                 The output of step B's extractor.
    location_cols : list of column names in spatial_df to decompose into.
                    If None, infers from spatial_df columns matching
                    '*_frac_in_*'.

    Returns DataFrame:
        species, count, locus, location_1, location_2, ...
    """
    import pandas as pd

    if location_cols is None:
        location_cols = [c for c in spatial_df.columns if '_frac_in_' in c]

    rows = []
    for species_name, count_val in counts.items():
        m = _LOCUS_RE.search(species_name)
        locus = m.group(1) if m else None
        # Pick the right family of frac columns based on species name prefix
        prefix = _species_prefix(species_name)
        prefix_cols = [c for c in location_cols if c.startswith(prefix + '_')]
        row = {
            'species': species_name,
            'count': float(count_val),
            'locus': locus,
            'prefix': prefix,
        }
        if locus is None or locus not in spatial_df.index:
            # No spatial info; leave fractions as NaN
            for c in prefix_cols:
                row[c] = float('nan')
        else:
            sp_row = spatial_df.loc[locus]
            for c in prefix_cols:
                frac = sp_row.get(c, float('nan'))
                if pd.notna(frac):
                    row[c] = float(frac * count_val)
                else:
                    row[c] = float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


def _species_prefix(name: str) -> str:
    """Pick the spatial-features prefix family for a species name.

    Maps row name prefixes to the spatial-extractor's categories
    (gene / rna / protein / membrane / translation).
    """
    if name.startswith('G_'):  return 'gene'
    if name.startswith('R_'):  return 'rna'
    if name.startswith('P_'):  return 'protein'
    if name.startswith('PM_'): return 'membrane'
    if name.startswith('RP_') or name.startswith('RPM_'):
        return 'translation'
    return 'protein'      # default fallback
