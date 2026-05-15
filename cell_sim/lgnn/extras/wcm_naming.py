"""Tier-3 prep: WCM internal naming → standard biology naming map.

From analysis/WCM_naming_scheme.xlsx (also a copy in
counts_and_fluxes_analysis/). Maps simulator-internal names (often
abbreviated or non-standard) to biology nomenclature.

NOT a model feature. Used by analysis/narrate_window to produce more
readable outputs ("Pyruvate kinase" instead of "R_PYK").

Activation: pass the result of load_wcm_naming_map() to
format_narrative()... wait, that's not wired in yet. This module
provides the loader; narrate_window can be enhanced to use it later
without retraining.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def load_wcm_naming_map(
    xlsx_path: Path | str,
    *,
    verbose: bool = True,
) -> Dict[str, str]:
    """Parse WCM_naming_scheme.xlsx and return internal → standard map.

    Returns empty dict on failure.
    """
    import pandas as pd

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        if verbose:
            print(f'  WARNING: WCM_naming_scheme missing at {xlsx_path}')
        return {}

    try:
        df = None
        for sheet in (0, 'WCM_naming_scheme', 'Sheet1'):
            try:
                df = pd.read_excel(xlsx_path, sheet_name=sheet)
                break
            except Exception:
                continue
        if df is None or len(df) == 0:
            return {}
    except Exception as e:
        if verbose:
            print(f'  WARNING: failed to read {xlsx_path}: {e}')
        return {}

    # Heuristic: find internal-name column and standard-name column.
    # Two-pass to avoid double-assigning the same column when both regexes
    # would match (e.g., 'WCM_name' matches both 'wcm' and 'name').
    internal_col = None
    standard_col = None
    # Pass 1: explicit internal-side markers
    for col in df.columns:
        cl = str(col).lower()
        if internal_col is None and ('wcm' in cl or 'internal' in cl or 'sim' in cl):
            internal_col = col
    # Pass 2: standard-side markers, excluding the column already taken
    for col in df.columns:
        if col == internal_col:
            continue
        cl = str(col).lower()
        if standard_col is None and ('standard' in cl or 'official' in cl
                                       or 'biolog' in cl or 'common' in cl
                                       or 'name' in cl):
            standard_col = col
    # Fallbacks
    if internal_col is None:
        internal_col = df.columns[0]
    if standard_col is None and len(df.columns) >= 2:
        standard_col = df.columns[1] if df.columns[1] != internal_col else df.columns[-1]

    name_map = {}
    for _, r in df.iterrows():
        k = str(r[internal_col]).strip()
        v = str(r[standard_col]).strip() if standard_col else ''
        if k and v and k != 'nan' and v != 'nan':
            name_map[k] = v

    if verbose:
        print(f'  WCM naming map: {len(name_map)} entries')
    return name_map


def apply_name_map(name: str, name_map: Dict[str, str]) -> str:
    """Resolve an internal name to its standard form. Returns the original
    name if no mapping exists (graceful fallback)."""
    return name_map.get(name, name)
