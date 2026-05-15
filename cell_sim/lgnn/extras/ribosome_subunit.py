"""Tier-3 prep: ribosome large-subunit assembly features.

From input_data/LargeSubunit.xlsx: which proteins constitute the
50S large ribosomal subunit, and (if listed) their assembly order
or grouping.

Used as static features:
  is_large_subunit_protein : 1 if this row is a 50S subunit protein
  assembly_stage           : ordinal position (0..1 normalized) if known,
                              else 0

Affects ~30-40 rows (the L1..L36 large-subunit ribosomal proteins).
Highly specialized. Default OFF in M7TrainConfig.

Note: this overlaps with complex_constraints.xlsx (which includes the
full ribosome). Activate this in ADDITION to complex constraints to
get fine-grained large-subunit info; activate alone for narrower
specialization.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

RIBOSOME_SUBUNIT_COLS = [
    'is_large_subunit_protein',
    'large_subunit_assembly_stage',
]
_LOCUS_RE = re.compile(r'_(\d{4})(?:_|$)')


def build_ribosome_subunit_features(
    xlsx_path: Path | str,
    row_names: Sequence[str],
    *,
    verbose: bool = True,
) -> torch.Tensor:
    import pandas as pd

    N = len(row_names)
    out = np.zeros((N, 2), dtype=np.float32)
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        if verbose:
            print(f'  WARNING: LargeSubunit missing at {xlsx_path}')
        return torch.from_numpy(out)

    try:
        df = None
        for sheet in (0, 'LargeSubunit', 'Sheet1'):
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

    # Heuristic: find the gene/locus column and optional ordering column
    locus_col = None
    order_col = None
    for col in df.columns:
        cl = str(col).lower()
        if ('locus' in cl or 'gene' in cl) and locus_col is None:
            locus_col = col
        if ('order' in cl or 'stage' in cl or 'step' in cl) and order_col is None:
            order_col = col
    if locus_col is None:
        locus_col = df.columns[0]

    # Build locus -> stage map
    import re as _re
    stage_map: dict = {}
    for idx, r in df.iterrows():
        g_val = r.get(locus_col, '')
        # Excel sometimes converts '0001' to int 1. Normalize to a 4-digit
        # string before regex matching so all of '0001', 1, 'JCVISYN3A_0001'
        # match the same locus.
        if pd.isna(g_val):
            continue
        if isinstance(g_val, (int, np.integer, float, np.floating)):
            try:
                g_str = str(int(g_val)).zfill(4)
            except (TypeError, ValueError):
                g_str = ''
        else:
            g_str = str(g_val)
        # Pattern matching, in order of specificity
        loci = _LOCUS_RE.findall(g_str)
        if not loci:
            loci = _re.findall(r'(\d{4})', g_str)
        if not loci:
            # Try padding any short digit run
            raw = _re.findall(r'\d+', g_str)
            loci = [d.zfill(4) for d in raw if 1 <= len(d) <= 4]
        # Stage: prefer explicit column; else use row index as proxy
        if order_col is not None:
            try:
                stage = float(r[order_col])
            except (TypeError, ValueError):
                stage = float(idx)
        else:
            stage = float(idx)
        for locus in loci:
            stage_map[locus] = stage

    if not stage_map:
        if verbose:
            print(f'  ribosome_subunit: no loci parsed from LargeSubunit.xlsx')
        return torch.from_numpy(out)

    # Normalize stage to 0..1
    max_stage = max(stage_map.values())
    if max_stage > 0:
        for k in stage_map:
            stage_map[k] = stage_map[k] / max_stage

    n_matched = 0
    for i, name in enumerate(row_names):
        m = _LOCUS_RE.search(name)
        if m is None:
            continue
        locus = m.group(1)
        if locus in stage_map and name.startswith(('P_', 'PM_', 'R_', 'G_')):
            out[i, 0] = 1.0
            out[i, 1] = float(stage_map[locus])
            n_matched += 1

    if verbose:
        print(f'  ribosome subunit features: {n_matched} rows matched')
    return torch.from_numpy(out)
