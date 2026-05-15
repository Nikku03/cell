"""Tier-3 prep: simulator metadata from sim_properties_1_9.pkl.

The Luthey-Schulten dataset includes a pickled metadata file:
  parameter_comparison/sim_properties_1_9.pkl (3.2 MB)

Contents are not documented in the released code. Likely candidates:
  - Per-replicate initial conditions
  - Per-replicate growth statistics
  - Cell volume / surface area trajectories
  - Random seeds used per replicate

This loader opens the pickle SAFELY (it can be malicious by design;
we use pickle.Unpickler with restricted globals where possible) and
returns the contents as a dict.

NOT a model feature - this is informational/diagnostic. Could be
used for:
  - Initial-condition validation (compare baseline rep0 init to pickled)
  - Augmented validation replicates (if extras exist beyond the 50)
  - Sanity-checking model predictions against simulator-recorded stats
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional


def load_sim_metadata(
    pkl_path: Path | str,
    *,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Open the sim_properties pickle. Returns dict-like contents on
    success, None on failure.

    Pickle loading executes Python code from the file, which is unsafe
    for files of unknown provenance. The Luthey-Schulten files are
    trusted; for arbitrary files, switch to a safer alternative.
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        if verbose:
            print(f'  WARNING: sim_properties pickle missing at {pkl_path}')
        return None

    try:
        with open(pkl_path, 'rb') as f:
            obj = pickle.load(f)
    except Exception as e:
        if verbose:
            print(f'  WARNING: failed to unpickle {pkl_path}: {e}')
        return None

    # Normalize to a dict for predictable access
    if hasattr(obj, '__dict__') and not isinstance(obj, dict):
        obj = vars(obj)

    if verbose:
        if isinstance(obj, dict):
            print(f'  sim_metadata: dict with {len(obj)} keys: '
                  f'{list(obj.keys())[:8]}{"..." if len(obj) > 8 else ""}')
        elif hasattr(obj, '__len__'):
            print(f'  sim_metadata: {type(obj).__name__} of length {len(obj)}')
        else:
            print(f'  sim_metadata: type={type(obj).__name__}')

    return obj if isinstance(obj, dict) else {'_raw': obj}


def summarize_sim_metadata(metadata: Optional[Dict[str, Any]]) -> str:
    """Return a human-readable summary of the metadata contents."""
    if metadata is None:
        return 'no metadata available'
    lines = ['=== Sim metadata ===']
    if isinstance(metadata, dict):
        for k, v in list(metadata.items())[:20]:
            if hasattr(v, 'shape'):
                lines.append(f'  {k}: array shape={tuple(v.shape)}')
            elif hasattr(v, '__len__') and not isinstance(v, str):
                lines.append(f'  {k}: {type(v).__name__} len={len(v)}')
            else:
                lines.append(f'  {k}: {v!r}'[:100])
    return '\n'.join(lines)
