"""Stoichiometric matrix loader for the PINN output head.

The matrix S is shape (n_species, n_reactions), int32, encoding which
species are produced (+) or consumed (-) by each reaction at what
stoichiometric ratio. Loaded from any of the four MinCell_*.lm files
(they share the same Model/Reaction/StoichiometricMatrix array).

The PINN output head uses this matrix to enforce mass conservation at
the architecture level: Δx = S · v, where v is the per-reaction rate
vector predicted by the GNN.

The spatial simulator tracks 5489 species; our LGNN tracks 8572. This
loader maps the spatial S onto the LGNN's row order, padding unmapped
species (LGNN-only rows like RP_*, RPM_*, PM_*, DM_*, F_*, M_*) with
zero. Those species' counts are then updated by other mechanisms (e.g.
the central-dogma simulator edges in species_graph.py) rather than by
the stoichiometric matrix.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np
import torch


def _decode_speciesnames(raw) -> List[str]:
    """Robust decoder mirroring static_node_features._decode."""
    out = []
    for x in raw:
        v = x
        while hasattr(v, '__len__') and not isinstance(v, (bytes, str, np.bytes_)):
            if len(v) == 0:
                v = b''; break
            v = v[0]
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode('utf-8', 'replace').rstrip('\x00')
        else:
            v = str(v)
        out.append(v)
    return out


def load_stoichiometric_matrix(
    lm_path: str | Path,
    lgnn_row_names: Sequence[str],
) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
    """Load S from a MinCell_*.lm file and map onto LGNN row order.

    Args
    ----
    lm_path        : path to MinCell_<rep>.lm
    lgnn_row_names : list of LGNN species names in node order (length S_lgnn)

    Returns
    -------
    S_lgnn       : torch.FloatTensor (S_lgnn, n_reactions), with rows zero-
                    padded for species in lgnn_row_names that have no
                    spatial counterpart.
    mapped_mask  : (S_lgnn,) bool ndarray, True where the row was mapped
                    from the spatial S, False where it's a zero pad.
    spatial_names: list[str] — the original spatial species names, length 5489
    """
    lm_path = Path(lm_path)
    with h5py.File(lm_path, 'r') as f:
        raw = f['Parameters']['SpeciesNames'][:]
        spatial_names = _decode_speciesnames(raw)
        S_spatial = np.asarray(f['Model']['Reaction']['StoichiometricMatrix'])
        # shape (n_species_spatial, n_reactions), int32
    n_sp, n_rxn = S_spatial.shape
    assert n_sp == len(spatial_names), \
        f'spatial S has {n_sp} species but {len(spatial_names)} names'

    # Build lookup: spatial_name -> row index in S_spatial
    spatial_idx = {n: i for i, n in enumerate(spatial_names)}

    S_lgnn = np.zeros((len(lgnn_row_names), n_rxn), dtype=np.float32)
    mapped_mask = np.zeros(len(lgnn_row_names), dtype=bool)
    n_mapped = 0
    for i, name in enumerate(lgnn_row_names):
        if name in spatial_idx:
            S_lgnn[i] = S_spatial[spatial_idx[name]].astype(np.float32)
            mapped_mask[i] = True
            n_mapped += 1

    print(f'  spatial S: ({n_sp}, {n_rxn})')
    print(f'  LGNN rows mapped to S: {n_mapped}/{len(lgnn_row_names)} '
          f'({100*n_mapped/len(lgnn_row_names):.1f}%)')
    print(f'  LGNN rows zero-padded (no spatial counterpart): '
          f'{len(lgnn_row_names) - n_mapped}')

    return torch.from_numpy(S_lgnn), mapped_mask, spatial_names
