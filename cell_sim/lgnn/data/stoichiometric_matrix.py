"""Stoichiometric matrix loader for the PINN output head.

Two loaders:

  load_stoichiometric_matrix:
    Loads the raw spatial stoichiometric matrix (5489 species, 3556 reactions)
    from MinCell_*.lm and maps onto the LGNN's 8572-row species order. Returns
    a (S_lgnn, n_reactions_all) tensor including reactions that don't have
    F_* tracking.

  load_stoichiometric_matrix_for_pinn:
    Specialized for PINN training. Filters to ONLY the reactions that have
    a corresponding F_<rxn> species in the LGNN row order (so each column
    of S corresponds to one F_* node whose hidden state will encode that
    reaction's rate). Returns:
      S_pinn       : (S_lgnn, n_pinn_reactions) float tensor
      flux_indices : (n_pinn_reactions,) long tensor — index into row_names
                      pointing to each reaction's F_* species
      reaction_ids : list[str] — reaction name per column, for debugging

The spatial simulator tracks 5489 species; our LGNN tracks 8572. Species
in the LGNN but not in spatial (RP_*, RPM_*, PM_*, DM_*, F_*, M_*) get
zero-padded rows. Their counts evolve via the central-dogma simulator
edges in the GNN encoder, not via the PINN's stoichiometric update.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

# h5py is imported lazily inside the loader functions so the module is
# importable even when h5py is not installed (the PINN model can be
# instantiated with a pre-loaded matrix without needing h5py).


def _decode(raw) -> List[str]:
    """Robust decoder for HDF5 string arrays (handles dtype O / S / 2D)."""
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


def _decode_reaction_names(f) -> List[str]:
    """Reaction names live in Parameters.attrs['reactionConstNames'] as a
    comma-separated bytes string (similar to siteTypeNames).
    """
    raw = f['Parameters'].attrs.get('reactionConstNames', b'')
    if isinstance(raw, bytes):
        return raw.decode('utf-8', 'replace').split(',')
    if isinstance(raw, np.ndarray) and raw.dtype.kind == 'S':
        return [s.decode('utf-8', 'replace') for s in raw]
    return [str(x) for x in raw]


def load_stoichiometric_matrix(
    lm_path: str | Path,
    lgnn_row_names: Sequence[str],
) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
    """Load full (S_lgnn, n_reactions_all) matrix.

    Returns
    -------
    S_lgnn       : torch.FloatTensor (S_lgnn, n_reactions), zero-padded for
                    species in lgnn_row_names that have no spatial counterpart
    mapped_mask  : (S_lgnn,) bool ndarray, True where mapped from spatial
    spatial_names: list[str] of length 5489
    """
    import h5py
    lm_path = Path(lm_path)
    with h5py.File(lm_path, 'r') as f:
        raw = f['Parameters']['SpeciesNames'][:]
        spatial_names = _decode(raw)
        S_spatial = np.asarray(f['Model']['Reaction']['StoichiometricMatrix'])
    n_sp, n_rxn = S_spatial.shape
    assert n_sp == len(spatial_names), \
        f'spatial S has {n_sp} species but {len(spatial_names)} names'

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
    print(f'  LGNN rows mapped: {n_mapped}/{len(lgnn_row_names)} '
          f'({100*n_mapped/len(lgnn_row_names):.1f}%)')
    return torch.from_numpy(S_lgnn), mapped_mask, spatial_names


def load_stoichiometric_matrix_for_pinn(
    lm_path: str | Path,
    lgnn_row_names: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor, List[str], np.ndarray]:
    """Filter the stoichiometric matrix to ONLY reactions with F_* tracking.

    For each F_<rxn> species in lgnn_row_names, find the matching column in
    the full spatial S. The returned S_pinn has columns aligned with the F_*
    species in the LGNN, so PINNHead can pull hidden state at F_* indices
    and use it as the rate vector v.

    Returns
    -------
    S_pinn       : torch.FloatTensor (S_lgnn, n_pinn_reactions)
                    Rows in LGNN order, columns aligned with flux_indices.
                    Rows for species not in spatial are zero-padded.
                    Rows for F_* species (which are reaction-rate trackers,
                    not species) are also zero — the PINN doesn't update them
                    via stoichiometry.
    flux_indices : torch.LongTensor (n_pinn_reactions,)
                    Index in lgnn_row_names of each F_* species, in column order
    reaction_ids : list[str] — reaction name per column (for debug/log)
    mapped_mask  : (S_lgnn,) bool ndarray, True where the row is mapped
                    from spatial S (i.e., this species participates in the
                    PINN-handled reactions)
    """
    import h5py
    lm_path = Path(lm_path)
    with h5py.File(lm_path, 'r') as f:
        spatial_names = _decode(f['Parameters']['SpeciesNames'][:])
        S_spatial = np.asarray(f['Model']['Reaction']['StoichiometricMatrix'])
        reaction_names_all = _decode_reaction_names(f)
    n_sp, n_rxn = S_spatial.shape
    if len(reaction_names_all) != n_rxn:
        print(f'WARNING: reactionConstNames has {len(reaction_names_all)} '
              f'entries but S has {n_rxn} columns. Trying to proceed by '
              f'index alone.')

    # Build lookup: reaction_name -> column index in S_spatial
    rxn_idx_by_name = {n: i for i, n in enumerate(reaction_names_all)}
    spatial_idx_by_name = {n: i for i, n in enumerate(spatial_names)}

    # For each F_<rxn> species in LGNN, find its reaction column
    pinn_columns: List[int] = []      # which spatial S columns correspond to F_*
    flux_indices: List[int] = []       # which LGNN rows are the F_* species
    reaction_ids: List[str] = []       # reaction name (debug)
    missing_rxns: List[str] = []
    for i, name in enumerate(lgnn_row_names):
        if not name.startswith('F_'):
            continue
        rxn_id = name[2:]   # strip 'F_' prefix
        # The .lm reactionConstNames typically carries the SBML 'R_' prefix
        # (e.g. R_PGI) while the LGNN F_* species drops it (F_PGI). Try the
        # raw form, with R_ added, and both with/without an '_end' suffix
        # (period-averaged vs end-of-step bookkeeping).
        base = rxn_id[:-4] if rxn_id.endswith('_end') else rxn_id
        candidates: List[str] = []
        for prefix in ('', 'R_'):
            for suffix in ('', '_end'):
                c = f'{prefix}{base}{suffix}'
                if c not in candidates:
                    candidates.append(c)
        col = None
        for c in candidates:
            if c in rxn_idx_by_name:
                col = rxn_idx_by_name[c]
                break
        if col is None:
            missing_rxns.append(name)
            continue
        pinn_columns.append(col)
        flux_indices.append(i)
        reaction_ids.append(rxn_id)

    n_pinn_rxn = len(pinn_columns)
    print(f'  spatial S: ({n_sp}, {n_rxn})')
    print(f'  F_* species in LGNN: {len([n for n in lgnn_row_names if n.startswith("F_")])}')
    print(f'  PINN-mapped reactions: {n_pinn_rxn}')
    print(f'  unmappable F_* species: {len(missing_rxns)} '
          f'(first few: {missing_rxns[:5]})')

    # Build S_pinn: (n_lgnn, n_pinn_rxn). For each LGNN species row, look up
    # its spatial counterpart and copy only the pinn_columns.
    S_pinn = np.zeros((len(lgnn_row_names), n_pinn_rxn), dtype=np.float32)
    mapped_mask = np.zeros(len(lgnn_row_names), dtype=bool)
    n_species_with_rows = 0
    pinn_col_array = np.array(pinn_columns, dtype=np.int64)
    for i, name in enumerate(lgnn_row_names):
        if name in spatial_idx_by_name:
            sp_row = spatial_idx_by_name[name]
            S_pinn[i] = S_spatial[sp_row, pinn_col_array].astype(np.float32)
            mapped_mask[i] = True
            n_species_with_rows += 1
    print(f'  LGNN rows with non-zero S: {int((S_pinn != 0).any(axis=1).sum())}')
    print(f'  LGNN rows mapped from spatial: {n_species_with_rows}')

    # F_* rows themselves should be zero (they're reaction trackers, not species
    # that get updated by the PINN's mass-balance step)
    for fi in flux_indices:
        S_pinn[fi] = 0.0

    return (
        torch.from_numpy(S_pinn),
        torch.tensor(flux_indices, dtype=torch.long),
        reaction_ids,
        mapped_mask,
    )
