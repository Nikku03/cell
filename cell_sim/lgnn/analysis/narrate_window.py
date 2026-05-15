"""narrate_window: explain what happened in a 1-second cell window.

No model. No GPU. No training. Just data analysis on the trajectory.

Given a (replicate, t) pair, this function:

  1. Reads the state at time t and time t+1 from the trajectory.
  2. Computes the per-species change dx = x[t+1] - x[t].
  3. From the F_* rows (cumulative flux counters), reads how many times
     each reaction fired in the 1-second window directly. No inference -
     this number is in the data.
  4. For every non-F_* row that changed, looks up which reactions in the
     SBML model can produce/consume that species and attributes the
     observed delta to them via stoichiometry × n_fires.
  5. Optionally enriches the result with gene-product context from the
     GenBank annotations (syn3A.gb) and per-locus spatial summaries
     from the lm-derived parquet.

The output is a structured dict (machine-readable) plus a formatter that
produces a human-readable narrative.

Sanity column: each species row gets a `residual = change - net_attributed`.
A residual of zero means the SBML reactions fully account for the
observed change. Non-zero residual is informative - usually it means
either (a) a non-SBML reaction (translation, transcription, degradation
- handled by the simulator template, not the metabolic model) or (b) a
cumulative counter (PM_/RPM_/DM_) whose stoichiometry is unusual.

Cache helpers: load_genbank() and load_stoich() are exposed so callers
making thousands of narrate_window calls don't reparse the same SBML
matrix and GenBank file every time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

_LOCUS_RE = re.compile(r'_(\d{4})(?:_|$)')


# ----------------------------------------------------------------------
# Data class
# ----------------------------------------------------------------------

@dataclass
class NarrationResult:
    """One window's narrative result.

    Attributes
    ----------
    replicate : int
    t_start, t_end : int
        Window covers [t_start, t_end). For window_size=1, t_end = t_start + 1.
    reaction_events : Dict[str, dict]
        Reactions that fired in the window. Keys are reaction ids (e.g.
        'R_PYK'); values include n_fires (signed; usually positive, can
        be negative when a counter drops), flux_row_idx (which row in
        the LGNN species list holds this reaction's F_* counter), and
        S_col_idx (which column in the stoichiometric matrix).
    species_changes : Dict[str, dict]
        Non-F_* species whose count changed in the window. Values include
        change, x_t, x_t1, produced_by (list of reaction contributions
        with positive stoichiometry), consumed_by (negative stoichiometry),
        net_attributed (sum of contributions), residual (change - net,
        zero when fully explained), and optionally gene_context and
        spatial.
    """
    replicate: int
    t_start: int
    t_end: int
    reaction_events: Dict[str, dict] = field(default_factory=dict)
    species_changes: Dict[str, dict] = field(default_factory=dict)

    def n_reactions_fired(self) -> int:
        return len(self.reaction_events)

    def n_species_changed(self) -> int:
        return len(self.species_changes)

    def total_event_count(self) -> int:
        return sum(abs(e['n_fires']) for e in self.reaction_events.values())


# ----------------------------------------------------------------------
# Cache helpers
# ----------------------------------------------------------------------

def load_genbank(gb_path: Union[str, Path]) -> Dict[str, dict]:
    """Parse syn3A.gb once; return locus_tag -> {gene, product, function, ec}.

    Wraps `cell_sim.layer0_genome.syn3a_real.parse_genbank` for use by
    narrate_window. Returns an empty dict if BioPython isn't installed,
    so the rest of the tool still works without gene context.
    """
    try:
        from cell_sim.layer0_genome.syn3a_real import parse_genbank
    except ImportError:
        return {}
    try:
        genes, _ = parse_genbank(Path(gb_path))
        return genes
    except Exception as e:
        print(f'WARNING: GenBank parse failed ({e}); gene context disabled')
        return {}


def load_stoich(sbml_path: Union[str, Path],
                row_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the PINN stoichiometric matrix once and return numpy arrays.

    Returns
    -------
    S : np.ndarray of shape (n_species_lgnn, n_pinn_reactions), fp32
    flux_indices : np.ndarray of shape (n_pinn_reactions,), int64.
        flux_indices[c] = row index in `row_names` of the F_* species
        for the c-th PINN reaction.
    reaction_ids : list[str], length n_pinn_reactions
    """
    # Local import to avoid pulling torch into pure-analysis callers
    # that don't need it - but the loader does require torch (it returns
    # tensors), so we convert at the boundary.
    from cell_sim.lgnn.data.stoichiometric_matrix import (
        load_stoichiometric_matrix_for_pinn,
    )
    S_t, flux_idx_t, reaction_ids, _mask = load_stoichiometric_matrix_for_pinn(
        Path(sbml_path), row_names,
    )
    return (
        S_t.cpu().numpy().astype(np.float32),
        flux_idx_t.cpu().numpy().astype(np.int64),
        list(reaction_ids),
    )


# ----------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------

def narrate_window(
    replicate: int,
    t: int,
    *,
    # Data sources - pass either the raw module + paths, or pre-loaded objects
    lsdata=None,
    sbml_path: Optional[Union[str, Path]] = None,
    gb_path: Optional[Union[str, Path]] = None,
    spatial_parquet: Optional[Union[str, Path]] = None,
    # Pre-loaded objects (use these for repeated calls to avoid re-parsing)
    trajectory: Optional[Any] = None,  # pd.DataFrame from lsdata.load_replicate
    row_names: Optional[List[str]] = None,
    stoich: Optional[Tuple[np.ndarray, np.ndarray, List[str]]] = None,
    genes: Optional[Dict[str, dict]] = None,
    spatial_df: Optional[Any] = None,
    # Behaviour
    window_size: int = 1,
) -> NarrationResult:
    """Explain what happened in `[t, t+window_size)` for the given replicate.

    Required arguments: either (lsdata + sbml_path) for the slow path, or
    pre-loaded (trajectory + stoich + row_names) for the cached path.
    Optional gene/spatial enrichment passes through if provided.

    Returns a NarrationResult.
    """
    # ---- 1. Trajectory ----
    if trajectory is None:
        if lsdata is None:
            raise ValueError('either trajectory or lsdata must be provided')
        trajectory = lsdata.load_replicate(replicate)
    # trajectory has shape (n_species, n_timesteps), index = species names
    if row_names is None:
        row_names = list(trajectory.index)
    n_t = trajectory.shape[1]
    if t < 0 or t + window_size >= n_t:
        raise ValueError(
            f't={t} window_size={window_size} out of range '
            f'(replicate has {n_t} timesteps)'
        )

    x_t  = trajectory.iloc[:, t].to_numpy()
    x_t1 = trajectory.iloc[:, t + window_size].to_numpy()
    delta = (x_t1 - x_t).astype(np.float64)

    # ---- 2. Stoichiometric matrix ----
    if stoich is None:
        if sbml_path is None:
            raise ValueError('either stoich or sbml_path must be provided')
        stoich = load_stoich(sbml_path, row_names)
    S, flux_indices, reaction_ids = stoich

    # ---- 3. Reactions that fired (from F_* row deltas) ----
    flux_idx_set = set(int(i) for i in flux_indices.tolist())
    reaction_events: Dict[str, dict] = {}
    for col_idx, (flux_row, rxn_id) in enumerate(
        zip(flux_indices.tolist(), reaction_ids)
    ):
        flux_row = int(flux_row)
        n_fires = float(delta[flux_row])
        if abs(n_fires) > 0.5:    # any fractional drift gets ignored
            reaction_events[rxn_id] = {
                'n_fires': int(round(n_fires)),
                'flux_row_idx': flux_row,
                'S_col_idx': col_idx,
                'flux_row_name': row_names[flux_row],
            }

    # ---- 4. Non-flux species that changed ----
    species_changes: Dict[str, dict] = {}
    for i, change in enumerate(delta):
        if i in flux_idx_set or abs(change) < 0.5:
            continue
        species_changes[row_names[i]] = {
            'change': float(change),
            'row_idx': i,
            'x_t': float(x_t[i]),
            'x_t1': float(x_t1[i]),
        }

    # ---- 5. Attribute species changes to reactions ----
    for sp_name, info in species_changes.items():
        sp_idx = info['row_idx']
        produced_by: List[dict] = []
        consumed_by: List[dict] = []
        net_attributed = 0.0
        # Walk only reactions that fired (others contribute zero)
        for rxn_id, evt in reaction_events.items():
            col = evt['S_col_idx']
            stoich_coef = float(S[sp_idx, col])
            if stoich_coef == 0.0:
                continue
            n_fires = evt['n_fires']
            contribution = stoich_coef * n_fires
            entry = {
                'reaction': rxn_id,
                'stoich_coef': stoich_coef,
                'n_fires': n_fires,
                'contribution': contribution,
            }
            if stoich_coef > 0:
                produced_by.append(entry)
            else:
                consumed_by.append(entry)
            net_attributed += contribution
        info['produced_by'] = produced_by
        info['consumed_by'] = consumed_by
        info['net_attributed'] = net_attributed
        info['residual'] = info['change'] - net_attributed

    # ---- 6. Gene-product enrichment ----
    if genes is None and gb_path is not None:
        genes = load_genbank(gb_path)
    if genes:
        for sp_name, info in species_changes.items():
            locus = _extract_locus(sp_name)
            if locus is None:
                continue
            gene_locus = f'JCVISYN3A_{locus}'
            if gene_locus in genes:
                gi = genes[gene_locus]
                info['gene_context'] = {
                    'locus_tag': gene_locus,
                    'gene_name': gi.get('gene', '') or '',
                    'product':   gi.get('product', '') or '',
                    'function':  gi.get('function', '') or '',
                    'ec_number': gi.get('ec', '') or '',
                }

    # ---- 7. Spatial enrichment ----
    if spatial_df is None and spatial_parquet is not None:
        import pandas as pd  # local import
        spatial_df = pd.read_parquet(spatial_parquet)
        # Use 'locus' as index if it's a column
        if 'locus' in spatial_df.columns:
            spatial_df = spatial_df.set_index('locus')
    if spatial_df is not None:
        for sp_name, info in species_changes.items():
            locus = _extract_locus(sp_name)
            if locus is None:
                continue
            if locus in spatial_df.index:
                info['spatial'] = {
                    k: float(v) if isinstance(v, (int, float, np.floating, np.integer))
                       else v
                    for k, v in spatial_df.loc[locus].to_dict().items()
                }

    return NarrationResult(
        replicate=int(replicate),
        t_start=int(t),
        t_end=int(t + window_size),
        reaction_events=reaction_events,
        species_changes=species_changes,
    )


# ----------------------------------------------------------------------
# Formatter
# ----------------------------------------------------------------------

def format_narrative(
    result: NarrationResult,
    top_n_reactions: int = 15,
    top_n_species: int = 20,
    show_residual: bool = True,
) -> str:
    """Render a NarrationResult as a readable string."""
    out: List[str] = []
    out.append(
        f'=== Cell narrative: rep {result.replicate} '
        f'second {result.t_start} -> {result.t_end} '
        f'({result.t_end - result.t_start}s window) ==='
    )

    # Reactions
    rxns = result.reaction_events
    n_rxn = len(rxns)
    total_events = result.total_event_count()
    out.append(
        f'\n{n_rxn} reaction(s) fired ({total_events:,} total events) '
        f'- top {min(top_n_reactions, n_rxn)} by event count:'
    )
    sorted_rxns = sorted(rxns.items(),
                         key=lambda kv: abs(kv[1]['n_fires']),
                         reverse=True)
    for rxn_id, evt in sorted_rxns[:top_n_reactions]:
        sign = '+' if evt['n_fires'] >= 0 else ''
        out.append(f'  {rxn_id:32s}  {sign}{evt["n_fires"]:>6,d} events')
    if n_rxn > top_n_reactions:
        out.append(f'  ...and {n_rxn - top_n_reactions} more reactions')

    # Species changes
    changes = result.species_changes
    n_change = len(changes)
    out.append(
        f'\n{n_change} non-flux species changed '
        f'- top {min(top_n_species, n_change)} by magnitude:'
    )
    sorted_changes = sorted(changes.items(),
                            key=lambda kv: abs(kv[1]['change']),
                            reverse=True)
    for sp_name, info in sorted_changes[:top_n_species]:
        sign = '+' if info['change'] >= 0 else ''
        out.append(
            f'\n  {sp_name}: {sign}{info["change"]:.0f}'
            f'  (x[t]={info["x_t"]:.0f} -> x[t+1]={info["x_t1"]:.0f})'
        )
        # Production attributions
        if info['produced_by']:
            for entry in sorted(info['produced_by'],
                                key=lambda e: -abs(e['contribution'])):
                out.append(
                    f'    + {entry["reaction"]:30s} '
                    f'fired {entry["n_fires"]:>5,d}x'
                    f'  x stoich {entry["stoich_coef"]:+.0f}'
                    f'  = +{entry["contribution"]:.0f}'
                )
        # Consumption attributions
        if info['consumed_by']:
            for entry in sorted(info['consumed_by'],
                                key=lambda e: -abs(e['contribution'])):
                out.append(
                    f'    - {entry["reaction"]:30s} '
                    f'fired {entry["n_fires"]:>5,d}x'
                    f'  x stoich {entry["stoich_coef"]:+.0f}'
                    f'  = {entry["contribution"]:.0f}'
                )
        # Sanity residual
        if show_residual and abs(info['residual']) >= 1.0:
            out.append(
                f'    ! unexplained residual: {info["residual"]:+.0f}'
                f' (chemistry outside SBML, e.g. translation/translocation)'
            )
        # Gene context
        if 'gene_context' in info:
            gc = info['gene_context']
            line = f'    [gene] {gc["locus_tag"]}'
            if gc['gene_name']:
                line += f' | {gc["gene_name"]}'
            if gc['product']:
                line += f' | "{gc["product"][:60]}"'
            if gc['ec_number']:
                line += f' | EC={gc["ec_number"]}'
            out.append(line)
        # Spatial context (compact summary)
        if 'spatial' in info:
            spdata = info['spatial']
            interesting_keys = [
                k for k in spdata
                if 'dist_to_membrane' in k or 'frac_in' in k
            ]
            if interesting_keys:
                # Pick the most representative ones
                shown = []
                for key in interesting_keys[:3]:
                    val = spdata[key]
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        shown.append(f'{key}={val:.2f}')
                if shown:
                    out.append(f'    [spatial] {", ".join(shown)}')

    if n_change > top_n_species:
        out.append(f'\n  ...and {n_change - top_n_species} more species changes')

    return '\n'.join(out)


# ----------------------------------------------------------------------
# Higher-level scanner
# ----------------------------------------------------------------------

def find_interesting_seconds(
    trajectory: Any,
    *,
    flux_indices: Optional[np.ndarray] = None,
    row_names: Optional[List[str]] = None,
    top_n: int = 20,
    criterion: str = 'most_events',
) -> List[Tuple[int, float]]:
    """Scan all 1-second windows in a replicate and rank them.

    Args
    ----
    trajectory : pd.DataFrame, shape (n_species, n_timesteps)
    flux_indices : optional list of F_* row indices. If provided, the
        'most_events' criterion uses these directly. Required if the
        criterion is 'most_events'.
    criterion :
        'most_events'    - total |reaction events| in the window (default)
        'most_changes'   - number of distinct species rows that changed
        'biggest_swing'  - sum of |delta| across all rows
    top_n : int, return the top N seconds.

    Returns
    -------
    List of (t, score) tuples, sorted descending by score.
    """
    arr = trajectory.to_numpy()                  # (n_species, n_timesteps)
    deltas = np.diff(arr, axis=1)                # (n_species, n_timesteps - 1)
    if criterion == 'most_events':
        if flux_indices is None:
            raise ValueError('flux_indices required for most_events criterion')
        # Sum |delta| over F_* rows for each time window
        score = np.abs(deltas[flux_indices, :]).sum(axis=0)
    elif criterion == 'most_changes':
        score = (np.abs(deltas) > 0.5).sum(axis=0)
    elif criterion == 'biggest_swing':
        score = np.abs(deltas).sum(axis=0)
    else:
        raise ValueError(f'unknown criterion {criterion!r}')
    order = np.argsort(-score)[:top_n]
    return [(int(t), float(score[t])) for t in order]


# ----------------------------------------------------------------------
# Internal
# ----------------------------------------------------------------------

def _extract_locus(name: str) -> Optional[str]:
    """Pull the 4-digit locus number out of a species name, or None."""
    m = _LOCUS_RE.search(name)
    return m.group(1) if m else None
