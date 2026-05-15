"""Smoke tests for cell_sim.lgnn.analysis.narrate_window."""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent.parent))

from cell_sim.lgnn.analysis import (
    narrate_window, format_narrative, find_interesting_seconds,
    NarrationResult,
)
from cell_sim.lgnn.analysis.narrate_window import _extract_locus


def _toy_trajectory() -> pd.DataFrame:
    """Build a tiny 4-species, 5-timestep trajectory + matching SBML+S.

    Species (in this order):
      M_A     : metabolite, count starts at 100, declines as R1 fires
      M_B     : metabolite, count starts at 0, grows as R1 fires
      F_R1    : flux counter for reaction R1, increments by # fires per step
      G_0001  : non-SBML gene row, constant at 2
    Reactions:
      R1 : M_A -> M_B  (one A consumed, one B produced per fire)
    """
    rows = ['M_A', 'M_B', 'F_R1', 'G_0001']
    T = 5
    arr = np.zeros((len(rows), T), dtype=np.float32)
    # t=0
    arr[:, 0] = [100, 0, 0, 2]
    # t=1: R1 fires 3 times  -> A -=3, B +=3, F_R1 +=3
    arr[:, 1] = [97, 3, 3, 2]
    # t=2: R1 fires 5 times  -> A -=5, B +=5, F_R1 +=5
    arr[:, 2] = [92, 8, 8, 2]
    # t=3: R1 fires 0 times  -> nothing changes
    arr[:, 3] = [92, 8, 8, 2]
    # t=4: R1 fires 2 times + spurious change in G_0001
    arr[:, 4] = [90, 10, 10, 3]
    return pd.DataFrame(arr, index=rows)


def _toy_stoich():
    """Build the matching S, flux_indices, reaction_ids for the toy data."""
    # n_species = 4, n_reactions = 1
    # Reaction R1: M_A (-1) -> M_B (+1)
    S = np.array([
        [-1.0],   # M_A (row 0): consumed by R1
        [ 1.0],   # M_B (row 1): produced by R1
        [ 0.0],   # F_R1 (row 2): flux counter, not in S
        [ 0.0],   # G_0001 (row 3): not in SBML
    ], dtype=np.float32)
    flux_indices = np.array([2], dtype=np.int64)   # F_R1 lives at row 2
    reaction_ids = ['R1']
    return S, flux_indices, reaction_ids


# ----------------------------------------------------------------------
# Core narrative tests
# ----------------------------------------------------------------------

def test_narrate_window_basic():
    """End-to-end: 3 fires of R1 should be attributed correctly."""
    traj = _toy_trajectory()
    stoich = _toy_stoich()
    res = narrate_window(
        replicate=0, t=0,
        trajectory=traj, row_names=list(traj.index), stoich=stoich,
    )
    # Reaction R1 should be in reaction_events with n_fires=3
    assert 'R1' in res.reaction_events
    assert res.reaction_events['R1']['n_fires'] == 3
    assert res.reaction_events['R1']['flux_row_idx'] == 2

    # M_A should be in species_changes with change=-3 attributed to R1
    assert 'M_A' in res.species_changes
    a = res.species_changes['M_A']
    assert a['change'] == -3.0
    assert len(a['produced_by']) == 0
    assert len(a['consumed_by']) == 1
    assert a['consumed_by'][0]['reaction'] == 'R1'
    assert a['consumed_by'][0]['contribution'] == -3.0
    assert a['residual'] == 0.0   # fully explained

    # M_B should be +3, attributed to R1 (production)
    assert 'M_B' in res.species_changes
    b = res.species_changes['M_B']
    assert b['change'] == 3.0
    assert len(b['produced_by']) == 1
    assert b['produced_by'][0]['contribution'] == 3.0
    assert b['residual'] == 0.0


def test_narrate_window_unexplained_residual():
    """Non-SBML row changes should show up as unexplained residual."""
    traj = _toy_trajectory()
    stoich = _toy_stoich()
    # Window 3 -> 4: G_0001 goes from 2 -> 3 (no reaction in SBML explains this)
    res = narrate_window(
        replicate=0, t=3,
        trajectory=traj, row_names=list(traj.index), stoich=stoich,
    )
    assert 'G_0001' in res.species_changes
    g = res.species_changes['G_0001']
    assert g['change'] == 1.0
    assert g['net_attributed'] == 0.0
    assert g['residual'] == 1.0   # NOT explained by SBML


def test_narrate_window_quiet_step_returns_empty_events():
    """t=2 -> t=3 has no changes; both fields should be empty."""
    traj = _toy_trajectory()
    stoich = _toy_stoich()
    res = narrate_window(
        replicate=0, t=2,
        trajectory=traj, row_names=list(traj.index), stoich=stoich,
    )
    assert res.n_reactions_fired() == 0
    assert res.n_species_changed() == 0


def test_narrate_window_oob_t_raises():
    traj = _toy_trajectory()
    stoich = _toy_stoich()
    # T=5, window_size=1 -> last valid t is 3
    res = narrate_window(replicate=0, t=3,
                         trajectory=traj, row_names=list(traj.index),
                         stoich=stoich)  # ok
    assert isinstance(res, NarrationResult)
    # t=4 with window=1 would need x[5], which doesn't exist
    try:
        narrate_window(replicate=0, t=4,
                       trajectory=traj, row_names=list(traj.index),
                       stoich=stoich)
        assert False, 'expected ValueError'
    except ValueError:
        pass


# ----------------------------------------------------------------------
# Locus extraction
# ----------------------------------------------------------------------

def test_extract_locus():
    assert _extract_locus('G_0234') == '0234'
    assert _extract_locus('R_0234_d') == '0234'
    assert _extract_locus('RP_0234_C1') == '0234'
    assert _extract_locus('PM_0612') == '0612'
    assert _extract_locus('M_atp_c') is None
    assert _extract_locus('F_R_PYK') is None


def test_gene_context_enrichment():
    """If genes dict is provided, matching loci get gene_context attached."""
    traj = _toy_trajectory()
    stoich = _toy_stoich()
    fake_genes = {
        'JCVISYN3A_0001': {
            'gene': 'pyk', 'product': 'Pyruvate kinase',
            'function': 'glycolysis', 'ec': '2.7.1.40',
        },
    }
    res = narrate_window(
        replicate=0, t=3,
        trajectory=traj, row_names=list(traj.index), stoich=stoich,
        genes=fake_genes,
    )
    assert 'gene_context' in res.species_changes['G_0001']
    gc = res.species_changes['G_0001']['gene_context']
    assert gc['gene_name'] == 'pyk'
    assert gc['product'] == 'Pyruvate kinase'
    assert gc['ec_number'] == '2.7.1.40'


def test_format_narrative_runs():
    """Formatter should produce a non-empty string without crashing."""
    traj = _toy_trajectory()
    stoich = _toy_stoich()
    res = narrate_window(
        replicate=0, t=0,
        trajectory=traj, row_names=list(traj.index), stoich=stoich,
    )
    s = format_narrative(res)
    assert isinstance(s, str)
    assert 'R1' in s          # the reaction should be named in the output
    assert 'M_A' in s         # changed species named
    assert 'M_B' in s


# ----------------------------------------------------------------------
# find_interesting_seconds scanner
# ----------------------------------------------------------------------

def test_find_interesting_seconds_most_events():
    traj = _toy_trajectory()
    # The 'most_events' criterion needs flux_indices
    out = find_interesting_seconds(
        traj, flux_indices=np.array([2], dtype=np.int64),
        top_n=5, criterion='most_events',
    )
    # Reactions per window: t=0->1: 3 fires, t=1->2: 5, t=2->3: 0, t=3->4: 2
    # Sort descending: [(1, 5), (0, 3), (3, 2), (2, 0)]
    assert len(out) == 4
    assert out[0] == (1, 5.0)
    assert out[1] == (0, 3.0)
    assert out[2] == (3, 2.0)
    assert out[3] == (2, 0.0)


def test_find_interesting_seconds_most_changes():
    traj = _toy_trajectory()
    out = find_interesting_seconds(traj, top_n=5, criterion='most_changes')
    # Number of changed species per window:
    #   t=0->1: M_A, M_B, F_R1 changed = 3
    #   t=1->2: M_A, M_B, F_R1 changed = 3
    #   t=2->3: nothing changed = 0
    #   t=3->4: M_A, M_B, F_R1, G_0001 = 4
    assert out[0] == (3, 4.0)


def test_find_interesting_seconds_biggest_swing():
    traj = _toy_trajectory()
    out = find_interesting_seconds(traj, top_n=5, criterion='biggest_swing')
    # |delta| sum:
    #   t=0->1: 3+3+3 = 9
    #   t=1->2: 5+5+5 = 15
    #   t=2->3: 0
    #   t=3->4: 2+2+2+1 = 7
    assert out[0] == (1, 15.0)
    assert out[1] == (0, 9.0)
