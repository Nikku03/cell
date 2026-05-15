"""Smoke tests for the four backward-inference tools."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent.parent))

from cell_sim.lgnn.analysis import (
    audit_species_economy,
    find_coupled_species,
    decompose_counts_by_location,
)


# ----------------------------------------------------------------------
# Fixtures (shared with the narrate_window tests but redeclared here so
# this test file is self-contained)
# ----------------------------------------------------------------------

def _toy_trajectory_with_flux() -> pd.DataFrame:
    """5-timestep, 4-species toy.

    R1: M_A -> M_B  (fires 3, 5, 0, 2 events in the 4 windows)
    """
    rows = ['M_A', 'M_B', 'F_R1', 'G_0001']
    arr = np.array([
        [100, 97, 92, 92, 90],     # M_A
        [  0,  3,  8,  8, 10],     # M_B
        [  0,  3,  8,  8, 10],     # F_R1 (cumulative)
        [  2,  2,  2,  2,  3],     # G_0001 (jumps in last window)
    ], dtype=np.float32)
    return pd.DataFrame(arr, index=rows)


def _toy_stoich():
    """S, flux_indices, reaction_ids for the toy."""
    S = np.array([
        [-1.0],   # M_A consumed
        [+1.0],   # M_B produced
        [ 0.0],   # F_R1 (flux row)
        [ 0.0],   # G_0001
    ], dtype=np.float32)
    flux_indices = np.array([2], dtype=np.int64)
    return S, flux_indices, ['R1']


# ----------------------------------------------------------------------
# species_economy
# ----------------------------------------------------------------------

def test_audit_species_economy_simple_balance():
    """For M_B, window t=0..t=2: consumed 0, produced 8, observed Δ = +8."""
    traj = _toy_trajectory_with_flux()
    audit = audit_species_economy(
        species_name='M_B',
        trajectory=traj,
        row_names=list(traj.index),
        stoich=_toy_stoich(),
        t_start=0, t_end=2,
    )
    assert audit.species == 'M_B'
    assert audit.observed_change == 8.0
    assert audit.total_produced == 8.0
    assert audit.total_consumed == 0.0
    assert audit.residual == 0.0
    assert 'R1' in audit.produced_by
    assert audit.produced_by['R1'] == 8.0


def test_audit_species_economy_unexplained_residual():
    """G_0001 jumps 2->3 in last window with no SBML reaction; residual=1."""
    traj = _toy_trajectory_with_flux()
    audit = audit_species_economy(
        species_name='G_0001',
        trajectory=traj,
        row_names=list(traj.index),
        stoich=_toy_stoich(),
        t_start=3, t_end=4,
    )
    assert audit.observed_change == 1.0
    assert audit.net_attributed == 0.0
    assert audit.residual == 1.0


def test_audit_species_economy_summary_string_renders():
    traj = _toy_trajectory_with_flux()
    audit = audit_species_economy(
        species_name='M_A',
        trajectory=traj,
        row_names=list(traj.index),
        stoich=_toy_stoich(),
    )
    s = audit.summary()
    assert 'M_A' in s
    assert 'R1' in s


def test_audit_species_economy_oob_raises():
    traj = _toy_trajectory_with_flux()
    try:
        audit_species_economy(
            species_name='UNKNOWN',
            trajectory=traj,
            row_names=list(traj.index),
            stoich=_toy_stoich(),
        )
        assert False, 'expected ValueError on unknown species'
    except ValueError:
        pass


# ----------------------------------------------------------------------
# functional_coupling
# ----------------------------------------------------------------------

def test_find_coupled_species_pearson_finds_correlated():
    """Two perfectly correlated species + one independent. The correlated
    pair should rank first."""
    rng = np.random.RandomState(0)
    base = rng.randn(100)
    arr = np.stack([
        base,                       # species 0
        2 * base + 0.01 * rng.randn(100),     # species 1 - linear with 0
        rng.randn(100),             # species 2 - independent
    ])
    df = pd.DataFrame(arr, index=['A', 'B', 'C'])
    coupled = find_coupled_species(df, list(df.index), top_n=3)
    assert len(coupled) == 3
    # The (A, B) pair should be the top hit
    top = coupled.iloc[0]
    assert (top['species_a'], top['species_b']) in [('A', 'B'), ('B', 'A')]
    assert abs(top['correlation']) > 0.95


def test_find_coupled_species_filter_works():
    rng = np.random.RandomState(0)
    arr = rng.randn(5, 50)
    df = pd.DataFrame(arr, index=['G_0001', 'G_0002', 'R_0001', 'R_0002', 'P_0001'])
    out = find_coupled_species(df, list(df.index),
                                species_filter=['G_0001', 'G_0002'],
                                top_n=10)
    # Only G_0001 and G_0002 should appear
    species_in_out = set(out['species_a']) | set(out['species_b'])
    assert species_in_out == {'G_0001', 'G_0002'}


def test_find_coupled_species_low_variance_filtered():
    """Constant rows should be dropped (their variance is below threshold)."""
    arr = np.array([
        [1, 1, 1, 1, 1],    # constant -> should be filtered
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
    ], dtype=np.float32)
    df = pd.DataFrame(arr, index=['flat', 'up', 'down'])
    out = find_coupled_species(df, list(df.index),
                                min_variance=0.1, top_n=10)
    species = set(out['species_a']) | set(out['species_b'])
    assert 'flat' not in species
    assert 'up' in species and 'down' in species


# ----------------------------------------------------------------------
# spatial_decompose
# ----------------------------------------------------------------------

def test_decompose_counts_basic():
    """For a protein with 80% membrane / 20% cytoplasm fingerprint and
    count 1000, decomposition should give 800/200."""
    spatial_df = pd.DataFrame({
        'protein_frac_in_membrane':  [0.8],
        'protein_frac_in_cytoplasm': [0.2],
        'gene_frac_in_DNA':          [1.0],   # different family
    }, index=['0001'])
    counts = {'P_0001': 1000.0}
    out = decompose_counts_by_location(counts, spatial_df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row['protein_frac_in_membrane']  == 800.0
    assert row['protein_frac_in_cytoplasm'] == 200.0


def test_decompose_counts_nan_when_no_spatial():
    """Species without a matching locus in spatial_df get NaN fractions."""
    spatial_df = pd.DataFrame({
        'protein_frac_in_membrane': [0.5],
    }, index=['0001'])
    counts = {'P_0999': 100.0}      # no matching locus
    out = decompose_counts_by_location(counts, spatial_df)
    assert len(out) == 1
    assert pd.isna(out.iloc[0]['protein_frac_in_membrane'])


def test_decompose_counts_uses_prefix_family():
    """Gene rows should pick up gene_* spatial columns, not protein_* ones."""
    spatial_df = pd.DataFrame({
        'gene_frac_in_DNA':            [1.0],
        'gene_frac_in_cytoplasm':      [0.0],
        'protein_frac_in_cytoplasm':   [0.5],
    }, index=['0001'])
    counts = {'G_0001': 100.0}
    out = decompose_counts_by_location(counts, spatial_df)
    assert out.iloc[0]['gene_frac_in_DNA'] == 100.0
    # protein_frac_in_cytoplasm shouldn't be applied to a gene row
    if 'protein_frac_in_cytoplasm' in out.columns:
        assert pd.isna(out.iloc[0]['protein_frac_in_cytoplasm'])
