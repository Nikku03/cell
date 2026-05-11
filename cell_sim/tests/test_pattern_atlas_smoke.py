"""Smoke tests for the pattern atlas microscope.

Verifies whole_cell_atlas and per_species_atlas return well-formed
outputs with non-degenerate clustering on synthetic multi-regime
trajectory data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent.parent))

from cell_sim.lgnn.microscope.pattern_atlas import (
    whole_cell_atlas,
    per_species_atlas,
    characterize_whole_cell_clusters,
    save_atlas,
    load_atlas,
)


def _synthetic_multi_regime_trajectory(T=300, S=20, n_regimes=4, seed=0):
    """A trajectory that switches between n_regimes distinct mean
    levels. Whole-cell atlas should recover ≥ n_regimes clusters."""
    rng = np.random.default_rng(seed)
    regime_means = rng.normal(0, 1, (n_regimes, S))
    schedule = np.repeat(np.arange(n_regimes), T // n_regimes)
    X = regime_means[schedule] + 0.1 * rng.normal(0, 1, (len(schedule), S))
    return X.astype(np.float32)


def test_whole_cell_atlas_recovers_regime_count():
    """4-regime synthetic data → atlas should produce ≥4 active clusters."""
    X = _synthetic_multi_regime_trajectory(T=400, S=15, n_regimes=4)
    atlas = whole_cell_atlas(X, K=8, seed=0)
    assert atlas['T'] == 400 and atlas['S'] == 15
    assert atlas['labels'].shape == (400,)
    assert atlas['centroids'].shape == (8, 15)
    # All 4 regimes should be discoverable — atlas active_K ≥ 4
    assert atlas['active_K'] >= 4
    # No single cluster should dominate everything
    assert atlas['timestep_density'].max() < 0.95 * 400


def test_whole_cell_atlas_separates_regimes():
    """Two distinct regimes → the cluster labels in regime 1 should be
    largely disjoint from the cluster labels in regime 2. Within each
    regime can be further subdivided (k > n_regimes), but the BREAK
    at the regime boundary must be clear."""
    X = _synthetic_multi_regime_trajectory(T=200, S=10, n_regimes=2, seed=42)
    atlas = whole_cell_atlas(X, K=4, seed=0)
    first_half_labels  = set(np.unique(atlas['labels'][:100]).tolist())
    second_half_labels = set(np.unique(atlas['labels'][100:]).tolist())
    overlap = first_half_labels & second_half_labels
    union   = first_half_labels | second_half_labels
    # The two halves' label sets should be almost disjoint
    assert len(overlap) <= 1
    assert len(union) >= 2


def test_characterize_whole_cell_clusters_output():
    X = _synthetic_multi_regime_trajectory(T=200, S=10, n_regimes=3)
    atlas = whole_cell_atlas(X, K=6, seed=0)
    row_names = [f's_{i}' for i in range(10)]
    char_df = characterize_whole_cell_clusters(atlas, row_names,
                                                  top_n_distinctive=3)
    assert {'cluster_id', 'density', 'top_species',
            'top_signed_zscore'} <= set(char_df.columns)
    assert len(char_df) >= 1
    # Each row's top_species should be 3 species names from the input
    for _, r in char_df.iterrows():
        assert len(r['top_species']) == 3
        for name in r['top_species']:
            assert name in row_names


def test_per_species_atlas_shape_and_labels():
    X = _synthetic_multi_regime_trajectory(T=100, S=8, n_regimes=2)
    atlas = per_species_atlas(X, K=10, window=3, seed=0)
    expected_pairs = (100 - 3) * 8
    assert atlas['n_pairs'] == expected_pairs
    assert atlas['labels'].shape == (expected_pairs,)
    assert atlas['centroids'].shape == (10, 3)
    assert atlas['active_K'] >= 2


def test_per_species_atlas_with_subset():
    """species_subset restricts which species get clustered."""
    X = _synthetic_multi_regime_trajectory(T=100, S=12, n_regimes=2)
    subset = [0, 3, 7]
    atlas = per_species_atlas(X, K=4, window=2, species_subset=subset)
    expected_pairs = (100 - 2) * len(subset)
    assert atlas['n_pairs'] == expected_pairs
    np.testing.assert_array_equal(atlas['species_subset'], np.array(subset))
    # Every (species, time) pair must have species ∈ subset
    assert set(atlas['species_per_pair'].tolist()) <= set(subset)


def test_save_and_load_atlas_roundtrip(tmp_path):
    X = _synthetic_multi_regime_trajectory(T=80, S=6, n_regimes=2)
    atlas = whole_cell_atlas(X, K=4, seed=0)
    out_path = tmp_path / 'atlas.npz'
    save_atlas(atlas, out_path)
    loaded = load_atlas(out_path)
    np.testing.assert_array_equal(loaded['labels'], atlas['labels'])
    np.testing.assert_allclose(loaded['centroids'], atlas['centroids'])
    assert int(loaded['active_K']) == atlas['active_K']


if __name__ == '__main__':
    import tempfile
    test_whole_cell_atlas_recovers_regime_count()
    test_whole_cell_atlas_separates_regimes()
    test_characterize_whole_cell_clusters_output()
    test_per_species_atlas_shape_and_labels()
    test_per_species_atlas_with_subset()
    with tempfile.TemporaryDirectory() as d:
        test_save_and_load_atlas_roundtrip(Path(d))
    print('OK: pattern-atlas smoke tests passed.')
