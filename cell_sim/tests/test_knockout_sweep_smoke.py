"""Smoke tests for the knockout sweep.

Three layers:
1. knockout_sweep on a toy linear model where divergence has a known
   ground truth (clamping a high-coefficient input must produce more
   divergence than clamping a zero-coefficient input).
2. gene_row_mapping_from_loci: verify prefix filtering and locus
   parsing match the species-graph regex behavior.
3. evaluate_against_essentiality: synthetic ranking + ground truth
   give known MCC, precision, recall.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent.parent))

from cell_sim.lgnn.microscope.knockout_sweep import (
    knockout_sweep,
    gene_row_mapping_from_loci,
    evaluate_against_essentiality,
)


class _ToyLinearDynamics(nn.Module):
    """dx = W @ x, with W chosen so row 0 is strongly upstream of
    everything else. Knocking out row 0 should produce LARGE divergence
    vs knocking out an isolated row (e.g., the last row which has no
    outgoing weight)."""
    def __init__(self, n_species: int = 10):
        super().__init__()
        W = torch.zeros(n_species, n_species)
        # Row 0 influences rows 1..3 strongly
        W[1, 0] = 0.3
        W[2, 0] = 0.2
        W[3, 0] = 0.4
        # Row 4 has no outgoing influence on anyone
        # Mild diagonal so trajectories don't blow up
        W -= 0.1 * torch.eye(n_species)
        self.W = nn.Parameter(W, requires_grad=False)

    def forward(self, x):
        # x: (B, S)  →  dx: (B, S)
        return x @ self.W.T


def test_knockout_of_upstream_node_produces_more_divergence():
    """Sanity: KO of row 0 (which influences rows 1, 2, 3) must produce
    more divergence than KO of row 4 (no downstream influence)."""
    torch.manual_seed(0)
    model = _ToyLinearDynamics(n_species=10)
    model.eval()
    initial = torch.randn(10) + 2.0   # nonzero initial state

    df = knockout_sweep(
        model, initial,
        species_to_test=[0, 4],
        n_rollout_steps=20,
    )
    # df is sorted by divergence descending; row 0 should come first
    by_idx = df.set_index('species_idx')['divergence_excluding_ko']
    assert by_idx[0] > by_idx[4], (
        f'KO of upstream row 0 ({by_idx[0]:.4f}) should diverge more '
        f'than KO of dangling row 4 ({by_idx[4]:.4f})'
    )


def test_knockout_sweep_returns_dataframe_with_correct_columns():
    model = _ToyLinearDynamics(n_species=8)
    model.eval()
    initial = torch.randn(8) + 1.0
    df = knockout_sweep(
        model, initial,
        species_to_test=[0, 1, 2],
        n_rollout_steps=5,
        row_names=[f'r_{i}' for i in range(8)],
    )
    assert set(df.columns) >= {
        'species_idx', 'species_name', 'divergence',
        'divergence_excluding_ko',
    }
    assert len(df) == 3
    assert df['species_name'].tolist() == sorted(df['species_name'].tolist(),
        key=lambda n: -df.set_index('species_name')['divergence_excluding_ko'][n])


def test_knockout_sweep_batched_matches_unbatched():
    """Batching is a perf optimization; results must match across
    batch_size choices."""
    torch.manual_seed(42)
    model = _ToyLinearDynamics(n_species=12)
    model.eval()
    initial = torch.randn(12) + 1.0
    targets = [0, 1, 2, 3, 4, 5]

    df_all = knockout_sweep(model, initial, targets,
                              n_rollout_steps=10, batch_size=None)
    df_two = knockout_sweep(model, initial, targets,
                              n_rollout_steps=10, batch_size=2)
    a = df_all.sort_values('species_idx').reset_index(drop=True)
    b = df_two.sort_values('species_idx').reset_index(drop=True)
    np.testing.assert_allclose(
        a['divergence_excluding_ko'].values,
        b['divergence_excluding_ko'].values,
        rtol=1e-5, atol=1e-7,
    )


def test_gene_row_mapping_filters_by_prefix():
    rows = ['G_0001', 'R_0001', 'P_0001',
            'G_0042', 'RP_0042', 'PM_0042',
            'M_atp_c',                    # not a gene row
            'F_PFK',                      # flux row
            'G_999_C1']                   # gene with state suffix
    m = gene_row_mapping_from_loci(rows, knockout_prefix='G')
    assert 'JCVISYN3A_0001' in m
    assert 'JCVISYN3A_0042' in m
    assert 'JCVISYN3A_999' in m
    # M_atp_c and F_PFK are filtered out by the regex (no 3-digit locus)
    assert m['JCVISYN3A_0001'] == rows.index('G_0001')
    assert m['JCVISYN3A_0042'] == rows.index('G_0042')

    # With a different prefix, gets the R rows
    m_r = gene_row_mapping_from_loci(rows, knockout_prefix='R')
    assert 'JCVISYN3A_0001' in m_r
    assert m_r['JCVISYN3A_0001'] == rows.index('R_0001')
    assert 'JCVISYN3A_0042' not in m_r   # no R_0042 in the test list


def test_evaluate_against_essentiality_recovers_known_metrics():
    """Synthetic ranking where the top-3 by divergence are the true
    essentials. Compute MCC/precision/recall by hand and verify."""
    # 5 genes; 3 are essential. Divergence values rank them right.
    div_df = pd.DataFrame({
        'gene_id':    ['G1', 'G2', 'G3', 'G4', 'G5'],
        'species_idx': [0, 1, 2, 3, 4],
        'divergence':  [0.9, 0.8, 0.7, 0.1, 0.05],
        'divergence_excluding_ko': [0.9, 0.8, 0.7, 0.1, 0.05],
    })
    ess_df = pd.DataFrame({
        'locus_tag': ['G1', 'G2', 'G3', 'G4', 'G5'],
        'essential': [True, True, True, False, False],
    })
    res = evaluate_against_essentiality(
        div_df, ess_df,
        gene_id_col='locus_tag', essential_col='essential',
        top_k=3, divergence_col='divergence_excluding_ko',
    )
    # Perfect prediction: top-3 are exactly the 3 essentials
    assert res['n_genes_evaluated'] == 5
    assert res['n_essential_true'] == 3
    assert res['n_essential_predicted'] == 3
    assert res['tp'] == 3
    assert res['fp'] == 0
    assert res['fn'] == 0
    assert res['tn'] == 2
    assert res['MCC'] == 1.0
    assert res['precision'] == 1.0
    assert res['recall'] == 1.0


def test_evaluate_against_essentiality_imperfect_ranking():
    """Top-3 includes 2 true + 1 false essential. Verify MCC < 1."""
    div_df = pd.DataFrame({
        'gene_id': ['G1', 'G2', 'G3', 'G4', 'G5'],
        'species_idx': [0, 1, 2, 3, 4],
        'divergence_excluding_ko': [0.9, 0.5, 0.8, 0.7, 0.05],
    })
    ess_df = pd.DataFrame({
        'locus_tag': ['G1', 'G2', 'G3', 'G4', 'G5'],
        'essential': [True, True, True, False, False],
    })
    res = evaluate_against_essentiality(
        div_df, ess_df, top_k=3,
    )
    # Top 3 by divergence: G1 (0.9), G3 (0.8), G4 (0.7)
    # G1=essential ✓, G3=essential ✓, G4=non-ess (FP)
    # G2 has divergence 0.5 but IS essential (FN)
    assert res['tp'] == 2
    assert res['fp'] == 1
    assert res['fn'] == 1
    assert res['tn'] == 1
    assert 0 < res['MCC'] < 1


def test_knockout_sweep_outputs_finite_nonneg_divergences():
    """Both divergence metrics must be finite and non-negative for
    every knockout. (The earlier ordering assumption between
    `divergence` and `divergence_excluding_ko` was wrong: which is
    bigger depends on whether the clamped row contributed more or
    less than the average row to the sum-of-squares.)"""
    model = _ToyLinearDynamics(n_species=8)
    model.eval()
    initial = torch.randn(8) + 2.0
    targets = [0, 3, 5]
    df = knockout_sweep(model, initial, targets, n_rollout_steps=15)
    for _, row in df.iterrows():
        assert np.isfinite(row['divergence'])
        assert np.isfinite(row['divergence_excluding_ko'])
        assert row['divergence'] >= 0.0
        assert row['divergence_excluding_ko'] >= 0.0


if __name__ == '__main__':
    test_knockout_of_upstream_node_produces_more_divergence()
    test_knockout_sweep_returns_dataframe_with_correct_columns()
    test_knockout_sweep_batched_matches_unbatched()
    test_gene_row_mapping_filters_by_prefix()
    test_evaluate_against_essentiality_recovers_known_metrics()
    test_evaluate_against_essentiality_imperfect_ranking()
    test_knockout_sweep_outputs_finite_nonneg_divergences()
    print('OK: knockout-sweep smoke tests passed.')
