"""Smoke tests for the edge-attribution microscope.

Runs capture_attention on a tiny M1+axis2 model and verifies:
  - α has the expected shape (T_sampled, E)
  - α rows sum to 1 per destination (proper softmax)
  - Self-loop edges have α=0 (Bug 1 fix is preserved)
  - aggregate_attention produces correct columns
  - top_attended_per_node returns top-K per dst with rank labels
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent.parent))

from cell_sim.lgnn.data.species_graph import build_species_graph, EdgeKind
from cell_sim.lgnn.models.gnn_v1_axis2 import CellGNNv1Axis2
from cell_sim.lgnn.microscope.edge_attribution import (
    capture_attention,
    aggregate_attention,
    top_attended_per_node,
)


def _write_fake_cobra(path: Path):
    model = {
        'metabolites': [{'id': 'A'}, {'id': 'B'}, {'id': 'C'}],
        'reactions': [
            {'id': 'R1', 'metabolites': {'A': -1.0, 'B': 1.0}},
            {'id': 'R2', 'metabolites': {'B': -1.0, 'C': -1.0, 'A': 1.0}},
        ],
    }
    path.write_text(json.dumps(model))


def _tiny_graph_and_model(tmp: Path, hidden=16, n_layers=2):
    cobra = tmp / 'mini.json'
    _write_fake_cobra(cobra)
    rows = ['A', 'B', 'C',
            'G_0001', 'RP_0001', 'R_0001', 'RPM_0001',
            'P_0001', 'PM_0001', 'DM_0001', 'D_0001',
            'F_R1']
    g = build_species_graph(rows, cobra, include_simulator_edges=True)
    model = CellGNNv1Axis2(g, hidden=hidden, n_layers=n_layers,
                            use_checkpoint=False).eval()
    return g, model, rows


def test_capture_attention_shape_and_finite(tmp_path):
    g, model, rows = _tiny_graph_and_model(tmp_path)
    X = torch.randn(50, g.n_nodes)
    capture = capture_attention(model, X, layer_idx=0, sample_every=5)
    assert capture['T_sampled'] == 10            # 50 / 5
    assert capture['alpha'].shape == (10, g.n_edges)
    assert np.isfinite(capture['alpha']).all()


def test_capture_attention_softmax_normalized(tmp_path):
    """Per (timestep, destination), α should sum to 1.0 across all
    incoming edges (excluding self-loops which are masked to 0)."""
    g, model, rows = _tiny_graph_and_model(tmp_path)
    X = torch.randn(20, g.n_nodes)
    capture = capture_attention(model, X, layer_idx=0, sample_every=10)
    alpha = capture['alpha']                     # (T_sampled, E)
    dst = capture['edge_index'][1]
    # For each (timestep, dst), sum of α over incoming should be 1 or 0
    # (0 only if all incoming edges are self-loops; rare for our graph)
    for t in range(alpha.shape[0]):
        sums_per_dst = np.zeros(g.n_nodes)
        for e in range(alpha.shape[1]):
            sums_per_dst[dst[e]] += alpha[t, e]
        # Sums should be either 0 (no non-self incoming) or 1
        for d in range(g.n_nodes):
            assert (sums_per_dst[d] < 1e-5) or abs(sums_per_dst[d] - 1.0) < 1e-3


def test_capture_attention_self_loops_have_zero_alpha(tmp_path):
    """Bug 1 fix: every self-loop edge must have α=0 in the capture."""
    g, model, rows = _tiny_graph_and_model(tmp_path)
    X = torch.randn(10, g.n_nodes)
    capture = capture_attention(model, X, layer_idx=0, sample_every=5)
    alpha = capture['alpha']
    is_self = capture['edge_kind'] == int(EdgeKind.SELF_LOOP)
    self_loop_alphas = alpha[:, is_self]
    assert (self_loop_alphas == 0).all()


def test_aggregate_attention_columns_and_values(tmp_path):
    g, model, rows = _tiny_graph_and_model(tmp_path)
    X = torch.randn(30, g.n_nodes)
    capture = capture_attention(model, X, sample_every=3)
    df = aggregate_attention(capture, row_names=rows)
    expected_cols = {
        'edge_idx', 'src_idx', 'dst_idx', 'src_name', 'dst_name',
        'edge_kind', 'mean_alpha', 'std_alpha',
        'q25_alpha', 'q50_alpha', 'q75_alpha', 'max_alpha',
        'frac_above_0.5',
    }
    assert expected_cols <= set(df.columns)
    assert (df['mean_alpha'] >= 0).all()
    assert (df['mean_alpha'] <= 1.0 + 1e-6).all()
    # All self-loop edges should have mean_alpha = 0
    self_rows = df[df['edge_kind'] == int(EdgeKind.SELF_LOOP)]
    assert (self_rows['mean_alpha'] == 0).all()


def test_top_attended_per_node(tmp_path):
    g, model, rows = _tiny_graph_and_model(tmp_path)
    X = torch.randn(20, g.n_nodes)
    capture = capture_attention(model, X, sample_every=4)
    df = aggregate_attention(capture, row_names=rows)
    top = top_attended_per_node(df, top_k=2, metric='mean_alpha')
    # Every dst with at least one incoming non-self edge should appear
    assert 'rank' in top.columns
    assert (top['rank'] >= 1).all() and (top['rank'] <= 2).all()
    # Per dst, ranks should be 1, 2 (no duplicates within a dst)
    for dst, group in top.groupby('dst_idx'):
        assert sorted(group['rank'].tolist()) == list(
            range(1, len(group) + 1)
        )
        # Higher rank = lower mean_alpha (sorted descending)
        if len(group) > 1:
            sorted_by_rank = group.sort_values('rank')
            alphas = sorted_by_rank['mean_alpha'].tolist()
            assert all(a >= b for a, b in zip(alphas, alphas[1:]))


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d)
        test_capture_attention_shape_and_finite(tp)
        test_capture_attention_softmax_normalized(tp)
        test_capture_attention_self_loops_have_zero_alpha(tp)
        test_aggregate_attention_columns_and_values(tp)
        test_top_attended_per_node(tp)
    print('OK: edge-attribution smoke tests passed.')
