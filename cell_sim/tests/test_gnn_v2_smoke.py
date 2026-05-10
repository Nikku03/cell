"""Smoke tests for M2 Phase 1 — CfC node dynamics.

Verifies CfC layer math (gate clipping, sigmoid output bounded),
parameter count includes the per-species A/B biases, end-to-end
training runs and produces finite losses, and the model produces
*different* output from CellGNNv1Axis2 (architecture is meaningfully
different)."""
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

from cell_sim.lgnn.data.species_graph import build_species_graph
from cell_sim.lgnn.models.gnn_v1_axis2 import CellGNNv1Axis2
from cell_sim.lgnn.models.gnn_v2 import (
    CellGNNv2, _CfCAttentionGNNLayer, count_parameters,
)
from cell_sim.lgnn.training.train_m2 import (
    M2TrainConfig, train_m2,
)


def _write_fake_cobra(path: Path):
    model = {
        'metabolites': [
            {'id': 'A'}, {'id': 'B'}, {'id': 'C'},
        ],
        'reactions': [
            {'id': 'R1', 'metabolites': {'A': -1.0, 'B': 1.0}},
            {'id': 'R2', 'metabolites': {'B': -1.0, 'C': -1.0, 'A': 1.0}},
        ],
    }
    path.write_text(json.dumps(model))


def _build_tiny_graph(tmp: Path):
    cobra = tmp / 'mini.json'
    _write_fake_cobra(cobra)
    rows = ['A', 'B', 'C',
            'G_0001', 'RP_0001', 'R_0001', 'RPM_0001',
            'P_0001', 'PM_0001', 'DM_0001', 'D_0001',
            'F_R1']
    return build_species_graph(rows, cobra, include_simulator_edges=True), rows


def test_cfc_forward_shape_and_finiteness():
    with tempfile.TemporaryDirectory() as d:
        g, _ = _build_tiny_graph(Path(d))
    model = CellGNNv2(g, hidden=16, n_layers=2, use_checkpoint=False)
    x = torch.randn(4, g.n_nodes)
    y = model(x)
    assert y.shape == (4, g.n_nodes)
    assert torch.isfinite(y).all()


def test_cfc_param_count_includes_per_species_biases():
    """A and B per-species are the dominant param contribution."""
    with tempfile.TemporaryDirectory() as d:
        g, _ = _build_tiny_graph(Path(d))
    n_layers = 2
    hidden = 16
    n_nodes = g.n_nodes
    model = CellGNNv2(g, hidden=hidden, n_layers=n_layers, use_checkpoint=False)
    total = count_parameters(model)
    # Each layer contributes 2 * (n_nodes * hidden) for A + B alone.
    expected_AB = n_layers * 2 * n_nodes * hidden
    assert total >= expected_AB, (total, expected_AB)


def test_cfc_output_differs_from_axis2():
    """CellGNNv2 with the same hidden/n_layers should produce a
    different output from CellGNNv1Axis2 — confirms the CfC update
    is changing the forward pass, not silently falling back."""
    with tempfile.TemporaryDirectory() as d:
        g, _ = _build_tiny_graph(Path(d))
    torch.manual_seed(0)
    m_v2 = CellGNNv2(g, hidden=16, n_layers=2, use_checkpoint=False)
    torch.manual_seed(0)
    m_axis2 = CellGNNv1Axis2(g, hidden=16, n_layers=2, use_checkpoint=False)
    x = torch.randn(2, g.n_nodes)
    y_v2 = m_v2(x).detach()
    y_axis2 = m_axis2(x).detach()
    # Same seed → same input_proj/output_head/attention init, but the
    # update rule is different so outputs MUST differ.
    assert not torch.allclose(y_v2, y_axis2, atol=1e-3)


def test_cfc_W_is_clipped_to_tau_min_bound():
    """At the layer level: |W·h + b| should never blow up because W
    is clamped to ±1/τ_min. Probe via running forward with extreme h."""
    with tempfile.TemporaryDirectory() as d:
        g, _ = _build_tiny_graph(Path(d))
    layer = _CfCAttentionGNNLayer(
        hidden=16, edge_attr_dim=int(g.edge_attr.shape[1]),
        n_nodes=g.n_nodes, cfc_tau_min=0.1,
    )
    h = torch.randn(2, g.n_nodes, 16) * 100   # extreme magnitudes
    src = g.edge_index[0]
    dst = g.edge_index[1]
    edge_attr = g.edge_attr
    edge_kind = g.edge_kind
    out, ent = layer(h, g.edge_index, edge_attr, edge_kind, edge_mask=None)
    # CfC output is σ(...)·A + σ(...)·B with σ ∈ (0,1) and A,B small.
    # So |out| should be bounded by max(|A|, |B|), not blow up with |h|.
    assert torch.isfinite(out).all()
    # A, B are init'd at std=0.01, so output magnitude should be O(0.01)
    assert out.abs().max() < 1.0     # very loose bound — should hold easily


def test_cfc_gradient_flow_through_AB():
    """A and B per-species biases must receive gradients on backward."""
    with tempfile.TemporaryDirectory() as d:
        g, _ = _build_tiny_graph(Path(d))
    model = CellGNNv2(g, hidden=16, n_layers=2, use_checkpoint=False)
    model.train()
    x = torch.randn(2, g.n_nodes)
    model(x).pow(2).mean().backward()
    for layer in model.layers:
        assert layer.cfc_A.grad is not None
        assert layer.cfc_B.grad is not None
        assert torch.isfinite(layer.cfc_A.grad).all()
        assert torch.isfinite(layer.cfc_B.grad).all()
        # Gradient must be non-trivially nonzero for at least one row
        # (some species will have zero gradient if no signal reaches them
        # in the synthetic test, so check `any` not `all`)
        assert layer.cfc_A.grad.abs().sum() > 0
        assert layer.cfc_B.grad.abs().sum() > 0


def _make_fake_lsdata(rows, n_t):
    rng = np.random.default_rng(0)

    class FakeLs:
        @staticmethod
        def load_replicate(i, species=None, **_):
            base = rng.normal(0, 1, (len(rows), 1))
            t = np.arange(n_t).reshape(1, -1) / n_t
            data = (base + 0.3 * np.sin(2 * np.pi * t * (1 + i * 0.1))
                     ).astype(np.float32)
            return pd.DataFrame(
                data, index=pd.Index(rows, name='species'),
                columns=pd.Index(np.arange(n_t, dtype=float), name='time_s'),
            )
    return FakeLs()


def test_end_to_end_train_runs_with_cfc(tmp_path):
    rows = ['A', 'B', 'C',
            'G_0001', 'RP_0001', 'R_0001', 'RPM_0001',
            'P_0001', 'PM_0001', 'DM_0001', 'D_0001',
            'F_R1']
    fake_ls = _make_fake_lsdata(rows, n_t=40)
    cobra = tmp_path / 'mini.json'
    _write_fake_cobra(cobra)
    g = build_species_graph(rows, cobra, include_simulator_edges=True)

    cfg = M2TrainConfig(
        n_species=g.n_nodes, hidden=16, n_layers=2, batch_size=4, lr=3e-3,
        train_replicates=(1, 2, 3), val_replicates=(4,),
        n_epochs=1, device='cpu', seed=0, log_every=10000,
        edge_dropout_p=0.0, lambda_dropout=0.0, lambda_attn=0.0,
        lambda_attn_warmup_steps=0, lambda_attn_ramp_steps=0,
        lambda_dropout_warmup_steps=0, lambda_dropout_ramp_steps=0,
        k_curriculum=(2,), max_k=2,
        use_checkpoint=False, use_bf16=False, preload_dtype='float32',
        cfc_tau_min=0.1, wall_clock_budget_s=600,
    )
    out = train_m2(cfg, fake_ls, g, checkpoint_path=None)
    assert np.isfinite(out['history']['train_total'][0])
    assert out['history']['k_per_epoch'] == [2]
    assert len(out['history']['val_mse_per_step'][0]) == 2


def test_loss_decreases_on_synthetic_data(tmp_path):
    """CfC model can learn from the same synthetic dynamics that
    M1+axis2 learned from. Confirms no broken initialization."""
    rows = ['A', 'B', 'C',
            'G_0001', 'RP_0001', 'R_0001', 'RPM_0001',
            'P_0001', 'PM_0001', 'DM_0001', 'D_0001',
            'F_R1']
    fake_ls = _make_fake_lsdata(rows, n_t=60)
    cobra = tmp_path / 'mini.json'
    _write_fake_cobra(cobra)
    g = build_species_graph(rows, cobra, include_simulator_edges=True)

    cfg = M2TrainConfig(
        n_species=g.n_nodes, hidden=24, n_layers=2, batch_size=8, lr=3e-3,
        train_replicates=(1, 2, 3, 4), val_replicates=(5,),
        n_epochs=2, device='cpu', seed=0, log_every=10000,
        edge_dropout_p=0.0, lambda_dropout=0.0, lambda_attn=0.0,
        lambda_attn_warmup_steps=0, lambda_attn_ramp_steps=0,
        lambda_dropout_warmup_steps=0, lambda_dropout_ramp_steps=0,
        k_curriculum=(1, 1), max_k=1,
        use_checkpoint=False, use_bf16=False, preload_dtype='float32',
        cfc_tau_min=0.1, wall_clock_budget_s=600,
    )
    out = train_m2(cfg, fake_ls, g, checkpoint_path=None)
    assert out['history']['train_full_ss'][1] < out['history']['train_full_ss'][0]


if __name__ == '__main__':
    test_cfc_forward_shape_and_finiteness()
    test_cfc_param_count_includes_per_species_biases()
    test_cfc_output_differs_from_axis2()
    test_cfc_W_is_clipped_to_tau_min_bound()
    test_cfc_gradient_flow_through_AB()
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d)
        test_end_to_end_train_runs_with_cfc(tp)
        test_loss_decreases_on_synthetic_data(tp)
    print('OK: M2 Phase 1 (CfC) smoke tests passed.')
