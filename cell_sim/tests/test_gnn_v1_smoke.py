"""Smoke tests for the M1 hetero-edge GNN.

Builds a tiny graph (3 SBML metabolites + 3 chain rows + 1 flux row),
verifies forward shape, gradient flow, parameter count, checkpoint
round-trip, and that loss decreases on synthetic data.

The synthetic data is constructed so the GNN's task is learnable in
a few hundred steps: target dx is a fixed linear function of input x
masked through the graph adjacency. If the model can't drop the loss
on this, message passing is wired wrong.
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

from cell_sim.lgnn.data.species_graph import (
    build_species_graph, EdgeKind,
)
from cell_sim.lgnn.models.gnn_v1 import CellGNNv1, count_parameters
from cell_sim.lgnn.training.train_gnn import GNNTrainConfig, train_gnn


def _write_fake_cobra(path: Path):
    model = {
        'metabolites': [
            {'id': 'A', 'compartment': 'c'},
            {'id': 'B', 'compartment': 'c'},
            {'id': 'C', 'compartment': 'c'},
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
    # Full central-dogma chain so every simulator EdgeKind is exercised:
    # G ↔ RP ↔ R ↔ RPM ↔ {P, PM},  P ↔ PM (translocation),
    # R ↔ DM ↔ D (degradation). Plus F_R1 for FLUX_COUPLING.
    rows = ['A', 'B', 'C',
            'G_0001', 'RP_0001', 'R_0001', 'RPM_0001',
            'P_0001', 'PM_0001', 'DM_0001', 'D_0001',
            'F_R1']
    return build_species_graph(rows, cobra, include_simulator_edges=True)


def test_forward_shape_matches_input():
    with tempfile.TemporaryDirectory() as d:
        g = _build_tiny_graph(Path(d))
    model = CellGNNv1(graph=g, hidden=16, n_layers=2)
    x = torch.randn(4, g.n_nodes)
    y = model(x)
    assert y.shape == (4, g.n_nodes), y.shape


def test_backward_produces_gradients_on_every_param():
    with tempfile.TemporaryDirectory() as d:
        g = _build_tiny_graph(Path(d))
    model = CellGNNv1(graph=g, hidden=16, n_layers=2, use_checkpoint=False)
    model.train()
    x = torch.randn(2, g.n_nodes, requires_grad=False)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f'no grad for {name}'
        assert torch.isfinite(p.grad).all(), f'non-finite grad for {name}'


def test_checkpointing_matches_non_checkpointed_forward():
    """checkpoint vs no-checkpoint forward must agree to fp32 noise."""
    with tempfile.TemporaryDirectory() as d:
        g = _build_tiny_graph(Path(d))
    torch.manual_seed(0)
    m1 = CellGNNv1(graph=g, hidden=16, n_layers=2, use_checkpoint=True)
    torch.manual_seed(0)
    m2 = CellGNNv1(graph=g, hidden=16, n_layers=2, use_checkpoint=False)
    x = torch.randn(3, g.n_nodes)
    # Both in train mode (only train mode triggers the checkpoint branch)
    m1.train(); m2.train()
    y1 = m1(x)
    y2 = m2(x)
    assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-5), (y1 - y2).abs().max()


def test_parameter_count_scales_with_hidden():
    """M1's param count should be O(hidden² × n_layers × N_EDGE_KINDS)."""
    with tempfile.TemporaryDirectory() as d:
        g = _build_tiny_graph(Path(d))
    p_h16 = count_parameters(CellGNNv1(graph=g, hidden=16, n_layers=2))
    p_h32 = count_parameters(CellGNNv1(graph=g, hidden=32, n_layers=2))
    # Rough check: doubling hidden roughly quadruples params (MLPs dominate)
    assert p_h32 > 3 * p_h16, (p_h16, p_h32)


def test_loss_decreases_on_synthetic_graph_aware_data():
    """Construct dx so it's a learnable linear function of the
    incoming-message structure. The GNN should drop loss substantially
    in 200 steps; if it doesn't, message passing is broken."""
    with tempfile.TemporaryDirectory() as d:
        g = _build_tiny_graph(Path(d))
    torch.manual_seed(0)
    model = CellGNNv1(graph=g, hidden=32, n_layers=2, use_checkpoint=False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.MSELoss()

    # Random "true" linear projection of incoming-degree-weighted x
    deg = torch.zeros(g.n_nodes)
    deg.index_add_(0, g.edge_index[1], torch.ones_like(g.edge_index[1],
                                                       dtype=torch.float))
    # Synthetic target: dx_i = 0.5 * deg_i * tanh(x_i)
    losses = []
    for step in range(250):
        x = torch.randn(8, g.n_nodes) * 0.5
        target = 0.5 * deg * torch.tanh(x)
        pred = model(x)
        loss = loss_fn(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    # Final 20-step mean should be < 25% of initial 20-step mean
    assert sum(losses[-20:]) / 20 < 0.25 * sum(losses[:20]) / 20, \
        f'no progress: start {sum(losses[:20])/20:.4f} end {sum(losses[-20:])/20:.4f}'


def test_checkpoint_roundtrip(tmp_path):
    g = _build_tiny_graph(tmp_path)
    m1 = CellGNNv1(graph=g, hidden=16, n_layers=2)
    pth = tmp_path / 'gnn.pt'
    torch.save({'state_dict': m1.state_dict()}, pth)
    m2 = CellGNNv1(graph=g, hidden=16, n_layers=2)
    m2.load_state_dict(torch.load(pth, weights_only=False)['state_dict'])
    x = torch.randn(2, g.n_nodes)
    m1.eval(); m2.eval()
    y1 = m1(x); y2 = m2(x)
    assert torch.allclose(y1, y2, atol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end training smoke test (uses fake lsdata module)
# ---------------------------------------------------------------------------
def _make_fake_lsdata(tmp: Path, n_replicates: int = 4,
                      n_timepoints: int = 32):
    g_rows = ['A', 'B', 'C',
              'G_0001', 'RP_0001', 'R_0001', 'RPM_0001',
              'P_0001', 'PM_0001', 'DM_0001', 'D_0001',
              'F_R1']
    rng = np.random.default_rng(0)

    class FakeLs:
        @staticmethod
        def load_replicate(i, species=None, **_):
            data = rng.normal(0, 1, (len(g_rows), n_timepoints)).astype(np.float32)
            df = pd.DataFrame(data,
                              index=pd.Index(g_rows, name='species'),
                              columns=pd.Index(np.arange(n_timepoints,
                                                          dtype=float),
                                                name='time_s'))
            return df
    return FakeLs(), g_rows


def test_end_to_end_train_runs(tmp_path):
    fake_ls, rows = _make_fake_lsdata(tmp_path)
    cobra = tmp_path / 'mini.json'
    _write_fake_cobra(cobra)
    g = build_species_graph(list(rows), cobra, include_simulator_edges=True)

    cfg = GNNTrainConfig(
        n_species=g.n_nodes,
        hidden=16, n_layers=2,
        batch_size=4, lr=3e-3,
        train_replicates=(1, 2, 3),
        val_replicates=(4,),
        n_epochs=1, device='cpu', seed=0, log_every=10000,
        use_checkpoint=False,         # synthetic data is too small
    )
    out = train_gnn(cfg, fake_ls, g, checkpoint_path=None)
    assert 'history' in out
    assert len(out['history']['train_loss']) == 1
    # Sanity: loss is a finite number
    assert np.isfinite(out['history']['train_loss'][0])


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d)
        test_forward_shape_matches_input()
        test_backward_produces_gradients_on_every_param()
        test_checkpointing_matches_non_checkpointed_forward()
        test_parameter_count_scales_with_hidden()
        test_loss_decreases_on_synthetic_graph_aware_data()
        test_checkpoint_roundtrip(tp)
        test_end_to_end_train_runs(tp)
    print('OK: M1 GNN smoke tests passed.')
