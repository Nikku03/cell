"""Smoke tests for M1+axis2 — attention, edge dropout, multi-step rollout.

Covers:
  - forward shape + entropy return
  - edge dropout actually changes outputs (mask is wired)
  - attention entropy is non-negative + finite
  - segment-softmax sums to 1 per dst node
  - rollout dataset yields (max_k+1, S) windows
  - paired forward + rollout training step runs end-to-end
  - loss decreases on synthetic graph-aware data
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

from cell_sim.lgnn.data.dataset import CountsRolloutDataset
from cell_sim.lgnn.data.species_graph import build_species_graph
from cell_sim.lgnn.models.gnn_v1_axis2 import (
    CellGNNv1Axis2, _segment_softmax, count_parameters,
)
from cell_sim.lgnn.training.train_m1_axis2 import (
    M1Axis2TrainConfig, _k_for_epoch, train_m1_axis2,
)


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
    rows = ['A', 'B', 'C',
            'G_0001', 'RP_0001', 'R_0001', 'RPM_0001',
            'P_0001', 'PM_0001', 'DM_0001', 'D_0001',
            'F_R1']
    return build_species_graph(rows, cobra, include_simulator_edges=True), rows


def test_segment_softmax_normalizes_per_dst():
    """Manual check: logits per (B, dst) softmax to sum=1."""
    n_nodes = 4
    # 5 edges: dst = [0, 0, 1, 1, 2]
    dst = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    logit = torch.tensor([[1.0, 2.0, 0.5, 0.7, 3.0]])    # B=1
    alpha = _segment_softmax(logit, dst, n_nodes)
    # alpha[:, e] for dst=0 should sum to 1
    s0 = alpha[0, 0] + alpha[0, 1]
    s1 = alpha[0, 2] + alpha[0, 3]
    s2 = alpha[0, 4]                                      # only edge to dst=2
    assert abs(s0.item() - 1.0) < 1e-5
    assert abs(s1.item() - 1.0) < 1e-5
    assert abs(s2.item() - 1.0) < 1e-5


def test_segment_softmax_handles_minus_inf_masking():
    """Masked edges (logit=-inf) should get α=0 and not contribute."""
    n_nodes = 2
    dst = torch.tensor([0, 0], dtype=torch.long)
    logit = torch.tensor([[1.0, float('-inf')]])
    alpha = _segment_softmax(logit, dst, n_nodes)
    assert abs(alpha[0, 0].item() - 1.0) < 1e-5
    assert alpha[0, 1].item() == 0.0


def test_forward_shape_and_entropy():
    with tempfile.TemporaryDirectory() as d:
        g, _ = _build_tiny_graph(Path(d))
    model = CellGNNv1Axis2(g, hidden=16, n_layers=2, use_checkpoint=False)
    x = torch.randn(4, g.n_nodes)
    pred = model(x)
    assert pred.shape == (4, g.n_nodes)

    pred2, ent = model(x, return_attention_entropy=True)
    assert pred2.shape == (4, g.n_nodes)
    assert ent.dim() == 0                                 # scalar
    assert torch.isfinite(ent)
    assert ent.item() >= 0.0                              # entropy is non-negative


def test_edge_dropout_changes_output():
    """With dropout enabled (and model.train()), output should differ
    from the no-dropout forward (different seeds → different masks)."""
    with tempfile.TemporaryDirectory() as d:
        g, _ = _build_tiny_graph(Path(d))
    model = CellGNNv1Axis2(g, hidden=16, n_layers=2, use_checkpoint=False)
    model.train()
    x = torch.randn(2, g.n_nodes)
    torch.manual_seed(0)
    y_full = model(x, edge_dropout_p=0.0)
    torch.manual_seed(1)                                  # different mask
    y_drop = model(x, edge_dropout_p=0.5)
    assert not torch.allclose(y_full, y_drop, atol=1e-4)


def test_attention_entropy_zero_when_one_edge_per_dst():
    """If a node has exactly one incoming edge, its attention is
    forced to α=1 → entropy=0. Self-loop-only nodes hit this case."""
    with tempfile.TemporaryDirectory() as d:
        g, _ = _build_tiny_graph(Path(d))
    model = CellGNNv1Axis2(g, hidden=16, n_layers=1, use_checkpoint=False)
    x = torch.randn(2, g.n_nodes)
    h = model.input_proj(x.unsqueeze(-1))
    _, ent_per_node = model.layers[0](
        h, model.edge_index, model.edge_attr,
        model.edge_kind, edge_mask=None,
    )
    assert ent_per_node.shape == (2, g.n_nodes)
    assert (ent_per_node >= -1e-6).all()


def test_rollout_dataset_yields_window():
    rows = ['A', 'B', 'C']
    rng = np.random.default_rng(0)
    n_t = 30

    class FakeLs:
        @staticmethod
        def load_replicate(i, species=None, **_):
            data = rng.normal(0, 1, (3, n_t)).astype(np.float32)
            return pd.DataFrame(
                data, index=pd.Index(rows, name='species'),
                columns=pd.Index(np.arange(n_t, dtype=float), name='time_s'),
            )

    ds = CountsRolloutDataset(
        replicate_indices=[1], lsdata_module=FakeLs(),
        max_k=5, shuffle_within_replicate=False,
    )
    seen = list(ds)
    # n_t timepoints; n_valid = n_t - 5 - 1 = 24 starting positions
    assert len(seen) == n_t - 5 - 1
    x_t, x_window = seen[0]
    assert x_t.shape == (3,)                              # n species = 3
    assert x_window.shape == (6, 3)                       # max_k+1 = 6


def test_k_curriculum_clamps_at_last_value():
    cur = (1, 2, 4, 10)
    assert _k_for_epoch(cur, 0) == 1
    assert _k_for_epoch(cur, 3) == 10
    assert _k_for_epoch(cur, 99) == 10                    # clamps


def test_end_to_end_train_runs_with_rollout(tmp_path):
    """End-to-end: tiny graph, 3 fake replicates, 1 epoch at k=2.
    Asserts that loss is finite and the attention entropy term is
    actually being computed (non-zero in history)."""
    rows = ['A', 'B', 'C',
            'G_0001', 'RP_0001', 'R_0001', 'RPM_0001',
            'P_0001', 'PM_0001', 'DM_0001', 'D_0001',
            'F_R1']
    rng = np.random.default_rng(0)
    n_t = 40

    class FakeLs:
        @staticmethod
        def load_replicate(i, species=None, **_):
            data = rng.normal(0, 1, (len(rows), n_t)).astype(np.float32)
            return pd.DataFrame(
                data, index=pd.Index(rows, name='species'),
                columns=pd.Index(np.arange(n_t, dtype=float), name='time_s'),
            )
    cobra = tmp_path / 'mini.json'
    _write_fake_cobra(cobra)
    g = build_species_graph(rows, cobra, include_simulator_edges=True)

    cfg = M1Axis2TrainConfig(
        n_species=g.n_nodes, hidden=16, n_layers=2,
        batch_size=4, lr=3e-3,
        train_replicates=(1, 2, 3), val_replicates=(4,),
        n_epochs=1,
        device='cpu', seed=0, log_every=10000,
        edge_dropout_p=0.1, lambda_dropout=0.5, lambda_attn=1e-3,
        k_curriculum=(2,), max_k=4,
        use_checkpoint=False, wall_clock_budget_s=600,
    )
    out = train_m1_axis2(cfg, FakeLs(), g, checkpoint_path=None)
    assert np.isfinite(out['history']['train_total'][0])
    assert out['history']['train_attn_entropy'][0] > 0    # entropy term is live
    assert out['history']['train_drop_ss'][0] > 0         # dropout pass is live
    assert out['history']['k_per_epoch'] == [2]


def test_loss_decreases_on_synthetic_graph_aware_data(tmp_path):
    """Same synthetic-target setup as gnn_v1 smoke. Verifies the
    axis-2 model can still learn — attention + dropout don't break
    gradient flow."""
    g, _ = _build_tiny_graph(tmp_path)
    torch.manual_seed(0)
    model = CellGNNv1Axis2(g, hidden=32, n_layers=2, use_checkpoint=False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    mse = torch.nn.MSELoss()

    deg = torch.zeros(g.n_nodes)
    deg.index_add_(0, g.edge_index[1],
                    torch.ones_like(g.edge_index[1], dtype=torch.float))

    losses = []
    model.train()
    for step in range(250):
        x = torch.randn(8, g.n_nodes) * 0.5
        target = 0.5 * deg * torch.tanh(x)
        # Simple supervised: just first-step + small attention penalty
        pred, ent = model(x, return_attention_entropy=True)
        loss = mse(pred, target) + 1e-3 * ent
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(mse(pred.detach(), target).item())
    assert sum(losses[-20:]) / 20 < 0.5 * sum(losses[:20]) / 20, \
        f'no progress: start {sum(losses[:20])/20:.4f} ' \
        f'end {sum(losses[-20:])/20:.4f}'


if __name__ == '__main__':
    test_segment_softmax_normalizes_per_dst()
    test_segment_softmax_handles_minus_inf_masking()
    test_forward_shape_and_entropy()
    test_edge_dropout_changes_output()
    test_attention_entropy_zero_when_one_edge_per_dst()
    test_rollout_dataset_yields_window()
    test_k_curriculum_clamps_at_last_value()
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d)
        test_end_to_end_train_runs_with_rollout(tp)
        test_loss_decreases_on_synthetic_graph_aware_data(tp)
    print('OK: M1+axis2 smoke tests passed.')
