"""Smoke tests for the fast M1+axis2 training path.

Verifies preload-to-GPU semantics, on-GPU index iterator,
fancy-index window gathering, and end-to-end training run on
synthetic data without DataLoader.
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

from cell_sim.lgnn.data.species_graph import build_species_graph
from cell_sim.lgnn.training.train_m1_axis2_fast import (
    M1Axis2FastTrainConfig, _gather_windows, _index_iterator,
    preload_to_gpu, train_m1_axis2_fast,
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


def _make_fake_lsdata(rows, n_t):
    rng = np.random.default_rng(0)

    class FakeLs:
        @staticmethod
        def load_replicate(i, species=None, **_):
            data = rng.normal(0, 1, (len(rows), n_t)).astype(np.float32)
            return pd.DataFrame(
                data, index=pd.Index(rows, name='species'),
                columns=pd.Index(np.arange(n_t, dtype=float), name='time_s'),
            )

    return FakeLs()


def test_gather_windows_shape_and_correctness():
    """Verify _gather_windows extracts the right slice from (R, T, S)."""
    R, T, S, B, K = 3, 20, 5, 4, 3
    data = torch.arange(R * T * S, dtype=torch.float32).reshape(R, T, S)
    rep_idx = torch.tensor([0, 1, 2, 0])
    t_idx   = torch.tensor([0, 5, 2, 10])
    out = _gather_windows(data, rep_idx, t_idx, K + 1)
    assert out.shape == (B, K + 1, S)
    # Spot-check: out[0, 0, :] should equal data[0, 0, :]
    assert torch.equal(out[0, 0], data[0, 0])
    # out[1, 2, :] should equal data[1, 5+2, :] = data[1, 7, :]
    assert torch.equal(out[1, 2], data[1, 7])
    # out[2, 3, :] = data[2, 2+3, :] = data[2, 5, :]
    assert torch.equal(out[2, 3], data[2, 5])


def test_index_iterator_covers_all_pairs_once():
    """Permutation should yield every (rep, t) pair exactly once."""
    R, n_valid, B = 3, 7, 4
    gen = torch.Generator(device='cpu').manual_seed(0)
    seen = set()
    n_yielded = 0
    for rep_b, t_b in _index_iterator(R, n_valid, B, gen,
                                       device=torch.device('cpu')):
        for r, t in zip(rep_b.tolist(), t_b.tolist()):
            assert (r, t) not in seen, f'duplicate ({r},{t})'
            seen.add((r, t))
            n_yielded += 1
    assert n_yielded == R * n_valid
    assert len(seen) == R * n_valid


def test_preload_to_gpu_concatenates_replicates_in_order():
    """Replicate i becomes data[i]; order matches replicate_indices arg.
    Uses an idempotent fake (load_replicate(i) returns data deterministic
    in i) so we can verify the preload's row-ordering is correct."""
    rows = ['A', 'B', 'C']
    n_t = 10

    class IdempotentFake:
        @staticmethod
        def load_replicate(i, species=None, **_):
            # Each replicate's data is constant value = i. After
            # signed-log1p the value becomes log1p(i).
            data = np.full((len(rows), n_t), float(i), dtype=np.float32)
            return pd.DataFrame(
                data, index=pd.Index(rows, name='species'),
                columns=pd.Index(np.arange(n_t, dtype=float), name='time_s'),
            )

    big = preload_to_gpu(
        replicate_indices=[5, 7, 2],
        lsdata_module=IdempotentFake(),
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    assert big.shape == (3, n_t, len(rows))
    assert big.dtype == torch.float32
    # row 0 is replicate 5 → value log1p(5)
    expected_5 = np.log1p(5.0)
    assert torch.allclose(big[0], torch.full_like(big[0], expected_5))
    # row 1 is replicate 7 → value log1p(7)
    expected_7 = np.log1p(7.0)
    assert torch.allclose(big[1], torch.full_like(big[1], expected_7))
    # row 2 is replicate 2 → value log1p(2)
    expected_2 = np.log1p(2.0)
    assert torch.allclose(big[2], torch.full_like(big[2], expected_2))


def test_end_to_end_fast_training_runs(tmp_path):
    """Tiny graph + 3 fake replicates + 1 epoch — verifies the whole
    fast training path runs and produces finite losses."""
    rows = ['A', 'B', 'C',
            'G_0001', 'RP_0001', 'R_0001', 'RPM_0001',
            'P_0001', 'PM_0001', 'DM_0001', 'D_0001',
            'F_R1']
    fake = _make_fake_lsdata(rows, n_t=40)
    cobra = tmp_path / 'mini.json'
    _write_fake_cobra(cobra)
    g = build_species_graph(rows, cobra, include_simulator_edges=True)

    cfg = M1Axis2FastTrainConfig(
        n_species=g.n_nodes, hidden=16, n_layers=2,
        batch_size=4, lr=3e-3,
        train_replicates=(1, 2, 3), val_replicates=(4,),
        n_epochs=1,
        device='cpu',           # CPU run for the test (no GPU in CI)
        seed=0, log_every=10000,
        edge_dropout_p=0.1, lambda_dropout=0.5, lambda_attn=1e-3,
        k_curriculum=(2,), max_k=2,
        use_checkpoint=False,
        use_bf16=False,         # cpu doesn't support bf16 autocast cleanly
        preload_dtype='float32',
        wall_clock_budget_s=600,
    )
    out = train_m1_axis2_fast(cfg, fake, g, checkpoint_path=None)
    assert np.isfinite(out['history']['train_total'][0])
    assert out['history']['train_attn_entropy'][0] > 0
    assert out['history']['train_drop_ss'][0] > 0
    assert out['history']['k_per_epoch'] == [2]
    assert out['history']['samples_per_sec'][0] > 0


def test_loss_decreases_on_synthetic_data(tmp_path):
    """Same gradient-flow check as the slow variant — must learn."""
    rows = ['A', 'B', 'C',
            'G_0001', 'RP_0001', 'R_0001', 'RPM_0001',
            'P_0001', 'PM_0001', 'DM_0001', 'D_0001',
            'F_R1']
    n_t = 60
    rng = np.random.default_rng(0)

    class FakeLs:
        @staticmethod
        def load_replicate(i, species=None, **_):
            # Smoothly varying data so the dynamics are learnable
            base = rng.normal(0, 1, (len(rows), 1))
            t = np.arange(n_t).reshape(1, -1) / n_t
            data = (base + 0.3 * np.sin(2 * np.pi * t * (1 + i * 0.1))).astype(np.float32)
            return pd.DataFrame(
                data, index=pd.Index(rows, name='species'),
                columns=pd.Index(np.arange(n_t, dtype=float), name='time_s'),
            )

    cobra = tmp_path / 'mini.json'
    _write_fake_cobra(cobra)
    g = build_species_graph(rows, cobra, include_simulator_edges=True)

    cfg = M1Axis2FastTrainConfig(
        n_species=g.n_nodes, hidden=24, n_layers=2,
        batch_size=8, lr=3e-3,
        train_replicates=(1, 2, 3, 4), val_replicates=(5,),
        n_epochs=2,
        device='cpu', seed=0, log_every=10000,
        edge_dropout_p=0.0, lambda_dropout=0.0, lambda_attn=0.0,
        k_curriculum=(1, 1), max_k=1,
        use_checkpoint=False, use_bf16=False, preload_dtype='float32',
        wall_clock_budget_s=600,
    )
    out = train_m1_axis2_fast(cfg, FakeLs(), g, checkpoint_path=None)
    # full_ss must drop epoch 0 → epoch 1
    assert out['history']['train_full_ss'][1] < out['history']['train_full_ss'][0], \
        out['history']['train_full_ss']


if __name__ == '__main__':
    test_gather_windows_shape_and_correctness()
    test_index_iterator_covers_all_pairs_once()
    test_preload_to_gpu_concatenates_replicates_in_order()
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d)
        test_end_to_end_fast_training_runs(tp)
        test_loss_decreases_on_synthetic_data(tp)
    print('OK: M1+axis2 fast smoke tests passed.')
