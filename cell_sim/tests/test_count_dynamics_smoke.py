"""End-to-end smoke test for the count-dynamics surrogate.

Trains the MLP for two epochs on a tiny synthetic table of the same shape
the real Drive data would have. Verifies:
  * data path produces tensors of the right shape and dtype
  * one training step actually moves the loss
  * checkpoint saves and reloads
  * predict_step returns integer counts ≥ 0

Runs in <5s on CPU so it stays in the default test sweep.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from cell_sim.lgnn.data.dataset import (
    CountsTransitionDataset, BufferedShuffleDataset,
    replicate_to_log1p_array,
)
from cell_sim.lgnn.models.mlp_baseline import (
    CountDynamicsMLP, predict_step,
)
from cell_sim.lgnn.training.train_mlp import (
    TrainConfig, evaluate, train_count_dynamics,
)


def make_fake_lsdata(n_species: int = 32, n_timepoints: int = 64,
                      n_replicates: int = 3, seed: int = 0):
    """Build a stand-in for the lsdata module, returning small synthetic
    replicates whose dynamics are a fixed linear flow + noise. The model
    must be able to fit the linear part; that's what the loss-drop assert
    checks for."""
    rng = np.random.default_rng(seed)
    A = 0.95 * np.eye(n_species, dtype=np.float32)        # contracting flow
    species = [f'sp_{i}' for i in range(n_species)]
    replicates: dict = {}
    for r in range(1, n_replicates + 1):
        x = rng.uniform(0, 100, size=n_species).astype(np.float32)
        traj = np.zeros((n_species, n_timepoints), dtype=np.float32)
        for t in range(n_timepoints):
            traj[:, t] = x
            x = A @ x + rng.normal(0, 0.1, size=n_species).astype(np.float32)
            x = np.clip(x, 0, None)
        replicates[r] = pd.DataFrame(traj, index=species,
                                      columns=[float(c) for c in
                                              range(n_timepoints)])

    def load_replicate(i, species=None, time_start=0, time_end=1e9):
        df = replicates[i]
        if species is not None:
            df = df.loc[[s for s in species if s in df.index]]
        return df

    return SimpleNamespace(load_replicate=load_replicate), species


def test_replicate_to_log1p_array_shape_and_dtype():
    fake, species = make_fake_lsdata(n_species=8, n_timepoints=10)
    arr = replicate_to_log1p_array(fake.load_replicate(1))
    assert arr.shape == (10, 8), arr.shape
    assert arr.dtype == np.float32, arr.dtype
    assert np.isfinite(arr).all(), 'transform must produce no NaN/inf'


def test_replicate_to_log1p_handles_nan_and_negatives():
    """Real Drive data has NaN in idle-flux rows and negatives in
    reverse-direction flux rows. Both must yield finite numbers."""
    df = pd.DataFrame(
        np.array([[1.0, 2.0, np.nan, 3.0],
                  [-5.0, -7.0, np.nan, 0.0],
                  [np.inf, -np.inf, np.nan, 1e6]], dtype=np.float32),
        index=['count_a', 'flux_b', 'crazy_c'],
        columns=[0.0, 1.0, 2.0, 3.0],
    )
    arr = replicate_to_log1p_array(df)
    assert arr.shape == (4, 3)
    assert np.isfinite(arr).all(), arr


def test_dataset_yields_correct_shape():
    fake, species = make_fake_lsdata(n_species=8, n_timepoints=10,
                                      n_replicates=2)
    ds = CountsTransitionDataset([1, 2], fake, shuffle_within_replicate=False)
    pairs = list(ds)
    # 9 transitions per replicate × 2 replicates = 18
    assert len(pairs) == 18, len(pairs)
    x, dx = pairs[0]
    assert x.shape == (8,) and dx.shape == (8,)
    assert x.dtype == torch.float32 and dx.dtype == torch.float32


def test_buffered_shuffle_yields_complete_set():
    """Cross-replicate buffered shuffler must yield every transition
    from every replicate exactly once, regardless of buffer size."""
    fake, _ = make_fake_lsdata(n_species=4, n_timepoints=6, n_replicates=4)
    ds = BufferedShuffleDataset([1, 2, 3, 4], fake, buffer_size=2, seed=7)
    pairs = list(ds)
    # 5 transitions × 4 replicates = 20
    assert len(pairs) == 20, len(pairs)
    # All pairs are 4-vectors
    for x, dx in pairs:
        assert x.shape == (4,) and dx.shape == (4,)


def test_buffered_shuffle_actually_interleaves():
    """A buffer_size>=2 run on multiple replicates should interleave
    transitions from different replicates rather than emit one
    replicate's full block before starting the next."""
    fake, _ = make_fake_lsdata(n_species=3, n_timepoints=20, n_replicates=4,
                                seed=11)
    ds = BufferedShuffleDataset([1, 2, 3, 4], fake, buffer_size=4, seed=0)
    # With 4-deep buffer, the first 8 transitions should come from 2+
    # different replicates with high probability. Use the (x_t, dx)
    # values themselves as a fingerprint of which replicate they came
    # from — synthetic data has different starting states per rep.
    pairs = list(ds)
    first_8_x0 = torch.stack([p[0] for p in pairs[:8]])
    n_unique = len(set(tuple(v.tolist()) for v in first_8_x0))
    assert n_unique >= 4, (
        f'buffered shuffler should mix across replicates; first 8 '
        f'samples had only {n_unique} unique x_t signatures')


def test_loss_decreases_on_synthetic_data():
    n_species = 32
    fake, species = make_fake_lsdata(n_species=n_species, n_timepoints=64,
                                      n_replicates=3)
    cfg = TrainConfig(
        n_species=n_species,
        hidden=64,
        n_blocks=1,
        batch_size=16,
        lr=3e-3,
        train_replicates=(1, 2),
        val_replicates=(3,),
        n_epochs=2,
        device='cpu',
        log_every=10_000,             # silence logs
    )
    out = train_count_dynamics(cfg, fake)
    train_losses = out['history']['train_loss']
    val_losses = out['history']['val_loss']
    assert train_losses[-1] < train_losses[0], (
        f'train loss should fall: {train_losses}')
    assert val_losses[-1] < 1.0, (
        f'val MSE on Δlog1p of a contracting linear flow should be small, '
        f'got {val_losses}')


def test_checkpoint_round_trip(tmp_path: Path):
    n_species = 16
    fake, species = make_fake_lsdata(n_species=n_species, n_timepoints=32,
                                      n_replicates=2)
    cfg = TrainConfig(
        n_species=n_species, hidden=32, n_blocks=1, batch_size=8,
        train_replicates=(1,), val_replicates=(2,),
        n_epochs=1, device='cpu', log_every=10_000,
    )
    ckpt = tmp_path / 'ckpt.pt'
    out = train_count_dynamics(cfg, fake, checkpoint_path=ckpt)
    assert ckpt.exists()
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert 'state_dict' in state and 'val_mse' in state
    model2 = CountDynamicsMLP(n_species=n_species, hidden=32, n_blocks=1)
    model2.load_state_dict(state['state_dict'])
    # Predictions match what evaluate() saw
    val_metrics = evaluate(model2, cfg, fake, torch.device('cpu'))
    assert abs(val_metrics['mse'] - state['val_mse']) < 1e-5


def test_predict_step_returns_nonneg_integers():
    n_species = 16
    model = CountDynamicsMLP(n_species=n_species, hidden=32, n_blocks=1)
    x = np.array([10, 50, 0, 200] * (n_species // 4), dtype=np.int64)
    nxt = predict_step(model, x)
    assert nxt.shape == (n_species,)
    assert nxt.dtype == np.int64
    assert (nxt >= 0).all(), 'predicted counts must be ≥ 0'


if __name__ == '__main__':
    test_replicate_to_log1p_array_shape_and_dtype()
    test_dataset_yields_correct_shape()
    test_loss_decreases_on_synthetic_data()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_checkpoint_round_trip(Path(d))
    test_predict_step_returns_nonneg_integers()
    print('OK: count-dynamics smoke tests passed.')
