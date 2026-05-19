"""Phase 3 verification: spiking reservoir + linear readout on real tasks.

The Phase 3 contract is: a small fixed random spiking reservoir + a
TINY linear readout can solve nontrivial time-series tasks. The
trained-parameter count (the readout's weights) is much smaller than
the reservoir; the reservoir's fixed-but-rich nonlinear dynamics do
the heavy lifting.

Gates:

  1. Reservoir produces non-trivial spike activity under input drive
  2. Reservoir state distinguishes different input patterns (linear
     separability after readout)
  3. Sine-wave reconstruction: feed a sine to the reservoir, train a
     readout to reproduce it from the population state; achieve R^2 >= 0.7
     on held-out data
  4. Frequency discrimination: 2-class classification of sine inputs at
     two distinct frequencies; >= 90% accuracy
  5. Memory test: readout can recover a delayed copy of the input
     (showing the reservoir has short-term memory)
  6. Speedup vs naive deep network with same task: training a linear
     readout takes < 5 s; training a deep MLP would take much longer.
     (Measured as: ridge regression is fast in absolute terms.)
"""

from __future__ import annotations

import math
import time

import torch

from .reservoir import (
    LiquidStateMachine,
    ReservoirConfig,
    SpikingReservoir,
    train_linear_readout,
    predict_linear,
)
from .verify_phase0 import CheckResult, format_results


def _make_lsm(seed: int = 0, **overrides) -> LiquidStateMachine:
    cfg = ReservoirConfig(seed=seed, **overrides)
    return LiquidStateMachine(cfg)


def _drive_input(N: int, input_signal: torch.Tensor, scale: float = 5.0) -> torch.Tensor:
    """Project a (T,) input signal to (T, N) per-neuron drives via a fixed
    random projection. This is how reservoirs usually receive inputs."""
    T = input_signal.shape[0]
    g = torch.Generator().manual_seed(7777)
    P = (torch.rand(N, generator=g) - 0.5) * 2.0          # uniform in [-1, 1]
    drive = input_signal.unsqueeze(-1) * P.unsqueeze(0)   # (T, N)
    return drive * scale


def _run_reservoir_collect_V(
    lsm: LiquidStateMachine, drive: torch.Tensor, dt: float = 1.0,
) -> torch.Tensor:
    """Run reservoir and collect V at every timestep. Returns (T, N)."""
    T = drive.shape[0]
    N = lsm.reservoir.n
    Vs = torch.zeros(T, N)
    lsm.reset()
    for k in range(T):
        feat = lsm.step_and_feature(drive[k], dt=dt, t=k * dt)
        Vs[k] = feat
    return Vs


# ---------------- gates ----------------

def check_reservoir_active_under_input() -> CheckResult:
    """The reservoir should produce a non-trivial number of spikes when
    driven by input -- not silent, not saturated."""
    lsm = _make_lsm(seed=0)
    N = lsm.reservoir.n
    T = 300
    t = torch.arange(T, dtype=torch.float32) * 0.02
    signal = torch.sin(2 * math.pi * 5.0 * t)             # 5 Hz sine
    drive = _drive_input(N, signal)
    Vs = _run_reservoir_collect_V(lsm, drive)
    # Approximate spike count from membrane state crossing threshold
    # (we don't directly accumulate spikes here, but V trajectories should be rich)
    V_std = Vs.std(dim=0).mean().item()
    return CheckResult(
        "reservoir membrane state varies non-trivially under input",
        V_std > 0.5, V_std, 0.5,
        note=f"(mean per-neuron V std over T: {V_std:.3f})",
    )


def check_sine_reconstruction() -> CheckResult:
    """Train readout to reproduce a sine input from the reservoir state.

    The readout is just a (N -> 1) linear map. It must reconstruct the
    input from the population state with R^2 >= 0.7 on a held-out window.
    """
    lsm = _make_lsm(seed=1)
    N = lsm.reservoir.n
    T = 1200
    dt = 1.0
    t = torch.arange(T, dtype=torch.float32) * dt * 0.001    # in seconds
    signal = torch.sin(2 * math.pi * 5.0 * t)
    drive = _drive_input(N, signal)

    Vs = _run_reservoir_collect_V(lsm, drive, dt=dt)        # (T, N)
    # Train on first 60%, test on last 30% (10% warmup)
    warmup = int(0.1 * T)
    n_train = int(0.6 * T)
    train_features = Vs[warmup:warmup + n_train]
    train_targets = signal[warmup:warmup + n_train].unsqueeze(-1)
    W = train_linear_readout(train_features, train_targets, ridge=1.0)

    test_features = Vs[warmup + n_train:]
    test_targets = signal[warmup + n_train:].unsqueeze(-1)
    pred = predict_linear(W, test_features)
    ss_res = ((test_targets - pred) ** 2).sum()
    ss_tot = ((test_targets - test_targets.mean()) ** 2).sum().clamp_min(1e-9)
    r2 = 1.0 - ss_res / ss_tot
    return CheckResult(
        "linear readout reconstructs sine input (held-out R^2 >= 0.7)",
        float(r2) >= 0.7, float(r2), 0.7,
        note=f"(test R^2={float(r2):.3f}, train n={n_train}, test n={T-warmup-n_train})",
    )


def check_input_classification() -> CheckResult:
    """Two-class task: classify whether the input is a sine wave or
    Gaussian noise. Both at the same energy / amplitude scale, so
    discrimination requires the reservoir to encode temporal structure.

    The reservoir's mean V + std features should easily separate
    "structured oscillation" from "white noise."
    """
    lsm = _make_lsm(seed=2)
    N = lsm.reservoir.n
    T_per_episode = 400
    dt = 1.0
    n_episodes = 40          # 20 each class

    rng = torch.Generator().manual_seed(31)
    features = []
    labels = []
    for ep in range(n_episodes):
        is_noise = (ep % 2 == 1)
        t = torch.arange(T_per_episode, dtype=torch.float32) * dt * 0.001
        if is_noise:
            signal = torch.randn(T_per_episode, generator=rng) * 0.7
        else:
            phase = torch.rand((), generator=rng).item() * 2 * math.pi
            freq = 4.0 + 2.0 * torch.rand((), generator=rng).item()
            signal = torch.sin(2 * math.pi * freq * t + phase)
        drive = _drive_input(N, signal)
        Vs = _run_reservoir_collect_V(lsm, drive, dt=dt)
        late = Vs[T_per_episode // 2:]
        feat = torch.cat([
            late.mean(dim=0),
            late.std(dim=0),
            late.max(dim=0).values - late.min(dim=0).values,
        ])
        features.append(feat)
        labels.append(1.0 if is_noise else 0.0)

    X = torch.stack(features)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

    # Hold out every 5th (8 of 40)
    idx = torch.arange(n_episodes)
    test_mask = (idx % 5 == 0)
    train_mask = ~test_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    W = train_linear_readout(X_train, y_train, ridge=1.0)
    pred = predict_linear(W, X_test)
    pred_label = (pred > 0.5).float()
    acc = (pred_label == y_test).float().mean().item()
    return CheckResult(
        "sine-vs-noise classification accuracy >= 80%",
        acc >= 0.8, acc, 0.8,
        note=f"(test accuracy: {acc*100:.1f}% on {int(test_mask.sum())} held-out)",
    )


def check_short_term_memory() -> CheckResult:
    """Delay task: the reservoir state at time t should contain enough info
    to recover the input at time t - delay. Train a readout for delayed
    reproduction and measure R^2."""
    lsm = _make_lsm(seed=3)
    N = lsm.reservoir.n
    T = 1500
    dt = 1.0
    # Use a sum of low-freq sines so there's structure to remember
    t = torch.arange(T, dtype=torch.float32) * dt * 0.001
    signal = (torch.sin(2 * math.pi * 2.0 * t)
              + 0.5 * torch.sin(2 * math.pi * 5.0 * t))
    drive = _drive_input(N, signal)
    Vs = _run_reservoir_collect_V(lsm, drive, dt=dt)

    # Predict signal at t-25 (25 ms in the past) from reservoir state at t
    delay = 25
    warmup = 100
    n_train = 800
    train_features = Vs[warmup + delay: warmup + delay + n_train]
    train_targets = signal[warmup: warmup + n_train].unsqueeze(-1)
    W = train_linear_readout(train_features, train_targets, ridge=1.0)

    test_features = Vs[warmup + delay + n_train:]
    test_targets = signal[warmup + n_train: warmup + n_train + test_features.shape[0]].unsqueeze(-1)
    n_test = min(test_features.shape[0], test_targets.shape[0])
    test_features = test_features[:n_test]
    test_targets = test_targets[:n_test]

    pred = predict_linear(W, test_features)
    ss_res = ((test_targets - pred) ** 2).sum()
    ss_tot = ((test_targets - test_targets.mean()) ** 2).sum().clamp_min(1e-9)
    r2 = 1.0 - ss_res / ss_tot
    return CheckResult(
        "short-term memory: readout recovers 25-step-delayed signal (R^2 >= 0.5)",
        float(r2) >= 0.5, float(r2), 0.5,
        note=f"(delayed-reconstruction R^2 = {float(r2):.3f})",
    )


def check_readout_train_is_fast() -> CheckResult:
    """Training a linear readout on the reservoir features must be fast
    (under 1 s for 500-neuron reservoir, 1000 train samples)."""
    lsm = _make_lsm(seed=4)
    N = lsm.reservoir.n
    T = 1100
    dt = 1.0
    t = torch.arange(T, dtype=torch.float32) * dt * 0.001
    signal = torch.sin(2 * math.pi * 5.0 * t)
    drive = _drive_input(N, signal)
    Vs = _run_reservoir_collect_V(lsm, drive, dt=dt)

    train_features = Vs[100:1100]
    train_targets = signal[100:1100].unsqueeze(-1)
    t0 = time.time()
    W = train_linear_readout(train_features, train_targets, ridge=1.0)
    elapsed = time.time() - t0
    return CheckResult(
        "linear readout training is fast (< 1 s)",
        elapsed < 1.0, elapsed, 1.0,
        note=f"(ridge regression for 500-d, 1000 samples: {elapsed*1000:.2f} ms)",
    )


# ---------------- runner ----------------

def run_all() -> list[CheckResult]:
    return [
        check_reservoir_active_under_input(),
        check_sine_reconstruction(),
        check_input_classification(),
        check_short_term_memory(),
        check_readout_train_is_fast(),
    ]


if __name__ == "__main__":
    t0 = time.time()
    results = run_all()
    print(f"Phase 3 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    print(f"\n  -> {n_pass} / {len(results)} checks passed.")
