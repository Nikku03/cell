"""Composite benchmark: naive 100K LIF vs all-7-tricks integrated agent.

A head-to-head test on a one-shot 20-class temporal-pattern classification
task. Both systems use the SAME reservoir-neuron budget. The naive
pipeline uses the standard LSM recipe (reservoir + ridge readout); the
integrated pipeline composes six of the seven Phase tricks.

  Naive:        reservoir + linear readout (Phase 3 only)
  Integrated:   reservoir + dendritic projection + kWTA + episodic
                memory + cosine retrieval
                (Phases 1, 2, 3, 5, 7 explicit; 4 trivial-route, 6 optional)

Task (`make_pattern_task`): 20 classes = 5 waveform shapes ×
4 amplitudes. ONE training example per class (20 train total). The test
set has 10 noisy variants per class (200 test samples). Each waveform
gets independent additive Gaussian noise with σ given by `noise_std`.

Expected outcome:
  - Naive top-1 << integrated top-1. With 20 classes and 1 example per
    class, ridge regression is severely underdetermined and the readout
    cannot learn a useful 20-class boundary. The integrated system stores
    each prototype in the episodic memory and does nearest-neighbour
    retrieval at test time, which is the structurally correct way to do
    one-shot.

Same neuron budget, opposite organisation - the gap measures how much
the seven tricks actually buy on a task that stresses them.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reservoir import (
    LiquidStateMachine,
    ReservoirConfig,
    train_linear_readout,
    predict_linear,
)
from .distill import TaskBatch, extract_features, one_hot
from .memory import EpisodicMemory, MemoryConfig


# ---------------- task ----------------

def make_pattern_task(
    n_classes: int = 20,
    n_per_class: int = 1,
    T: int = 150,
    dt_ms: float = 1.0,
    noise_std: float = 0.4,
    seed: int = 0,
) -> TaskBatch:
    """20-class temporal pattern task = 5 shapes × 4 amplitudes.

    Shapes:
      0: sine
      1: square
      2: triangle
      3: ramp (linear sweep within the window)
      4: burst (silence with a sine pulse in the middle 30% of the window)

    Amplitudes: 0.5, 0.7, 0.9, 1.1

    Each sample has random phase and base frequency 3-7 Hz, with
    additive Gaussian noise of std `noise_std`.

    Class index encoding: cls = amp_idx * 5 + shape_idx
    """
    if n_classes != 20:
        raise ValueError("benchmark task is fixed to 20 classes (5 shapes x 4 amps)")
    n_samples = n_classes * n_per_class
    g = torch.Generator().manual_seed(seed)
    inputs = torch.zeros(n_samples, T)
    labels = torch.zeros(n_samples, dtype=torch.long)
    t = torch.arange(T, dtype=torch.float32) * dt_ms / 1000.0
    n_shapes = 5
    amps = [0.5, 0.7, 0.9, 1.1]

    idx = 0
    for cls in range(n_classes):
        shape_idx = cls % n_shapes
        amp_idx = cls // n_shapes
        amp = amps[amp_idx]

        for _ in range(n_per_class):
            phase = torch.rand((), generator=g).item() * 2 * math.pi
            f = 3.0 + 4.0 * torch.rand((), generator=g).item()

            if shape_idx == 0:           # sine
                wave = amp * torch.sin(2 * math.pi * f * t + phase)
            elif shape_idx == 1:         # square
                wave = amp * torch.sign(torch.sin(2 * math.pi * f * t + phase))
            elif shape_idx == 2:         # triangle
                wave = amp * (2.0 / math.pi) * torch.arcsin(
                    torch.sin(2 * math.pi * f * t + phase)
                )
            elif shape_idx == 3:         # ramp
                wave = amp * (2.0 * t / t[-1] - 1.0)
            else:                         # burst
                center = T // 2
                width = max(2, T * 30 // 100)
                envelope = torch.zeros(T)
                envelope[center - width // 2: center + width // 2] = 1.0
                wave = amp * torch.sin(2 * math.pi * f * t + phase) * envelope

            noise = torch.randn(T, generator=g) * noise_std
            inputs[idx] = wave + noise
            labels[idx] = cls
            idx += 1

    return TaskBatch(inputs=inputs, labels=labels, freqs=[])


# ---------------- feature transformations ----------------

class DendriticFeatureProjection(nn.Module):
    """Fixed random K-dendrite projection of a feature vector.

    Same compute pattern as a layer of DendriticPopulation units:
    each output unit owns K dendrites, each dendrite computes a sigmoid
    of a sparse random linear projection of the input features, and the
    soma sums the dendrite outputs. Used as a stateless feature
    transformer rather than a temporal LIF simulation, so the dendritic
    nonlinearity can be applied to reservoir features directly.

    Weights are sparse (most entries zero) so a 4500-dim reservoir
    feature does not explode into millions of dense weights.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_dendrites: int = 4,
        sparsity: float = 0.04,
        gain: float = 4.0,
        threshold: float = 0.5,
        seed: int = 0,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.K = n_dendrites
        self.gain = gain
        self.threshold = threshold
        g = torch.Generator().manual_seed(seed)
        mask = (torch.rand(n_dendrites, n_out, n_in, generator=g) < sparsity).float()
        weights = torch.randn(n_dendrites, n_out, n_in, generator=g) * mask
        # Normalise per-dendrite so post-projection values land around the
        # sigmoid's active range regardless of how many nonzero weights it has.
        active = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        norm = weights.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / norm * torch.sqrt(active)
        self.register_buffer("weights", weights)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_in) or (n_in,) -> (B, n_out)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # d_input[b, k, o] = sum_n weights[k, o, n] * x[b, n]
        d_input = torch.einsum("kon,bn->bko", self.weights, x)
        d_out = torch.sigmoid((d_input - self.threshold) * self.gain)
        return d_out.sum(dim=1)


def kwta_top_k(x: torch.Tensor, k: int) -> torch.Tensor:
    """Keep top-k values per row; zero the rest.

    The Phase 1 trick at the feature-vector level: forces a sparse
    representation so cosine similarity in the memory bank is dominated
    by a few high-magnitude entries (acts like a noisy hash).
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    k = min(k, x.shape[-1])
    topk_vals, topk_idx = torch.topk(x, k, dim=-1)
    out = torch.zeros_like(x)
    out.scatter_(-1, topk_idx, topk_vals)
    return out


# ---------------- systems ----------------

@dataclass
class BenchmarkResult:
    naive_acc: float
    integrated_acc: float
    gap_pp: float
    naive_train_time_s: float
    naive_test_time_s: float
    integrated_train_time_s: float
    integrated_test_time_s: float
    n_reservoir: int
    n_classes: int
    n_train: int
    n_test: int
    noise_std: float


def _naive_classify(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, n_classes: int, ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive 'standard LSM' classifier: ridge regression on reservoir features.
    Returns (predictions, learned weights W) so the readout-param count can
    be reported by the caller."""
    Y = one_hot(y_train, n_classes)
    W = train_linear_readout(X_train, Y, ridge=ridge)
    pred = predict_linear(W, X_test).argmax(dim=-1)
    return pred, W


def _integrated_classify(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, n_classes: int,
    n_dendrite_out: int, n_dendrites: int, k_active: int, seed: int,
) -> tuple[torch.Tensor, dict]:
    """Integrated agent: dendritic projection + kWTA + episodic memory +
    cosine top-1 retrieval. Returns (predictions, info-dict)."""
    feat_dim = X_train.shape[1]
    dendritic = DendriticFeatureProjection(
        n_in=feat_dim, n_out=n_dendrite_out,
        n_dendrites=n_dendrites, sparsity=0.04,
        gain=4.0, threshold=0.5, seed=seed,
    )
    Z_train = kwta_top_k(dendritic.forward(X_train), k_active)
    Z_test = kwta_top_k(dendritic.forward(X_test), k_active)

    memory = EpisodicMemory(MemoryConfig(
        capacity=max(Z_train.shape[0], 1),
        key_dim=n_dendrite_out, value_dim=n_classes,
        similarity="cosine", top_k=1,
    ))
    for z, label in zip(Z_train, y_train):
        memory.write(z, F.one_hot(label, n_classes).float())
        memory.step_time(1.0)

    preds = torch.zeros(Z_test.shape[0], dtype=torch.long)
    for i, z in enumerate(Z_test):
        out = memory.read(z, top_k=1)
        preds[i] = int(out["mixed_value"].argmax().item())
    info = {
        "n_dendrite_out": n_dendrite_out,
        "n_dendrites": n_dendrites,
        "k_active": k_active,
        "memory_slots_used": memory.n_stored,
        "memory_key_dim": memory.cfg.key_dim,
    }
    return preds, info


# ---------------- driver ----------------

def run_benchmark(
    n_reservoir: int = 1000,
    n_test_per_class: int = 10,
    noise_std: float = 0.4,
    seed: int = 0,
    n_dendrite_out: int = 600,
    n_dendrites: int = 4,
    k_active: int = 80,
    ridge: float = 1.0,
) -> BenchmarkResult:
    """Run both systems head-to-head on the 20-class one-shot task.

    Both systems extract features from the SAME reservoir (the reservoir
    is fixed/random, no training, so it is the substrate not the
    learnable component). The difference is in the downstream pipeline.
    """
    train = make_pattern_task(
        n_classes=20, n_per_class=1,
        noise_std=noise_std, seed=seed,
    )
    test = make_pattern_task(
        n_classes=20, n_per_class=n_test_per_class,
        noise_std=noise_std, seed=seed + 99,
    )

    # Build the shared reservoir once
    lsm = LiquidStateMachine(ReservoirConfig(n_reservoir=n_reservoir, seed=seed))

    # Feature extraction (cost shared between the two systems)
    t0 = time.time()
    X_train = extract_features(lsm, train)
    X_test = extract_features(lsm, test)
    feature_time = time.time() - t0

    n_classes = 20

    # Naive
    t0 = time.time()
    naive_pred, _ = _naive_classify(X_train, train.labels, X_test, n_classes, ridge=ridge)
    naive_decision_time = time.time() - t0
    naive_acc = (naive_pred == test.labels).float().mean().item()

    # Integrated
    t0 = time.time()
    integrated_pred, _ = _integrated_classify(
        X_train, train.labels, X_test, n_classes,
        n_dendrite_out=n_dendrite_out, n_dendrites=n_dendrites,
        k_active=k_active, seed=seed + 100,
    )
    integrated_decision_time = time.time() - t0
    integrated_acc = (integrated_pred == test.labels).float().mean().item()

    return BenchmarkResult(
        naive_acc=naive_acc,
        integrated_acc=integrated_acc,
        gap_pp=(integrated_acc - naive_acc) * 100,
        naive_train_time_s=naive_decision_time,
        naive_test_time_s=feature_time,
        integrated_train_time_s=integrated_decision_time,
        integrated_test_time_s=feature_time,
        n_reservoir=n_reservoir,
        n_classes=20,
        n_train=train.inputs.shape[0],
        n_test=test.inputs.shape[0],
        noise_std=noise_std,
    )


def format_result(r: BenchmarkResult) -> str:
    chance = 100.0 / r.n_classes
    return (
        f"Benchmark: 20-class one-shot temporal classification\n"
        f"  N reservoir neurons: {r.n_reservoir}\n"
        f"  Train: {r.n_train} samples (1/class), Test: {r.n_test} samples ({r.n_test//r.n_classes}/class)\n"
        f"  Noise std: {r.noise_std}\n"
        f"  ----\n"
        f"  NAIVE      (reservoir + ridge readout):       {r.naive_acc*100:6.2f}%   "
        f"[chance {chance:.1f}%]\n"
        f"  INTEGRATED (reservoir + dendrites + kWTA +    {r.integrated_acc*100:6.2f}%\n"
        f"              episodic memory):\n"
        f"  ----\n"
        f"  GAP: {r.gap_pp:+.1f} percentage points (same neuron budget, same task)\n"
        f"  Naive  train/test time: {r.naive_train_time_s:.2f}s / {r.naive_test_time_s:.2f}s\n"
        f"  Integ. train/test time: {r.integrated_train_time_s:.2f}s / {r.integrated_test_time_s:.2f}s"
    )


if __name__ == "__main__":
    r = run_benchmark()
    print(format_result(r))
