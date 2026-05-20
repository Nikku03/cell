"""RL task generation utilities for the snn_scaling agents.

Provides the 5-class temporal-pattern task, reservoir feature extraction
(centered + L2-normalised), and reversal-style + mixed-regime phase-map
generators. Used by the colab cells and by the ASI-Evolve mount.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .distill import TaskBatch as _DistillTaskBatch, _drive_input


# Re-export TaskBatch under the rl_tasks namespace for convenience
TaskBatch = _DistillTaskBatch


def make_5class_task(n_per_class: int = 30, T: int = 200,
                       noise_std: float = 0.4, seed: int = 0) -> TaskBatch:
    """5 waveform shapes (sine, square, triangle, ramp, burst) at unit amp,
    additive Gaussian noise. Same task as colab_reversal_benchmark.py."""
    n_classes = 5
    n_samples = n_classes * n_per_class
    g = torch.Generator().manual_seed(seed)
    inputs = torch.zeros(n_samples, T)
    labels = torch.zeros(n_samples, dtype=torch.long)
    t = torch.arange(T, dtype=torch.float32) * 0.001
    idx = 0
    for cls in range(n_classes):
        for _ in range(n_per_class):
            phase = torch.rand((), generator=g).item() * 2 * math.pi
            f = 3.0 + 4.0 * torch.rand((), generator=g).item()
            if cls == 0:
                wave = torch.sin(2 * math.pi * f * t + phase)
            elif cls == 1:
                wave = torch.sign(torch.sin(2 * math.pi * f * t + phase))
            elif cls == 2:
                wave = (2 / math.pi) * torch.arcsin(torch.sin(2 * math.pi * f * t + phase))
            elif cls == 3:
                wave = (2 * t / t[-1] - 1)
            else:
                env = torch.zeros(T); env[T // 2 - T * 15 // 100: T // 2 + T * 15 // 100] = 1.0
                wave = torch.sin(2 * math.pi * f * t + phase) * env
            inputs[idx] = wave + torch.randn(T, generator=g) * noise_std
            labels[idx] = cls
            idx += 1
    return TaskBatch(inputs=inputs, labels=labels, freqs=[])


@torch.no_grad()
def extract_centered_features(reservoir, batch: TaskBatch, device: str = "cpu",
                                 center: bool = True, l2_normalize: bool = True) -> torch.Tensor:
    """Run the reservoir on each sample, collect (mean, std, range) of late-
    half membrane V, then optionally center + L2-normalise the features.

    Centering subtracts the train-batch mean (removes the strong common-mode
    direction that the reservoir produces near rest potential). L2 makes
    cosine similarity equal to dot product, useful for memory keys.
    """
    N = reservoir.n
    out = torch.zeros(batch.inputs.shape[0], 3 * N, device=device)
    for i in range(batch.inputs.shape[0]):
        drive = _drive_input(N, batch.inputs[i]).to(device=device)
        T = drive.shape[0]
        Vs = torch.zeros(T, N, device=device)
        reservoir.reset()
        for k in range(T):
            reservoir.step(drive[k], dt=1.0, t=k * 1.0)
            Vs[k] = reservoir.cluster.population.V.clone()
        late = Vs[T // 2:]
        out[i] = torch.cat([
            late.mean(dim=0),
            late.std(dim=0),
            late.max(dim=0).values - late.min(dim=0).values,
        ])
    if center:
        out = out - out.mean(dim=0, keepdim=True)
    if l2_normalize:
        out = F.normalize(out, dim=-1)
    return out


@torch.no_grad()
def extract_features_and_traces(reservoir, batch: TaskBatch, device: str = "cpu",
                                   decimate: int = 4):
    """Run the reservoir ONCE per sample and return both:

      features: (n_samples, 3*N) standard (mean, std, range) of late-half
                membrane V, centered + L2-normalised. Identical recipe to
                extract_centered_features() - the FIXED feature set used
                by the baseline agents.
      traces:   (n_samples, T//decimate, N) decimated raw membrane traces -
                the raw material a mutable feature transform can operate on
                (temporal bins, derivatives, FFT, spike proxies, etc.).

    One reservoir pass produces both, so exposing an evolvable feature
    extractor costs no extra simulation time.
    """
    N = reservoir.n
    n = batch.inputs.shape[0]
    T = batch.inputs.shape[1]
    dec_idx = torch.arange(0, T, decimate)
    feats = torch.zeros(n, 3 * N, device=device)
    traces = torch.zeros(n, dec_idx.numel(), N, device=device)
    for i in range(n):
        drive = _drive_input(N, batch.inputs[i]).to(device=device)
        Ti = drive.shape[0]
        Vs = torch.zeros(Ti, N, device=device)
        reservoir.reset()
        for k in range(Ti):
            reservoir.step(drive[k], dt=1.0, t=k * 1.0)
            Vs[k] = reservoir.cluster.population.V.clone()
        late = Vs[Ti // 2:]
        feats[i] = torch.cat([
            late.mean(dim=0),
            late.std(dim=0),
            late.max(dim=0).values - late.min(dim=0).values,
        ])
        traces[i] = Vs[dec_idx.to(device=Vs.device)]
    feats = feats - feats.mean(dim=0, keepdim=True)
    feats = F.normalize(feats, dim=-1)
    return feats, traces


def make_reversal_phase_maps(n_phases: int = 4, n_classes: int = 5,
                                n_actions: int = 3, seed: int = 0) -> list[torch.Tensor]:
    """Full reversal: every class gets a fresh random action each phase.
    Successive phases never repeat the previous mapping verbatim."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    maps = []
    last = None
    for p in range(n_phases):
        while True:
            m = torch.randint(0, n_actions, (n_classes,), generator=g)
            if last is None or not torch.equal(m, last):
                break
        maps.append(m)
        last = m
    return maps


def make_mixed_phase_maps(n_phases: int = 4, n_classes: int = 5, n_actions: int = 3,
                            n_stationary: int = 3, seed: int = 0):
    """Mixed regime: n_stationary classes keep a fixed correct-action across
    all phases; the remaining classes reshuffle each phase.
    Returns (list of phase maps, is_stationary bool tensor)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    stationary_idx = torch.randperm(n_classes, generator=g)[:n_stationary]
    is_stationary = torch.zeros(n_classes, dtype=torch.bool)
    is_stationary[stationary_idx] = True
    fixed_actions = torch.randint(0, n_actions, (n_classes,), generator=g)
    maps = []
    for p in range(n_phases):
        m = fixed_actions.clone()
        for cls in range(n_classes):
            if not is_stationary[cls]:
                while True:
                    a = int(torch.randint(0, n_actions, (1,), generator=g).item())
                    if p == 0 or a != int(maps[p - 1][cls].item()):
                        m[cls] = a
                        break
        maps.append(m)
    return maps, is_stationary
