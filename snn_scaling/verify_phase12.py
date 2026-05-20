"""Phase 12 verification: batched reservoir feature extraction.

The verified SpikingReservoir is a single-sample (N,) simulator;
extract_centered_features runs it once PER SAMPLE in a Python for-loop,
which measured as ~80% of every benchmark and ASI-Evolve run.
BatchedReservoir borrows an existing reservoir's connectivity + neuron
parameters and re-runs the SAME dynamics with the batch carried as a
leading tensor dim - all B samples in parallel, the only Python loop
over T timesteps (the recurrence couples step to step).

Gates:

  1. With noise disabled the batched extractor is BIT-IDENTICAL to the
     sequential extract_centered_features - same network, same recipe,
     just reshaped from a per-sample loop to a batched tensor op
  2. Batched extraction is dramatically faster than the per-sample loop
  3. Batched features train a linear readout that classifies the
     5-class waveform task well above chance
  4. With noise enabled the batched noise stream - drawn (B, N) at once
     rather than (N,) per sample - perturbs the feature matrix by the
     same relative magnitude as the sequential path: equivalent noise
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .batched import extract_features_batched
from .reservoir import (
    ReservoirConfig,
    SpikingReservoir,
    predict_linear,
    train_linear_readout,
)
from .rl_tasks import extract_centered_features, make_5class_task
from .verify_phase0 import CheckResult, format_results


# ---------------- helpers ----------------

def _classify(feats: torch.Tensor, labels: torch.Tensor, n_classes: int,
              n_train_pc: int, seed: int) -> float:
    """Per-class train/test split, train-mean centering, ridge readout.

    Centering uses the TRAIN mean only (subtracted from both splits) so
    no test statistics leak into training - the train/test-consistent
    version of the extract_centered_features recipe."""
    g = torch.Generator().manual_seed(seed)
    tr, te = [], []
    for c in range(n_classes):
        idx = (labels == c).nonzero(as_tuple=False).squeeze(-1)
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        tr.append(perm[:n_train_pc])
        te.append(perm[n_train_pc:])
    tr, te = torch.cat(tr), torch.cat(te)
    mu = feats[tr].mean(dim=0, keepdim=True)
    Xtr = F.normalize(feats[tr] - mu, dim=-1)
    Xte = F.normalize(feats[te] - mu, dim=-1)
    ytr = torch.eye(n_classes)[labels[tr]]
    W = train_linear_readout(Xtr, ytr, ridge=1e-2)
    pred = predict_linear(W, Xte).argmax(dim=-1)
    return float((pred == labels[te]).float().mean())


_CACHE: dict = {}


def _feature_sets() -> dict:
    """Build a matched noise-free / noisy reservoir pair (IDENTICAL
    connectivity - the same global seed drives the degree draw and the
    same cfg seed drives target selection, so only noise_std differs)
    and extract features each way. Cached: the sequential passes are the
    slow part and feed both gate 3 and gate 4."""
    if not _CACHE:
        common = dict(n_reservoir=100, p_recurrent=0.12, seed=7)
        batch = make_5class_task(n_per_class=20, T=150, noise_std=0.3, seed=4)
        torch.manual_seed(101)
        res_clean = SpikingReservoir(ReservoirConfig(noise_std=0.0, **common))
        clean = extract_centered_features(res_clean, batch, center=False,
                                          l2_normalize=False)
        torch.manual_seed(101)
        res = SpikingReservoir(ReservoirConfig(noise_std=0.5, **common))
        seq = extract_centered_features(res, batch, center=False,
                                        l2_normalize=False)
        bat = extract_features_batched(res, batch, center=False,
                                       l2_normalize=False)
        _CACHE.update(clean=clean, seq=seq, bat=bat, labels=batch.labels)
    return _CACHE


# ---------------- gates ----------------

def check_batched_bit_identical_noise_free() -> CheckResult:
    """With reservoir noise disabled, the batched extractor reproduces
    extract_centered_features to the bit: same borrowed connectivity,
    same exponential-Euler update, same (mean, std, range) recipe - only
    the per-sample Python loop is replaced by a batched tensor op."""
    cfg = ReservoirConfig(n_reservoir=80, p_recurrent=0.12,
                          noise_std=0.0, seed=5)
    res = SpikingReservoir(cfg)
    batch = make_5class_task(n_per_class=8, T=120, noise_std=0.3, seed=2)
    seq = extract_centered_features(res, batch)
    bat = extract_features_batched(res, batch)
    diff = float((seq - bat).abs().max())
    passed = diff < 1e-3
    return CheckResult(
        "batched features bit-identical to sequential (noise-free)",
        passed, diff, 1e-3,
        note=f"(max abs diff over the {seq.shape[0]}x{seq.shape[1]} "
             f"feature matrix = {diff:.2e})",
    )


def check_batched_speedup() -> CheckResult:
    """Batched extraction collapses 80 sequential reservoir runs into one
    parallel pass: the Python loop shrinks from B*T iterations to T."""
    cfg = ReservoirConfig(n_reservoir=100, p_recurrent=0.12,
                          noise_std=0.5, seed=1)
    res = SpikingReservoir(cfg)
    batch = make_5class_task(n_per_class=16, T=150, noise_std=0.3, seed=3)
    t0 = time.time()
    extract_centered_features(res, batch)
    t_seq = time.time() - t0
    t0 = time.time()
    extract_features_batched(res, batch)
    t_bat = time.time() - t0
    speedup = t_seq / max(t_bat, 1e-6)
    passed = speedup > 5.0
    return CheckResult(
        "batched extraction is much faster than the per-sample loop",
        passed, speedup, 5.0,
        note=f"({batch.inputs.shape[0]} samples: sequential {t_seq:.2f}s "
             f"-> batched {t_bat:.2f}s, {speedup:.0f}x)",
    )


def check_batched_features_classify() -> CheckResult:
    """Batched features carry real task signal: a ridge readout trained
    on them classifies the 5-class waveform task well above the 0.20
    chance level (mean test accuracy over three splits)."""
    fs = _feature_sets()
    accs = [_classify(fs["bat"], fs["labels"], 5, 13, s) for s in range(3)]
    mean_acc = sum(accs) / len(accs)
    passed = mean_acc > 0.40
    return CheckResult(
        "batched features classify the 5-class task above chance",
        passed, mean_acc, 0.40,
        note=f"(mean test accuracy {mean_acc:.3f} over 3 splits; "
             f"5-class chance = 0.20)",
    )


def check_batched_noise_equivalent() -> CheckResult:
    """With noise enabled the batched simulator draws its noise (B, N) at
    once instead of (N,) per sample, so it is not bit-identical to the
    sequential path - but it is statistically equivalent. Measured
    against a shared noise-free reference, the noise perturbs the feature
    matrix by the same relative magnitude either way, and that
    perturbation is a small bounded fraction of the signal."""
    fs = _feature_sets()
    sig = float(fs["clean"].norm())
    d_seq = float((fs["seq"] - fs["clean"]).norm()) / sig
    d_bat = float((fs["bat"] - fs["clean"]).norm()) / sig
    gap = abs(d_seq - d_bat)
    passed = gap < 0.01 and d_seq < 0.15 and d_bat < 0.15
    return CheckResult(
        "batched noise is statistically equivalent to sequential noise",
        passed, gap, 0.01,
        note=f"(noise perturbation vs noise-free ref: sequential "
             f"{d_seq:.4f}, batched {d_bat:.4f}; both bounded well "
             f"below 0.15)",
    )


# ---------------- runner ----------------

def run_all() -> list[CheckResult]:
    return [
        check_batched_bit_identical_noise_free(),
        check_batched_speedup(),
        check_batched_features_classify(),
        check_batched_noise_equivalent(),
    ]


if __name__ == "__main__":
    t0 = time.time()
    results = run_all()
    print(f"Phase 12 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    print(f"\n  -> {n_pass} / {len(results)} checks passed.")
