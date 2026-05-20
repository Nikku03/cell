"""Phase 8 verification: surrogate-gradient SNN training.

Phases 0-7 never train the SNN itself - the spiking substrate is fixed
and random, only a closed-form readout is fit. Phase 8 makes the spiking
weights trainable by backpropagation through time, using the
surrogate-gradient trick for the non-differentiable spike threshold.

Gates:

  1. The surrogate gradient actually flows to the spiking-layer weights
     (without it, the Heaviside's true derivative is 0 and training is
     impossible)
  2. The forward pass emits exact binary spikes with the right shapes
  3. A surrogate-trained SNN learns the 3-class temporal task
     (sine / square / noise) well above chance
  4. Training the SNN weights beats leaving them frozen-random - i.e.
     surrogate-gradient training adds value over a fixed substrate
  5. The firing-rate regulariser cuts spiking activity toward a target
     while keeping classification accuracy
  6. The spike-budget penalty reduces total spikes per inference while
     keeping classification accuracy
"""

from __future__ import annotations

import statistics
import time

import torch
import torch.nn.functional as F

from .surrogate import (
    SurrogateSNNClassifier,
    SurrogateTrainConfig,
    train_surrogate_snn,
    evaluate_surrogate_snn,
    build_freq_dataset,
    spike_fn,
)
from .verify_phase0 import CheckResult, format_results


# ---------------- gates ----------------

def check_surrogate_gradient_flows() -> CheckResult:
    """A backward pass produces finite, non-zero gradients on the spiking
    layer's input weights. With a plain Heaviside (derivative 0 a.e.) the
    gradient would be exactly zero and the SNN could not be trained."""
    x = torch.tensor([-0.4, 0.0, 0.6], requires_grad=True)
    s = spike_fn(x, 10.0)
    assert bool(((s == 0) | (s == 1)).all()), "spike_fn forward must be binary"
    s.sum().backward()
    surrogate_nonzero = x.grad.abs().sum().item() > 0.0

    model = SurrogateSNNClassifier(n_hidden=64, n_classes=3, train_lif=True, seed=0)
    xtr, ytr, _, _ = build_freq_dataset(n_train=48, n_test=12, T=60, seed=1)
    logits, _ = model(xtr[:16])
    F.cross_entropy(logits, ytr[:16]).backward()
    g = model.input_proj.weight.grad
    finite = bool(torch.isfinite(g).all())
    gnorm = float(g.norm())
    passed = surrogate_nonzero and finite and gnorm > 1e-8
    return CheckResult(
        "surrogate gradient flows to the spiking-layer weights",
        passed, gnorm, 1e-8,
        note=f"(input_proj grad norm={gnorm:.4e}, finite={finite}, "
             f"surrogate non-zero={surrogate_nonzero})",
    )


def check_spikes_are_binary() -> CheckResult:
    """The forward pass emits exact binary {0,1} spikes (the surrogate is
    backward-only) and produces correctly shaped logits + spike tensors."""
    n_hidden, T, B = 64, 60, 16
    model = SurrogateSNNClassifier(n_hidden=n_hidden, n_classes=3, seed=0)
    xtr, _, _, _ = build_freq_dataset(n_train=48, n_test=12, T=T, seed=2)
    logits, spikes = model(xtr[:B])
    uniq = torch.unique(spikes)
    is_binary = bool(((uniq == 0) | (uniq == 1)).all())
    shape_ok = (tuple(logits.shape) == (B, 3)
                and tuple(spikes.shape) == (B, T, n_hidden))
    passed = is_binary and shape_ok
    return CheckResult(
        "forward pass emits exact binary spikes with correct shapes",
        passed, float(uniq.numel()), 2.0,
        note=f"(spike values={uniq.tolist()}, logits={tuple(logits.shape)}, "
             f"spikes={tuple(spikes.shape)})",
    )


def check_trained_snn_learns() -> CheckResult:
    """A surrogate-trained SNN classifies the 3-class temporal task well
    above the 33% chance level."""
    xtr, ytr, xte, yte = build_freq_dataset(
        n_train=240, n_test=120, T=80, noise_std=0.5, seed=3)
    model = SurrogateSNNClassifier(n_hidden=128, n_classes=3, train_lif=True, seed=3)
    train_surrogate_snn(model, xtr, ytr, SurrogateTrainConfig(steps=140, seed=3))
    ev = evaluate_surrogate_snn(model, xte, yte)
    return CheckResult(
        "surrogate-trained SNN learns the 3-class task (accuracy >= 70%)",
        ev["accuracy"] >= 0.70, ev["accuracy"], 0.70,
        note=f"(test accuracy {ev['accuracy']*100:.1f}%, chance 33%)",
    )


def check_trained_beats_frozen() -> CheckResult:
    """Training the spiking-layer weights (surrogate-gradient) clearly
    beats leaving them frozen at random init - the frozen case is a
    fixed-random-substrate (reservoir-style) baseline that only fits the
    readout.

    Tested in the regime where it matters: a SMALL spiking layer (32
    neurons) on a HARDER 4-class task (sine / square / noise / chirp).
    A 32-neuron random substrate has only 32 random features and
    genuinely struggles; training makes each neuron class-informative.
    (On an easy 3-class task with 128 neurons a random substrate is
    already good and training adds only ~2-3pp - the gap is real but
    small. The small-network / hard-task regime is where surrogate
    training earns its keep.) Averaged over 3 seeds to remove single-run
    noise.
    """
    n_seeds = 3
    full_accs, frozen_accs = [], []
    for seed in range(n_seeds):
        xtr, ytr, xte, yte = build_freq_dataset(
            n_train=240, n_test=160, T=80, noise_std=0.6, n_classes=4, seed=seed)
        full = SurrogateSNNClassifier(n_hidden=32, n_classes=4, train_lif=True, seed=seed)
        frozen = SurrogateSNNClassifier(n_hidden=32, n_classes=4, train_lif=False, seed=seed)
        train_surrogate_snn(full, xtr, ytr, SurrogateTrainConfig(steps=160, seed=seed))
        train_surrogate_snn(frozen, xtr, ytr, SurrogateTrainConfig(steps=160, seed=seed))
        full_accs.append(evaluate_surrogate_snn(full, xte, yte)["accuracy"])
        frozen_accs.append(evaluate_surrogate_snn(frozen, xte, yte)["accuracy"])
    a_full = statistics.mean(full_accs)
    a_frozen = statistics.mean(frozen_accs)
    gap = a_full - a_frozen
    return CheckResult(
        "training the SNN weights beats a frozen random substrate",
        gap >= 0.08, gap, 0.08,
        note=f"(3-seed mean: trained {a_full*100:.1f}%, "
             f"frozen-random {a_frozen*100:.1f}%, gap {gap*100:+.1f}pp)",
    )


def check_firing_rate_regularizer() -> CheckResult:
    """The L1 firing-rate regulariser, trained toward a low target,
    materially cuts the network's spiking activity relative to an
    unregularised run - while keeping classification above chance."""
    xtr, ytr, xte, yte = build_freq_dataset(
        n_train=200, n_test=100, T=80, noise_std=0.5, seed=5)
    unreg = SurrogateSNNClassifier(n_hidden=128, n_classes=3, train_lif=True, seed=5)
    train_surrogate_snn(unreg, xtr, ytr,
                         SurrogateTrainConfig(steps=120, lambda_rate=0.0, seed=5))
    e0 = evaluate_surrogate_snn(unreg, xte, yte)

    reg = SurrogateSNNClassifier(n_hidden=128, n_classes=3, train_lif=True, seed=5)
    train_surrogate_snn(reg, xtr, ytr, SurrogateTrainConfig(
        steps=120, lambda_rate=3.0, target_rate=0.05, seed=5))
    e1 = evaluate_surrogate_snn(reg, xte, yte)

    cut = e1["firing_rate"] < 0.85 * e0["firing_rate"]
    still_classifies = e1["accuracy"] > 0.5
    passed = cut and still_classifies
    return CheckResult(
        "firing-rate regulariser cuts spiking activity, keeps accuracy",
        passed, e0["firing_rate"] - e1["firing_rate"], 0.0,
        note=f"(rate unreg={e0['firing_rate']:.3f} -> reg={e1['firing_rate']:.3f}; "
             f"reg accuracy {e1['accuracy']*100:.1f}%)",
    )


def check_spike_budget() -> CheckResult:
    """The spike-budget hinge penalty reduces total spikes per inference
    relative to an unbudgeted run - while keeping accuracy above chance."""
    xtr, ytr, xte, yte = build_freq_dataset(
        n_train=200, n_test=100, T=80, noise_std=0.5, seed=6)
    unbudg = SurrogateSNNClassifier(n_hidden=128, n_classes=3, train_lif=True, seed=6)
    train_surrogate_snn(unbudg, xtr, ytr, SurrogateTrainConfig(steps=120, seed=6))
    e0 = evaluate_surrogate_snn(unbudg, xte, yte)
    budget = 0.5 * e0["total_spikes_per_sample"]

    budg = SurrogateSNNClassifier(n_hidden=128, n_classes=3, train_lif=True, seed=6)
    train_surrogate_snn(budg, xtr, ytr, SurrogateTrainConfig(
        steps=120, lambda_budget=1.0e-3, budget_per_sample=budget, seed=6))
    e1 = evaluate_surrogate_snn(budg, xte, yte)

    reduced = e1["total_spikes_per_sample"] < e0["total_spikes_per_sample"]
    still_classifies = e1["accuracy"] > 0.5
    passed = reduced and still_classifies
    return CheckResult(
        "spike-budget penalty reduces total spikes, keeps accuracy",
        passed,
        e0["total_spikes_per_sample"] - e1["total_spikes_per_sample"], 0.0,
        note=f"(spikes/sample unbudgeted={e0['total_spikes_per_sample']:.0f} -> "
             f"budgeted={e1['total_spikes_per_sample']:.0f} (budget={budget:.0f}); "
             f"budgeted accuracy {e1['accuracy']*100:.1f}%)",
    )


# ---------------- runner ----------------

def run_all() -> list[CheckResult]:
    return [
        check_surrogate_gradient_flows(),
        check_spikes_are_binary(),
        check_trained_snn_learns(),
        check_trained_beats_frozen(),
        check_firing_rate_regularizer(),
        check_spike_budget(),
    ]


if __name__ == "__main__":
    t0 = time.time()
    results = run_all()
    print(f"Phase 8 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    print(f"\n  -> {n_pass} / {len(results)} checks passed.")
