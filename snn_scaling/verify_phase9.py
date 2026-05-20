"""Phase 9 verification: e-prop local online learning (no BPTT).

Phase 8 trains the SNN by backpropagation through time - it unrolls the
whole T-step graph and runs a backward pass through it. Phase 9 trains
the same kind of SNN with e-prop (Bellec et al. 2020): forward-only,
local eligibility traces multiplied by a per-neuron learning signal.
Autograd is never invoked during e-prop training.

Gates:

  1. The e-prop forward pass emits a correctly shaped, non-zero
     eligibility trace and carries NO autograd graph; the weights are
     plain (non-grad) tensors
  2. e-prop trains the SNN forward-only (autograd never called) and
     learns the 3-class temporal task well above chance
  3. e-prop reaches test accuracy comparable to a BPTT-trained SNN
  4. The e-prop weight gradient positively tracks the true BPTT
     gradient - i.e. e-prop genuinely approximates the gradient rather
     than doing something unrelated
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .eprop import (
    EpropSNN,
    EpropConfig,
    train_eprop,
    evaluate_eprop,
    eprop_gradients,
)
from .surrogate import (
    build_freq_dataset,
    SurrogateSNNClassifier,
    SurrogateTrainConfig,
    train_surrogate_snn,
    evaluate_surrogate_snn,
)
from .verify_phase0 import CheckResult, format_results


# ---------------- gates ----------------

def check_eprop_forward_no_autograd() -> CheckResult:
    """The e-prop forward pass builds a correctly shaped, non-zero
    eligibility trace, carries no autograd graph (it runs under
    no_grad), and operates on plain non-grad weight tensors."""
    cfg = EpropConfig(input_dim_D=10, n_hidden=48, n_classes=3)
    model = EpropSNN(cfg, seed=0)
    xtr, _, _, _ = build_freq_dataset(n_train=60, n_test=12, T=50,
                                       n_classes=3, seed=1)
    logits, rate, G = model.forward_with_eligibility(xtr[:16])
    no_graph = (logits.grad_fn is None) and (G.grad_fn is None)
    shape_ok = (tuple(logits.shape) == (16, 3)
                and tuple(G.shape) == (16, 48, 10))
    G_nonzero = G.abs().sum().item() > 0.0
    rate_valid = bool((rate >= 0).all() and (rate <= 1).all())
    weights_plain = (not model.W_in.requires_grad)
    passed = no_graph and shape_ok and G_nonzero and rate_valid and weights_plain
    return CheckResult(
        "e-prop forward: eligibility trace built, no autograd graph",
        passed, float(G.abs().mean()), 0.0,
        note=f"(G shape={tuple(G.shape)}, no_graph={no_graph}, "
             f"weights_plain={weights_plain}, rate in [0,1]={rate_valid})",
    )


def check_eprop_trains_no_autograd() -> CheckResult:
    """e-prop trains the SNN entirely forward-only - autograd is never
    invoked (the weights stay plain non-grad tensors with .grad == None)
    - and the trained network learns the 3-class task above chance."""
    xtr, ytr, xte, yte = build_freq_dataset(
        n_train=240, n_test=120, T=70, noise_std=0.5, n_classes=3, seed=2)
    model = EpropSNN(EpropConfig(input_dim_D=12, n_hidden=96, n_classes=3), seed=2)
    train_eprop(model, xtr, ytr, steps=150, lr=0.5, seed=2)
    ev = evaluate_eprop(model, xte, yte)
    no_autograd = (model.W_in.grad is None) and (not model.W_in.requires_grad)
    passed = ev["accuracy"] >= 0.70 and no_autograd
    return CheckResult(
        "e-prop trains the SNN forward-only (no autograd) and learns",
        passed, ev["accuracy"], 0.70,
        note=f"(test accuracy {ev['accuracy']*100:.1f}%, chance 33%; "
             f"autograd never used={no_autograd})",
    )


def check_eprop_matches_bptt() -> CheckResult:
    """e-prop reaches test accuracy comparable to a BPTT-trained SNN on
    the same task. e-prop is a gradient APPROXIMATION, so being a little
    behind BPTT is expected and honest - the gate allows a 15pp margin."""
    xtr, ytr, xte, yte = build_freq_dataset(
        n_train=240, n_test=120, T=70, noise_std=0.5, n_classes=3, seed=3)
    eprop_model = EpropSNN(
        EpropConfig(input_dim_D=12, n_hidden=96, n_classes=3), seed=3)
    train_eprop(eprop_model, xtr, ytr, steps=150, lr=0.5, seed=3)
    a_eprop = evaluate_eprop(eprop_model, xte, yte)["accuracy"]

    bptt_model = SurrogateSNNClassifier(n_hidden=96, n_classes=3,
                                          train_lif=True, seed=3)
    train_surrogate_snn(bptt_model, xtr, ytr,
                         SurrogateTrainConfig(steps=150, seed=3))
    a_bptt = evaluate_surrogate_snn(bptt_model, xte, yte)["accuracy"]

    passed = a_eprop >= a_bptt - 0.15
    return CheckResult(
        "e-prop reaches accuracy comparable to BPTT (within 15pp)",
        passed, a_eprop - a_bptt, -0.15,
        note=f"(e-prop {a_eprop*100:.1f}%, BPTT {a_bptt*100:.1f}%, "
             f"gap {(a_eprop-a_bptt)*100:+.1f}pp)",
    )


def check_eprop_gradient_tracks_bptt() -> CheckResult:
    """The e-prop weight gradient positively tracks the TRUE BPTT
    gradient. Cosine similarity of the two W_in gradients must be clearly
    positive - this proves e-prop genuinely approximates the gradient
    (forward-only) rather than producing an unrelated update."""
    xtr, ytr, _, _ = build_freq_dataset(
        n_train=240, n_test=24, T=70, noise_std=0.5, n_classes=3, seed=4)
    model = EpropSNN(EpropConfig(input_dim_D=12, n_hidden=96, n_classes=3), seed=4)
    train_eprop(model, xtr, ytr, steps=40, lr=0.5, seed=4)   # partially trained
    dW_eprop, _, _, _ = eprop_gradients(model, xtr[:64], ytr[:64])
    dW_bptt = model.bptt_gradient_W_in(xtr[:64], ytr[:64])
    cos = float(F.cosine_similarity(
        dW_eprop.flatten(), dW_bptt.flatten(), dim=0))
    return CheckResult(
        "e-prop gradient positively tracks the true BPTT gradient",
        cos > 0.2, cos, 0.2,
        note=f"(cosine similarity of e-prop vs BPTT W_in gradient = {cos:.3f})",
    )


# ---------------- runner ----------------

def run_all() -> list[CheckResult]:
    return [
        check_eprop_forward_no_autograd(),
        check_eprop_trains_no_autograd(),
        check_eprop_matches_bptt(),
        check_eprop_gradient_tracks_bptt(),
    ]


if __name__ == "__main__":
    t0 = time.time()
    results = run_all()
    print(f"Phase 9 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    print(f"\n  -> {n_pass} / {len(results)} checks passed.")
