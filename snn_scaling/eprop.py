"""Trick 9 / SparseSpike-GEMM steal #5: e-prop local online learning.

BPTT (Phase 8) trains an SNN by unrolling the whole T-step graph and
backpropagating through time - it must keep every intermediate state and
is not biologically plausible (no synapse has a backward channel running
backwards through time). e-prop (Bellec et al. 2020) is the alternative:
the gradient of a loss w.r.t. a synaptic weight factorises as

    dL/dW_ji  =  sum_t  L_j[t]  *  e_ji[t]

where the ELIGIBILITY TRACE e_ji[t] is a purely LOCAL quantity computed
FORWARD in time (it sees only the pre-synaptic input and the
post-synaptic membrane state) and L_j[t] is a per-neuron LEARNING
SIGNAL. No backward pass through time, no stored activation history.

This module implements e-prop for a feed-forward LIF classifier:

  - the 1-D input signal is DELAY-EMBEDDED into a length-D sliding
    window, so each neuron's input weight W_in[j,:] is a genuine D-tap
    temporal filter that e-prop can shape
  - eligibility vector   eps_i[t] = beta * eps_i[t-1] + xenc_i[t]
    (the leaky-integrated encoded input - it is shared across post
    neurons because the pre-synaptic input does not depend on j)
  - eligibility trace     e_ji[t] = psi_j[t] * eps_i[t]
    (psi = the fast-sigmoid surrogate pseudo-derivative of the spike)
  - learning signal       L_j = sum_c readout_w[c,j] * output_error_c
    (symmetric e-prop: a ONE-layer spatial projection of the output
    error - e-prop removes backprop through TIME, not through the
    single readout layer)

`train_eprop` never calls autograd: it computes every gradient
explicitly, forward-only. `EpropSNN.bptt_gradient_W_in` is provided
ONLY so the verification can confirm the e-prop gradient approximates
the true BPTT gradient; it is not used for training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .surrogate import spike_fn, build_freq_dataset  # noqa: F401  (re-export)


# ---------------- input delay embedding ----------------

def delay_embed(x: torch.Tensor, D: int) -> torch.Tensor:
    """Turn a (B, T) scalar signal into (B, T, D): at each step t the
    feature is the length-D window [x[t-D+1], ..., x[t]] (zero-padded at
    the start). This gives the LIF layer a D-tap temporal receptive
    field, so W_in becomes a genuine (N, D) learnable filter bank."""
    B, T = x.shape
    padded = torch.cat([torch.zeros(B, D - 1, dtype=x.dtype, device=x.device), x], dim=1)
    return torch.stack([padded[:, t:t + D] for t in range(T)], dim=1)  # (B, T, D)


# ---------------- e-prop SNN classifier ----------------

@dataclass
class EpropConfig:
    input_dim_D: int = 12        # delay-embedding window length
    n_hidden: int = 96
    n_classes: int = 3
    beta: float = 0.9            # membrane leak (also the eligibility leak)
    v_thresh: float = 1.0
    surrogate_beta: float = 10.0


class EpropSNN:
    """A feed-forward LIF classifier trained by e-prop.

    Weights (W_in, readout_w, readout_b) are PLAIN tensors with
    requires_grad=False - e-prop updates them with explicitly computed,
    forward-only gradients, so autograd is never invoked during training.
    """

    def __init__(self, cfg: EpropConfig | None = None, seed: int = 0):
        self.cfg = cfg or EpropConfig()
        c = self.cfg
        g = torch.Generator().manual_seed(seed)
        self.W_in = torch.randn(c.n_hidden, c.input_dim_D, generator=g) / (c.input_dim_D ** 0.5)
        self.readout_w = torch.randn(c.n_classes, c.n_hidden, generator=g) / (c.n_hidden ** 0.5)
        self.readout_b = torch.zeros(c.n_classes)
        # all plain tensors, requires_grad stays False
        self.n_hidden = c.n_hidden
        self.n_classes = c.n_classes
        self.D = c.input_dim_D
        self.beta = c.beta
        self.v_thresh = c.v_thresh
        self.surrogate_beta = c.surrogate_beta

    @torch.no_grad()
    def forward_with_eligibility(self, x: torch.Tensor):
        """x: (B, T). Forward-only pass.

        Returns (logits, rate, G):
          logits (B, n_classes)
          rate   (B, n_hidden)            per-neuron mean firing rate
          G      (B, n_hidden, D)         accumulated eligibility,
                                          G[b,j,i] = sum_t psi_j[t]*eps_i[t]
        """
        B, T = x.shape
        N, D = self.n_hidden, self.D
        xenc = delay_embed(x, D)                       # (B, T, D)
        V = torch.zeros(B, N)
        s = torch.zeros(B, N)
        eps = torch.zeros(B, D)                        # eligibility vector
        G = torch.zeros(B, N, D)                       # eligibility trace sum
        spike_sum = torch.zeros(B, N)
        for t in range(T):
            xt = xenc[:, t]                            # (B, D)
            I = xt @ self.W_in.t()                     # (B, N)
            V = self.beta * V + I - self.v_thresh * s  # membrane (reset-by-subtraction)
            x_thr = V - self.v_thresh
            s = (x_thr > 0).to(x.dtype)                # exact binary spike
            psi = 1.0 / (1.0 + self.surrogate_beta * x_thr.abs()) ** 2   # pseudo-derivative
            eps = self.beta * eps + xt                 # leaky input trace
            G = G + psi.unsqueeze(2) * eps.unsqueeze(1)  # accumulate (B,N,D)
            spike_sum = spike_sum + s
        rate = spike_sum / T
        logits = rate @ self.readout_w.t() + self.readout_b
        return logits, rate, G

    def bptt_gradient_W_in(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """The TRUE BPTT gradient of W_in, via autograd through the
        surrogate spike function. Used ONLY by the verification to check
        that e-prop's forward-only gradient approximates it - never used
        for training."""
        W = self.W_in.clone().detach().requires_grad_(True)
        ro_w = self.readout_w.clone().detach()
        ro_b = self.readout_b.clone().detach()
        B, T = x.shape
        N, D = self.n_hidden, self.D
        xenc = delay_embed(x, D)
        V = torch.zeros(B, N)
        s = torch.zeros(B, N)
        spike_sum = torch.zeros(B, N)
        for t in range(T):
            xt = xenc[:, t]
            I = xt @ W.t()
            V = self.beta * V + I - self.v_thresh * s
            s = spike_fn(V - self.v_thresh, self.surrogate_beta)
            spike_sum = spike_sum + s
        rate = spike_sum / T
        logits = rate @ ro_w.t() + ro_b
        F.cross_entropy(logits, labels).backward()
        return W.grad.detach()


# ---------------- e-prop gradients + training ----------------

def eprop_gradients(model: EpropSNN, xb: torch.Tensor, yb: torch.Tensor):
    """Compute the e-prop gradients (forward-only, no autograd).

    Returns (dW_in, d_readout_w, d_readout_b, ce_loss).
    """
    logits, rate, G = model.forward_with_eligibility(xb)
    B, T = xb.shape
    probs = torch.softmax(logits, dim=-1)
    onehot = F.one_hot(yb, model.n_classes).to(probs.dtype)
    error = probs - onehot                               # dL/dlogits  (B, n_classes)
    ce = float(-(onehot * torch.log(probs.clamp_min(1e-9))).sum(-1).mean())
    # readout: standard one-layer gradients (logits = rate @ ro_w.T + ro_b)
    d_readout_w = error.t() @ rate / B                   # (n_classes, n_hidden)
    d_readout_b = error.mean(dim=0)                       # (n_classes,)
    # e-prop gradient for W_in:
    #   learning signal L_j = sum_c readout_w[c,j] * error_c
    #   dL/dW_in[j,i] = (1/T) * mean_b ( L[b,j] * G[b,j,i] )
    L = error @ model.readout_w                           # (B, n_hidden)
    dW_in = (L.unsqueeze(2) * G).mean(dim=0) / T          # (n_hidden, D)
    return dW_in, d_readout_w, d_readout_b, ce


def train_eprop(model: EpropSNN, inputs: torch.Tensor, labels: torch.Tensor,
                 steps: int = 150, lr: float = 0.5, batch_size: int = 64,
                 momentum: float = 0.9, seed: int = 0) -> list[float]:
    """Train an EpropSNN with e-prop (SGD + momentum). Forward-only -
    autograd is never called. Returns the per-step cross-entropy history.
    """
    g = torch.Generator().manual_seed(seed)
    n = inputs.shape[0]
    vW = torch.zeros_like(model.W_in)
    vRw = torch.zeros_like(model.readout_w)
    vRb = torch.zeros_like(model.readout_b)
    history: list[float] = []
    for _ in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        dW_in, dro_w, dro_b, ce = eprop_gradients(model, inputs[idx], labels[idx])
        vW = momentum * vW + dW_in
        vRw = momentum * vRw + dro_w
        vRb = momentum * vRb + dro_b
        model.W_in = model.W_in - lr * vW
        model.readout_w = model.readout_w - lr * vRw
        model.readout_b = model.readout_b - lr * vRb
        history.append(ce)
    return history


@torch.no_grad()
def evaluate_eprop(model: EpropSNN, inputs: torch.Tensor,
                    labels: torch.Tensor) -> dict:
    """Accuracy + mean firing rate on a held-out set."""
    logits, rate, _ = model.forward_with_eligibility(inputs)
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return {"accuracy": acc, "firing_rate": rate.mean().item()}
