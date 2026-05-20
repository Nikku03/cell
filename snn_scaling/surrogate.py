"""Trick 8 / SparseSpike-GEMM steal #3: surrogate-gradient SNN training.

Phases 0-7 use a FIXED random spiking substrate plus a closed-form linear
readout - nothing inside the SNN is trained. This module makes the
spiking weights themselves trainable by backpropagation through time
(BPTT), using the surrogate-gradient trick for the non-differentiable
spike threshold (Neftci, Mostafa & Zenke 2019).

Why a separate module: `LIFPopulation` / `SparseSynapses` are
`@torch.no_grad` biophysical-unit simulators - correct for the verified
phases but not differentiable. `DifferentiableLIF` here is a deliberately
distinct, normalised-unit (V_thresh = 1.0, leak beta in (0,1)) layer
whose forward pass is an autograd-traceable unrolled loop.

Components
----------
  SpikeFunction        autograd.Function: forward = Heaviside,
                       backward = fast-sigmoid surrogate derivative
  DifferentiableLIF    a layer of LIF neurons, BPTT-differentiable
  SurrogateSNNClassifier   input projection -> DifferentiableLIF ->
                       per-neuron firing rate -> linear class readout
  firing_rate_loss     L1 penalty pulling firing rate toward a target
  spike_budget_loss    hinge penalty for exceeding a per-sample spike cap
  train_surrogate_snn  the BPTT training loop (Adam)

The activity regulariser and the spike-budget constraint are the
"enforcing biological sparsity during training" ideas from the spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- surrogate-gradient spike function ----------------

class SpikeFunction(torch.autograd.Function):
    """Heaviside spike with a fast-sigmoid surrogate gradient.

      forward:   s = 1 if x > 0 else 0          (x = V - V_thresh)
      backward:  ds/dx ~= 1 / (1 + b*|x|)^2     (Zenke & Ganguli 2018)

    The true derivative of the Heaviside is 0 everywhere (and undefined
    at 0), so without the surrogate no gradient reaches the spiking
    weights and training is impossible. The surrogate is used ONLY on
    the backward pass; the forward pass emits exact binary spikes.
    """

    @staticmethod
    def forward(ctx, x, surrogate_beta):
        ctx.save_for_backward(x)
        ctx.surrogate_beta = float(surrogate_beta)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        b = ctx.surrogate_beta
        surrogate = 1.0 / (1.0 + b * x.abs()) ** 2
        return grad_output * surrogate, None


def spike_fn(x: torch.Tensor, surrogate_beta: float = 10.0) -> torch.Tensor:
    """Apply the surrogate-gradient spike nonlinearity."""
    return SpikeFunction.apply(x, surrogate_beta)


# ---------------- differentiable LIF layer ----------------

@dataclass
class SurrogateLIFConfig:
    beta: float = 0.9            # membrane leak per step = exp(-dt / tau)
    v_thresh: float = 1.0        # spike threshold (normalised units)
    surrogate_beta: float = 10.0 # steepness of the fast-sigmoid surrogate
    reset: str = "subtract"      # "subtract" (V -= v_thresh) or "zero"


class DifferentiableLIF(nn.Module):
    """A layer of leaky integrate-and-fire neurons, differentiable via BPTT.

    Normalised units: V starts at 0, threshold 1.0, leak beta in (0,1).
    Per timestep:

        V <- beta * V + input_current[t]
        s  = SpikeFunction(V - v_thresh)
        V <- V - v_thresh * s        # reset-by-subtraction (differentiable)

    Has no learnable parameters of its own; the trainable weights live in
    the input projection that feeds `input_current`. The whole unrolled
    loop is autograd-traceable, so gradients flow back through every
    timestep and every spike (via the surrogate).
    """

    def __init__(self, cfg: SurrogateLIFConfig | None = None):
        super().__init__()
        self.cfg = cfg or SurrogateLIFConfig()

    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """input_current: (B, T, N) -> spikes: (B, T, N), values in {0, 1}."""
        B, T, N = input_current.shape
        device = input_current.device
        V = torch.zeros(B, N, device=device, dtype=input_current.dtype)
        s = torch.zeros(B, N, device=device, dtype=input_current.dtype)
        spikes = []
        for t in range(T):
            V = self.cfg.beta * V + input_current[:, t]
            s = spike_fn(V - self.cfg.v_thresh, self.cfg.surrogate_beta)
            spikes.append(s)
            if self.cfg.reset == "subtract":
                V = V - self.cfg.v_thresh * s
            else:                                    # hard reset toward 0
                V = V * (1.0 - s)
        return torch.stack(spikes, dim=1)


# ---------------- surrogate-gradient SNN classifier ----------------

class SurrogateSNNClassifier(nn.Module):
    """input projection -> DifferentiableLIF -> firing-rate readout.

    The input is a (B, T) scalar temporal signal. `input_proj` maps the
    per-step scalar to an N-dim per-neuron drive; the LIF layer turns
    that into spike trains; the per-neuron mean firing rate over time is
    the feature vector for a linear class readout.

    `train_lif=False` freezes the input projection at its random init -
    that is the fixed-random-substrate (reservoir-style) baseline, where
    only the readout learns. `train_lif=True` trains the projection too,
    with gradients flowing through the spikes via the surrogate. The
    difference between the two is exactly the value of training the SNN.
    """

    def __init__(self, n_hidden: int = 128, n_classes: int = 3,
                 lif_cfg: SurrogateLIFConfig | None = None,
                 train_lif: bool = True, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.input_proj = nn.Linear(1, n_hidden)
        self.lif = DifferentiableLIF(lif_cfg or SurrogateLIFConfig())
        self.readout = nn.Linear(n_hidden, n_classes)
        self.train_lif = train_lif
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        if not train_lif:
            for p in self.input_proj.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        """x: (B, T) -> (logits (B, n_classes), spikes (B, T, n_hidden))."""
        I = self.input_proj(x.unsqueeze(-1))      # (B, T, n_hidden)
        spikes = self.lif(I)                       # (B, T, n_hidden)
        rate = spikes.mean(dim=1)                  # (B, n_hidden)
        logits = self.readout(rate)
        return logits, spikes


# ---------------- sparsity losses ----------------

def firing_rate_loss(spikes: torch.Tensor, target_rate: float) -> torch.Tensor:
    """L1 penalty pulling the per-neuron mean firing rate toward a target.
    spikes: (B, T, N). The brain runs at a few percent activity; a low
    target_rate trains the network toward that regime."""
    rate = spikes.mean(dim=(0, 1))                 # (N,)
    return (rate - target_rate).abs().mean()


def spike_budget_loss(spikes: torch.Tensor, budget_per_sample: float) -> torch.Tensor:
    """Hinge penalty for exceeding a per-sample total-spike budget.
    Pushes the network to spend spikes only where they matter."""
    total = spikes.sum(dim=(1, 2))                 # (B,)
    return F.relu(total - budget_per_sample).mean()


# ---------------- training loop ----------------

@dataclass
class SurrogateTrainConfig:
    steps: int = 120
    lr: float = 2e-3
    batch_size: int = 64
    lambda_rate: float = 0.0
    target_rate: float = 0.10
    lambda_budget: float = 0.0
    budget_per_sample: float = 1.0e9
    seed: int = 0


def train_surrogate_snn(model: SurrogateSNNClassifier,
                         inputs: torch.Tensor, labels: torch.Tensor,
                         cfg: SurrogateTrainConfig | None = None) -> dict:
    """BPTT training of a SurrogateSNNClassifier.

    inputs: (n_samples, T) float, labels: (n_samples,) long.
    Returns a history dict (ce loss, total loss per step).
    """
    cfg = cfg or SurrogateTrainConfig()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    g = torch.Generator().manual_seed(cfg.seed)
    n = inputs.shape[0]
    ce_hist, loss_hist = [], []
    model.train()
    for step in range(cfg.steps):
        idx = torch.randint(0, n, (cfg.batch_size,), generator=g)
        xb, yb = inputs[idx], labels[idx]
        logits, spikes = model(xb)
        ce = F.cross_entropy(logits, yb)
        loss = ce
        if cfg.lambda_rate > 0:
            loss = loss + cfg.lambda_rate * firing_rate_loss(spikes, cfg.target_rate)
        if cfg.lambda_budget > 0:
            loss = loss + cfg.lambda_budget * spike_budget_loss(spikes, cfg.budget_per_sample)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ce_hist.append(float(ce.detach()))
        loss_hist.append(float(loss.detach()))
    return {"ce": ce_hist, "loss": loss_hist}


@torch.no_grad()
def evaluate_surrogate_snn(model: SurrogateSNNClassifier,
                            inputs: torch.Tensor, labels: torch.Tensor) -> dict:
    """Evaluate accuracy + spiking statistics on a held-out set."""
    model.eval()
    logits, spikes = model(inputs)
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return {
        "accuracy": acc,
        "firing_rate": spikes.mean().item(),                       # global mean
        "total_spikes_per_sample": spikes.sum(dim=(1, 2)).mean().item(),
    }


# ---------------- dataset helper ----------------

def build_freq_dataset(n_train: int = 240, n_test: int = 120, T: int = 80,
                         noise_std: float = 0.5, n_classes: int = 3,
                         seed: int = 0):
    """Temporal waveform-classification task from distill's make_freq_task,
    standardised with train-set statistics.

    n_classes in {2,3,4}: sine / square / white-noise / chirp. The chirp
    (class 3) is sine-like and harder to separate, so 4-class is a
    genuinely harder task than 3-class.

    Returns (x_train, y_train, x_test, y_test) with x: (n, T) float,
    y: (n,) long.
    """
    from .distill import make_freq_task
    train = make_freq_task(n_classes=n_classes, n_samples=n_train, T=T,
                            noise_std=noise_std, seed=seed)
    test = make_freq_task(n_classes=n_classes, n_samples=n_test, T=T,
                           noise_std=noise_std, seed=seed + 9991)
    mu = train.inputs.mean()
    sd = train.inputs.std().clamp_min(1e-6)
    x_train = (train.inputs - mu) / sd
    x_test = (test.inputs - mu) / sd
    return x_train, train.labels, x_test, test.labels
