"""Trick 3: reservoir computing (echo state / liquid state machine).

The reservoir computing insight: a fixed, randomly connected recurrent
network produces rich high-dimensional nonlinear dynamics. You can
extract task-relevant information from those dynamics with a tiny
LINEAR readout trained by ridge regression -- no end-to-end gradient
backprop through the recurrence.

For spiking neurons, this is called a Liquid State Machine (Maass 2002).
The reservoir is a recurrent SNN driven by inputs; its instantaneous
spike pattern is projected to outputs via a learned linear map.

Why this scales effective compute: a small reservoir of N neurons,
sampled over T timesteps, produces an N*T-dimensional feature trajectory.
The readout sees that whole trajectory, so the effective representational
capacity scales with N*T, not N alone.

This module provides:
  - SpikingReservoir: a small random-recurrent SNN driven by input
                        spikes / currents
  - extract_readout_features: turn reservoir spike history into a
                                fixed-dim feature vector
  - train_linear_readout: ridge-regression on collected features
  - LiquidStateMachine: convenience class combining the above
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .population import (
    LIFPopulation,
    NeuronParams,
    SparseSynapses,
    SNNCluster,
    SYN_EXC,
    SYN_INH,
)


@dataclass
class ReservoirConfig:
    n_reservoir: int = 500
    p_recurrent: float = 0.10
    g_exc: float = 0.04
    g_inh: float = 0.30
    exc_fraction: float = 0.8
    tau_syn: float = 5.0
    noise_std: float = 0.5
    spectral_radius: float | None = None       # if set, rescale to this
    seed: int = 0


class SpikingReservoir(nn.Module):
    """A random recurrent spiking network. Fixed weights; never trained.

    Inputs are injected as per-neuron currents (see `step_with_input`).
    Spike outputs are read directly off the population.
    """

    def __init__(self, cfg: ReservoirConfig, device: torch.device | str = "cpu"):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        N = cfg.n_reservoir
        device = torch.device(device)

        pop = LIFPopulation(N, params=NeuronParams(noise_std=cfg.noise_std), device=device)

        # Random recurrent connectivity. E/I split: first exc_fraction are E.
        n_e = int(round(cfg.exc_fraction * N))
        # Each neuron has on average p_recurrent * N out-edges.
        # Sample per-source binomial.
        total_pre, total_post, total_kind, total_g = [], [], [], []
        counts = torch.distributions.Binomial(
            total_count=N - 1, probs=torch.tensor(cfg.p_recurrent, dtype=torch.float64),
        ).sample(sample_shape=(N,)).long()
        for i in range(N):
            c = int(counts[i].item())
            if c == 0:
                continue
            # Sample c targets without replacement, excluding self
            mask = torch.ones(N, dtype=torch.bool)
            mask[i] = False
            avail = torch.arange(N)[mask]
            chosen = avail[torch.randperm(N - 1, generator=g)[:c]]
            total_pre.append(torch.full((c,), i, dtype=torch.long))
            total_post.append(chosen)
            kind = SYN_EXC if i < n_e else SYN_INH
            gmax = cfg.g_exc if kind == SYN_EXC else cfg.g_inh
            total_kind.append(torch.full((c,), kind, dtype=torch.long))
            total_g.append(torch.full((c,), gmax))
        if total_pre:
            pre = torch.cat(total_pre)
            post = torch.cat(total_post)
            kind = torch.cat(total_kind)
            gmax = torch.cat(total_g)
        else:
            pre = torch.empty(0, dtype=torch.long)
            post = torch.empty(0, dtype=torch.long)
            kind = torch.empty(0, dtype=torch.long)
            gmax = torch.empty(0)
        tau = torch.full((pre.numel(),), cfg.tau_syn)
        syn = SparseSynapses(
            n_pre=N, n_post=N,
            edge_pre=pre, edge_post=post,
            kind=kind, g_max=gmax, tau_syn=tau,
            device=device,
        )
        self.cluster = SNNCluster(pop)
        self.cluster.add_synapses(syn)
        self.n = N

    def reset(self) -> None:
        self.cluster.reset_state()

    @torch.no_grad()
    def step(self, input_current: torch.Tensor | None, dt: float, t: float) -> torch.Tensor:
        return self.cluster.step(dt=dt, t=t, external_current=input_current)

    @torch.no_grad()
    def run(
        self,
        inputs: torch.Tensor,           # (T, N) per-step external current
        dt: float = 1.0,
        record_spike_rates: bool = True,
        record_membrane: bool = False,
        bin_ms: float = 5.0,
    ) -> dict:
        """Run for T steps with `inputs[t]` as per-neuron drive.

        Returns time-binned spike rates and / or membrane potentials.
        """
        T = inputs.shape[0]
        assert inputs.shape[1] == self.n
        if record_spike_rates:
            n_bins = max(1, int(round(T * dt / bin_ms)))
            steps_per_bin = max(1, T // n_bins)
            rates = torch.zeros(n_bins, self.n)
            count_in_bin = 0
            cur_bin = 0
        if record_membrane:
            membranes = torch.zeros(T, self.n)

        for k in range(T):
            spike = self.cluster.step(dt=dt, t=k * dt, external_current=inputs[k])
            if record_spike_rates:
                rates[cur_bin] += spike.float()
                count_in_bin += 1
                if count_in_bin >= steps_per_bin and cur_bin < n_bins - 1:
                    rates[cur_bin] = rates[cur_bin] / count_in_bin * 1000.0 / dt
                    cur_bin += 1
                    count_in_bin = 0
            if record_membrane:
                membranes[k] = self.cluster.population.V.clone()
        if record_spike_rates and count_in_bin > 0:
            rates[cur_bin] = rates[cur_bin] / count_in_bin * 1000.0 / dt

        out: dict = {}
        if record_spike_rates:
            out["rates"] = rates           # (n_bins, n)
            out["bin_ms"] = bin_ms
        if record_membrane:
            out["V"] = membranes           # (T, n)
        return out


# ---------------- linear readout ----------------

def train_linear_readout(
    features: torch.Tensor,         # (M_samples, F_features)
    targets: torch.Tensor,          # (M_samples, K_outputs) or (M_samples,)
    ridge: float = 1e-2,
) -> torch.Tensor:
    """Ridge regression: returns W of shape (F, K).

    Solves (X^T X + ridge*I) W = X^T y in closed form.
    """
    if targets.dim() == 1:
        targets = targets.unsqueeze(-1)
    X = features
    y = targets.to(dtype=X.dtype)
    F = X.shape[1]
    XtX = X.t() @ X
    reg = ridge * torch.eye(F, dtype=X.dtype, device=X.device)
    XtY = X.t() @ y
    W = torch.linalg.solve(XtX + reg, XtY)
    return W


def predict_linear(W: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Apply readout matrix to features."""
    return features @ W


def quantize_ternary(W: torch.Tensor) -> torch.Tensor:
    """BitNet-style ternary quantization of a weight matrix.

    Every weight is snapped to the nearest of three levels {-s, 0, +s},
    where the shared scale s = mean(|W|). The returned tensor therefore
    holds at most 3 distinct values and can drop straight into
    predict_linear in place of the float32 W.

    A spiking substrate's recurrent synapses are already effectively
    ternary (uniform conductance per E/I kind + zero for absent edges).
    The one component with genuine continuous weights is the trained
    linear readout - so this is what "ternary weights" means for our
    architecture: a 2-bit readout. Classification only depends on which
    side of a boundary a sample lands, so it tolerates this quantization
    far better than regression would.
    """
    scale = W.abs().mean().clamp_min(1e-12)
    q = torch.clamp(torch.round(W / scale), -1.0, 1.0)
    return q * scale


# ---------------- end-to-end LSM ----------------

class LiquidStateMachine:
    """SpikingReservoir + ridge-trained linear readout.

    Standard usage:
      lsm = LiquidStateMachine(cfg)
      features_train, y_train = collect_episode_features(lsm, train_inputs, train_labels)
      lsm.fit(features_train, y_train)
      preds = lsm.predict(features_test)
    """

    def __init__(self, cfg: ReservoirConfig, device: torch.device | str = "cpu"):
        self.reservoir = SpikingReservoir(cfg, device=device)
        self.W: torch.Tensor | None = None

    def reset(self) -> None:
        self.reservoir.reset()

    def step_and_feature(
        self, input_current: torch.Tensor, dt: float, t: float,
    ) -> torch.Tensor:
        """One step; return the readout feature for this step (= spike state)."""
        spike = self.reservoir.step(input_current, dt=dt, t=t)
        # Feature is the current membrane state (richer than just spikes).
        return self.reservoir.cluster.population.V.clone()

    def fit(self, features: torch.Tensor, targets: torch.Tensor, ridge: float = 1e-2) -> None:
        self.W = train_linear_readout(features, targets, ridge=ridge)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if self.W is None:
            raise RuntimeError("readout not trained; call fit() first")
        return predict_linear(self.W, features)
