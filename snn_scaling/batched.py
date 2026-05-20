"""Batched reservoir: the feature-extraction speedup.

The verified SpikingReservoir / LIFPopulation / SparseSynapses are
single-sample (N,) simulators. extract_centered_features runs them once
PER SAMPLE in a Python for-loop, so a 220-sample feature pass is 220
sequential reservoir runs - which measured as ~80% of every benchmark
and ASI-Evolve run.

BatchedReservoir runs all B samples in parallel: membrane state is
(B, N), synapse conductance is (B, E), and the only Python loop is over
T timesteps (unavoidable - the recurrence couples step to step). The
LIF exponential-Euler update and the sparse synaptic scatter are
elementwise / index_add ops that batch cleanly.

It does NOT modify the verified classes - it borrows an existing
SpikingReservoir's connectivity and parameters (so it is the SAME
reservoir) and re-runs the dynamics batched. With noise disabled it is
bit-identical to the sequential simulator; with noise it is
statistically equivalent (the noise stream is drawn (B, N) at once
instead of (N,) per sample).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .reservoir import SpikingReservoir


def batched_drive(inputs: torch.Tensor, N: int, scale: float = 5.0,
                   proj_seed: int = 7777, device: torch.device | str = "cpu") -> torch.Tensor:
    """Project a batch of (B, T) signals to (B, T, N) per-neuron drive,
    using the same fixed random projection as distill._drive_input."""
    g = torch.Generator().manual_seed(proj_seed)
    P = ((torch.rand(N, generator=g) - 0.5) * 2.0).to(device=device)
    return inputs.to(device=device).unsqueeze(-1) * P * scale          # (B, T, N)


class BatchedReservoir:
    """Runs B independent samples through a reservoir's dynamics in
    parallel. Borrows connectivity + neuron params from an existing
    SpikingReservoir, so for a given config seed it is the SAME network."""

    def __init__(self, reservoir: SpikingReservoir, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.n = reservoir.n
        pop = reservoir.cluster.population
        syn = reservoir.cluster.synapses[0]
        # LIF per-neuron params, shape (N,) - broadcast over the batch dim
        self._tau_m = pop._tau_m.to(self.device)
        self._V_rest = pop._V_rest.to(self.device)
        self._V_thresh = pop._V_thresh.to(self.device)
        self._V_reset = pop._V_reset.to(self.device)
        self._R_m = pop._R_m.to(self.device)
        self._refractory = pop._refractory.to(self.device)
        self._noise_std = pop._noise_std.to(self.device)
        # synapse edge tensors
        self.edge_pre = syn.edge_pre.to(self.device)
        self.edge_post = syn.edge_post.to(self.device)
        self.E_syn = syn.E_syn.to(self.device)
        self.g_max = syn.g_max.to(self.device)
        self.tau_syn = syn.tau_syn.to(self.device)
        self.E = int(self.edge_pre.numel())

    @torch.no_grad()
    def run_collect_V(self, drive: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """drive: (B, T, N) per-step per-neuron input current.
        Returns Vs: (B, T, N) - the membrane trace, all samples in parallel.

        Mirrors SNNCluster.step exactly (post-currents -> LIF exponential
        Euler -> deliver spikes -> decay), batched over B."""
        B, T, N = drive.shape
        dev = self.device
        V = self._V_rest.expand(B, N).clone()                # all samples at V_rest
        refr = torch.zeros(B, N, device=dev)
        g = torch.zeros(B, self.E, device=dev)               # per-(sample, edge) conductance
        Vs = torch.zeros(B, T, N, device=dev)
        noisy = bool(torch.any(self._noise_std > 0))
        sqrt_dt = math.sqrt(dt)

        for k in range(T):
            # --- synaptic post-currents (batched scatter) ---
            V_at_post = V[:, self.edge_post]                          # (B, E)
            I_edge = g * (self.E_syn - V_at_post)                     # (B, E)
            I_syn = torch.zeros(B, N, device=dev)
            g_syn = torch.zeros(B, N, device=dev)
            I_syn.index_add_(1, self.edge_post, I_edge)
            g_syn.index_add_(1, self.edge_post, g)
            I_total = I_syn + drive[:, k]

            # --- LIF exponential-Euler step (batched) ---
            was_refr = refr > 0
            refr = (refr - dt).clamp_(min=0.0)
            one_plus = 1.0 + self._R_m * g_syn
            V_ss = (self._V_rest + self._R_m * (I_total + g_syn * V)) / one_plus
            tau_eff = self._tau_m / one_plus
            decay = torch.exp(-dt / tau_eff)
            V_new = V_ss + (V - V_ss) * decay
            if noisy:
                V_new = V_new + torch.randn(B, N, device=dev) * self._noise_std * sqrt_dt
            V_new = torch.where(was_refr, self._V_reset, V_new)
            spike = (V_new >= self._V_thresh) & (~was_refr)
            V_new = torch.where(spike, self._V_reset, V_new)
            refr = torch.where(spike, self._refractory, refr)
            V = V_new
            Vs[:, k] = V

            # --- deliver spikes + decay (batched) ---
            fire_e = spike[:, self.edge_pre]                          # (B, E)
            g = g + self.g_max * fire_e.to(g.dtype)
            g = g * torch.exp(-dt / self.tau_syn)

        return Vs


@torch.no_grad()
def extract_features_batched(reservoir: SpikingReservoir, batch,
                              device: torch.device | str = "cpu",
                              center: bool = True, l2_normalize: bool = True,
                              scale: float = 5.0, proj_seed: int = 7777) -> torch.Tensor:
    """Batched drop-in for rl_tasks.extract_centered_features.

    Returns (n_samples, 3*N): per-sample (mean, std, range) of the
    late-half membrane trace, optionally centered + L2-normalised - the
    identical feature recipe, computed for every sample in one pass.
    """
    br = BatchedReservoir(reservoir, device=device)
    drive = batched_drive(batch.inputs, br.n, scale=scale,
                           proj_seed=proj_seed, device=device)
    Vs = br.run_collect_V(drive)                                   # (B, T, N)
    T = Vs.shape[1]
    late = Vs[:, T // 2:]
    feats = torch.cat([
        late.mean(dim=1),
        late.std(dim=1),
        late.max(dim=1).values - late.min(dim=1).values,
    ], dim=-1)
    if center:
        feats = feats - feats.mean(dim=0, keepdim=True)
    if l2_normalize:
        feats = F.normalize(feats, dim=-1)
    return feats
