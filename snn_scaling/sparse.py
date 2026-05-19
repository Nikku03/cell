"""Trick 1: enforced sparse activation (k-winners-take-all).

Cortex maintains ~1-2% active neurons at any moment via global lateral
inhibition. Naive SNN training often produces higher activity, which
hurts both representational efficiency (collisions) and downstream
compute (every spike triggers synaptic work).

kWTA gates the spike-emission step so that **at most K neurons fire per
timestep**, chosen as the top-K most depolarised (closest to threshold).
This is a hard, global constraint, not a soft penalty.

The result:
  - exact control of active fraction (K/N) across the simulation
  - signal-to-noise improvement (the K most-driven neurons carry
    information; the rest stay silent)
  - per-step compute proportional to K, not N, for spike-propagation
    work (synapse banks index by spiked pre, so fewer spikes = less work)

Implementation: subclass `LIFPopulation` to inject a top-K mask between
the threshold check and the actual spike emission. The membrane
dynamics are unchanged; only which neurons are *allowed* to fire is.
"""

from __future__ import annotations

import math

import torch

from .population import LIFPopulation, NeuronParams


class kWTAPopulation(LIFPopulation):
    """LIFPopulation with a global k-winners-take-all gate.

    At each step, the membrane is updated normally (exponential Euler).
    Neurons whose V crossed threshold are candidates. From those, only
    the top-K by V are allowed to spike; the rest have V clamped just
    below threshold (so the missed spike's "intent" lives on into the
    next step rather than being completely lost).

    `k` can be set explicitly, or as a fraction of `n` via `k_fraction`.
    """

    def __init__(
        self,
        n_neurons: int,
        k: int | None = None,
        k_fraction: float | None = None,
        params: NeuronParams | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        retain_below_threshold: bool = True,
    ):
        super().__init__(n_neurons, params=params, device=device, dtype=dtype)
        if k is None and k_fraction is None:
            raise ValueError("specify k or k_fraction")
        if k is None:
            k = max(1, int(round(k_fraction * n_neurons)))
        assert 1 <= k <= n_neurons
        self.k = int(k)
        self.retain_below_threshold = bool(retain_below_threshold)

    @torch.no_grad()
    def step(
        self,
        I_syn: torch.Tensor,
        g_syn: torch.Tensor,
        dt: float,
        t: float,
    ) -> torch.Tensor:
        # Copy of LIFPopulation.step, with a top-K gate between threshold and emission.
        was_refr = self.refr > 0
        self.refr.sub_(dt).clamp_(min=0.0)

        one_plus = 1.0 + self._R_m * g_syn
        V_ss = (self._V_rest + self._R_m * (I_syn + g_syn * self.V)) / one_plus
        tau_eff = self._tau_m / one_plus
        decay = torch.exp(-dt / tau_eff)
        V_new = V_ss + (self.V - V_ss) * decay

        if torch.any(self._noise_std > 0):
            V_new = V_new + torch.randn_like(V_new) * self._noise_std * math.sqrt(dt)

        V_new = torch.where(was_refr, self._V_reset, V_new)

        # Threshold crossing candidates
        candidates = (V_new >= self._V_thresh) & (~was_refr)
        n_cand = int(candidates.sum().item())

        if n_cand <= self.k:
            # All candidates are allowed to fire
            spike = candidates
        else:
            # Keep only the top-K by membrane potential
            # Restrict topk to candidates by masking V_new to -inf elsewhere
            v_masked = torch.where(candidates, V_new, torch.full_like(V_new, float("-inf")))
            _, top_idx = torch.topk(v_masked, self.k)
            spike = torch.zeros_like(candidates)
            spike[top_idx] = True
            # The k-th-place + below winners had V >= V_thresh but did not fire.
            # Optionally retain them just below threshold so the bias toward firing
            # carries into the next step. Otherwise they'd get reset like spikers.
            if self.retain_below_threshold:
                missed = candidates & (~spike)
                V_new = torch.where(missed, self._V_thresh - 1e-3, V_new)

        V_new = torch.where(spike, self._V_reset, V_new)
        self.refr = torch.where(spike, self._refractory, self.refr)
        self.last_spike_t = torch.where(
            spike, torch.full_like(self.last_spike_t, t), self.last_spike_t
        )

        self.V = V_new
        self.spiked = spike
        return spike

    def active_fraction(self) -> float:
        """Convenience: K / N as a number for diagnostics."""
        return self.k / self.n
