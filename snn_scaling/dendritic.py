"""Trick 7: dendritic compartments (per-neuron sub-network).

A point neuron sums its synaptic input linearly: I_soma = Σ_e w_e · x_e.
That is a single linear map followed by a threshold, so a point neuron
can only realise linearly-separable functions of its inputs.

A pyramidal cell, in contrast, has many dendritic branches. Each branch
sums inputs locally and applies a SATURATING / SIGMOIDAL nonlinearity
BEFORE the signal reaches the soma. So the soma actually sees

  I_soma = Σ_k σ(Σ_{e ∈ branch k} w_e · x_e)

which is a 2-layer network embedded inside a single cell. Poirazi & Mel
showed that this lets one pyramidal cell compute XOR, parity, and other
non-linearly-separable functions of its inputs. The effective compute
of a network of N such cells is N · K (with K dendrites each), without
the connection-count cost of a wider point-neuron layer.

This module provides:
  - DendriticParams: extends NeuronParams with dendrite-count + nonlinearity
  - DendriticPopulation: N LIF somata, each with K compartments
  - DendriticSynapses: sparse synapses targeting (neuron, dendrite)
  - DendriticCluster: population + synapse banks (mirrors SNNCluster)
  - build_xor_neuron: a 2-dendrite single-neuron preset that computes XOR

By default the dendrites are stateless (the nonlinearity is applied to
the instantaneous per-compartment synaptic current). Setting
`dendrite_leak > 0` turns each branch into a LEAKY ACCUMULATOR:

  d_b   <- dendrite_leak * d_b + input_b      # per-branch integration
  g_b   = nonlinear((d_b - threshold) * gain) # nonlinearity on the state

so a branch integrates its input over time and can detect temporal
coincidence (sub-threshold inputs that accumulate). dendrite_leak=0
recovers the original stateless model exactly. Per-neuron state is then
N (soma V) + N*K (branch accumulators).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .population import (
    LIFPopulation,
    NeuronParams,
    SparseSynapses,
    SYN_EXC,
    SYN_INH,
    _E_SYN,
)


# ---------------- params ----------------

@dataclass
class DendriticParams:
    """All point-neuron params + dendrite count + nonlinearity choice."""
    # LIF soma params
    tau_m: float = 20.0
    V_rest: float = -70.0
    V_thresh: float = -50.0
    V_reset: float = -75.0
    R_m: float = 10.0
    refractory: float = 2.0
    noise_std: float = 0.0
    # Dendritic compartment params
    n_dendrites: int = 4
    nonlinearity: str = "sigmoid"          # "sigmoid", "relu", "tanh"
    dendrite_gain: float = 4.0             # input scaling before nonlinearity
    dendrite_threshold: float = 0.5        # half-activation point
    soma_weight: float = 1.0               # post-nonlinearity weight into soma
    dendrite_leak: float = 0.0             # per-branch leaky-accumulator retention;
                                           # 0.0 = stateless (original model),
                                           # 0<leak<1 = integrates input over time

    def to_neuron_params(self) -> NeuronParams:
        return NeuronParams(
            tau_m=self.tau_m, V_rest=self.V_rest, V_thresh=self.V_thresh,
            V_reset=self.V_reset, R_m=self.R_m, refractory=self.refractory,
            noise_std=self.noise_std,
        )


_NONLIN = {
    "sigmoid": torch.sigmoid,
    "relu": F.relu,
    "tanh": torch.tanh,
}


# ---------------- dendritic population ----------------

class DendriticPopulation(nn.Module):
    """N LIF neurons, each with K dendritic compartments.

    Per timestep, given per-dendrite input current d_in of shape (N, K):

      d_out  = nonlinear((d_in - threshold) * gain)        # (N, K)
      I_soma = (d_out * soma_weight).sum(dim=-1)           # (N,)
      [soma]   standard LIF step with I_soma as external drive

    The dendrite outputs are CURRENTS (not conductances) summed into the
    soma's external_current. Synaptic conductance + reversal-potential
    physics live in the DendriticSynapses bank, which aggregates per
    (neuron, dendrite) and converts to per-compartment current at the
    soma's V before handing the (N, K) tensor to this step().
    """

    def __init__(
        self,
        n_neurons: int,
        params: DendriticParams | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        params = params or DendriticParams()
        self.n = n_neurons
        self.K = params.n_dendrites
        self.device = torch.device(device)
        self.dtype = dtype
        self.params = params

        # Soma is a standard LIF population
        self.soma = LIFPopulation(
            n_neurons, params=params.to_neuron_params(), device=device, dtype=dtype,
        )

        # Per-dendrite nonlinearity params. Scalar for now; can be made
        # per-(neuron, dendrite) tensors later for heterogeneous behaviour.
        if params.nonlinearity not in _NONLIN:
            raise ValueError(f"unknown nonlinearity: {params.nonlinearity}")
        self.nonlinear_name = params.nonlinearity
        self.nonlinear: Callable = _NONLIN[params.nonlinearity]
        self.gain = float(params.dendrite_gain)
        self.threshold = float(params.dendrite_threshold)
        self.soma_weight = float(params.soma_weight)
        self.dendrite_leak = float(params.dendrite_leak)

        # Per-branch leaky accumulator state (N, K). With dendrite_leak=0
        # this is overwritten by the input each step (stateless, the
        # original Phase 7 model); with dendrite_leak>0 each branch
        # integrates its input over time.
        self.register_buffer(
            "d", torch.zeros(n_neurons, self.K, dtype=dtype, device=self.device)
        )

    @property
    def V(self) -> torch.Tensor:
        return self.soma.V

    @torch.no_grad()
    def reset_state(self) -> None:
        self.soma.reset_state()
        self.d.zero_()

    @torch.no_grad()
    def dendritic_output(self, d_input: torch.Tensor) -> torch.Tensor:
        """Apply the per-dendrite nonlinearity. d_input: (N, K) -> (N, K)."""
        return self.nonlinear((d_input - self.threshold) * self.gain)

    @torch.no_grad()
    def step(
        self,
        d_input: torch.Tensor,                # (N, K) per-dendrite current
        g_syn: torch.Tensor | None = None,    # (N,)   soma-level conductance
        dt: float = 1.0,
        t: float = 0.0,
        soma_extra_current: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One step. Returns soma spike mask (N,) bool.

        Pipeline:
          d      <- dendrite_leak * d + d_input      # leaky branch integration
          d_out  = nonlinear((d - threshold) * gain) # branch nonlinearity
          I_soma = (d_out * soma_weight).sum(dim=-1)
          [soma]   standard LIF step
        With dendrite_leak=0 the first line collapses to d == d_input, i.e.
        the original stateless behaviour.
        """
        assert d_input.shape == (self.n, self.K), \
            f"d_input must be (N={self.n}, K={self.K}), got {tuple(d_input.shape)}"
        # Leaky per-branch integration (in place on the registered buffer).
        self.d.mul_(self.dendrite_leak).add_(d_input)
        d_out = self.dendritic_output(self.d)
        I_soma = d_out.sum(dim=-1) * self.soma_weight
        if soma_extra_current is not None:
            I_soma = I_soma + soma_extra_current.to(device=self.device, dtype=self.dtype)
        if g_syn is None:
            g_syn = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        return self.soma.step(I_soma, g_syn, dt=dt, t=t)


# ---------------- dendritic synapses ----------------

class DendriticSynapses(nn.Module):
    """Sparse synapses where each post target is a (neuron, dendrite) pair.

    edge_pre[e]            : pre-synaptic neuron index
    edge_post_neuron[e]    : post-synaptic neuron index
    edge_post_dendrite[e]  : which compartment (0..K-1) of the post-neuron
    kind[e], g_max[e], tau_syn[e] : as in SparseSynapses

    Aggregation produces a (n_post, K) per-compartment current tensor by
    scattering edge currents into the (neuron * K + dendrite) flat index.
    Conductance-based reversal: I_e = g[e] * (E_syn[e] - V_post_soma).
    Each compartment's V is the soma's V (simplified model — no separate
    dendritic V state).
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        n_dendrites: int,
        edge_pre: torch.Tensor,
        edge_post_neuron: torch.Tensor,
        edge_post_dendrite: torch.Tensor,
        kind: torch.Tensor,
        g_max: torch.Tensor,
        tau_syn: torch.Tensor | float = 5.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        device = torch.device(device)
        assert edge_pre.shape == edge_post_neuron.shape == edge_post_dendrite.shape \
            == kind.shape == g_max.shape, "edge tensors must share shape"
        assert int(edge_post_dendrite.max().item() if edge_post_dendrite.numel() else 0) < n_dendrites
        E = edge_pre.numel()
        self.n_pre = n_pre
        self.n_post = n_post
        self.K = n_dendrites
        self.E = E
        self.device = device
        self.dtype = dtype

        self.register_buffer("edge_pre", edge_pre.to(device=device, dtype=torch.long))
        self.register_buffer("edge_post_neuron", edge_post_neuron.to(device=device, dtype=torch.long))
        self.register_buffer("edge_post_dendrite", edge_post_dendrite.to(device=device, dtype=torch.long))
        self.register_buffer("kind", kind.to(device=device, dtype=torch.long))
        self.register_buffer("g_max", g_max.to(device=device, dtype=dtype))
        if isinstance(tau_syn, torch.Tensor):
            tau_t = tau_syn.to(device=device, dtype=dtype)
        else:
            tau_t = torch.full((E,), float(tau_syn), device=device, dtype=dtype)
        self.register_buffer("tau_syn", tau_t)

        E_syn = torch.where(
            self.kind == SYN_EXC,
            torch.full((E,), _E_SYN[SYN_EXC], device=device, dtype=dtype),
            torch.full((E,), _E_SYN[SYN_INH], device=device, dtype=dtype),
        )
        self.register_buffer("E_syn", E_syn)
        self.register_buffer("g", torch.zeros(E, device=device, dtype=dtype))

        # Pre-computed flat post index = neuron * K + dendrite
        flat = self.edge_post_neuron * self.K + self.edge_post_dendrite
        self.register_buffer("edge_post_flat", flat.to(dtype=torch.long))

    @torch.no_grad()
    def reset_state(self) -> None:
        self.g.zero_()

    @torch.no_grad()
    def decay(self, dt: float) -> None:
        if self.E == 0:
            return
        self.g.mul_(torch.exp(-dt / self.tau_syn))

    @torch.no_grad()
    def deliver_spikes(self, pre_spiked: torch.Tensor) -> None:
        if self.E == 0:
            return
        fire_e = pre_spiked[self.edge_pre]
        self.g.add_(self.g_max * fire_e.to(self.dtype))

    @torch.no_grad()
    def post_currents(self, V_soma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate (I_per_dendrite, g_per_neuron) post-currents.

        Returns:
          I_dend: (n_post, K)  per-compartment input current
          g_soma: (n_post,)    per-neuron conductance (summed across dendrites;
                                 used by soma for membrane-time-constant
                                 modulation in LIF integration)
        """
        if self.E == 0:
            return (torch.zeros(self.n_post, self.K, device=self.device, dtype=self.dtype),
                    torch.zeros(self.n_post, device=self.device, dtype=self.dtype))
        V_at_post = V_soma.index_select(0, self.edge_post_neuron)
        I_per_edge = self.g * (self.E_syn - V_at_post)
        I_flat = torch.zeros(self.n_post * self.K, device=self.device, dtype=self.dtype)
        I_flat.index_add_(0, self.edge_post_flat, I_per_edge)
        I_dend = I_flat.view(self.n_post, self.K)
        g_soma = torch.zeros(self.n_post, device=self.device, dtype=self.dtype)
        g_soma.index_add_(0, self.edge_post_neuron, self.g)
        return I_dend, g_soma


# ---------------- cluster ----------------

class DendriticCluster(nn.Module):
    """A DendriticPopulation + one or more DendriticSynapses banks.

    Per step:
      1. external drive (per-dendrite, optional)
      2. each synapse bank computes (I_dend, g_soma) at current soma V
      3. soma steps with summed I, g
      4. spike mask delivered to all banks; banks decay
    """

    def __init__(self, population: DendriticPopulation):
        super().__init__()
        self.population = population
        self.synapses: list[DendriticSynapses] = nn.ModuleList()  # type: ignore[assignment]

    def add_synapses(self, syn: DendriticSynapses) -> None:
        assert syn.n_post == self.population.n, "synapse n_post must match population size"
        assert syn.K == self.population.K, "synapse n_dendrites must match population K"
        self.synapses.append(syn)

    @torch.no_grad()
    def reset_state(self) -> None:
        self.population.reset_state()
        for s in self.synapses:
            s.reset_state()

    @torch.no_grad()
    def step(
        self,
        dt: float,
        t: float,
        external_dendritic_current: torch.Tensor | None = None,   # (N, K)
        soma_extra_current: torch.Tensor | None = None,           # (N,)
    ) -> torch.Tensor:
        N, K = self.population.n, self.population.K
        device, dtype = self.population.device, self.population.dtype

        I_dend = torch.zeros(N, K, device=device, dtype=dtype)
        g_soma = torch.zeros(N, device=device, dtype=dtype)
        if external_dendritic_current is not None:
            I_dend = I_dend + external_dendritic_current.to(device=device, dtype=dtype)

        V_soma = self.population.V
        for s in self.synapses:
            I, g = s.post_currents(V_soma)
            I_dend = I_dend + I
            g_soma = g_soma + g

        spike = self.population.step(
            I_dend, g_syn=g_soma, dt=dt, t=t,
            soma_extra_current=soma_extra_current,
        )

        for s in self.synapses:
            s.deliver_spikes(spike)
            s.decay(dt)

        return spike


# ---------------- single-neuron XOR preset ----------------

def build_xor_neuron(
    device: torch.device | str = "cpu",
) -> DendriticPopulation:
    """A 1-neuron, 2-dendrite preset that computes XOR(x1, x2).

    Wiring:
      dendrite 0 fires when x1 high AND x2 low   (excitatory x1, inhibitory x2)
      dendrite 1 fires when x2 high AND x1 low   (excitatory x2, inhibitory x1)
      soma sums: fires when EITHER dendrite is high

    No synapses needed -- the test injects per-dendrite current directly,
    so this is the cleanest dendritic-XOR demonstration. The wiring is
    encoded in how the caller routes the two binary inputs into the
    (1, 2)-shaped d_input tensor.

    Returns:
      pop: DendriticPopulation with N=1, K=2, sigmoid nonlinearity
    """
    params = DendriticParams(
        n_dendrites=2,
        nonlinearity="sigmoid",
        dendrite_gain=12.0,         # steep enough that 0->1 transition is decisive
        dendrite_threshold=0.5,
        soma_weight=80.0,           # convert sigmoid 0..1 output into nA-scale drive
        V_rest=-70.0, V_thresh=-50.0, V_reset=-75.0,
        tau_m=20.0, R_m=10.0, refractory=2.0,
        noise_std=0.0,
    )
    return DendriticPopulation(n_neurons=1, params=params, device=device)


def xor_dendrite_input(x1: float, x2: float) -> torch.Tensor:
    """Convert two binary inputs into the (1, 2) per-dendrite drive that
    realises XOR when fed to a build_xor_neuron() population.

      dendrite 0: x1 - x2      (-> 1 only when x1=1, x2=0)
      dendrite 1: x2 - x1      (-> 1 only when x2=1, x1=0)

    Soma sees Σ σ(...) which is high exactly when x1 XOR x2.
    """
    return torch.tensor([[x1 - x2, x2 - x1]], dtype=torch.float32)


def build_point_neuron_baseline(device: torch.device | str = "cpu") -> LIFPopulation:
    """A 1-neuron LIF baseline for the XOR comparison: same membrane params
    as the dendritic preset, no compartment nonlinearity.

    A point neuron sees the LINEAR sum of its inputs. We feed it
    (x1 + x2) * soma_weight directly. It can fire for AND/OR but not XOR.
    """
    params = NeuronParams(
        V_rest=-70.0, V_thresh=-50.0, V_reset=-75.0,
        tau_m=20.0, R_m=10.0, refractory=2.0, noise_std=0.0,
    )
    return LIFPopulation(n_neurons=1, params=params, device=device)
