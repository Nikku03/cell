"""Tensor-vectorised LIF population + sparse synapses.

Foundation for everything else. Numerically equivalent to the per-object
LIF + chemical synapses in `bio_neural_net`, but every neuron and every
synapse is a slice of a tensor that the same code-path updates in one
shot. This is what lets us scale from 500 to 100k+ neurons.

Integration: exponential Euler (same as bio_neural_net post-fix), exact
for the linear membrane ODE between spikes, unconditionally stable for
arbitrary synaptic conductance.

  V(t+dt) = V_ss + (V(t) - V_ss) * exp(-dt / tau_eff)

where
  tau_eff = tau_m / (1 + R_m * g_syn)
  V_ss    = (V_rest + R_m * (I_syn + g_syn * V)) / (1 + R_m * g_syn)

Spikes: V crosses V_thresh -> V := V_reset, set refractory.
Synapses: each spike adds g_max to the per-edge conductance; conductance
decays exponentially with tau_syn between events.

Conventions
-----------
All voltages in mV, currents in nA-equivalent, times in ms. Synapse
reversal potentials: E_exc = 0 mV, E_inh = -80 mV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------- neuron model ----------------

@dataclass
class NeuronParams:
    tau_m: float = 20.0
    V_rest: float = -70.0
    V_thresh: float = -50.0
    V_reset: float = -75.0
    R_m: float = 10.0
    refractory: float = 2.0
    noise_std: float = 0.0


class LIFPopulation(nn.Module):
    """N leaky integrate-and-fire neurons as parallel tensors.

    Per-neuron params can be scalars (homogeneous) or (N,) tensors
    (heterogeneous). The integrator is exponential Euler so even with
    very large synaptic conductances the dynamics stay bounded.
    """

    def __init__(
        self,
        n_neurons: int,
        params: NeuronParams | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        params = params or NeuronParams()
        self.n = n_neurons
        self.device = torch.device(device)
        self.dtype = dtype

        # State (each (N,))
        self.register_buffer("V", torch.full((n_neurons,), params.V_rest, dtype=dtype, device=self.device))
        self.register_buffer("refr", torch.zeros(n_neurons, dtype=dtype, device=self.device))
        self.register_buffer("spiked", torch.zeros(n_neurons, dtype=torch.bool, device=self.device))
        self.register_buffer("last_spike_t", torch.full((n_neurons,), -1e9, dtype=dtype, device=self.device))

        # Per-neuron params (kept as tensors so per-cell-type heterogeneity is free)
        self._set_param("tau_m", params.tau_m)
        self._set_param("V_rest", params.V_rest)
        self._set_param("V_thresh", params.V_thresh)
        self._set_param("V_reset", params.V_reset)
        self._set_param("R_m", params.R_m)
        self._set_param("refractory", params.refractory)
        self._set_param("noise_std", params.noise_std)

    def _set_param(self, name: str, value: float | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            assert value.shape == (self.n,), f"{name} must be scalar or (N,)"
            t = value.to(device=self.device, dtype=self.dtype)
        else:
            t = torch.full((self.n,), float(value), dtype=self.dtype, device=self.device)
        self.register_buffer(f"_{name}", t)

    def set_heterogeneous(self, **kwargs: torch.Tensor) -> None:
        """Replace one or more per-neuron params with (N,) tensors.

        Example: pop.set_heterogeneous(noise_std=noise_tensor) to give
        each neuron a different noise scale.
        """
        for k, v in kwargs.items():
            assert isinstance(v, torch.Tensor) and v.shape == (self.n,)
            setattr(self, f"_{k}", v.to(device=self.device, dtype=self.dtype))

    @torch.no_grad()
    def reset_state(self) -> None:
        self.V.fill_(0.0)
        self.V += self._V_rest
        self.refr.zero_()
        self.spiked.zero_()
        self.last_spike_t.fill_(-1e9)

    @torch.no_grad()
    def step(
        self,
        I_syn: torch.Tensor,
        g_syn: torch.Tensor,
        dt: float,
        t: float,
    ) -> torch.Tensor:
        """Advance all neurons by dt; return spike mask (N,) bool.

        I_syn: (N,) synaptic current at current V (sum_e g_e * (E_e - V))
        g_syn: (N,) total synaptic conductance (sum_e g_e)
        """
        # Refractory countdown (decrement on every step, clamp at 0)
        was_refr = self.refr > 0
        self.refr.sub_(dt).clamp_(min=0.0)

        # Exponential Euler step
        # tau_eff = tau_m / (1 + R_m * g_syn)
        # V_ss   = (V_rest + R_m * (I_syn + g_syn * V)) / (1 + R_m * g_syn)
        one_plus = 1.0 + self._R_m * g_syn
        V_ss = (self._V_rest + self._R_m * (I_syn + g_syn * self.V)) / one_plus
        tau_eff = self._tau_m / one_plus
        decay = torch.exp(-dt / tau_eff)
        V_new = V_ss + (self.V - V_ss) * decay

        # Noise (Euler-Maruyama)
        # Only add noise to non-refractory neurons; refractory ones are clamped to V_reset anyway.
        if torch.any(self._noise_std > 0):
            noise = torch.randn_like(V_new) * self._noise_std * math.sqrt(dt)
            V_new = V_new + noise

        # Refractory neurons: clamp to V_reset (no membrane dynamics during refractory)
        V_new = torch.where(was_refr, self._V_reset, V_new)

        # Threshold crossing
        spike = (V_new >= self._V_thresh) & (~was_refr)
        V_new = torch.where(spike, self._V_reset, V_new)
        self.refr = torch.where(spike, self._refractory, self.refr)
        self.last_spike_t = torch.where(spike, torch.full_like(self.last_spike_t, t), self.last_spike_t)

        self.V = V_new
        self.spiked = spike
        return spike


# ---------------- synapses ----------------

# Synapse kind enum as ints (so it stays inside tensors).
SYN_EXC = 0
SYN_INH = 1
_E_SYN = {SYN_EXC: 0.0, SYN_INH: -80.0}


class SparseSynapses(nn.Module):
    """Sparse synapses as parallel edge tensors.

    Each synapse is identified by its row in the edge tensors:
      edge_pre[e]   : index of pre-synaptic neuron
      edge_post[e]  : index of post-synaptic neuron
      kind[e]       : SYN_EXC or SYN_INH
      g_max[e]      : conductance increment on each pre-spike
      tau_syn[e]    : decay time constant (ms)

    Per-edge running conductance g[e] decays exponentially between
    pre-spikes; each pre-spike adds g_max[e]. The post-side aggregation
    is a scatter-add (`index_add_`) into per-neuron arrays.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        edge_pre: torch.Tensor,
        edge_post: torch.Tensor,
        kind: torch.Tensor,
        g_max: torch.Tensor,
        tau_syn: torch.Tensor | float = 5.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        device = torch.device(device)
        assert edge_pre.shape == edge_post.shape == kind.shape == g_max.shape, \
            "all edge tensors must have the same shape"
        E = edge_pre.numel()
        self.n_pre = n_pre
        self.n_post = n_post
        self.E = E
        self.device = device
        self.dtype = dtype

        self.register_buffer("edge_pre", edge_pre.to(device=device, dtype=torch.long))
        self.register_buffer("edge_post", edge_post.to(device=device, dtype=torch.long))
        self.register_buffer("kind", kind.to(device=device, dtype=torch.long))
        self.register_buffer("g_max", g_max.to(device=device, dtype=dtype))
        if isinstance(tau_syn, torch.Tensor):
            tau_t = tau_syn.to(device=device, dtype=dtype)
        else:
            tau_t = torch.full((E,), float(tau_syn), device=device, dtype=dtype)
        self.register_buffer("tau_syn", tau_t)

        # Per-edge reversal potential (from kind)
        E_syn = torch.where(
            self.kind == SYN_EXC,
            torch.full((E,), _E_SYN[SYN_EXC], device=device, dtype=dtype),
            torch.full((E,), _E_SYN[SYN_INH], device=device, dtype=dtype),
        )
        self.register_buffer("E_syn", E_syn)

        # Running conductance per edge
        self.register_buffer("g", torch.zeros(E, device=device, dtype=dtype))

        # Effective-synaptic-event counter: a synaptic event is a spike
        # crossing a synapse. This is the hardware-agnostic efficiency
        # metric the SparseSpike-GEMM design measures energy per. Kept as
        # a tensor accumulator so deliver_spikes never needs a CPU sync.
        self.register_buffer("_n_events", torch.zeros((), dtype=torch.long, device=device))

    @torch.no_grad()
    def reset_state(self) -> None:
        self.g.zero_()
        self._n_events.zero_()

    @property
    def n_synaptic_events(self) -> int:
        """Total synaptic events (spike-crossing-synapse) since the last
        reset_state(). Each event is one signed accumulate on real
        hardware; this count is the substrate's compute/energy proxy."""
        return int(self._n_events.item())

    @torch.no_grad()
    def decay(self, dt: float) -> None:
        """Exponential decay of all edge conductances. Applied once per timestep."""
        if self.E == 0:
            return
        self.g.mul_(torch.exp(-dt / self.tau_syn))

    @torch.no_grad()
    def deliver_spikes(self, pre_spiked: torch.Tensor) -> None:
        """For every edge whose pre fires, add g_max to its conductance.

        pre_spiked: (n_pre,) bool mask.
        """
        if self.E == 0:
            return
        fire_e = pre_spiked[self.edge_pre]
        # Each fired edge is one effective synaptic event (tensor-accumulated,
        # no CPU sync, so this adds negligible overhead to the hot loop).
        self._n_events += fire_e.sum()
        # add g_max only where fire_e is True
        self.g.add_(self.g_max * fire_e.to(self.dtype))

    @torch.no_grad()
    def post_currents(self, V_post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate (I_syn, g_syn) per post-neuron.

        For each edge e:
          I_e = g[e] * (E_syn[e] - V_post[edge_post[e]])
          g_e = g[e]
        Sum to per-post-neuron via index_add_.

        Returns (I_syn_per_post (n_post,), g_syn_per_post (n_post,)).
        """
        if self.E == 0:
            zeros = torch.zeros(self.n_post, device=self.device, dtype=self.dtype)
            return zeros, zeros.clone()
        V_at_post = V_post.index_select(0, self.edge_post)
        I_per_edge = self.g * (self.E_syn - V_at_post)
        I_per_post = torch.zeros(self.n_post, device=self.device, dtype=self.dtype)
        g_per_post = torch.zeros(self.n_post, device=self.device, dtype=self.dtype)
        I_per_post.index_add_(0, self.edge_post, I_per_edge)
        g_per_post.index_add_(0, self.edge_post, self.g)
        return I_per_post, g_per_post


# ---------------- efficiency accounting ----------------

def synaptic_event_energy_pj(n_events: int, pj_per_event: float = 0.1) -> float:
    """Estimate the synaptic-arithmetic energy of a run, in picojoules.

    `n_events` is the hardware-agnostic count (see
    SparseSynapses.n_synaptic_events). `pj_per_event` is the ONLY
    hardware assumption: the SparseSpike-GEMM design projects a
    signed-accumulate synaptic event at roughly 0.05-0.5 pJ on a fused
    on-chip kernel (vs ~10^4-10^5x more for dense FP16). The default
    0.1 pJ sits in that projected band. The count itself is exact; the
    pJ figure is only as good as this per-event assumption.
    """
    return float(n_events) * float(pj_per_event)


# ---------------- cluster ----------------

class SNNCluster(nn.Module):
    """A LIF population + one or more sparse synapse banks.

    Runs the full per-step pipeline:
      1. external_current(t) gives optional drive
      2. each synapse bank computes (I_post, g_post) at current V
      3. neurons step with combined I, g
      4. spike mask is delivered to all synapse banks
      5. all synapse conductances decay

    The synapse decay is applied AFTER the step so the current step's
    post-current calculation sees the pre-decay g (matches bio_neural_net's
    behaviour).
    """

    def __init__(self, population: LIFPopulation):
        super().__init__()
        self.population = population
        self.synapses: list[SparseSynapses] = nn.ModuleList()  # type: ignore[assignment]

    def add_synapses(self, syn: SparseSynapses) -> None:
        assert syn.n_post == self.population.n, "synapse n_post must match population size"
        self.synapses.append(syn)

    @torch.no_grad()
    def reset_state(self) -> None:
        self.population.reset_state()
        for s in self.synapses:
            s.reset_state()

    @torch.no_grad()
    def step(self, dt: float, t: float, external_current: torch.Tensor | None = None) -> torch.Tensor:
        """Run one dt step. Returns spike mask (n,) bool.

        external_current: (n,) tensor, optional injected current.
        """
        N = self.population.n
        device = self.population.device
        dtype = self.population.dtype

        I_total = torch.zeros(N, device=device, dtype=dtype)
        g_total = torch.zeros(N, device=device, dtype=dtype)
        if external_current is not None:
            I_total = I_total + external_current.to(device=device, dtype=dtype)

        # Aggregate synaptic input at current V
        V = self.population.V
        for s in self.synapses:
            I, g = s.post_currents(V)
            I_total = I_total + I
            g_total = g_total + g

        # Neuron update
        spike = self.population.step(I_total, g_total, dt=dt, t=t)

        # Deliver spikes to all synapse banks (then decay)
        for s in self.synapses:
            s.deliver_spikes(spike)
            s.decay(dt)

        return spike

    @torch.no_grad()
    def run(
        self,
        duration_ms: float,
        dt: float = 1.0,
        external_current_fn=None,
        record_spikes: bool = True,
    ) -> dict:
        """Simulate for `duration_ms`. Returns dict with optional spike record.

        external_current_fn(t) -> (n,) tensor or None.
        """
        n_steps = int(round(duration_ms / dt))
        if record_spikes:
            spike_times: list[float] = []
            spike_neurons: list[int] = []
        for k in range(n_steps):
            t = k * dt
            ext = None
            if external_current_fn is not None:
                ext = external_current_fn(t)
            spike = self.step(dt, t, external_current=ext)
            if record_spikes and spike.any():
                idx = spike.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
                spike_neurons.extend(idx)
                spike_times.extend([t] * len(idx))
        out: dict = {"n_steps": n_steps, "dt": dt, "duration_ms": duration_ms}
        if record_spikes:
            out["spike_times"] = torch.tensor(spike_times, dtype=torch.float32)
            out["spike_neurons"] = torch.tensor(spike_neurons, dtype=torch.long)
            out["n_spikes"] = len(spike_neurons)
        return out


# ---------------- E/I microcircuit builder ----------------

def build_ei_microcircuit(
    n_e: int = 400,
    n_i: int = 100,
    p_ei: float = 0.15,
    p_ie: float = 0.15,
    p_ee: float = 0.04,
    p_ii: float = 0.05,
    g_ee: float = 0.08,
    g_ei: float = 0.30,
    g_ie: float = 0.80,
    g_ii: float = 0.40,
    tau_syn_e: float = 5.0,
    tau_syn_i: float = 5.0,
    noise_std_e: float = 5.0,
    noise_std_i: float = 1.0,
    seed: int = 0,
    device: torch.device | str = "cpu",
    degree_mode: bool = False,
) -> SNNCluster:
    """Build an excitation/inhibition microcircuit.

    Default (degree_mode=False): connection probabilities are constant
    densities (good for small N). Total synapse count scales as N^2.

    degree_mode=True: hold the EXPECTED IN-DEGREE PER POST-NEURON constant
    as N grows (biological: each cortical neuron has ~7000 synapses
    regardless of total brain size). The default p_xy values are
    re-interpreted as "the density at N=(400, 100)" and rescaled. This
    keeps memory O(N) and is the right model for cortical-style scaling.

    At N_e=400, N_i=100 with default p's:
      E-E mean in-degree = 400 * 0.04 = 16
      E-I in-degree on I = 400 * 0.15 = 60
      I-E in-degree on E = 100 * 0.15 = 15
      I-I in-degree on I = 100 * 0.05 = 5
    """
    if degree_mode:
        ref_n_e, ref_n_i = 400, 100
        # Rescale: p_new = p_old * ref_n / new_n keeps in-degree constant
        p_ee = p_ee * ref_n_e / max(n_e, 1)
        p_ei = p_ei * ref_n_e / max(n_e, 1)
        p_ie = p_ie * ref_n_i / max(n_i, 1)
        p_ii = p_ii * ref_n_i / max(n_i, 1)
    """Vectorised equivalent of bio_neural_net's clustered microcircuit.

    Neurons 0..n_e-1 are excitatory; n_e..n_e+n_i-1 are inhibitory.
    Recurrent E-E, E-I, I-E, I-I edges sampled at the given densities.
    Per-population noise is set after construction so E and I have
    different membrane fluctuations.
    """
    n = n_e + n_i
    pop = LIFPopulation(n, device=device)

    # Set heterogeneous noise: E noisier than I
    noise = torch.zeros(n)
    noise[:n_e] = noise_std_e
    noise[n_e:] = noise_std_i
    pop.set_heterogeneous(noise_std=noise)

    g = torch.Generator(device="cpu").manual_seed(seed)

    def sample_edges(src_lo, src_hi, dst_lo, dst_hi, p, g_max, kind):
        """Sparse Bernoulli edge sampling that does NOT build an O(src*dst)
        mask. For large N this is the difference between a few MB and
        hundreds of GB. Strategy: for each source, draw the number of
        target connections from Binomial(dst_n, p), then sample without
        replacement which targets.
        """
        src_n = src_hi - src_lo
        dst_n = dst_hi - dst_lo
        if src_n == 0 or dst_n == 0 or p <= 0:
            return None
        # Per-source connection count ~ Binomial(dst_n, p).
        # Tensor binomial sample (torch.distributions.Binomial is slow elementwise; build manually).
        counts = torch.distributions.Binomial(
            total_count=dst_n, probs=torch.tensor(p, dtype=torch.float64)
        ).sample(sample_shape=(src_n,)).long()
        # Optional self-loop avoidance: with sparse sampling, the rare
        # accidental self-edge from a per-source draw can be filtered out
        # in a post-pass; for large N this is negligible compared to
        # building a dense mask.
        total_edges = int(counts.sum().item())
        if total_edges == 0:
            return None
        # Allocate output tensors and fill per source via random permutation.
        pre = torch.empty(total_edges, dtype=torch.long)
        post = torch.empty(total_edges, dtype=torch.long)
        off = 0
        same_block = (src_lo == dst_lo and src_hi == dst_hi)
        for i in range(src_n):
            c = int(counts[i].item())
            if c == 0:
                continue
            # Sample c distinct targets out of dst_n. randperm is O(dst_n);
            # for large dst_n with small c, randomly drawing c integers
            # and rejecting duplicates is faster, but randperm is simple
            # and correct.
            chosen = torch.randperm(dst_n, generator=g)[:c]
            if same_block:
                # Drop the diagonal (avoid self-loop)
                self_idx_local = i
                mask = chosen != self_idx_local
                chosen = chosen[mask]
            n_kept = chosen.numel()
            pre[off:off + n_kept] = src_lo + i
            post[off:off + n_kept] = dst_lo + chosen
            off += n_kept
        pre = pre[:off]
        post = post[:off]
        E = pre.numel()
        if E == 0:
            return None
        return {
            "pre": pre,
            "post": post,
            "kind": torch.full((E,), kind, dtype=torch.long),
            "g_max": torch.full((E,), float(g_max)),
            "tau_syn": float(tau_syn_e if kind == SYN_EXC else tau_syn_i),
        }

    chunks = []
    for src_lo, src_hi, dst_lo, dst_hi, p, g_max, kind in [
        (0, n_e, 0, n_e, p_ee, g_ee, SYN_EXC),
        (0, n_e, n_e, n, p_ei, g_ei, SYN_EXC),
        (n_e, n, 0, n_e, p_ie, g_ie, SYN_INH),
        (n_e, n, n_e, n, p_ii, g_ii, SYN_INH),
    ]:
        c = sample_edges(src_lo, src_hi, dst_lo, dst_hi, p, g_max, kind)
        if c is not None:
            chunks.append(c)

    if chunks:
        pre = torch.cat([c["pre"] for c in chunks])
        post = torch.cat([c["post"] for c in chunks])
        kind = torch.cat([c["kind"] for c in chunks])
        g_max = torch.cat([c["g_max"] for c in chunks])
        tau_syn = torch.cat([
            torch.full((c["pre"].numel(),), c["tau_syn"]) for c in chunks
        ])
    else:
        pre = torch.empty(0, dtype=torch.long)
        post = torch.empty(0, dtype=torch.long)
        kind = torch.empty(0, dtype=torch.long)
        g_max = torch.empty(0)
        tau_syn = torch.empty(0)

    syn = SparseSynapses(
        n_pre=n, n_post=n,
        edge_pre=pre, edge_post=post,
        kind=kind, g_max=g_max, tau_syn=tau_syn,
        device=device,
    )

    cluster = SNNCluster(pop)
    cluster.add_synapses(syn)
    return cluster
