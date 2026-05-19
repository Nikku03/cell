"""Trick 2: recurrent depth — let the network think longer.

A feedforward network with N neurons and 1 pass does ~N units of work.
A recurrent network with N neurons and T passes does ~N×T units of
work using the SAME synaptic weights. Pattern-completion in
attractor networks is the canonical example: a small clustered
network resolves a corrupted pattern over many timesteps by re-using
the same recurrent loops.

This module provides:

  ClusteredAssemblyNetwork  -- builder for a clustered E/I network
                               where each assembly forms a local attractor
  measure_completion       -- track per-assembly firing rate over time
                               given a partial cue
  RecurrentDepthBenchmark  -- run completion at multiple T, return
                               accuracy curves

The Phase 2 contract: completion improves with T up to a plateau, the
plateau is selective (only the cued assembly's other half lights up),
and the total work done is ~ K_active × T (so T scales effective
compute linearly).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .population import (
    LIFPopulation,
    NeuronParams,
    SparseSynapses,
    SNNCluster,
    SYN_EXC,
    SYN_INH,
)


# ---------------- clustered assembly network ----------------

@dataclass
class ClusteredAssemblyNetwork:
    cluster: SNNCluster
    n_assemblies: int
    assembly_size: int
    n_e_assembly_total: int
    n_e_free: int
    n_i: int
    assemblies: list[torch.Tensor]   # each: (assembly_size,) E-neuron indices

    @property
    def n_total(self) -> int:
        return self.n_e_assembly_total + self.n_e_free + self.n_i


def build_clustered_assembly_network(
    n_assemblies: int = 4,
    assembly_size: int = 80,
    n_e_free: int = 80,
    n_i: int = 100,
    p_within: float = 0.25,
    p_across: float = 0.005,
    p_free: float = 0.04,
    p_ei: float = 0.12,
    p_ie: float = 0.12,
    p_ii: float = 0.05,
    g_within: float = 0.20,
    g_across: float = 0.04,
    g_free: float = 0.04,
    g_ei: float = 0.30,
    g_ie: float = 0.70,
    g_ii: float = 0.40,
    tau_syn: float = 5.0,
    noise_std_e: float = 1.8,
    noise_std_i: float = 0.3,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> ClusteredAssemblyNetwork:
    """Build a 4-assembly clustered network with the same topology as
    bio_neural_net's cognitive_microcircuit, but tensor-vectorised so we
    can scale.

    Layout:
      E neurons 0 .. n_e_assembly_total-1 are split into n_assemblies
        of `assembly_size` each, dense intra-assembly, sparse across.
      E neurons n_e_assembly_total .. n_e_assembly_total + n_e_free -1
        are "free" (sparse all-to-all)
      I neurons follow
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_e_assembly_total = n_assemblies * assembly_size
    n_e = n_e_assembly_total + n_e_free
    n = n_e + n_i

    pop = LIFPopulation(n, device=device)
    # Heterogeneous noise: E noisier than I
    noise = torch.zeros(n)
    noise[:n_e] = noise_std_e
    noise[n_e:] = noise_std_i
    pop.set_heterogeneous(noise_std=noise)

    # Assembly index lists
    assemblies = []
    for k in range(n_assemblies):
        idx = torch.arange(k * assembly_size, (k + 1) * assembly_size)
        assemblies.append(idx)

    pre_list, post_list, kind_list, gmax_list = [], [], [], []

    def add_pair_block(src_lo, src_hi, dst_lo, dst_hi, p, g_max, kind, exclude_self=False):
        src_n = src_hi - src_lo
        dst_n = dst_hi - dst_lo
        if src_n == 0 or dst_n == 0 or p <= 0:
            return
        # Per-source binomial sampling (memory-light)
        counts = torch.distributions.Binomial(
            total_count=dst_n, probs=torch.tensor(p, dtype=torch.float64)
        ).sample(sample_shape=(src_n,)).long()
        total = int(counts.sum().item())
        if total == 0:
            return
        pre_buf = torch.empty(total, dtype=torch.long)
        post_buf = torch.empty(total, dtype=torch.long)
        off = 0
        for i in range(src_n):
            c = int(counts[i].item())
            if c == 0:
                continue
            chosen = torch.randperm(dst_n, generator=g)[:c]
            if exclude_self:
                chosen = chosen[chosen != i]
            kept = chosen.numel()
            pre_buf[off:off + kept] = src_lo + i
            post_buf[off:off + kept] = dst_lo + chosen
            off += kept
        pre_buf = pre_buf[:off]
        post_buf = post_buf[:off]
        pre_list.append(pre_buf)
        post_list.append(post_buf)
        kind_list.append(torch.full((pre_buf.numel(),), kind, dtype=torch.long))
        gmax_list.append(torch.full((pre_buf.numel(),), float(g_max)))

    # Within-assembly E-E (dense, strong)
    for k in range(n_assemblies):
        a_lo = k * assembly_size
        a_hi = (k + 1) * assembly_size
        add_pair_block(a_lo, a_hi, a_lo, a_hi, p_within, g_within, SYN_EXC, exclude_self=True)

    # Across-assembly E-E (sparse, weak)
    for k in range(n_assemblies):
        for m in range(n_assemblies):
            if k == m:
                continue
            add_pair_block(
                k * assembly_size, (k + 1) * assembly_size,
                m * assembly_size, (m + 1) * assembly_size,
                p_across, g_across, SYN_EXC,
            )

    # Free E to all E (sparse)
    if n_e_free > 0:
        add_pair_block(
            n_e_assembly_total, n_e_assembly_total + n_e_free,
            0, n_e, p_free, g_free, SYN_EXC,
            exclude_self=False,
        )

    # E -> I (all E to all I, sparse uniform)
    add_pair_block(0, n_e, n_e, n, p_ei, g_ei, SYN_EXC)
    # I -> E
    add_pair_block(n_e, n, 0, n_e, p_ie, g_ie, SYN_INH)
    # I -> I
    add_pair_block(n_e, n, n_e, n, p_ii, g_ii, SYN_INH, exclude_self=True)

    pre = torch.cat(pre_list) if pre_list else torch.empty(0, dtype=torch.long)
    post = torch.cat(post_list) if post_list else torch.empty(0, dtype=torch.long)
    kind = torch.cat(kind_list) if kind_list else torch.empty(0, dtype=torch.long)
    gmax = torch.cat(gmax_list) if gmax_list else torch.empty(0)
    tau = torch.full((pre.numel(),), tau_syn)
    syn = SparseSynapses(
        n_pre=n, n_post=n,
        edge_pre=pre, edge_post=post,
        kind=kind, g_max=gmax, tau_syn=tau,
        device=device,
    )
    cluster = SNNCluster(pop)
    cluster.add_synapses(syn)
    return ClusteredAssemblyNetwork(
        cluster=cluster,
        n_assemblies=n_assemblies,
        assembly_size=assembly_size,
        n_e_assembly_total=n_e_assembly_total,
        n_e_free=n_e_free,
        n_i=n_i,
        assemblies=assemblies,
    )


# ---------------- pattern-completion measurement ----------------

@dataclass
class CompletionResult:
    cued_rate: list[float]       # firing rate of cued half over time bins
    uncued_rate: list[float]     # firing rate of uncued half over time bins
    other_assemblies_rate: list[list[float]]   # rates of OTHER assemblies (per-bin)
    bin_ms: float                # ms per bin


def measure_completion(
    network: ClusteredAssemblyNetwork,
    target_assembly: int,
    cue_fraction: float = 0.5,
    cue_current: float = 6.0,
    duration_ms: float = 400.0,
    bin_ms: float = 25.0,
    dt: float = 1.0,
    seed: int = 0,
) -> CompletionResult:
    """Cue half of target_assembly, run, return time-binned rates.

    Returns time-resolved firing of: cued half, uncued half, and each
    of the other (non-target) assemblies' E populations.
    """
    cluster = network.cluster
    cluster.reset_state()
    n_total = network.n_total
    asm_size = network.assembly_size

    target_idx = network.assemblies[target_assembly]
    n_cue = max(1, int(round(cue_fraction * asm_size)))
    cued = target_idx[:n_cue]
    uncued = target_idx[n_cue:]

    other_assemblies = [
        network.assemblies[k] for k in range(network.n_assemblies) if k != target_assembly
    ]

    base_drive = torch.zeros(n_total)
    base_drive[cued] = cue_current

    n_steps = int(round(duration_ms / dt))
    n_bins = int(round(duration_ms / bin_ms))
    steps_per_bin = max(1, n_steps // n_bins)
    cued_rate = [0.0] * n_bins
    uncued_rate = [0.0] * n_bins
    other_rates = [[0.0] * n_bins for _ in other_assemblies]

    cur_bin = 0
    spikes_in_bin_cued = 0
    spikes_in_bin_uncued = 0
    spikes_in_bin_others = [0 for _ in other_assemblies]

    def commit_bin(bin_i, n_steps_this_bin):
        cued_rate[bin_i] = (
            1000.0 * spikes_in_bin_cued / (len(cued) * n_steps_this_bin * dt)
        )
        uncued_rate[bin_i] = (
            1000.0 * spikes_in_bin_uncued / (len(uncued) * n_steps_this_bin * dt)
            if len(uncued) > 0 else 0.0
        )
        for j, _ in enumerate(other_assemblies):
            other_rates[j][bin_i] = (
                1000.0 * spikes_in_bin_others[j]
                / (asm_size * n_steps_this_bin * dt)
            )

    n_steps_this_bin = 0
    for step in range(n_steps):
        spike = cluster.step(dt=dt, t=step * dt, external_current=base_drive)
        spikes_in_bin_cued += int(spike[cued].sum().item())
        spikes_in_bin_uncued += int(spike[uncued].sum().item())
        for j, asm_other in enumerate(other_assemblies):
            spikes_in_bin_others[j] += int(spike[asm_other].sum().item())
        n_steps_this_bin += 1
        if (step + 1) % steps_per_bin == 0 and cur_bin < n_bins:
            commit_bin(cur_bin, n_steps_this_bin)
            cur_bin += 1
            spikes_in_bin_cued = 0
            spikes_in_bin_uncued = 0
            spikes_in_bin_others = [0 for _ in other_assemblies]
            n_steps_this_bin = 0

    return CompletionResult(
        cued_rate=cued_rate,
        uncued_rate=uncued_rate,
        other_assemblies_rate=other_rates,
        bin_ms=bin_ms,
    )


# ---------------- recurrent-depth scan ----------------

@dataclass
class DepthScanResult:
    durations_ms: list[float]
    final_uncued_rate: list[float]     # final-bin firing rate of uncued half
    final_other_rate: list[float]      # mean final-bin rate of other assemblies
    selectivity: list[float]           # uncued / (other + epsilon)


def scan_recurrent_depth(
    network: ClusteredAssemblyNetwork,
    target_assembly: int = 0,
    durations_ms: list[float] = (50.0, 100.0, 200.0, 400.0, 800.0),
    cue_fraction: float = 0.5,
    cue_current: float = 6.0,
    bin_ms: float = 25.0,
) -> DepthScanResult:
    """Run completion at multiple durations T. Report uncued half rate,
    other-assembly rate, and selectivity at each T.

    Selectivity is the headline number: uncued-half firing / mean
    other-assembly firing. Higher = pattern completion is working.
    """
    final_uncued = []
    final_other = []
    selectivity = []
    for T in durations_ms:
        res = measure_completion(
            network, target_assembly=target_assembly,
            cue_fraction=cue_fraction, cue_current=cue_current,
            duration_ms=T, bin_ms=bin_ms,
        )
        # Take the last bin's value
        uncued = res.uncued_rate[-1]
        if res.other_assemblies_rate:
            others = sum(r[-1] for r in res.other_assemblies_rate) / len(res.other_assemblies_rate)
        else:
            others = 0.0
        final_uncued.append(uncued)
        final_other.append(others)
        selectivity.append(uncued / max(others, 0.1))
    return DepthScanResult(
        durations_ms=list(durations_ms),
        final_uncued_rate=final_uncued,
        final_other_rate=final_other,
        selectivity=selectivity,
    )
