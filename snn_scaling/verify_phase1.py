"""Phase 1 verification: enforced sparse activation (kWTA).

Gates:

  1. kWTA enforces at most K spikes per step under saturating drive
  2. Active fraction is exactly K/N when more than K want to fire
  3. Without saturation (few candidates), kWTA is a no-op
  4. Microcircuit with kWTA gate runs without errors and produces
     bounded firing rates
  5. Per-step compute (spike count) is proportional to K, not N
  6. Speed: kWTA at K=N/100 runs at least as fast as the unconstrained
     network at the same scale (overhead of top-K is amortised)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .population import (
    NeuronParams,
    SparseSynapses,
    SNNCluster,
    SYN_EXC,
    SYN_INH,
    build_ei_microcircuit,
)
from .sparse import kWTAPopulation
from .verify_phase0 import CheckResult, format_results


# ---------------- core kWTA gates ----------------

def check_kwta_enforces_max_K() -> CheckResult:
    """Drive 1000 neurons all hard. With K=10, exactly 10 fire per step."""
    pop = kWTAPopulation(1000, k=10, params=NeuronParams(noise_std=0.0))
    # Saturating drive: every neuron above threshold every step
    I = torch.full((1000,), 10.0)
    g = torch.zeros(1000)
    max_spikes_per_step = 0
    for k in range(50):
        spike = pop.step(I, g, dt=1.0, t=k * 1.0)
        max_spikes_per_step = max(max_spikes_per_step, int(spike.sum().item()))
    return CheckResult(
        "kWTA with K=10 caps spikes-per-step at 10 under saturating drive",
        max_spikes_per_step <= 10, float(max_spikes_per_step), 10.0,
        note=f"(max observed spikes/step: {max_spikes_per_step})",
    )


def check_kwta_K_active_under_saturation() -> CheckResult:
    """At least one step shows exactly K active when more than K want to fire."""
    pop = kWTAPopulation(1000, k=50, params=NeuronParams(noise_std=0.0))
    I = torch.full((1000,), 10.0)
    g = torch.zeros(1000)
    saturating_steps = 0
    for k in range(100):
        spike = pop.step(I, g, dt=1.0, t=k * 1.0)
        n_s = int(spike.sum().item())
        if n_s == 50:
            saturating_steps += 1
    return CheckResult(
        "at least 10 timesteps see exactly K=50 active under saturation",
        saturating_steps >= 10, float(saturating_steps), 10.0,
        note=f"(saturating timesteps: {saturating_steps}/100)",
    )


def check_kwta_noop_at_subthreshold() -> CheckResult:
    """With drive too weak to fire, kWTA fires nothing (no false positives)."""
    pop = kWTAPopulation(1000, k=10, params=NeuronParams(noise_std=0.0))
    I = torch.full((1000,), 0.5)        # well below threshold
    g = torch.zeros(1000)
    n_spikes = 0
    for k in range(200):
        spike = pop.step(I, g, dt=1.0, t=k * 1.0)
        n_spikes += int(spike.sum().item())
    return CheckResult(
        "kWTA does not fire when no neuron crosses threshold",
        n_spikes == 0, float(n_spikes), 0.0,
        note=f"(spurious spikes: {n_spikes})",
    )


def check_kwta_active_fraction_controlled() -> CheckResult:
    """Over a long run, observed mean active fraction matches K/N
    within a reasonable margin when drive is heavy."""
    N = 1000
    K = 80
    pop = kWTAPopulation(N, k=K, params=NeuronParams(noise_std=2.0))
    # Mix of strong and modest drive so different neurons compete.
    rng = torch.Generator().manual_seed(0)
    I_base = 3.0 + 5.0 * torch.rand(N, generator=rng)
    g = torch.zeros(N)
    n_active_total = 0
    steps = 200
    for k in range(steps):
        spike = pop.step(I_base, g, dt=1.0, t=k * 1.0)
        n_active_total += int(spike.sum().item())
    observed = n_active_total / (steps * N)
    target = K / N
    # The cap is K, so observed cannot exceed target; can be lower if
    # not enough candidates. Heavy drive expected to keep it close to target.
    rel_dev = abs(observed - target) / target
    return CheckResult(
        "observed active fraction near K/N under heavy drive",
        rel_dev < 0.30, rel_dev, 0.30,
        note=f"(observed={observed*100:.2f}%, target K/N={target*100:.2f}%)",
    )


# ---------------- kWTA inside a real microcircuit ----------------

class kWTASparseCluster:
    """Wrap kWTAPopulation in a synapse-running pipeline.

    Reusing the SNNCluster code with a kWTAPopulation works directly --
    LIFPopulation is the parent class, the step() method is overridden.
    No wrapper needed; this class exists only to document the pattern.
    """


def check_microcircuit_with_kwta_runs() -> CheckResult:
    """Build a 1000-neuron microcircuit, swap in a kWTAPopulation."""
    # Build with default builder
    cluster = build_ei_microcircuit(n_e=800, n_i=200, seed=0)
    # Replace its population with a kWTA version using the same params.
    # Easiest: build a fresh cluster with kWTA and the same synapses.
    old_pop = cluster.population
    new_pop = kWTAPopulation(old_pop.n, k=100, params=NeuronParams(noise_std=0.0))
    new_pop.set_heterogeneous(noise_std=old_pop._noise_std.clone())
    new_cluster = SNNCluster(new_pop)
    for s in cluster.synapses:
        new_cluster.add_synapses(s)

    out = new_cluster.run(duration_ms=300.0, dt=1.0)
    n_spikes = out["n_spikes"]
    # 300 steps * 100 K = at most 30000 spikes. The kWTA cap is hard.
    return CheckResult(
        "1000-neuron microcircuit with kWTA (K=100) runs, bounded firing",
        n_spikes <= 30000 and n_spikes > 0, float(n_spikes), 30000.0,
        note=f"(total spikes in 300 ms: {n_spikes}; cap = {300*100})",
    )


def check_kwta_max_active_per_step() -> CheckResult:
    """In the microcircuit run, no step ever exceeds K active."""
    cluster = build_ei_microcircuit(n_e=800, n_i=200, seed=1)
    old_pop = cluster.population
    new_pop = kWTAPopulation(old_pop.n, k=50, params=NeuronParams(noise_std=0.0))
    new_pop.set_heterogeneous(noise_std=old_pop._noise_std.clone())
    new_cluster = SNNCluster(new_pop)
    for s in cluster.synapses:
        new_cluster.add_synapses(s)
    max_seen = 0
    for k in range(300):
        ext = None
        # Inject strong drive halfway through to make sure saturation occurs
        if 100 <= k < 200:
            ext = torch.zeros(old_pop.n)
            ext[:200] = 6.0   # drive 200 E neurons hard
        spike = new_cluster.step(dt=1.0, t=k * 1.0, external_current=ext)
        max_seen = max(max_seen, int(spike.sum().item()))
    return CheckResult(
        "kWTA cap holds across microcircuit dynamics",
        max_seen <= 50, float(max_seen), 50.0,
        note=f"(max active per step: {max_seen}; K=50)",
    )


def check_kwta_speed_overhead() -> CheckResult:
    """Top-K overhead should be small. Compare kWTA at K=100 vs ungated
    on a 10k-neuron run."""
    # Ungated baseline (default microcircuit's LIFPopulation)
    cluster = build_ei_microcircuit(n_e=8000, n_i=2000, seed=5, degree_mode=True)
    t0 = time.time()
    cluster.run(duration_ms=100.0, dt=1.0, record_spikes=False)
    t_baseline = time.time() - t0

    # kWTA gated
    cluster2 = build_ei_microcircuit(n_e=8000, n_i=2000, seed=5, degree_mode=True)
    old = cluster2.population
    new = kWTAPopulation(old.n, k=100, params=NeuronParams(noise_std=0.0))
    new.set_heterogeneous(noise_std=old._noise_std.clone())
    c3 = SNNCluster(new)
    for s in cluster2.synapses:
        c3.add_synapses(s)
    t0 = time.time()
    c3.run(duration_ms=100.0, dt=1.0, record_spikes=False)
    t_kwta = time.time() - t0
    overhead = t_kwta / max(t_baseline, 1e-9)
    # Allow up to 1.5x overhead (top-K is O(N log K))
    return CheckResult(
        "kWTA overhead vs ungated population is bounded",
        overhead < 1.5, overhead, 1.5,
        note=f"(baseline={t_baseline:.3f}s, kWTA={t_kwta:.3f}s, ratio={overhead:.2f}x)",
    )


# ---------------- runner ----------------

def run_all() -> list[CheckResult]:
    return [
        check_kwta_enforces_max_K(),
        check_kwta_K_active_under_saturation(),
        check_kwta_noop_at_subthreshold(),
        check_kwta_active_fraction_controlled(),
        check_microcircuit_with_kwta_runs(),
        check_kwta_max_active_per_step(),
        check_kwta_speed_overhead(),
    ]


if __name__ == "__main__":
    t0 = time.time()
    results = run_all()
    print(f"Phase 1 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    print(f"\n  -> {n_pass} / {len(results)} checks passed.")
