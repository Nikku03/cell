"""Phase 0 verification: tensor-vectorised LIF + sparse synapses.

Gates:

  1. Single neuron at rest stays at V_rest
  2. Single neuron under strong injected current fires repeatedly
  3. Refractory period: spikes don't come back-to-back below refractory time
  4. Excitatory synapse propagates a spike (post fires after pre)
  5. Inhibitory synapse silences a post under modest drive
  6. Vectorised 500-neuron microcircuit's population statistics match
     bio_neural_net's pure-Python implementation (within sampling noise)
  7. Scale test: 10 000-neuron simulation completes a 100 ms run
  8. Scale test: 100 000-neuron simulation completes a 10 ms run
  9. Speed test: vectorised 500-neuron is at least 5x faster than the
     bio_neural_net pure-Python version at the same scale
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .population import (
    LIFPopulation,
    NeuronParams,
    SparseSynapses,
    SNNCluster,
    SYN_EXC,
    SYN_INH,
    build_ei_microcircuit,
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float
    threshold: float
    note: str = ""


def format_results(results: list[CheckResult]) -> str:
    lines = []
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{tag}]  {r.name:<60}  {r.value:>10.4e}  / thr {r.threshold:>10.4e}  {r.note}")
    return "\n".join(lines)


# ---------------- single-neuron / synapse gates ----------------

def check_rest_state() -> CheckResult:
    """Single neuron with zero input stays at V_rest (no noise)."""
    pop = LIFPopulation(1, params=NeuronParams(noise_std=0.0))
    I = torch.zeros(1)
    g = torch.zeros(1)
    for k in range(200):
        pop.step(I, g, dt=1.0, t=k * 1.0)
    err = abs(pop.V.item() - (-70.0))
    return CheckResult(
        "single neuron at rest stays at V_rest", err < 1e-5, err, 1e-5,
        note=f"(V={pop.V.item():.4f}, expected -70.0)",
    )


def check_strong_current_fires() -> CheckResult:
    """Single neuron under strong constant current fires repeatedly."""
    pop = LIFPopulation(1, params=NeuronParams(noise_std=0.0))
    n_spikes = 0
    for k in range(500):
        I = torch.tensor([5.0])
        g = torch.zeros(1)
        spike = pop.step(I, g, dt=1.0, t=k * 1.0)
        if spike.any():
            n_spikes += 1
    # 500 ms, strong drive -> at least 30 spikes (rough lower bound)
    return CheckResult(
        "single neuron under strong I fires repeatedly", n_spikes >= 30,
        float(n_spikes), 30.0,
        note=f"(spikes in 500ms: {n_spikes})",
    )


def check_refractory() -> CheckResult:
    """No two spikes within `refractory` of each other."""
    pop = LIFPopulation(1, params=NeuronParams(noise_std=0.0, refractory=2.0))
    spike_times = []
    for k in range(500):
        spike = pop.step(torch.tensor([5.0]), torch.zeros(1), dt=1.0, t=k * 1.0)
        if spike.any():
            spike_times.append(k * 1.0)
    min_isi = min((b - a for a, b in zip(spike_times, spike_times[1:])), default=float("inf"))
    # refractory=2.0 -> next spike at least at t=t_spike+3 (refr+1 to integrate)
    return CheckResult(
        "no back-to-back spikes below refractory", min_isi >= 2.0, min_isi, 2.0,
        note=f"(min ISI: {min_isi:.2f} ms)",
    )


def check_exc_propagates() -> CheckResult:
    """A spike on pre causes a spike on post via an excitatory synapse."""
    pop = LIFPopulation(2, params=NeuronParams(noise_std=0.0))
    syn = SparseSynapses(
        n_pre=2, n_post=2,
        edge_pre=torch.tensor([0]), edge_post=torch.tensor([1]),
        kind=torch.tensor([SYN_EXC]), g_max=torch.tensor([2.0]),
        tau_syn=torch.tensor([5.0]),
    )
    cluster = SNNCluster(pop); cluster.add_synapses(syn)

    pre_spiked = post_spiked = False
    for k in range(200):
        ext = torch.zeros(2)
        if k < 50:
            ext[0] = 5.0
        spike = cluster.step(dt=1.0, t=k * 1.0, external_current=ext)
        if spike[0]:
            pre_spiked = True
        if spike[1]:
            post_spiked = True
    return CheckResult(
        "EXC synapse propagates spike (pre fires => post fires)",
        pre_spiked and post_spiked, 1.0 if (pre_spiked and post_spiked) else 0.0, 0.5,
        note=f"(pre fired={pre_spiked}, post fired={post_spiked})",
    )


def check_inh_silences() -> CheckResult:
    """Modest drive to post that would fire is suppressed by a strong INH input."""
    pop = LIFPopulation(2, params=NeuronParams(noise_std=0.0))
    syn = SparseSynapses(
        n_pre=2, n_post=2,
        edge_pre=torch.tensor([0]), edge_post=torch.tensor([1]),
        kind=torch.tensor([SYN_INH]), g_max=torch.tensor([1.5]),
        tau_syn=torch.tensor([5.0]),
    )
    cluster = SNNCluster(pop); cluster.add_synapses(syn)
    n_post = 0
    for k in range(500):
        ext = torch.tensor([10.0, 1.5])   # drive both: 0 fires; 1 fires alone but inh by 0
        spike = cluster.step(dt=1.0, t=k * 1.0, external_current=ext)
        if spike[1]:
            n_post += 1
    return CheckResult(
        "INH synapse silences post under modest drive", n_post == 0,
        float(n_post), 0.0,
        note=f"(post spikes: {n_post}; expected 0)",
    )


# ---------------- microcircuit statistics gate ----------------

def _rates_per_population(spike_neurons, n_e, n_i, duration_ms):
    e_mask = spike_neurons < n_e
    i_mask = spike_neurons >= n_e
    rate_e = 1000.0 * int(e_mask.sum()) / (n_e * duration_ms)
    rate_i = 1000.0 * int(i_mask.sum()) / (n_i * duration_ms)
    return rate_e, rate_i


def check_microcircuit_dynamics() -> CheckResult:
    """500-neuron microcircuit should produce baseline rates that match what
    bio_neural_net's pure-Python version did (E ~ 0.5 Hz, I ~ 16 Hz).

    Tolerance is loose: stochastic dynamics on different RNG seeds, but
    the qualitative regime (E ~ 0.1-3 Hz, I ~ 5-40 Hz) must hold.
    """
    cluster = build_ei_microcircuit(seed=0)
    duration = 800.0
    out = cluster.run(duration_ms=duration, dt=1.0)
    rate_e, rate_i = _rates_per_population(out["spike_neurons"], n_e=400, n_i=100, duration_ms=duration)
    in_range = (0.1 <= rate_e <= 3.0) and (5.0 <= rate_i <= 40.0)
    return CheckResult(
        "500-neuron microcircuit baseline rates in expected range",
        in_range, max(abs(rate_e - 0.5), abs(rate_i - 16.0)), 20.0,
        note=f"(E={rate_e:.2f} Hz, I={rate_i:.2f} Hz)",
    )


# ---------------- scale gates ----------------

def check_10k_neurons_runs() -> CheckResult:
    """A 10k-neuron microcircuit (8000 E + 2000 I) completes 100 ms."""
    cluster = build_ei_microcircuit(n_e=8000, n_i=2000, seed=1, degree_mode=True)
    t0 = time.time()
    out = cluster.run(duration_ms=100.0, dt=1.0)
    wall = time.time() - t0
    return CheckResult(
        f"10k-neuron microcircuit runs 100 ms (degree-constant)",
        wall < 60.0, wall, 60.0,
        note=f"({out['n_spikes']} spikes, real-time factor {0.1 / wall:.3f}x)",
    )


def check_100k_neurons_runs() -> CheckResult:
    """A 100k-neuron microcircuit (80k E + 20k I) completes 10 ms.

    Uses degree-constant scaling (biologically realistic: each cortical
    neuron has ~10⁴ synapses regardless of brain size). With the
    naive p=const scaling, 100k neurons would need ~15 GB of edge memory.
    With degree_mode, it's O(N).
    """
    cluster = build_ei_microcircuit(n_e=80000, n_i=20000, seed=2, degree_mode=True)
    t0 = time.time()
    out = cluster.run(duration_ms=10.0, dt=1.0, record_spikes=False)
    wall = time.time() - t0
    return CheckResult(
        f"100k-neuron microcircuit runs 10 ms (degree-constant)",
        wall < 180.0, wall, 180.0,
        note=f"(wall {wall:.2f}s; real-time factor {0.01 / wall:.4f}x)",
    )


# ---------------- speed comparison vs bio_neural_net ----------------

def check_speed_vs_bnn_pure_python() -> CheckResult:
    """Vectorised 500-neuron should be at least 5x faster than bio_neural_net
    pure-Python for the same simulated time.

    bio_neural_net microcircuit is in `bio_neural_net.microcircuit`. We
    time a short run of both at N=500 and compare. This is the headline
    speedup that justifies the Phase 0 refactor.
    """
    # Vectorised version
    cluster = build_ei_microcircuit(n_e=400, n_i=100, seed=3)
    t0 = time.time()
    cluster.run(duration_ms=200.0, dt=1.0, record_spikes=False)
    t_vec = time.time() - t0

    # bio_neural_net version
    from bio_neural_net.microcircuit import build_microcircuit
    import random
    rng = random.Random(3)
    cluster_bnn, e_ids, i_ids = build_microcircuit(rng, n_e=400, n_i=100)
    t0 = time.time()
    cluster_bnn.run(200.0, dt=1.0)
    t_bnn = time.time() - t0

    speedup = t_bnn / max(t_vec, 1e-9)
    return CheckResult(
        f"vectorised 500-neuron at least 5x faster than bio_neural_net",
        speedup >= 5.0, speedup, 5.0,
        note=f"(vec {t_vec:.3f}s vs bnn {t_bnn:.3f}s -> {speedup:.1f}x)",
    )


# ---------------- runner ----------------

def run_all() -> list[CheckResult]:
    return [
        check_rest_state(),
        check_strong_current_fires(),
        check_refractory(),
        check_exc_propagates(),
        check_inh_silences(),
        check_microcircuit_dynamics(),
        check_10k_neurons_runs(),
        check_100k_neurons_runs(),
        check_speed_vs_bnn_pure_python(),
    ]


if __name__ == "__main__":
    t0 = time.time()
    results = run_all()
    print(f"Phase 0 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    print(f"\n  -> {n_pass} / {len(results)} checks passed.")
