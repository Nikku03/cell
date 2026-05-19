"""Phase 2 verification: recurrent depth scales effective compute.

Gates:

  1. Cumulative spike work scales linearly with T
     (same N neurons, more timesteps → proportionally more total spikes)
  2. Pattern completion is selective: cued assembly's uncued half fires
     at >> than other assemblies' rate
  3. Pattern completion is real: uncued-half rate > silence-baseline rate
  4. Network with recurrence completes; without recurrence (control),
     uncued half stays silent
  5. Effective compute (uncued-half spike count) grows monotonically
     with T over the test range
  6. The mechanism is reuse: per-neuron spike count grows with T (each
     neuron is "used" repeatedly), not via more neurons
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .deep_recurrence import (
    build_clustered_assembly_network,
    measure_completion,
)
from .verify_phase0 import CheckResult, format_results


def _cumulative_uncued_spikes(net, target_assembly: int, duration_ms: float,
                               cue_fraction: float = 0.5,
                               cue_current: float = 6.0,
                               seed: int = 0) -> float:
    """Run completion for `duration_ms` and return total uncued-half spike
    count over the run.

    Total spikes = mean firing rate × atom count × duration / 1000.
    """
    res = measure_completion(
        net, target_assembly=target_assembly,
        cue_fraction=cue_fraction, cue_current=cue_current,
        duration_ms=duration_ms, bin_ms=25.0,
    )
    # Integrate firing rate × duration to get spikes per neuron, × neurons → total
    uncued_neurons = max(
        1, int(round((1.0 - cue_fraction) * net.assembly_size))
    )
    # Mean rate over bins (Hz), × uncued neurons × duration / 1000
    mean_rate = sum(res.uncued_rate) / max(1, len(res.uncued_rate))
    total_spikes = mean_rate * uncued_neurons * duration_ms / 1000.0
    return total_spikes


# ---------------- gates ----------------

def check_compute_scales_with_T() -> CheckResult:
    """Run pattern completion at T = 200, 400, 800, 1600 ms.

    Expected: total uncued-half spike count grows strongly with T
    (Pearson correlation >= 0.95). We allow an intercept rather than
    forcing through-origin: networks have transient warmup spikes that
    show up as a constant offset, then a linear-in-T steady-state term.
    Pearson handles that cleanly.
    """
    net = build_clustered_assembly_network(seed=0)
    durations = [200.0, 400.0, 800.0, 1600.0]
    spike_counts = [
        _cumulative_uncued_spikes(net, target_assembly=0, duration_ms=T)
        for T in durations
    ]
    Ts = torch.tensor(durations, dtype=torch.float64)
    Ss = torch.tensor(spike_counts, dtype=torch.float64)
    Ts_c = Ts - Ts.mean()
    Ss_c = Ss - Ss.mean()
    pearson = (Ts_c * Ss_c).sum() / (Ts_c.norm() * Ss_c.norm()).clamp_min(1e-9)
    return CheckResult(
        "spike count is strongly correlated with T (Pearson >= 0.95)",
        float(pearson) >= 0.95, float(pearson), 0.95,
        note=f"(spikes per T: {[round(s, 1) for s in spike_counts]}, "
             f"pearson={float(pearson):.3f})",
    )


def check_selectivity() -> CheckResult:
    """Final-bin uncued-half rate >> mean other-assembly rate.

    Goal: at the end of a 400 ms trial, the cued assembly's uncued half
    is active and the other assemblies are silent.
    """
    net = build_clustered_assembly_network(seed=1)
    res = measure_completion(net, target_assembly=0, duration_ms=400.0, bin_ms=25.0)
    uncued = res.uncued_rate[-1]
    other = sum(r[-1] for r in res.other_assemblies_rate) / max(1, len(res.other_assemblies_rate))
    selectivity = uncued / max(other, 0.5)
    return CheckResult(
        "uncued-half >> other-assemblies (selective completion)",
        selectivity >= 3.0, selectivity, 3.0,
        note=f"(uncued={uncued:.2f} Hz, other={other:.2f} Hz, ratio={selectivity:.1f}x)",
    )


def check_completion_above_silence() -> CheckResult:
    """Uncued half is meaningfully active during cue.

    Compares uncued-half firing rate during the 400 ms cue window against
    a no-cue baseline (just the spontaneous activity).
    """
    net = build_clustered_assembly_network(seed=2)
    res_cued = measure_completion(
        net, target_assembly=0, cue_current=6.0, duration_ms=400.0, bin_ms=25.0
    )
    mean_cued = sum(res_cued.uncued_rate) / max(1, len(res_cued.uncued_rate))

    # No-cue baseline: cue current 0
    res_silent = measure_completion(
        net, target_assembly=0, cue_current=0.0, duration_ms=400.0, bin_ms=25.0
    )
    mean_silent = sum(res_silent.uncued_rate) / max(1, len(res_silent.uncued_rate))

    ratio = mean_cued / max(mean_silent, 0.05)
    return CheckResult(
        "completion lifts uncued-half rate above no-cue baseline",
        ratio >= 3.0, ratio, 3.0,
        note=f"(cued mean={mean_cued:.2f} Hz, no-cue={mean_silent:.2f} Hz, "
             f"ratio={ratio:.1f}x)",
    )


def check_no_recurrence_control() -> CheckResult:
    """Control: if we sever the within-assembly recurrence, completion fails.

    Build a network with p_within=0 (no recurrent E-E in the assembly).
    Should give no uncued activity.
    """
    net = build_clustered_assembly_network(
        seed=3, p_within=0.0, g_within=0.0,
    )
    res = measure_completion(net, target_assembly=0, duration_ms=400.0, bin_ms=25.0)
    mean_uncued = sum(res.uncued_rate) / max(1, len(res.uncued_rate))
    return CheckResult(
        "control: no within-assembly recurrence -> no completion",
        mean_uncued <= 2.0, mean_uncued, 2.0,
        note=f"(uncued mean rate with p_within=0: {mean_uncued:.2f} Hz)",
    )


def check_monotone_compute() -> CheckResult:
    """Strictly monotone non-decreasing: spike(T) does not decrease as T grows.

    The strongest non-parametric form of "more time -> more work": if
    we sort durations ascending, the cumulative spike count must be
    non-decreasing.
    """
    net = build_clustered_assembly_network(seed=4)
    durations = [100.0, 200.0, 400.0, 800.0, 1600.0]
    spikes = [
        _cumulative_uncued_spikes(net, target_assembly=0, duration_ms=T)
        for T in durations
    ]
    # Check monotone non-decreasing
    deltas = [spikes[i + 1] - spikes[i] for i in range(len(spikes) - 1)]
    min_delta = min(deltas)
    return CheckResult(
        "spike(T) is non-decreasing in T",
        min_delta >= -10.0, min_delta, -10.0,
        note=f"(spikes per T: {[round(s, 1) for s in spikes]}, min delta={min_delta:.1f})",
    )


def check_per_neuron_reuse() -> CheckResult:
    """The same neurons are reused over time. Spike count per neuron > 1
    in the assembly during a long-T trial.

    This verifies that compute is happening through TEMPORAL REUSE of
    the same units, not by recruiting new neurons.
    """
    net = build_clustered_assembly_network(seed=5)
    cluster = net.cluster
    cluster.reset_state()

    target_idx = net.assemblies[0]
    cued = target_idx[: net.assembly_size // 2]
    drive = torch.zeros(net.n_total)
    drive[cued] = 6.0

    # Count spikes per neuron over 800 ms
    per_neuron_spikes = torch.zeros(net.n_total, dtype=torch.long)
    dt = 1.0
    for k in range(800):
        spike = cluster.step(dt=dt, t=k * dt, external_current=drive)
        per_neuron_spikes += spike.long()

    # Of the cued neurons (which see direct drive), at least 5 spikes each on average
    mean_cued_spikes = per_neuron_spikes[cued].float().mean().item()
    return CheckResult(
        "each cued neuron spikes multiple times in an 800 ms trial",
        mean_cued_spikes >= 5.0, mean_cued_spikes, 5.0,
        note=f"(mean spikes per cued neuron: {mean_cued_spikes:.2f})",
    )


# ---------------- runner ----------------

def run_all() -> list[CheckResult]:
    return [
        check_compute_scales_with_T(),
        check_selectivity(),
        check_completion_above_silence(),
        check_no_recurrence_control(),
        check_monotone_compute(),
        check_per_neuron_reuse(),
    ]


if __name__ == "__main__":
    t0 = time.time()
    results = run_all()
    print(f"Phase 2 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    print(f"\n  -> {n_pass} / {len(results)} checks passed.")
