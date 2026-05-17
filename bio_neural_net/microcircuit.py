"""A 500-neuron cortical-style microcircuit.

This is the smallest scale at which the asynchronous-irregular balanced
state - the resting condition cortex spends most of its time in - can
be sensibly demonstrated. The microcircuit has:

  - 400 excitatory + 100 inhibitory leaky integrate-and-fire neurons
    (the canonical 80 : 20 ratio observed in mammalian cortex)
  - sparse random recurrent wiring (~5 % E->E, ~10 % E->I, ~20 % I->E)
  - membrane noise on E neurons standing in for missing external inputs
  - a port for an extra external current driving a subset of E neurons

What we measure
---------------

  1. Baseline:    no external drive, just noise. E pop fires at a low,
                  irregular rate, sustained by recurrent excitation but
                  bounded by recurrent inhibition.
  2. Stimulus:    drive a subset of E neurons. The whole circuit's rate
                  jumps; the subset jumps more.
  3. Decay:       remove the stimulus. Activity returns to the baseline
                  rate within tens of ms.

What this does not (yet) show
-----------------------------

  - Persistent working memory in the circuit itself (would need targeted
    strong recurrence between a subset of E neurons).
  - STDP learning (we can enable it, but a single-circuit STDP demo
    needs structured input, which the cluster benchmark already covers).
"""

from __future__ import annotations

import random
import time

from .network import NeuralCluster
from .neuron import NeuronParams
from .synapse import SynapseKind


# ----------------------- builders -----------------------

def build_microcircuit(
    rng: random.Random,
    n_e: int = 400,
    n_i: int = 100,
    p_ee: float = 0.04,    # weak E recurrence
    p_ei: float = 0.15,
    p_ie: float = 0.15,
    p_ii: float = 0.05,
    g_ee: float = 0.08,
    g_ei: float = 0.30,
    g_ie: float = 0.80,
    g_ii: float = 0.40,
    noise_std_e: float = 5.0,
    noise_std_i: float = 1.0,
):
    """Wire a balanced E/I microcircuit. Returns (cluster, e_ids, i_ids)."""
    cluster = NeuralCluster(rng=rng)
    e_ids = cluster.add_neurons(n_e, NeuronParams(noise_std=noise_std_e))
    i_ids = cluster.add_neurons(n_i, NeuronParams(noise_std=noise_std_i))

    def wire(srcs, dsts, p, g, kind):
        for a in srcs:
            for b in dsts:
                if a != b and rng.random() < p:
                    cluster.connect(a, b, kind, g_max=g)

    wire(e_ids, e_ids, p_ee, g_ee, SynapseKind.EXCITATORY)
    wire(e_ids, i_ids, p_ei, g_ei, SynapseKind.EXCITATORY)
    wire(i_ids, e_ids, p_ie, g_ie, SynapseKind.INHIBITORY)
    wire(i_ids, i_ids, p_ii, g_ii, SynapseKind.INHIBITORY)
    return cluster, e_ids, i_ids


# ----------------------- analysis helpers -----------------------

def population_rate(record, neuron_ids, t_lo, t_hi) -> float:
    """Mean firing rate (Hz) of the given population in [t_lo, t_hi) ms."""
    ids = set(neuron_ids)
    n = sum(
        1 for t, nid in zip(record.times, record.neuron_ids)
        if nid in ids and t_lo <= t < t_hi
    )
    return 1000.0 * n / (len(neuron_ids) * (t_hi - t_lo))


def per_neuron_isi_cv(record, neuron_ids) -> float:
    """Coefficient of variation of inter-spike intervals (proxy for irregularity).
    1.0 = Poisson, 0.0 = perfectly regular. Computed over neurons with >= 3 spikes."""
    times_by_neuron: dict[int, list[float]] = {nid: [] for nid in neuron_ids}
    for t, nid in zip(record.times, record.neuron_ids):
        if nid in times_by_neuron:
            times_by_neuron[nid].append(t)
    cvs: list[float] = []
    for spikes in times_by_neuron.values():
        if len(spikes) < 3:
            continue
        isis = [b - a for a, b in zip(spikes, spikes[1:])]
        mean = sum(isis) / len(isis)
        if mean <= 0:
            continue
        var = sum((x - mean) ** 2 for x in isis) / len(isis)
        cvs.append(var ** 0.5 / mean)
    return sum(cvs) / len(cvs) if cvs else 0.0


def rate_over_time(record, neuron_ids, total_ms, bin_ms=20.0):
    n_bins = int(total_ms / bin_ms)
    ids = set(neuron_ids)
    counts = [0] * n_bins
    for t, nid in zip(record.times, record.neuron_ids):
        if nid in ids and t < total_ms:
            counts[int(t / bin_ms)] += 1
    return [1000.0 * c / (bin_ms * len(neuron_ids)) for c in counts]


# ----------------------- demo -----------------------

def _print_rate_trace(rates, bin_ms, stim_lo=None, stim_hi=None, scale=0.3):
    print(f"  {'time (ms)':>14}  {'rate (Hz)':>10}  population activity")
    print(f"  {'-'*14}  {'-'*10}  {'-'*40}")
    for i, r in enumerate(rates):
        t = i * bin_ms
        bar = "#" * int(r * scale)
        in_stim = (stim_lo is not None and stim_lo <= t < stim_hi)
        marker = "  <-- stim" if in_stim else ""
        print(f"  {t:>6.0f}-{t+bin_ms:>5.0f}  {r:>10.2f}  {bar:<40}{marker}")


def main() -> None:
    print("=" * 72)
    print("  500-neuron microcircuit: balanced E/I dynamics")
    print("=" * 72)

    print("\nBuilding 400 E + 100 I microcircuit...", flush=True)
    t0 = time.time()
    cluster, e_ids, i_ids = build_microcircuit(random.Random(0))
    n_syn = len(cluster.synapses)
    print(f"  {n_syn} synapses, built in {time.time()-t0:.1f}s")

    # --------------- Test 1: baseline ---------------
    print("\n--- Baseline: 800 ms, no external input ---", flush=True)
    t0 = time.time()
    duration = 800.0
    rec = cluster.run(duration, dt=1.0)
    print(f"  simulated in {time.time()-t0:.1f}s ({len(rec)} spikes)")

    e_rate = population_rate(rec, e_ids, 100.0, duration)
    i_rate = population_rate(rec, i_ids, 100.0, duration)
    isi_cv = per_neuron_isi_cv(rec, e_ids)
    print(f"  E pop rate: {e_rate:5.2f} Hz   I pop rate: {i_rate:5.2f} Hz   "
          f"ISI CV: {isi_cv:.2f}  (1.0 = Poisson)")

    # --------------- Test 2: stimulus pulse ---------------
    # Build a FRESH circuit so test 2 is independent of test 1's final state.
    print("\n--- Stimulus: drive 25% of E neurons (0.5-0.7 s of a 1.0 s run) ---",
          flush=True)
    cluster, e_ids, i_ids = build_microcircuit(random.Random(1))
    stim_lo, stim_hi = 500.0, 700.0
    duration = 1000.0
    target = e_ids[:100]
    target_set = set(target)
    n_total = len(cluster.neurons)
    base_row = [0.0] * n_total
    stim_row = list(base_row)
    for nid in target:
        stim_row[nid] = 5.0

    def I(t: float):
        if stim_lo <= t < stim_hi:
            return stim_row
        return base_row

    t0 = time.time()
    rec = cluster.run(duration, dt=1.0, external_current=I)
    print(f"  simulated in {time.time()-t0:.1f}s ({len(rec)} spikes)")

    bin_ms = 25.0
    rates_target = rate_over_time(rec, target, duration, bin_ms)
    rates_other = rate_over_time(rec, [i for i in e_ids if i not in target_set],
                                  duration, bin_ms)
    rates_inh = rate_over_time(rec, i_ids, duration, bin_ms)

    base_target = sum(rates_target[2:int(stim_lo/bin_ms)]) / max(
        1, int(stim_lo/bin_ms) - 2)
    peak_target = max(rates_target[int(stim_lo/bin_ms):int(stim_hi/bin_ms)])
    base_other = sum(rates_other[2:int(stim_lo/bin_ms)]) / max(
        1, int(stim_lo/bin_ms) - 2)
    peak_other = max(rates_other[int(stim_lo/bin_ms):int(stim_hi/bin_ms)])

    print(f"\n  target E neurons (the stimulated 100):   base={base_target:5.2f} Hz "
          f"   peak={peak_target:5.2f} Hz")
    print(f"  other  E neurons (the unstimulated 300): base={base_other:5.2f} Hz "
          f"   peak={peak_other:5.2f} Hz")
    print(f"  I population peak during stimulus:        "
          f"{max(rates_inh[int(stim_lo/bin_ms):int(stim_hi/bin_ms)]):5.2f} Hz")

    print("\n  Target-population rate over the 1-second run:")
    _print_rate_trace(rates_target, bin_ms, stim_lo, stim_hi, scale=0.6)

    # --------------- Verdict ---------------
    print("\nVerdict:")
    base_ok = 0.05 < base_target < 5.0
    peak_ok = peak_target > base_target * 3.0
    last_window_rate = sum(rates_target[-4:]) / 4
    decay_ok = last_window_rate < max(base_target * 3.0, 1.5)
    sparse_ok = peak_other < base_other * 1.5  # unstimulated stays at or below baseline

    if base_ok:
        print(f"  - Baseline ({base_target:.2f} Hz) is in the irregular-low-rate regime.")
    else:
        print(f"  - Baseline ({base_target:.2f} Hz) is outside the target 0.05-5 Hz range.")
    if peak_ok:
        print(f"  - Stimulus produces a clear response "
              f"(peak {peak_target:.1f} Hz, ~{peak_target/max(base_target, 0.01):.0f}x baseline).")
    else:
        print(f"  - Stimulus response is too weak (peak/base = "
              f"{peak_target/max(base_target, 0.01):.1f}).")
    if decay_ok:
        print(f"  - Activity returns to baseline after stimulus ({last_window_rate:.2f} Hz at end).")
    else:
        print(f"  - WARNING: activity persists ({last_window_rate:.1f} Hz at end).")
    if sparse_ok:
        print(f"  - Lateral inhibition through I keeps the response sparse: unstimulated")
        print(f"    E neurons stay at {peak_other:.2f} Hz vs baseline {base_other:.2f} Hz.")


if __name__ == "__main__":
    main()
