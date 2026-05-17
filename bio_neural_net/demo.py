"""Four demonstrations of the claim: a single neuron does nothing intelligent,
but a *cluster* with the right wiring does brain-like things.

Run::

    python -m bio_neural_net.demo

Each demo prints a spike raster (one row per neuron, time runs left to right).
"""

from __future__ import annotations

import random

from .network import NeuralCluster
from .neuron import NeuronParams
from .plasticity import STDP
from .synapse import SynapseKind


# ----------------------------- raster printing -----------------------------

def raster(record, n_neurons: int, duration_ms: float, width: int = 80) -> str:
    bins = [["." for _ in range(width)] for _ in range(n_neurons)]
    for t, nid in zip(record.times, record.neuron_ids):
        col = min(width - 1, int(t / duration_ms * width))
        bins[nid][col] = "|"
    lines = []
    for nid, row in enumerate(bins):
        spike_count = sum(1 for c in row if c == "|")
        lines.append(f"n{nid:02d} |{''.join(row)}|  ({spike_count} spikes)")
    return "\n".join(lines)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ----------------------------- demo 1: lone neuron -----------------------------

def demo_single_neuron() -> None:
    section("DEMO 1 — A single neuron in isolation")
    print(
        "Inject increasing current into one LIF neuron. All it does is spike\n"
        "faster as the input grows. No representation, no memory, no decision —\n"
        "it is a leaky threshold detector."
    )

    net = NeuralCluster()
    net.add_neuron()

    duration = 300.0
    def I(t: float):
        # Three 100 ms steps with growing amplitude.
        if t < 100.0:  return [1.5]
        if t < 200.0:  return [2.5]
        return [3.5]

    rec = net.run(duration, dt=0.1, external_current=I)
    print()
    print(raster(rec, 1, duration))
    print("\nKey point: changing the input amplitude only changes the firing rate.")
    print("There is no 'concept' that this neuron represents on its own.")


# --------------------- demo 2: winner-take-all (decision) ---------------------

def demo_winner_take_all() -> None:
    section("DEMO 2 — Winner-take-all: a 4-neuron cluster makes a decision")
    print(
        "Three excitatory 'option' neurons each receive a slightly different\n"
        "input current. They all excite a single inhibitor, which in turn\n"
        "inhibits all three. The strongest option suppresses the others —\n"
        "the cluster chooses. No individual neuron has this property.\n"
    )

    net = NeuralCluster(rng=random.Random(1))
    a, b, c = net.add_neurons(3)
    inh = net.add_neuron()

    # Each option drives the inhibitor; the inhibitor pulls them all down.
    for opt in (a, b, c):
        net.connect(opt, inh, SynapseKind.EXCITATORY, g_max=0.6)
        net.connect(inh, opt, SynapseKind.INHIBITORY, g_max=2.0)

    duration = 400.0
    # Slightly different drives: b is strongest.
    drives = [2.2, 2.6, 2.4, 0.0]
    rec = net.run(duration, dt=0.1, external_current=lambda t: drives)

    print(raster(rec, 4, duration))
    counts = [len(rec.for_neuron(i)) for i in range(4)]
    winner = max(range(3), key=lambda i: counts[i])
    print(f"\nSpike counts: a={counts[0]}, b={counts[1]}, c={counts[2]}, inh={counts[3]}")
    print(f"Winner: neuron {chr(ord('a') + winner)}. The cluster's *connectivity*")
    print("turned three near-equal inputs into a clean decision.")


# --------------- demo 3: persistent memory (recurrence holds state) ----------

def demo_persistent_memory() -> None:
    section("DEMO 3 — Working memory: a cluster remembers after input ends")
    print(
        "Two excitatory neurons connected to each other. A brief input pulse\n"
        "(first 50 ms) starts them firing; their mutual excitation then keeps\n"
        "them firing long after the pulse is gone. The information ('input\n"
        "happened') is held in the *activity pattern of the cluster*, not in\n"
        "any one neuron's state. This is how prefrontal cortex holds a\n"
        "phone number for a few seconds.\n"
    )

    net = NeuralCluster(rng=random.Random(2))
    a, b = net.add_neurons(2)
    net.connect(a, b, SynapseKind.EXCITATORY, g_max=1.4, tau_syn=15.0)
    net.connect(b, a, SynapseKind.EXCITATORY, g_max=1.4, tau_syn=15.0)

    duration = 500.0
    def I(t: float):
        if t < 50.0:
            return [3.0, 3.0]   # brief kick
        return [0.0, 0.0]       # no external input
    rec = net.run(duration, dt=0.1, external_current=I)

    print(raster(rec, 2, duration))
    late = sum(1 for t in rec.times if t > 100.0)
    print(f"\nSpikes after the input pulse ended (t > 100 ms): {late}")
    print("A single isolated neuron, given the same brief kick, would fall")
    print("silent within ~30 ms. The memory lives in the loop, not the neuron.")


# ---------------- demo 4: Hebbian learning recognises a pattern --------------

def demo_hebbian_pattern() -> None:
    section("DEMO 4 — Hebbian/STDP learning, no gradients, no labels")
    print(
        "Four input neurons feed into one output neuron. Inputs {0, 1} are\n"
        "driven together; inputs {2, 3} fire independently as noise. STDP\n"
        "strengthens the {0, 1} -> output synapses (because they reliably\n"
        "fire just before the output) and weakens the noisy ones.\n"
        "\n"
        "After training, we present each pair alone and count output spikes.\n"
    )

    rng = random.Random(3)
    net = NeuralCluster(rng=rng)
    inputs = net.add_neurons(4)
    out = net.add_neuron()
    # Initial g_max is deliberately low: a *single* input cannot drive the
    # output to threshold. Only the simultaneous arrival of two inputs can.
    syns = [net.connect(i, out, SynapseKind.EXCITATORY, g_max=0.7) for i in inputs]
    net.enable_plasticity(STDP(A_plus=0.04, A_minus=0.04, g_max_cap=3.0))

    PULSE_I = 12.0
    PULSE_MS = 5.0
    CYCLE_MS = 50.0            # period of coordinated firing -> ~20 Hz
    NOISE_RATE_HZ = 10.0       # mean rate of each noise channel
    p_noise_per_step = NOISE_RATE_HZ * 1e-3 * 0.1
    noise_on_until = [0.0, 0.0]

    def train_current(t: float):
        coord_on = ((t % CYCLE_MS) < PULSE_MS)
        I = [
            PULSE_I if coord_on else 0.0,
            PULSE_I if coord_on else 0.0,
            0.0,
            0.0,
        ]
        for k, nid in enumerate((2, 3)):
            if t < noise_on_until[k]:
                I[nid] = PULSE_I
            elif rng.random() < p_noise_per_step:
                noise_on_until[k] = t + PULSE_MS
                I[nid] = PULSE_I
        return I + [0.0]

    net.run(3000.0, dt=0.1, external_current=train_current)

    g_after = [s.g_max for s in syns]
    print(f"Synaptic strengths after training (g_max):")
    print(f"  input 0 -> out: {g_after[0]:.2f}   (co-fires with 1)")
    print(f"  input 1 -> out: {g_after[1]:.2f}   (co-fires with 0)")
    print(f"  input 2 -> out: {g_after[2]:.2f}   (noise)")
    print(f"  input 3 -> out: {g_after[3]:.2f}   (noise)")

    # Disable plasticity during recall so the test does not retrain.
    net.stdp = None

    def probe(active: tuple[int, int], duration_ms: float = 200.0) -> int:
        for n in net.neurons:
            n.reset()
        for s in net.synapses:
            s.g = 0.0
            s._pending.clear()
        def I(t):
            row = [0.0] * 5
            for i in active:
                row[i] = 12.0 if ((t % 50.0) < 5.0) else 0.0
            return row
        rec = net.run(duration_ms, dt=0.1, external_current=I)
        return len(rec.for_neuron(4))

    learned_count = probe((0, 1))
    noise_count = probe((2, 3))
    print(f"\nRecall — output spikes in 200 ms with only the learned pair (0,1): {learned_count}")
    print(f"Recall — output spikes in 200 ms with only the noise pair    (2,3): {noise_count}")
    print("\nThe cluster has formed a selective detector through purely local,")
    print("time-based synaptic changes. No backpropagation. No teacher signal.")


def main() -> None:
    demo_single_neuron()
    demo_winner_take_all()
    demo_persistent_memory()
    demo_hebbian_pattern()
    print()


if __name__ == "__main__":
    main()
