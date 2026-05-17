"""Smoke tests for bio_neural_net. Run with: python -m bio_neural_net.tests.test_basic"""

from __future__ import annotations

from ..network import NeuralCluster
from ..neuron import LIFNeuron, NeuronParams
from ..synapse import SynapseKind


def test_neuron_at_rest_does_not_spike() -> None:
    n = LIFNeuron()
    for k in range(1000):
        spiked = n.step(I_syn=0.0, dt=0.1, t=k * 0.1)
        assert not spiked
    assert abs(n.V - n.p.V_rest) < 1e-9


def test_neuron_fires_under_strong_current() -> None:
    n = LIFNeuron()
    fired = False
    for k in range(1000):
        if n.step(I_syn=5.0, dt=0.1, t=k * 0.1):
            fired = True
            break
    assert fired


def test_refractory_period() -> None:
    n = LIFNeuron(NeuronParams(refractory=5.0))
    spike_times: list[float] = []
    for k in range(5000):
        t = k * 0.1
        if n.step(I_syn=10.0, dt=0.1, t=t):
            spike_times.append(t)
    assert len(spike_times) >= 2
    gaps = [b - a for a, b in zip(spike_times, spike_times[1:])]
    assert min(gaps) >= 5.0 - 1e-6


def test_excitatory_synapse_propagates_spike() -> None:
    # With an excitatory synapse, driving only the pre should also cause post to fire.
    net = NeuralCluster()
    pre, post = net.add_neurons(2)
    net.connect(pre, post, SynapseKind.EXCITATORY, g_max=1.5)
    rec = net.run(100.0, dt=0.1, external_current=lambda t: [3.0, 0.0])
    assert len(rec.for_neuron(0)) > 0
    assert len(rec.for_neuron(1)) > 0

    # Without the synapse, the post (no external drive) stays silent.
    net2 = NeuralCluster()
    net2.add_neurons(2)
    rec2 = net2.run(100.0, dt=0.1, external_current=lambda t: [3.0, 0.0])
    assert len(rec2.for_neuron(1)) == 0


def test_inhibitory_synapse_silences_post() -> None:
    net = NeuralCluster()
    pre, post = net.add_neurons(2)
    net.connect(pre, post, SynapseKind.INHIBITORY, g_max=5.0, tau_syn=20.0)

    # Drive both. Without inhibition both would fire many times.
    rec = net.run(200.0, dt=0.1, external_current=lambda t: [3.0, 2.2])
    post_count_inh = len(rec.for_neuron(1))

    net2 = NeuralCluster()
    net2.add_neurons(2)
    rec2 = net2.run(200.0, dt=0.1, external_current=lambda t: [3.0, 2.2])
    post_count_free = len(rec2.for_neuron(1))

    assert post_count_inh < post_count_free


def main() -> None:
    tests = [
        test_neuron_at_rest_does_not_spike,
        test_neuron_fires_under_strong_current,
        test_refractory_period,
        test_excitatory_synapse_propagates_spike,
        test_inhibitory_synapse_silences_post,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
