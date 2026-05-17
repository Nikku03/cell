"""Cluster of spiking neurons connected by chemical synapses.

A `NeuralCluster` is just a bag of `LIFNeuron`s and a list of `Synapse`s
between them, advanced one timestep at a time. The whole point of this code
base is to show that *behavior lives at this level, not at the neuron level*:
patterns of connectivity in the cluster produce winner-take-all, persistent
activity, oscillation, pattern recognition, etc. None of which a single
neuron can do.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .neuron import LIFNeuron, NeuronParams
from .plasticity import STDP
from .synapse import Synapse, SynapseKind


@dataclass
class SpikeRecord:
    times: list[float] = field(default_factory=list)
    neuron_ids: list[int] = field(default_factory=list)

    def add(self, t: float, nid: int) -> None:
        self.times.append(t)
        self.neuron_ids.append(nid)

    def __len__(self) -> int:
        return len(self.times)

    def for_neuron(self, nid: int) -> list[float]:
        return [t for t, i in zip(self.times, self.neuron_ids) if i == nid]


class NeuralCluster:
    def __init__(self, rng: random.Random | None = None):
        self.neurons: list[LIFNeuron] = []
        self.synapses: list[Synapse] = []
        self.stdp: STDP | None = None
        self._rng = rng or random.Random(0)

    def add_neuron(self, params: NeuronParams | None = None) -> int:
        self.neurons.append(LIFNeuron(params, rng=self._rng))
        return len(self.neurons) - 1

    def add_neurons(self, n: int, params: NeuronParams | None = None) -> list[int]:
        return [self.add_neuron(params) for _ in range(n)]

    def connect(
        self,
        pre_id: int,
        post_id: int,
        kind: SynapseKind = SynapseKind.EXCITATORY,
        g_max: float = 0.5,
        tau_syn: float = 5.0,
        release_prob: float = 1.0,
        delay: float = 1.0,
    ) -> Synapse:
        if pre_id == post_id:
            raise ValueError("autapses (self-connections) are not supported")
        syn = Synapse(
            self.neurons[pre_id],
            self.neurons[post_id],
            kind=kind,
            g_max=g_max,
            tau_syn=tau_syn,
            release_prob=release_prob,
            delay=delay,
            rng=self._rng,
        )
        self.synapses.append(syn)
        return syn

    def enable_plasticity(self, stdp: STDP | None = None) -> STDP:
        self.stdp = stdp or STDP()
        return self.stdp

    def _build_routing(self) -> tuple[list[list[Synapse]], list[list[Synapse]]]:
        n = len(self.neurons)
        idx = {id(neu): i for i, neu in enumerate(self.neurons)}
        incoming: list[list[Synapse]] = [[] for _ in range(n)]
        outgoing: list[list[Synapse]] = [[] for _ in range(n)]
        for s in self.synapses:
            incoming[idx[id(s.post)]].append(s)
            outgoing[idx[id(s.pre)]].append(s)
        return incoming, outgoing

    def run(
        self,
        duration_ms: float,
        dt: float = 0.1,
        external_current: Callable[[float], Sequence[float]] | None = None,
    ) -> SpikeRecord:
        """Simulate for `duration_ms`. `external_current(t)` returns one current per neuron."""
        record = SpikeRecord()
        incoming, outgoing = self._build_routing()
        n = len(self.neurons)
        steps = int(round(duration_ms / dt))

        for k in range(steps):
            t = k * dt
            ext = external_current(t) if external_current is not None else [0.0] * n

            if self.stdp is not None:
                self.stdp.decay(dt)

            spiked_now: list[int] = []
            for i, neu in enumerate(self.neurons):
                I = ext[i]
                for s in incoming[i]:
                    I += s.current(t, dt)
                if neu.step(I, dt, t):
                    spiked_now.append(i)
                    record.add(t, i)

            for i in spiked_now:
                for s in outgoing[i]:
                    s.on_pre_spike(t)
                    if self.stdp is not None:
                        self.stdp.on_pre_spike(s)
                if self.stdp is not None:
                    for s in incoming[i]:
                        self.stdp.on_post_spike(s)

        return record
