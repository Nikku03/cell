"""Chemical synapse with conductance dynamics.

A real synapse is *not* a scalar weight w that gets multiplied with the
pre-synaptic activation. When the pre-synaptic neuron spikes, it (probabilistically)
releases neurotransmitter, which opens ion channels on the post-synaptic membrane.
Those channels have a reversal potential E_syn: the current they pass depends on
how far the membrane is from E_syn, not on a scalar input value.

  I_syn = g(t) * (V_post - E_syn)

  - Excitatory (AMPA-like):   E_syn ~ 0 mV   -> pushes V toward 0, depolarizing
  - Inhibitory (GABA-like):  E_syn ~ -80 mV -> pulls V toward -80, hyperpolarizing

Inhibition is therefore *not* "a negative weight" - it is the same multiplicative
conductance machinery, just with a different reversal potential. Two key
consequences of this that scalar-weight nets cannot reproduce naturally:

  1. shunting inhibition: a strong inhibitory conductance near V_rest produces
     little current on its own but divisively reduces the impact of excitation.
  2. inhibition becomes weaker as V approaches E_inh, capping how hyperpolarized
     the membrane can get.
"""

from __future__ import annotations

import enum
import math
import random


class SynapseKind(enum.Enum):
    EXCITATORY = "exc"
    INHIBITORY = "inh"


_E_SYN = {SynapseKind.EXCITATORY: 0.0, SynapseKind.INHIBITORY: -80.0}


class Synapse:
    def __init__(
        self,
        pre,                    # LIFNeuron
        post,                   # LIFNeuron
        kind: SynapseKind = SynapseKind.EXCITATORY,
        g_max: float = 0.5,
        tau_syn: float = 5.0,
        release_prob: float = 1.0,
        delay: float = 1.0,
        rng: random.Random | None = None,
    ):
        self.pre = pre
        self.post = post
        self.kind = kind
        self.E_syn = _E_SYN[kind]
        self.g_max = g_max
        self.tau_syn = tau_syn
        self.release_prob = release_prob
        self.delay = delay
        self.g = 0.0
        self._pending: list[float] = []  # spike-arrival times (post-delay)
        self._rng = rng or random.Random()

    def on_pre_spike(self, t: float) -> None:
        if self._rng.random() < self.release_prob:
            self._pending.append(t + self.delay)

    def current(self, t: float, dt: float) -> float:
        # Deliver any pending vesicle releases whose delay has elapsed.
        still_pending = []
        for arrival in self._pending:
            if arrival <= t:
                self.g += self.g_max
            else:
                still_pending.append(arrival)
        self._pending = still_pending

        # Exponential decay of the conductance.
        self.g *= math.exp(-dt / self.tau_syn)

        # Driving-force current.
        return self.g * (self.E_syn - self.post.V)
