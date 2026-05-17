"""Spike-timing-dependent plasticity (STDP).

Hebb's rule, refined: synapses strengthen if the pre-synaptic neuron fires
shortly *before* the post-synaptic neuron (the pre helped cause the post to
fire); they weaken if the pre fires *after* the post. There is no global loss
function and no gradient - just a local rule, applied per-synapse, that uses
only times of nearby spikes.

We implement the standard exponential-window STDP via pre/post traces:

  every step:                  x_pre  *= exp(-dt/tau_plus)
                               x_post *= exp(-dt/tau_minus)
  on pre  spike  (synapse s):  s.g_max += -A_minus * x_post   # post fired before
                               x_pre   += 1
  on post spike  (synapse s):  s.g_max +=  A_plus  * x_pre    # pre fired before
                               x_post  += 1

g_max is clamped to [0, g_max_cap].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .synapse import Synapse


@dataclass
class STDP:
    A_plus: float = 0.01
    A_minus: float = 0.012
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    g_max_cap: float = 4.0

    _pre_traces: dict[int, float] = field(default_factory=dict)
    _post_traces: dict[int, float] = field(default_factory=dict)

    def _pre(self, neuron) -> float:
        return self._pre_traces.get(id(neuron), 0.0)

    def _post(self, neuron) -> float:
        return self._post_traces.get(id(neuron), 0.0)

    def decay(self, dt: float) -> None:
        kp = math.exp(-dt / self.tau_plus)
        km = math.exp(-dt / self.tau_minus)
        for k in list(self._pre_traces):
            self._pre_traces[k] *= kp
        for k in list(self._post_traces):
            self._post_traces[k] *= km

    def on_pre_spike(self, syn: "Synapse") -> None:
        syn.g_max = max(0.0, syn.g_max - self.A_minus * self._post(syn.post))
        self._pre_traces[id(syn.pre)] = self._pre(syn.pre) + 1.0

    def on_post_spike(self, syn: "Synapse") -> None:
        syn.g_max = min(self.g_max_cap, syn.g_max + self.A_plus * self._pre(syn.pre))
        self._post_traces[id(syn.post)] = self._post(syn.post) + 1.0
