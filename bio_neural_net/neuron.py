"""Leaky integrate-and-fire (LIF) neuron.

The membrane is modelled as a leaky capacitor. Synaptic current charges or
discharges it; if the potential V crosses the firing threshold V_thresh, the
neuron emits a spike, V is reset to V_reset, and the neuron is unresponsive
for a refractory period.

This is the simplest model that captures the *qualitative* biology that makes
spiking networks different from rate-coded ones:

  - communication is discrete events in time, not continuous activations
  - the neuron has internal state (V, refractory timer) that persists between inputs
  - thresholding is hard (a spike or no spike), not a soft sigmoid
  - optional Gaussian membrane noise lets a *population* recover sub-threshold
    signals no single deterministic neuron could detect (stochastic resonance)

All voltages are in millivolts, times in milliseconds, currents in nanoamps.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass


@dataclass
class NeuronParams:
    tau_m: float = 20.0       # membrane time constant (ms)
    V_rest: float = -70.0     # resting potential (mV)
    V_thresh: float = -50.0   # firing threshold (mV)
    V_reset: float = -75.0    # post-spike reset (mV)
    R_m: float = 10.0         # membrane resistance (MOhm)
    refractory: float = 2.0   # absolute refractory period (ms)
    noise_std: float = 0.0    # Gaussian membrane noise, mV / sqrt(ms)


class LIFNeuron:
    def __init__(
        self,
        params: NeuronParams | None = None,
        rng: _random.Random | None = None,
    ):
        self.p = params or NeuronParams()
        self.V = self.p.V_rest
        self.refr_remaining = 0.0
        self.spiked = False
        self.last_spike_t = -1e9  # for STDP traces
        self._rng = rng

    def reset(self) -> None:
        self.V = self.p.V_rest
        self.refr_remaining = 0.0
        self.spiked = False
        self.last_spike_t = -1e9

    def step(self, I_syn: float, dt: float, t: float) -> bool:
        """Advance one timestep. Returns True iff this step emitted a spike."""
        self.spiked = False

        if self.refr_remaining > 0.0:
            self.refr_remaining -= dt
            self.V = self.p.V_reset
            return False

        # Euler integration of tau_m * dV/dt = -(V - V_rest) + R_m * I_syn
        dV = (-(self.V - self.p.V_rest) + self.p.R_m * I_syn) * (dt / self.p.tau_m)
        self.V += dV

        # Optional Gaussian membrane noise (Euler-Maruyama: sigma * sqrt(dt) * Z).
        if self.p.noise_std > 0.0 and self._rng is not None:
            self.V += self._rng.gauss(0.0, self.p.noise_std) * math.sqrt(dt)

        if self.V >= self.p.V_thresh:
            self.V = self.p.V_reset
            self.refr_remaining = self.p.refractory
            self.spiked = True
            self.last_spike_t = t
            return True
        return False
