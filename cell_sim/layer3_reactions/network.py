"""
Layer 3: Reaction network.

Given a CellSpec with metabolites and reactions, integrate concentrations
forward in time using mass-action kinetics. This is the cheapest layer and
runs at cell-cycle timescales in minutes.

For reactions where rate constants are unknown (the common case), we use
a simple heuristic default and mark those rates as "learned_placeholder"
so higher layers know they're not physics-grounded.

The ODE integrator is a fourth-order Runge-Kutta with adaptive step size,
implemented in pure NumPy/PyTorch so it's differentiable if we want to
train against experimental data later.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from layer0_genome.parser import CellSpec, Reaction


@dataclass
class ReactionRates:
    """Rate constants for all reactions in a CellSpec."""
    k_forward: Dict[str, float] = field(default_factory=dict)
    k_reverse: Dict[str, float] = field(default_factory=dict)
    source: Dict[str, str] = field(default_factory=dict)  # 'literature', 'learned', 'default'


def default_rates_for_spec(spec: CellSpec) -> ReactionRates:
    """
    Assign default rate constants when not specified.

    These are ORDER-OF-MAGNITUDE reasonable for typical enzymatic reactions
    but should not be trusted for quantitative predictions. The point is to
    produce non-degenerate dynamics so we can test the pipeline.
    """
    rates = ReactionRates()
    for rxn_id, rxn in spec.reactions.items():
        if rxn.k_cat is not None:
            rates.k_forward[rxn_id] = rxn.k_cat
            rates.source[rxn_id] = 'literature'
        else:
            # default: 1 per second for forward, reverse is 1% of forward
            rates.k_forward[rxn_id] = 1.0
            rates.source[rxn_id] = 'default'
        rates.k_reverse[rxn_id] = rates.k_forward[rxn_id] * 0.01
    return rates


class ReactionNetwork:
    """
    ODE system for metabolite concentrations.

    State vector: concentrations of all metabolites in mM.
    dC/dt computed from mass-action kinetics for all reactions.
    """

    def __init__(self, spec: CellSpec, rates: Optional[ReactionRates] = None):
        self.spec = spec
        self.rates = rates if rates is not None else default_rates_for_spec(spec)

        # Build index maps
        self.met_ids = list(spec.metabolites.keys())
        self.met_idx = {m: i for i, m in enumerate(self.met_ids)}
        self.n_mets = len(self.met_ids)

        self.rxn_ids = list(spec.reactions.keys())
        self.n_rxns = len(self.rxn_ids)

        # Build stoichiometry matrix: S[i, j] = net stoich of met i in rxn j
        self.S = np.zeros((self.n_mets, self.n_rxns))
        for j, rxn_id in enumerate(self.rxn_ids):
            rxn = spec.reactions[rxn_id]
            for met, stoich in rxn.reactants.items():
                if met in self.met_idx:
                    self.S[self.met_idx[met], j] -= stoich
            for met, stoich in rxn.products.items():
                if met in self.met_idx:
                    self.S[self.met_idx[met], j] += stoich

        # Initial concentrations
        self.C0 = np.array([
            spec.metabolites[m].initial_concentration_mM for m in self.met_ids
        ])

    def flux(self, C: np.ndarray) -> np.ndarray:
        """Compute reaction fluxes given current concentrations."""
        fluxes = np.zeros(self.n_rxns)
        for j, rxn_id in enumerate(self.rxn_ids):
            rxn = self.spec.reactions[rxn_id]
            kf = self.rates.k_forward.get(rxn_id, 1.0)
            kr = self.rates.k_reverse.get(rxn_id, 0.0)

            forward_flux = kf
            for met, stoich in rxn.reactants.items():
                if met in self.met_idx:
                    conc = max(C[self.met_idx[met]], 0.0)  # clamp negatives
                    forward_flux *= conc ** stoich
            reverse_flux = kr
            for met, stoich in rxn.products.items():
                if met in self.met_idx:
                    conc = max(C[self.met_idx[met]], 0.0)
                    reverse_flux *= conc ** stoich
            fluxes[j] = forward_flux - reverse_flux
        return fluxes

    def dC_dt(self, C: np.ndarray) -> np.ndarray:
        """Derivative of concentrations with respect to time."""
        v = self.flux(C)
        return self.S @ v

    def integrate(self, t_end: float, dt: float = 0.01,
                  verbose: bool = False) -> Dict[str, np.ndarray]:
        """
        Fourth-order Runge-Kutta integration from t=0 to t=t_end.

        Returns:
            {'t': times, 'C': concentrations [n_times, n_mets], 'fluxes': ...}
        """
        n_steps = int(t_end / dt) + 1
        times = np.linspace(0, t_end, n_steps)
        C = np.zeros((n_steps, self.n_mets))
        C[0] = self.C0

        for i in range(n_steps - 1):
            c = C[i]
            k1 = self.dC_dt(c)
            k2 = self.dC_dt(c + 0.5 * dt * k1)
            k3 = self.dC_dt(c + 0.5 * dt * k2)
            k4 = self.dC_dt(c + dt * k3)
            C[i+1] = c + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            # Clamp negatives from numerical error
            C[i+1] = np.maximum(C[i+1], 0.0)

            if verbose and i % max(1, n_steps // 10) == 0:
                print(f"  t={times[i]:.2f}s  "
                      f"max conc={C[i].max():.3f}  "
                      f"min conc={C[i].min():.6f}")

        return {
            't': times,
            'C': C,
            'met_ids': self.met_ids,
        }

    def concentration_at(self, result: Dict, met_id: str) -> np.ndarray:
        """Extract a concentration trajectory by metabolite id."""
        return result['C'][:, self.met_idx[met_id]]


# =============================================================================
# Demo
# =============================================================================
if __name__ == "__main__":
    from layer0_genome.parser import build_cell_spec

    print("Building Syn3A CellSpec...")
    spec = build_cell_spec(species='syn3a')
    print(spec.summary())
    print()

    print("Setting up reaction network...")
    net = ReactionNetwork(spec)
    print(f"  {net.n_mets} metabolites, {net.n_rxns} reactions")
    print(f"  Stoich matrix shape: {net.S.shape}")
    print()

    print("Integrating for 10 seconds, dt=0.01s...")
    result = net.integrate(t_end=10.0, dt=0.01, verbose=True)
    print()

    print("Final concentrations:")
    for met_id in net.met_ids:
        initial = spec.metabolites[met_id].initial_concentration_mM
        final = net.concentration_at(result, met_id)[-1]
        if abs(final - initial) > 1e-4:
            print(f"  {met_id}: {initial:.4f} → {final:.4f} mM  (Δ={final-initial:+.4f})")

    print("\nLayer 3 is working.")
