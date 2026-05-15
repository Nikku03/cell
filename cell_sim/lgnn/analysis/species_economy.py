"""Per-species production/consumption audit over a time window.

Given a species name and a time window, walk the trajectory and the
stoichiometric matrix to produce a complete accounting:
  - total molecules produced (and which reactions produced them)
  - total molecules consumed (and which reactions consumed them)
  - balance check: produced - consumed should equal observed Δ
  - residual: any unexplained change (chemistry outside SBML)

This is narrate_window's logic re-focused on a single species, summed
across many seconds. Useful for the "where did this come from" question
applied to ATP or any other species over an interesting time interval.

Designed to be runnable standalone - takes a pre-loaded trajectory and
stoichiometric matrix; doesn't depend on M7 training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EconomyAudit:
    species: str
    replicate: int
    t_start: int
    t_end: int
    observed_change: float
    total_produced: float
    total_consumed: float
    net_attributed: float
    residual: float
    produced_by: Dict[str, float] = field(default_factory=dict)
    consumed_by: Dict[str, float] = field(default_factory=dict)
    per_second_change: Optional[List[float]] = None
    per_second_events: Optional[List[int]] = None

    def summary(self) -> str:
        lines = [
            f'=== {self.species} economy audit ===',
            f'  replicate {self.replicate}, seconds {self.t_start} -> {self.t_end} '
            f'({self.t_end - self.t_start}s window)',
            f'  observed change in count : {self.observed_change:+,.0f}',
            f'  produced (SBML):           +{self.total_produced:,.0f}',
            f'  consumed (SBML):           {self.total_consumed:,.0f}',
            f'  net attributed:            {self.net_attributed:+,.0f}',
            f'  residual (non-SBML):       {self.residual:+,.0f}',
        ]
        if self.produced_by:
            lines.append(f'  top producers:')
            top = sorted(self.produced_by.items(), key=lambda kv: -abs(kv[1]))[:8]
            for rxn, amt in top:
                lines.append(f'    +{amt:>8,.0f}  {rxn}')
        if self.consumed_by:
            lines.append(f'  top consumers:')
            top = sorted(self.consumed_by.items(), key=lambda kv: -abs(kv[1]))[:8]
            for rxn, amt in top:
                lines.append(f'    {amt:>+8,.0f}  {rxn}')
        return '\n'.join(lines)


def audit_species_economy(
    species_name: str,
    *,
    trajectory: Any,
    row_names: List[str],
    stoich,            # tuple from load_stoich: (S, flux_indices, reaction_ids)
    replicate: int = 0,
    t_start: int = 0,
    t_end: Optional[int] = None,
) -> EconomyAudit:
    """Account for all molecules of `species_name` produced or consumed
    in [t_start, t_end) by summing reaction firings × stoichiometry."""
    if species_name not in row_names:
        raise ValueError(f'species {species_name!r} not in row_names')
    sp_idx = row_names.index(species_name)

    arr = trajectory.to_numpy() if hasattr(trajectory, 'to_numpy') else trajectory
    if t_end is None:
        t_end = arr.shape[1] - 1     # last valid timestep index
    if t_end <= t_start or t_end >= arr.shape[1]:
        raise ValueError(
            f't_start={t_start}, t_end={t_end} invalid for trajectory '
            f'with {arr.shape[1]} timesteps (valid t_end: 1..{arr.shape[1]-1})'
        )

    S, flux_indices, reaction_ids = stoich
    flux_indices = np.asarray(flux_indices)
    S = np.asarray(S)

    # window has shape (n_species, t_end - t_start + 1) - inclusive of both endpoints
    window = arr[:, t_start:t_end + 1]
    per_step_state = window[sp_idx]
    per_step_delta = np.diff(per_step_state)
    observed_change = float(window[sp_idx, -1] - window[sp_idx, 0])

    # For each reaction, total events across window = ΔF over window
    total_events_per_rxn = (
        arr[flux_indices, t_end] - arr[flux_indices, t_start]
    )  # shape (n_reactions,)

    produced_by: Dict[str, float] = {}
    consumed_by: Dict[str, float] = {}
    total_produced = 0.0
    total_consumed = 0.0
    for col_idx, (rxn_id, n_events) in enumerate(
        zip(reaction_ids, total_events_per_rxn.tolist())
    ):
        stoich_coef = float(S[sp_idx, col_idx])
        if stoich_coef == 0 or abs(n_events) < 0.5:
            continue
        contribution = stoich_coef * n_events
        if stoich_coef > 0:
            produced_by[rxn_id] = contribution
            total_produced += contribution
        else:
            consumed_by[rxn_id] = contribution
            total_consumed += contribution

    net_attributed = total_produced + total_consumed
    residual = observed_change - net_attributed

    return EconomyAudit(
        species=species_name,
        replicate=int(replicate),
        t_start=int(t_start),
        t_end=int(t_end),
        observed_change=observed_change,
        total_produced=total_produced,
        total_consumed=total_consumed,
        net_attributed=net_attributed,
        residual=residual,
        produced_by=produced_by,
        consumed_by=consumed_by,
        per_second_change=per_step_delta.tolist(),
        per_second_events=None,
    )
