"""Fission demo driver.

Take a toy bilayer vesicle (see :mod:`.vesicle`), apply a Gaussian radial
constriction force at the equator, run the MD integrator, and observe the
membrane pinch into two daughter compartments.

This is the smallest self-assembly experiment that proves the AtomUnit
primitive works end-to-end:
  1. Atoms with finite valence and categorical identity form bonds.
  2. Bonded + non-bonded forces drive the geometry.
  3. An external driving force reshapes the topology.
  4. A bookkeeping layer (event log + connected-components) reports
     whether a topology change happened, without any fixed time grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .force_field import ForceFieldConfig
from .integrator import IntegratorConfig, SimState, current_temperature_K, step
from .vesicle import (
    VesicleSpec,
    build_vesicle,
    count_connected_components,
    equatorial_split_metric,
    hemisphere_split_balance,
    is_bimodal_along_axis,
)


@dataclass
class FissionConfig:
    # Vesicle shape
    vesicle: VesicleSpec = field(default_factory=VesicleSpec)
    # MD
    dt_ps: float = 0.005
    target_temperature_K: float = 300.0
    thermostat_tau_ps: float = 1.0
    # Force field
    lj_cutoff_nm: float = 1.2
    constriction_width_nm: float = 0.8
    constriction_k_kj_per_nm2: float = 800.0
    constriction_ramp_ps: float = 10.0
    max_force_kj_per_nm: float = 2.0e4
    # Run
    equilibration_steps: int = 500
    production_steps: int = 20_000
    report_every: int = 500
    break_fraction: float = 1.8           # chain bonds are tough


@dataclass
class FissionResult:
    # Trajectory of scalar metrics, one entry per `report_every` steps.
    t_ps: list[float] = field(default_factory=list)
    temperature_K: list[float] = field(default_factory=list)
    neck_fraction: list[float] = field(default_factory=list)
    n_components: list[int] = field(default_factory=list)
    hemisphere_imbalance: list[float] = field(default_factory=list)
    bimodal: list[bool] = field(default_factory=list)
    # Summary
    initial_components: int = 0
    final_components: int = 0
    pinch_time_ps: Optional[float] = None         # neck-fraction → 0
    bimodal_time_ps: Optional[float] = None       # both hemispheres populated
    completed_fission: bool = False               # bimodal AND balanced
    # Event log
    bonds_formed: int = 0
    bonds_broken: int = 0
    n_atoms: int = 0
    n_bonds_initial: int = 0


def run_fission(
    cfg: FissionConfig,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[SimState, FissionResult]:
    """Run the equilibration + constriction + pinch-off sequence."""
    atoms, bonds = build_vesicle(cfg.vesicle)
    state = SimState(atoms=atoms, bonds=bonds)

    result = FissionResult(
        n_atoms=len(atoms),
        n_bonds_initial=len(bonds),
        initial_components=count_connected_components(atoms, bonds),
    )

    # Equilibration: no constriction.
    ff_eq = ForceFieldConfig(
        lj_cutoff_nm=cfg.lj_cutoff_nm,
        max_force_kj_per_nm=cfg.max_force_kj_per_nm,
        use_constriction=False,
    )
    int_cfg = IntegratorConfig(
        dt_ps=cfg.dt_ps,
        thermostat_tau_ps=cfg.thermostat_tau_ps,
        target_temperature_K=cfg.target_temperature_K,
        dynamic_bonding=False,
        bond_break_fraction=cfg.break_fraction,
    )
    if progress is not None:
        progress(f"equilibrating ({cfg.equilibration_steps} steps)")

    forces = None
    for k in range(cfg.equilibration_steps):
        forces = step(state, ff_eq, int_cfg, forces)
        if progress is not None and (k + 1) % cfg.report_every == 0:
            progress(f"eq step {k + 1}/{cfg.equilibration_steps} "
                     f"T={current_temperature_K(state.atoms):.1f} K")

    # Production: ramp up constriction, let fission proceed.
    ff_prod = ForceFieldConfig(
        lj_cutoff_nm=cfg.lj_cutoff_nm,
        max_force_kj_per_nm=cfg.max_force_kj_per_nm,
        use_constriction=True,
        constriction_axis=2,
        constriction_width_nm=cfg.constriction_width_nm,
        constriction_k_kj_per_nm2=cfg.constriction_k_kj_per_nm2,
        constriction_ramp_ps=cfg.constriction_ramp_ps,
    )

    # Time reference for the ramp: reset so ramp starts at 0 ps of production.
    t0 = state.t_ps

    if progress is not None:
        progress(f"production ({cfg.production_steps} steps, "
                 f"dt={cfg.dt_ps} ps = "
                 f"{cfg.production_steps * cfg.dt_ps:.1f} ps total)")
    # The ramp inside compute_forces is measured against absolute state.t_ps.
    # t0 is the start of production; we want full constriction reached after
    # cfg.constriction_ramp_ps of *production* time, so the effective ramp
    # endpoint is t0 + constriction_ramp_ps.
    ff_prod.constriction_ramp_ps = t0 + cfg.constriction_ramp_ps
    forces = None
    for k in range(cfg.production_steps):
        forces = step(state, ff_prod, int_cfg, forces)
        if (k + 1) % cfg.report_every == 0 or k == cfg.production_steps - 1:
            t = state.t_ps
            T = current_temperature_K(state.atoms)
            neck = equatorial_split_metric(state.atoms, axis=2)
            nc = count_connected_components(state.atoms, state.bonds)
            imb = hemisphere_split_balance(state.atoms, axis=2)
            bi = is_bimodal_along_axis(state.atoms, axis=2)
            result.t_ps.append(t)
            result.temperature_K.append(T)
            result.neck_fraction.append(neck)
            result.n_components.append(nc)
            result.hemisphere_imbalance.append(imb)
            result.bimodal.append(bi)
            if progress is not None:
                progress(f"prod step {k + 1}/{cfg.production_steps} "
                         f"t={t:.1f} ps  T={T:.1f} K  "
                         f"neck={neck:.2f}  comps={nc}  "
                         f"imbalance={imb:.2f}  bimodal={bi}")
            if result.pinch_time_ps is None and neck < 0.05:
                result.pinch_time_ps = t
            if result.bimodal_time_ps is None and bi:
                result.bimodal_time_ps = t

    result.bonds_formed = state.events_bonds_formed
    result.bonds_broken = state.events_bonds_broken
    result.final_components = count_connected_components(state.atoms, state.bonds)
    # A successful fission leaves both hemispheres populated with a nearly
    # empty neck and a roughly balanced split.
    final_bi = is_bimodal_along_axis(state.atoms, axis=2)
    final_imb = hemisphere_split_balance(state.atoms, axis=2)
    result.completed_fission = bool(final_bi and final_imb < 0.4)
    return state, result
