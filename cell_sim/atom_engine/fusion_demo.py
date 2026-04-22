"""Fusion demo driver (field-only sanity version).

Build two small vesicles separated along z, then apply a uniform axial
attractor field to pull them together. The field is simply

    F_z(atom) = -strength * tanh(z / 0.5 nm) * ramp(t)

i.e. a constant-magnitude "downward" pull above z=0 and "upward" pull
below. The ``tanh`` avoids a discontinuity at the midplane.

There is no explicit solvent in this version. Environment-awareness in
the honest sense (lipids sensing a surrounding water shell) is added in
a later step; this script exists first as a sanity check that the
integrator + bilayer can execute the approach → contact → merge
sequence at all.

Success criterion: the pair of vesicles, initially 2 connected components
by the tail-tail proximity metric, becomes 1 connected component while
preserving the total atom count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .force_field import ForceFieldConfig
from .integrator import IntegratorConfig, SimState, current_temperature_K, step
from .vesicle import (
    VesicleSpec,
    build_two_vesicles,
    count_connected_components,
    vesicle_com_separation,
)


@dataclass
class FusionConfig:
    # Each vesicle's shape
    vesicle: VesicleSpec = field(default_factory=lambda: VesicleSpec(
        n_per_leaflet=80, radius_nm=2.0, bilayer_thickness_nm=0.9))
    z_offset_nm: float = 3.5          # center-to-center = 2 * this
    # MD
    dt_ps: float = 0.005
    target_temperature_K: float = 300.0
    thermostat_tau_ps: float = 1.0
    # Force field
    lj_cutoff_nm: float = 1.2
    max_force_kj_per_nm: float = 2.0e4
    # Driving field
    attractor_strength_kj_per_nm: float = 4.0
    attractor_ramp_ps: float = 15.0
    # Run
    equilibration_steps: int = 500
    production_steps: int = 20_000
    report_every: int = 500
    break_fraction: float = 1.8


@dataclass
class FusionResult:
    t_ps: list[float] = field(default_factory=list)
    temperature_K: list[float] = field(default_factory=list)
    n_components: list[int] = field(default_factory=list)
    com_separation_nm: list[float] = field(default_factory=list)
    initial_components: int = 0
    final_components: int = 0
    contact_time_ps: Optional[float] = None   # first time com sep < 2 * R
    merge_time_ps: Optional[float] = None     # first time comps == 1
    completed_fusion: bool = False
    bonds_formed: int = 0
    bonds_broken: int = 0
    n_atoms: int = 0


def run_fusion(
    cfg: FusionConfig,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[SimState, FusionResult]:
    atoms, bonds = build_two_vesicles(cfg.vesicle, z_offset_nm=cfg.z_offset_nm)
    state = SimState(atoms=atoms, bonds=bonds)

    result = FusionResult(
        n_atoms=len(atoms),
        initial_components=count_connected_components(atoms, bonds),
    )

    # Equilibration: no field, just let each vesicle relax at target T.
    ff_eq = ForceFieldConfig(
        lj_cutoff_nm=cfg.lj_cutoff_nm,
        max_force_kj_per_nm=cfg.max_force_kj_per_nm,
    )
    int_cfg = IntegratorConfig(
        dt_ps=cfg.dt_ps,
        thermostat_tau_ps=cfg.thermostat_tau_ps,
        target_temperature_K=cfg.target_temperature_K,
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

    # Production: turn on the attractor, watch them come together.
    ff_prod = ForceFieldConfig(
        lj_cutoff_nm=cfg.lj_cutoff_nm,
        max_force_kj_per_nm=cfg.max_force_kj_per_nm,
        use_axial_attractor=True,
        axial_attractor_axis=2,
        axial_attractor_strength_kj_per_nm=cfg.attractor_strength_kj_per_nm,
        axial_attractor_ramp_ps=state.t_ps + cfg.attractor_ramp_ps,
    )
    if progress is not None:
        progress(f"production ({cfg.production_steps} steps, "
                 f"{cfg.production_steps * cfg.dt_ps:.1f} ps total)")

    contact_threshold_nm = 2.0 * cfg.vesicle.radius_nm  # surfaces touch

    forces = None
    for k in range(cfg.production_steps):
        forces = step(state, ff_prod, int_cfg, forces)
        if (k + 1) % cfg.report_every == 0 or k == cfg.production_steps - 1:
            t = state.t_ps
            T = current_temperature_K(state.atoms)
            sep = vesicle_com_separation(state.atoms, axis=2)
            nc = count_connected_components(state.atoms, state.bonds)
            result.t_ps.append(t)
            result.temperature_K.append(T)
            result.n_components.append(nc)
            result.com_separation_nm.append(sep)
            if progress is not None:
                progress(f"prod step {k + 1}/{cfg.production_steps} "
                         f"t={t:.1f} ps  T={T:.1f} K  "
                         f"sep={sep:.2f} nm  comps={nc}")
            if result.contact_time_ps is None and sep < contact_threshold_nm:
                result.contact_time_ps = t
            if result.merge_time_ps is None and nc == 1:
                result.merge_time_ps = t

    result.bonds_formed = state.events_bonds_formed
    result.bonds_broken = state.events_bonds_broken
    result.final_components = count_connected_components(state.atoms, state.bonds)
    result.completed_fusion = result.final_components == 1 and result.initial_components == 2
    return state, result
