"""Pre-seeded-molecule chemistry demo.

Instead of starting from atomic soup, build a mixture of small
molecules (H2, O2, CH4, NH3, H2O, ...), turn on dynamic bonding, and
watch the molecular populations evolve at an elevated temperature.

Success signal: bond-break events on the starting molecules and
bond-form events between fragments, producing molecules that were not
in the initial mix — e.g. (2 H2 + O2) producing H2O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .atom_unit import BondType
from .force_field import ForceFieldConfig
from .integrator import IntegratorConfig, SimState, current_temperature_K, step
from .molecule_builder import build_mixture, classify_molecules


@dataclass
class ChemistryConfig:
    composition: dict[str, int] = field(default_factory=lambda: {
        "H2": 80, "O2": 40,
    })
    radius_nm: float = 3.0
    # MD
    dt_ps: float = 0.0005                      # 0.5 fs — bonds are stiff
    thermostat_tau_ps: float = 0.1
    target_temperature_K: float = 3000.0
    # Force field
    lj_cutoff_nm: float = 1.0
    use_confinement: bool = True
    confinement_k_kj_per_nm2: float = 2.0e3
    max_force_kj_per_nm: float = 5.0e4
    use_neighbor_list: bool = True
    neighbor_skin_nm: float = 0.3
    neighbor_rebuild_every: int = 10
    # Shrink LJ sigma for reactive elements so atoms can actually reach
    # bond-forming distance through the LJ core. 0.3 = sigma / 3, which
    # puts the LJ well roughly at 0.08-0.10 nm for H/C/N/O — same order
    # as their covalent bond lengths.
    reactive_sigma_scale: float = 0.3
    # Dynamic bonding — default to a SOFT toy-chemistry bond (k = 5e4
    # kJ/mol/nm^2, bond energy ~90 kJ/mol) instead of a realistic
    # stiff covalent bond (k = 3e5, ~400 kJ/mol). Soft bonds let the
    # system explore break -> reform cycles at 3000 K within a few ps.
    # Use ``bond_k_kj_per_nm2`` to override both the initial and
    # dynamically-formed bond stiffness.
    bond_form_distance_nm: float = 0.20     # outer gate (neighbor pre-filter)
    bond_form_ratio: float = 1.3             # form if r < form_ratio * r0_pair
    bond_form_k_kj_per_nm2: float = 5.0e4
    bond_form_r0_nm: float = 0.12
    bond_break_fraction: float = 1.8         # break if r > 1.8 * r0_pair
    bond_form_kind: BondType = BondType.COVALENT_SINGLE
    initial_bond_k_kj_per_nm2: Optional[float] = 5.0e4
    # Run
    equilibration_steps: int = 500
    steps: int = 20_000
    report_every: int = 1000


@dataclass
class ChemistryResult:
    t_ps: list[float] = field(default_factory=list)
    temperature_K: list[float] = field(default_factory=list)
    formula_snapshots: list[dict[str, int]] = field(default_factory=list)
    total_bonds_formed: int = 0
    total_bonds_broken: int = 0
    initial_formulas: dict[str, int] = field(default_factory=dict)
    final_formulas: dict[str, int] = field(default_factory=dict)
    n_atoms: int = 0
    n_bonds_initial: int = 0


def _diff_formulas(a: dict[str, int],
                   b: dict[str, int]) -> dict[str, int]:
    """Return ``b - a`` entry-wise (positive = appeared in b, negative = lost)."""
    keys = set(a) | set(b)
    return {k: b.get(k, 0) - a.get(k, 0) for k in keys
            if b.get(k, 0) != a.get(k, 0)}


def run_chemistry(
    cfg: ChemistryConfig,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[SimState, ChemistryResult]:
    atoms, bonds, angles = build_mixture(
        cfg.composition,
        radius_nm=cfg.radius_nm,
        temperature_K=cfg.target_temperature_K,
        bond_k_kj_per_nm2=cfg.initial_bond_k_kj_per_nm2,
    )
    state = SimState(atoms=atoms, bonds=bonds, angles=angles)

    result = ChemistryResult(
        n_atoms=len(atoms),
        n_bonds_initial=len(bonds),
        initial_formulas=classify_molecules(atoms),
    )

    ff = ForceFieldConfig(
        lj_cutoff_nm=cfg.lj_cutoff_nm,
        max_force_kj_per_nm=cfg.max_force_kj_per_nm,
        use_confinement=cfg.use_confinement,
        confinement_radius_nm=cfg.radius_nm,
        confinement_k_kj_per_nm2=cfg.confinement_k_kj_per_nm2,
        use_neighbor_list=cfg.use_neighbor_list,
        neighbor_skin_nm=cfg.neighbor_skin_nm,
        reactive_sigma_scale=cfg.reactive_sigma_scale,
    )
    int_cfg_eq = IntegratorConfig(
        dt_ps=cfg.dt_ps,
        thermostat_tau_ps=cfg.thermostat_tau_ps,
        target_temperature_K=cfg.target_temperature_K,
        dynamic_bonding=False,         # keep topology fixed while warming up
        bond_break_fraction=cfg.bond_break_fraction,
        neighbor_rebuild_every=cfg.neighbor_rebuild_every,
    )
    int_cfg = IntegratorConfig(
        dt_ps=cfg.dt_ps,
        thermostat_tau_ps=cfg.thermostat_tau_ps,
        target_temperature_K=cfg.target_temperature_K,
        dynamic_bonding=True,
        bond_form_distance_nm=cfg.bond_form_distance_nm,
        bond_form_ratio=cfg.bond_form_ratio,
        bond_form_k_kj_per_nm2=cfg.bond_form_k_kj_per_nm2,
        bond_form_r0_nm=cfg.bond_form_r0_nm,
        bond_break_fraction=cfg.bond_break_fraction,
        bond_form_kind=cfg.bond_form_kind,
        neighbor_rebuild_every=cfg.neighbor_rebuild_every,
    )

    if progress is not None:
        progress(f"atoms={len(atoms)} bonds_initial={len(bonds)} "
                 f"T_target={cfg.target_temperature_K:.0f} K "
                 f"dt={cfg.dt_ps} ps")
        progress(f"initial: {result.initial_formulas}")

    # Equilibrate without reactive bonding.
    forces = None
    for _ in range(cfg.equilibration_steps):
        forces = step(state, ff, int_cfg_eq, forces)

    if progress is not None:
        progress(f"equilibrated to T={current_temperature_K(state.atoms):.0f} K; "
                 f"starting reactive phase")

    forces = None
    for k in range(cfg.steps):
        forces = step(state, ff, int_cfg, forces)
        if (k + 1) % cfg.report_every == 0 or k == cfg.steps - 1:
            t = state.t_ps
            T = current_temperature_K(state.atoms)
            snap = classify_molecules(state.atoms)
            result.t_ps.append(t)
            result.temperature_K.append(T)
            result.formula_snapshots.append(snap)
            if progress is not None:
                top = sorted(snap.items(), key=lambda kv: -kv[1])[:6]
                progress(f"step {k+1}/{cfg.steps} t={t:.2f} ps "
                         f"T={T:.0f} K formed={state.events_bonds_formed} "
                         f"broken={state.events_bonds_broken} "
                         f"top={top}")

    result.total_bonds_formed = state.events_bonds_formed
    result.total_bonds_broken = state.events_bonds_broken
    result.final_formulas = classify_molecules(state.atoms)
    return state, result
