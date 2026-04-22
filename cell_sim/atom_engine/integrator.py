"""Velocity-Verlet integrator with Berendsen thermostat.

Tight, deterministic, built on numpy. All forces in kJ/(mol·nm), masses in
Da, times in ps — accelerations in nm/ps^2 by unit consistency
(1 kJ/mol / 1 Da / 1 nm = 1 nm/ps^2).

The integrator also drives bond-forming / bond-breaking events:

- Bonds break when `r > break_fraction * equilibrium_length` (default 1.5).
- Bonds can OPTIONALLY form between atoms of compatible elements when they
  approach within `form_distance_nm` and both still have valence remaining.

Every event is appended to the atom history via :meth:`AtomUnit.break_bond`
/ :meth:`AtomUnit.form_bond` — so the full trajectory of bond changes is
preserved without any fixed time grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .atom_unit import AtomUnit, Bond, BondType
from .element import pair_is_bondable
from .force_field import ForceFieldConfig, build_neighbor_list, compute_forces


@dataclass
class IntegratorConfig:
    dt_ps: float = 0.005                     # 5 fs timestep
    thermostat_tau_ps: float = 1.0
    target_temperature_K: float = 300.0
    dynamic_bonding: bool = False            # whether to form/break bonds
    bond_form_distance_nm: float = 0.18
    bond_break_fraction: float = 1.5          # break if r > 1.5 * r0
    bond_form_kind: BondType = BondType.COVALENT_SINGLE
    bond_form_k_kj_per_nm2: float = 2.0e4
    bond_form_r0_nm: float = 0.15
    # Neighbor-list scheduler: rebuild every N steps when the force field
    # has use_neighbor_list=True. 0 disables (caller manages the list).
    neighbor_rebuild_every: int = 10
    # Callback invoked each step with (t_ps, n_bonds_formed, n_bonds_broken).
    step_callback: Optional[callable] = None


@dataclass
class SimState:
    atoms: list[AtomUnit]
    bonds: list[Bond] = field(default_factory=list)
    t_ps: float = 0.0
    step: int = 0
    events_bonds_formed: int = 0
    events_bonds_broken: int = 0
    # Cached neighbor list (iu, ju). Rebuilt by the integrator when stale.
    _neighbor_iu: Optional[np.ndarray] = None
    _neighbor_ju: Optional[np.ndarray] = None
    _neighbor_built_at_step: int = -1


# ---------- array views onto atom state -------------------------------


def _gather_positions(atoms: Sequence[AtomUnit]) -> np.ndarray:
    # np.array over a list of 3-lists is faster than per-atom indexed writes.
    return np.array([a.position for a in atoms], dtype=np.float64)


def _gather_velocities(atoms: Sequence[AtomUnit]) -> np.ndarray:
    return np.array([a.velocity for a in atoms], dtype=np.float64)


def _scatter_positions(atoms: Sequence[AtomUnit], pos: np.ndarray) -> None:
    # tolist() turns numpy rows into cheap Python lists in one C call.
    rows = pos.tolist()
    for i, a in enumerate(atoms):
        p = rows[i]
        a.position[0] = p[0]
        a.position[1] = p[1]
        a.position[2] = p[2]


def _scatter_velocities(atoms: Sequence[AtomUnit], vel: np.ndarray) -> None:
    rows = vel.tolist()
    for i, a in enumerate(atoms):
        v = rows[i]
        a.velocity[0] = v[0]
        a.velocity[1] = v[1]
        a.velocity[2] = v[2]


def _gather_masses(atoms: Sequence[AtomUnit]) -> np.ndarray:
    return np.array([a.mass_da for a in atoms], dtype=np.float64)


# ---------- diagnostics -----------------------------------------------


def _kinetic_energy_kj_per_mol(atoms: Sequence[AtomUnit]) -> float:
    e = 0.0
    for a in atoms:
        vx, vy, vz = a.velocity
        e += 0.5 * a.mass_da * (vx * vx + vy * vy + vz * vz)
    return e


def current_temperature_K(atoms: Sequence[AtomUnit]) -> float:
    """T = (2 KE) / (3 N k_B), KE in kJ/mol, k_B = 0.00831 kJ/(mol·K)."""
    n = len(atoms)
    if n == 0:
        return 0.0
    ke = _kinetic_energy_kj_per_mol(atoms)
    return (2.0 * ke) / (3.0 * n * 0.00831446)


# ---------- bond event kernels ----------------------------------------


def _maybe_break_bonds(state: SimState, break_fraction: float) -> int:
    broken = 0
    for bond in list(state.bonds):
        if bond.death_time_ps is not None:
            continue
        dx = bond.b.position[0] - bond.a.position[0]
        dy = bond.b.position[1] - bond.a.position[1]
        dz = bond.b.position[2] - bond.a.position[2]
        r = (dx * dx + dy * dy + dz * dz) ** 0.5
        if r > 0.0 and r > break_fraction * bond.equilibrium_length_nm:
            bond.a.break_bond(bond, state.t_ps, reason=f"overstretched r={r:.3f}nm")
            broken += 1
            state.events_bonds_broken += 1
    state.bonds = [b for b in state.bonds if b.death_time_ps is None]
    return broken


def _maybe_form_bonds(
    state: SimState,
    cfg: IntegratorConfig,
    neighbor_pairs: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> int:
    """Form bonds between unbonded pairs within form_distance.

    Uses the cached neighbor list (if any) to restrict the candidate set;
    otherwise falls back to the full upper triangle.
    """
    atoms = state.atoms
    n = len(atoms)
    if n < 2:
        return 0
    formed = 0
    bonded_ids = {
        frozenset((b.a.atom_id, b.b.atom_id))
        for b in state.bonds if b.death_time_ps is None
    }
    r_max = cfg.bond_form_distance_nm
    r_max2 = r_max * r_max

    if neighbor_pairs is not None:
        iu, ju = neighbor_pairs
    else:
        iu, ju = np.triu_indices(n, k=1)

    if iu.size == 0:
        return 0

    pos = _gather_positions(atoms)
    d = pos[ju] - pos[iu]
    r2 = np.einsum("ij,ij->i", d, d)
    close = r2 < r_max2
    if not close.any():
        return 0
    iu = iu[close]
    ju = ju[close]

    # Loop only over the surviving candidates — usually a tiny subset.
    for i, j in zip(iu.tolist(), ju.tolist()):
        ai = atoms[i]
        aj = atoms[j]
        if ai.valence_remaining <= 0 or aj.valence_remaining <= 0:
            continue
        if frozenset((ai.atom_id, aj.atom_id)) in bonded_ids:
            continue
        if not pair_is_bondable(ai.element, aj.element):
            continue
        bond = ai.form_bond(
            aj,
            kind=cfg.bond_form_kind,
            t_ps=state.t_ps,
            equilibrium_length_nm=cfg.bond_form_r0_nm,
            spring_constant_kj_per_nm2=cfg.bond_form_k_kj_per_nm2,
        )
        state.bonds.append(bond)
        bonded_ids.add(frozenset((ai.atom_id, aj.atom_id)))
        formed += 1
        state.events_bonds_formed += 1
    return formed


# ---------- velocity-Verlet ------------------------------------------


def step(
    state: SimState,
    ff_cfg: ForceFieldConfig,
    int_cfg: IntegratorConfig,
    forces_prev: Optional[np.ndarray] = None,
) -> np.ndarray:
    """One velocity-Verlet step. Returns the new forces array so the caller
    can pass it in as `forces_prev` on the next step."""
    dt = int_cfg.dt_ps
    atoms = state.atoms

    def _neighbors() -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not ff_cfg.use_neighbor_list:
            return None
        every = max(1, int_cfg.neighbor_rebuild_every)
        stale = (
            state._neighbor_iu is None
            or (state.step - state._neighbor_built_at_step) >= every
            or state._neighbor_built_at_step < 0
        )
        if stale:
            pos_arr = _gather_positions(atoms)
            cutoff_with_skin = ff_cfg.lj_cutoff_nm + ff_cfg.neighbor_skin_nm
            state._neighbor_iu, state._neighbor_ju = build_neighbor_list(
                pos_arr, cutoff_with_skin
            )
            state._neighbor_built_at_step = state.step
        return (state._neighbor_iu, state._neighbor_ju)

    if forces_prev is None:
        forces_prev = compute_forces(atoms, state.bonds, state.t_ps, ff_cfg,
                                     neighbor_pairs=_neighbors())

    masses = _gather_masses(atoms)[:, None]              # (N, 1)
    vel = _gather_velocities(atoms)
    pos = _gather_positions(atoms)

    # Half-step velocity + full position update.
    vel += 0.5 * dt * forces_prev / masses
    pos += dt * vel
    _scatter_positions(atoms, pos)

    state.t_ps += dt
    state.step += 1

    forces_new = compute_forces(atoms, state.bonds, state.t_ps, ff_cfg,
                                 neighbor_pairs=_neighbors())

    # Second half-step velocity.
    vel += 0.5 * dt * forces_new / masses

    # Berendsen thermostat (scale all velocities).
    t_now = _temperature_from_arrays(vel, masses[:, 0])
    if t_now <= 0.0:
        t_now = 1.0
    tau = max(int_cfg.thermostat_tau_ps, 1e-3)
    lam = (1.0 + (dt / tau) * (int_cfg.target_temperature_K / t_now - 1.0)) ** 0.5
    vel *= lam
    _scatter_velocities(atoms, vel)

    broken = _maybe_break_bonds(state, int_cfg.bond_break_fraction)
    formed = 0
    if int_cfg.dynamic_bonding:
        formed = _maybe_form_bonds(state, int_cfg, neighbor_pairs=_neighbors())

    if int_cfg.step_callback is not None:
        int_cfg.step_callback(state.t_ps, formed, broken)

    return forces_new


def _temperature_from_arrays(vel: np.ndarray, masses: np.ndarray) -> float:
    ke = 0.5 * float(np.sum(masses * np.sum(vel * vel, axis=1)))
    n = vel.shape[0]
    if n == 0:
        return 0.0
    return (2.0 * ke) / (3.0 * n * 0.00831446)


def run(
    state: SimState,
    n_steps: int,
    ff_cfg: ForceFieldConfig,
    int_cfg: IntegratorConfig,
) -> SimState:
    forces = None
    for _ in range(n_steps):
        forces = step(state, ff_cfg, int_cfg, forces)
    return state
