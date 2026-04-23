"""Water + peptide biophysics demo.

Exercises the new physics (Coulomb + angles + Langevin thermostat)
on biologically relevant setups:

  1. ``run_water_box``: TIP3P-ish water at 300 K. Checks that the
     O-H bonds stay near 0.096 nm and the H-O-H angle stays near
     104.5 deg (the template's equilibrium). Reports the mean O-O
     nearest-neighbour distance (~0.28 nm in real water) as a
     simple sanity check for hydrogen-bond-like structure.

  2. ``run_glycine_in_water``: one glycine zwitterion immersed in
     a water shell at 300 K. Checks that glycine's backbone
     geometry (N-Cα-C angle, Cα-C-O angle) survives immersion +
     Langevin thermostat for a few ps. Reports the final bond
     lengths and angle errors.

Both demos use:
  - ``thermostat="langevin"`` — correct canonical ensemble
  - ``use_coulomb=True`` — partial charges on water/glycine
  - angles supplied by the templates via ``build_mixture``
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .atom_unit import AngleBond
from .force_field import ForceFieldConfig
from .integrator import IntegratorConfig, SimState, current_temperature_K, step
from .molecule_builder import build_mixture


@dataclass
class WaterBoxConfig:
    n_water: int = 30
    radius_nm: float = 1.3
    temperature_K: float = 300.0
    dt_ps: float = 0.0002                  # 0.2 fs — required for the stiff
                                            # 8000 kJ/mol/rad^2 H-O-H bend
    steps: int = 10_000                    # 2 ps of simulated time
    report_every: int = 1000
    langevin_gamma_inv_ps: float = 5.0
    min_center_separation_nm: float = 0.45
    seed: int = 42


@dataclass
class WaterBoxResult:
    t_ps: list[float] = field(default_factory=list)
    temperature_K: list[float] = field(default_factory=list)
    mean_oh_nm: list[float] = field(default_factory=list)
    mean_hoh_deg: list[float] = field(default_factory=list)
    mean_nearest_oo_nm: list[float] = field(default_factory=list)
    hbonds_per_water: list[float] = field(default_factory=list)
    n_atoms: int = 0
    elapsed_s: float = 0.0


def _o_h_bond_stats(atoms, bonds) -> tuple[float, int]:
    """Mean O-H bond length and count."""
    from .element import Element
    rs = []
    for b in bonds:
        if b.death_time_ps is not None:
            continue
        if {b.a.element, b.b.element} != {Element.O, Element.H}:
            continue
        d = np.array(b.b.position) - np.array(b.a.position)
        rs.append(float(np.linalg.norm(d)))
    return (float(np.mean(rs)) if rs else 0.0, len(rs))


def _hoh_angle_stats(atoms, angles) -> tuple[float, int]:
    """Mean H-O-H angle in degrees and count."""
    from .element import Element
    thetas = []
    for ang in angles:
        if ang.j.element is not Element.O:
            continue
        r1 = np.array(ang.i.position) - np.array(ang.j.position)
        r2 = np.array(ang.k.position) - np.array(ang.j.position)
        n1 = np.linalg.norm(r1)
        n2 = np.linalg.norm(r2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        c = np.clip(float(r1 @ r2) / (n1 * n2), -1.0, 1.0)
        thetas.append(math.degrees(math.acos(c)))
    return (float(np.mean(thetas)) if thetas else 0.0, len(thetas))


def _mean_nearest_oo(atoms) -> float:
    """Mean nearest-neighbour O-O distance. ~0.28 nm in liquid water."""
    from .element import Element
    ox = np.array([a.position for a in atoms if a.element is Element.O])
    if len(ox) < 2:
        return 0.0
    d = ox[None, :, :] - ox[:, None, :]
    r2 = (d * d).sum(-1)
    np.fill_diagonal(r2, np.inf)
    nearest2 = r2.min(axis=1)
    return float(np.mean(np.sqrt(nearest2)))


def count_hbonds(
    atoms,
    bonds,
    r_cutoff_nm: float = 0.35,
    angle_cutoff_deg: float = 30.0,
) -> tuple[int, float]:
    """Count water-water hydrogen bonds.

    A hydrogen bond is defined here as an O-H...O geometry where:
      - donor hydrogen and acceptor oxygen are on DIFFERENT water
        molecules (different parent_molecule tags);
      - the donor-O to acceptor-O distance < r_cutoff_nm (0.35 nm);
      - the O-H...O angle deviates from 180 deg by <=
        angle_cutoff_deg.

    Returns ``(n_hbonds, hbonds_per_water)``. Real liquid water sits
    around 3.5 H-bonds per water; at our dilute box density and short
    runs we expect something lower.
    """
    from .element import Element
    # Index donors (H atoms bonded to an O, grouped by molecule tag) and
    # acceptors (all water O atoms).
    id_to = {id(a): i for i, a in enumerate(atoms)}
    o_atoms = [(i, a) for i, a in enumerate(atoms) if a.element is Element.O]
    # For each H atom bonded to an O, record (h_idx, its O partner).
    h_donors: list[tuple[int, int]] = []
    for bond in bonds:
        if bond.death_time_ps is not None:
            continue
        a, b = bond.a, bond.b
        if {a.element, b.element} != {Element.O, Element.H}:
            continue
        if a.element is Element.H:
            h, o = a, b
        else:
            h, o = b, a
        if not h.parent_molecule.startswith(("water", "H2O")):
            continue
        h_idx = id_to[id(h)]
        o_idx = id_to[id(o)]
        h_donors.append((h_idx, o_idx))
    if not h_donors:
        return 0, 0.0
    n_water = max(1, len({atoms[o_idx].parent_molecule
                          for _, o_idx in h_donors}))
    cos_min = math.cos(math.radians(180.0 - angle_cutoff_deg))
    r_cut2 = r_cutoff_nm * r_cutoff_nm
    n_hb = 0
    for h_idx, o_donor in h_donors:
        h_pos = np.array(atoms[h_idx].position)
        od_pos = np.array(atoms[o_donor].position)
        for acc_idx, acc_atom in o_atoms:
            if acc_atom.parent_molecule == atoms[h_idx].parent_molecule:
                continue
            a_pos = np.array(acc_atom.position)
            r_oo = a_pos - od_pos
            if r_oo @ r_oo > r_cut2:
                continue
            v1 = h_pos - od_pos
            v2 = a_pos - h_pos
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            # Angle at H: donor-O -> H -> acceptor. Close to 180 for a
            # real H-bond (H points at the acceptor).
            cos_ohd = (v1 @ v2) / (n1 * n2)
            if cos_ohd >= cos_min:      # close enough to 180 deg
                n_hb += 1
                break                   # each donor H counts once
    return n_hb, n_hb / n_water


def run_water_box(
    cfg: WaterBoxConfig,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[SimState, WaterBoxResult]:
    import time
    atoms, bonds, angles, dihedrals = build_mixture(
        {"H2O": cfg.n_water},
        radius_nm=cfg.radius_nm,
        temperature_K=cfg.temperature_K,
        seed=cfg.seed,
        min_center_separation_nm=cfg.min_center_separation_nm,
    )
    state = SimState(atoms=atoms, bonds=bonds, angles=angles,
                     dihedrals=dihedrals)
    result = WaterBoxResult(n_atoms=len(atoms))

    ff = ForceFieldConfig(
        lj_cutoff_nm=1.0, use_confinement=True,
        confinement_radius_nm=cfg.radius_nm,
        use_neighbor_list=True,
        use_coulomb=True,
    )
    int_cfg = IntegratorConfig(
        dt_ps=cfg.dt_ps,
        target_temperature_K=cfg.temperature_K,
        thermostat="langevin",
        langevin_gamma_inv_ps=cfg.langevin_gamma_inv_ps,
    )

    if progress is not None:
        progress(f"water box: {cfg.n_water} waters ({len(atoms)} atoms), "
                 f"T={cfg.temperature_K:.0f} K, "
                 f"Langevin gamma={cfg.langevin_gamma_inv_ps} /ps, "
                 f"dt={cfg.dt_ps} ps, {cfg.steps} steps")
    t0 = time.time()
    forces = None
    for k in range(cfg.steps):
        forces = step(state, ff, int_cfg, forces)
        if (k + 1) % cfg.report_every == 0 or k == cfg.steps - 1:
            T = current_temperature_K(state.atoms)
            mean_oh, _ = _o_h_bond_stats(state.atoms, state.bonds)
            mean_hoh, _ = _hoh_angle_stats(state.atoms, state.angles)
            mean_oo = _mean_nearest_oo(state.atoms)
            _, hb_per = count_hbonds(state.atoms, state.bonds)
            result.t_ps.append(state.t_ps)
            result.temperature_K.append(T)
            result.mean_oh_nm.append(mean_oh)
            result.mean_hoh_deg.append(mean_hoh)
            result.mean_nearest_oo_nm.append(mean_oo)
            result.hbonds_per_water.append(hb_per)
            if progress is not None:
                progress(f"step {k+1}/{cfg.steps} t={state.t_ps:.2f} ps "
                         f"T={T:.0f} K  OH={mean_oh:.3f} nm  "
                         f"HOH={mean_hoh:.1f} deg  <d_OO>={mean_oo:.3f} nm  "
                         f"HB/water={hb_per:.2f}")
    result.elapsed_s = time.time() - t0
    return state, result


@dataclass
class GlycineInWaterConfig:
    n_water: int = 40
    n_glycine: int = 1
    radius_nm: float = 1.8
    temperature_K: float = 300.0
    dt_ps: float = 0.0002                  # 0.2 fs for stiff angles
    steps: int = 10_000                    # 2 ps simulated
    report_every: int = 1000
    langevin_gamma_inv_ps: float = 5.0
    min_center_separation_nm: float = 0.55
    seed: int = 42


@dataclass
class GlycineResult:
    t_ps: list[float] = field(default_factory=list)
    temperature_K: list[float] = field(default_factory=list)
    n_cal_c_nm: list[float] = field(default_factory=list)   # N-Ca and Ca-C bond
    ncac_deg: list[float] = field(default_factory=list)     # N-Ca-C backbone angle
    n_atoms: int = 0
    n_broken_bonds: int = 0
    elapsed_s: float = 0.0


def _glycine_backbone_stats(atoms, bonds, angles) -> tuple[float, float]:
    """Return (mean N-Cα + Cα-C bond length, N-Cα-C angle in degrees)
    across all glycine residues present."""
    from .element import Element
    # Find bonds on glycine-tagged atoms.
    nca_rs = []
    cac_rs = []
    ncac = []
    for b in bonds:
        if b.death_time_ps is not None:
            continue
        if not b.a.parent_molecule.startswith("glycine"):
            continue
        pair = {b.a.element, b.b.element}
        if pair == {Element.N, Element.C}:
            d = np.array(b.b.position) - np.array(b.a.position)
            nca_rs.append(float(np.linalg.norm(d)))
        elif pair == {Element.C}:
            d = np.array(b.b.position) - np.array(b.a.position)
            cac_rs.append(float(np.linalg.norm(d)))
    bond_lens = nca_rs + cac_rs
    for ang in angles:
        if ang.j.parent_molecule.startswith("glycine") \
           and ang.j.element is Element.C \
           and {ang.i.element, ang.k.element} == {Element.N}:
            # this isn't the right angle — skip
            continue
        # Find the "N-Cα-C" angle: vertex Cα, neighbours N and C (heavy).
        if ang.j.element is Element.C \
           and ang.i.element is Element.N \
           and ang.k.element is Element.C:
            r1 = np.array(ang.i.position) - np.array(ang.j.position)
            r2 = np.array(ang.k.position) - np.array(ang.j.position)
            n1 = np.linalg.norm(r1)
            n2 = np.linalg.norm(r2)
            if n1 > 1e-6 and n2 > 1e-6:
                c = np.clip(float(r1 @ r2) / (n1 * n2), -1.0, 1.0)
                ncac.append(math.degrees(math.acos(c)))
    mean_bond = float(np.mean(bond_lens)) if bond_lens else 0.0
    mean_ang = float(np.mean(ncac)) if ncac else 0.0
    return mean_bond, mean_ang


def run_glycine_in_water(
    cfg: GlycineInWaterConfig,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[SimState, GlycineResult]:
    import time
    atoms, bonds, angles, dihedrals = build_mixture(
        {"glycine": cfg.n_glycine, "H2O": cfg.n_water},
        radius_nm=cfg.radius_nm,
        temperature_K=cfg.temperature_K,
        seed=cfg.seed,
        min_center_separation_nm=cfg.min_center_separation_nm,
    )
    state = SimState(atoms=atoms, bonds=bonds, angles=angles,
                     dihedrals=dihedrals)
    result = GlycineResult(n_atoms=len(atoms))

    ff = ForceFieldConfig(
        lj_cutoff_nm=1.0, use_confinement=True,
        confinement_radius_nm=cfg.radius_nm,
        use_neighbor_list=True,
        use_coulomb=True,
    )
    int_cfg = IntegratorConfig(
        dt_ps=cfg.dt_ps,
        target_temperature_K=cfg.temperature_K,
        thermostat="langevin",
        langevin_gamma_inv_ps=cfg.langevin_gamma_inv_ps,
        bond_break_fraction=2.5,     # be generous; don't want spontaneous
                                      # bond breaks in a stable biomolecule
    )

    if progress is not None:
        progress(f"glycine-in-water: {cfg.n_glycine} Gly + {cfg.n_water} water "
                 f"({len(atoms)} atoms, {len(bonds)} bonds, "
                 f"{len(angles)} angles)")
    t0 = time.time()
    initial_bonds = len(state.bonds)
    forces = None
    for k in range(cfg.steps):
        forces = step(state, ff, int_cfg, forces)
        if (k + 1) % cfg.report_every == 0 or k == cfg.steps - 1:
            T = current_temperature_K(state.atoms)
            mean_bond, mean_ang = _glycine_backbone_stats(
                state.atoms, state.bonds, state.angles
            )
            result.t_ps.append(state.t_ps)
            result.temperature_K.append(T)
            result.n_cal_c_nm.append(mean_bond)
            result.ncac_deg.append(mean_ang)
            if progress is not None:
                progress(f"step {k+1}/{cfg.steps} t={state.t_ps:.2f} ps "
                         f"T={T:.0f} K  bb_bond={mean_bond:.3f} nm  "
                         f"N-Ca-C={mean_ang:.1f} deg  "
                         f"bonds={len(state.bonds)}")
    result.n_broken_bonds = initial_bonds - len(state.bonds)
    result.elapsed_s = time.time() - t0
    return state, result
