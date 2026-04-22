"""Minimal force field for the AtomUnit MD toy.

NumPy-vectorized. Covers:
  - Harmonic bond potential: U(r) = 0.5 * k * (r - r0)^2
  - Lennard-Jones 12-6 non-bonded: U(r) = 4 * eps * ((s/r)^12 - (s/r)^6)
  - Optional radial constriction force (used by the fission demo)

Units: lengths in nm, energies in kJ/mol, times in ps, masses in Da. With
these, acceleration is `force_kj_per_mol_nm / mass_da` in nm/ps^2 (1 Da =
1 g/mol; 1 kJ/mol / (g/mol) / nm = 1 nm/ps^2 exactly, which is why MD
codes like these units).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .atom_unit import AtomUnit, Bond
from .element import Element, props

# ---------- LJ special-pair modifiers ---------------------------------

# Hydrophobic tails like each other more than LB says; tails avoid solvent.
# A small boost helps the toy bilayer hold together under the constriction
# driving force in the fission demo.
_TAIL_TAIL_EPS_BOOST = 1.4
_TAIL_WATER_EPS_PENALTY = 0.1
_HEAD_TAIL_EPS_FACTOR = 0.3


@dataclass
class ForceFieldConfig:
    lj_cutoff_nm: float = 1.2
    kT_kj_per_mol: float = 2.49                 # 300 K in kJ/mol
    # --- fission driver: radial constriction along axis ---
    use_constriction: bool = False
    constriction_axis: int = 2                  # 0=x, 1=y, 2=z
    constriction_width_nm: float = 2.0          # Gaussian half-width along axis
    constriction_k_kj_per_nm2: float = 50.0     # radial spring strength
    constriction_ramp_ps: float = 100.0         # ramp k from 0 at t=0
    # --- fusion driver: directional axial attractor toward the midplane ---
    use_axial_attractor: bool = False
    axial_attractor_axis: int = 2               # usually same axis
    axial_attractor_strength_kj_per_nm: float = 0.1   # constant per-atom pull
    axial_attractor_ramp_ps: float = 20.0       # ramp strength from 0 at t=0
    # --- soft spherical confinement (keeps an atom soup from flying apart) ---
    use_confinement: bool = False
    confinement_radius_nm: float = 2.0          # walls start here
    confinement_k_kj_per_nm2: float = 2.0e3     # wall spring constant
    # --- safety ---
    max_force_kj_per_nm: float = 2.0e4          # per-atom force cap


# ---------- Legacy scalar helpers (kept for reference / tests) --------


def lj_params_for(a: AtomUnit, b: AtomUnit) -> tuple[float, float]:
    """Lorentz-Berthelot combining rules. Returns (sigma, epsilon)."""
    pa = props(a.element)
    pb = props(b.element)
    sigma = 0.5 * (pa.lj_sigma_nm + pb.lj_sigma_nm)
    eps = math.sqrt(max(pa.lj_epsilon_kj, 0.0) * max(pb.lj_epsilon_kj, 0.0))
    # Apply same element-pair modifiers so scalar and vector paths agree.
    if a.element is Element.COARSE_TAIL and b.element is Element.COARSE_TAIL:
        eps *= _TAIL_TAIL_EPS_BOOST
    if {a.element, b.element} == {Element.COARSE_TAIL, Element.COARSE_SOLVENT}:
        eps *= _TAIL_WATER_EPS_PENALTY
    if {a.element, b.element} == {Element.COARSE_HEAD, Element.COARSE_TAIL}:
        eps *= _HEAD_TAIL_EPS_FACTOR
    return sigma, eps


def _bond_force_mag(r: float, bond: Bond) -> float:
    """Harmonic bond: F = -k (r - r0). Positive = inward (r > r0)."""
    return -bond.spring_constant_kj_per_nm2 * (r - bond.equilibrium_length_nm)


# ---------- Per-element property arrays (cached per call) -------------


def _element_arrays(atoms: Sequence[AtomUnit]):
    n = len(atoms)
    sigmas = np.empty(n, dtype=np.float64)
    epsilons = np.empty(n, dtype=np.float64)
    elem_codes = np.empty(n, dtype=np.int32)
    for i, a in enumerate(atoms):
        p = props(a.element)
        sigmas[i] = p.lj_sigma_nm
        epsilons[i] = max(p.lj_epsilon_kj, 0.0)
        elem_codes[i] = a.element.value
    return sigmas, epsilons, elem_codes


# ---------- Main force kernel -----------------------------------------


def compute_forces(
    atoms: Sequence[AtomUnit],
    bonds: Iterable[Bond],
    t_ps: float,
    cfg: ForceFieldConfig,
    neighbor_pairs: Iterable[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Compute total force on each atom. Returns (N, 3) array, kJ/mol/nm.

    LJ is evaluated fully vectorized across the upper triangle. For N up to
    ~3000 this is ~10-30 ms per step on a laptop — fast enough for a
    fission demo.
    """
    n = len(atoms)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pos = np.empty((n, 3), dtype=np.float64)
    for i, a in enumerate(atoms):
        pos[i, 0] = a.position[0]
        pos[i, 1] = a.position[1]
        pos[i, 2] = a.position[2]

    forces = np.zeros((n, 3), dtype=np.float64)

    # Stable index for each atom by id(). Fast lookup for bonded pairs.
    id_to_idx = {id(a): i for i, a in enumerate(atoms)}

    # --- bonded forces (sparse) ---
    bond_mask = np.zeros((n, n), dtype=bool)
    live_bonds = [b for b in bonds if b.death_time_ps is None]
    for bond in live_bonds:
        i = id_to_idx.get(id(bond.a))
        j = id_to_idx.get(id(bond.b))
        if i is None or j is None:
            continue
        d = pos[j] - pos[i]
        r = float(np.linalg.norm(d))
        if r < 1e-6:
            continue
        mag = _bond_force_mag(r, bond)    # + pulls j toward i
        f = (mag / r) * d
        forces[i] -= f
        forces[j] += f
        bond_mask[i, j] = True
        bond_mask[j, i] = True

    # --- vectorized non-bonded LJ ---
    sigmas, epsilons, elem_codes = _element_arrays(atoms)

    # All pairs (upper triangle)
    iu, ju = np.triu_indices(n, k=1)

    # Vector from i to j
    dvec = pos[ju] - pos[iu]                               # (M, 3)
    r2 = np.einsum("ij,ij->i", dvec, dvec)                 # (M,)

    cutoff = cfg.lj_cutoff_nm
    cutoff2 = cutoff * cutoff

    keep = (r2 > 1e-8) & (r2 < cutoff2) & ~bond_mask[iu, ju]
    if neighbor_pairs is not None:
        pair_set = {(min(i, j), max(i, j)) for i, j in neighbor_pairs}
        nb_keep = np.array([(i, j) in pair_set for i, j in zip(iu, ju)], dtype=bool)
        keep &= nb_keep

    iu = iu[keep]
    ju = ju[keep]
    if iu.size:
        dvec = dvec[keep]
        r2 = r2[keep]
        r = np.sqrt(r2)

        sig = 0.5 * (sigmas[iu] + sigmas[ju])
        eps = np.sqrt(epsilons[iu] * epsilons[ju])

        # Element-pair modifiers (vectorized)
        TAIL = Element.COARSE_TAIL.value
        SOLV = Element.COARSE_SOLVENT.value
        HEAD = Element.COARSE_HEAD.value
        ei = elem_codes[iu]
        ej = elem_codes[ju]
        m_tt = (ei == TAIL) & (ej == TAIL)
        m_ts = ((ei == TAIL) & (ej == SOLV)) | ((ei == SOLV) & (ej == TAIL))
        m_ht = ((ei == HEAD) & (ej == TAIL)) | ((ei == TAIL) & (ej == HEAD))
        eps = np.where(m_tt, eps * _TAIL_TAIL_EPS_BOOST, eps)
        eps = np.where(m_ts, eps * _TAIL_WATER_EPS_PENALTY, eps)
        eps = np.where(m_ht, eps * _HEAD_TAIL_EPS_FACTOR, eps)

        sr = sig / r
        sr6 = sr ** 6
        sr12 = sr6 * sr6
        mag = 24.0 * eps * (2.0 * sr12 - sr6) / r          # + outward (repulsive)

        fvec = (mag / r)[:, None] * dvec                   # force on j
        np.add.at(forces, ju, fvec)
        np.add.at(forces, iu, -fvec)

    # --- optional axial attractor (fusion driver) ---
    if cfg.use_axial_attractor:
        ax = cfg.axial_attractor_axis
        ramp = min(1.0, t_ps / max(cfg.axial_attractor_ramp_ps, 1e-6))
        strength = cfg.axial_attractor_strength_kj_per_nm * ramp
        # Force = -strength * sign(z) scaled by tanh(z / 0.5 nm). The tanh
        # has two effects: (1) continuous zero-crossing at z=0, and
        # (2) naturally weakens when |z| is already small (atoms near the
        # midplane are barely pulled), so once the two vesicles have met
        # the field stops compressing the contact zone.
        coord = pos[:, ax]
        forces[:, ax] -= strength * np.tanh(coord / 0.5)

    # --- soft spherical confinement wall ---
    if cfg.use_confinement:
        r = np.linalg.norm(pos, axis=1)
        out = r > cfg.confinement_radius_nm
        if out.any():
            radial_unit = pos[out] / np.maximum(r[out], 1e-9)[:, None]
            excess = (r[out] - cfg.confinement_radius_nm)[:, None]
            forces[out] -= cfg.confinement_k_kj_per_nm2 * excess * radial_unit

    # --- optional radial constriction (fission driver) ---
    if cfg.use_constriction:
        axis = cfg.constriction_axis
        w = cfg.constriction_width_nm
        ramp = min(1.0, t_ps / max(cfg.constriction_ramp_ps, 1e-6))
        k = cfg.constriction_k_kj_per_nm2 * ramp
        z = pos[:, axis]
        gauss = np.exp(-(z * z) / (2.0 * w * w))
        # Apply -k * x * gauss and -k * y * gauss for the two radial axes.
        for ax in (0, 1, 2):
            if ax == axis:
                continue
            forces[:, ax] -= k * pos[:, ax] * gauss

    # --- safety cap ---
    norms = np.linalg.norm(forces, axis=1)
    mask = norms > cfg.max_force_kj_per_nm
    if mask.any():
        forces[mask] *= (cfg.max_force_kj_per_nm / norms[mask])[:, None]

    return forces
