"""Toy amphiphilic vesicle generator.

Builds a roughly spherical bilayer of ``N`` lipid-like amphiphiles. Each
lipid is 2 coarse atoms: a COARSE_HEAD and a COARSE_TAIL, connected by a
single bond.

Topology:
  - Outer leaflet: heads at ``r = R``, tails at ``r = R - t/2`` (pointing
    inward).
  - Inner leaflet: heads at ``r = R - t``, tails at ``r = R - t/2``
    (pointing outward).
  - The two tail shells overlap at the midplane → they form the
    hydrophobic core that holds the bilayer together.

Atom positions are sampled on each shell using a Fibonacci-sphere-like
distribution for even coverage. Velocities are drawn from the
Maxwell-Boltzmann distribution at the target temperature.

This is a toy model. The intent is to have just enough structure that the
integrator can replicate membrane fission when we apply a radial
constriction force at the equator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .atom_unit import AtomUnit, Bond, BondType
from .element import Element, mass

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


@dataclass
class VesicleSpec:
    n_per_leaflet: int = 200            # lipids per leaflet (total = 2x this)
    radius_nm: float = 3.0              # outer leaflet radius
    bilayer_thickness_nm: float = 1.0   # head-to-head distance across bilayer
    temperature_K: float = 300.0
    bond_k_kj_per_nm2: float = 5.0e3
    bond_r0_nm: float = 0.45            # head-tail equilibrium
    parent_molecule: str = "vesicle"
    seed: Optional[int] = 42


def _fibonacci_sphere(n: int, radius: float,
                      angular_offset: float = 0.0,
                      latitude_offset: float = 0.0) -> np.ndarray:
    """Return (n, 3) of points roughly uniformly distributed on a sphere.

    ``angular_offset`` rotates the longitude; ``latitude_offset`` (in
    [-1, 1]) shifts the y-sweep. Used to de-register the inner leaflet
    so its tails don't sit exactly on top of the outer leaflet's tails.
    """
    pts = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        y = 1.0 - (2.0 * i + 1.0) / n + latitude_offset / n
        y = max(-1.0, min(1.0, y))
        r_xy = math.sqrt(max(0.0, 1.0 - y * y))
        theta = GOLDEN_ANGLE * i + angular_offset
        x = math.cos(theta) * r_xy
        z = math.sin(theta) * r_xy
        pts[i, 0] = x * radius
        pts[i, 1] = y * radius
        pts[i, 2] = z * radius
    return pts


def _maxwell_boltzmann(mass_da: float, T_K: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Draw a 3D velocity in nm/ps from MB at temperature T.

    sigma_v = sqrt(k_B * T / m). With k_B = 0.00831446 kJ/(mol·K) and
    mass in Da (= g/mol), result comes out in nm/ps.
    """
    k_B = 0.00831446
    sigma = math.sqrt(k_B * T_K / mass_da)
    return rng.normal(0.0, sigma, size=3)


def build_vesicle(spec: VesicleSpec) -> tuple[list[AtomUnit], list[Bond]]:
    """Build a bilayer vesicle centered at the origin.

    Returns (atoms, bonds). Atoms carry head-tail bond pairs. No dynamic
    bonds between lipids — cohesion is purely via LJ tail-tail attraction.
    """
    rng = np.random.default_rng(spec.seed)
    R_out = spec.radius_nm
    t = spec.bilayer_thickness_nm
    R_in = R_out - t
    R_mid = R_out - 0.5 * t              # tail shell radius
    if R_in <= 0.2:
        raise ValueError(f"vesicle too small: R_in={R_in:.2f} nm")

    atoms: list[AtomUnit] = []
    bonds: list[Bond] = []

    def _add_leaflet(n: int, r_head: float, r_tail: float,
                     angular_offset: float = 0.0,
                     latitude_offset: float = 0.0) -> None:
        heads_pos = _fibonacci_sphere(n, r_head,
                                      angular_offset=angular_offset,
                                      latitude_offset=latitude_offset)
        # Tail sits along the radial direction between head and core.
        # Use unit vector from head to origin (or reversed) scaled to r_tail.
        for i in range(n):
            hp = heads_pos[i]
            unit = hp / (np.linalg.norm(hp) + 1e-12)
            tp = unit * r_tail
            head = AtomUnit.create(
                element=Element.COARSE_HEAD,
                position=(float(hp[0]), float(hp[1]), float(hp[2])),
                velocity=tuple(_maxwell_boltzmann(mass(Element.COARSE_HEAD),
                                                  spec.temperature_K, rng)),
                parent_molecule=spec.parent_molecule,
            )
            tail = AtomUnit.create(
                element=Element.COARSE_TAIL,
                position=(float(tp[0]), float(tp[1]), float(tp[2])),
                velocity=tuple(_maxwell_boltzmann(mass(Element.COARSE_TAIL),
                                                  spec.temperature_K, rng)),
                parent_molecule=spec.parent_molecule,
            )
            bond = head.form_bond(
                tail,
                kind=BondType.COARSE_CHAIN,
                t_ps=0.0,
                equilibrium_length_nm=spec.bond_r0_nm,
                spring_constant_kj_per_nm2=spec.bond_k_kj_per_nm2,
            )
            atoms.append(head)
            atoms.append(tail)
            bonds.append(bond)

    _add_leaflet(spec.n_per_leaflet, r_head=R_out, r_tail=R_mid)
    # Stagger the inner leaflet by half a golden-angle step + half a
    # latitude step so its tails interleave with the outer leaflet's at
    # the midplane, rather than sitting on top of them.
    _add_leaflet(spec.n_per_leaflet, r_head=R_in, r_tail=R_mid,
                 angular_offset=0.5 * GOLDEN_ANGLE,
                 latitude_offset=0.5)

    _zero_net_momentum(atoms)
    return atoms, bonds


def _zero_net_momentum(atoms: list[AtomUnit]) -> None:
    """Subtract center-of-mass velocity so the vesicle doesn't drift."""
    total_mass = sum(a.mass_da for a in atoms)
    if total_mass <= 0.0:
        return
    px = sum(a.mass_da * a.velocity[0] for a in atoms) / total_mass
    py = sum(a.mass_da * a.velocity[1] for a in atoms) / total_mass
    pz = sum(a.mass_da * a.velocity[2] for a in atoms) / total_mass
    for a in atoms:
        a.velocity[0] -= px
        a.velocity[1] -= py
        a.velocity[2] -= pz


# ---------- diagnostics -----------------------------------------------


def build_two_vesicles(
    spec: VesicleSpec,
    z_offset_nm: float = 4.0,
) -> tuple[list[AtomUnit], list[Bond]]:
    """Build two identical vesicles centered at z = +offset and z = -offset.

    Returns a combined (atoms, bonds) where each vesicle's atoms carry
    distinct ``parent_molecule`` tags ("vesicle_upper" / "vesicle_lower"),
    which makes it trivial to tell which atom started on which side.
    """
    upper_spec = VesicleSpec(
        n_per_leaflet=spec.n_per_leaflet,
        radius_nm=spec.radius_nm,
        bilayer_thickness_nm=spec.bilayer_thickness_nm,
        temperature_K=spec.temperature_K,
        bond_k_kj_per_nm2=spec.bond_k_kj_per_nm2,
        bond_r0_nm=spec.bond_r0_nm,
        parent_molecule="vesicle_upper",
        seed=spec.seed,
    )
    lower_spec = VesicleSpec(
        n_per_leaflet=spec.n_per_leaflet,
        radius_nm=spec.radius_nm,
        bilayer_thickness_nm=spec.bilayer_thickness_nm,
        temperature_K=spec.temperature_K,
        bond_k_kj_per_nm2=spec.bond_k_kj_per_nm2,
        bond_r0_nm=spec.bond_r0_nm,
        parent_molecule="vesicle_lower",
        seed=(spec.seed + 1) if spec.seed is not None else None,
    )
    upper_atoms, upper_bonds = build_vesicle(upper_spec)
    lower_atoms, lower_bonds = build_vesicle(lower_spec)
    for a in upper_atoms:
        a.position[2] += z_offset_nm
    for a in lower_atoms:
        a.position[2] -= z_offset_nm
    atoms = upper_atoms + lower_atoms
    bonds = upper_bonds + lower_bonds
    return atoms, bonds


def count_connected_components(atoms: list[AtomUnit],
                               bonds: list[Bond],
                               link_cutoff_nm: float = 0.7) -> int:
    """Cluster atoms by (live-bond) + (LJ-proximity) connectivity.

    Two atoms are considered connected if they share a live bond OR are
    within ``link_cutoff_nm`` (a tail-tail cohesion distance). After
    fission the vesicle should split into exactly 2 components.
    """
    n = len(atoms)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    id_to_idx = {id(a): i for i, a in enumerate(atoms)}
    for b in bonds:
        if b.death_time_ps is not None:
            continue
        i = id_to_idx.get(id(b.a))
        j = id_to_idx.get(id(b.b))
        if i is None or j is None:
            continue
        union(i, j)

    # Distance-based linking only between COARSE_TAIL atoms (the cohesive glue).
    tail_idx = [i for i, a in enumerate(atoms) if a.element is Element.COARSE_TAIL]
    if tail_idx:
        pos = np.array([atoms[i].position for i in tail_idx], dtype=np.float64)
        m = len(tail_idx)
        cutoff2 = link_cutoff_nm * link_cutoff_nm
        for i in range(m):
            d = pos[i + 1:] - pos[i]
            r2 = np.einsum("ij,ij->i", d, d)
            near = np.where(r2 < cutoff2)[0]
            for k in near:
                union(tail_idx[i], tail_idx[i + 1 + int(k)])

    roots = {find(i) for i in range(n)}
    return len(roots)


def equatorial_split_metric(atoms: list[AtomUnit], axis: int = 2) -> float:
    """Fraction of atoms in the "neck" slab (|axis| < 0.5 nm), normalized
    by the fraction in an unperturbed sphere.

    Returns ~1.0 before fission, → 0 as the neck closes.
    """
    if not atoms:
        return 0.0
    coords = np.array([a.position[axis] for a in atoms])
    neck = float(np.mean(np.abs(coords) < 0.5))
    # For a uniformly-filled sphere of radius R, the fraction of volume in
    # |z| < h is (3h/2R - h^3/(2R^3)). For h=0.5, R=3 we expect ~0.25.
    return neck / 0.25 if neck > 0.0 else 0.0


def hemisphere_split_balance(atoms: list[AtomUnit], axis: int = 2) -> float:
    """Return |N_plus - N_minus| / N_total. For a symmetric split: 0.
    For all-atoms-on-one-side: 1.
    """
    if not atoms:
        return 0.0
    coords = np.array([a.position[axis] for a in atoms])
    n_plus = int(np.sum(coords > 0.0))
    n_minus = int(np.sum(coords < 0.0))
    return abs(n_plus - n_minus) / max(1, len(atoms))


def is_bimodal_along_axis(atoms: list[AtomUnit], axis: int = 2,
                          neck_halfwidth_nm: float = 0.5) -> bool:
    """Simple bimodality heuristic: the neck slab is near-empty AND each
    hemisphere has a nontrivial fraction of atoms."""
    if not atoms:
        return False
    coords = np.array([a.position[axis] for a in atoms])
    n = len(atoms)
    neck = float(np.mean(np.abs(coords) < neck_halfwidth_nm))
    plus = float(np.mean(coords > neck_halfwidth_nm))
    minus = float(np.mean(coords < -neck_halfwidth_nm))
    return neck < 0.05 and plus > 0.3 and minus > 0.3


def vesicle_com_separation(atoms: list[AtomUnit], axis: int = 2) -> float:
    """Distance between centers of mass of atoms tagged ``vesicle_upper`` vs.
    ``vesicle_lower`` along ``axis``. Zero once they fully interpenetrate.
    """
    upper = [a for a in atoms if a.parent_molecule == "vesicle_upper"]
    lower = [a for a in atoms if a.parent_molecule == "vesicle_lower"]
    if not upper or not lower:
        return 0.0
    u = sum(a.mass_da * a.position[axis] for a in upper) / sum(a.mass_da for a in upper)
    l = sum(a.mass_da * a.position[axis] for a in lower) / sum(a.mass_da for a in lower)
    return abs(u - l)
