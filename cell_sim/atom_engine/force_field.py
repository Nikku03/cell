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

# ---------- Optional Numba acceleration -------------------------------
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ---------- Optional Rust acceleration --------------------------------
try:
    import cell_sim_rust as _rust
    _HAS_RUST_LJ = hasattr(_rust, "lj_forces")
except ImportError:
    _rust = None
    _HAS_RUST_LJ = False


if _HAS_NUMBA:
    @njit(cache=True)
    def _build_neighbor_list_numba(pos, cutoff):
        """Counting-sort spatial hash, JIT-compiled.

        Two-pass: (1) count atoms per cell + count pairs per atom, then
        (2) allocate exact output arrays + fill. Avoids dynamic lists
        and set lookups that kill Numba throughput.
        """
        n = pos.shape[0]
        cutoff2 = cutoff * cutoff

        # Bounding box.
        min_x = pos[0, 0]
        min_y = pos[0, 1]
        min_z = pos[0, 2]
        max_x = min_x
        max_y = min_y
        max_z = min_z
        for i in range(1, n):
            x = pos[i, 0]
            y = pos[i, 1]
            z = pos[i, 2]
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            if z < min_z: min_z = z
            if x > max_x: max_x = x
            if y > max_y: max_y = y
            if z > max_z: max_z = z
        min_x -= cutoff
        min_y -= cutoff
        min_z -= cutoff

        sx = int((max_x - min_x) / cutoff) + 2
        sy = int((max_y - min_y) / cutoff) + 2
        sz = int((max_z - min_z) / cutoff) + 2
        n_cells = sx * sy * sz

        # Cell index per atom + per-cell count.
        cell_idx = np.empty(n, dtype=np.int64)
        cell_count = np.zeros(n_cells, dtype=np.int64)
        for i in range(n):
            ix = int((pos[i, 0] - min_x) / cutoff)
            iy = int((pos[i, 1] - min_y) / cutoff)
            iz = int((pos[i, 2] - min_z) / cutoff)
            c = (ix * sy + iy) * sz + iz
            cell_idx[i] = c
            cell_count[c] += 1

        # Prefix sum for cell starts.
        cell_start = np.zeros(n_cells + 1, dtype=np.int64)
        for c in range(n_cells):
            cell_start[c + 1] = cell_start[c] + cell_count[c]

        # Counting-sort atoms into cells.
        write_pos = cell_start[:n_cells].copy()
        sorted_atom = np.empty(n, dtype=np.int64)
        for i in range(n):
            c = cell_idx[i]
            sorted_atom[write_pos[c]] = i
            write_pos[c] += 1

        # Pass 1: count pairs.
        n_pairs = 0
        for i in range(n):
            ci = cell_idx[i]
            iz = ci % sz
            rem = ci // sz
            iy = rem % sy
            ix = rem // sy
            pxi = pos[i, 0]
            pyi = pos[i, 1]
            pzi = pos[i, 2]
            for dx in range(-1, 2):
                nx = ix + dx
                if nx < 0 or nx >= sx:
                    continue
                for dy in range(-1, 2):
                    ny = iy + dy
                    if ny < 0 or ny >= sy:
                        continue
                    for dz in range(-1, 2):
                        nz = iz + dz
                        if nz < 0 or nz >= sz:
                            continue
                        c = (nx * sy + ny) * sz + nz
                        for k in range(cell_start[c], cell_start[c + 1]):
                            j = sorted_atom[k]
                            if j <= i:
                                continue
                            dxp = pos[j, 0] - pxi
                            dyp = pos[j, 1] - pyi
                            dzp = pos[j, 2] - pzi
                            if dxp * dxp + dyp * dyp + dzp * dzp < cutoff2:
                                n_pairs += 1

        # Pass 2: fill exact arrays.
        pair_i = np.empty(n_pairs, dtype=np.int64)
        pair_j = np.empty(n_pairs, dtype=np.int64)
        idx = 0
        for i in range(n):
            ci = cell_idx[i]
            iz = ci % sz
            rem = ci // sz
            iy = rem % sy
            ix = rem // sy
            pxi = pos[i, 0]
            pyi = pos[i, 1]
            pzi = pos[i, 2]
            for dx in range(-1, 2):
                nx = ix + dx
                if nx < 0 or nx >= sx:
                    continue
                for dy in range(-1, 2):
                    ny = iy + dy
                    if ny < 0 or ny >= sy:
                        continue
                    for dz in range(-1, 2):
                        nz = iz + dz
                        if nz < 0 or nz >= sz:
                            continue
                        c = (nx * sy + ny) * sz + nz
                        for k in range(cell_start[c], cell_start[c + 1]):
                            j = sorted_atom[k]
                            if j <= i:
                                continue
                            dxp = pos[j, 0] - pxi
                            dyp = pos[j, 1] - pyi
                            dzp = pos[j, 2] - pzi
                            if dxp * dxp + dyp * dyp + dzp * dzp < cutoff2:
                                pair_i[idx] = i
                                pair_j[idx] = j
                                idx += 1
        return pair_i, pair_j

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
    # --- neighbor list (required for N >> a few thousand) ---
    use_neighbor_list: bool = False
    neighbor_skin_nm: float = 0.3               # padding added to cutoff when building
    # --- reactive chemistry mode ---
    # Scale LJ sigma for atoms of reactive elements (H/C/N/O/P/S) by
    # this factor. Default 1.0 = OPLS-style sigma, appropriate when
    # atoms sit at their saturated valence (vesicles, membrane demos).
    # Values < 1.0 shrink the LJ core so atoms can approach within
    # bond-forming distance; required for any dynamic-bonding demo to
    # form new cross-element bonds.
    reactive_sigma_scale: float = 1.0
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
    """Element property arrays (sigma, epsilon, element_code) per atom.

    Expensive at large N (Python loop + dict lookups). Callers that hit
    this in a hot loop should use ``element_arrays_cached`` which skips
    the rebuild when the atom list identity is unchanged.
    """
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


# Module-level cache keyed by (id(atoms), len(atoms)). Element identity
# does not change once an atom exists, so the only cache-bust is an atom
# added to / removed from the list (which changes len or id).
_ELEMENT_CACHE: dict = {}


def element_arrays_cached(atoms: Sequence[AtomUnit]):
    key = (id(atoms), len(atoms))
    cached = _ELEMENT_CACHE.get(key)
    if cached is not None:
        return cached
    result = _element_arrays(atoms)
    _ELEMENT_CACHE[key] = result
    # Cap the cache to avoid unbounded growth when callers create many
    # ephemeral atom lists.
    if len(_ELEMENT_CACHE) > 8:
        _ELEMENT_CACHE.pop(next(iter(_ELEMENT_CACHE)))
    return result


# ---------- Main force kernel -----------------------------------------


def build_neighbor_list(pos: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Spatial-hash neighbor list. Returns (iu, ju) arrays of pair indices
    (i < j) whose positions are within ``cutoff``. Expected O(N) for
    roughly-uniform density.

    Uses a Numba-JIT counting-sort implementation when available; falls
    back to a pure-Python dict-based spatial hash otherwise.
    """
    n = pos.shape[0]
    if n < 2:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    if n < 128:
        iu, ju = np.triu_indices(n, k=1)
        return iu.astype(np.int64), ju.astype(np.int64)

    if _HAS_NUMBA:
        return _build_neighbor_list_numba(np.ascontiguousarray(pos, dtype=np.float64),
                                          float(cutoff))

    # --- pure-Python fallback ---
    cell = cutoff
    min_c = pos.min(axis=0) - cell
    cell_idx = np.floor((pos - min_c) / cell).astype(np.int64)

    grid: dict[tuple, list[int]] = {}
    for i in range(n):
        k = (int(cell_idx[i, 0]), int(cell_idx[i, 1]), int(cell_idx[i, 2]))
        grid.setdefault(k, []).append(i)

    cutoff2 = cutoff * cutoff
    pair_i: list[int] = []
    pair_j: list[int] = []
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1)
               for dy in (-1, 0, 1)
               for dz in (-1, 0, 1)]
    for key, cell_atoms in grid.items():
        for dx, dy, dz in offsets:
            nk = (key[0] + dx, key[1] + dy, key[2] + dz)
            nbrs = grid.get(nk)
            if nbrs is None:
                continue
            for i in cell_atoms:
                for j in nbrs:
                    if j <= i:
                        continue
                    d = pos[j] - pos[i]
                    if d @ d < cutoff2:
                        pair_i.append(i)
                        pair_j.append(j)
    return (np.asarray(pair_i, dtype=np.int64),
            np.asarray(pair_j, dtype=np.int64))


def compute_forces(
    atoms: Sequence[AtomUnit],
    bonds: Iterable[Bond],
    t_ps: float,
    cfg: ForceFieldConfig,
    neighbor_pairs: (Iterable[tuple[int, int]]
                    | tuple[np.ndarray, np.ndarray]
                    | None) = None,
) -> np.ndarray:
    """Compute total force on each atom. Returns (N, 3) array, kJ/mol/nm.

    Two paths:
      - ``neighbor_pairs=None`` + ``cfg.use_neighbor_list=False``: full
        O(N^2) pair enumeration via ``np.triu_indices``. Fast up to a
        few thousand atoms; allocates O(N^2) memory so it is hard-capped.
      - ``neighbor_pairs`` provided OR ``cfg.use_neighbor_list=True``:
        only the given pairs (or pairs found by the internal spatial-hash
        neighbor list) are evaluated. Required for N in the 10 000+ range.
    """
    n = len(atoms)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Batch gather — np.array over a list of 3-lists is ~10x faster than
    # a per-atom indexed write loop at N >> 1000.
    pos = np.array([a.position for a in atoms], dtype=np.float64)
    forces = np.zeros((n, 3), dtype=np.float64)

    # Stable index for each atom by id(). Fast lookup for bonded pairs.
    id_to_idx = {id(a): i for i, a in enumerate(atoms)}

    # --- bonded forces (sparse) — also collect the bonded-pair set for
    # exclusion from the LJ calculation.
    bonded_set: set[tuple[int, int]] = set()
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
        bonded_set.add((min(i, j), max(i, j)))

    # --- vectorized non-bonded LJ ---
    sigmas, epsilons, elem_codes = element_arrays_cached(atoms)
    if cfg.reactive_sigma_scale != 1.0:
        # Reactive elements: H=1, C=6, N=7, O=8, P=15, S=16. Shrink sigma
        # so atoms can approach within bond-forming distance. Coarse
        # pseudo-elements (code >= 100) keep their membrane sigmas.
        reactive_mask = (
            (elem_codes == 1) | (elem_codes == 6) | (elem_codes == 7)
            | (elem_codes == 8) | (elem_codes == 15) | (elem_codes == 16)
        )
        sigmas = np.where(reactive_mask, sigmas * cfg.reactive_sigma_scale, sigmas)

    cutoff = cfg.lj_cutoff_nm
    cutoff2 = cutoff * cutoff

    # Pick the pair source: explicit argument > config flag > full O(N^2).
    if neighbor_pairs is None and cfg.use_neighbor_list:
        iu, ju = build_neighbor_list(pos, cutoff)
    elif neighbor_pairs is None:
        iu, ju = np.triu_indices(n, k=1)
    elif isinstance(neighbor_pairs, tuple) and len(neighbor_pairs) == 2 \
            and isinstance(neighbor_pairs[0], np.ndarray):
        iu, ju = neighbor_pairs          # already array form
    else:
        pairs = list(neighbor_pairs)
        if pairs:
            arr = np.asarray(pairs, dtype=np.int64)
            iu = np.minimum(arr[:, 0], arr[:, 1])
            ju = np.maximum(arr[:, 0], arr[:, 1])
        else:
            iu = np.empty(0, dtype=np.int64)
            ju = np.empty(0, dtype=np.int64)

    if iu.size:
        dvec = pos[ju] - pos[iu]                           # (M, 3)
        r2 = np.einsum("ij,ij->i", dvec, dvec)             # (M,)
        keep = (r2 > 1e-8) & (r2 < cutoff2)
        if bonded_set:
            # Encode bonded pairs as single ints (i * n + j). Vectorized
            # exclusion via np.isin beats a Python set-lookup per pair.
            bonded_codes = np.fromiter(
                (i * n + j for (i, j) in bonded_set),
                dtype=np.int64, count=len(bonded_set),
            )
            pair_codes = iu.astype(np.int64) * n + ju.astype(np.int64)
            keep &= ~np.isin(pair_codes, bonded_codes)
        iu = iu[keep]
        ju = ju[keep]
        dvec = dvec[keep]
        r2 = r2[keep]
    if iu.size:
        max_elem = int(elem_codes.max()) if elem_codes.size else 0
        has_coarse = max_elem >= 100       # any COARSE_* pseudo-element?

        if _HAS_RUST_LJ:
            # Rust kernel handles the whole distance/eps/force math plus
            # the scatter in one tight loop. ~5x faster than numpy +
            # bincount at N=10000.
            forces += _rust.lj_forces(
                np.ascontiguousarray(pos, dtype=np.float64),
                np.ascontiguousarray(iu, dtype=np.int64),
                np.ascontiguousarray(ju, dtype=np.int64),
                np.ascontiguousarray(sigmas, dtype=np.float64),
                np.ascontiguousarray(epsilons, dtype=np.float64),
                np.ascontiguousarray(elem_codes, dtype=np.int32),
                float(cutoff),
                bool(has_coarse),
            )
        else:
            # Pure-NumPy fallback.
            r = np.sqrt(r2)

            sig = 0.5 * (sigmas[iu] + sigmas[ju])
            eps = np.sqrt(epsilons[iu] * epsilons[ju])

            if has_coarse:
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
            mag = 24.0 * eps * (2.0 * sr12 - sr6) / r

            factor = mag / r
            fx = factor * dvec[:, 0]
            fy = factor * dvec[:, 1]
            fz = factor * dvec[:, 2]
            forces[:, 0] += (np.bincount(ju, weights=fx, minlength=n)
                             - np.bincount(iu, weights=fx, minlength=n))
            forces[:, 1] += (np.bincount(ju, weights=fy, minlength=n)
                             - np.bincount(iu, weights=fy, minlength=n))
            forces[:, 2] += (np.bincount(ju, weights=fz, minlength=n)
                             - np.bincount(iu, weights=fz, minlength=n))

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
