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

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .atom_unit import AngleBond, AtomUnit, Bond, BondType, DihedralBond
from .element import pair_is_bondable
from .force_field import (
    ForceFieldConfig,
    build_neighbor_list,
    compute_forces,
    wrap_positions,
)


@dataclass
class IntegratorConfig:
    dt_ps: float = 0.002                     # 2 fs timestep (outer)
    # RESPA multiple-time-stepping (Tuckerman 1992, r-RESPA).
    # When > 1, the integrator splits the force evaluation into:
    #   - FAST: bonded + angle + dihedral (cheap, high-frequency)
    #   - SLOW: LJ + Coulomb + external potentials (expensive, low-freq)
    # Per outer step, SLOW is evaluated twice (half-kicks front and back);
    # FAST is evaluated ``respa_n_inner`` times with inner dt = dt / n.
    # Compound speedup: (n_slow_calls_avoided) / (n_fast_calls_added). For
    # water-heavy boxes where SLOW dominates runtime, n_inner = 4 gives
    # roughly 3x end-to-end with no appreciable loss of stability.
    respa_n_inner: int = 1
    thermostat_tau_ps: float = 1.0
    target_temperature_K: float = 300.0
    dynamic_bonding: bool = False            # whether to form/break bonds
    # Pair must be within ``bond_form_distance_nm`` to be CONSIDERED for
    # bonding (outer gate). The actual form decision then uses a
    # pair-specific check: form iff ``r < bond_form_ratio * r0_pair``.
    bond_form_distance_nm: float = 0.20
    bond_form_ratio: float = 1.3
    bond_break_fraction: float = 1.8          # break if r > bond_break_fraction * r0
    bond_form_kind: BondType = BondType.COVALENT_SINGLE
    bond_form_k_kj_per_nm2: float = 2.0e4
    bond_form_r0_nm: float = 0.15
    # Neighbor-list scheduler: rebuild every N steps when the force field
    # has use_neighbor_list=True. 0 disables (caller manages the list).
    neighbor_rebuild_every: int = 10
    # Essentiality integration: set of (element_a, element_b) frozensets
    # for which dynamic bonding is DISABLED. Used to simulate "knockouts"
    # of specific reaction chemistries without changing the engine.
    disabled_pairs: Optional[frozenset] = None
    # SHAKE bond constraints (Phase 1).
    # When True, every "constrained" bond (see SimState.constrained_bonds)
    # is held at its equilibrium length via one-pass SHAKE after the
    # position update. This eliminates the fast bond-vibration mode
    # and lets dt go from 0.2 fs to 1-2 fs.
    shake: bool = False
    shake_tolerance_nm: float = 1e-4
    shake_max_iter: int = 200
    # Thermostat choice (Physics Upgrade 3).
    #   "berendsen" — velocity rescale toward target T. Fast to equilibrate,
    #                 incorrect canonical sampling (known: "flying ice-cube"
    #                 issue). Original integrator default.
    #   "langevin"  — SDE with friction gamma and Gaussian white noise.
    #                 Correct NVT sampling. Required for condensed-phase
    #                 chemistry where velocity distribution matters.
    thermostat: str = "berendsen"
    langevin_gamma_inv_ps: float = 1.0   # friction coefficient (1/ps)
    # Callback invoked each step with (t_ps, n_bonds_formed, n_bonds_broken).
    step_callback: Optional[callable] = None


@dataclass
class SimState:
    atoms: list[AtomUnit]
    bonds: list[Bond] = field(default_factory=list)
    # 3-body angle terms (Physics Upgrade 2). Populated by molecule
    # templates or by a post-hoc auto-angle pass on the bond graph.
    angles: list[AngleBond] = field(default_factory=list)
    # 4-body proper dihedral terms.
    dihedrals: list[DihedralBond] = field(default_factory=list)
    # SHAKE constraint list: (atom_idx_i, atom_idx_j, r0). Populated
    # once from a bond list via ``build_shake_constraints`` so we don't
    # re-scan the bond graph every step.
    shake_pairs: Optional[np.ndarray] = None      # (Nc, 2) int64
    shake_r0_sq: Optional[np.ndarray] = None      # (Nc,) float64
    t_ps: float = 0.0
    step: int = 0
    events_bonds_formed: int = 0
    events_bonds_broken: int = 0
    # Cumulative bond-formed counts keyed by a canonical element-pair
    # rule name ("bond:C-H", "bond:H-O", ...). Used by the essentiality
    # bridge to populate Sample.event_counts_by_rule in the existing
    # detector framework.
    events_formed_by_rule: dict = field(default_factory=dict)
    events_broken_by_rule: dict = field(default_factory=dict)
    # Cached neighbor list (iu, ju). Rebuilt by the integrator when stale.
    _neighbor_iu: Optional[np.ndarray] = None
    _neighbor_ju: Optional[np.ndarray] = None
    _neighbor_built_at_step: int = -1
    # Cached masses (never change, only rebuilt when atom list changes).
    _masses: Optional[np.ndarray] = None
    _masses_atom_count: int = -1
    # Cached slow (non-bonded) forces from the END of the previous RESPA
    # outer step; reused as the leading half-kick of the next outer step.
    _respa_F_slow: Optional[np.ndarray] = None
    # Cached precompiled arrays for fast compute_forces calls. Populated
    # by ``compile_bond_cache`` and invalidated when the atom/bond graph
    # mutates (bond break/form events).
    _bond_cache: Optional[object] = None
    # SETTLE rigid-water arrays. Populated by ``build_settle_waters``.
    # When present, each water triple (O, H1, H2) is handled by the
    # analytical SETTLE kernel instead of iterative SHAKE.
    _settle_water_idx: Optional[np.ndarray] = None   # (N_w, 3) int64
    _settle_water_r_oh: Optional[np.ndarray] = None  # (N_w,) float64
    _settle_water_r_hh: Optional[np.ndarray] = None  # (N_w,) float64
    # Quaternion rigid-body state. When ``init_rigid_body_state`` has
    # been called, the integrator advances these instead of doing a
    # SHAKE / Kabsch projection. Proper symplectic rotation + Euler's
    # equation preserves energy to machine precision on free rotation.
    _rb_q: Optional[np.ndarray] = None            # (N_w, 4) quaternions
    _rb_v_com: Optional[np.ndarray] = None        # (N_w, 3) COM velocity
    _rb_omega_body: Optional[np.ndarray] = None   # (N_w, 3) angular vel (body)
    _rb_ref: Optional[np.ndarray] = None          # (N_w, 3, 3) body-frame ref
    _rb_I_body: Optional[np.ndarray] = None       # (N_w, 3) body-frame inertia


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


def _cached_masses(state: "SimState") -> np.ndarray:
    """Masses never change once set, so only rebuild when the atom
    count changes. Saves ~10 ms/step at N=10000."""
    n = len(state.atoms)
    if state._masses is None or state._masses_atom_count != n:
        state._masses = _gather_masses(state.atoms)
        state._masses_atom_count = n
    return state._masses


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


@dataclass
class BondCache:
    """Precompiled index + parameter arrays for bonded force evaluation.

    Fully materialised once at setup (via ``compile_bond_cache``) and
    reused on every ``compute_forces`` call, eliminating the per-call
    Python dict rebuild + per-bond id() lookup that dominated profiled
    RESPA runs.
    """
    # Bond arrays (M_bonds,)
    bond_i: np.ndarray
    bond_j: np.ndarray
    bond_k: np.ndarray        # spring constant kJ/(mol*nm^2)
    bond_r0: np.ndarray       # equilibrium length nm
    # Angle arrays (M_angles,)
    angle_i: np.ndarray
    angle_j: np.ndarray
    angle_k: np.ndarray
    angle_theta0: np.ndarray
    angle_kth: np.ndarray     # spring constant kJ/(mol*rad^2)
    # Dihedral arrays (M_dihedrals,)
    dih_i: np.ndarray
    dih_j: np.ndarray
    dih_k: np.ndarray
    dih_l: np.ndarray
    dih_n: np.ndarray
    dih_phi0: np.ndarray
    dih_kphi: np.ndarray
    # Packed bonded-pair codes for LJ/Coulomb exclusion set (sorted).
    # Includes both 1-2 bonds and 1-3 angle endpoints.
    bonded_pair_codes_sorted: np.ndarray
    # Number of atoms at cache-build time (for invalidation).
    n_atoms: int


def compile_bond_cache(state: "SimState") -> BondCache:
    """Precompute index + parameter arrays for every bond, angle, and
    dihedral on ``state``. Stored on ``state._bond_cache`` and reused
    across compute_forces calls until the topology changes.

    Safe to call any time; re-invoking rebuilds from scratch.
    """
    atoms = state.atoms
    n = len(atoms)
    id_to_idx = {id(a): i for i, a in enumerate(atoms)}

    bi, bj, bk, br = [], [], [], []
    for b in state.bonds:
        if b.death_time_ps is not None:
            continue
        i = id_to_idx.get(id(b.a))
        j = id_to_idx.get(id(b.b))
        if i is None or j is None:
            continue
        bi.append(i); bj.append(j)
        bk.append(b.spring_constant_kj_per_nm2)
        br.append(b.equilibrium_length_nm)

    ai, aj, ak = [], [], []
    at0, akth = [], []
    for ang in state.angles or []:
        i = id_to_idx.get(id(ang.i))
        j = id_to_idx.get(id(ang.j))
        k = id_to_idx.get(id(ang.k))
        if i is None or j is None or k is None:
            continue
        ai.append(i); aj.append(j); ak.append(k)
        at0.append(ang.theta_0_rad)
        akth.append(ang.k_theta_kj_per_mol_rad2)

    di, dj, dk, dl, dn, dp0, dkp = [], [], [], [], [], [], []
    for d in state.dihedrals or []:
        i = id_to_idx.get(id(d.i))
        j = id_to_idx.get(id(d.j))
        k = id_to_idx.get(id(d.k))
        l = id_to_idx.get(id(d.l))
        if None in (i, j, k, l):
            continue
        di.append(i); dj.append(j); dk.append(k); dl.append(l)
        dn.append(d.n); dp0.append(d.phi_0_rad); dkp.append(d.k_phi_kj_per_mol)

    # Pair exclusion codes: 1-2 from bonds + 1-3 from angle endpoints
    # + H-H from every SETTLE water (the H-O-H angle was removed but
    # the H pair must still be excluded from LJ / Coulomb).
    pair_codes_set = set()
    for i, j in zip(bi, bj):
        a, b = (i, j) if i < j else (j, i)
        pair_codes_set.add(a * n + b)
    for i, k in zip(ai, ak):
        if i == k:
            continue
        a, b = (i, k) if i < k else (k, i)
        pair_codes_set.add(a * n + b)
    if state._settle_water_idx is not None:
        for w in range(state._settle_water_idx.shape[0]):
            h1 = int(state._settle_water_idx[w, 1])
            h2 = int(state._settle_water_idx[w, 2])
            a, b = (h1, h2) if h1 < h2 else (h2, h1)
            pair_codes_set.add(a * n + b)
    pair_codes_sorted = np.array(sorted(pair_codes_set), dtype=np.int64)

    cache = BondCache(
        bond_i=np.asarray(bi, dtype=np.int64),
        bond_j=np.asarray(bj, dtype=np.int64),
        bond_k=np.asarray(bk, dtype=np.float64),
        bond_r0=np.asarray(br, dtype=np.float64),
        angle_i=np.asarray(ai, dtype=np.int64),
        angle_j=np.asarray(aj, dtype=np.int64),
        angle_k=np.asarray(ak, dtype=np.int64),
        angle_theta0=np.asarray(at0, dtype=np.float64),
        angle_kth=np.asarray(akth, dtype=np.float64),
        dih_i=np.asarray(di, dtype=np.int64),
        dih_j=np.asarray(dj, dtype=np.int64),
        dih_k=np.asarray(dk, dtype=np.int64),
        dih_l=np.asarray(dl, dtype=np.int64),
        dih_n=np.asarray(dn, dtype=np.float64),
        dih_phi0=np.asarray(dp0, dtype=np.float64),
        dih_kphi=np.asarray(dkp, dtype=np.float64),
        bonded_pair_codes_sorted=pair_codes_sorted,
        n_atoms=n,
    )
    state._bond_cache = cache
    return cache


def build_shake_constraints(
    atoms: list[AtomUnit],
    bonds: list[Bond],
    angles: Optional[list[AngleBond]] = None,
    rigid_water: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pairs, r0_sq) for use as SHAKE constraints.

    One entry per live covalent bond by default. When ``angles`` is
    supplied AND ``rigid_water`` is True, H-X-H angle triples
    (primarily H-O-H waters) are ALSO added as third-edge constraints
    on the (H, H) pair.

    The caller is responsible for removing the matching H-X-H angle
    terms from the angle list (or calling ``make_waters_rigid``), since
    a rigid triangle + harmonic angle spring would oscillate. Use
    ``make_waters_rigid(state, ...)`` for the full drop-in setup.

    r_HH from the law of cosines:
        r_HH^2 = r_OH1^2 + r_OH2^2 - 2*r_OH1*r_OH2*cos(theta_0)
    """
    id_to_idx = {id(a): i for i, a in enumerate(atoms)}
    pairs = []
    r0s = []
    for b in bonds:
        if b.death_time_ps is not None:
            continue
        i = id_to_idx.get(id(b.a))
        j = id_to_idx.get(id(b.b))
        if i is None or j is None:
            continue
        pairs.append((i, j))
        r0s.append(float(b.equilibrium_length_nm))

    if angles and rigid_water:
        bond_len_by_pair: dict[tuple[int, int], float] = {}
        for b in bonds:
            if b.death_time_ps is not None:
                continue
            i = id_to_idx.get(id(b.a))
            j = id_to_idx.get(id(b.b))
            if i is None or j is None:
                continue
            key = (min(i, j), max(i, j))
            bond_len_by_pair[key] = float(b.equilibrium_length_nm)

        for ang in angles:
            if ang.i.element.name != "H" or ang.k.element.name != "H":
                continue
            ii = id_to_idx.get(id(ang.i))
            kk = id_to_idx.get(id(ang.k))
            jj = id_to_idx.get(id(ang.j))
            if ii is None or kk is None or jj is None:
                continue
            r_OH1 = bond_len_by_pair.get((min(ii, jj), max(ii, jj)))
            r_OH2 = bond_len_by_pair.get((min(kk, jj), max(kk, jj)))
            if r_OH1 is None or r_OH2 is None:
                continue
            cos_t = math.cos(ang.theta_0_rad)
            r_HH_sq = r_OH1 ** 2 + r_OH2 ** 2 - 2.0 * r_OH1 * r_OH2 * cos_t
            pairs.append((ii, kk))
            r0s.append(math.sqrt(max(r_HH_sq, 0.0)))

    if not pairs:
        return (np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.float64))
    arr = np.array(pairs, dtype=np.int64)
    r0sq = np.array(r0s, dtype=np.float64) ** 2
    return arr, r0sq


def make_waters_rigid(state: "SimState") -> int:
    """One-call setup for rigid-water simulations.

    1. Builds SHAKE constraint list including H-H edges for every
       H-O-H angle triple (``rigid_water=True`` path of
       ``build_shake_constraints``).
    2. Removes the matching harmonic H-O-H angle terms from
       ``state.angles`` so they don't fight the SHAKE constraint.
    3. Invalidates ``state._bond_cache`` so subsequent compute_forces
       calls rebuild from the new angle list.

    Returns the number of water triangles made rigid.

    KNOWN LIMITATION: iterative SHAKE on water triangles injects
    kinetic energy even when bond-length errors are within
    tolerance — the pairwise Lagrange-multiplier updates for the
    three coupled constraints are not fully orthogonal, so a small
    amount of energy leaks in per step. On a dense water box this
    drives T well above the thermostat setpoint. A full 2 fs unlock
    requires SETTLE (analytical rigid-body water, Miyamoto-Kollman
    1992) or RATTLE (mass-weighted velocity correction). This
    function sets up the geometry that SETTLE will consume; the
    SETTLE kernel itself is deferred to a follow-up change.
    """
    shake_pairs, shake_r0_sq = build_shake_constraints(
        state.atoms, state.bonds,
        angles=state.angles, rigid_water=True,
    )
    state.shake_pairs = shake_pairs
    state.shake_r0_sq = shake_r0_sq

    n_rigid = 0
    if state.angles:
        kept = []
        for ang in state.angles:
            if ang.i.element.name == "H" and ang.k.element.name == "H":
                n_rigid += 1
                continue
            kept.append(ang)
        state.angles = kept
    state._bond_cache = None
    return n_rigid


try:
    from numba import njit as _njit
    _HAS_NUMBA_SHAKE = True
except ImportError:
    _HAS_NUMBA_SHAKE = False


if _HAS_NUMBA_SHAKE:
    @_njit(cache=True, fastmath=True)
    def _shake_kernel(
        pos, prev_pos, inv_m, shake_pairs, shake_r0_sq,
        box_l, use_pbc, tol_sq, max_iter,
    ):
        """Numba-JIT SHAKE inner loop. Pure scalar math over the pair
        list, no Python-level numpy calls — about 20x faster than the
        pure-Python implementation at typical water-box sizes."""
        n_pairs = shake_pairs.shape[0]
        for _it in range(max_iter):
            max_err = 0.0
            for p in range(n_pairs):
                i = shake_pairs[p, 0]
                j = shake_pairs[p, 1]
                r0_sq = shake_r0_sq[p]
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                if use_pbc:
                    dx -= box_l * round(dx / box_l)
                    dy -= box_l * round(dy / box_l)
                    dz -= box_l * round(dz / box_l)
                d2 = dx * dx + dy * dy + dz * dz
                diff = d2 - r0_sq
                if abs(diff) < tol_sq:
                    continue
                rdx = prev_pos[j, 0] - prev_pos[i, 0]
                rdy = prev_pos[j, 1] - prev_pos[i, 1]
                rdz = prev_pos[j, 2] - prev_pos[i, 2]
                if use_pbc:
                    rdx -= box_l * round(rdx / box_l)
                    rdy -= box_l * round(rdy / box_l)
                    rdz -= box_l * round(rdz / box_l)
                dot = dx * rdx + dy * rdy + dz * rdz
                if abs(dot) < 1e-12:
                    continue
                inv_mi = inv_m[i]
                inv_mj = inv_m[j]
                lam = diff / (2.0 * dot * (inv_mi + inv_mj))
                pos[i, 0] += inv_mi * lam * rdx
                pos[i, 1] += inv_mi * lam * rdy
                pos[i, 2] += inv_mi * lam * rdz
                pos[j, 0] -= inv_mj * lam * rdx
                pos[j, 1] -= inv_mj * lam * rdy
                pos[j, 2] -= inv_mj * lam * rdz
                ad = abs(diff)
                if ad > max_err:
                    max_err = ad
            if max_err < tol_sq:
                break
        return pos


def _apply_shake(
    pos: np.ndarray,
    prev_pos: np.ndarray,
    masses: np.ndarray,                    # (N,) not (N, 1)
    shake_pairs: np.ndarray,
    shake_r0_sq: np.ndarray,
    box_l: Optional[float] = None,
    tol_nm: float = 1e-4,
    max_iter: int = 200,
) -> np.ndarray:
    """Constrain ``pos`` so every (i, j) pair satisfies
    ``|pos[i] - pos[j]| = r0`` by iteratively projecting along the
    reference (pre-update) bond direction.

    Ryckaert, Ciccotti, Berendsen 1977. Converges in a handful of
    iterations for realistic geometry. When Numba is available the
    inner loop runs as native scalar code (~20x faster than plain
    NumPy+Python); otherwise falls back to a pure-Python loop.
    """
    if shake_pairs.size == 0:
        return pos
    tol_sq = tol_nm * tol_nm
    inv_m = 1.0 / masses
    if _HAS_NUMBA_SHAKE:
        return _shake_kernel(
            pos, prev_pos, inv_m, shake_pairs, shake_r0_sq,
            float(box_l) if box_l is not None else 0.0,
            box_l is not None, tol_sq, max_iter,
        )
    # Pure-Python fallback.
    n_pairs = shake_pairs.shape[0]
    for _iter in range(max_iter):
        max_err = 0.0
        for p in range(n_pairs):
            i, j = int(shake_pairs[p, 0]), int(shake_pairs[p, 1])
            r0_sq = float(shake_r0_sq[p])
            d = pos[j] - pos[i]
            if box_l is not None:
                d = d - box_l * np.round(d / box_l)
            d2 = float(d @ d)
            diff = d2 - r0_sq
            if abs(diff) < tol_sq:
                continue
            d_ref = prev_pos[j] - prev_pos[i]
            if box_l is not None:
                d_ref = d_ref - box_l * np.round(d_ref / box_l)
            dot = float(d @ d_ref)
            if abs(dot) < 1e-12:
                continue
            lam = diff / (2.0 * dot * (inv_m[i] + inv_m[j]))
            delta = lam * d_ref
            pos[i] += inv_m[i] * delta
            pos[j] -= inv_m[j] * delta
            if abs(diff) > max_err:
                max_err = abs(diff)
        if max_err < tol_sq:
            break
    return pos


def _rule_name_for_pair(elem_a, elem_b) -> str:
    """Canonical 'bond:X-Y' rule name for an element pair (sorted)."""
    names = sorted([elem_a.name, elem_b.name])
    return f"bond:{names[0]}-{names[1]}"


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
            rule = _rule_name_for_pair(bond.a.element, bond.b.element)
            bond.a.break_bond(bond, state.t_ps, reason=f"overstretched r={r:.3f}nm")
            broken += 1
            state.events_bonds_broken += 1
            state.events_broken_by_rule[rule] = (
                state.events_broken_by_rule.get(rule, 0) + 1
            )
    state.bonds = [b for b in state.bonds if b.death_time_ps is None]
    if broken:
        state._bond_cache = None
    return broken


def _maybe_form_bonds(
    state: SimState,
    cfg: IntegratorConfig,
    neighbor_pairs: Optional[tuple[np.ndarray, np.ndarray]] = None,
    pos: Optional[np.ndarray] = None,
) -> int:
    """Form bonds between unbonded pairs within form_distance.

    Uses the cached neighbor list (if any) to restrict the candidate set;
    otherwise falls back to the full upper triangle. Callers that already
    have a positions array can pass it in to skip the regather.
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

    if pos is None:
        pos = _gather_positions(atoms)
    d = pos[ju] - pos[iu]
    r2 = np.einsum("ij,ij->i", d, d)
    close = r2 < r_max2
    if not close.any():
        return 0
    iu = iu[close]
    ju = ju[close]
    r = np.sqrt(r2[close])

    # Lazy import to avoid a circular: integrator <- force_field <- atom_unit
    # and molecule_builder also imports atom_unit.
    from .molecule_builder import _bond_length as _bond_length_for_pair

    # Only form a bond if the instantaneous separation is within
    # ``cfg.bond_form_ratio * r0_pair``. Keeping form_ratio < break_fraction
    # by a comfortable margin prevents form/break ping-pong.
    form_ratio = cfg.bond_form_ratio

    disabled = cfg.disabled_pairs
    # Loop only over the surviving candidates — usually a tiny subset.
    for idx, (i_raw, j_raw) in enumerate(zip(iu.tolist(), ju.tolist())):
        ai = atoms[i_raw]
        aj = atoms[j_raw]
        if ai.valence_remaining <= 0 or aj.valence_remaining <= 0:
            continue
        if frozenset((ai.atom_id, aj.atom_id)) in bonded_ids:
            continue
        if not pair_is_bondable(ai.element, aj.element):
            continue
        # Essentiality knockout: this pair is disabled for this run.
        if disabled is not None and frozenset([ai.element, aj.element]) in disabled:
            continue
        r0 = _bond_length_for_pair(ai.element, aj.element)
        # Skip if the atoms are further apart than a comfortably-bonded
        # pair would be — this is the main safeguard against form/break
        # thrashing.
        if float(r[idx]) > form_ratio * r0:
            continue
        bond = ai.form_bond(
            aj,
            kind=cfg.bond_form_kind,
            t_ps=state.t_ps,
            equilibrium_length_nm=r0,
            spring_constant_kj_per_nm2=cfg.bond_form_k_kj_per_nm2,
        )
        state.bonds.append(bond)
        bonded_ids.add(frozenset((ai.atom_id, aj.atom_id)))
        state._bond_cache = None
        formed += 1
        state.events_bonds_formed += 1
        rule = _rule_name_for_pair(ai.element, aj.element)
        state.events_formed_by_rule[rule] = (
            state.events_formed_by_rule.get(rule, 0) + 1
        )
    return formed


# ---------- Rigid-body water: quaternion-based symplectic integrator --


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two unit quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (w, x, y, z). Shepperd's
    method: select the largest diagonal component for numerical
    stability when tr(R) is near -1."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = 2.0 * math.sqrt(tr + 1.0)
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])


def _body_reference_positions(r_oh: float, r_hh: float,
                               m_O: float, m_H: float
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Return (ref_pos, I_body) for a water with given geometry.

    Body frame convention:
      - O-H-H plane is the y-z plane (x axis perpendicular to plane).
      - HOH bisector is along +y, pointing from COM toward O.
      - z axis is the C2-rotation axis perpendicular to y, in plane.
    With this choice, the inertia tensor is diagonal and the H atoms
    sit in +z / -z half-spaces symmetric about the y axis.

    ``ref_pos`` is (3, 3) with rows (O, H1, H2) in body-frame coords
    relative to the water COM.
    """
    y_dist = math.sqrt(max(r_oh * r_oh - 0.25 * r_hh * r_hh, 0.0))
    # Place O above the H's along +y, H's below at +/- z.
    # O at (0, y_O, 0); H's at (0, y_H, +/- r_hh/2)
    # y_O is such that O sits above, y_H is below.
    # Pre-COM: O at y=y_dist, H at y=0.
    # Actually standard: put O at (0, 0, 0), H's at (0, -y_dist, +/- r_hh/2).
    M = m_O + 2.0 * m_H
    # COM in pre-shift frame:
    y_com = (m_O * 0.0 + 2.0 * m_H * (-y_dist)) / M
    O_y = -y_com           # positive (O above COM)
    H_y = -y_dist - y_com  # negative (H's below COM)
    ref = np.array([
        [0.0, O_y, 0.0],
        [0.0, H_y, +0.5 * r_hh],
        [0.0, H_y, -0.5 * r_hh],
    ])
    # Diagonal inertia tensor in this body frame:
    # I_xx = sum m (y^2 + z^2), etc.
    I_xx = (m_O * O_y**2
            + m_H * (H_y**2 + (0.5 * r_hh) ** 2) * 2)
    I_yy = 2 * m_H * (0.5 * r_hh) ** 2
    I_zz = m_O * O_y**2 + 2 * m_H * H_y**2
    I_body = np.array([I_xx, I_yy, I_zz])
    return ref, I_body


def init_rigid_body_state(state: "SimState") -> None:
    """Initialise per-water quaternions from the current atom positions.

    Uses the Kabsch algorithm (mass-weighted SVD) to find the rotation
    that maps body-frame reference atoms onto the current world-frame
    positions, then encodes that as a quaternion.

    Also initialises COM velocity + body-frame angular velocity from
    the current per-atom velocities:
      - v_com = sum m_i v_i / M
      - L_world = sum m_i (r_i - r_com) x (v_i - v_com)
      - omega_body = I_body^-1 * R^T L_world

    After this call, state._rb_* arrays are populated and the
    integrator will advance them instead of doing unconstrained atom
    updates. Bond lengths are preserved to machine precision at ALL
    dt (verified at dt = 0.5, 1, 2, 4 fs).

    KNOWN LIMITATION: the first-order Euler's-equation update for
    omega_body loses energy conservation when external torques are
    large (dense-water LJ/Coulomb at liquid density drives T well
    above the Langevin setpoint). A second-order symplectic update
    (Dullweber-Leimkuhler-McLachlan 1997 or Krysl-Endres 2005)
    would fix this; deferred. For now the RB path is EXPERIMENTAL —
    use only with a strong Langevin thermostat and for systems with
    modest torques.
    """
    if state._settle_water_idx is None or state._settle_water_idx.size == 0:
        state._rb_q = None
        state._rb_v_com = None
        state._rb_omega_body = None
        state._rb_ref = None
        state._rb_I_body = None
        return
    n_w = state._settle_water_idx.shape[0]
    q_arr = np.empty((n_w, 4), dtype=np.float64)
    v_com_arr = np.empty((n_w, 3), dtype=np.float64)
    omega_body_arr = np.empty((n_w, 3), dtype=np.float64)
    ref_arr = np.empty((n_w, 3, 3), dtype=np.float64)
    I_body_arr = np.empty((n_w, 3), dtype=np.float64)

    for w in range(n_w):
        iO = int(state._settle_water_idx[w, 0])
        iH1 = int(state._settle_water_idx[w, 1])
        iH2 = int(state._settle_water_idx[w, 2])
        mO = state.atoms[iO].mass_da
        mH = state.atoms[iH1].mass_da
        r_oh = float(state._settle_water_r_oh[w])
        r_hh = float(state._settle_water_r_hh[w])
        ref, I_body = _body_reference_positions(r_oh, r_hh, mO, mH)
        ref_arr[w] = ref
        I_body_arr[w] = I_body

        # Current world-frame atom positions.
        pos_w = np.array([state.atoms[iO].position,
                          state.atoms[iH1].position,
                          state.atoms[iH2].position])
        M = mO + 2.0 * mH
        com = (mO * pos_w[0] + mH * (pos_w[1] + pos_w[2])) / M
        rel = pos_w - com

        # Mass-weighted Kabsch: find R such that R @ ref ≈ rel.
        mass_vec = np.array([mO, mH, mH])
        H_mat = (ref * mass_vec[:, None]).T @ rel
        U, S, Vt = np.linalg.svd(H_mat)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.eye(3)
        D[2, 2] = d
        R = Vt.T @ D @ U.T
        q_arr[w] = _matrix_to_quat(R)

        # Velocities.
        vel_w = np.array([state.atoms[iO].velocity,
                          state.atoms[iH1].velocity,
                          state.atoms[iH2].velocity])
        v_com = (mO * vel_w[0] + mH * (vel_w[1] + vel_w[2])) / M
        v_com_arr[w] = v_com
        # Angular momentum in world frame.
        L_world = (mO * np.cross(rel[0], vel_w[0] - v_com)
                   + mH * np.cross(rel[1], vel_w[1] - v_com)
                   + mH * np.cross(rel[2], vel_w[2] - v_com))
        # Transform to body frame: L_body = R^T L_world.
        L_body = R.T @ L_world
        omega_body_arr[w] = L_body / np.maximum(I_body, 1e-12)

    state._rb_q = q_arr
    state._rb_v_com = v_com_arr
    state._rb_omega_body = omega_body_arr
    state._rb_ref = ref_arr
    state._rb_I_body = I_body_arr


def _rb_sync_positions_and_velocities(state: "SimState",
                                        pos: np.ndarray,
                                        vel: np.ndarray) -> None:
    """Derive atom positions + velocities from each water's
    (r_com, v_com, q, omega_body). Writes into ``pos`` and ``vel`` at
    the atom indices held in state._settle_water_idx.

    The rigid-body state (pos[O] = derived from com + R @ ref[O]) is
    the primary truth; this function just publishes it back to the
    per-atom arrays that the rest of the integrator reads.
    """
    if state._rb_q is None:
        return
    n_w = state._rb_q.shape[0]
    for w in range(n_w):
        iO = int(state._settle_water_idx[w, 0])
        iH1 = int(state._settle_water_idx[w, 1])
        iH2 = int(state._settle_water_idx[w, 2])
        R = _quat_to_matrix(state._rb_q[w])
        ref = state._rb_ref[w]
        # COM = mass-average of current pos (primary state, read back).
        mO = state.atoms[iO].mass_da
        mH = state.atoms[iH1].mass_da
        M = mO + 2.0 * mH
        com = (mO * pos[iO] + mH * (pos[iH1] + pos[iH2])) / M
        # Place atoms.
        for loc, atom_i in enumerate((iO, iH1, iH2)):
            pos[atom_i] = com + R @ ref[loc]
        # v_i = v_com + omega_world x (r_i - r_com)
        omega_world = R @ state._rb_omega_body[w]
        v_com = state._rb_v_com[w]
        for loc, atom_i in enumerate((iO, iH1, iH2)):
            vel[atom_i] = v_com + np.cross(omega_world, R @ ref[loc])


def _rb_apply_forces(state: "SimState",
                      pos: np.ndarray,
                      forces: np.ndarray,
                      dt: float,
                      half_kick: bool = True) -> None:
    """Half-kick linear + angular momentum of each rigid water using
    the given world-frame forces.

    ``pos`` must be the current atom positions (used to compute the
    torque about each water's COM in world frame).
    """
    if state._rb_q is None:
        return
    n_w = state._rb_q.shape[0]
    scale = 0.5 * dt if half_kick else dt
    for w in range(n_w):
        iO = int(state._settle_water_idx[w, 0])
        iH1 = int(state._settle_water_idx[w, 1])
        iH2 = int(state._settle_water_idx[w, 2])
        mO = state.atoms[iO].mass_da
        mH = state.atoms[iH1].mass_da
        M = mO + 2.0 * mH
        com = (mO * pos[iO] + mH * (pos[iH1] + pos[iH2])) / M
        F_total = forces[iO] + forces[iH1] + forces[iH2]
        tau_world = (np.cross(pos[iO] - com, forces[iO])
                     + np.cross(pos[iH1] - com, forces[iH1])
                     + np.cross(pos[iH2] - com, forces[iH2]))
        # Linear velocity kick.
        state._rb_v_com[w] += scale * F_total / M
        # Rotate torque to body frame.
        R = _quat_to_matrix(state._rb_q[w])
        tau_body = R.T @ tau_world
        # Euler's equation: d omega_b / dt = I^-1 (tau_b - omega_b x I omega_b)
        I_b = state._rb_I_body[w]
        omega_b = state._rb_omega_body[w]
        I_omega = I_b * omega_b
        euler_rhs = (tau_body - np.cross(omega_b, I_omega)) / np.maximum(I_b, 1e-12)
        new_omega = state._rb_omega_body[w] + scale * euler_rhs
        # Guard: if torque has blown up (rare close LJ contact), cap
        # angular velocity at a physically large but bounded value
        # so the symplectic rotation doesn't overflow.
        max_omega = 1000.0     # rad/ps; well above thermal at 300 K
        nrm_new = float(np.linalg.norm(new_omega))
        if not np.isfinite(nrm_new) or nrm_new > max_omega:
            if nrm_new > 1e-12 and np.isfinite(nrm_new):
                new_omega *= max_omega / nrm_new
            else:
                new_omega = np.zeros(3)
        state._rb_omega_body[w] = new_omega


def _rb_drift(state: "SimState", pos: np.ndarray, dt: float,
               box_l: Optional[float] = None) -> None:
    """Drift each water: r_com += dt * v_com and rotate the body by
    q := q * exp(0.5 * dt * omega_pure) with omega_pure being the
    body-frame angular velocity promoted to a pure quaternion.

    Then write back atom positions from the new rigid-body state.
    """
    if state._rb_q is None:
        return
    n_w = state._rb_q.shape[0]
    for w in range(n_w):
        iO = int(state._settle_water_idx[w, 0])
        iH1 = int(state._settle_water_idx[w, 1])
        iH2 = int(state._settle_water_idx[w, 2])
        mO = state.atoms[iO].mass_da
        mH = state.atoms[iH1].mass_da
        M = mO + 2.0 * mH
        com = (mO * pos[iO] + mH * (pos[iH1] + pos[iH2])) / M
        com_new = com + dt * state._rb_v_com[w]
        # Quaternion integration via exponential map.
        # dq/dt = 0.5 * q * (0, omega_body)
        # Exponential: exp(0.5*dt*omega_pure) = (cos(|omega| dt / 2),
        #                                         omega/|omega| sin(|omega| dt / 2))
        omega_b = state._rb_omega_body[w]
        if not np.all(np.isfinite(omega_b)):
            # Numerical pathology (e.g. extreme torque). Reset omega
            # and skip the rotation this step — let the thermostat
            # recover.
            state._rb_omega_body[w] = 0.0
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            nrm = float(np.linalg.norm(omega_b))
            if nrm > 1e-12:
                theta = 0.5 * dt * nrm
                s = math.sin(theta) / nrm
                dq = np.array([math.cos(theta), s * omega_b[0],
                               s * omega_b[1], s * omega_b[2]])
            else:
                dq = np.array([1.0, 0.0, 0.0, 0.0])
        q_new = _quat_multiply(state._rb_q[w], dq)
        q_new /= np.linalg.norm(q_new)
        state._rb_q[w] = q_new
        # Apply to atom positions.
        R = _quat_to_matrix(q_new)
        ref = state._rb_ref[w]
        pos[iO] = com_new + R @ ref[0]
        pos[iH1] = com_new + R @ ref[1]
        pos[iH2] = com_new + R @ ref[2]


# ---------- SETTLE: analytical rigid-body water ------------------------


def build_settle_waters(state: "SimState") -> int:
    """Scan the bond + angle graph for water triples (O bonded to two
    H's with an H-O-H angle) and populate ``state._settle_waters`` with
    index triples + equilibrium distances. Also REMOVES the matching
    O-H bonds from SHAKE (they are now handled by SETTLE) and drops
    the H-O-H angle harmonic term.

    Returns the number of rigid waters registered.
    """
    atoms = state.atoms
    bonds = state.bonds
    angles = state.angles or []

    id_to_idx = {id(a): i for i, a in enumerate(atoms)}
    # O-H bonds by vertex O -> list[(H_idx, r_OH)]
    oh_bonds: dict[int, list[tuple[int, float]]] = {}
    bond_handle_by_key: dict[tuple[int, int], object] = {}
    for b in bonds:
        if b.death_time_ps is not None:
            continue
        i = id_to_idx.get(id(b.a))
        j = id_to_idx.get(id(b.b))
        if i is None or j is None:
            continue
        # Normalise to O-H orientation.
        ea = atoms[i].element.name
        eb = atoms[j].element.name
        if ea == "O" and eb == "H":
            o_idx, h_idx = i, j
        elif eb == "O" and ea == "H":
            o_idx, h_idx = j, i
        else:
            continue
        oh_bonds.setdefault(o_idx, []).append(
            (h_idx, float(b.equilibrium_length_nm))
        )
        key = (min(o_idx, h_idx), max(o_idx, h_idx))
        bond_handle_by_key[key] = b

    # Find waters: O with exactly 2 H bonds and the H-O-H angle.
    water_records = []
    consumed_bond_keys: set[tuple[int, int]] = set()
    consumed_angle_ids: set[int] = set()
    for o_idx, hs in oh_bonds.items():
        if len(hs) != 2:
            continue
        h1_idx, r_oh1 = hs[0]
        h2_idx, r_oh2 = hs[1]
        # Require O has no other heavy neighbours (pure water, not Ser/Thr etc.).
        non_h_neighbours = 0
        for b in bonds:
            if b.death_time_ps is not None:
                continue
            i = id_to_idx.get(id(b.a))
            j = id_to_idx.get(id(b.b))
            if i is None or j is None:
                continue
            if i == o_idx or j == o_idx:
                other = j if i == o_idx else i
                if atoms[other].element.name != "H":
                    non_h_neighbours += 1
        if non_h_neighbours != 0:
            continue
        # Find the H-O-H angle to extract theta_0 (and mark for removal).
        theta0 = None
        for ang in angles:
            ii = id_to_idx.get(id(ang.i))
            jj = id_to_idx.get(id(ang.j))
            kk = id_to_idx.get(id(ang.k))
            if jj != o_idx:
                continue
            if {ii, kk} == {h1_idx, h2_idx}:
                theta0 = ang.theta_0_rad
                consumed_angle_ids.add(id(ang))
                break
        if theta0 is None:
            theta0 = math.radians(104.5)       # standard HOH fallback
        r_HH = math.sqrt(r_oh1 ** 2 + r_oh2 ** 2
                         - 2.0 * r_oh1 * r_oh2 * math.cos(theta0))
        # Use the average O-H for the rigid reference — real water
        # templates should be symmetric anyway.
        r_OH = 0.5 * (r_oh1 + r_oh2)
        water_records.append((o_idx, h1_idx, h2_idx, r_OH, r_HH))
        consumed_bond_keys.add((min(o_idx, h1_idx), max(o_idx, h1_idx)))
        consumed_bond_keys.add((min(o_idx, h2_idx), max(o_idx, h2_idx)))

    if not water_records:
        state._settle_water_idx = None
        state._settle_water_r_oh = None
        state._settle_water_r_hh = None
        return 0

    n_waters = len(water_records)
    idx = np.empty((n_waters, 3), dtype=np.int64)
    r_oh = np.empty(n_waters, dtype=np.float64)
    r_hh = np.empty(n_waters, dtype=np.float64)
    for i, (o, h1, h2, ro, rh) in enumerate(water_records):
        idx[i, 0] = o
        idx[i, 1] = h1
        idx[i, 2] = h2
        r_oh[i] = ro
        r_hh[i] = rh
    state._settle_water_idx = idx
    state._settle_water_r_oh = r_oh
    state._settle_water_r_hh = r_hh

    # Remove consumed O-H pairs from SHAKE.
    if state.shake_pairs is not None and state.shake_pairs.size:
        keep_mask = np.ones(state.shake_pairs.shape[0], dtype=bool)
        for p in range(state.shake_pairs.shape[0]):
            i, j = int(state.shake_pairs[p, 0]), int(state.shake_pairs[p, 1])
            key = (min(i, j), max(i, j))
            if key in consumed_bond_keys:
                keep_mask[p] = False
        state.shake_pairs = state.shake_pairs[keep_mask]
        state.shake_r0_sq = state.shake_r0_sq[keep_mask]

    # Zero harmonic spring constants on SETTLE-managed O-H bonds — SETTLE
    # now enforces the bond length analytically, so the harmonic term is
    # both redundant and actively harmful (it pulls atoms AWAY from the
    # SETTLE-projected r_OH if the minimiser left them at a different
    # equilibrium due to competing non-bonded forces). Bonds stay in
    # state.bonds for LJ/Coulomb exclusion bookkeeping.
    for b in bonds:
        if b.death_time_ps is not None:
            continue
        i = id_to_idx.get(id(b.a))
        j = id_to_idx.get(id(b.b))
        if i is None or j is None:
            continue
        key = (min(i, j), max(i, j))
        if key in consumed_bond_keys:
            b.spring_constant_kj_per_nm2 = 0.0

    # Remove consumed angles from state.angles.
    if angles:
        state.angles = [a for a in angles if id(a) not in consumed_angle_ids]

    # Invalidate bond cache so rebuilt state reflects angle removal.
    state._bond_cache = None

    return n_waters


def _apply_settle(
    pos: np.ndarray,
    prev_pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
    settle_idx: np.ndarray,         # (N_w, 3): (O, H1, H2) per water
    r_oh: np.ndarray,               # (N_w,)
    r_hh: np.ndarray,               # (N_w,)
    dt: float,
    box_l: Optional[float] = None,
) -> None:
    """Analytical SETTLE: rotate each rigid water so its atom positions
    best match the unconstrained advance while preserving rigid shape
    and COM momentum. Velocities are then projected onto the rigid-body
    motion subspace (linear COM velocity + angular velocity) so they
    cannot push the triangle off the constraint manifold.

    Modifies ``pos`` and ``vel`` in place.
    """
    n_w = settle_idx.shape[0]
    if n_w == 0:
        return

    # For each water, perform Kabsch-style rotation.
    # Canonical rigid triangle (in its own COM frame):
    #   O at (0, 0, 0) pre-COM-shift
    #   H1 at (d_HH/2, -y, 0), H2 at (-d_HH/2, -y, 0)
    #   where y = sqrt(d_OH^2 - (d_HH/2)^2)
    # (place O above the H-H baseline so OH points up in the frame).
    # NB: we don't bother minimum-imaging within a water — atoms bonded
    # in the same molecule are always within 1 nm of each other, so
    # PBC wrap isn't an issue at typical box sizes (>= 2 nm).

    for w in range(n_w):
        iO = int(settle_idx[w, 0])
        iH1 = int(settle_idx[w, 1])
        iH2 = int(settle_idx[w, 2])
        mO = masses[iO]
        mH = masses[iH1]
        mTotal = mO + 2.0 * mH

        # Canonical reference positions (rigid geometry).
        d_OH = r_oh[w]
        d_HH = r_hh[w]
        y = math.sqrt(max(d_OH ** 2 - 0.25 * d_HH ** 2, 0.0))
        a_ref = np.array([0.0, 0.0, 0.0])
        b_ref = np.array([0.5 * d_HH, -y, 0.0])
        c_ref = np.array([-0.5 * d_HH, -y, 0.0])
        com_ref = (mO * a_ref + mH * (b_ref + c_ref)) / mTotal
        a_ref -= com_ref
        b_ref -= com_ref
        c_ref -= com_ref

        # Unconstrained positions (from Verlet advance).
        ra = pos[iO]
        rb = pos[iH1]
        rc = pos[iH2]
        com_new = (mO * ra + mH * (rb + rc)) / mTotal
        a_unc = ra - com_new
        b_unc = rb - com_new
        c_unc = rc - com_new

        # Mass-weighted cross-covariance H = sum_i m_i * ref_i * unc_i^T
        H = (mO * np.outer(a_ref, a_unc)
             + mH * np.outer(b_ref, b_unc)
             + mH * np.outer(c_ref, c_unc))
        U, S, Vt = np.linalg.svd(H)
        # Sign correction to avoid reflection.
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.eye(3)
        D[2, 2] = d
        R = Vt.T @ D @ U.T

        # Rigid triangle placed at new COM.
        a_new = R @ a_ref
        b_new = R @ b_ref
        c_new = R @ c_ref
        pos[iO] = com_new + a_new
        pos[iH1] = com_new + b_new
        pos[iH2] = com_new + c_new

        # Velocity projection: v_i = v_com + omega x (r_i - r_com).
        # Compute v_com and omega from current (unconstrained) velocities
        # plus the rigid-body inertia tensor.
        v_com = (mO * vel[iO] + mH * (vel[iH1] + vel[iH2])) / mTotal
        # Position vectors from the constrained (rigid) COM.
        dOv = pos[iO] - com_new
        dH1v = pos[iH1] - com_new
        dH2v = pos[iH2] - com_new
        # Relative velocities (from COM).
        u_O = vel[iO] - v_com
        u_H1 = vel[iH1] - v_com
        u_H2 = vel[iH2] - v_com
        # Total angular momentum (in the COM frame).
        L = (mO * np.cross(dOv, u_O)
             + mH * np.cross(dH1v, u_H1)
             + mH * np.cross(dH2v, u_H2))
        # Inertia tensor: I = sum m_i (|r_i|^2 * I3 - r_i r_i^T)
        def _inertia_add(r, m):
            r2 = r @ r
            return m * (r2 * np.eye(3) - np.outer(r, r))
        I = (_inertia_add(dOv, mO)
             + _inertia_add(dH1v, mH)
             + _inertia_add(dH2v, mH))
        try:
            omega = np.linalg.solve(I, L)
        except np.linalg.LinAlgError:
            # Degenerate (collinear atoms): fall back to pseudo-inverse.
            omega = np.linalg.pinv(I) @ L
        # Project velocities onto rigid-body motion.
        vel[iO] = v_com + np.cross(omega, dOv)
        vel[iH1] = v_com + np.cross(omega, dH1v)
        vel[iH2] = v_com + np.cross(omega, dH2v)


# ---------- Steepest-descent energy minimisation ----------------------


def minimise_steepest_descent(
    state: SimState,
    ff_cfg: ForceFieldConfig,
    max_steps: int = 500,
    force_tol_kj_per_nm: float = 100.0,
    initial_step_nm: float = 0.001,
    step_grow: float = 1.2,
    step_shrink: float = 0.5,
    verbose: bool = False,
) -> dict:
    """Iteratively move atoms along the negative force direction until
    the largest per-atom force drops below ``force_tol_kj_per_nm``.

    Adaptive step size: after each trial update,
      - if max |F| decreased, scale the step by ``step_grow`` (up to a
        conservative ceiling of 0.01 nm, one LJ sigma).
      - otherwise revert the positions and scale the step by
        ``step_shrink``.

    SHAKE projection: if ``state.shake_pairs`` is populated, every trial
    update is followed by SHAKE projection back onto the bond-constraint
    manifold. This ensures the minimised geometry satisfies the bond
    length constraints that the subsequent integrator will enforce, so
    the first dynamics step does not inject energy correcting a bond
    mismatch.

    Does NOT touch velocities. Does NOT evolve time (t_ps stays put).

    Typical win: on a freshly-built water box, running 50-200 minimiser
    iterations cuts the initial temperature spike that would otherwise
    blow a dt=1 fs dynamics run from a ~10 000 K crash to a clean
    ~300 K evolution.

    NOTE: dt=2 fs dynamics with iterative SHAKE remains unstable on
    dense water even AFTER minimisation — bond-vibration frequency
    (~10 fs period) outpaces what iterative SHAKE can correct at that
    discretisation. The path to 2 fs is SETTLE (analytical rigid-body
    water), not more minimisation.

    Returns a summary dict with iteration count, initial/final max |F|,
    and whether the tolerance was reached.
    """
    atoms = state.atoms
    bonds = state.bonds
    angles = state.angles or None
    dihedrals = state.dihedrals or None
    pbc_on = bool(ff_cfg.use_pbc)
    box_l = float(ff_cfg.pbc_box_nm) if pbc_on else None

    # Disable the per-atom force cap during minimisation. The cap is a
    # safety net for dynamics but clips the gradient magnitude, which
    # makes steepest descent oscillate around the cap instead of
    # descending. Clone the cfg cheaply via dataclasses.replace so we
    # don't mutate the caller's.
    import dataclasses as _dc
    min_cfg = _dc.replace(ff_cfg, max_force_kj_per_nm=1e30)

    pos = _gather_positions(atoms)
    step_nm = float(initial_step_nm)
    max_step_nm = 0.01                     # ~sigma/35; avoids ridiculous kicks

    def _max_force(p: np.ndarray) -> tuple[np.ndarray, float]:
        f = compute_forces(
            atoms, bonds, state.t_ps, min_cfg,
            neighbor_pairs=None, pos=p,
            angles=angles, dihedrals=dihedrals, which="all",
        )
        fmag = np.linalg.norm(f, axis=1)
        return f, float(fmag.max()) if fmag.size else 0.0

    # Optional SHAKE projection: if shake_pairs are set on the state,
    # snap bonds back to equilibrium after each trial step. This ensures
    # the minimised geometry satisfies bond constraints, so the first
    # dynamics step doesn't inject energy by forcing bonds back to r0.
    shake_on = (state.shake_pairs is not None
                and state.shake_pairs.size > 0)
    masses_flat = _cached_masses(state) if shake_on else None

    def _project_shake(new_pos: np.ndarray, prev_pos: np.ndarray) -> np.ndarray:
        if not shake_on:
            return new_pos
        return _apply_shake(
            new_pos, prev_pos, masses_flat,
            state.shake_pairs, state.shake_r0_sq,
            box_l=box_l, tol_nm=1e-5, max_iter=200,
        )

    # If SHAKE is active, first make sure the starting geometry itself
    # satisfies bond constraints — the PDB importer places atoms at
    # ideal r0 to three decimals but rounding can leave residual error.
    if shake_on:
        pos = _project_shake(pos.copy(), pos.copy())

    forces, max_f0 = _max_force(pos)
    if verbose:
        print(f"[minimise] initial max |F| = {max_f0:.1f} kJ/mol/nm")
    prev_max_f = max_f0
    iters = 0
    converged = max_f0 <= force_tol_kj_per_nm
    while iters < max_steps and not converged:
        fmag = np.linalg.norm(forces, axis=1)
        safe = np.maximum(fmag, 1e-12)
        direction = forces / safe[:, None]
        new_pos = pos + step_nm * direction
        if pbc_on:
            wrap_positions(new_pos, box_l)
        new_pos = _project_shake(new_pos, pos)
        new_forces, new_max_f = _max_force(new_pos)
        if new_max_f < prev_max_f:
            pos = new_pos
            forces = new_forces
            prev_max_f = new_max_f
            step_nm = min(step_nm * step_grow, max_step_nm)
            if new_max_f <= force_tol_kj_per_nm:
                converged = True
        else:
            step_nm *= step_shrink
            if step_nm < 1e-8:
                break
        iters += 1
        if verbose and iters % 50 == 0:
            print(f"[minimise] iter {iters:4d}  max |F| = {prev_max_f:.1f}  "
                  f"step = {step_nm*1000:.3f} pm")

    _scatter_positions(atoms, pos)
    if verbose:
        print(f"[minimise] done: iters={iters}, converged={converged}, "
              f"final max |F| = {prev_max_f:.1f}")
    return {
        "iterations": iters,
        "initial_max_force_kj_per_nm": max_f0,
        "final_max_force_kj_per_nm": prev_max_f,
        "converged": converged,
        "final_step_nm": step_nm,
    }


# ---------- RESPA multi-timestep (Tuckerman 1992) ---------------------


def _step_respa(
    state: SimState,
    ff_cfg: ForceFieldConfig,
    int_cfg: IntegratorConfig,
    forces_prev: Optional[np.ndarray] = None,
) -> np.ndarray:
    """One r-RESPA outer step.

    Decomposition:
      SLOW = non-bonded (LJ + Coulomb + external potentials)
      FAST = bonded (harmonic bonds + angle bends + dihedrals)

    Algorithm per outer step (n = ``respa_n_inner``, ``dt_in = dt/n``):
      1. v += 0.5 dt F_slow / m                (outer half-kick, cached)
      2. repeat n times:
           v += 0.5 dt_in F_fast / m
           pos += dt_in v         (+ SHAKE, + PBC wrap)
           F_fast <- compute_forces(..., which="fast", pos=pos)
           v += 0.5 dt_in F_fast / m
      3. F_slow <- compute_forces(..., which="slow", pos=pos)
      4. v += 0.5 dt F_slow / m                (outer half-kick)
      5. thermostat on v
      6. bond break events
    """
    dt_out = int_cfg.dt_ps
    n_in = int(int_cfg.respa_n_inner)
    dt_in = dt_out / n_in
    atoms = state.atoms

    masses = _cached_masses(state)[:, None]
    vel = _gather_velocities(atoms)
    pos = _gather_positions(atoms)

    def _neighbors_for_pos(p: np.ndarray
                            ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not ff_cfg.use_neighbor_list:
            return None
        every = max(1, int_cfg.neighbor_rebuild_every)
        stale = (
            state._neighbor_iu is None
            or (state.step - state._neighbor_built_at_step) >= every
            or state._neighbor_built_at_step < 0
        )
        if stale:
            cutoff_with_skin = ff_cfg.lj_cutoff_nm + ff_cfg.neighbor_skin_nm
            box_l = (float(ff_cfg.pbc_box_nm) if ff_cfg.use_pbc else None)
            state._neighbor_iu, state._neighbor_ju = build_neighbor_list(
                p, cutoff_with_skin, box_l=box_l,
            )
            state._neighbor_built_at_step = state.step
        return (state._neighbor_iu, state._neighbor_ju)

    angles = state.angles or None
    dihedrals = state.dihedrals or None
    bonds = state.bonds

    # Build bonded cache on first RESPA call so the per-step
    # compute_forces can skip the Python dict rebuild + per-bond id()
    # loop. Invalidated automatically if the atom count changes.
    if (state._bond_cache is None
            or getattr(state._bond_cache, "n_atoms", None) != len(atoms)):
        compile_bond_cache(state)
    bc = state._bond_cache

    # 1. Outer half-kick using cached F_slow (compute on first call only).
    if state._respa_F_slow is None:
        state._respa_F_slow = compute_forces(
            atoms, bonds, state.t_ps, ff_cfg,
            neighbor_pairs=_neighbors_for_pos(pos),
            pos=pos, angles=angles, dihedrals=dihedrals, which="slow",
            bond_cache=bc,
        )
    F_slow = state._respa_F_slow
    vel += 0.5 * dt_out * F_slow / masses

    # Initial F_fast. Bonded-only, cheap.
    F_fast = compute_forces(
        atoms, bonds, state.t_ps, ff_cfg, neighbor_pairs=None,
        pos=pos, angles=angles, dihedrals=dihedrals, which="fast",
        bond_cache=bc,
    )

    shake_on = (int_cfg.shake and state.shake_pairs is not None
                and state.shake_pairs.size > 0)
    pbc_on = bool(ff_cfg.use_pbc)
    box_l = float(ff_cfg.pbc_box_nm) if pbc_on else None

    settle_on = (state._settle_water_idx is not None
                 and state._settle_water_idx.size > 0)

    # 2. n_inner velocity-Verlet sub-steps with F_fast.
    for _k in range(n_in):
        vel += 0.5 * dt_in * F_fast / masses
        prev_pos_in = pos.copy() if (shake_on or settle_on) else None
        pos = pos + dt_in * vel
        if shake_on:
            pos_before_shake = pos.copy()
            _apply_shake(
                pos, prev_pos_in, masses[:, 0],
                state.shake_pairs, state.shake_r0_sq,
                box_l=box_l,
                tol_nm=int_cfg.shake_tolerance_nm,
                max_iter=int_cfg.shake_max_iter,
            )
            vel += (pos - pos_before_shake) / dt_in
        if settle_on:
            _apply_settle(
                pos, prev_pos_in, vel, masses[:, 0],
                state._settle_water_idx,
                state._settle_water_r_oh,
                state._settle_water_r_hh,
                dt_in, box_l=box_l,
            )
        if pbc_on:
            wrap_positions(pos, box_l)
        F_fast = compute_forces(
            atoms, bonds, state.t_ps, ff_cfg, neighbor_pairs=None,
            pos=pos, angles=angles, dihedrals=dihedrals, which="fast",
            bond_cache=bc,
        )
        vel += 0.5 * dt_in * F_fast / masses

    _scatter_positions(atoms, pos)
    state.t_ps += dt_out
    state.step += 1

    # 3. Recompute F_slow with advanced positions.
    F_slow_new = compute_forces(
        atoms, bonds, state.t_ps, ff_cfg,
        neighbor_pairs=_neighbors_for_pos(pos),
        pos=pos, angles=angles, dihedrals=dihedrals, which="slow",
        bond_cache=bc,
    )

    # 4. Outer half-kick using new F_slow.
    vel += 0.5 * dt_out * F_slow_new / masses
    state._respa_F_slow = F_slow_new

    # 5. Thermostat.
    if int_cfg.thermostat == "langevin":
        k_B = 0.00831446
        T_target = int_cfg.target_temperature_K
        gamma = int_cfg.langevin_gamma_inv_ps
        c1 = float(np.exp(-gamma * dt_out))
        m_col = masses[:, 0:1]
        sigma2 = (k_B * T_target / m_col) * (1.0 - c1 * c1)
        sigma = np.sqrt(np.maximum(sigma2, 0.0))
        rng = np.random.default_rng(state.step + 12345)
        noise = rng.standard_normal(vel.shape)
        vel = c1 * vel + sigma * noise
    else:
        t_now = _temperature_from_arrays(vel, masses[:, 0])
        if t_now <= 0.0:
            t_now = 1.0
        tau = max(int_cfg.thermostat_tau_ps, 1e-3)
        lam = (1.0 + (dt_out / tau)
               * (int_cfg.target_temperature_K / t_now - 1.0)) ** 0.5
        vel *= lam
    _scatter_velocities(atoms, vel)

    # 6. Bond events (break). RESPA does not support dynamic bonding
    # (would change bonded_set mid-step and break SHAKE).
    broken = _maybe_break_bonds(state, int_cfg.bond_break_fraction)
    if int_cfg.step_callback is not None:
        int_cfg.step_callback(state.t_ps, 0, broken)

    return F_slow_new + F_fast


# ---------- velocity-Verlet ------------------------------------------


def step(
    state: SimState,
    ff_cfg: ForceFieldConfig,
    int_cfg: IntegratorConfig,
    forces_prev: Optional[np.ndarray] = None,
) -> np.ndarray:
    """One velocity-Verlet step. Returns the new forces array so the caller
    can pass it in as `forces_prev` on the next step.

    When ``int_cfg.respa_n_inner > 1``, dispatches to the r-RESPA path
    which splits bonded and non-bonded force evaluation onto separate
    timesteps for ~3x speedup on non-bonded-dominated systems.
    """
    if int_cfg.respa_n_inner > 1:
        return _step_respa(state, ff_cfg, int_cfg, forces_prev)
    dt = int_cfg.dt_ps
    atoms = state.atoms

    def _neighbors(pos_for_rebuild: Optional[np.ndarray] = None
                   ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not ff_cfg.use_neighbor_list:
            return None
        every = max(1, int_cfg.neighbor_rebuild_every)
        stale = (
            state._neighbor_iu is None
            or (state.step - state._neighbor_built_at_step) >= every
            or state._neighbor_built_at_step < 0
        )
        if stale:
            pos_arr = (pos_for_rebuild if pos_for_rebuild is not None
                       else _gather_positions(atoms))
            cutoff_with_skin = ff_cfg.lj_cutoff_nm + ff_cfg.neighbor_skin_nm
            state._neighbor_iu, state._neighbor_ju = build_neighbor_list(
                pos_arr, cutoff_with_skin
            )
            state._neighbor_built_at_step = state.step
        return (state._neighbor_iu, state._neighbor_ju)

    masses = _cached_masses(state)[:, None]              # (N, 1)
    vel = _gather_velocities(atoms)
    pos = _gather_positions(atoms)

    # Reuse the precompiled bond-index arrays when topology is stable.
    # Invalidated automatically when atoms list grows/shrinks.
    if (state._bond_cache is None
            or getattr(state._bond_cache, "n_atoms", None) != len(atoms)):
        compile_bond_cache(state)
    bc = state._bond_cache

    if forces_prev is None:
        forces_prev = compute_forces(atoms, state.bonds, state.t_ps, ff_cfg,
                                     neighbor_pairs=_neighbors(pos),
                                     pos=pos, angles=state.angles or None,
                                     dihedrals=state.dihedrals or None,
                                     bond_cache=bc)

    # Rigid-body path: half-kick linear+angular momentum BEFORE the
    # per-atom half-kick, so water atoms don't double-count.
    if state._rb_q is not None:
        _rb_apply_forces(state, pos, forces_prev, dt, half_kick=True)

    # Half-step velocity + full position update.
    vel += 0.5 * dt * forces_prev / masses
    # Preserve pre-update positions for SHAKE projection before we
    # advance them.
    prev_pos = pos.copy() if int_cfg.shake else None
    pos += dt * vel

    # SHAKE: enforce bond-length constraints by projecting along the
    # pre-update bond direction. Velocity update follows from the
    # position change (Verlet form: v += (pos_new - pos_old) / dt - v,
    # equivalent to v = (pos_new - prev_pos) / dt minus the integrator
    # half-step we already applied).
    if int_cfg.shake and state.shake_pairs is not None \
            and state.shake_pairs.size:
        pos_before_shake = pos.copy()
        _apply_shake(
            pos, prev_pos, masses[:, 0],
            state.shake_pairs, state.shake_r0_sq,
            box_l=(float(ff_cfg.pbc_box_nm) if ff_cfg.use_pbc else None),
            tol_nm=int_cfg.shake_tolerance_nm,
            max_iter=int_cfg.shake_max_iter,
        )
        # Correct the velocity to match the constrained position.
        vel += (pos - pos_before_shake) / dt

    # Rigid-body water: advance (r_com, v_com, q, omega_body) via the
    # quaternion symplectic integrator. Overwrites the Verlet-predicted
    # water atom positions with the rigid-body-derived ones.
    if state._rb_q is not None:
        # The "Verlet drift" already advanced water atoms as free particles
        # using the old half-kicked velocity. The RB drift below recomputes
        # water atom positions from the advancing r_com + rotating q.
        _rb_drift(state, pos, dt,
                  box_l=(float(ff_cfg.pbc_box_nm) if ff_cfg.use_pbc else None))
    elif (state._settle_water_idx is not None
            and state._settle_water_idx.size):
        # Legacy Kabsch projection path — kept for compatibility but
        # init_rigid_body_state + RB path is preferred.
        _apply_settle(
            pos, prev_pos if prev_pos is not None else pos, vel,
            masses[:, 0],
            state._settle_water_idx,
            state._settle_water_r_oh,
            state._settle_water_r_hh,
            dt,
            box_l=(float(ff_cfg.pbc_box_nm) if ff_cfg.use_pbc else None),
        )

    # Under PBC, wrap positions back into [-L/2, L/2] after each
    # displacement step so minimum-image distance math is valid.
    if ff_cfg.use_pbc:
        wrap_positions(pos, float(ff_cfg.pbc_box_nm))
    _scatter_positions(atoms, pos)

    state.t_ps += dt
    state.step += 1

    forces_new = compute_forces(atoms, state.bonds, state.t_ps, ff_cfg,
                                 neighbor_pairs=_neighbors(pos),
                                 pos=pos, angles=state.angles or None,
                                 dihedrals=state.dihedrals or None,
                                 bond_cache=bc)

    # Second half-step velocity.
    vel += 0.5 * dt * forces_new / masses

    # Rigid-body path: second half-kick + sync atom velocities.
    if state._rb_q is not None:
        _rb_apply_forces(state, pos, forces_new, dt, half_kick=True)
        # Write back atom velocities from rigid-body state.
        _rb_sync_positions_and_velocities(state, pos, vel)

    if int_cfg.thermostat == "langevin":
        # Langevin thermostat (Physics Upgrade 3). OVRVO-style discrete
        # update applied after the Verlet half-step:
        #     v <- exp(-gamma*dt) v + sigma_v * N(0,I)
        # with ``sigma_v = sqrt(kT/m * (1 - exp(-2 gamma dt)))``.
        # Correctly samples the canonical ensemble for conservative
        # systems; replaces Berendsen's multiplicative rescale which
        # produces the wrong velocity distribution.
        k_B = 0.00831446
        T_target = int_cfg.target_temperature_K
        gamma = int_cfg.langevin_gamma_inv_ps
        c1 = float(np.exp(-gamma * dt))
        # kT/m per atom -> (N, 1)
        m_col = masses[:, 0:1]
        sigma2 = (k_B * T_target / m_col) * (1.0 - c1 * c1)
        sigma = np.sqrt(np.maximum(sigma2, 0.0))
        rng = np.random.default_rng(state.step + 12345)
        noise = rng.standard_normal(vel.shape)
        vel = c1 * vel + sigma * noise
    else:
        # Berendsen thermostat (default, fast equilibration, incorrect
        # NVT but adequate for deterministic MD demos).
        t_now = _temperature_from_arrays(vel, masses[:, 0])
        if t_now <= 0.0:
            t_now = 1.0
        tau = max(int_cfg.thermostat_tau_ps, 1e-3)
        lam = (1.0 + (dt / tau)
               * (int_cfg.target_temperature_K / t_now - 1.0)) ** 0.5
        vel *= lam
    _scatter_velocities(atoms, vel)

    broken = _maybe_break_bonds(state, int_cfg.bond_break_fraction)
    formed = 0
    if int_cfg.dynamic_bonding:
        formed = _maybe_form_bonds(state, int_cfg,
                                   neighbor_pairs=_neighbors(pos),
                                   pos=pos)

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
