"""
DMVC P3b: Compartment-aware stamps.

DIAGNOSIS FROM P3: our stamps encoded only (C,H,O,N,P,S, charge) and were
IDENTICAL for chemically-identical molecules across compartments (e.g.
M_h_c and M_h_e both had stamp = [0,1,0,0,0,0,+1,0,...]).

This made TRANSPORT reactions invisible to per-compartment atom readouts,
because transporting H+ from extra to cyto doesn't change the stamp field
at all -- the stamps cancel in the sum.

The FIX: expand the stamp subspace to include compartment identity. For each
(atom_type, compartment) pair, dedicate one stamp dimension. For C compartments
and K_atoms atom types, the stamp subspace has K_atoms * C atom-dims + C
charge-dims, so K_stamp = (K_atoms+1) * C.

Concretely for our 3-compartment setup (extra=0, mem=1, cyto=2):
  stamp[0..K_atoms-1]:             atoms in extra
  stamp[K_atoms..2K_atoms-1]:      atoms in mem
  stamp[2K_atoms..3K_atoms-1]:     atoms in cyto
  stamp[3K_atoms]:                 charge in extra
  stamp[3K_atoms+1]:               charge in mem
  stamp[3K_atoms+2]:               charge in cyto

Each molecule contributes its atom counts to the stamp slice corresponding
to its compartment, and zero to the other compartments' slices.

This way:
  - A transport reaction's stamp has NEGATIVE entries in one compartment
    (where species leaves) and POSITIVE in another (where species appears),
    so it's NOT zero. Per-compartment atom readouts correctly see transport.
  - Total atoms (summing across compartment slices) remains conserved by
    balanced reactions because the total stoichiometric balance still holds.
  - Internal reactions still cancel within their one compartment's slice.

Tests:
  T1: internal reaction affects only its compartment's stamp (same as P3)
  T2: per-compartment atom totals preserved by internal reactions (same)
  T3: per-compartment atom totals VISIBLY MOVE under transport reactions
      (while global totals are preserved) -- the thing P3 couldn't see
  T4: global totals preserved across all reaction types (stronger version of T6)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict
from enum import IntEnum

import sys
sys.path.insert(0, "/home/claude/dmvc")
from prototype_p3_compartments import (
    Compartment, COMPARTMENT_NAMES,
    extract_molecules_with_compartments,
    extract_reactions, classify_by_compartment, RxnKind,
    build_spherical_cell, Molecule, SBMLReaction,
    ATOM_TYPES, K_atoms, load_sbml_model, SBML_PATH,
    find_proton_and_water,
)


# =============================================================================
# Compartment-aware stamp dimensions
# =============================================================================
N_COMPARTMENTS = 3  # EXTRA, MEMBRANE, CYTO
K_STAMP = (K_atoms + 1) * N_COMPARTMENTS   # atoms+charge per compartment
# Layout: [atoms_EXTRA | atoms_MEM | atoms_CYTO | charge_EXTRA | charge_MEM | charge_CYTO]
# size:    K_atoms       K_atoms    K_atoms       1              1            1


def stamp_idx_for_atom(atom_idx: int, comp: Compartment) -> int:
    """Index into stamp subspace for atom-count of given type in given compartment."""
    return int(comp) * K_atoms + atom_idx


def stamp_idx_for_charge(comp: Compartment) -> int:
    """Index into stamp for charge in given compartment."""
    return N_COMPARTMENTS * K_atoms + int(comp)


def build_embeddings_compartment_aware(mols: Dict[str, Molecule], D: int,
                                         rng: np.random.Generator):
    """
    For each molecule with full info:
        stamp[atom k, in its compartment] = atom_count[k]
        stamp[charge, in its compartment] = charge
        flavor = random unit-norm vector in dims K_STAMP..D-1
    """
    assert D > K_STAMP, f"Need D > {K_STAMP} for compartment-aware stamps"
    for m in mols.values():
        if m.atom_count is None or m.charge is None or m.compartment is None:
            continue
        stamp = np.zeros(D)
        for k in range(K_atoms):
            stamp[stamp_idx_for_atom(k, m.compartment)] = m.atom_count[k]
        stamp[stamp_idx_for_charge(m.compartment)] = m.charge
        flavor = np.zeros(D)
        rnd = rng.standard_normal(D - K_STAMP)
        rnd /= np.linalg.norm(rnd) + 1e-12
        flavor[K_STAMP:] = rnd
        m.embedding = stamp + flavor


# =============================================================================
# Per-compartment readouts using the new stamp layout
# =============================================================================
@dataclass
class CellStateCA:
    Nx: int; Ny: int; Nz: int; L: float
    Psi: np.ndarray
    compartment: np.ndarray

    @property
    def D(self): return self.Psi.shape[-1]
    @property
    def dV(self): return (self.L / self.Nx) ** 3
    def mask(self, c: Compartment) -> np.ndarray:
        return self.compartment == c


def seed_state(Nx, Ny, Nz, D, L, rng, amp=0.2) -> CellStateCA:
    """
    Important: Psi is initialized only in the stamp dimensions that match
    the voxel's own compartment. Stamps of OTHER compartments are zero at
    that voxel. Otherwise per-compartment readouts would show phantom atoms
    in compartments where those molecules don't exist.
    """
    comp = build_spherical_cell(Nx, Ny, Nz, L)
    Psi = np.zeros((Nx, Ny, Nz, D))
    # Start with small random flavor everywhere
    Psi[..., K_STAMP:] = amp * rng.standard_normal((Nx, Ny, Nz, D - K_STAMP))
    # For each voxel, seed small atom counts of THAT compartment's atom-slots
    for c in Compartment:
        mask = (comp == c)
        if not mask.any(): continue
        # small random nonnegative-ish occupancy
        for k in range(K_atoms):
            idx = stamp_idx_for_atom(k, c)
            Psi[..., idx] = amp * rng.standard_normal((Nx, Ny, Nz))[...] * mask
        # charge
        ic = stamp_idx_for_charge(c)
        Psi[..., ic] = amp * rng.standard_normal((Nx, Ny, Nz)) * mask
    return CellStateCA(Nx, Ny, Nz, L, Psi, comp)


def atom_totals_by_compartment(state: CellStateCA) -> Dict[Compartment, np.ndarray]:
    """Per-compartment atom totals from the compartment-specific stamp slices."""
    out = {}
    for c in Compartment:
        atoms = np.zeros(K_atoms)
        for k in range(K_atoms):
            atoms[k] = state.Psi[..., stamp_idx_for_atom(k, c)].sum() * state.dV
        out[c] = atoms
    return out


def atom_totals_global(state: CellStateCA) -> np.ndarray:
    """Global atom totals summed over compartments."""
    totals = np.zeros(K_atoms)
    for c in Compartment:
        for k in range(K_atoms):
            totals[k] += state.Psi[..., stamp_idx_for_atom(k, c)].sum() * state.dV
    return totals


def charge_by_compartment(state: CellStateCA) -> Dict[Compartment, float]:
    out = {}
    for c in Compartment:
        out[c] = float(state.Psi[..., stamp_idx_for_charge(c)].sum() * state.dV)
    return out


def charge_total_global(state: CellStateCA) -> float:
    total = 0.0
    for c in Compartment:
        total += state.Psi[..., stamp_idx_for_charge(c)].sum() * state.dV
    return float(total)


# =============================================================================
# Reaction latent + application (same structure as P3, but stamps now differ)
# =============================================================================
def build_reaction_latent(rxn: SBMLReaction,
                           mols: Dict[str, Molecule]) -> Optional[np.ndarray]:
    r = None
    for sid, coef in rxn.stoichiometry.items():
        m = mols[sid]
        if m.embedding is None: return None
        term = coef * m.embedding
        r = term if r is None else r + term
    return r


def apply_internal(state: CellStateCA, r_lat: np.ndarray, rate_field: np.ndarray,
                    target: Compartment, dt: float):
    mask = state.mask(target).astype(np.float64)
    state.Psi += dt * (rate_field * mask)[..., None] * r_lat[None, None, None, :]


def apply_transport(state: CellStateCA, r_lat: np.ndarray, rate_field: np.ndarray,
                     dt: float):
    mask = state.mask(Compartment.MEMBRANE).astype(np.float64)
    state.Psi += dt * (rate_field * mask)[..., None] * r_lat[None, None, None, :]


# =============================================================================
# Rebalance with compartment-correct proton/water (copied+adapted from P3)
# =============================================================================
def find_h_and_water_per_compartment(mols: Dict[str, Molecule]):
    p = {Compartment.CYTO: None, Compartment.EXTRA: None}
    w = {Compartment.CYTO: None, Compartment.EXTRA: None}
    for sid, m in mols.items():
        if m.atom_count is None or m.charge is None or m.compartment is None:
            continue
        is_proton = (np.sum(m.atom_count) == 1 and m.atom_count[1] == 1 and m.charge == 1)
        is_water = (m.atom_count[1] == 2 and m.atom_count[2] == 1
                     and m.charge == 0 and np.sum(m.atom_count) == 3)
        if is_proton and m.compartment in p:
            p[m.compartment] = sid
        if is_water and m.compartment in w:
            w[m.compartment] = sid
    return p, w


def rebalance_reaction(rxn: SBMLReaction, mols: Dict[str, Molecule],
                         p_ids, w_ids) -> Optional[SBMLReaction]:
    if rxn.is_exchange or rxn.is_biomass:
        return rxn
    # Figure out dominant compartment (where most species live)
    counts = defaultdict(int)
    for sid in rxn.stoichiometry:
        m = mols.get(sid)
        if m and m.compartment:
            counts[m.compartment] += 1
    if not counts: return None
    dominant = max(counts, key=counts.get)
    if dominant not in p_ids: return rxn
    h_id, w_id = p_ids[dominant], w_ids[dominant]
    if h_id is None: return rxn
    atom_res = np.zeros(K_atoms); charge_res = 0.0
    for sid, coef in rxn.stoichiometry.items():
        m = mols[sid]
        if m.atom_count is None: return None
        atom_res += coef * m.atom_count
        charge_res += coef * m.charge
    if (abs(atom_res[0]) > 1e-6 or abs(atom_res[3]) > 1e-6
        or abs(atom_res[4]) > 1e-6 or abs(atom_res[5]) > 1e-6):
        return None
    a_proton = -charge_res
    O_res = atom_res[2]; b_water = -O_res; H_res = atom_res[1]
    if abs(2 * b_water - (-H_res - a_proton)) > 1e-6:
        return None
    if abs(b_water) > 1e-6 and w_id is None:
        return None
    new_stoich = dict(rxn.stoichiometry)
    if abs(a_proton) > 1e-6:
        new_stoich[h_id] = new_stoich.get(h_id, 0.0) + a_proton
    if abs(b_water) > 1e-6 and w_id is not None:
        new_stoich[w_id] = new_stoich.get(w_id, 0.0) + b_water
    new_stoich = {k: v for k, v in new_stoich.items() if abs(v) > 1e-6}
    return SBMLReaction(rxn.sbml_id, rxn.name, new_stoich,
                         rxn.is_biomass, rxn.is_exchange)


# =============================================================================
# Main test
# =============================================================================
def main():
    print("=" * 76)
    print("DMVC P3b: compartment-aware stamps make transport visible")
    print("=" * 76)

    model = load_sbml_model(SBML_PATH)
    mols = extract_molecules_with_compartments(model)
    rxns = extract_reactions(model)
    print(f"\n{len(mols)} molecules, {len(rxns)} reactions")

    p_ids, w_ids = find_h_and_water_per_compartment(mols)
    print(f"proton per compartment: "
          + "  ".join(f"{c.name}={s}" for c, s in p_ids.items()))
    print(f"water  per compartment: "
          + "  ".join(f"{c.name}={s}" for c, s in w_ids.items()))

    rebalanced, skipped, failed = [], [], []
    for r in rxns:
        if r.is_exchange or r.is_biomass:
            skipped.append(r); continue
        nr = rebalance_reaction(r, mols, p_ids, w_ids)
        if nr is None: failed.append(r)
        else: rebalanced.append(nr)
    print(f"rebalance: {len(rebalanced)} OK, {len(failed)} failed, "
          f"{len(skipped)} skipped")

    by_kind = defaultdict(list)
    for r in rebalanced:
        by_kind[classify_by_compartment(r, mols)].append(r)
    print("by kind:", {k.name: len(v) for k, v in by_kind.items()})

    D = 192  # need D > K_STAMP = 21
    rng = np.random.default_rng(42)
    print(f"\n[building embeddings, D={D}, K_STAMP={K_STAMP}]")
    build_embeddings_compartment_aware(mols, D, rng)

    Nx = Ny = Nz = 16
    L = 1.0
    state = seed_state(Nx, Ny, Nz, D, L, rng)

    # -----------------------------------------------------------------------
    # T1: internal reaction has zero stamp contribution to OTHER compartments
    # -----------------------------------------------------------------------
    print("\n[T1] Internal cyto reaction: stamp is zero in EXTRA and MEMBRANE slices")
    rxn_int = by_kind[RxnKind.INTERNAL_CYTO][0]
    r_lat = build_reaction_latent(rxn_int, mols)
    # stamp slices for each compartment:
    extra_slice = r_lat[:K_atoms]
    mem_slice   = r_lat[K_atoms:2*K_atoms]
    cyto_slice  = r_lat[2*K_atoms:3*K_atoms]
    ext_q = r_lat[stamp_idx_for_charge(Compartment.EXTRA)]
    mem_q = r_lat[stamp_idx_for_charge(Compartment.MEMBRANE)]
    cyto_q = r_lat[stamp_idx_for_charge(Compartment.CYTO)]
    print(f"     reaction: {rxn_int.sbml_id}")
    print(f"     stamp[EXTRA atoms]: {extra_slice}, charge: {ext_q}")
    print(f"     stamp[MEM atoms]:   {mem_slice}, charge: {mem_q}")
    print(f"     stamp[CYTO atoms]:  {cyto_slice}, charge: {cyto_q}")
    t1 = (np.max(np.abs(extra_slice)) < 1e-9
          and np.max(np.abs(mem_slice)) < 1e-9
          and abs(ext_q) < 1e-9 and abs(mem_q) < 1e-9
          and np.max(np.abs(cyto_slice)) < 1e-9  # balanced cyto: cancels
          and abs(cyto_q) < 1e-9)
    print(f"     {'PASS' if t1 else 'FAIL'}")

    # -----------------------------------------------------------------------
    # T2: transport reaction has NON-ZERO compartment-local stamps that CANCEL
    #     across compartments (global balance)
    # -----------------------------------------------------------------------
    print("\n[T2] Transport reaction: non-zero per-compartment stamp, total zero")
    rxn_t = None
    # pick a transport with a readable stoichiometry pattern
    for r in by_kind[RxnKind.TRANSPORT]:
        if "pyr" in r.sbml_id.lower() or "glc" in r.sbml_id.lower() or "na" in r.sbml_id.lower():
            rxn_t = r; break
    if rxn_t is None and by_kind[RxnKind.TRANSPORT]:
        rxn_t = by_kind[RxnKind.TRANSPORT][0]
    if rxn_t is None:
        print("     SKIP no transport available")
        t2 = False
    else:
        print(f"     reaction: {rxn_t.sbml_id}")
        for sid, c in rxn_t.stoichiometry.items():
            m = mols[sid]
            print(f"       {c:+.1f} {sid}  ({m.compartment.name})")
        r_lat = build_reaction_latent(rxn_t, mols)
        extra_slice = r_lat[:K_atoms]
        cyto_slice = r_lat[2*K_atoms:3*K_atoms]
        ext_q = r_lat[stamp_idx_for_charge(Compartment.EXTRA)]
        cyto_q = r_lat[stamp_idx_for_charge(Compartment.CYTO)]
        total_atoms = extra_slice + r_lat[K_atoms:2*K_atoms] + cyto_slice
        print(f"     stamp[EXTRA atoms]: {extra_slice}, charge: {ext_q:+.3f}")
        print(f"     stamp[CYTO atoms]:  {cyto_slice}, charge: {cyto_q:+.3f}")
        print(f"     sum over compartments (should be 0): {total_atoms}")
        some_nonzero = (np.max(np.abs(extra_slice)) > 1e-6
                         or np.max(np.abs(cyto_slice)) > 1e-6)
        total_zero = np.max(np.abs(total_atoms)) < 1e-9 and abs(ext_q + cyto_q) < 1e-9
        t2 = some_nonzero and total_zero
        print(f"     per-compartment nonzero: {some_nonzero}   total zero: {total_zero}")
        print(f"     {'PASS' if t2 else 'FAIL'}")

    # -----------------------------------------------------------------------
    # T3: transport reaction VISIBLY moves atoms between compartments
    # -----------------------------------------------------------------------
    print("\n[T3] Applying transport reaction moves per-compartment atoms...")
    if rxn_t is not None:
        state3 = seed_state(Nx, Ny, Nz, D, L, rng)
        A_c_0 = atom_totals_by_compartment(state3)
        A_g_0 = atom_totals_global(state3)
        r_lat = build_reaction_latent(rxn_t, mols)
        rate = 0.5 * np.ones((Nx, Ny, Nz))
        for _ in range(20):
            apply_transport(state3, r_lat, rate, 0.01)
        A_c = atom_totals_by_compartment(state3)
        A_g = atom_totals_global(state3)
        dC = A_c[Compartment.CYTO] - A_c_0[Compartment.CYTO]
        dE = A_c[Compartment.EXTRA] - A_c_0[Compartment.EXTRA]
        dG = A_g - A_g_0
        print(f"     CYTO delta:   {[f'{v:+.3e}' for v in dC]}")
        print(f"     EXTRA delta:  {[f'{v:+.3e}' for v in dE]}")
        print(f"     GLOBAL delta: {[f'{v:+.3e}' for v in dG]}")
        some_moved = np.max(np.abs(dC)) > 1e-3 or np.max(np.abs(dE)) > 1e-3
        global_preserved = np.max(np.abs(dG)) < 1e-9
        t3 = some_moved and global_preserved
        print(f"     stuff moved: {some_moved}   global preserved: {global_preserved}")
        print(f"     {'PASS' if t3 else 'FAIL'}")
    else:
        t3 = False

    # -----------------------------------------------------------------------
    # T4: full evolution -- global conservation across mixed reaction types
    # -----------------------------------------------------------------------
    print("\n[T4] Full evolution over all reactions, many cycles, global conservation")
    state4 = seed_state(Nx, Ny, Nz, D, L, rng)
    A0 = atom_totals_global(state4); Q0 = charge_total_global(state4)
    rate_rng = np.random.default_rng(5)
    max_a_d = max_q_d = 0.0
    for cycle in range(3):
        for kind, rxn_list in by_kind.items():
            for rxn in rxn_list:
                r_lat = build_reaction_latent(rxn, mols)
                if r_lat is None: continue
                rate = 0.05 * rate_rng.standard_normal((Nx, Ny, Nz))
                if kind == RxnKind.INTERNAL_CYTO:
                    apply_internal(state4, r_lat, rate, Compartment.CYTO, 0.01)
                elif kind == RxnKind.INTERNAL_EXTRA:
                    apply_internal(state4, r_lat, rate, Compartment.EXTRA, 0.01)
                elif kind == RxnKind.TRANSPORT:
                    apply_transport(state4, r_lat, rate, 0.01)
                A = atom_totals_global(state4); Q = charge_total_global(state4)
                max_a_d = max(max_a_d, float(np.max(np.abs(A - A0))))
                max_q_d = max(max_q_d, float(abs(Q - Q0)))
    print(f"     max global atom drift:   {max_a_d:.3e}")
    print(f"     max global charge drift: {max_q_d:.3e}")
    t4 = max_a_d < 1e-9 and max_q_d < 1e-9
    print(f"     {'PASS' if t4 else 'FAIL'}")

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    results = [("T1", t1, "internal reaction has zero stamp everywhere (balanced)"),
                ("T2", t2, "transport reaction has non-zero per-compartment but zero total"),
                ("T3", t3, "transport visibly moves atoms, global preserved"),
                ("T4", t4, "full evolution preserves global atoms + charge")]
    for lab, ok, desc in results:
        print(f"  [{lab}] {'PASS' if ok else 'FAIL'}  {desc}")


if __name__ == "__main__":
    main()
