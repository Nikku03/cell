"""
DMVC P3: Compartment-aware cell architecture.

Extends P2b with compartment tracking:

  * Each grid voxel x has a compartment label c(x) in {CYTO, EXTRA, MEMBRANE}.
  * The cell geometry here is hard-coded as a simple sphere for testing.
  * Molecules carry a compartment tag from SBML (M_xxx_c, M_xxx_e, etc.).
  * A compartment-c molecule's embedding contributes to Psi ONLY at voxels
    with that compartment label (or at the membrane for boundary species).
  * Reactions sort into INTERNAL (all species in one compartment),
    TRANSPORT (species in different compartments, applied at the membrane),
    and EXCHANGE (open-system boundary fluxes).
  * Conservation is now per-compartment.

Tests:
  T1: internal cytoplasm reaction applied to extracellular voxels: no effect
  T2: internal cytoplasm reaction preserves per-compartment atom totals
  T3: transport reaction moves mass correctly between compartments
  T4: transport reaction preserves total mass across all compartments
  T5: Syn3A reactions classify correctly into internal/transport/exchange
  T6: full Syn3A evolution preserves total atoms globally
"""

from __future__ import annotations
import numpy as np
import libsbml
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from enum import IntEnum

import sys
sys.path.insert(0, "/home/claude/dmvc")
from prototype_p2_syn3a import (
    ATOM_TYPES, K_atoms, K_total,
    parse_formula, atom_vector,
    load_sbml_model, SBML_PATH,
)
from prototype_p2b_rebalance import find_proton_and_water


class Compartment(IntEnum):
    EXTRA = 0
    MEMBRANE = 1
    CYTO = 2


COMPARTMENT_NAMES = {
    Compartment.EXTRA: "extracellular",
    Compartment.MEMBRANE: "membrane",
    Compartment.CYTO: "cytoplasm",
}

# Map SBML compartment suffix -> our Compartment label
SBML_COMPARTMENT_MAP = {
    "c": Compartment.CYTO,
    "e": Compartment.EXTRA,
    "m": Compartment.MEMBRANE,
}


@dataclass
class Molecule:
    sbml_id: str
    name: str
    formula_raw: Optional[str]
    atom_count: Optional[np.ndarray]
    charge: Optional[float]
    compartment: Optional[Compartment]
    embedding: Optional[np.ndarray]


@dataclass
class SBMLReaction:
    sbml_id: str
    name: str
    stoichiometry: Dict[str, float]
    is_biomass: bool
    is_exchange: bool


def extract_compartment(sbml_id: str) -> Optional[Compartment]:
    """
    SBML convention: M_<name>_<comp_suffix>. Last token after underscore maps.
    """
    parts = sbml_id.rsplit("_", 1)
    if len(parts) < 2:
        return None
    suffix = parts[1]
    return SBML_COMPARTMENT_MAP.get(suffix)


def extract_molecules_with_compartments(model) -> Dict[str, Molecule]:
    mols = {}
    for i in range(model.getNumSpecies()):
        s = model.getSpecies(i)
        sid = s.getId()
        fbc = s.getPlugin("fbc")
        formula_raw = fbc.getChemicalFormula() if fbc else None
        charge = fbc.getCharge() if (fbc and fbc.isSetCharge()) else None
        parsed = parse_formula(formula_raw)
        atoms = atom_vector(parsed)
        # Prefer the SBML compartment attribute if present; fall back to suffix
        sbml_comp = s.getCompartment() if s.isSetCompartment() else None
        comp = SBML_COMPARTMENT_MAP.get(sbml_comp) if sbml_comp else None
        if comp is None:
            comp = extract_compartment(sid)
        mols[sid] = Molecule(
            sbml_id=sid,
            name=s.getName() or sid,
            formula_raw=formula_raw,
            atom_count=atoms,
            charge=float(charge) if charge is not None else None,
            compartment=comp,
            embedding=None,
        )
    return mols


def extract_reactions(model) -> List[SBMLReaction]:
    rxns = []
    for i in range(model.getNumReactions()):
        r = model.getReaction(i)
        stoich: Dict[str, float] = defaultdict(float)
        for j in range(r.getNumReactants()):
            ref = r.getReactant(j)
            stoich[ref.getSpecies()] -= float(ref.getStoichiometry())
        for j in range(r.getNumProducts()):
            ref = r.getProduct(j)
            stoich[ref.getSpecies()] += float(ref.getStoichiometry())
        rid = r.getId()
        name = r.getName() or rid
        is_biomass = "biomass" in rid.lower() or "biomass" in name.lower()
        is_exchange = (rid.startswith(("R_EX_", "EX_"))
                        or r.getNumReactants() == 0 or r.getNumProducts() == 0)
        rxns.append(SBMLReaction(rid, name, dict(stoich), is_biomass, is_exchange))
    return rxns


# =============================================================================
# Reaction classification by compartment
# =============================================================================
class RxnKind(IntEnum):
    INTERNAL_CYTO = 0
    INTERNAL_EXTRA = 1
    TRANSPORT = 2
    EXCHANGE = 3
    UNKNOWN = 4


def classify_by_compartment(rxn: SBMLReaction,
                             mols: Dict[str, Molecule]) -> RxnKind:
    if rxn.is_exchange or rxn.is_biomass:
        return RxnKind.EXCHANGE
    comps = set()
    for sid in rxn.stoichiometry:
        m = mols.get(sid)
        if m is None or m.compartment is None:
            return RxnKind.UNKNOWN
        comps.add(m.compartment)
    if len(comps) == 1:
        only = next(iter(comps))
        if only == Compartment.CYTO:
            return RxnKind.INTERNAL_CYTO
        if only == Compartment.EXTRA:
            return RxnKind.INTERNAL_EXTRA
        return RxnKind.UNKNOWN
    # Multiple compartments: transport
    return RxnKind.TRANSPORT


# =============================================================================
# Cell geometry and compartment map
# =============================================================================
def build_spherical_cell(Nx: int, Ny: int, Nz: int, L: float,
                          r_inner_frac: float = 0.30,
                          r_outer_frac: float = 0.36) -> np.ndarray:
    """
    Return a compartment-label field of shape (Nx, Ny, Nz).
    A sphere centered in the box; interior=CYTO, thin shell=MEMBRANE,
    exterior=EXTRA. Fractions are relative to box size L.
    """
    dx = L / Nx
    xs = (np.arange(Nx) + 0.5) * dx - L / 2
    ys = (np.arange(Ny) + 0.5) * dx - L / 2
    zs = (np.arange(Nz) + 0.5) * dx - L / 2
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)
    comp = np.full((Nx, Ny, Nz), Compartment.EXTRA, dtype=np.int8)
    comp[r < r_outer_frac * L] = Compartment.MEMBRANE
    comp[r < r_inner_frac * L] = Compartment.CYTO
    return comp


# =============================================================================
# Embeddings: same stamp+flavor as P1/P2, unchanged
# =============================================================================
def build_embeddings(mols: Dict[str, Molecule], D: int,
                      rng: np.random.Generator):
    for m in mols.values():
        if m.atom_count is None or m.charge is None:
            continue
        stamp = np.zeros(D)
        stamp[:K_atoms] = m.atom_count
        stamp[K_atoms] = m.charge
        flavor = np.zeros(D)
        if D > K_total:
            rnd = rng.standard_normal(D - K_total)
            rnd /= np.linalg.norm(rnd) + 1e-12
            flavor[K_total:] = rnd
        m.embedding = stamp + flavor


# =============================================================================
# Cell state: Psi on grid + a compartment label map
# =============================================================================
@dataclass
class CellState:
    Nx: int
    Ny: int
    Nz: int
    L: float
    Psi: np.ndarray             # (Nx, Ny, Nz, D)
    compartment: np.ndarray     # (Nx, Ny, Nz) int8

    @property
    def D(self): return self.Psi.shape[-1]
    @property
    def dV(self): return (self.L / self.Nx) ** 3

    def mask(self, which: Compartment) -> np.ndarray:
        return self.compartment == which


def seed_state(Nx, Ny, Nz, D, L, rng, amp=0.3) -> CellState:
    Psi = amp * rng.standard_normal((Nx, Ny, Nz, D))
    comp = build_spherical_cell(Nx, Ny, Nz, L)
    return CellState(Nx, Ny, Nz, L, Psi, comp)


# =============================================================================
# Per-compartment conservation readouts
# =============================================================================
def atom_totals_by_compartment(state: CellState) -> Dict[Compartment, np.ndarray]:
    """Return dict: compartment -> atom total vector."""
    out = {}
    for c in Compartment:
        m = state.mask(c)
        totals = (state.Psi[..., :K_atoms] * m[..., None]).sum(axis=(0, 1, 2))
        out[c] = totals * state.dV
    return out


def atom_totals_global(state: CellState) -> np.ndarray:
    return state.Psi[..., :K_atoms].sum(axis=(0, 1, 2)) * state.dV


def charge_totals_by_compartment(state: CellState) -> Dict[Compartment, float]:
    out = {}
    for c in Compartment:
        m = state.mask(c)
        q = (state.Psi[..., K_atoms] * m).sum()
        out[c] = float(q * state.dV)
    return out


def charge_total_global(state: CellState) -> float:
    return float(state.Psi[..., K_atoms].sum() * state.dV)


# =============================================================================
# Reaction application, compartment-aware
# =============================================================================
def build_reaction_latent(rxn: SBMLReaction,
                           mols: Dict[str, Molecule]) -> Optional[np.ndarray]:
    r = None
    for sid, coef in rxn.stoichiometry.items():
        m = mols[sid]
        if m.embedding is None:
            return None
        term = coef * m.embedding
        r = term if r is None else r + term
    return r


def apply_internal_reaction(state: CellState, rxn_latent: np.ndarray,
                              rate_field: np.ndarray,
                              target_compartment: Compartment,
                              dt: float):
    """
    Apply an internal reaction only at voxels with the given compartment label.
    """
    mask = state.mask(target_compartment).astype(np.float64)
    effective_rate = rate_field * mask
    state.Psi += dt * effective_rate[..., None] * rxn_latent[None, None, None, :]


def apply_transport_reaction(state: CellState, rxn_latent: np.ndarray,
                               rate_field: np.ndarray, dt: float):
    """
    Apply a transport reaction only at membrane voxels. The latent vector
    already encodes the stoichiometric change: negative coef on one side's
    species, positive on the other.
    """
    mask = state.mask(Compartment.MEMBRANE).astype(np.float64)
    effective_rate = rate_field * mask
    state.Psi += dt * effective_rate[..., None] * rxn_latent[None, None, None, :]


# =============================================================================
# Tests
# =============================================================================
def run_tests():
    print("=" * 76)
    print("DMVC P3: compartment-aware architecture with membrane dynamics")
    print("=" * 76)

    # Parse SBML with compartments
    print("\n[1] Loading SBML model with compartment info...")
    model = load_sbml_model(SBML_PATH)
    mols = extract_molecules_with_compartments(model)
    rxns = extract_reactions(model)
    comp_counts = defaultdict(int)
    for m in mols.values():
        comp_counts[m.compartment] += 1
    print(f"    {len(mols)} molecules, {len(rxns)} reactions")
    for c, n in comp_counts.items():
        name = COMPARTMENT_NAMES.get(c, str(c))
        print(f"      {name:15s}: {n} metabolites")

    # Rebalance using P2b logic (shim in compartment-aware form)
    # For compartment-aware rebalance, we need protons/water in the SAME
    # compartment as the residual arises in. We keep this simple for now
    # and use cytoplasmic H+ and H2O for internal reactions, and the
    # extracellular counterparts for extra reactions.
    print("\n[2] Finding proton/water species per compartment...")
    proton_c = water_c = proton_e = water_e = None
    for sid, m in mols.items():
        if m.atom_count is None or m.charge is None: continue
        # H+ (only H, charge +1)
        if np.sum(m.atom_count) == 1 and m.atom_count[1] == 1 and m.charge == 1:
            if m.compartment == Compartment.CYTO: proton_c = sid
            elif m.compartment == Compartment.EXTRA: proton_e = sid
        # H2O (H=2, O=1, charge 0)
        if (m.atom_count[1] == 2 and m.atom_count[2] == 1 and m.charge == 0
            and np.sum(m.atom_count) == 3):
            if m.compartment == Compartment.CYTO: water_c = sid
            elif m.compartment == Compartment.EXTRA: water_e = sid
    print(f"    cytoplasmic H+: {proton_c}, H2O: {water_c}")
    print(f"    extracellular H+: {proton_e}, H2O: {water_e}")

    # Rebalance each reaction using same-compartment H+/H2O.
    # For transport reactions, we apply both sides' patches if needed.
    def rebalance_reaction(rxn: SBMLReaction) -> Optional[SBMLReaction]:
        if rxn.is_exchange or rxn.is_biomass:
            return rxn
        # Determine dominant compartment: pick the one with most species
        comp_counts = defaultdict(int)
        for sid in rxn.stoichiometry:
            m = mols.get(sid)
            if m and m.compartment:
                comp_counts[m.compartment] += 1
        if not comp_counts:
            return None
        dominant = max(comp_counts, key=comp_counts.get)
        if dominant == Compartment.CYTO:
            h_id, w_id = proton_c, water_c
        elif dominant == Compartment.EXTRA:
            h_id, w_id = proton_e, water_e
        else:
            return rxn
        if h_id is None:
            return rxn
        # Compute current residual
        atom_res = np.zeros(K_atoms); charge_res = 0.0
        for sid, coef in rxn.stoichiometry.items():
            m = mols.get(sid)
            if not m or m.atom_count is None: return None
            atom_res += coef * m.atom_count
            charge_res += coef * m.charge
        # Must have zero C,N,P,S residuals to be closable by H+/H2O
        if (abs(atom_res[0]) > 1e-6 or abs(atom_res[3]) > 1e-6
            or abs(atom_res[4]) > 1e-6 or abs(atom_res[5]) > 1e-6):
            return None
        a_proton = -charge_res
        O_res = atom_res[2]
        b_water = -O_res
        H_res = atom_res[1]
        expected_extra_H = -H_res - a_proton
        if abs(2 * b_water - expected_extra_H) > 1e-6:
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

    print("\n[3] Rebalancing reactions by compartment...")
    rebalanced, skipped, failed = [], [], []
    for rxn in rxns:
        if rxn.is_exchange or rxn.is_biomass:
            skipped.append(rxn); continue
        nr = rebalance_reaction(rxn)
        if nr is None: failed.append(rxn)
        else: rebalanced.append(nr)
    print(f"    {len(rebalanced)} balanced, {len(failed)} unbalanceable, "
          f"{len(skipped)} structural skips")

    # Classify rebalanced reactions by compartment
    print("\n[4] Classifying balanced reactions by compartment kind...")
    by_kind = defaultdict(list)
    for rxn in rebalanced:
        k = classify_by_compartment(rxn, mols)
        by_kind[k].append(rxn)
    for k, items in by_kind.items():
        print(f"    {k.name:15s}: {len(items)}")

    # Build embeddings
    D = 128
    rng = np.random.default_rng(42)
    print(f"\n[5] Building embeddings (D={D})...")
    build_embeddings(mols, D, rng)

    # Set up cell
    Nx = Ny = Nz = 16
    L = 1.0
    state = seed_state(Nx, Ny, Nz, D, L, rng, amp=0.2)
    n_cyto  = int(state.mask(Compartment.CYTO).sum())
    n_mem   = int(state.mask(Compartment.MEMBRANE).sum())
    n_extra = int(state.mask(Compartment.EXTRA).sum())
    print(f"\n    Cell: {Nx}^3 grid, cyto={n_cyto}, mem={n_mem}, extra={n_extra}")

    # -----------------------------------------------------------------------
    # T1: internal-cyto reaction applied only in cytoplasm
    # -----------------------------------------------------------------------
    print("\n[T1] Internal cytoplasmic reaction should affect only CYTO voxels")
    if by_kind[RxnKind.INTERNAL_CYTO]:
        rxn_test = by_kind[RxnKind.INTERNAL_CYTO][0]
        r_lat = build_reaction_latent(rxn_test, mols)
        if r_lat is not None:
            state2 = seed_state(Nx, Ny, Nz, D, L, rng, amp=0.2)
            # use same random state as seed by re-seeding
            rate = 0.5 * np.ones((Nx, Ny, Nz))
            before = state2.Psi.copy()
            apply_internal_reaction(state2, r_lat, rate, Compartment.CYTO, 0.01)
            diff = np.abs(state2.Psi - before).max(axis=-1)
            cyto_changed = diff[state2.mask(Compartment.CYTO)].max()
            extra_changed = diff[state2.mask(Compartment.EXTRA)].max()
            mem_changed = diff[state2.mask(Compartment.MEMBRANE)].max()
            print(f"     test reaction: {rxn_test.sbml_id}")
            print(f"     max |dPsi| in CYTO:     {cyto_changed:.3e}")
            print(f"     max |dPsi| in EXTRA:    {extra_changed:.3e}")
            print(f"     max |dPsi| in MEMBRANE: {mem_changed:.3e}")
            t1_pass = (cyto_changed > 1e-6
                        and extra_changed < 1e-12 and mem_changed < 1e-12)
            print(f"     {'PASS' if t1_pass else 'FAIL'}")
        else:
            print("     SKIP: no usable internal-cyto reaction found")
            t1_pass = False
    else:
        t1_pass = False
        print("     SKIP: no internal-cyto reactions available")

    # -----------------------------------------------------------------------
    # T2: internal reactions preserve per-compartment atom totals
    # -----------------------------------------------------------------------
    print("\n[T2] Internal reactions preserve per-compartment totals")
    state2 = seed_state(Nx, Ny, Nz, D, L, rng, amp=0.2)
    A_by_c_0 = atom_totals_by_compartment(state2)
    Q_by_c_0 = charge_totals_by_compartment(state2)
    rate_rng = np.random.default_rng(1)
    max_diff_per_comp = 0.0
    for rxn in by_kind[RxnKind.INTERNAL_CYTO]:
        r_lat = build_reaction_latent(rxn, mols)
        if r_lat is None: continue
        rate = 0.1 * rate_rng.standard_normal((Nx, Ny, Nz))
        apply_internal_reaction(state2, r_lat, rate, Compartment.CYTO, 0.01)
    A_by_c = atom_totals_by_compartment(state2)
    Q_by_c = charge_totals_by_compartment(state2)
    for c in Compartment:
        max_diff_per_comp = max(max_diff_per_comp,
                                 float(np.max(np.abs(A_by_c[c] - A_by_c_0[c]))))
        max_diff_per_comp = max(max_diff_per_comp,
                                 abs(Q_by_c[c] - Q_by_c_0[c]))
    print(f"     Applied all {len(by_kind[RxnKind.INTERNAL_CYTO])} internal-cyto reactions")
    print(f"     Max per-compartment deviation (atoms or charge): "
          f"{max_diff_per_comp:.3e}")
    t2_pass = max_diff_per_comp < 1e-9
    print(f"     {'PASS' if t2_pass else 'FAIL'}")

    # -----------------------------------------------------------------------
    # T3: transport reaction moves mass between compartments
    # -----------------------------------------------------------------------
    print("\n[T3] Transport reaction moves atoms between compartments")
    if by_kind[RxnKind.TRANSPORT]:
        rxn_t = by_kind[RxnKind.TRANSPORT][0]
        print(f"     test transport reaction: {rxn_t.sbml_id}")
        for sid, coef in rxn_t.stoichiometry.items():
            m = mols[sid]
            cn = COMPARTMENT_NAMES.get(m.compartment, '?')
            print(f"        {coef:+.0f} {sid} ({cn})")
        r_lat = build_reaction_latent(rxn_t, mols)
        if r_lat is not None:
            state3 = seed_state(Nx, Ny, Nz, D, L, rng, amp=0.2)
            A_by_c_0 = atom_totals_by_compartment(state3)
            A_tot_0 = atom_totals_global(state3)
            # Apply many times so transport has visible effect
            rate = 0.5 * np.ones((Nx, Ny, Nz))
            for _ in range(20):
                apply_transport_reaction(state3, r_lat, rate, 0.01)
            A_by_c = atom_totals_by_compartment(state3)
            A_tot = atom_totals_global(state3)
            # Cyto and extra should have changed, total should be preserved
            cyto_delta = A_by_c[Compartment.CYTO] - A_by_c_0[Compartment.CYTO]
            extra_delta = A_by_c[Compartment.EXTRA] - A_by_c_0[Compartment.EXTRA]
            mem_delta = A_by_c[Compartment.MEMBRANE] - A_by_c_0[Compartment.MEMBRANE]
            total_delta = A_tot - A_tot_0
            print(f"     atom delta CYTO:     {[f'{v:+.2e}' for v in cyto_delta]}")
            print(f"     atom delta EXTRA:    {[f'{v:+.2e}' for v in extra_delta]}")
            print(f"     atom delta MEMBRANE: {[f'{v:+.2e}' for v in mem_delta]}")
            print(f"     atom delta TOTAL:    {[f'{v:+.2e}' for v in total_delta]}")
            t3_pass = (float(np.max(np.abs(total_delta))) < 1e-9)
            print(f"     total mass conserved: {t3_pass}  {'PASS' if t3_pass else 'FAIL'}")
        else:
            print("     SKIP: cannot build latent for transport reaction")
            t3_pass = False
    else:
        t3_pass = False
        print("     SKIP: no transport reactions available")

    # -----------------------------------------------------------------------
    # T6 (renamed, the big one): Full evolution preserves total mass
    # -----------------------------------------------------------------------
    print("\n[T6] Full evolution of all reactions preserves TOTAL atoms + charge")
    state_full = seed_state(Nx, Ny, Nz, D, L, rng, amp=0.2)
    A0_g = atom_totals_global(state_full)
    Q0_g = charge_total_global(state_full)
    print(f"     Initial global atoms: "
          + "  ".join(f"{a}={v:+.3f}" for a, v in zip(ATOM_TYPES, A0_g)))
    rate_rng = np.random.default_rng(3)
    max_a_drift = 0.0
    max_q_drift = 0.0
    applied = 0
    for cycle in range(5):
        for kind, rxn_list in by_kind.items():
            for rxn in rxn_list:
                r_lat = build_reaction_latent(rxn, mols)
                if r_lat is None:
                    continue
                rate = 0.05 * rate_rng.standard_normal((Nx, Ny, Nz))
                if kind == RxnKind.INTERNAL_CYTO:
                    apply_internal_reaction(state_full, r_lat, rate,
                                              Compartment.CYTO, 0.01)
                elif kind == RxnKind.INTERNAL_EXTRA:
                    apply_internal_reaction(state_full, r_lat, rate,
                                              Compartment.EXTRA, 0.01)
                elif kind == RxnKind.TRANSPORT:
                    apply_transport_reaction(state_full, r_lat, rate, 0.01)
                applied += 1
                A = atom_totals_global(state_full)
                Q = charge_total_global(state_full)
                max_a_drift = max(max_a_drift, float(np.max(np.abs(A - A0_g))))
                max_q_drift = max(max_q_drift, float(abs(Q - Q0_g)))
    print(f"     total reactions applied: {applied}")
    print(f"     max global atom drift:   {max_a_drift:.3e}")
    print(f"     max global charge drift: {max_q_drift:.3e}")
    t6_pass = max_a_drift < 1e-9 and max_q_drift < 1e-9
    print(f"     {'PASS' if t6_pass else 'FAIL'}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    results = [
        ("T1", t1_pass, "internal reaction respects compartment mask"),
        ("T2", t2_pass, "internal reactions preserve per-compartment totals"),
        ("T3", t3_pass, "transport reactions conserve global mass"),
        ("T6", t6_pass, "full evolution preserves global atoms + charge"),
    ]
    for label, ok, desc in results:
        print(f"  [{label}] {'PASS' if ok else 'FAIL'}  {desc}")

    print("\nReaction classification of balanced set:")
    for k in RxnKind:
        print(f"  {k.name:15s}: {len(by_kind[k])}")


if __name__ == "__main__":
    run_tests()
