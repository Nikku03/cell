"""
DMVC P2b: Auto-rebalance Syn3A reactions.

P2 found that 159 of 244 Syn3A reactions are "unbalanced" in the SBML file.
Investigation showed the imbalances are mostly of specific patterns:

  (H=-1, charge=-1): a proton is missing as a product.
  (H=+1, charge=+1): a proton is missing as a reactant.
  (H=-2, O=-1):      a water is missing as a product.
  (H=+2, O=+1):      a water is missing as a reactant.

These are stylistic choices in the Luthey-Schulten SBML (charged-form
metabolites + implicit proton tracking + "No H2O" model variant).
They are NOT curation errors in the underlying chemistry.

Strategy: systematically add H+ and/or H2O coefficients to close the balance.
After this step, every reaction should satisfy:
    sum_m nu_m * atom_m = 0 (per atom)
    sum_m nu_m * charge_m = 0
structurally, and then our P1 architecture handles them with zero drift.

Tests:
  (a) How many additional reactions become balanced after adding H+ and H2O?
  (b) Do the rebalanced reactions still pass the conservation test on the
      cell-state field?
  (c) How many genuinely-unbalanced reactions remain (not closable by H+/H2O)?
"""

from __future__ import annotations
import numpy as np
import libsbml
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Reuse the P2 module
import sys
sys.path.insert(0, "/home/claude/dmvc")
from prototype_p2_syn3a import (
    ATOM_TYPES, K_atoms, K_total,
    parse_formula, atom_vector, Molecule, SBMLReaction,
    load_sbml_model, extract_molecules, extract_reactions,
    reaction_residuals, build_embeddings,
    CellState, seed_state, atom_totals, charge_total,
    apply_reaction_field, build_reaction_latent,
    SBML_PATH,
)


def find_proton_and_water(mols: Dict[str, Molecule]) -> Tuple[Optional[str], Optional[str]]:
    """Identify the proton and water species by their formulas+charges."""
    proton_id = None
    water_id = None
    for sid, m in mols.items():
        if m.atom_count is None or m.charge is None:
            continue
        # Proton: H=1 only, charge=+1
        if (m.atom_count[1] == 1 and m.charge == 1
            and np.sum(m.atom_count) == 1):
            proton_id = sid
        # Water: H=2, O=1 only, charge=0
        if (m.atom_count[1] == 2 and m.atom_count[2] == 1 and m.charge == 0
            and np.sum(m.atom_count) == 3):
            water_id = sid
    return proton_id, water_id


def try_rebalance(rxn: SBMLReaction, mols: Dict[str, Molecule],
                   proton_id: Optional[str], water_id: Optional[str],
                   tol: float = 1e-6) -> Optional[SBMLReaction]:
    """
    Try to balance rxn by adding H+ and H2O species.

    Algorithm:
      1. Compute current atom residual (C, H, O, N, P, S) and charge residual.
      2. If C, N, P, S residuals are all ~0, the reaction is closeable by
         H+/H2O (which only have C=N=P=S=0). Proceed.
      3. Solve for coefficients a (H+) and b (H2O):
            a * [0, 1, 0, 0, 0, 0] + b * [0, 2, 1, 0, 0, 0] = -atom_res
            a * (+1)               + b * 0                  = -charge_res
         i.e., a = -charge_res, then from H-balance:
            a + 2b = -H_res
            b = (-H_res - a) / 2
         Then check O-balance: b = -O_res (since O_res should only come from H2O).
      4. If the integer/fractional coefficients work, add them.
    """
    atom_res, charge_res, bad = reaction_residuals(rxn, mols)
    if bad:
        return None
    # already balanced?
    if np.max(np.abs(atom_res)) < tol and abs(charge_res) < tol:
        return rxn
    # Must have zero residual in C, N, P, S to be closeable by H+/H2O
    if (abs(atom_res[0]) > tol or abs(atom_res[3]) > tol
        or abs(atom_res[4]) > tol or abs(atom_res[5]) > tol):
        return None
    # Determine coefficients
    if proton_id is None or water_id is None:
        return None
    a_proton = -charge_res
    H_res = atom_res[1]
    O_res = atom_res[2]
    b_water = -O_res     # water is the only O source in our two-species patch
    implied_H_from_b = 2 * b_water
    expected_H_remaining = -H_res - a_proton
    if abs(implied_H_from_b - expected_H_remaining) > tol:
        return None  # not closeable by H+ and H2O alone
    # Apply the new coefficients to a copy of the stoichiometry
    new_stoich = dict(rxn.stoichiometry)
    if abs(a_proton) > tol:
        new_stoich[proton_id] = new_stoich.get(proton_id, 0.0) + a_proton
    if abs(b_water) > tol:
        new_stoich[water_id] = new_stoich.get(water_id, 0.0) + b_water
    # Clean zero entries
    new_stoich = {k: v for k, v in new_stoich.items() if abs(v) > tol}
    return SBMLReaction(sbml_id=rxn.sbml_id, name=rxn.name,
                         stoichiometry=new_stoich,
                         is_biomass=rxn.is_biomass, is_exchange=rxn.is_exchange)


def main():
    print("=" * 76)
    print("DMVC P2b: auto-rebalance Syn3A reactions by adding H+/H2O")
    print("=" * 76)

    print("\n[1] Loading model...")
    model = load_sbml_model(SBML_PATH)
    mols = extract_molecules(model)
    rxns = extract_reactions(model)
    print(f"    {len(mols)} molecules, {len(rxns)} reactions")

    print("\n[2] Identifying proton and water species...")
    proton_id, water_id = find_proton_and_water(mols)
    print(f"    Proton species: {proton_id}  ({mols[proton_id].name if proton_id else '-'})")
    print(f"    Water  species: {water_id}  ({mols[water_id].name if water_id else '-'})")
    if proton_id is None:
        print("    ERROR: cannot find a proton species (H+, charge +1); aborting.")
        return
    if water_id is None:
        print("    WARNING: no water species found in model.")
        print("    (Note: this is the 'NoH2O' variant; water may have been removed.")
        print("     We'll add water as a virtual species for rebalancing purposes.)")
        # Synthesize a virtual water entry for rebalancing
        virtual_water = Molecule(
            sbml_id="M_h2o_virtual",
            name="water (virtual, added for rebalancing)",
            formula_raw="H2O",
            formula_parsed={"H": 2, "O": 1},
            atom_count=np.array([0, 2, 1, 0, 0, 0], dtype=np.float64),
            charge=0.0,
            embedding=None,
            other_elements={},
        )
        mols["M_h2o_virtual"] = virtual_water
        water_id = "M_h2o_virtual"
        print(f"    Added virtual water: {water_id}")

    print("\n[3] Attempting to rebalance each unbalanced reaction...")
    rebalanced = []
    still_unbalanced = []
    skipped_structural = []
    for rxn in rxns:
        if rxn.is_biomass or rxn.is_exchange:
            skipped_structural.append(rxn)
            continue
        attempt = try_rebalance(rxn, mols, proton_id, water_id)
        if attempt is None:
            still_unbalanced.append(rxn)
        else:
            rebalanced.append(attempt)

    print(f"    Balanced (or rebalanced to balanced): {len(rebalanced)}")
    print(f"    Still unbalanced:                     {len(still_unbalanced)}")
    print(f"    Structural skip (biomass/exchange):   {len(skipped_structural)}")

    # What remained unbalanced -- examine
    if still_unbalanced:
        print(f"\n    First 5 reactions that remain unbalanced:")
        for rxn in still_unbalanced[:5]:
            atom_res, charge_res, bad = reaction_residuals(rxn, mols)
            res_str = ", ".join(
                f"{a}={v:+.0f}" for a, v in zip(ATOM_TYPES, atom_res)
                if abs(v) > 1e-6
            )
            if abs(charge_res) > 1e-6:
                res_str += f", charge={charge_res:+.1f}"
            # Also list non-CHONPS elements involved
            nonstd = set()
            for sid in rxn.stoichiometry:
                m = mols.get(sid)
                if m and m.other_elements:
                    nonstd.update(m.other_elements.keys())
            nonstd_str = f"  (involves: {sorted(nonstd)})" if nonstd else ""
            print(f"      {rxn.sbml_id:15s}  {res_str}{nonstd_str}")

    # -----------------------------------------------------------------------
    # Build embeddings and run conservation test on the rebalanced set
    # -----------------------------------------------------------------------
    D = 128
    rng = np.random.default_rng(42)
    print(f"\n[4] Building embeddings (D={D})...")
    build_embeddings(mols, D, rng)

    mol_id_to_idx = {sid: i for i, sid in enumerate(mols.keys())}
    latent_vecs = []
    max_stamp_err = 0.0
    for rxn in rebalanced:
        r_lat = build_reaction_latent(rxn, mols, mol_id_to_idx)
        if r_lat is None:
            continue
        latent_vecs.append((rxn, r_lat))
        max_stamp_err = max(max_stamp_err, float(np.max(np.abs(r_lat[:K_total]))))

    print(f"    {len(latent_vecs)} reactions have complete embeddings")
    print(f"    Max stamp component across all: {max_stamp_err:.3e} (should be ~0)")

    # -----------------------------------------------------------------------
    # Conservation test on the rebalanced set
    # -----------------------------------------------------------------------
    print("\n[5] Conservation test: 10 cycles through all rebalanced reactions...")
    Nx = Ny = Nz = 8
    state = seed_state(Nx, Ny, Nz, D, rng)
    A0 = atom_totals(state)
    Q0 = charge_total(state)

    rate_rng = np.random.default_rng(7)
    dt = 0.01
    max_atom_drift = 0.0
    max_charge_drift = 0.0
    for cycle in range(10):
        for rxn, r_lat in latent_vecs:
            rate = 0.1 * rate_rng.standard_normal((Nx, Ny, Nz))
            apply_reaction_field(state, r_lat, rate, dt)
        A = atom_totals(state)
        Q = charge_total(state)
        max_atom_drift = max(max_atom_drift, float(np.max(np.abs(A - A0))))
        max_charge_drift = max(max_charge_drift, float(abs(Q - Q0)))

    n_applied = 10 * len(latent_vecs)
    print(f"    Reactions applied: {n_applied}")
    print(f"    Max atom drift:    {max_atom_drift:.3e}")
    print(f"    Max charge drift:  {max_charge_drift:.3e}")
    conservation_ok = max_atom_drift < 1e-9 and max_charge_drift < 1e-9
    print(f"    {'PASS' if conservation_ok else 'FAIL'}")

    # -----------------------------------------------------------------------
    # Category analysis of what remains unbalanced
    # -----------------------------------------------------------------------
    print("\n[6] What elements appear in still-unbalanced reactions?")
    element_counter = defaultdict(int)
    for rxn in still_unbalanced:
        atom_res, charge_res, _ = reaction_residuals(rxn, mols)
        for i, atom in enumerate(ATOM_TYPES):
            if abs(atom_res[i]) > 1e-6:
                element_counter[atom] += 1
        if abs(charge_res) > 1e-6:
            element_counter["charge"] += 1
        for sid in rxn.stoichiometry:
            m = mols.get(sid)
            if m and m.other_elements:
                for e in m.other_elements:
                    element_counter[f"other:{e}"] += 1
    for elem, n in sorted(element_counter.items(), key=lambda x: -x[1]):
        print(f"    {elem:15s} appears in {n} still-unbalanced reactions")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    recovered = len(rebalanced) - sum(
        1 for rxn in rebalanced
        if np.max(np.abs(reaction_residuals(rxn, mols)[0])) < 1e-6
        and abs(reaction_residuals(rxn, mols)[1]) < 1e-6
    )  # this reports zero but just for clarity
    print(f"Total reactions:              {len(rxns)}")
    print(f"Structural skip:              {len(skipped_structural)}")
    print(f"Balanced after H+/H2O patch:  {len(rebalanced)}")
    print(f"Still unbalanced:             {len(still_unbalanced)}")
    print(f"Fraction usable:              "
          f"{len(rebalanced) / (len(rxns) - len(skipped_structural)) * 100:.1f}%")
    print(f"Conservation on patched set:  "
          f"{'PASS (drift=' + f'{max_atom_drift:.2e})' if conservation_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
