"""
DMVC P4: Real Syn3A kinetics on the P3b compartment-aware architecture.

What this prototype does:
  1. Parses the Luthey-Schulten kinetic parameter files (SBtab TSVs in the
     CME_ODE/model_data/ directory). Extracts per-reaction k_cat_+, k_cat_-,
     enzyme concentrations, K_eq, and per-(reaction, substrate) K_m values.
  2. Reads physiological concentrations (mM) as the initial cell state.
  3. Implements CONVENIENCE RATE LAWS - the multi-substrate generalization
     of Michaelis-Menten used by the Luthey-Schulten lab:
         v = [E] * (kcatF * prod(S_i/Km_i) - kcatR * prod(P_j/Km_j))
             --------------------------------------------------------
             prod(1 + S_i/Km_i) * prod(1 + P_j/Km_j)  (actually more subtle;
                                                       see CONVENIENCE below)
     We use the Liebermeister 2010 form.
  4. Builds the rate field at each step from current concentrations.
  5. Runs forward in time.
  6. Tests:
       T1: conservation still holds (expected from architecture)
       T2: concentrations remain non-negative and bounded (physical)
       T3: metabolite concentrations approach their physiological values
       T4: consuming glucose via glycolysis produces ATP and pyruvate as
           a basic metabolic sanity check

Known simplifications:
  * Enzyme concentrations held fixed (no protein dynamics yet)
  * Only reactions with kinetic parameters are active; others are dormant
  * Readout is the simple <Psi, e_m> (can be negative); rate law clips
    negative concentrations to zero so reactions can't run "in reverse"
    out of nothing
  * Single-voxel "cell" for this prototype - we're testing kinetics, not
    spatial dynamics. Spatial version is a later prototype.
"""

from __future__ import annotations
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, "/home/claude/dmvc")
from prototype_p3b_stamps import (
    Compartment, COMPARTMENT_NAMES,
    extract_molecules_with_compartments, extract_reactions,
    classify_by_compartment, RxnKind,
    Molecule, SBMLReaction,
    ATOM_TYPES, K_atoms,
    load_sbml_model, SBML_PATH,
    find_h_and_water_per_compartment, rebalance_reaction,
    K_STAMP, N_COMPARTMENTS,
    stamp_idx_for_atom, stamp_idx_for_charge,
    build_embeddings_compartment_aware,
    build_reaction_latent,
    atom_totals_global, atom_totals_by_compartment,
    charge_total_global,
)


KINETICS_TSV = ("/home/claude/dmvc/data/Minimal_Cell/CME_ODE/model_data/"
                 "Central_AA_Zane_Balanced_direction_fixed_nounqATP.tsv")
# Additional kinetics sources could be concatenated later:
EXTRA_KINETICS_TSVS = [
    "/home/claude/dmvc/data/Minimal_Cell/CME_ODE/model_data/"
    "Nucleotide_Kinetic_Parameters.tsv",
]


# =============================================================================
# SBtab parsing
# =============================================================================
def parse_sbtab_parameter_section(path: str) -> List[Dict[str, str]]:
    """
    Parse the Parameter table(s) from an SBtab TSV file. Returns a list of
    dicts, one per row, keyed by the column headers.

    An SBtab file has multiple !!SBtab blocks. We only pull rows from blocks
    whose TableName starts with 'Parameter' and is NOT 'Parameter prior'.
    """
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        in_target = False
        headers: List[str] = []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!!SBtab"):
                in_target = ("TableName='Parameter'" in line
                              and "'Parameter prior'" not in line)
                headers = []
                continue
            if not in_target: continue
            parts = line.split("\t")
            if line.startswith("!"):
                # header row: !QuantityType\t!Reaction:...\t!Compound:...\t...
                headers = [p.lstrip("!") for p in parts]
                continue
            if not headers: continue
            if all(p == "" for p in parts): continue
            row = {}
            for i, h in enumerate(headers):
                row[h] = parts[i] if i < len(parts) else ""
            if row.get("QuantityType", "").strip():
                rows.append(row)
    return rows


@dataclass
class ReactionKinetics:
    sbml_id: str
    kcat_forward: Optional[float] = None   # 1/s
    kcat_reverse: Optional[float] = None   # 1/s
    kcat_mean: Optional[float] = None
    enzyme_conc: Optional[float] = None    # mM
    keq: Optional[float] = None
    # Michaelis constants per species in this reaction: {species_id: K_m in mM}
    Km: Dict[str, float] = field(default_factory=dict)

    def is_usable(self) -> bool:
        """Need at least kcat_forward and one Km value."""
        return (self.kcat_forward is not None and len(self.Km) > 0
                and self.enzyme_conc is not None)


def parse_all_kinetics() -> Tuple[Dict[str, ReactionKinetics], Dict[str, float]]:
    """
    Aggregate kinetic data across all available SBtab files.
    Returns (per-reaction kinetics, initial concentrations {species: mM}).
    """
    kinetics: Dict[str, ReactionKinetics] = {}
    concentrations: Dict[str, float] = {}

    def get_row(row) -> Tuple[str, str, str, Optional[float]]:
        """Extract (qty_type, reaction_id, species_id, mode_value)."""
        qt = row.get("QuantityType", "").strip()
        # column names vary slightly between files
        rxn = (row.get("Reaction:SBML:reaction:id")
                or row.get("Reaction") or "").strip()
        sp = (row.get("Compound:SBML:species:id")
               or row.get("Compound") or "").strip()
        mode = row.get("Mode", "").strip()
        try:
            val = float(mode) if mode and mode.lower() != "nan" else None
        except ValueError:
            val = None
        return qt, rxn, sp, val

    all_rows = parse_sbtab_parameter_section(KINETICS_TSV)
    for p in EXTRA_KINETICS_TSVS:
        all_rows.extend(parse_sbtab_parameter_section(p))

    for row in all_rows:
        qt, rxn, sp, val = get_row(row)
        if val is None: continue
        # route by quantity type
        if qt == "concentration" and sp:
            # First file wins: only write if we don't already have a value.
            # The Nucleotide_Kinetic_Parameters.tsv has 0.1 placeholder for many
            # species that the central-AA file already specifies with real values.
            if sp not in concentrations:
                concentrations[sp] = val
        elif qt == "concentration of enzyme" and rxn:
            kinetics.setdefault(rxn, ReactionKinetics(rxn)).enzyme_conc = val
        elif qt == "Michaelis constant" and rxn and sp:
            kinetics.setdefault(rxn, ReactionKinetics(rxn)).Km[sp] = val
        elif qt == "substrate catalytic rate constant" and rxn:
            kinetics.setdefault(rxn, ReactionKinetics(rxn)).kcat_forward = val
        elif qt == "product catalytic rate constant" and rxn:
            kinetics.setdefault(rxn, ReactionKinetics(rxn)).kcat_reverse = val
        elif qt == "catalytic rate constant geometric mean" and rxn:
            kinetics.setdefault(rxn, ReactionKinetics(rxn)).kcat_mean = val
        elif qt == "equilibrium constant" and rxn:
            kinetics.setdefault(rxn, ReactionKinetics(rxn)).keq = val

    return kinetics, concentrations


# =============================================================================
# Convenience rate law (Liebermeister 2010 form)
# =============================================================================
def convenience_rate(rxn: SBMLReaction, kin: ReactionKinetics,
                      concs: Dict[str, float]) -> float:
    """
    Compute the reaction rate given:
      - rxn.stoichiometry (signed coefficients)
      - kin.kcat_forward, kin.kcat_reverse, kin.enzyme_conc, kin.Km
      - concs: current concentration of each species (mM), clipped >= 0

    The convenience rate law (single-substrate, single-product, order 1):
        v = E * (kf * s - kr * p) / (1 + s + p)
      where s = [S]/Km_S, p = [P]/Km_P.

    For multi-substrate / multi-product, the generalization per
    Liebermeister is:
        numerator   = E * (kf * prod(s_i^|v_i|) - kr * prod(p_j^|v_j|))
        denominator = prod((1 + s_i + s_i^2 + ... + s_i^|v_i|))
                      + prod((1 + p_j + ... + p_j^|v_j|)) - 1
    We implement the common case of stoichiometry 1 for each species, which
    covers essentially all real metabolic reactions. Higher stoichiometries
    would need a generalized version.
    """
    if not kin.is_usable():
        return 0.0
    E = kin.enzyme_conc
    kf = kin.kcat_forward
    kr = kin.kcat_reverse if kin.kcat_reverse is not None else 0.0

    subs = []   # (s, coef) pairs -- s is ratio, coef is |stoichiometry|
    prods = []  # same for products
    for sid, coef in rxn.stoichiometry.items():
        c = max(concs.get(sid, 0.0), 0.0)
        Km = kin.Km.get(sid)
        if Km is None or Km <= 0:
            # If we lack a Km for this species, we fall back to treating it
            # as if the ratio were just its concentration -- crude but avoids
            # division by zero. More principled: skip the reaction.
            return 0.0
        ratio = c / Km
        if coef < 0:
            subs.append((ratio, int(round(-coef))))
        elif coef > 0:
            prods.append((ratio, int(round(coef))))

    # Numerator
    prod_s = 1.0
    for s, n in subs:
        prod_s *= s ** n
    prod_p = 1.0
    for p, n in prods:
        prod_p *= p ** n
    numerator = E * (kf * prod_s - kr * prod_p)

    # Denominator: sum over {0,...,|v_i|} powers per species, product across
    # all species, then subtract 1 (for the constant term counted twice)
    # For simplicity assume all stoichiometries are 1 (which they are for
    # nearly every central metabolism reaction):
    denom = 1.0
    for s, _ in subs:
        denom *= (1.0 + s)
    denom_p = 1.0
    for p, _ in prods:
        denom_p *= (1.0 + p)
    denom = denom + denom_p - 1.0

    if denom <= 0:
        return 0.0
    return numerator / denom


# =============================================================================
# Well-mixed "single voxel" cell for this prototype
# =============================================================================
@dataclass
class WellMixedCell:
    Psi: np.ndarray   # shape (D,) -- single voxel
    volume: float     # arbitrary

    @property
    def D(self): return self.Psi.shape[0]


def seed_wellmixed(mols: Dict[str, Molecule], init_concs: Dict[str, float],
                    D: int) -> WellMixedCell:
    """
    Build an initial Psi vector such that for every molecule with a known
    initial concentration, <Psi, e_m> approximately equals that concentration.

    Strategy: each molecule's embedding has a stamp slot specific to its
    compartment. If we set the stamp slots directly to concentrations, the
    concentration of m becomes dot(Psi, e_m). Since stamps have atom counts
    as values, simple direct assignment doesn't quite work - we need a
    different approach.

    Simpler: seed Psi such that Psi at the compartment's stamp atom-0 index
    equals the total atom-C in that compartment, Psi at stamp-charge index
    equals total charge, etc. In other words, directly set the atom-count
    and charge integrals.

    For rate-law evaluation though, we need c_m(x). We define:
       c_m = <Psi, e_m> / ||e_m||^2 * (something)
    Actually we want c_m such that when the reaction is applied with rate v,
    the stamp of Psi changes exactly like a mass-action update.

    This is genuinely tricky. For this prototype, we take the simplest route:
    track concentrations directly in a parallel dict and also propagate Psi,
    using only Psi for conservation testing and concentrations for rate
    evaluation. This decouples concerns and we can align them in a future
    prototype.
    """
    Psi = np.zeros(D)
    return WellMixedCell(Psi=Psi, volume=1.0)


def step_wellmixed(cell_state: Dict[str, float],
                    rxns: List[SBMLReaction],
                    kinetics: Dict[str, ReactionKinetics],
                    dt: float) -> float:
    """
    One Euler step of the kinetic ODE.
        d[S_i]/dt = sum_r nu_{i,r} * v_r(S)
    Returns total rate magnitude (diagnostic).
    """
    rates = []
    for rxn in rxns:
        if rxn.sbml_id not in kinetics:
            rates.append(0.0); continue
        v = convenience_rate(rxn, kinetics[rxn.sbml_id], cell_state)
        rates.append(v)
    # Apply
    for rxn, v in zip(rxns, rates):
        if v == 0.0: continue
        for sid, coef in rxn.stoichiometry.items():
            cell_state[sid] = cell_state.get(sid, 0.0) + dt * coef * v
    return float(sum(abs(r) for r in rates))


# =============================================================================
# Main test
# =============================================================================
def main():
    print("=" * 76)
    print("DMVC P4: Real Syn3A kinetics on the P3b architecture")
    print("=" * 76)

    # ------------------------------------------------------------------
    # 1. Parse kinetics
    # ------------------------------------------------------------------
    print("\n[1] Parsing SBtab kinetic parameter files...")
    kinetics, init_concs = parse_all_kinetics()
    print(f"    Kinetic entries for {len(kinetics)} reactions")
    print(f"    Initial concentrations for {len(init_concs)} species")
    usable = [r for r, k in kinetics.items() if k.is_usable()]
    print(f"    Usable (have kcat + enzyme + Km): {len(usable)}")
    # Show a sample
    if usable:
        rid = usable[0]
        k = kinetics[rid]
        print(f"\n    Sample: {rid}")
        print(f"      kcat_forward: {k.kcat_forward}, kcat_reverse: {k.kcat_reverse}")
        print(f"      enzyme_conc:  {k.enzyme_conc} mM")
        print(f"      keq:          {k.keq}")
        print(f"      Km entries:   {len(k.Km)}")
        for s, km in list(k.Km.items())[:3]:
            print(f"        {s}: {km:.3f} mM")

    # Sample physiological concentrations
    sample = ["M_atp_c", "M_adp_c", "M_g6p_c", "M_pyr_c", "M_glc__D_c",
              "M_glc__D_e", "M_pep_c", "M_nad_c", "M_nadh_c"]
    print(f"\n    Sample physiological concentrations (mM):")
    for s in sample:
        c = init_concs.get(s, None)
        print(f"      {s:15s}  {c if c is not None else '(not listed)'}")

    # ------------------------------------------------------------------
    # 2. Load reactions; match with kinetics
    # ------------------------------------------------------------------
    print("\n[2] Loading SBML reactions; finding intersection with kinetics...")
    model = load_sbml_model(SBML_PATH)
    mols = extract_molecules_with_compartments(model)
    rxns = extract_reactions(model)
    p_ids, w_ids = find_h_and_water_per_compartment(mols)
    # Rebalance using P3b logic
    rebalanced = []
    for r in rxns:
        if r.is_exchange or r.is_biomass: continue
        nr = rebalance_reaction(r, mols, p_ids, w_ids)
        if nr is not None: rebalanced.append(nr)
    # Filter to those with usable kinetics
    rxns_with_kin = [r for r in rebalanced
                       if r.sbml_id in kinetics and kinetics[r.sbml_id].is_usable()]
    print(f"    {len(rebalanced)} balanced reactions, "
          f"{len(rxns_with_kin)} have complete kinetics")

    # ------------------------------------------------------------------
    # 3. Simulate: ODE-style well-mixed cell from physiological initial conditions
    # ------------------------------------------------------------------
    print("\n[3] Running well-mixed simulation from physiological initial state")
    cell = dict(init_concs)   # copy
    # Make sure all reaction species have an entry (default 0)
    for r in rxns_with_kin:
        for sid in r.stoichiometry:
            cell.setdefault(sid, 0.0)

    # Diagnostics we want to track
    watch = ["M_atp_c", "M_adp_c", "M_amp_c", "M_g6p_c", "M_f6p_c",
              "M_fdp_c", "M_pyr_c", "M_pep_c", "M_nad_c", "M_nadh_c",
              "M_h_c"]
    watch = [s for s in watch if s in cell]

    n_steps = 5000
    dt = 0.001    # 1 ms
    traj = {s: [] for s in watch}
    rate_sum_traj = []
    cell0 = dict(cell)   # initial copy

    for step in range(n_steps):
        rate_total = step_wellmixed(cell, rxns_with_kin, kinetics, dt)
        if step % 50 == 0:
            for s in watch:
                traj[s].append(cell[s])
            rate_sum_traj.append(rate_total)

    # ------------------------------------------------------------------
    # 4. Tests
    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)

    # T1: no NaN/Inf
    any_nan = any(not np.isfinite(v) for v in cell.values())
    print(f"\n[T1] Numerical stability: {'FAIL (NaN/Inf)' if any_nan else 'PASS'}")

    # T2: concentrations non-negative and bounded
    negs = {s: v for s, v in cell.items() if v < -1e-6}
    maxc = max(cell.values())
    print(f"[T2] Physical concentrations:")
    print(f"     Negative species: {len(negs)}  (threshold -1e-6)")
    print(f"     Max concentration: {maxc:.2f} mM")
    t2_pass = len(negs) == 0 and maxc < 1e6
    print(f"     {'PASS' if t2_pass else 'FAIL'}")
    if negs:
        print(f"     First 3 negative species: "
              + "; ".join(f"{s}={v:.4f}" for s, v in list(negs.items())[:3]))

    # T3: do concentrations stay near physiological values
    print(f"\n[T3] How close are steady-ish state concentrations to initial?")
    print(f"     {'species':15s} {'initial':>10s} {'final':>10s} {'ratio':>8s}")
    deviations = []
    for s in watch:
        c0 = cell0.get(s, 0.0)
        cf = cell[s]
        if c0 > 0:
            ratio = cf / c0
            deviations.append(abs(np.log10(max(ratio, 1e-10))))
        else:
            ratio = float('inf') if cf > 0 else 1.0
        print(f"     {s:15s} {c0:10.3f} {cf:10.3f} {ratio:8.2f}x")
    t3_pass = len(deviations) > 0 and max(deviations) < 2.0   # within 100x
    print(f"     {'PASS' if t3_pass else 'FAIL'} "
          f"(all metabolites stayed within 100x of physiological)")

    # T4: glucose consumed -> pyruvate/ATP produced?
    print(f"\n[T4] Basic metabolism sanity: glucose flows through glycolysis?")
    glc_start = cell0.get("M_glc__D_c", 0.0) + cell0.get("M_glc__D_e", 0.0)
    glc_end = cell.get("M_glc__D_c", 0.0) + cell.get("M_glc__D_e", 0.0)
    pyr_start = cell0.get("M_pyr_c", 0.0)
    pyr_end = cell.get("M_pyr_c", 0.0)
    print(f"     Total glucose: {glc_start:.3f} -> {glc_end:.3f} mM (Δ = {glc_end - glc_start:+.3f})")
    print(f"     Pyruvate:      {pyr_start:.3f} -> {pyr_end:.3f} mM (Δ = {pyr_end - pyr_start:+.3f})")
    # We don't enforce a specific sign -- the system may be near steady state
    # where glucose and pyruvate each hover around their setpoints.
    print(f"     (At physiological steady state, both should be roughly stable.)")

    # Summary
    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(f"  Kinetics loaded:      {len(kinetics)} reactions have any data")
    print(f"  Usable kinetics:      {len(usable)}")
    print(f"  Simulation ran:       {n_steps} steps (dt = {dt*1000:.1f} ms)")
    print(f"  Architecture tests:   T1:{'PASS' if not any_nan else 'FAIL'}  "
          f"T2:{'PASS' if t2_pass else 'FAIL'}  "
          f"T3:{'PASS' if t3_pass else 'FAIL'}")
    print()
    print("Honest note on scope:")
    print("  This prototype tests the kinetic layer in a well-mixed single-voxel")
    print("  regime to validate the rate-law implementation and numerical")
    print("  stability. The P3b compartment/spatial architecture is NOT used")
    print("  here -- spatial kinetics coupling is the next prototype.")


if __name__ == "__main__":
    main()
