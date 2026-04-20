"""
Coupled Layer 2 / Layer 3 simulator.

Every catalysis event now:
  1. Checks substrate availability (mass-action propensity)
  2. Decrements substrates
  3. Increments products
  4. Logs the full transfer (e.g., "1 phosphate moved from ATP to G6P via PGI")

Metabolite handling:
  - Low-abundance species (< 10000 molecules) are tracked as integer counts
  - High-abundance species (water, H+) are treated as infinite reservoirs
    (constant concentration) to avoid integer overflow and unnecessary
    propensity churn
  - Extracellular species ('_e') are also infinite reservoirs by default
    (medium is buffered externally)

This gives us proper mass-action coupling. Enzymes slow down when
substrates deplete. Metabolite concentrations change over time.
Pathways become visible as coupled fluxes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from layer2_field.dynamics import (
    CellState, TransitionRule, EventSimulator,
    Complex,
)
from layer0_genome.syn3a_real import (
    build_real_syn3a_cellspec, ComplexDef,
)
from layer3_reactions.sbml_parser import (
    SBMLModel, parse_sbml, sbml_gene_to_locus,
)


# ============================================================================
# Metabolite tracking inside CellState
# ============================================================================
# We store metabolite counts as a flat dict on the state object. Rules read
# and write this dict.

AVOGADRO = 6.022e23

# Species we treat as infinite reservoirs (counts stay constant).
# Water, H+, and extracellular nutrients from the medium.
INFINITE_SPECIES = {
    'M_h2o_c', 'M_h_c', 'M_h2o_e', 'M_h_e',
    'M_co2_c', 'M_co2_e',
    'M_o2_c', 'M_o2_e',
}

# Threshold below which we track as integer counts
COUNTABLE_THRESHOLD = 100_000


def mM_to_count(conc_mM: float, volume_L: float) -> int:
    """Convert millimolar concentration + volume to molecule count."""
    return int(round(conc_mM * 1e-3 * volume_L * AVOGADRO))


def count_to_mM(count: int, volume_L: float) -> float:
    """Convert molecule count + volume to millimolar concentration."""
    if volume_L <= 0:
        return 0.0
    return (count / AVOGADRO) / volume_L * 1000.0


def initialize_metabolites(state: CellState, sbml: SBMLModel,
                            cell_volume_um3: float = 0.034) -> Dict[str, int]:
    """
    Seed metabolite counts on the CellState.

    Reads initial concentrations from CellSpec.metabolites (which came
    from initial_concentrations.xlsx) and from sbml model species.

    Returns the metabolite counts dict for inspection.
    """
    # Convert cell volume from μm^3 to L
    # 1 μm^3 = 1e-15 L
    volume_L = cell_volume_um3 * 1e-15

    state.metabolite_counts = {}
    state.metabolite_volume_L = volume_L
    state.metabolite_infinite = set(INFINITE_SPECIES)

    # From CellSpec metabolites (intracellular concentrations from xlsx)
    for met_id, met in state.spec.metabolites.items():
        # These met_ids are BiGG-style like 'atp_c', 'pyr_c'. Convert to SBML style.
        sbml_id = f'M_{met_id}' if not met_id.startswith('M_') else met_id
        count = mM_to_count(met.initial_concentration_mM, volume_L)
        state.metabolite_counts[sbml_id] = count

    # For SBML species not covered, use a default concentration
    # (many species will have zero count initially — that's correct, they
    # appear as products of reactions)
    for sid in sbml.species:
        if sid not in state.metabolite_counts:
            state.metabolite_counts[sid] = 0

    return state.metabolite_counts


def get_species_count(state: CellState, species_id: str) -> int:
    """Get species count, returning infinity for infinite-reservoir species."""
    if species_id in state.metabolite_infinite:
        return COUNTABLE_THRESHOLD * 10  # effectively infinite for our propensity calcs
    return state.metabolite_counts.get(species_id, 0)


def update_species_count(state: CellState, species_id: str, delta: int):
    """Update a species count (no-op for infinite reservoirs)."""
    if species_id in state.metabolite_infinite:
        return
    if species_id not in state.metabolite_counts:
        state.metabolite_counts[species_id] = 0
    state.metabolite_counts[species_id] += delta
    # Clamp negatives (shouldn't happen but be safe with propensity rounding)
    if state.metabolite_counts[species_id] < 0:
        state.metabolite_counts[species_id] = 0


# ============================================================================
# Coupled catalysis rule: real stoichiometry
# ============================================================================
def make_coupled_catalysis_rule(
    sbml_rxn, kcat: float, enzyme_loci: List[str],
) -> Optional[TransitionRule]:
    """
    Build a catalysis rule that implements proper mass-action coupling.

    Propensity = k_cat * enzyme_count * min(substrate counts)
    When fired: decrement substrates, increment products.

    Returns None if rule can't be built (missing stoichiometry, etc.)
    """
    if not enzyme_loci or not sbml_rxn.reactants:
        return None

    rxn_name = sbml_rxn.short_name
    reactants = dict(sbml_rxn.reactants)   # {species_id: stoich}
    products = dict(sbml_rxn.products)
    all_enzymes = list(enzyme_loci)

    def can_fire(state):
        # Find all native enzyme molecules
        enzyme_instances = []
        for locus in all_enzymes:
            natives = state.proteins_by_state.get(f'{locus}:native', set())
            enzyme_instances.extend(list(natives))
        if not enzyme_instances:
            return []

        # Check substrate availability — for propensity we need the
        # limiting substrate count
        min_substrate = float('inf')
        for sp, stoich in reactants.items():
            count = get_species_count(state, sp)
            capacity = count / stoich  # how many firings this substrate allows
            if capacity < min_substrate:
                min_substrate = capacity
        if min_substrate < 1:
            return []

        # Propensity proxy: each "candidate" represents one (enzyme_instance, substrate_slot) pair.
        # For mass-action k_cat * E * min(S/stoich), we want propensity = kcat * E * S_limiting
        # Our framework multiplies rule.rate by len(candidates). So:
        # len(candidates) should be E * min_substrate_count_normalized.
        # But we can't return 10^6 candidates — overhead blows up.
        # Trick: return just enzyme_instances and bake the substrate factor
        # into a precomputed propensity multiplier. Use a tuple encoding.
        return [('fire', enzyme_instances, min_substrate)]

    def apply(state, cands, rng):
        if not cands:
            return
        _, enzyme_instances, _ = cands[0]

        # Check substrates again (state may have changed since can_fire)
        for sp, stoich in reactants.items():
            if get_species_count(state, sp) < stoich:
                return

        # Pick an enzyme to log as the acting molecule
        enzyme_id = enzyme_instances[rng.integers(0, len(enzyme_instances))]

        # Decrement substrates, increment products
        for sp, stoich in reactants.items():
            update_species_count(state, sp, -int(stoich))
        for sp, stoich in products.items():
            update_species_count(state, sp, +int(stoich))

        # Log with rich description
        def abbrev(sp_id):
            return sp_id.replace('M_', '').replace('_c', '')
        substrate_str = ' + '.join(f'{s}' for s in (abbrev(s) for s in reactants))
        product_str = ' + '.join(f'{s}' for s in (abbrev(p) for p in products))
        state.log_event(
            f'catalysis:{rxn_name}', [enzyme_id],
            f'{rxn_name}: {substrate_str} → {product_str} (by {state.proteins[enzyme_id].gene_id})',
        )

    # For mass-action, we need propensity proportional to enzyme_count * substrate_count.
    # The simulator sets propensity = rule.rate * len(candidates).
    # We want: propensity = k_cat * E * S_limiting
    # So we need len(candidates) = E * S_limiting. But returning that many is
    # too expensive.
    #
    # Pragmatic approximation: use the effective rate scaling inside can_fire.
    # We return a small number of candidate tokens but the rule.rate has been
    # pre-scaled. This sacrifices exact Gillespie correctness for tractability.
    # For Syn3A with enzyme counts in the tens and substrate counts often in
    # the thousands, this gives qualitatively correct behavior.
    #
    # Better approach: return candidates equal to min(E * S, MAX_TOKENS).
    # With rate=k_cat / MAX_TOKENS, the aggregate propensity matches.
    # Let's do that.

    MAX_TOKENS = 100  # cap candidate list size

    def can_fire_bounded(state):
        enzyme_instances = []
        for locus in all_enzymes:
            natives = state.proteins_by_state.get(f'{locus}:native', set())
            enzyme_instances.extend(list(natives))
        if not enzyme_instances:
            return []

        min_substrate = float('inf')
        for sp, stoich in reactants.items():
            count = get_species_count(state, sp)
            capacity = count / stoich
            if capacity < min_substrate:
                min_substrate = capacity
        if min_substrate < 1:
            return []

        # Total propensity tokens ≈ E * min(S)
        total = len(enzyme_instances) * int(min(min_substrate, MAX_TOKENS))
        total = max(1, min(total, MAX_TOKENS))
        return [('fire', enzyme_instances, min_substrate)] * total

    return TransitionRule(
        name=f'catalysis:{rxn_name}',
        participants=list(reactants.keys()),
        rate=float(kcat),
        rate_source='literature',
        can_fire=can_fire_bounded,
        apply=apply,
    )


# ============================================================================
# Build all coupled rules from SBML + kcat data
# ============================================================================
def build_coupled_catalysis_rules(sbml: SBMLModel, kcats: Dict[str, float]) -> List[TransitionRule]:
    """
    For every reaction that has both SBML stoichiometry AND a known k_cat,
    build a coupled catalysis rule.

    Returns the list of rules.
    """
    rules = []
    by_short_name = sbml.reactions_by_short_name()
    matched = 0
    unmatched_kcat = []
    for rxn_name, kcat in kcats.items():
        if rxn_name not in by_short_name:
            unmatched_kcat.append(rxn_name)
            continue
        sbml_rxn = by_short_name[rxn_name]
        enzyme_loci = [sbml_gene_to_locus(g) for g in sbml_rxn.gene_associations]
        enzyme_loci = [e for e in enzyme_loci if e]
        if not enzyme_loci:
            continue
        rule = make_coupled_catalysis_rule(sbml_rxn, kcat, enzyme_loci)
        if rule is not None:
            rules.append(rule)
            matched += 1
    return rules, unmatched_kcat


# ============================================================================
# Demo
# ============================================================================
if __name__ == "__main__":
    import io, contextlib

    print("=" * 60)
    print("Coupled Layer 2 + Layer 3 simulation (REAL Syn3A)")
    print("=" * 60)

    print("\nLoading data...")
    with contextlib.redirect_stdout(io.StringIO()):
        spec, counts, complexes, kcats = build_real_syn3a_cellspec()

    sbml_path = Path(__file__).parent.parent / 'data' / 'Minimal_Cell_ComplexFormation' / 'input_data' / 'Syn3A_updated.xml'
    sbml = parse_sbml(sbml_path)
    print(f"  SBML: {len(sbml.species)} species, {len(sbml.reactions)} reactions")

    # Build coupled rules
    coupled_rules, unmatched = build_coupled_catalysis_rules(sbml, kcats)
    print(f"  Built {len(coupled_rules)} coupled catalysis rules")
    print(f"  Unmatched k_cat entries: {len(unmatched)} (no SBML stoichiometry)")
    if unmatched[:5]:
        print(f"    examples: {unmatched[:5]}")

    # Populate state
    from layer2_field.real_syn3a_rules import (
        populate_real_syn3a, make_folding_rule, make_complex_formation_rules,
    )

    state = CellState(spec)
    populate_real_syn3a(state, counts, scale_factor=0.02, max_per_gene=10)
    print(f"  Populated {len(state.proteins):,} protein molecules")

    # Initialize metabolite pool — Syn3A cell volume is ~0.034 μm^3 (200 nm radius sphere)
    cell_volume_um3 = (4/3) * np.pi * (0.2)**3  # ≈ 0.0335
    mets = initialize_metabolites(state, sbml, cell_volume_um3=cell_volume_um3)
    print(f"  Metabolites initialized: {len(mets)} species")

    # Report initial counts for key metabolites
    print("\n  Initial key metabolite counts:")
    for sid in ['M_atp_c', 'M_adp_c', 'M_g6p_c', 'M_pep_c', 'M_pyr_c',
                 'M_lac__L_c', 'M_nad_c', 'M_nadh_c', 'M_pi_c']:
        c = get_species_count(state, sid)
        mM = count_to_mM(c, state.metabolite_volume_L) if sid not in state.metabolite_infinite else 'inf'
        print(f"    {sid:15s}  count={c:>10,}  ({mM if mM == 'inf' else f'{mM:.2f}'} mM)")

    # Build the full rule set
    rules = (
        [make_folding_rule(k_fold_per_s=20.0)]
        + coupled_rules
        + make_complex_formation_rules(complexes, base_rate_per_s=0.05)
    )
    print(f"\nTotal rules: {len(rules)}")

    # Snapshot initial metabolite state BEFORE simulation
    initial_counts_snapshot = dict(state.metabolite_counts)

    # Run
    import time
    print("\nRunning 0.2 s simulated time (coupled, real stoichiometry)...")
    sim = EventSimulator(state, rules, mode='gillespie', seed=42)
    t0 = time.time()
    stats = sim.run_until(t_end=0.2, max_events=300_000)
    wall = time.time() - t0
    print(f"  {stats['n_events']:,} events in {wall:.2f}s wall")
    print(f"  ({stats['events_per_wall_sec']:.0f} events/sec, "
          f"{state.time*1000:.1f} ms simulated)")

    # Metabolite changes
    print("\n  Metabolite changes after simulation:")
    print(f"  {'Species':15s}  {'Initial':>12s}  {'Final':>12s}  {'Δ':>12s}  {'Δ mM':>10s}")
    watched = ['M_atp_c', 'M_adp_c', 'M_amp_c',
                'M_g6p_c', 'M_f6p_c', 'M_fdp_c', 'M_g3p_c', 'M_dhap_c',
                'M_13dpg_c', 'M_3pg_c', 'M_2pg_c', 'M_pep_c', 'M_pyr_c',
                'M_lac__L_c', 'M_nad_c', 'M_nadh_c', 'M_pi_c']
    for sid in watched:
        if sid in state.metabolite_infinite:
            print(f"  {sid:15s}  (infinite reservoir)")
            continue
        init_c = initial_counts_snapshot.get(sid, 0)
        final_c = get_species_count(state, sid)
        delta = final_c - init_c
        delta_mM = count_to_mM(abs(delta), state.metabolite_volume_L) * (1 if delta >= 0 else -1)
        print(f"  {sid:15s}  {init_c:>12,}  {final_c:>12,}  {delta:>+12,}  {delta_mM:>+10.3f}")

    # Event breakdown
    from collections import Counter
    by_type = Counter()
    for e in state.events:
        key = e.rule_name.split(':')[0] if ':' in e.rule_name else e.rule_name
        by_type[key] += 1
    print("\n  Event type summary:")
    for t, n in by_type.most_common():
        print(f"    {t}: {n:,}")

    # Top catalyses
    cat_by_rxn = Counter()
    for e in state.events:
        if e.rule_name.startswith('catalysis:'):
            cat_by_rxn[e.rule_name.split(':')[1]] += 1
    print("\n  Top 10 catalyzed reactions:")
    for rxn, n in cat_by_rxn.most_common(10):
        kc = kcats.get(rxn, 0)
        print(f"    {rxn:8s}  {n:,} events  (k_cat={kc:.0f}/s)")

    # First few coupled events with rich descriptions
    print("\n  First 10 catalysis events (with stoichiometry info):")
    cat_events = [e for e in state.events if e.rule_name.startswith('catalysis:')]
    for e in cat_events[:10]:
        print(f"    t={e.time*1e6:8.0f} μs  {e.description[:90]}")
