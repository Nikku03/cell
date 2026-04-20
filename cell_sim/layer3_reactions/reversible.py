"""
Reversible Michaelis-Menten catalysis rules.

Upgrades the Priority 1 coupled simulator with:
  1. Forward + reverse rules for reversible reactions
  2. Proper saturation via K_m values (not just mass-action)
  3. Transport reactions from the medium (buffered external species)

Michaelis-Menten propensity (simplified for Gillespie):
  For substrates S_i with K_m^S_i and k_cat:
    v_fwd = k_cat × E × Π(c_i/K_m^i) / (1 + Π(c_i/K_m^i) + Π(c_j/K_m^j))
  where c_i = concentration (mM) of species i
  This gives saturation: when c_i >> K_m, v ≈ k_cat × E (full rate)
                        when c_i << K_m, v ≈ k_cat × E × c_i/K_m (linear)

We approximate this for Gillespie by using saturation-aware rates per
enzyme instance instead of per-substrate-molecule.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from layer2_field.dynamics import (
    CellState, TransitionRule, EventSimulator,
)
from layer3_reactions.sbml_parser import (
    SBMLModel, parse_sbml, sbml_gene_to_locus, SBMLReaction,
)
from layer3_reactions.kinetics import (
    ReactionKinetics, load_all_kinetics, load_medium, MediumSpecies,
)
from layer3_reactions.coupled import (
    get_species_count, update_species_count, count_to_mM, mM_to_count,
    initialize_metabolites, AVOGADRO,
)


# ============================================================================
# Michaelis-Menten saturation factor
# ============================================================================
def mm_saturation_factor(state: CellState,
                          species_ids: List[str],
                          Km_map: Dict[str, float]) -> float:
    """
    Compute the Michaelis-Menten-style saturation for a set of species.

    Returns a factor in [0, 1] representing how saturated the enzyme is
    with these substrates. Full saturation (c >> K_m) → 1.0.
    Starved (c << K_m) → c/K_m (approaches 0).

    For multiple substrates we use the product rule (random binding):
    factor = Π (c_i/(c_i + K_m_i))
    """
    product = 1.0
    vol_L = state.metabolite_volume_L
    for sp in species_ids:
        km = Km_map.get(sp)
        if km is None or km <= 0:
            continue
        count = get_species_count(state, sp)
        c_mM = count_to_mM(count, vol_L) if sp not in state.metabolite_infinite else 1000.0
        # Saturation factor for this substrate
        ratio = c_mM / (c_mM + km)
        product *= ratio
    return product


# ============================================================================
# Build forward + reverse rules for a reversible reaction
# ============================================================================
def make_reversible_rules(
    sbml_rxn: SBMLReaction,
    kinetics: ReactionKinetics,
    enzyme_loci: List[str],
    include_saturation: bool = True,
) -> List[TransitionRule]:
    """
    Build up to two rules per reaction: forward and (if reversible) reverse.

    Each rule fires at saturated rate k_cat × E × saturation_factor when
    substrates are available, and decrements substrates + increments products.
    """
    rules = []
    if not enzyme_loci:
        return rules

    rxn_name = sbml_rxn.short_name
    all_enzymes = list(enzyme_loci)

    # ----- Forward rule -----
    if kinetics.kcat_forward > 0 and sbml_rxn.reactants:
        fwd_rule = _build_one_direction_rule(
            name=f'catalysis:{rxn_name}',
            rxn_name=rxn_name,
            direction='fwd',
            substrates=dict(sbml_rxn.reactants),
            products=dict(sbml_rxn.products),
            enzyme_loci=all_enzymes,
            kcat=kinetics.kcat_forward,
            Km=dict(kinetics.Km),
            include_saturation=include_saturation,
        )
        if fwd_rule is not None:
            rules.append(fwd_rule)

    # ----- Reverse rule -----
    if kinetics.is_reversible and kinetics.kcat_reverse > 0 and sbml_rxn.products:
        rev_rule = _build_one_direction_rule(
            name=f'catalysis:{rxn_name}:rev',
            rxn_name=rxn_name,
            direction='rev',
            substrates=dict(sbml_rxn.products),  # products become substrates in reverse
            products=dict(sbml_rxn.reactants),
            enzyme_loci=all_enzymes,
            kcat=kinetics.kcat_reverse,
            Km=dict(kinetics.Km),
            include_saturation=include_saturation,
        )
        if rev_rule is not None:
            rules.append(rev_rule)

    return rules


def _build_one_direction_rule(
    name: str,
    rxn_name: str,
    direction: str,
    substrates: Dict[str, float],
    products: Dict[str, float],
    enzyme_loci: List[str],
    kcat: float,
    Km: Dict[str, float],
    include_saturation: bool,
) -> Optional[TransitionRule]:
    """Build a single-direction Michaelis-Menten catalysis rule."""

    substrate_ids = list(substrates.keys())
    all_enzymes = list(enzyme_loci)

    MAX_TOKENS = 100

    def can_fire(state):
        enzyme_instances = []
        for locus in all_enzymes:
            natives = state.proteins_by_state.get(f'{locus}:native', set())
            enzyme_instances.extend(list(natives))
        if not enzyme_instances:
            return []

        # Check minimum substrate availability
        min_avail = float('inf')
        for sp, stoich in substrates.items():
            c = get_species_count(state, sp)
            capacity = c / stoich
            if capacity < min_avail:
                min_avail = capacity
        if min_avail < 1:
            return []

        # Compute saturation factor (in [0, 1])
        if include_saturation:
            sat = mm_saturation_factor(state, substrate_ids, Km)
        else:
            sat = 1.0

        # Effective rate per enzyme = kcat × saturation
        # Number of candidate tokens ≈ E * min(S, MAX_TOKENS) * sat
        # We bake saturation into the token count by using round(E * sat_factor)
        n_effective = max(1, int(round(len(enzyme_instances) * sat)))
        n_effective = min(n_effective, MAX_TOKENS)
        return [(enzyme_instances, min_avail)] * n_effective

    def apply(state, cands, rng):
        if not cands:
            return
        enzyme_instances, _ = cands[0]

        # Re-verify substrate availability (state may have shifted)
        for sp, stoich in substrates.items():
            if get_species_count(state, sp) < stoich:
                return

        enzyme_id = enzyme_instances[rng.integers(0, len(enzyme_instances))]

        # Apply stoichiometry
        for sp, stoich in substrates.items():
            update_species_count(state, sp, -int(stoich))
        for sp, stoich in products.items():
            update_species_count(state, sp, +int(stoich))

        # Log
        def abbrev(sp_id):
            return sp_id.replace('M_', '').replace('_c', '').replace('_e', '(ext)')
        sstr = ' + '.join(abbrev(s) for s in substrates)
        pstr = ' + '.join(abbrev(s) for s in products)
        tag = '←' if direction == 'rev' else '→'
        state.log_event(
            name, [enzyme_id],
            f'{rxn_name}{"(rev)" if direction == "rev" else ""}: {sstr} {tag} {pstr} (by {state.proteins[enzyme_id].gene_id})',
        )

    return TransitionRule(
        name=name,
        participants=substrate_ids,
        rate=float(kcat),
        rate_source='literature_mm',
        can_fire=can_fire,
        apply=apply,
    )


# ============================================================================
# Build reversible rules for all metabolic reactions
# ============================================================================
def build_reversible_catalysis_rules(
    sbml: SBMLModel,
    kinetics: Dict[str, ReactionKinetics],
    include_saturation: bool = True,
) -> Tuple[List[TransitionRule], Dict[str, int]]:
    """
    Build forward+reverse rules for all reactions with both SBML and kinetic data.

    Returns (rules, stats_dict).
    """
    rules = []
    stats = {
        'total_reactions': 0,
        'with_kinetics': 0,
        'with_enzymes': 0,
        'forward_rules': 0,
        'reverse_rules': 0,
    }
    by_short = sbml.reactions_by_short_name()

    for rxn_name, kin in kinetics.items():
        stats['total_reactions'] += 1
        if rxn_name not in by_short:
            continue
        stats['with_kinetics'] += 1
        sbml_rxn = by_short[rxn_name]
        enzyme_loci = [sbml_gene_to_locus(g) for g in sbml_rxn.gene_associations]
        enzyme_loci = [e for e in enzyme_loci if e]
        if not enzyme_loci:
            continue
        stats['with_enzymes'] += 1

        rxn_rules = make_reversible_rules(
            sbml_rxn, kin, enzyme_loci,
            include_saturation=include_saturation,
        )
        for r in rxn_rules:
            if r.name.endswith(':rev'):
                stats['reverse_rules'] += 1
            else:
                stats['forward_rules'] += 1
        rules.extend(rxn_rules)

    return rules, stats


# ============================================================================
# Transport rules — uptake and efflux from the medium
# ============================================================================
def initialize_medium(state: CellState, medium: Dict[str, MediumSpecies]):
    """
    Add medium species as infinite reservoirs on the state.

    Medium species (extracellular, ending in '_e') are treated as buffered:
    their concentration is fixed by the medium composition. This prevents
    the cell from draining the external world.
    """
    if not hasattr(state, 'metabolite_medium_mM'):
        state.metabolite_medium_mM = {}
    for sid, m in medium.items():
        # Set count high enough that they behave as reservoirs
        state.metabolite_counts[sid] = 10_000_000_000  # ~16 mM equivalent, plenty
        state.metabolite_infinite.add(sid)
        state.metabolite_medium_mM[sid] = m.conc_mM


def build_transport_rules(
    sbml: SBMLModel,
    kinetics: Dict[str, ReactionKinetics],
    include_saturation: bool = True,
) -> List[TransitionRule]:
    """
    Build transport rules. Transport reactions in SBML have reactants in one
    compartment and products in the other (e.g., M_glc__D_e + M_atp_c →
    M_glc__D_c + M_adp_c + M_pi_c).

    The kinetics for transport reactions are in the Transport sheet.
    We already handle them in build_reversible_catalysis_rules() since
    they have kcat_forward. The only difference is that some species are
    extracellular ('_e') and treated as infinite reservoirs.
    """
    # Transport rules use the same infrastructure; they're already built
    # in build_reversible_catalysis_rules if we include Transport kinetics.
    # This function exists as a hook for future transport-specific logic
    # (e.g., electrochemical gradients).
    return []


# ============================================================================
# Demo
# ============================================================================
if __name__ == "__main__":
    import io, contextlib
    from layer0_genome.syn3a_real import build_real_syn3a_cellspec
    from layer2_field.real_syn3a_rules import (
        populate_real_syn3a, make_folding_rule, make_complex_formation_rules,
    )

    print("=" * 60)
    print("Priority 1.5: Reversible MM + medium transport")
    print("=" * 60)

    print("\nLoading data...")
    with contextlib.redirect_stdout(io.StringIO()):
        spec, counts, complexes, _ = build_real_syn3a_cellspec()

    sbml_path = Path(__file__).parent.parent / 'data' / 'Minimal_Cell_ComplexFormation' / 'input_data' / 'Syn3A_updated.xml'
    sbml = parse_sbml(sbml_path)
    kinetics = load_all_kinetics()
    medium = load_medium()
    print(f"  SBML: {len(sbml.species)} species, {len(sbml.reactions)} reactions")
    print(f"  Kinetics: {len(kinetics)} reactions "
          f"({sum(1 for k in kinetics.values() if k.is_reversible)} reversible)")
    print(f"  Medium: {len(medium)} external species")

    # Build rules
    rev_rules, stats = build_reversible_catalysis_rules(sbml, kinetics, include_saturation=True)
    print(f"\nReversible rules built:")
    print(f"  With kinetics data: {stats['with_kinetics']}")
    print(f"  With enzyme mapping: {stats['with_enzymes']}")
    print(f"  Forward rules:  {stats['forward_rules']}")
    print(f"  Reverse rules:  {stats['reverse_rules']}")
    print(f"  Total new rules: {len(rev_rules)}")

    # Populate state
    state = CellState(spec)
    populate_real_syn3a(state, counts, scale_factor=0.02, max_per_gene=10)
    cell_vol_um3 = (4/3) * np.pi * (0.2)**3
    initialize_metabolites(state, sbml, cell_volume_um3=cell_vol_um3)
    initialize_medium(state, medium)

    print(f"\n  Populated {len(state.proteins):,} proteins, "
          f"{len(state.metabolite_counts)} metabolites, "
          f"{len(state.metabolite_infinite)} infinite reservoirs")

    # Report key initial concentrations
    print("\n  Initial cytoplasm concentrations:")
    for sid, label in [('M_atp_c', 'ATP'), ('M_adp_c', 'ADP'),
                         ('M_g6p_c', 'G6P'), ('M_pep_c', 'PEP'),
                         ('M_pyr_c', 'Pyr'), ('M_dhap_c', 'DHAP'),
                         ('M_nad_c', 'NAD'), ('M_nadh_c', 'NADH')]:
        c = get_species_count(state, sid)
        mM = count_to_mM(c, state.metabolite_volume_L)
        print(f"    {label:5s}: {c:>10,} ({mM:.3f} mM)")

    print("\n  Medium reservoir concentrations:")
    for sid, m in list(medium.items())[:6]:
        print(f"    {sid:20s}: {m.conc_mM:.3f} mM ({m.name})")

    # Build full rule set and run
    all_rules = (
        [make_folding_rule(20.0)]
        + rev_rules
        + make_complex_formation_rules(complexes, base_rate_per_s=0.05)
    )
    print(f"\nTotal rule count: {len(all_rules)}")

    # Snapshot initial
    initial_counts = dict(state.metabolite_counts)

    # Run
    import time
    print("\nRunning 0.2 s simulated time...")
    sim = EventSimulator(state, all_rules, mode='gillespie', seed=42)
    t0 = time.time()
    stats_run = sim.run_until(t_end=0.2, max_events=500_000)
    wall = time.time() - t0
    print(f"  {stats_run['n_events']:,} events in {wall:.2f}s wall")
    print(f"  ({stats_run['events_per_wall_sec']:.0f} events/sec)")

    # Metabolite changes
    print("\n  Key metabolite changes:")
    print(f"  {'Species':14s}  {'Initial':>10s}  {'Final':>10s}  {'Δ':>10s}  {'Δ mM':>9s}")
    watched = ['M_atp_c', 'M_adp_c', 'M_g6p_c', 'M_f6p_c', 'M_fdp_c',
                'M_g3p_c', 'M_dhap_c', 'M_pep_c', 'M_pyr_c', 'M_lac__L_c',
                'M_nad_c', 'M_nadh_c']
    for sid in watched:
        if sid in state.metabolite_infinite:
            print(f"    {sid:14s}  (infinite)")
            continue
        init_c = initial_counts.get(sid, 0)
        fin_c = get_species_count(state, sid)
        d = fin_c - init_c
        dmM = count_to_mM(abs(d), state.metabolite_volume_L) * (1 if d >= 0 else -1)
        print(f"    {sid:14s}  {init_c:>10,}  {fin_c:>10,}  {d:>+10,}  {dmM:>+9.3f}")

    # Event breakdown
    from collections import Counter
    by_dir = Counter()
    for e in state.events:
        if e.rule_name.startswith('catalysis:'):
            by_dir['reverse' if ':rev' in e.rule_name else 'forward'] += 1
        else:
            by_dir[e.rule_name.split(':')[0]] += 1
    print("\n  Event direction breakdown:")
    for k, n in by_dir.most_common():
        print(f"    {k}: {n:,}")

    # Top reactions forward vs reverse
    cat_counts = Counter()
    for e in state.events:
        if e.rule_name.startswith('catalysis:'):
            parts = e.rule_name.split(':')
            rxn = parts[1]
            is_rev = len(parts) > 2 and parts[2] == 'rev'
            cat_counts[(rxn, is_rev)] += 1
    print("\n  Top reactions (showing fwd/rev balance):")
    seen_rxns = set()
    for (rxn, is_rev), n in cat_counts.most_common(30):
        if rxn in seen_rxns:
            continue
        fwd = cat_counts.get((rxn, False), 0)
        rev = cat_counts.get((rxn, True), 0)
        net = fwd - rev
        kf = kinetics[rxn].kcat_forward if rxn in kinetics else 0
        kr = kinetics[rxn].kcat_reverse if rxn in kinetics else 0
        print(f"    {rxn:10s}  fwd={fwd:>6,}  rev={rev:>6,}  net={net:>+7,}  "
              f"(k_f={kf:.1f} k_r={kr:.1f})")
        seen_rxns.add(rxn)
        if len(seen_rxns) >= 10:
            break
