"""
Layer 3 — novel substrate registration.

Bridge between the Layer 1 atomic engine and the Layer 2 simulator.
Given a novel substrate (SMILES) and a target enzyme class, this module:

  1. Queries the atomic engine for a k_cat estimate
  2. Picks the enzyme instances that will catalyze the novel reaction
  3. Generates a TransitionRule the simulator can fire on it
  4. Optionally adds a "dead-end product" flag (e.g., 2-deoxy-G6P cannot
     be processed further by PGI) to prevent the product from being
     consumed by downstream reactions

This is the architectural payoff of Priority 3: the user can now write

    add_novel_substrate(
        state, rules, engine,
        name='2-deoxyglucose',
        smiles='OC[C@H]1OC(O)C[C@@H](O)[C@@H]1O',
        target_reaction='HEX1',
        product_name='2dg6p',
        dead_end=True,
    )

and the simulator will predict, from a SMILES alone, what happens to
metabolism when this compound is added to the cell. No hand-coding of
k_cat, no hand-picking enzymes.

Limitations that matter
-----------------------
- We still need the user to tell us which enzyme class to try
  (target_reaction). A fully autonomous version would scan all enzymes
  and pick the one with highest estimated k_cat, but enzyme promiscuity
  is genuinely hard and I don't want to fake it.

- Dead-end behavior is flagged manually. A full implementation would
  query the engine for downstream enzyme k_cats (likely very low) and
  let the simulator find the dead-end naturally. That works with the
  current similarity backend if downstream enzymes are strict enough —
  MACE would do better.

- We assume stoichiometry mirrors the parent reaction. For HEX:
  substrate + ATP -> product + ADP. We don't re-derive this from
  structure; we reuse the parent enzyme's stoichiometric template.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from layer2_field.dynamics import CellState, TransitionRule
from layer1_atomic.engine import (
    AtomicEngine, EnzymeProfile, KcatEstimate, eyring_kcat, kcat_to_Ea,
)
from layer3_reactions.sbml_parser import (
    SBMLModel, SBMLReaction, sbml_gene_to_locus,
)
from layer3_reactions.kinetics import ReactionKinetics
from layer3_reactions.coupled import (
    get_species_count, update_species_count, AVOGADRO, count_to_mM,
    INFINITE_SPECIES,
)


# ============================================================================
# Substrate SMILES library — fill in from published data as needed.
# In a real deployment, these would come from a curated metabolite database
# keyed by BiGG ID. For now, a starter dict keyed by the xlsx species IDs.
# ============================================================================
METABOLITE_SMILES = {
    # Glucose family (for PTS/hexokinase references)
    'M_glc__D_c':   'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O',
    'M_glc__D_e':   'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O',
    'M_g6p_c':      'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1OP(=O)(O)O',
    'M_man__L_c':   'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@H]1O',
    'M_fru_c':      'OC[C@@H]1O[C@@](O)(CO)[C@@H](O)[C@@H]1O',
    # Glycerol family (for GLYK)
    'M_glyc_c':     'OCC(O)CO',                        # glycerol
    'M_glyc3p_c':   'OCC(O)COP(=O)(O)O',               # glycerol-3-phosphate
    # Nucleosides (for TMDK1, DGSNK, DCYTK, etc.)
    'M_thymd_c':    'CC1=CN([C@@H]2O[C@H](CO)[C@@H](O)C2)C(=O)NC1=O',  # thymidine
    'M_dtmp_c':     'CC1=CN([C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)C2)C(=O)NC1=O',
    'M_dad_2_c':    'Nc1ncnc2n([C@@H]3O[C@H](CO)[C@@H](O)C3)cnc12',    # deoxyadenosine
    'M_dcyt_c':     'OC[C@H]1O[C@@H](N2C=CC(N)=NC2=O)C[C@@H]1O',       # deoxycytidine
    'M_dgsn_c':     'OC[C@H]1O[C@@H](n2cnc3c2N=C(N)NC3=O)C[C@@H]1O',   # deoxyguanosine
    # Amino acids
    'M_ala__L_c':   'C[C@@H](N)C(=O)O',
    'M_gly_c':      'NCC(=O)O',
}


@dataclass
class NovelSubstrateInfo:
    """Record of what was registered for later inspection / reporting."""
    name: str
    smiles: str
    target_reaction: str
    parent_substrate_id: str
    product_name: str
    kcat_estimate: KcatEstimate
    dead_end: bool
    atomic_species_id: str      # the ID we'll use in state.metabolite_counts
    product_species_id: str


# ============================================================================
# Register a novel substrate
# ============================================================================
def add_novel_substrate(
    state: CellState,
    sbml: SBMLModel,
    kinetics: Dict[str, ReactionKinetics],
    engine: AtomicEngine,
    name: str,
    smiles: str,
    target_reaction: str,
    product_name: str,
    initial_count: int = 50_000,
    dead_end: bool = False,
    known_substrate_smiles: Optional[Dict[str, float]] = None,
) -> Tuple[TransitionRule, NovelSubstrateInfo]:
    """
    Add a novel substrate to the simulator.

    Parameters
    ----------
    state : the live CellState. We attach the new substrate count to it.
    sbml  : parsed SBML model (for enzyme->reaction mapping)
    kinetics : the loaded kinetic parameters (for k_cat of reference substrate)
    engine : AtomicEngine (similarity or MACE)
    name : human label for reporting
    smiles : SMILES string of the novel substrate
    target_reaction : short reaction name to hijack (e.g., 'HEX1')
    product_name : display name of the predicted product (for logging only)
    initial_count : starting molecule count for the novel substrate
    dead_end : if True, the product isn't consumed by any downstream reaction
               (it accumulates and can trap cofactors)
    known_substrate_smiles : if provided, overrides the auto-detected reference
               set. Dict of {substrate_SMILES: measured_kcat_per_s}.

    Returns
    -------
    (rule, info) : the TransitionRule ready to append to the simulator's
                   rule list, and a NovelSubstrateInfo summarizing the
                   registration.
    """

    # Look up the reference reaction and the enzymes that catalyze it
    rxns_by_short = sbml.reactions_by_short_name()
    if target_reaction not in rxns_by_short:
        raise ValueError(f'Reaction {target_reaction!r} not found in SBML')
    ref_rxn = rxns_by_short[target_reaction]
    enzyme_loci = [sbml_gene_to_locus(g) for g in ref_rxn.gene_associations]
    enzyme_loci = [e for e in enzyme_loci if e]
    if not enzyme_loci:
        raise ValueError(f'Reaction {target_reaction!r} has no enzyme mapping')

    if target_reaction not in kinetics:
        raise ValueError(
            f'Reaction {target_reaction!r} has no kinetic parameters')
    parent_kcat = kinetics[target_reaction].kcat_forward

    # Identify the "real" substrate of the parent reaction — the one we are
    # swapping out. Pick the first non-cofactor reactant (ignore ATP, H, H2O).
    cofactor_ids = {'M_atp_c', 'M_adp_c', 'M_amp_c', 'M_gtp_c', 'M_gdp_c',
                     'M_nad_c', 'M_nadh_c', 'M_nadp_c', 'M_nadph_c',
                     'M_h_c', 'M_h_e', 'M_h2o_c', 'M_h2o_e', 'M_pi_c', 'M_ppi_c',
                     'M_coa_c'}
    parent_substrate_id = None
    for sp in ref_rxn.reactants:
        if sp not in cofactor_ids:
            parent_substrate_id = sp
            break
    if parent_substrate_id is None:
        raise ValueError(f'Could not identify non-cofactor substrate in {target_reaction}')

    # Build enzyme profile for the engine
    if known_substrate_smiles is None:
        # Auto-assemble: parent substrate + any similar reactions we know
        known_substrate_smiles = {}
        if parent_substrate_id in METABOLITE_SMILES:
            known_substrate_smiles[METABOLITE_SMILES[parent_substrate_id]] = parent_kcat

    enzyme_profile = EnzymeProfile(
        name=target_reaction,
        reaction_class=_infer_reaction_class(ref_rxn),
        known_substrate_smiles=known_substrate_smiles,
    )

    # Estimate k_cat for the novel substrate
    estimate = engine.estimate_kcat(smiles, enzyme_profile)
    if estimate.kcat_per_s <= 0:
        raise ValueError(
            f'Engine returned non-positive k_cat ({estimate.kcat_per_s:.3g} /s) '
            f'for {name}. Source: {estimate.source}. Notes: {estimate.notes}')

    # Choose simulator species IDs
    novel_id = f'M_novel_{_slug(name)}_c'
    product_id = f'M_novel_{_slug(product_name)}_c'

    # Initialize the species on the state (substrate present, product absent)
    state.metabolite_counts[novel_id] = initial_count
    state.metabolite_counts[product_id] = 0

    # Build the stoichiometry for the new rule by cloning the parent's and
    # replacing the parent substrate with our novel substrate, and the main
    # product with our novel product.
    parent_main_product = None
    for sp in ref_rxn.products:
        if sp not in cofactor_ids:
            parent_main_product = sp
            break
    if parent_main_product is None:
        raise ValueError(
            f'Could not identify main product of {target_reaction}')

    novel_substrates_stoich: Dict[str, float] = {}
    novel_products_stoich: Dict[str, float] = {}
    for sp, coef in ref_rxn.reactants.items():
        if sp == parent_substrate_id:
            novel_substrates_stoich[novel_id] = coef
        else:
            novel_substrates_stoich[sp] = coef
    for sp, coef in ref_rxn.products.items():
        if sp == parent_main_product:
            novel_products_stoich[product_id] = coef
        else:
            novel_products_stoich[sp] = coef

    rule = _make_novel_catalysis_rule(
        name=f'catalysis:novel:{_slug(name)}',
        rxn_label=target_reaction,
        substrates=novel_substrates_stoich,
        products=novel_products_stoich,
        enzyme_loci=enzyme_loci,
        kcat=estimate.kcat_per_s,
    )

    info = NovelSubstrateInfo(
        name=name,
        smiles=smiles,
        target_reaction=target_reaction,
        parent_substrate_id=parent_substrate_id,
        product_name=product_name,
        kcat_estimate=estimate,
        dead_end=dead_end,
        atomic_species_id=novel_id,
        product_species_id=product_id,
    )

    # If the product is a metabolic dead-end, protect it from being consumed
    # by scanning all existing rules and ensuring none of them see it as a
    # substrate. In the current architecture this is a no-op (reactions use
    # the species IDs from SBML, not our novel ID), so the product naturally
    # won't be consumed by any other rule unless we explicitly add one.
    # We record the flag for reporting.
    return rule, info


def _infer_reaction_class(rxn: SBMLReaction) -> str:
    """Crude classification of reaction chemistry from its stoichiometry."""
    reactants = set(rxn.reactants)
    products = set(rxn.products)
    if 'M_atp_c' in reactants and 'M_adp_c' in products:
        return 'phospho_transfer'
    if 'M_nad_c' in reactants or 'M_nadh_c' in reactants:
        return 'oxidoreductase'
    if 'M_h2o_c' in reactants:
        return 'hydrolase'
    return 'isomerase'


def _slug(name: str) -> str:
    """Normalize a human name to a valid species-id slug."""
    return ''.join(c if c.isalnum() else '_' for c in name.lower()).strip('_')


# ============================================================================
# Rule factory — mirrors reversible.py but for a novel (irreversible) forward
# ============================================================================
def _make_novel_catalysis_rule(
    name: str,
    rxn_label: str,
    substrates: Dict[str, float],
    products: Dict[str, float],
    enzyme_loci: List[str],
    kcat: float,
    max_tokens: int = 100,
) -> TransitionRule:
    """Irreversible forward rule for a novel substrate (k_cat from engine)."""

    substrate_ids = list(substrates.keys())
    all_enzymes = list(enzyme_loci)

    def can_fire(state):
        enzyme_instances = []
        for locus in all_enzymes:
            for pid in state.proteins_by_state.get(f'{locus}:native', set()):
                enzyme_instances.append(pid)
        if not enzyme_instances:
            return []

        min_avail = float('inf')
        for sp, stoich in substrates.items():
            c = get_species_count(state, sp)
            cap = c / stoich
            if cap < min_avail:
                min_avail = cap
        if min_avail < 1:
            return []

        n_tok = min(len(enzyme_instances), max_tokens)
        return [(enzyme_instances, min_avail)] * n_tok

    def apply(state, cands, rng):
        if not cands:
            return
        enzyme_instances, _ = cands[0]

        for sp, stoich in substrates.items():
            if get_species_count(state, sp) < stoich:
                return

        for sp, stoich in substrates.items():
            update_species_count(state, sp, -int(stoich))
        for sp, stoich in products.items():
            update_species_count(state, sp, +int(stoich))

        pid = enzyme_instances[rng.integers(0, len(enzyme_instances))]
        gene = state.proteins[pid].gene_id
        sstr = ' + '.join(s.replace('M_', '').replace('_c', '') for s in substrates)
        pstr = ' + '.join(p.replace('M_', '').replace('_c', '') for p in products)
        state.log_event(
            name, [pid],
            f'{rxn_label}: {sstr} -> {pstr} (by {gene}) [NOVEL]',
        )

    return TransitionRule(
        name=name,
        participants=substrate_ids,
        rate=float(kcat),
        rate_source='layer1_atomic',
        can_fire=can_fire,
        apply=apply,
    )


# ============================================================================
# Smoke test
# ============================================================================
if __name__ == '__main__':
    import io, contextlib
    from layer0_genome.syn3a_real import build_real_syn3a_cellspec
    from layer2_field.dynamics import CellState, EventSimulator
    from layer2_field.real_syn3a_rules import (
        populate_real_syn3a, make_folding_rule, make_complex_formation_rules,
    )
    from layer3_reactions.sbml_parser import parse_sbml
    from layer3_reactions.kinetics import load_all_kinetics, load_medium
    from layer3_reactions.reversible import (
        build_reversible_catalysis_rules, initialize_medium,
    )
    from layer3_reactions.coupled import initialize_metabolites

    print('=' * 60)
    print('Priority 3 smoke test: register BrdU at TMDK1')
    print('=' * 60)

    with contextlib.redirect_stdout(io.StringIO()):
        spec, counts, complexes, _ = build_real_syn3a_cellspec()
    sbml = parse_sbml(Path(__file__).parent.parent / 'data' /
                      'Minimal_Cell_ComplexFormation' / 'input_data' /
                      'Syn3A_updated.xml')
    kinetics = load_all_kinetics()
    medium = load_medium()
    rev_rules, _ = build_reversible_catalysis_rules(sbml, kinetics)

    state = CellState(spec)
    populate_real_syn3a(state, counts, scale_factor=0.02, max_per_gene=10)
    initialize_metabolites(state, sbml, cell_volume_um3=(4/3)*np.pi*0.2**3)
    initialize_medium(state, medium)

    engine = AtomicEngine(backend_name='similarity')
    print(f'\nEngine: {engine.active_backend} backend\n')

    try:
        rule, info = add_novel_substrate(
            state, sbml, kinetics, engine,
            name='BrdU',
            smiles='Brc1cn([C@@H]2O[C@H](CO)[C@@H](O)C2)c(=O)[nH]c1=O',
            target_reaction='TMDK1',
            product_name='BrdU_monophosphate',
            initial_count=50_000,
            dead_end=True,
        )
    except ValueError as e:
        print(f'Registration error: {e}')
        raise

    print(f'Registered: {info.name}')
    print(f'  parent_reaction: {info.target_reaction}')
    print(f'  parent_substrate_id: {info.parent_substrate_id}')
    print(f'  novel_species_id: {info.atomic_species_id}')
    print(f'  product_species_id: {info.product_species_id}')
    print(f'  k_cat estimated: {info.kcat_estimate.kcat_per_s:.3f} /s')
    print(f'    confidence: {info.kcat_estimate.confidence:.3f}')
    print(f'    source: {info.kcat_estimate.source}')
    print(f'    notes: {info.kcat_estimate.notes}')

    rules = ([make_folding_rule(20.0)]
             + rev_rules
             + make_complex_formation_rules(complexes, 0.05)
             + [rule])

    print(f'\nRunning 100 ms with {len(rules)} rules (incl. novel BrdU rule)')
    sim = EventSimulator(state, rules, mode='gillespie', seed=42)
    sim.run_until(t_end=0.1, max_events=100_000)

    novel_count_start = 50_000
    novel_count_end = get_species_count(state, info.atomic_species_id)
    product_count_end = get_species_count(state, info.product_species_id)

    print(f'\nResult:')
    print(f'  BrdU                  consumed: {novel_count_start - novel_count_end:,}')
    print(f'  BrdU-monophosphate   produced: {product_count_end:,}')
    novel_events = [e for e in state.events
                     if e.rule_name.startswith('catalysis:novel:')]
    print(f'  Novel catalysis events:        {len(novel_events)}')
    if novel_events:
        print(f'  First event:   {novel_events[0].description}')
        print(f'  Last event:    {novel_events[-1].description}')
