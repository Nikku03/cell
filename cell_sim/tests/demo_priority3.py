"""
Priority 3 demonstration — atomic-scale k_cat for a novel substrate.

The claim: our simulator can handle substrates that are not in the
measured kinetic database. Given just a SMILES string and a target
enzyme, Layer 1 estimates a k_cat; Layer 3 turns that into a
simulator rule; Layer 2 fires it against real Syn3A enzyme molecules.

We demonstrate with 5-bromo-2'-deoxyuridine (BrdU), a real DNA
proliferation tracer that has no entry in the Luthey-Schulten kinetic
parameter database. BrdU is structurally similar to thymidine — the
only difference is a bromine atom in place of the methyl group on the
pyrimidine ring. So we expect Syn3A's thymidine kinase (TMDK1) to
phosphorylate it.

The simulator, knowing nothing about BrdU beyond its SMILES, should:
  1. Estimate k_cat from TMDK1's measured thymidine k_cat (19.26 /s)
     and the BrdU-thymidine structural similarity
  2. Fire catalysis events at that rate against the real TMDK1
     enzyme instances (JCVISYN3A_0140, tdk)
  3. Consume ATP and produce BrdU-monophosphate at a rate comparable
     to normal thymidine metabolism

This is the architectural move that the Luthey-Schulten model can't
make: they simulate Syn3A because they measured its k_cats. We can
simulate Syn3A + any new compound because we compute the k_cats.

Output: prints a side-by-side comparison of metabolism with and
without BrdU present. The with-BrdU run should show additional ATP
consumption by the new TMDK1 events and a growing pool of BrdU-MP.
"""

import sys, time, io, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from layer0_genome.syn3a_real import build_real_syn3a_cellspec
from layer2_field.dynamics import CellState
from layer2_field.fast_dynamics import FastEventSimulator as EventSimulator
from layer2_field.real_syn3a_rules import (
    populate_real_syn3a, make_folding_rule, make_complex_formation_rules,
)
from layer3_reactions.sbml_parser import parse_sbml
from layer3_reactions.kinetics import load_all_kinetics, load_medium
from layer3_reactions.reversible import (
    build_reversible_catalysis_rules, initialize_medium,
)
from layer3_reactions.coupled import (
    initialize_metabolites, get_species_count, count_to_mM,
)
from layer1_atomic.engine import AtomicEngine, EnzymeProfile
from layer3_reactions.novel_substrates import add_novel_substrate


# ============================================================================
# Config
# ============================================================================
SIM_TIME_S = 0.3           # simulated seconds
SCALE_FACTOR = 0.02
SEED = 42

NOVEL_SUBSTRATE_NAME = 'BrdU'
NOVEL_SUBSTRATE_SMILES = 'Brc1cn([C@@H]2O[C@H](CO)[C@@H](O)C2)c(=O)[nH]c1=O'
NOVEL_TARGET_REACTION = 'TMDK1'
NOVEL_PRODUCT_NAME = 'BrdU_monophosphate'
NOVEL_INITIAL_COUNT = 100_000     # ~5 mM analog concentration


# ============================================================================
# One simulator run
# ============================================================================
def run(with_novel: bool, engine: AtomicEngine = None):
    with contextlib.redirect_stdout(io.StringIO()):
        spec, counts, complexes, _ = build_real_syn3a_cellspec()

    data_root = Path(__file__).resolve().parent.parent / 'data' / 'Minimal_Cell_ComplexFormation' / 'input_data'
    sbml = parse_sbml(data_root / 'Syn3A_updated.xml')
    kinetics = load_all_kinetics()
    medium = load_medium()
    rev_rules, _ = build_reversible_catalysis_rules(sbml, kinetics)

    state = CellState(spec)
    populate_real_syn3a(state, counts, scale_factor=SCALE_FACTOR, max_per_gene=10)
    initialize_metabolites(state, sbml, cell_volume_um3=(4/3)*np.pi*0.2**3)
    initialize_medium(state, medium)

    rules = ([make_folding_rule(20.0)]
             + rev_rules
             + make_complex_formation_rules(complexes, base_rate_per_s=0.05))

    info = None
    if with_novel:
        if engine is None:
            engine = AtomicEngine(backend_name='similarity')
        rule, info = add_novel_substrate(
            state, sbml, kinetics, engine,
            name=NOVEL_SUBSTRATE_NAME,
            smiles=NOVEL_SUBSTRATE_SMILES,
            target_reaction=NOVEL_TARGET_REACTION,
            product_name=NOVEL_PRODUCT_NAME,
            initial_count=NOVEL_INITIAL_COUNT,
            dead_end=True,
        )
        rules.append(rule)

    initial = dict(state.metabolite_counts)

    sim = EventSimulator(state, rules, mode='gillespie', seed=SEED)
    t0 = time.time()
    sim.run_until(t_end=SIM_TIME_S, max_events=1_000_000)
    wall = time.time() - t0

    return {
        'state': state,
        'initial_counts': initial,
        'wall_time': wall,
        'info': info,
        'n_rules': len(rules),
    }


# ============================================================================
# Reporting helpers
# ============================================================================
WATCHED_METS = [
    ('M_atp_c',      'ATP'),
    ('M_adp_c',      'ADP'),
    ('M_thymd_c',    'thymidine'),
    ('M_dtmp_c',     'dTMP'),
    ('M_pi_c',       'Pi'),
]


def summarize(label, run_result):
    state = run_result['state']
    initial = run_result['initial_counts']
    vol_L = state.metabolite_volume_L

    print(f'\n{"="*64}')
    print(f'  {label}')
    print(f'{"="*64}')
    print(f'  rules: {run_result["n_rules"]}   '
          f'wall: {run_result["wall_time"]:.2f}s   '
          f'events: {len(state.events):,}')

    # Metabolite deltas
    print(f'\n  Metabolite changes:')
    print(f'    {"species":<14s} {"initial":>10s} {"final":>10s} {"Δ count":>10s} {"Δ mM":>8s}')
    for sid, name in WATCHED_METS:
        init_c = initial.get(sid, 0)
        fin_c = get_species_count(state, sid)
        d = fin_c - init_c
        dmM = count_to_mM(abs(d), vol_L) * (1 if d >= 0 else -1)
        print(f'    {name:<14s} {init_c:>10,} {fin_c:>10,} {d:>+10,} {dmM:>+8.3f}')

    # Novel substrate details
    if run_result['info'] is not None:
        info = run_result['info']
        novel_final = get_species_count(state, info.atomic_species_id)
        product_final = get_species_count(state, info.product_species_id)
        novel_consumed = NOVEL_INITIAL_COUNT - novel_final
        print(f'\n  Novel substrate ({info.name}):')
        print(f'    SMILES:             {info.smiles[:60]}{"..." if len(info.smiles)>60 else ""}')
        print(f'    Nearest known:      thymidine (Tanimoto {info.kcat_estimate.similarity:.3f})')
        print(f'    k_cat estimate:     {info.kcat_estimate.kcat_per_s:.3f} /s '
              f'(source: {info.kcat_estimate.source})')
        print(f'    Reference k_cat:    {19.26} /s (thymidine at TMDK1)')
        print(f'    Initial count:      {NOVEL_INITIAL_COUNT:,}')
        print(f'    Consumed:           {novel_consumed:>7,} '
              f'({count_to_mM(novel_consumed, vol_L):.3f} mM)')
        print(f'    Product formed:     {product_final:>7,} '
              f'(BrdU-MP, dead-end)')

    # Event breakdown
    from collections import Counter
    ev_types = Counter()
    for e in state.events:
        prefix = e.rule_name.split(':')[0]
        if ':novel:' in e.rule_name:
            prefix = 'catalysis[novel]'
        ev_types[prefix] += 1
    print(f'\n  Event breakdown:')
    for t, n in ev_types.most_common():
        print(f'    {t:24s} {n:>8,}')

    # How many TMDK1 events fired in each case?
    tmdk_fwd = sum(1 for e in state.events if e.rule_name == 'catalysis:TMDK1')
    tmdk_rev = sum(1 for e in state.events if e.rule_name == 'catalysis:TMDK1:rev')
    tmdk_novel = sum(1 for e in state.events
                      if e.rule_name.startswith('catalysis:novel:'))
    print(f'\n  Thymidine kinase activity:')
    print(f'    TMDK1 forward (thymidine -> dTMP):  {tmdk_fwd:>6,}')
    print(f'    TMDK1 reverse (dTMP -> thymidine):  {tmdk_rev:>6,}')
    print(f'    TMDK1 on novel substrate (BrdU):    {tmdk_novel:>6,}')
    net_tmdk_atp_consumption = tmdk_fwd - tmdk_rev + tmdk_novel
    print(f'    Net ATP consumption by TMDK1:       {net_tmdk_atp_consumption:>6,}')


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print('=' * 64)
    print('Priority 3 demonstration — novel substrate via atomic physics')
    print('=' * 64)
    print()
    print('Testing whether a SMILES string alone is enough to simulate a')
    print('compound that has no entry in the measured kinetic database.')
    print()
    print(f'Novel compound: {NOVEL_SUBSTRATE_NAME}')
    print(f'  SMILES:  {NOVEL_SUBSTRATE_SMILES}')
    print(f'  Target:  {NOVEL_TARGET_REACTION} (thymidine kinase, k_cat_thymd=19.26/s)')
    print(f'Simulation: {SIM_TIME_S*1000:.0f} ms at {SCALE_FACTOR*100:.0f}% scale, '
          f'seed={SEED}')

    # Engine setup (auto-detects MACE availability)
    engine = AtomicEngine(backend_name='auto')
    print(f'\nAtomic engine backend: {engine.active_backend}')
    if engine.active_backend == 'similarity':
        print('  (MACE would be used on GPU; falling back to RDKit similarity)')

    # Baseline: no novel substrate
    print('\n' + '-' * 64)
    print('Run 1: baseline (no novel substrate)')
    print('-' * 64)
    base = run(with_novel=False)
    summarize('BASELINE (no novel substrate)', base)

    # With novel substrate
    print('\n' + '-' * 64)
    print('Run 2: with BrdU added to the cell')
    print('-' * 64)
    with_novel = run(with_novel=True, engine=engine)
    summarize('WITH NOVEL SUBSTRATE (BrdU)', with_novel)

    # Side-by-side difference
    print(f'\n{"="*64}')
    print('  SIDE-BY-SIDE EFFECT OF NOVEL SUBSTRATE')
    print(f'{"="*64}')

    def delta_for(sid, run_result):
        return (get_species_count(run_result['state'], sid)
                - run_result['initial_counts'].get(sid, 0))

    print(f'\n  Metabolite: how much did the novel substrate shift things?')
    print(f'    {"species":<14s} {"baseline Δ":>13s} {"with-BrdU Δ":>13s} {"difference":>13s}')
    for sid, name in WATCHED_METS:
        d_base = delta_for(sid, base)
        d_novel = delta_for(sid, with_novel)
        diff = d_novel - d_base
        print(f'    {name:<14s} {d_base:>+13,} {d_novel:>+13,} {diff:>+13,}')

    # Summary
    if with_novel['info'] is not None:
        info = with_novel['info']
        novel_events = sum(1 for e in with_novel['state'].events
                            if ':novel:' in e.rule_name)
        total_events_diff = len(with_novel['state'].events) - len(base['state'].events)
        print()
        print('  Interpretation:')
        print(f'    {novel_events:,} novel catalysis events fired against real TMDK1 enzyme')
        print(f'    The cell consumed {NOVEL_INITIAL_COUNT - get_species_count(with_novel["state"], info.atomic_species_id):,} '
              f'BrdU molecules')
        print(f'    Total events: {len(base["state"].events):,} baseline vs '
              f'{len(with_novel["state"].events):,} with BrdU (+{total_events_diff:,})')
        print(f'    The k_cat was predicted from structure alone, with no measured data for BrdU')

    print()
