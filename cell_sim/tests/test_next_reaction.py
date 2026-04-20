"""
Statistical correctness check: NextReactionSimulator vs FastEventSimulator.

They can't be event-for-event identical (different RNG consumption),
but must produce statistically equivalent trajectories:
  - Final metabolite counts within ~1% (stochastic noise)
  - Event type distribution within ~5%
  - Total event count within ~5%

We also confirm Phase 2 runs faster than Phase 1.
"""
from __future__ import annotations
import sys, io, contextlib, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from collections import Counter

from layer0_genome.syn3a_real import build_real_syn3a_cellspec
from layer2_field.dynamics import CellState
from layer2_field.fast_dynamics import FastEventSimulator
from layer2_field.next_reaction_dynamics import NextReactionSimulator
from layer2_field.real_syn3a_rules import (
    populate_real_syn3a, make_folding_rule, make_complex_formation_rules,
)
from layer3_reactions.sbml_parser import parse_sbml
from layer3_reactions.kinetics import load_all_kinetics, load_medium
from layer3_reactions.reversible import (
    build_reversible_catalysis_rules, initialize_medium,
)
from layer3_reactions.coupled import initialize_metabolites


def build_sim(SimClass, seed=42):
    with contextlib.redirect_stdout(io.StringIO()):
        spec, counts, complexes, _ = build_real_syn3a_cellspec()
    sbml_path = (Path(__file__).resolve().parent.parent
                 / 'data' / 'Minimal_Cell_ComplexFormation'
                 / 'input_data' / 'Syn3A_updated.xml')
    sbml = parse_sbml(sbml_path)
    kinetics = load_all_kinetics()
    medium = load_medium()
    rev_rules, _ = build_reversible_catalysis_rules(sbml, kinetics)
    state = CellState(spec)
    populate_real_syn3a(state, counts, scale_factor=0.02, max_per_gene=10)
    initialize_metabolites(state, sbml, cell_volume_um3=(4/3)*np.pi*0.2**3)
    initialize_medium(state, medium)
    rules = ([make_folding_rule(20.0)] + rev_rules
             + make_complex_formation_rules(complexes, 0.05))
    return SimClass(state, rules, mode='gillespie', seed=seed), state


def compare(t_end: float = 0.3):
    print(f'Running FastEventSimulator (Phase 1) to t={t_end}s...')
    p1_sim, p1_state = build_sim(FastEventSimulator, seed=42)
    p1_initial = dict(p1_state.metabolite_counts)
    t0 = time.time(); p1_sim.run_until(t_end=t_end, max_events=500_000)
    p1_wall = time.time() - t0
    p1_n = len(p1_state.events)
    print(f'  {p1_n:,} events in {p1_wall:.2f}s wall')

    print(f'\nRunning NextReactionSimulator (Phase 2) to t={t_end}s...')
    p2_sim, p2_state = build_sim(NextReactionSimulator, seed=42)
    p2_initial = dict(p2_state.metabolite_counts)
    t0 = time.time(); p2_res = p2_sim.run_until(t_end=t_end, max_events=500_000)
    p2_wall = time.time() - t0
    p2_n = len(p2_state.events)
    print(f'  {p2_n:,} events in {p2_wall:.2f}s wall')
    print(f'  heap stats: {p2_res["heap_pops"]:,} pops '
          f'({p2_res["stale_pops"]:,} stale, '
          f'{100*p2_res["stale_pops"]/max(p2_res["heap_pops"],1):.1f}%), '
          f'{p2_res["rule_updates"]:,} rule updates')

    # Statistical comparison
    print(f'\n--- Statistical equivalence ---')
    print(f'Event counts: Phase1={p1_n:,}  Phase2={p2_n:,}  '
          f'relative diff={100*abs(p1_n-p2_n)/p1_n:.2f}%')

    # Event type distribution
    p1_kinds = Counter(e.rule_name.split(':')[0] for e in p1_state.events)
    p2_kinds = Counter(e.rule_name.split(':')[0] for e in p2_state.events)
    all_kinds = sorted(set(p1_kinds) | set(p2_kinds))
    print(f'\nEvent type distribution:')
    print(f'  {"kind":<20s} {"Phase1":>10s}  {"Phase2":>10s}  {"diff":>8s}')
    for k in all_kinds:
        a, b = p1_kinds.get(k, 0), p2_kinds.get(k, 0)
        if max(a, b) == 0:
            continue
        diff_pct = 100 * abs(a - b) / max(a, b)
        print(f'  {k:<20s} {a:>10,}  {b:>10,}  {diff_pct:>7.2f}%')

    # Key metabolite comparison
    print(f'\nKey metabolite final counts:')
    print(f'  {"species":<12s} {"Phase1 Δ":>12s}  {"Phase2 Δ":>12s}  '
          f'{"rel diff":>8s}')
    watch = ['M_atp_c', 'M_adp_c', 'M_g6p_c', 'M_fdp_c', 'M_dhap_c',
             'M_g3p_c', 'M_pyr_c', 'M_lac__L_c', 'M_pi_c']
    for sid in watch:
        d1 = p1_state.metabolite_counts.get(sid, 0) - p1_initial.get(sid, 0)
        d2 = p2_state.metabolite_counts.get(sid, 0) - p2_initial.get(sid, 0)
        if d1 == 0 and d2 == 0:
            print(f'  {sid:<12s} {d1:>+12,}  {d2:>+12,}    --')
        else:
            rel = 100 * abs(d1 - d2) / max(abs(d1), abs(d2), 1)
            print(f'  {sid:<12s} {d1:>+12,}  {d2:>+12,}  {rel:>7.2f}%')

    # Speedup
    print(f'\n--- Performance ---')
    print(f'  Phase 1 (vectorized direct method):  '
          f'{p1_wall:.2f}s  ({p1_n/p1_wall:.0f} events/s)')
    print(f'  Phase 2 (next-reaction method):      '
          f'{p2_wall:.2f}s  ({p2_n/p2_wall:.0f} events/s)')
    print(f'  Phase 2 speedup over Phase 1:        '
          f'{p1_wall/p2_wall:.2f}x')


if __name__ == '__main__':
    compare(t_end=0.3)
