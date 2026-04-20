"""
Correctness check: FastEventSimulator must produce identical events
to the reference EventSimulator given the same seed.

If this fails, the fast path is unsound and should not be used.
"""
from __future__ import annotations
import sys, io, contextlib, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from layer0_genome.syn3a_real import build_real_syn3a_cellspec
from layer2_field.dynamics import CellState, EventSimulator
from layer2_field.fast_dynamics import FastEventSimulator
from layer2_field.real_syn3a_rules import (
    populate_real_syn3a, make_folding_rule, make_complex_formation_rules,
)
from layer3_reactions.sbml_parser import parse_sbml
from layer3_reactions.kinetics import load_all_kinetics, load_medium
from layer3_reactions.reversible import (
    build_reversible_catalysis_rules, initialize_medium,
)
from layer3_reactions.coupled import initialize_metabolites


def build_priority_15_sim(SimClass, seed=42):
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
    initialize_metabolites(state, sbml,
                            cell_volume_um3=(4/3)*np.pi*0.2**3)
    initialize_medium(state, medium)

    rules = ([make_folding_rule(20.0)]
             + rev_rules
             + make_complex_formation_rules(complexes, 0.05))
    sim = SimClass(state, rules, mode='gillespie', seed=seed)
    return sim, state


def compare_runs(t_end=0.05):
    print(f'Running Python EventSimulator to t={t_end:.2f}s...')
    py_sim, py_state = build_priority_15_sim(EventSimulator)
    t0 = time.time()
    py_sim.run_until(t_end=t_end, max_events=1_000_000)
    py_wall = time.time() - t0
    print(f'  {len(py_state.events):,} events in {py_wall:.2f}s wall')

    print(f'\nRunning FastEventSimulator to t={t_end:.2f}s...')
    fs_sim, fs_state = build_priority_15_sim(FastEventSimulator)
    t0 = time.time()
    fs_sim.run_until(t_end=t_end, max_events=1_000_000)
    fs_wall = time.time() - t0
    print(f'  {len(fs_state.events):,} events in {fs_wall:.2f}s wall')

    # Identity check
    print(f'\n--- Identity check ---')
    py_n, fs_n = len(py_state.events), len(fs_state.events)
    print(f'  Event count: Py={py_n:,}  Fast={fs_n:,}  '
          f'{"MATCH" if py_n == fs_n else "MISMATCH"}')

    # Compare first N events and final-state metabolites
    common = min(py_n, fs_n)
    # Rule-name sequence over the first few hundred events
    disagreement = None
    for i in range(min(common, 500)):
        p, f = py_state.events[i], fs_state.events[i]
        if p.rule_name != f.rule_name:
            disagreement = (i, p, f)
            break
    if disagreement is None:
        print(f'  First {min(common, 500)} event rule names: MATCH')
    else:
        i, p, f = disagreement
        print(f'  Rule-name diverges at event {i}:')
        print(f'    Py:   {p.rule_name}  @ t={p.time*1000:.4f}ms')
        print(f'    Fast: {f.rule_name}  @ t={f.time*1000:.4f}ms')

    # Final metabolite comparison
    print(f'\n  Metabolite count comparison (top divergent):')
    diffs = []
    for sid in sorted(set(py_state.metabolite_counts) | set(fs_state.metabolite_counts)):
        p = py_state.metabolite_counts.get(sid, 0)
        f = fs_state.metabolite_counts.get(sid, 0)
        if p != f:
            diffs.append((sid, p, f, p - f))
    if not diffs:
        print(f'    All species match exactly.')
    else:
        for sid, p, f, d in sorted(diffs, key=lambda x: -abs(x[3]))[:10]:
            print(f'    {sid:20s}  Py={p:>10,}  Fast={f:>10,}  (Δ {d:+,})')
        print(f'    ... {len(diffs)} species disagree total')

    # Speedup
    print(f'\n--- Performance ---')
    print(f'  Python: {py_wall:.2f}s  ({py_n/max(py_wall, 1e-9):.0f} events/sec)')
    print(f'  Fast:   {fs_wall:.2f}s  ({fs_n/max(fs_wall, 1e-9):.0f} events/sec)')
    print(f'  Speedup: {py_wall/max(fs_wall, 1e-9):.2f}x')


if __name__ == '__main__':
    compare_runs(t_end=0.05)
