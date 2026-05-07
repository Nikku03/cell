"""
Bit-identity check for the vectorised gene-expression propensity path.

Builds the same Priority-2 setup gene_expression.py uses (Real Syn3A spec
+ folding + reversible catalysis + complex formation + 50 gene-expression
rules per locus across the top-50 expressed genes), runs the reference
EventSimulator and the FastEventSimulator with the same seed, and asserts
the two produce the same event stream.

If this fails the vectorised gex kernel has drifted from the reference
closures and must be reverted.
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from layer0_genome.syn3a_real import build_real_syn3a_cellspec
from layer2_field.dynamics import CellState, EventSimulator
from layer2_field.fast_dynamics import FastEventSimulator
from layer2_field.real_syn3a_rules import (
    populate_real_syn3a, make_folding_rule, make_complex_formation_rules,
)
from layer3_reactions.coupled import initialize_metabolites
from layer3_reactions.gene_expression import (
    initialize_gene_expression_state,
    make_transcription_rule, make_translation_rule,
    make_mrna_degradation_rule, make_protein_degradation_rule,
)
from layer3_reactions.kinetics import load_all_kinetics, load_medium
from layer3_reactions.reversible import (
    build_reversible_catalysis_rules, initialize_medium,
)
from layer3_reactions.sbml_parser import parse_sbml


def build_sim(SimClass, seed: int = 42, n_top_genes: int = 50):
    with contextlib.redirect_stdout(io.StringIO()):
        spec, counts, complexes, _ = build_real_syn3a_cellspec()
    sbml_path = (THIS.parent.parent / 'data' / 'Minimal_Cell_ComplexFormation'
                 / 'input_data' / 'Syn3A_updated.xml')
    sbml = parse_sbml(sbml_path)
    kinetics = load_all_kinetics()
    medium = load_medium()
    rev_rules, _ = build_reversible_catalysis_rules(sbml, kinetics)

    state = CellState(spec)
    populate_real_syn3a(state, counts, scale_factor=0.02, max_per_gene=10)
    initialize_metabolites(state, sbml,
                           cell_volume_um3=(4 / 3) * np.pi * 0.2 ** 3)
    initialize_medium(state, medium)
    initialize_gene_expression_state(state, scale_factor=0.1, seed=seed)

    top = sorted(state.spec.proteins.keys(),
                 key=lambda g: state.mrna_counts.get(g, 0),
                 reverse=True)[:n_top_genes]
    gex = []
    for g in top:
        p = spec.proteins[g]
        length_aa = max(10, p.length or 100)
        length_nt = length_aa * 3 + 100
        gex.append(make_transcription_rule(g, length_nt))
        gex.append(make_translation_rule(g, length_aa))
        gex.append(make_mrna_degradation_rule(g, length_nt))
        gex.append(make_protein_degradation_rule(g))

    rules = (
        [make_folding_rule(20.0)]
        + rev_rules
        + make_complex_formation_rules(complexes, 0.05)
        + gex
    )
    sim = SimClass(state, rules, mode='gillespie', seed=seed)
    return sim, state, len(rules), len(gex)


def assert_event_streams_match(t_end: float = 0.05, max_check: int = 500):
    print(f'Reference EventSimulator -> t_end={t_end:.3f}s')
    py_sim, py_state, n_rules, n_gex = build_sim(EventSimulator)
    print(f'  {n_rules} rules total, {n_gex} of them gene-expression')
    t0 = time.time()
    py_sim.run_until(t_end=t_end, max_events=2_000_000)
    py_wall = time.time() - t0
    print(f'  {len(py_state.events):,} events in {py_wall:.2f}s '
          f'({len(py_state.events) / max(py_wall, 1e-9):.0f} ev/s)')

    print(f'\nFastEventSimulator -> t_end={t_end:.3f}s')
    fs_sim, fs_state, _, _ = build_sim(FastEventSimulator)
    t0 = time.time()
    fs_sim.run_until(t_end=t_end, max_events=2_000_000)
    fs_wall = time.time() - t0
    print(f'  {len(fs_state.events):,} events in {fs_wall:.2f}s '
          f'({len(fs_state.events) / max(fs_wall, 1e-9):.0f} ev/s)')
    print(f'  speedup vs reference: {py_wall / max(fs_wall, 1e-9):.2f}x')

    py_n, fs_n = len(py_state.events), len(fs_state.events)
    assert py_n > 0, 'reference produced zero events; bad fixture'
    assert py_n == fs_n, f'event count mismatch: py={py_n} fast={fs_n}'

    common = min(py_n, fs_n, max_check)
    for i in range(common):
        p = py_state.events[i]
        f = fs_state.events[i]
        assert p.rule_name == f.rule_name, (
            f'rule_name diverged at event {i}: py={p.rule_name!r} '
            f'fast={f.rule_name!r} t_py={p.time*1e3:.4f}ms '
            f't_fast={f.time*1e3:.4f}ms')
        assert abs(p.time - f.time) < 1e-12, (
            f'time diverged at event {i}: py={p.time:.12f} '
            f'fast={f.time:.12f} ({p.rule_name})')
    print(f'  first {common} events: rule_name + time MATCH')

    diffs = []
    for sid in sorted(set(py_state.metabolite_counts)
                       | set(fs_state.metabolite_counts)):
        p = py_state.metabolite_counts.get(sid, 0)
        f = fs_state.metabolite_counts.get(sid, 0)
        if p != f:
            diffs.append((sid, p, f))
    if diffs:
        for sid, p, f in diffs[:5]:
            print(f'  METABOLITE DIFF  {sid}: py={p} fast={f}')
    assert not diffs, f'{len(diffs)} metabolite counts disagree'

    py_mrna = getattr(py_state, 'mrna_counts', {})
    fs_mrna = getattr(fs_state, 'mrna_counts', {})
    mrna_diffs = [(g, py_mrna[g], fs_mrna.get(g, 0))
                   for g in py_mrna
                   if py_mrna[g] != fs_mrna.get(g, 0)]
    assert not mrna_diffs, f'mrna_counts disagree: {mrna_diffs[:5]}'
    print('  metabolite + mrna_counts state: MATCH')


def test_gex_vectorized_matches_reference():
    assert_event_streams_match(t_end=0.05, max_check=500)


if __name__ == '__main__':
    assert_event_streams_match(t_end=0.05, max_check=500)
    print('\nOK: vectorised gex propensity is bit-identical to reference.')
