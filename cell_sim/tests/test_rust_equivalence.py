"""
Correctness check: RustBackedFastEventSimulator must produce identical
events to FastEventSimulator given the same seed.

If this fails, the Rust hot path is unsound and must be debugged before
shipping. The test is deterministic and comprehensive:
  - same event count
  - same rule-name sequence (all events, not just first 500)
  - same event times (exact float64 equality)
  - same final metabolite counts (exact integer equality)
  - same final protein states
"""
from __future__ import annotations
import sys, io, contextlib, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from layer0_genome.syn3a_real import build_real_syn3a_cellspec
from layer2_field.dynamics import CellState
from layer2_field.fast_dynamics import FastEventSimulator
from layer2_field.rust_dynamics import RustBackedFastEventSimulator
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
    initialize_metabolites(state, sbml, cell_volume_um3=(4/3)*np.pi*0.2**3)
    initialize_medium(state, medium)

    rules = ([make_folding_rule(20.0)]
             + rev_rules
             + make_complex_formation_rules(complexes, 0.05))
    sim = SimClass(state, rules, mode='gillespie', seed=seed)
    return sim, state


def compare_runs(t_end=0.3):
    print(f'Running FastEventSimulator to t={t_end:.2f}s...')
    py_sim, py_state = build_priority_15_sim(FastEventSimulator)
    t0 = time.time()
    py_sim.run_until(t_end=t_end, max_events=1_000_000)
    py_wall = time.time() - t0
    print(f'  {len(py_state.events):,} events in {py_wall:.2f}s wall')

    print(f'\nRunning RustBackedFastEventSimulator to t={t_end:.2f}s...')
    rs_sim, rs_state = build_priority_15_sim(RustBackedFastEventSimulator)
    t0 = time.time()
    rs_sim.run_until(t_end=t_end, max_events=1_000_000)
    rs_wall = time.time() - t0
    print(f'  {len(rs_state.events):,} events in {rs_wall:.2f}s wall')

    # ---- identity checks ----
    print(f'\n--- Identity checks ---')
    n_py, n_rs = len(py_state.events), len(rs_state.events)
    ok_count = (n_py == n_rs)
    print(f'  Event count:      Py={n_py:,}  Rust={n_rs:,}  '
          f'{"MATCH" if ok_count else "MISMATCH"}')

    # Full rule-name sequence
    common = min(n_py, n_rs)
    rule_mismatch = None
    time_mismatch = None
    for i in range(common):
        p, r = py_state.events[i], rs_state.events[i]
        if p.rule_name != r.rule_name and rule_mismatch is None:
            rule_mismatch = (i, p, r)
        if p.time != r.time and time_mismatch is None:
            time_mismatch = (i, p, r)
        if rule_mismatch and time_mismatch:
            break
    ok_rules = rule_mismatch is None
    ok_times = time_mismatch is None
    print(f'  Rule-name seq ({common:,} events): '
          f'{"MATCH" if ok_rules else "MISMATCH"}')
    if rule_mismatch is not None:
        i, p, r = rule_mismatch
        print(f'    Diverges at event {i}:')
        print(f'      Py:   {p.rule_name:60s}  t={p.time*1000:.6f}ms')
        print(f'      Rust: {r.rule_name:60s}  t={r.time*1000:.6f}ms')
    print(f'  Event times  ({common:,} events): '
          f'{"MATCH (exact f64)" if ok_times else "MISMATCH"}')
    if time_mismatch is not None:
        i, p, r = time_mismatch
        print(f'    Diverges at event {i}:')
        print(f'      Py time:   {p.time!r}')
        print(f'      Rust time: {r.time!r}')
        print(f'      Δ = {(r.time - p.time)!r}')

    # Final metabolite counts
    diffs = []
    all_sids = set(py_state.metabolite_counts) | set(rs_state.metabolite_counts)
    for sid in sorted(all_sids):
        p = py_state.metabolite_counts.get(sid, 0)
        r = rs_state.metabolite_counts.get(sid, 0)
        if p != r:
            diffs.append((sid, p, r))
    ok_metab = len(diffs) == 0
    print(f'  Metabolite counts ({len(all_sids)} species): '
          f'{"MATCH" if ok_metab else f"MISMATCH ({len(diffs)} species)"}')
    if diffs:
        for sid, p, r in sorted(diffs, key=lambda x: -abs(x[1]-x[2]))[:5]:
            print(f'    {sid:30s}  Py={p:>10,}  Rust={r:>10,}  Δ={r-p:+,}')

    # Protein state distribution
    py_prot_states = {k: len(v) for k, v in py_state.proteins_by_state.items()}
    rs_prot_states = {k: len(v) for k, v in rs_state.proteins_by_state.items()}
    ok_prot = py_prot_states == rs_prot_states
    print(f'  Protein states:   '
          f'{"MATCH" if ok_prot else "MISMATCH"}')

    # ---- performance ----
    print(f'\n--- Performance ---')
    speedup = py_wall / max(rs_wall, 1e-9)
    print(f'  Python Fast:  {py_wall:.2f}s  '
          f'({n_py/max(py_wall, 1e-9):.0f} events/sec)')
    print(f'  Rust-backed:  {rs_wall:.2f}s  '
          f'({n_rs/max(rs_wall, 1e-9):.0f} events/sec)')
    print(f'  Speedup:      {speedup:.2f}x over Python Fast')

    all_ok = ok_count and ok_rules and ok_times and ok_metab and ok_prot
    print(f'\n{"=" * 60}')
    if all_ok:
        print(f'  BIT-IDENTITY VERIFIED ({n_py:,} events)')
    else:
        print(f'  FAILED - Rust backend is NOT bit-identical to Python.')
    print(f'{"=" * 60}')
    return all_ok


if __name__ == '__main__':
    ok = compare_runs(t_end=0.3)
    sys.exit(0 if ok else 1)
