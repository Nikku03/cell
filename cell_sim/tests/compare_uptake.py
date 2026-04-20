"""
Compare whole-cell simulation WITH vs WITHOUT the nutrient-uptake patch.

Runs two simulations at the same scale/seed/t_end, one with the
`nutrient_uptake` patch enabled and one without, then writes a
side-by-side trajectory CSV and comparison plot.

Usage:
    python tests/compare_uptake.py

Env vars (all optional, same as render_whole_cell.py):
    WC_SCALE       scale factor (default 0.25 — tuning for Colab runtime)
    WC_T_END       biological time in seconds (default 1.0)
    WC_SEED        RNG seed (default 42)
    WC_USE_RUST    '1' to use the Rust backend

Output (under data/whole_cell_compare/):
    baseline_trajectory.csv    — no uptake patch
    uptake_trajectory.csv      — with patch
    comparison.png             — 4-panel chart
    summary.txt                — numeric comparison of final counts

At 25% scale × 1 s bio, each run is ~30 s wall. At 100%, ~20 min each.
Default is 25% to make the comparison fast and decisive.
"""
from __future__ import annotations

import os
import sys
import io
import time
import contextlib
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from layer0_genome.syn3a_real import build_real_syn3a_cellspec
from layer2_field.dynamics import CellState
from layer2_field.fast_dynamics import FastEventSimulator
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
from layer3_reactions.nutrient_uptake import build_missing_transport_rules


SCALE    = float(os.environ.get('WC_SCALE', '0.25'))
T_END    = float(os.environ.get('WC_T_END', '1.0'))
SEED     = int(os.environ.get('WC_SEED', '42'))
USE_RUST = os.environ.get('WC_USE_RUST', '0') == '1'

OUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'whole_cell_compare'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Max per gene scales with scale factor — same heuristic as render_whole_cell.py
def _max_per(scale):
    if scale <= 0.02: return 10
    if scale <= 0.10: return 50
    if scale <= 0.25: return 125
    if scale <= 0.50: return 250
    return 500

MAX_PER = _max_per(SCALE)

TRACKED = [
    'M_atp_c', 'M_adp_c', 'M_amp_c',
    'M_nad_c', 'M_nadh_c',
    'M_pi_c', 'M_ppi_c',
    'M_g6p_c', 'M_f6p_c', 'M_fdp_c',
    'M_dhap_c', 'M_g3p_c',
    'M_pep_c', 'M_pyr_c', 'M_lac__L_c',
    'M_ade_c', 'M_gua_c', 'M_ura_c',
    'M_fa_c', 'M_glyc_c', 'M_o2_c',
]


def pick_sim_class():
    if not USE_RUST:
        return FastEventSimulator, 'Python FastEventSimulator'
    try:
        from layer2_field.rust_dynamics import RustBackedFastEventSimulator
        return RustBackedFastEventSimulator, 'Rust-backed FastEventSimulator'
    except ImportError:
        print('  (WC_USE_RUST=1 but cell_sim_rust not installed; using Python Fast)')
        return FastEventSimulator, 'Python FastEventSimulator'


def build(with_uptake, scale, max_per, seed):
    """Build a simulator at the given scale, optionally with uptake patch."""
    with contextlib.redirect_stdout(io.StringIO()):
        spec, counts, complexes, _ = build_real_syn3a_cellspec()
    sbml = parse_sbml(Path(__file__).resolve().parent.parent /
                      'data' / 'Minimal_Cell_ComplexFormation' /
                      'input_data' / 'Syn3A_updated.xml')
    kinetics = load_all_kinetics()
    medium = load_medium()
    rev_rules, _ = build_reversible_catalysis_rules(sbml, kinetics)

    state = CellState(spec)
    populate_real_syn3a(state, counts, scale_factor=scale, max_per_gene=max_per)
    initialize_metabolites(state, sbml, cell_volume_um3=(4/3)*np.pi*0.2**3)
    initialize_medium(state, medium)

    extra = build_missing_transport_rules(sbml, kinetics) if with_uptake else []
    rules = ([make_folding_rule(20.0)]
             + rev_rules + extra
             + make_complex_formation_rules(complexes, 0.05))

    SimClass, _ = pick_sim_class()
    sim = SimClass(state, rules, mode='gillespie', seed=seed)
    return sim, state, rules, extra


def run_with_trajectory(sim, state, tracked, t_end,
                         traj_dt_ms=10.0, progress_dt_sec=10.0):
    """Run and record trajectory every traj_dt_ms of biological time."""
    initial = {sid: get_species_count(state, sid) for sid in tracked}
    traj = [(0.0, dict(initial))]
    next_record = traj_dt_ms * 1e-3
    traj_dt = traj_dt_ms * 1e-3

    t0 = time.time()
    last_print = t0
    chunk = 0.005

    while state.time < t_end:
        target = min(state.time + chunk, t_end)
        sim.run_until(t_end=target, max_events=100_000_000)

        while next_record <= state.time and next_record <= t_end:
            row = {sid: get_species_count(state, sid) for sid in tracked}
            traj.append((next_record, row))
            next_record += traj_dt

        now = time.time()
        if now - last_print >= progress_dt_sec or state.time >= t_end:
            wall = now - t0
            remaining = t_end - state.time
            bio_per_wall = state.time / max(wall, 1e-9)
            eta = remaining / max(bio_per_wall, 1e-9) if bio_per_wall > 0 else float('inf')
            print(f'    t_bio={state.time*1000:>7.1f} ms  '
                  f'events={len(state.events):>9,}  '
                  f'wall={wall:>5.1f}s  ETA={eta:>5.0f}s')
            last_print = now

        chunk = min(max(chunk, 0.005), 0.05)

    wall_total = time.time() - t0
    return traj, initial, wall_total


def write_trajectory(traj, tracked, path):
    with open(path, 'w') as f:
        f.write('t_ms,' + ','.join(tracked) + '\n')
        for t_bio, row in traj:
            vals = ','.join(str(row.get(sid, 0)) for sid in tracked)
            f.write(f'{t_bio*1000:.3f},{vals}\n')


def main():
    print('=' * 64)
    print(f'  Whole-cell uptake comparison')
    print('=' * 64)
    SimClass, sim_name = pick_sim_class()
    print(f'  scale           = {SCALE} ({int(SCALE*100)}%)')
    print(f'  max_per_gene    = {MAX_PER}')
    print(f'  biological time = {T_END} s')
    print(f'  seed            = {SEED}')
    print(f'  simulator       = {sim_name}')
    print()

    print('Run 1 / 2: BASELINE (no uptake patch)')
    sim_a, state_a, rules_a, _ = build(False, SCALE, MAX_PER, SEED)
    print(f'  rules: {len(rules_a)}')
    traj_a, init_a, wall_a = run_with_trajectory(sim_a, state_a, TRACKED, T_END)
    n_events_a = len(state_a.events)
    print(f'  DONE: {n_events_a:,} events in {wall_a:.1f}s wall')

    print()
    print('Run 2 / 2: WITH UPTAKE PATCH')
    sim_b, state_b, rules_b, extra_b = build(True, SCALE, MAX_PER, SEED)
    print(f'  rules: {len(rules_b)} ({len(extra_b)} new uptake rules)')
    traj_b, init_b, wall_b = run_with_trajectory(sim_b, state_b, TRACKED, T_END)
    n_events_b = len(state_b.events)
    print(f'  DONE: {n_events_b:,} events in {wall_b:.1f}s wall')

    write_trajectory(traj_a, TRACKED, OUT_DIR / 'baseline_trajectory.csv')
    write_trajectory(traj_b, TRACKED, OUT_DIR / 'uptake_trajectory.csv')
    print()
    print(f'Trajectories written to {OUT_DIR}')

    # ---- numeric summary ----
    print()
    lines = []
    lines.append('=' * 72)
    lines.append(f'  Uptake-patch comparison — final metabolite deltas (Δ = final - initial)')
    lines.append('=' * 72)
    lines.append(f'  scale={SCALE}  t_end={T_END}s  seed={SEED}')
    lines.append(f'  Baseline: {n_events_a:>9,} events, {wall_a:>5.1f}s wall, {len(rules_a)} rules')
    lines.append(f'  Patched:  {n_events_b:>9,} events, {wall_b:>5.1f}s wall, {len(rules_b)} rules')
    lines.append('')
    lines.append(f'  {"species":<18s}{"initial":>12s}{"Δ baseline":>14s}{"Δ patched":>14s}{"diff":>12s}')
    for sid in TRACKED:
        init = init_a.get(sid, 0)
        da = get_species_count(state_a, sid) - init
        db = get_species_count(state_b, sid) - init
        diff = db - da
        marker = ''
        if init > 0:
            pct_a = abs(da) / init
            if pct_a > 0.5 and abs(diff) > 100 and (diff > 0) == (da < 0):
                marker = '  ←rescued'
        lines.append(f'  {sid:<18s}{init:>12,}{da:>+14,}{db:>+14,}{diff:>+12,}{marker}')
    summary_text = '\n'.join(lines)
    with open(OUT_DIR / 'summary.txt', 'w') as f:
        f.write(summary_text + '\n')
    print(summary_text)

    # ---- comparison plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def series(traj, sid):
            return [t*1000 for t, _ in traj], [row[sid] for _, row in traj]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        ax = axes[0, 0]
        x, y_a = series(traj_a, 'M_atp_c'); _, y_b = series(traj_b, 'M_atp_c')
        ax.plot(x, y_a, color='#e63946', linewidth=1.8, label='baseline', linestyle='--')
        ax.plot(x, y_b, color='#e63946', linewidth=1.8, label='with uptake')
        _, y_a = series(traj_a, 'M_adp_c'); _, y_b = series(traj_b, 'M_adp_c')
        ax.plot(x, y_a, color='#457b9d', linewidth=1.8, linestyle='--')
        ax.plot(x, y_b, color='#457b9d', linewidth=1.8)
        ax.set_xlabel('biological time (ms)'); ax.set_ylabel('count')
        ax.set_title('ATP (red) and ADP (blue): dashed=baseline, solid=with uptake')
        ax.legend()

        ax = axes[0, 1]
        _, y_a = series(traj_a, 'M_amp_c'); _, y_b = series(traj_b, 'M_amp_c')
        ax.plot(x, y_a, color='#264653', linestyle='--', label='AMP baseline')
        ax.plot(x, y_b, color='#264653', label='AMP with uptake')
        ax.set_xlabel('biological time (ms)'); ax.set_ylabel('count')
        ax.set_title('AMP accumulation (should drop with uptake if salvage works)')
        ax.legend()

        ax = axes[1, 0]
        _, y_a = series(traj_a, 'M_ppi_c'); _, y_b = series(traj_b, 'M_ppi_c')
        ax.plot(x, y_a, color='#9d4edd', linestyle='--', label='baseline')
        ax.plot(x, y_b, color='#9d4edd', label='with uptake')
        ax.set_xlabel('biological time (ms)'); ax.set_ylabel('count')
        ax.set_title('Pyrophosphate (baseline crashed here)')
        ax.legend()

        ax = axes[1, 1]
        _, y_a = series(traj_a, 'M_g6p_c'); _, y_b = series(traj_b, 'M_g6p_c')
        ax.plot(x, y_a, color='#f77f00', linestyle='--', label='G6P baseline')
        ax.plot(x, y_b, color='#f77f00', label='G6P with uptake')
        _, y_a = series(traj_a, 'M_lac__L_c'); _, y_b = series(traj_b, 'M_lac__L_c')
        ax.plot(x, y_a, color='#e76f51', linestyle='--', label='lactate baseline')
        ax.plot(x, y_b, color='#e76f51', label='lactate with uptake')
        ax.set_xlabel('biological time (ms)'); ax.set_ylabel('count')
        ax.set_title('Carbon flow: G6P sourcing, lactate output')
        ax.legend()

        plt.tight_layout()
        plt.savefig(OUT_DIR / 'comparison.png', dpi=100, bbox_inches='tight')
        print(f'\nPlot written: {OUT_DIR / "comparison.png"}')
    except Exception as e:
        print(f'\n(plot skipped: {e})')


if __name__ == '__main__':
    main()
