"""
Whole-Syn3A-cell simulation at 100% scale.

All real proteomics counts (not scaled down), full metabolism, 1 second
of biological time.

Output:
  - data/whole_cell/summary.txt    : human-readable final report
  - data/whole_cell/trajectory.csv : key metabolites every 10 ms bio time
  - data/whole_cell/events.csv     : per-rule event counts (no per-event dump)
  - data/whole_cell/checkpoint.pkl : resumable state (optional)

Run from the cell_sim/ directory:
    python tests/render_whole_cell.py

Flags (environment variables, all optional):
  WC_SCALE       scale factor (default: 1.0 = 100%)
  WC_MAX_PER     per-gene cap for populate (default: 500)
  WC_T_END       biological time in seconds (default: 1.0)
  WC_SEED        RNG seed (default: 42)
  WC_USE_RUST    set to '1' to use RustBackedFastEventSimulator
  WC_CHECKPOINT  set to '1' to pickle state at end for resume
  WC_WITH_UPTAKE set to '0' to disable the nutrient-uptake patch
                 (default: '1' = patch ON, which fills in GLCpts, O2t,
                 FAt, GLYCt, etc. and synthetic nucleobase importers)

Wall time expectations (Colab High-RAM CPU):
  2%  scale, 1s bio :   ~4s
  10% scale, 1s bio :  ~60s
  50% scale, 1s bio : ~400s   (7 min)
  100% scale, 1s bio: ~1200s  (20 min with Rust; ~30 min pure Python)

Memory: ~2 GB at 100% scale.
"""
from __future__ import annotations

import os
import sys
import io
import time
import contextlib
import pickle
from collections import Counter
from pathlib import Path

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


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SCALE      = float(os.environ.get('WC_SCALE', '1.0'))
MAX_PER    = int(os.environ.get('WC_MAX_PER', '500'))
T_END      = float(os.environ.get('WC_T_END', '1.0'))
SEED       = int(os.environ.get('WC_SEED', '42'))
USE_RUST   = os.environ.get('WC_USE_RUST', '0') == '1'
CHECKPOINT = os.environ.get('WC_CHECKPOINT', '0') == '1'
WITH_UPTAKE = os.environ.get('WC_WITH_UPTAKE', '1') == '1'   # on by default

OUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'whole_cell'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAJECTORY_DT_MS = 10.0    # record one trajectory row per 10 ms bio time
PROGRESS_DT_SEC  = 10.0    # print progress line every 10 wall seconds


# ----------------------------------------------------------------------
# Metabolites to track in the trajectory CSV. Keep the list short —
# 2 million events generate a lot of data and we don't need per-species
# trajectories for every compound.
# ----------------------------------------------------------------------
TRACKED = [
    'M_atp_c', 'M_adp_c', 'M_amp_c',
    'M_nad_c', 'M_nadh_c',
    'M_pi_c', 'M_ppi_c',
    'M_g6p_c', 'M_f6p_c', 'M_fdp_c',
    'M_dhap_c', 'M_g3p_c',
    'M_pep_c', 'M_pyr_c', 'M_lac__L_c',
    'M_h_c', 'M_h2o_c',
]


def pick_simulator_class():
    """Use Rust backend if requested and available; fall back to Python Fast."""
    if not USE_RUST:
        return FastEventSimulator, 'Python FastEventSimulator'
    try:
        from layer2_field.rust_dynamics import RustBackedFastEventSimulator
        return RustBackedFastEventSimulator, 'Rust-backed FastEventSimulator'
    except ImportError:
        print('  (WC_USE_RUST=1 but cell_sim_rust not installed; using Python Fast)')
        return FastEventSimulator, 'Python FastEventSimulator'


def build_sim():
    """Build the simulator at the requested scale."""
    print(f'Building whole-cell simulation...')
    print(f'  scale           = {SCALE} ({int(SCALE*100)}%)')
    print(f'  max_per_gene    = {MAX_PER}')
    print(f'  biological time = {T_END} s')
    print(f'  seed            = {SEED}')

    with contextlib.redirect_stdout(io.StringIO()):
        spec, counts, complexes, _ = build_real_syn3a_cellspec()
    sbml_path = (Path(__file__).resolve().parent.parent
                 / 'data' / 'Minimal_Cell_ComplexFormation'
                 / 'input_data' / 'Syn3A_updated.xml')
    sbml = parse_sbml(sbml_path)
    kinetics = load_all_kinetics()
    medium = load_medium()
    rev_rules, _ = build_reversible_catalysis_rules(sbml, kinetics)

    # Missing nutrient uptake rules (GLCpts, GLYCt, O2t, FAt, CHOLt, TAGt,
    # and synthetic nucleobase imports). These fill in k_cats that the
    # Luthey-Schulten kinetic database doesn't provide but which the cell
    # needs to avoid starving. See layer3_reactions/nutrient_uptake.py.
    uptake_rules = []
    if WITH_UPTAKE:
        uptake_rules = build_missing_transport_rules(sbml, kinetics)

    state = CellState(spec)
    populate_real_syn3a(state, counts, scale_factor=SCALE, max_per_gene=MAX_PER)
    initialize_metabolites(state, sbml,
                           cell_volume_um3=(4/3)*np.pi*0.2**3)
    initialize_medium(state, medium)

    rules = ([make_folding_rule(20.0)]
             + rev_rules
             + uptake_rules
             + make_complex_formation_rules(complexes, 0.05))

    n_prot = sum(len(v) for v in state.proteins_by_state.values())
    n_metab = sum(v for v in state.metabolite_counts.values())
    print(f'  proteins        = {n_prot:>10,}')
    print(f'  metabolites     = {n_metab:>10,}')
    print(f'  rules           = {len(rules):>10,}')
    print(f'  uptake patch    = {"ON (" + str(len(uptake_rules)) + " extra rules)" if WITH_UPTAKE else "OFF"}')

    SimClass, sim_name = pick_simulator_class()
    print(f'  simulator       = {sim_name}')
    sim = SimClass(state, rules, mode='gillespie', seed=SEED)
    return sim, state, rules


def initial_counts(state, tracked):
    return {sid: get_species_count(state, sid) for sid in tracked}


def trajectory_row(state, tracked):
    return {sid: get_species_count(state, sid) for sid in tracked}


def run_with_progress(sim, state, rules, tracked):
    """
    Drive the simulator with periodic progress output and trajectory
    sampling. Returns (trajectory, event_counts, initial_counts).
    """
    initial = initial_counts(state, tracked)

    traj = []                  # list of (t_bio, {sid: count})
    traj_threshold = 0.0       # next bio time to record
    traj_dt = TRAJECTORY_DT_MS * 1e-3
    traj.append((0.0, dict(initial)))

    t0 = time.time()
    last_progress_wall = t0
    last_progress_bio = 0.0
    last_progress_events = 0

    # Run in small biological chunks so we can stream progress. Chunk size
    # grows as wall speed is measured.
    chunk = 0.005  # start: 5 ms bio time
    max_events = 100_000_000

    while state.time < T_END:
        target = min(state.time + chunk, T_END)
        events_before = len(state.events)
        sim.run_until(t_end=target, max_events=max_events)

        # Record trajectory point(s) passed during this chunk
        while traj_threshold <= state.time and traj_threshold <= T_END:
            traj.append((traj_threshold, trajectory_row(state, tracked)))
            traj_threshold += traj_dt

        # Progress stream
        now = time.time()
        if now - last_progress_wall >= PROGRESS_DT_SEC or state.time >= T_END:
            wall = now - t0
            n_events = len(state.events)
            events_per_sec_wall = n_events / max(wall, 1e-9)
            events_per_sec_bio = n_events / max(state.time, 1e-9)
            remaining_bio = T_END - state.time
            if state.time - last_progress_bio > 0:
                bio_per_wall = (state.time - last_progress_bio) / (now - last_progress_wall)
                eta = remaining_bio / max(bio_per_wall, 1e-9)
            else:
                eta = float('inf')
            print(f'  t_bio={state.time*1000:>7.1f} ms  '
                  f'events={n_events:>10,}  '
                  f'wall={wall:>6.1f}s  '
                  f'rate={events_per_sec_wall:>7,.0f}/s  '
                  f'ETA={eta:>5.0f}s')
            last_progress_wall = now
            last_progress_bio = state.time
            last_progress_events = n_events

        # Adapt chunk: aim for ~5 wall seconds per chunk
        wall_this_chunk = time.time() - (last_progress_wall - PROGRESS_DT_SEC)
        if wall_this_chunk > 0:
            dt_bio = target - (state.time - (target - state.time))
            # Keep chunk roughly 5s wall; don't grow unbounded
            chunk = min(max(chunk, 0.005), 0.1)

        if len(state.events) >= max_events:
            print(f'  hit max_events={max_events:,}, stopping')
            break

    wall_total = time.time() - t0
    event_counts = Counter(e.rule_name.split(':')[1] if ':' in e.rule_name else e.rule_name
                           for e in state.events)
    return traj, event_counts, initial, wall_total


def write_trajectory_csv(traj, tracked, path):
    """CSV with columns: t_ms, [sid_1, sid_2, ...]"""
    with open(path, 'w') as f:
        f.write('t_ms,' + ','.join(tracked) + '\n')
        for t_bio, row in traj:
            vals = ','.join(str(row.get(sid, 0)) for sid in tracked)
            f.write(f'{t_bio*1000:.3f},{vals}\n')


def write_events_csv(event_counts, path):
    """CSV of per-rule event counts (sorted descending)."""
    with open(path, 'w') as f:
        f.write('rule,count\n')
        for name, n in event_counts.most_common():
            f.write(f'{name},{n}\n')


def write_summary(state, initial, event_counts, wall_total, n_prot, path):
    """Human-readable summary of the run."""
    lines = []
    lines.append('=' * 64)
    lines.append(f'  Whole-Syn3A-cell simulation — final report')
    lines.append('=' * 64)
    lines.append(f'  Scale:            {int(SCALE*100)}%')
    lines.append(f'  Biological time:  {T_END} s')
    lines.append(f'  Wall time:        {wall_total:.1f} s')
    lines.append(f'  Proteins:         {n_prot:,}')
    lines.append(f'  Total events:     {len(state.events):,}')
    lines.append(f'  Events/s bio:     {len(state.events)/max(T_END, 1e-9):,.0f}')
    lines.append(f'  Events/s wall:    {len(state.events)/max(wall_total, 1e-9):,.0f}')
    lines.append('')
    lines.append('  Event breakdown (top 20):')
    for name, n in event_counts.most_common(20):
        pct = 100.0 * n / max(len(state.events), 1)
        lines.append(f'    {name:<32s} {n:>12,}  ({pct:5.2f}%)')
    lines.append('')
    lines.append('  Tracked metabolite changes:')
    lines.append(f'    {"species":<18s}{"initial":>12s}{"final":>12s}{"Δ count":>12s}{"Δ mM":>10s}')
    vol_L = state.metabolite_volume_L
    for sid in TRACKED:
        init = initial.get(sid, 0)
        final = get_species_count(state, sid)
        d = final - init
        dmM = count_to_mM(abs(d), vol_L) * (1 if d >= 0 else -1)
        lines.append(f'    {sid:<18s}{init:>12,}{final:>12,}{d:>+12,}{dmM:>+10.4f}')
    lines.append('')
    lines.append('  Output files written to: ' + str(OUT_DIR))
    lines.append('=' * 64)
    text = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(text + '\n')
    return text


def main():
    sim, state, rules = build_sim()
    print()
    print('Running...')
    traj, event_counts, initial, wall_total = run_with_progress(
        sim, state, rules, TRACKED)
    print()

    write_trajectory_csv(traj, TRACKED, OUT_DIR / 'trajectory.csv')
    write_events_csv(event_counts, OUT_DIR / 'events.csv')
    n_prot = sum(len(v) for v in state.proteins_by_state.values())
    summary = write_summary(state, initial, event_counts, wall_total,
                             n_prot, OUT_DIR / 'summary.txt')
    print(summary)

    if CHECKPOINT:
        # Minimal checkpoint: final metabolite counts + state.time + rng state.
        # Not the full state object (which contains unpickleable closures).
        ckpt = {
            'time': state.time,
            'metabolite_counts': dict(state.metabolite_counts),
            'event_count_by_rule': dict(event_counts),
            'seed': SEED, 'scale': SCALE, 't_end': T_END,
        }
        with open(OUT_DIR / 'checkpoint.pkl', 'wb') as f:
            pickle.dump(ckpt, f)
        print(f'  Checkpoint written: {OUT_DIR / "checkpoint.pkl"}')


if __name__ == '__main__':
    main()
