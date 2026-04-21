"""
Essential-vs-non-essential gene knockout test.

The thesis: a dynamical whole-cell simulator should produce measurably
DIFFERENT trajectories when you knock out a metabolically essential
gene vs a non-essential gene. If that signal exists, we have the
foundation for a full genome-wide essentiality screen — something
FBA cannot do mechanistically.

Test panel (from Breuer et al. 2019 "Essential metabolism for a
minimal cell", Fig. 4, explicitly labeled):

  Essential:
    - JCVISYN3A_0445 (pgi)   — glucose-6-phosphate isomerase, core glycolysis
    - JCVISYN3A_0779 (ptsG)  — glucose PTS transporter, primary sugar uptake

  Non-essential:
    - JCVISYN3A_0522 (ftsZ)  — cell division (not needed for 1s of metabolism)
    - JCVISYN3A_0305         — uncharacterized metallopeptidase

  Control:
    - Wild-type (no knockout)

Each condition runs the simulator at the same scale, seed, and bio
time. We record:
  - ATP trajectory (primary metabolic signal)
  - G6P trajectory (glucose entry point — should crash in ptsG/pgi KO)
  - Pyruvate trajectory (glycolysis output)
  - Total event count
  - Fraction of events for each knocked-out reaction (should be 0 for
    directly-blocked rules, unchanged for unrelated rules)

Pass criterion: the two essential KOs show visibly different trajectories
from the two non-essential KOs. "Visibly different" = deviation
from WT >20% on ATP or G6P by t=500ms, for essential; <5% for non-essential.

Usage:
    python tests/test_knockouts.py

Env vars:
  KO_SCALE     scale factor (default 0.5 — balances signal vs runtime)
  KO_T_END     biological time seconds (default 0.5)
  KO_SEED      RNG seed (default 42)
  KO_USE_RUST  '1' to use the Rust backend

Runtime: ~20-30 min at 50% scale × 0.5s × 5 conditions on Colab CPU.
"""
from __future__ import annotations

import os
import sys
import io
import time
import contextlib
import copy
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


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SCALE    = float(os.environ.get('KO_SCALE', '0.5'))
T_END    = float(os.environ.get('KO_T_END', '0.5'))
SEED     = int(os.environ.get('KO_SEED', '42'))
USE_RUST = os.environ.get('KO_USE_RUST', '0') == '1'

OUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'knockout_test'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def max_per_for(scale):
    if scale <= 0.02: return 10
    if scale <= 0.10: return 50
    if scale <= 0.25: return 125
    if scale <= 0.50: return 250
    return 500

MAX_PER = max_per_for(SCALE)


# ----------------------------------------------------------------------
# Conditions: (label, knockout_locus_or_None, expected_class)
# ----------------------------------------------------------------------
CONDITIONS = [
    ('wildtype',      None,             'control'),
    ('KO_pgi',        'JCVISYN3A_0445', 'essential'),
    ('KO_ptsG',       'JCVISYN3A_0779', 'essential'),
    ('KO_ftsZ',       'JCVISYN3A_0522', 'non-essential'),
    ('KO_0305',       'JCVISYN3A_0305', 'non-essential'),
]


TRACKED = [
    'M_atp_c', 'M_adp_c', 'M_amp_c',
    'M_g6p_c', 'M_f6p_c', 'M_fdp_c',
    'M_pyr_c', 'M_lac__L_c',
    'M_pi_c', 'M_ppi_c',
]


def pick_sim():
    if not USE_RUST:
        return FastEventSimulator, 'Python FastEventSimulator'
    try:
        from layer2_field.rust_dynamics import RustBackedFastEventSimulator
        return RustBackedFastEventSimulator, 'Rust-backed FastEventSimulator'
    except ImportError:
        print('  (KO_USE_RUST=1 but cell_sim_rust not installed; using Python Fast)')
        return FastEventSimulator, 'Python FastEventSimulator'


def build_state_with_knockout(knockout_locus=None):
    """
    Build a CellState identical to wild-type except the knocked-out
    gene has ZERO protein instances.

    We achieve this by filtering the `counts` dict before calling
    populate_real_syn3a. Setting count=0 naturally leads to n=max(1,...)
    which would still create 1 protein. Instead we pop the locus
    entirely so populate skips it.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        spec, counts, complexes, _ = build_real_syn3a_cellspec()

    # Knock out: remove the gene from the counts dict so no proteins get created
    if knockout_locus is not None:
        if knockout_locus in counts:
            del counts[knockout_locus]
        # Double safety: even if populate re-adds it somehow, we'll zero out below.

    sbml = parse_sbml(Path(__file__).resolve().parent.parent /
                     'data' / 'Minimal_Cell_ComplexFormation' /
                     'input_data' / 'Syn3A_updated.xml')
    kinetics = load_all_kinetics()
    medium = load_medium()
    rev_rules, _ = build_reversible_catalysis_rules(sbml, kinetics)

    state = CellState(spec)
    populate_real_syn3a(state, counts, scale_factor=SCALE, max_per_gene=MAX_PER)

    # Belt-and-suspenders: ensure truly zero instances of the knockout gene.
    # Remove any proteins that might have been created despite our filter.
    if knockout_locus is not None:
        # Find all protein instances for this locus and remove them from state
        ids_to_remove = [pid for pid, p in state.proteins.items()
                         if p.gene_id == knockout_locus]
        for pid in ids_to_remove:
            p = state.proteins.pop(pid, None)
            if p is not None:
                # Also remove from proteins_by_state buckets
                for key, bucket in list(state.proteins_by_state.items()):
                    if isinstance(bucket, set):
                        bucket.discard(pid)
                    elif isinstance(bucket, list) and pid in bucket:
                        bucket.remove(pid)

    initialize_metabolites(state, sbml, cell_volume_um3=(4/3)*np.pi*0.2**3)
    initialize_medium(state, medium)

    extra = build_missing_transport_rules(sbml, kinetics)
    rules = ([make_folding_rule(20.0)]
             + rev_rules + extra
             + make_complex_formation_rules(complexes, 0.05))

    return state, rules


def run_condition(label, knockout_locus, seed):
    """Run one simulation with progress streaming and trajectory capture."""
    print(f'\n{"="*60}')
    print(f'  Running: {label}' + (f' (knockout: {knockout_locus})' if knockout_locus else ''))
    print(f'{"="*60}')

    state, rules = build_state_with_knockout(knockout_locus)
    n_prot = sum(len(v) for v in state.proteins_by_state.values()
                  if isinstance(v, (list, set)))
    # Verify the knockout actually took effect
    if knockout_locus is not None:
        n_ko_proteins = sum(1 for p in state.proteins.values()
                             if p.gene_id == knockout_locus)
        print(f'  proteins (total)   = {n_prot:,}')
        print(f'  KO gene instances = {n_ko_proteins}  (should be 0)')
        assert n_ko_proteins == 0, f'Knockout failed: {n_ko_proteins} proteins of {knockout_locus} remain'
    else:
        print(f'  proteins (total)   = {n_prot:,}')

    SimClass, _ = pick_sim()
    sim = SimClass(state, rules, mode='gillespie', seed=seed)

    # Trajectory sampling
    initial = {sid: get_species_count(state, sid) for sid in TRACKED}
    traj = [(0.0, dict(initial))]
    dt_sample = 10e-3  # every 10 ms bio
    next_record = dt_sample

    t0 = time.time()
    last_print = t0
    chunk = 0.005

    while state.time < T_END:
        target = min(state.time + chunk, T_END)
        sim.run_until(t_end=target, max_events=50_000_000)

        while next_record <= state.time and next_record <= T_END:
            traj.append((next_record, {sid: get_species_count(state, sid) for sid in TRACKED}))
            next_record += dt_sample

        now = time.time()
        if now - last_print >= 10.0 or state.time >= T_END:
            wall = now - t0
            remain = T_END - state.time
            eta = (remain / max(state.time, 1e-9)) * wall if state.time > 0 else float('inf')
            print(f'    t_bio={state.time*1000:>6.1f} ms  events={len(state.events):>9,}'
                  f'  wall={wall:>5.1f}s  ETA={eta:>5.0f}s')
            last_print = now

        chunk = min(max(chunk, 0.005), 0.05)

    wall = time.time() - t0

    # Event counts by rule short-name
    cats = Counter()
    for e in state.events:
        if e.rule_name.startswith('catalysis:'):
            parts = e.rule_name.split(':')
            cats[parts[1]] += 1

    print(f'  DONE: {len(state.events):,} events in {wall:.1f}s')
    final = {sid: get_species_count(state, sid) for sid in TRACKED}
    return dict(
        label=label, knockout=knockout_locus,
        n_events=len(state.events), wall=wall,
        initial=initial, final=final, traj=traj, event_counts=cats,
    )


def write_trajectory_csv(results, path):
    """Write one CSV with all conditions side by side."""
    # Find shared time grid
    all_times = sorted({t for r in results for t, _ in r['traj']})
    with open(path, 'w') as f:
        header = ['t_ms']
        for r in results:
            for sid in TRACKED:
                header.append(f'{r["label"]}::{sid}')
        f.write(','.join(header) + '\n')

        # Build a {label: {t: row}} lookup
        lookups = {r['label']: {t: row for t, row in r['traj']} for r in results}

        for t in all_times:
            vals = [f'{t*1000:.3f}']
            for r in results:
                row = lookups[r['label']].get(t, {})
                for sid in TRACKED:
                    vals.append(str(row.get(sid, '')))
            f.write(','.join(vals) + '\n')


def write_summary(results, path):
    lines = []
    lines.append('=' * 80)
    lines.append(f'  Knockout test — scale={SCALE}  t_end={T_END}s  seed={SEED}')
    lines.append('=' * 80)
    lines.append('')

    wt = next(r for r in results if r['label'] == 'wildtype')
    wt_atp_delta = wt['final']['M_atp_c'] - wt['initial']['M_atp_c']
    wt_g6p_delta = wt['final']['M_g6p_c'] - wt['initial']['M_g6p_c']
    wt_events = wt['n_events']

    lines.append(f'{"condition":<16s}{"class":<14s}{"events":>10s}{"ΔATP":>10s}'
                 f'{"ΔG6P":>10s}{"ΔATP %WT":>12s}{"ΔG6P %WT":>12s}')
    lines.append('-' * 80)

    for r in results:
        label = r['label']
        cls = next((c for (lbl, _, c) in CONDITIONS if lbl == label), '?')
        atp_d = r['final']['M_atp_c'] - r['initial']['M_atp_c']
        g6p_d = r['final']['M_g6p_c'] - r['initial']['M_g6p_c']
        if label == 'wildtype':
            atp_pct = '—'
            g6p_pct = '—'
        else:
            atp_pct = f'{100*atp_d/(wt_atp_delta if wt_atp_delta != 0 else 1):+.0f}%'
            g6p_pct = f'{100*g6p_d/(wt_g6p_delta if wt_g6p_delta != 0 else 1):+.0f}%'
        lines.append(f'{label:<16s}{cls:<14s}{r["n_events"]:>10,}'
                     f'{atp_d:>+10,}{g6p_d:>+10,}{atp_pct:>12s}{g6p_pct:>12s}')

    lines.append('')
    lines.append('Interpretation:')
    lines.append('  - Essential KOs should show large deviation from wildtype')
    lines.append('  - Non-essential KOs should look similar to wildtype')
    lines.append('  - If signal is clear -> commit to full 452-gene screen')
    lines.append('  - If signal is noise -> need to fix wildtype decay first')

    # Also compute top-event-rule differences: does ptsG knockout actually stop GLCpts?
    lines.append('')
    lines.append('Event-count changes for 10 key reactions (events vs wildtype):')
    key_rxns = ['GLCpts', 'PGI', 'PFK', 'GAPD', 'PGK', 'PYK', 'LDH_L', 'ADK1', 'PTAr', 'GLYK']
    lines.append(f'  {"rxn":<10s}' + ''.join(f'{r["label"]:>14s}' for r in results))
    for rxn in key_rxns:
        row = f'  {rxn:<10s}'
        for r in results:
            n = r['event_counts'].get(rxn, 0)
            row += f'{n:>14,}'
        lines.append(row)

    text = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(text + '\n')
    return text


def plot_comparison(results, path):
    """Four-panel plot: ATP, G6P, pyruvate, lactate, one line per condition."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        colors = {
            'wildtype':  '#2a2a2a',
            'KO_pgi':    '#e63946',
            'KO_ptsG':   '#f77f00',
            'KO_ftsZ':   '#2a9d8f',
            'KO_0305':   '#457b9d',
        }
        linestyles = {
            'wildtype':  '-',
            'KO_pgi':    '-',
            'KO_ptsG':   '-',
            'KO_ftsZ':   '--',
            'KO_0305':   '--',
        }

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        def plot_panel(ax, metric, title):
            for r in results:
                times = [t*1000 for t, _ in r['traj']]
                vals = [row[metric] for _, row in r['traj']]
                ax.plot(times, vals,
                        label=r['label'],
                        color=colors.get(r['label'], 'gray'),
                        linestyle=linestyles.get(r['label'], '-'),
                        lw=1.8 if r['label'] == 'wildtype' else 1.4)
            ax.set_xlabel('biological time (ms)')
            ax.set_title(title)
            ax.legend(fontsize=9, loc='best')

        plot_panel(axes[0, 0], 'M_atp_c', 'ATP — essential KOs should diverge')
        plot_panel(axes[0, 1], 'M_g6p_c', 'G6P — ptsG/pgi KO should crash this')
        plot_panel(axes[1, 0], 'M_pyr_c', 'Pyruvate — downstream of glycolysis')
        plot_panel(axes[1, 1], 'M_lac__L_c', 'Lactate — terminal product')

        plt.suptitle(
            f'Gene knockout test — scale={SCALE}, t={T_END}s, seed={SEED}\n'
            f'Essential KOs (solid colors) should deviate from WT (black); '
            f'non-essential KOs (dashed) should overlap WT',
            fontsize=11, y=1.00)
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight')
        print(f'Saved: {path}')
    except Exception as e:
        print(f'(plot skipped: {e})')


def main():
    print(f'Gene knockout test — 5 conditions')
    print(f'  scale: {SCALE} ({int(SCALE*100)}%)')
    print(f'  max_per_gene: {MAX_PER}')
    print(f'  t_end: {T_END} s')
    print(f'  seed: {SEED}')
    SimClass, sim_name = pick_sim()
    print(f'  simulator: {sim_name}')

    results = []
    for label, ko_locus, _ in CONDITIONS:
        r = run_condition(label, ko_locus, SEED)
        results.append(r)

    write_trajectory_csv(results, OUT_DIR / 'knockout_trajectories.csv')
    summary_text = write_summary(results, OUT_DIR / 'knockout_summary.txt')
    plot_comparison(results, OUT_DIR / 'knockout_comparison.png')

    print()
    print(summary_text)


if __name__ == '__main__':
    main()
