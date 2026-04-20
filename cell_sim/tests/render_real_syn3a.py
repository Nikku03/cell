"""
Render an MP4 of the real Syn3A simulation.

This runs the event-driven simulator with real data from the
Luthey-Schulten lab:
  - Real proteome (458 genes from GenBank CP016816)
  - Real protein initial counts (from comparative proteomics)
  - Real k_cat values (from kinetic_params.xlsx)
  - Real 24 known complexes with gene compositions

Then it renders a 4-panel video showing the simulation over time.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time as time_module
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FFMpegWriter
from collections import defaultdict, deque, Counter

from layer2_field.real_syn3a_rules import (
    build_real_syn3a_cellspec, populate_real_syn3a, build_enzyme_map,
    make_folding_rule, make_catalysis_rules, make_complex_formation_rules,
)
from layer2_field.dynamics import CellState
from layer2_field.fast_dynamics import FastEventSimulator as EventSimulator

# ============================================================================
# Config
# ============================================================================
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'real_syn3a_movie'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE_FACTOR = 0.02
MAX_PER_GENE = 10
SIM_DURATION_S = 0.5          # 500 ms of real Syn3A life
FPS = 24
MOVIE_DURATION_S = 15.0
N_FRAMES = int(FPS * MOVIE_DURATION_S)

CELL_RADIUS = 1.0
BG_COLOR = '#0d1117'
CELL_MEMBRANE_COLOR = '#30363d'

STATE_COLORS = {
    'unfolded': '#ff6b6b',
    'native':   '#4dabf7',
    'bound':    '#51cf66',
    'active':   '#ffd43b',  # recently catalyzed
}

# ============================================================================
# Step 1: Run the simulation
# ============================================================================
print('=' * 60)
print('REAL Syn3A simulation')
print('=' * 60)

import io, contextlib
print('\nLoading real Syn3A data...')
with contextlib.redirect_stdout(io.StringIO()) as silent:
    spec, counts, complexes, kcats = build_real_syn3a_cellspec()
print(f'  {len(spec.proteins)} genes, {len(counts)} with counts, '
      f'{len(complexes)} complexes, {len(kcats)} reactions with k_cat')

enzyme_map = build_enzyme_map(spec, kcats.keys())

state = CellState(spec)
populate_real_syn3a(state, counts, scale_factor=SCALE_FACTOR, max_per_gene=MAX_PER_GENE)
print(f'  Populated {len(state.proteins):,} molecules')

rules = (
    [make_folding_rule(k_fold_per_s=20.0)]
    + make_catalysis_rules(kcats, enzyme_map)
    + make_complex_formation_rules(complexes, base_rate_per_s=0.1)
)
print(f'  Built {len(rules)} rules')

sim = EventSimulator(state, rules, mode='gillespie', seed=42)
print(f'\nRunning {SIM_DURATION_S}s simulated time...')
t0 = time_module.time()
stats = sim.run_until(t_end=SIM_DURATION_S, max_events=500_000)
print(f'  {stats["n_events"]:,} events in {time_module.time()-t0:.1f}s wall')
print(f'  {len(state.complexes)} complexes formed')

# Summarize event types
by_type = Counter()
for e in state.events:
    key = e.rule_name.split(':')[0] if ':' in e.rule_name else e.rule_name
    by_type[key] += 1
print('  Event breakdown:')
for t, n in by_type.most_common():
    print(f'    {t}: {n:,}')

# Assembly events specifically
assembly_events = [e for e in state.events if e.rule_name.startswith('assembly:')]
print(f'  Assembly events: {len(assembly_events)}')
for e in assembly_events[:5]:
    name = e.rule_name.split(':')[1]
    print(f'    t={e.time*1e6:8.0f} μs  {name}: {e.description[:80]}')

# ============================================================================
# Step 2: Assign 2D positions for visualization
# ============================================================================
print('\nAssigning visualization positions...')
all_inst_ids = sorted(state.proteins.keys())
rng = np.random.default_rng(0)
positions = {}
for inst_id in all_inst_ids:
    while True:
        x = rng.uniform(-CELL_RADIUS, CELL_RADIUS)
        y = rng.uniform(-CELL_RADIUS, CELL_RADIUS)
        if x*x + y*y < (CELL_RADIUS * 0.95)**2:
            positions[inst_id] = (x, y)
            break

# Bucket molecules by function class for color legends / filtering
fc_by_instance = {i: spec.proteins[state.proteins[i].gene_id].function_class
                  for i in all_inst_ids}

# ============================================================================
# Step 3: Build per-frame snapshots
# ============================================================================
print(f'Building {N_FRAMES} frames (sim {SIM_DURATION_S}s → {MOVIE_DURATION_S}s video)...')

sim_times = np.linspace(0, state.time, N_FRAMES + 1)[1:]
events_sorted = sorted(state.events, key=lambda e: e.time)

FLASH_FRAMES = 3

frame_states = []
ev_idx = 0
conformations = {i: state.proteins[i].conformation for i in all_inst_ids}
# But these are end-state; we need start-state. Let's reconstruct:
# Actually populate_real_syn3a set 25% unfolded, 75% native initially.
# We need to track 'initial' not current. Reconstruct from event log by working backwards.
# Simpler: re-initialize a mirror state to match beginning.
initial_conformations = {}
for inst_id in all_inst_ids:
    initial_conformations[inst_id] = state.proteins[inst_id].conformation
# Undo all folding events to get initial state
for e in events_sorted:
    if e.rule_name == 'folding':
        initial_conformations[e.participants[0]] = 'unfolded'
# Undo binding (mark as unbound at t=0)
initial_bound = {i: False for i in all_inst_ids}

current = {
    'conformations': dict(initial_conformations),
    'bound':         dict(initial_bound),
    'active_until':  {},   # inst_id -> time after which to stop highlighting as 'just catalyzed'
    'flashing':      {},   # inst_id -> (etype, frames_remaining)
    'n_complexes':   0,
    'complex_names_count': Counter(),
    'recent_events': deque(maxlen=5),
    'catalysis_count': 0,
    'catalysis_by_rxn': Counter(),
}

for frame_idx, t_frame in enumerate(sim_times):
    while ev_idx < len(events_sorted) and events_sorted[ev_idx].time <= t_frame:
        ev = events_sorted[ev_idx]
        if ev.rule_name == 'folding':
            pid = ev.participants[0]
            current['conformations'][pid] = 'native'
            current['flashing'][pid] = ('fold', FLASH_FRAMES)
        elif ev.rule_name.startswith('catalysis:'):
            pid = ev.participants[0]
            current['active_until'][pid] = t_frame + 0.005  # 5 ms highlight
            current['flashing'][pid] = ('active', 1)
            current['catalysis_count'] += 1
            current['catalysis_by_rxn'][ev.rule_name.split(':')[1]] += 1
        elif ev.rule_name.startswith('assembly:'):
            for pid in ev.participants:
                current['bound'][pid] = True
                current['flashing'][pid] = ('bind', FLASH_FRAMES * 2)
            current['n_complexes'] += 1
            current['complex_names_count'][ev.rule_name.split(':')[1]] += 1
        current['recent_events'].append((ev.time, ev.rule_name, ev.description))
        ev_idx += 1

    # Clean expired "active" markers
    expired = [i for i, tmax in current['active_until'].items() if tmax < t_frame]
    for i in expired:
        current['active_until'].pop(i, None)

    snap = {
        'sim_time': t_frame,
        'conformations': dict(current['conformations']),
        'bound':         dict(current['bound']),
        'active':        set(current['active_until'].keys()),
        'flashing':      dict(current['flashing']),
        'n_complexes':   current['n_complexes'],
        'complex_names_count': dict(current['complex_names_count']),
        'recent_events': list(current['recent_events']),
        'catalysis_count': current['catalysis_count'],
        'catalysis_by_rxn': dict(current['catalysis_by_rxn']),
    }
    frame_states.append(snap)

    # Decay flashes
    new_flash = {}
    for i, (et, r) in current['flashing'].items():
        if r > 1:
            new_flash[i] = (et, r - 1)
    current['flashing'] = new_flash

print(f'  Final: {frame_states[-1]["n_complexes"]} complexes, '
      f'{frame_states[-1]["catalysis_count"]:,} catalytic events')

# ============================================================================
# Step 4: Render the MP4
# ============================================================================
print(f'\nRendering {FPS} fps × {MOVIE_DURATION_S}s = {N_FRAMES} frames...')

plt.rcParams.update({
    'font.family': 'monospace',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': 'white',
    'axes.facecolor': BG_COLOR,
})

fig = plt.figure(figsize=(14, 8), facecolor=BG_COLOR)
gs = fig.add_gridspec(3, 3, left=0.04, right=0.98, top=0.94, bottom=0.06,
                      wspace=0.35, hspace=0.55)

# Main cell view
ax_cell = fig.add_subplot(gs[:, :2])
ax_cell.set_facecolor(BG_COLOR)
ax_cell.set_xlim(-CELL_RADIUS*1.15, CELL_RADIUS*1.15)
ax_cell.set_ylim(-CELL_RADIUS*1.15, CELL_RADIUS*1.15)
ax_cell.set_aspect('equal')
ax_cell.set_xticks([]); ax_cell.set_yticks([])
for s in ax_cell.spines.values(): s.set_color(CELL_MEMBRANE_COLOR)

membrane = Circle((0,0), CELL_RADIUS, fill=False,
                   color=CELL_MEMBRANE_COLOR, linewidth=3, zorder=1)
ax_cell.add_patch(membrane)
ax_cell.add_patch(Circle((0,0), CELL_RADIUS, fill=True, color='#1c2230',
                          alpha=0.5, zorder=0))

# One scatter per function class for separation
FC_MARKERS = {
    'enzyme':         ('o', 14, '#4dabf7'),
    'ribosomal':      ('D', 16, '#d946ef'),
    'transport':      ('s', 12, '#22d3ee'),
    'rna_processing': ('v', 10, '#a855f7'),
    'regulatory':     ('P', 10, '#f97316'),
    'structural_division': ('*', 18, '#f59e0b'),
    'unknown':        ('o', 8, '#94a3b8'),
    'other':          ('o', 10, '#64748b'),
}

scatters = {}
for fc, (marker, size, color) in FC_MARKERS.items():
    ids = [i for i in all_inst_ids if fc_by_instance[i] == fc]
    if not ids:
        continue
    xs = [positions[i][0] for i in ids]
    ys = [positions[i][1] for i in ids]
    sc = ax_cell.scatter(xs, ys, c=[color]*len(ids),
                          marker=marker, s=size, zorder=5,
                          edgecolors='white', linewidths=0.2, alpha=0.9)
    scatters[fc] = (sc, ids)

# Title + time
cell_title = ax_cell.text(0, CELL_RADIUS*1.17,
    'JCVI-Syn3A — real genome, real proteome, real kinetics',
    ha='center', va='bottom', fontsize=13, weight='bold', color='white')
cell_time = ax_cell.text(0, CELL_RADIUS*1.08, '',
    ha='center', va='bottom', fontsize=10, color='#8b9eb3')

# Legend (bottom)
legend_items = [
    ('enzyme',           FC_MARKERS['enzyme']),
    ('ribosomal',        FC_MARKERS['ribosomal']),
    ('transport',        FC_MARKERS['transport']),
    ('unknown',          FC_MARKERS['unknown']),
    ('structural_div.',  FC_MARKERS['structural_division']),
]
for i, (label, (marker, size, color)) in enumerate(legend_items):
    x = -CELL_RADIUS*1.1 + i * 0.48
    y = -CELL_RADIUS*1.12
    ax_cell.scatter([x], [y], c=[color], marker=marker, s=size*1.5,
                    edgecolors='white', linewidths=0.3)
    ax_cell.text(x + 0.06, y, label, fontsize=8, color='white', va='center')

# Right column panels
# Panel 1: population by state
ax_bars = fig.add_subplot(gs[0, 2])
ax_bars.set_facecolor(BG_COLOR)
for s in ax_bars.spines.values(): s.set_color('#30363d')
ax_bars.set_title('Molecule states', color='white', fontsize=10)
bar_labels = ['unfolded', 'native', 'bound', 'active']
bar_colors = [STATE_COLORS['unfolded'], STATE_COLORS['native'],
              STATE_COLORS['bound'], STATE_COLORS['active']]
bars = ax_bars.bar(bar_labels, [0]*4, color=bar_colors,
                    edgecolor='white', linewidth=0.3)
ax_bars.tick_params(labelsize=8)
ax_bars.set_ylabel('count', color='white', fontsize=8)

# Panel 2: catalytic activity (log scale)
ax_cat = fig.add_subplot(gs[1, 2])
ax_cat.set_facecolor(BG_COLOR)
for s in ax_cat.spines.values(): s.set_color('#30363d')
ax_cat.set_title('Catalytic turnovers', color='white', fontsize=10)
ax_cat.tick_params(labelsize=8)
cat_text = ax_cat.text(0.02, 0.95, '', fontsize=8, color='#c9d1d9',
                        family='monospace', va='top', ha='left',
                        transform=ax_cat.transAxes)
ax_cat.set_xticks([]); ax_cat.set_yticks([])

# Panel 3: Recent events log
ax_log = fig.add_subplot(gs[2, 2])
ax_log.set_facecolor(BG_COLOR)
for s in ax_log.spines.values(): s.set_color('#30363d')
ax_log.set_title('Recent events', color='white', fontsize=10)
ax_log.set_xticks([]); ax_log.set_yticks([])
log_text = ax_log.text(0.02, 0.95, '', fontsize=6.5, color='#c9d1d9',
                        family='monospace', va='top', ha='left',
                        transform=ax_log.transAxes)


def color_for(inst_id, snap):
    if inst_id in snap['flashing']:
        et, _ = snap['flashing'][inst_id]
        if et == 'fold':
            return '#ffffff'
        if et == 'bind':
            return '#ffb347'
        if et == 'active':
            return STATE_COLORS['active']
    if snap['bound'].get(inst_id):
        return STATE_COLORS['bound']
    if inst_id in snap['active']:
        return STATE_COLORS['active']
    if snap['conformations'].get(inst_id) == 'native':
        return None  # use the class default
    return STATE_COLORS['unfolded']


# Render frames
writer = FFMpegWriter(
    fps=FPS, codec='libx264', bitrate=3500,
    extra_args=['-pix_fmt', 'yuv420p', '-preset', 'medium'],
)

movie_path = OUTPUT_DIR / 'real_syn3a.mp4'

t_render_start = time_module.time()
with writer.saving(fig, str(movie_path), dpi=110):
    for frame_idx, snap in enumerate(frame_states):
        # Update molecule colors
        for fc, (sc, ids) in scatters.items():
            base_color = FC_MARKERS[fc][2]
            colors = []
            for i in ids:
                c = color_for(i, snap)
                colors.append(c if c is not None else base_color)
            sc.set_color(colors)

        cell_time.set_text(f'sim t = {snap["sim_time"]*1000:6.2f} ms   |   '
                            f'{snap["catalysis_count"]:,} catalytic events so far   |   '
                            f'{snap["n_complexes"]} complexes')

        # Update bar chart
        n_unf   = sum(1 for i in all_inst_ids
                       if snap['conformations'].get(i) == 'unfolded' and not snap['bound'].get(i))
        n_nat   = sum(1 for i in all_inst_ids
                       if snap['conformations'].get(i) == 'native' and not snap['bound'].get(i)
                       and i not in snap['active'])
        n_bound = sum(1 for i in all_inst_ids if snap['bound'].get(i))
        n_act   = len(snap['active'])
        counts_arr = [n_unf, n_nat, n_bound, n_act]
        for bar, c in zip(bars, counts_arr):
            bar.set_height(c)
        ax_bars.set_ylim(0, max(max(counts_arr) * 1.2, 10))

        # Update catalysis panel
        top_cat = sorted(snap['catalysis_by_rxn'].items(), key=lambda x: -x[1])[:8]
        cat_lines = ['Top reactions (real k_cat):']
        for rxn, n in top_cat:
            kc = kcats.get(rxn, 0)
            cat_lines.append(f'  {rxn:6s} {n:>6,} (k_cat={kc:>6.0f}/s)')
        if not top_cat:
            cat_lines.append('  (none yet)')
        cat_text.set_text('\n'.join(cat_lines))

        # Update event log
        lines = []
        for t, rule, desc in snap['recent_events'][-5:]:
            short = rule.split(':')[-1] if ':' in rule else rule
            lines.append(f'{t*1e6:8.0f}μs {short[:18]:18s}')
            lines.append(f'  {desc[:40]}')
        log_text.set_text('\n'.join(lines))

        writer.grab_frame()

        if (frame_idx + 1) % 30 == 0:
            elapsed = time_module.time() - t_render_start
            rate = (frame_idx + 1) / elapsed
            eta = (len(frame_states) - frame_idx - 1) / rate
            print(f'  Frame {frame_idx+1}/{len(frame_states)}  {rate:.1f} fps  ETA {eta:.0f}s')

plt.close(fig)
render_time = time_module.time() - t_render_start
size_mb = movie_path.stat().st_size / 1e6
print(f'\nSaved {movie_path}  ({size_mb:.2f} MB, rendered in {render_time:.1f}s)')
