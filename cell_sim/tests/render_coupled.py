"""
Render an MP4 of the COUPLED Syn3A simulation (Priority 1).

This video shows the payoff of Layer 2 ↔ Layer 3 coupling:
  - Enzymes catalyzing real reactions (with real k_cat)
  - Metabolites depleting and accumulating in real time
  - Mass conservation visible across ATP/ADP, NAD/NADH
  - Pathway flux: substrates → intermediates → products

Layout:
  Left:   Cell view with molecules (same as real_syn3a.mp4)
  Right top:    Metabolite concentration time-series (ATP, ADP, NADH, lactate)
  Right mid:    Energy balance (ATP vs ADP bar, NAD vs NADH bar)
  Right bot:    Event ticker + flux table
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time as time_module
import io, contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FFMpegWriter
from collections import defaultdict, deque, Counter

from layer0_genome.syn3a_real import build_real_syn3a_cellspec
from layer2_field.dynamics import CellState, EventSimulator
from layer2_field.real_syn3a_rules import (
    populate_real_syn3a, make_folding_rule, make_complex_formation_rules,
)
from layer3_reactions.sbml_parser import parse_sbml
from layer3_reactions.coupled import (
    build_coupled_catalysis_rules, initialize_metabolites,
    get_species_count, count_to_mM,
)


# ============================================================================
# Config
# ============================================================================
OUTPUT_DIR = Path('/home/claude/cell_sim/data/coupled_movie')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE_FACTOR = 0.02
MAX_PER_GENE = 10
SIM_DURATION_S = 0.2
FPS = 24
MOVIE_DURATION_S = 15.0
N_FRAMES = int(FPS * MOVIE_DURATION_S)

CELL_RADIUS = 1.0
BG = '#0d1117'
MEMBRANE = '#30363d'

# Metabolites to track in the chart (key glycolysis + energy)
WATCHED_METABOLITES = [
    ('M_atp_c',     'ATP',    '#ff6b6b'),
    ('M_adp_c',     'ADP',    '#ffa94d'),
    ('M_nadh_c',    'NADH',   '#4dabf7'),
    ('M_lac__L_c',  'lactate','#51cf66'),
    ('M_pyr_c',     'pyruvate','#ffd43b'),
    ('M_g6p_c',     'G6P',    '#d946ef'),
]


# ============================================================================
# Step 1: Run the coupled simulation, checkpointing metabolites
# ============================================================================
print('=' * 60)
print('Coupled Syn3A simulation + MP4 (Priority 1)')
print('=' * 60)

print('\nLoading data...')
with contextlib.redirect_stdout(io.StringIO()):
    spec, counts, complexes, kcats = build_real_syn3a_cellspec()

sbml_path = Path(__file__).parent.parent / 'data' / 'Minimal_Cell_ComplexFormation' / 'input_data' / 'Syn3A_updated.xml'
sbml = parse_sbml(sbml_path)

coupled_rules, _ = build_coupled_catalysis_rules(sbml, kcats)
print(f'  {len(coupled_rules)} coupled catalysis rules')

state = CellState(spec)
populate_real_syn3a(state, counts, scale_factor=SCALE_FACTOR, max_per_gene=MAX_PER_GENE)

cell_volume_um3 = (4/3) * np.pi * (0.2)**3
initialize_metabolites(state, sbml, cell_volume_um3=cell_volume_um3)
print(f'  {len(state.proteins):,} protein molecules')
print(f'  cell volume = {cell_volume_um3:.4f} μm³')

rules = (
    [make_folding_rule(k_fold_per_s=20.0)]
    + coupled_rules
    + make_complex_formation_rules(complexes, base_rate_per_s=0.05)
)

# Hook: checkpoint metabolite counts at each frame boundary.
# We run the simulation in small chunks and snapshot between chunks.
print(f'\nRunning {SIM_DURATION_S}s simulated time with per-frame checkpointing...')
sim = EventSimulator(state, rules, mode='gillespie', seed=42)

# Initial snapshot (t=0)
# Track all bar-chart species separately
BAR_SPECIES = ['M_atp_c', 'M_adp_c', 'M_nad_c', 'M_nadh_c', 'M_lac__L_c']

initial_counts = dict(state.metabolite_counts)
metabolite_trajectory = {sid: [get_species_count(state, sid)]
                         for sid, _, _ in WATCHED_METABOLITES}
# Also track bar-chart species that aren't in WATCHED
for sid in BAR_SPECIES:
    if sid not in metabolite_trajectory:
        metabolite_trajectory[sid] = [get_species_count(state, sid)]
time_points = [0.0]
event_count_trajectory = [0]
complex_count_trajectory = [0]

# Run in N_FRAMES chunks
chunk_times = np.linspace(0, SIM_DURATION_S, N_FRAMES + 1)[1:]
t0 = time_module.time()
for i, t_chunk_end in enumerate(chunk_times):
    sim.run_until(t_end=t_chunk_end, max_events=1_000_000)
    for sid in metabolite_trajectory:
        metabolite_trajectory[sid].append(get_species_count(state, sid))
    time_points.append(state.time)
    event_count_trajectory.append(len(state.events))
    complex_count_trajectory.append(len(state.complexes))
    if (i + 1) % 30 == 0:
        elapsed = time_module.time() - t0
        n_events = len(state.events)
        print(f'  Frame {i+1}/{N_FRAMES}  t={state.time*1000:.1f}ms  '
              f'events={n_events:,}  wall={elapsed:.1f}s')

sim_wall = time_module.time() - t0
print(f'\nSimulation complete: {len(state.events):,} events in {sim_wall:.1f}s wall')
print(f'  {len(state.complexes)} complexes assembled')

# Summary of metabolite changes
print('\nKey metabolite changes:')
for sid, label, _ in WATCHED_METABOLITES:
    init = initial_counts.get(sid, 0)
    final = get_species_count(state, sid)
    delta = final - init
    delta_mM = count_to_mM(abs(delta), state.metabolite_volume_L) * (1 if delta >= 0 else -1)
    print(f'  {label:8s} {init:>10,} → {final:>10,}  ({delta:>+8,}  {delta_mM:+.3f} mM)')

# ============================================================================
# Step 2: Prepare molecule positions & per-frame enzyme activity
# ============================================================================
print('\nAssigning molecule positions...')
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

fc_by_instance = {i: spec.proteins[state.proteins[i].gene_id].function_class
                  for i in all_inst_ids}

# Build per-frame "which enzymes fired in this chunk" for flash rendering.
# Use time_points as chunk boundaries.
events_sorted = sorted(state.events, key=lambda e: e.time)
chunk_events = [[] for _ in range(N_FRAMES)]
ev_idx = 0
for frame_idx in range(N_FRAMES):
    t_start = time_points[frame_idx]
    t_end = time_points[frame_idx + 1]
    while ev_idx < len(events_sorted) and events_sorted[ev_idx].time <= t_end:
        chunk_events[frame_idx].append(events_sorted[ev_idx])
        ev_idx += 1

# Rolling "recently active" enzymes for the cell visualization
active_duration_frames = 2  # stay yellow for 2 frames after firing
active_counter = {}  # inst_id -> frames remaining as 'active'
# Pre-compute per-frame active sets & recent-event logs & top-reactions
frame_data = []
recent_events = deque(maxlen=5)
cumulative_cat_by_rxn = Counter()

for frame_idx in range(N_FRAMES):
    # Decay active counters
    for iid in list(active_counter.keys()):
        active_counter[iid] -= 1
        if active_counter[iid] <= 0:
            del active_counter[iid]

    # Mark enzymes that fired in this chunk
    for ev in chunk_events[frame_idx]:
        if ev.rule_name.startswith('catalysis:') and ev.participants:
            active_counter[ev.participants[0]] = active_duration_frames
            rxn_short = ev.rule_name.split(':')[1]
            cumulative_cat_by_rxn[rxn_short] += 1
            recent_events.append((ev.time, rxn_short, ev.description))

    # Snapshot
    frame_data.append({
        't': time_points[frame_idx + 1],
        'active_set': set(active_counter.keys()),
        'recent_events': list(recent_events),
        'top_reactions': cumulative_cat_by_rxn.most_common(8),
        'n_events_cumulative': event_count_trajectory[frame_idx + 1],
        'n_complexes': complex_count_trajectory[frame_idx + 1],
    })

# ============================================================================
# Step 3: Build figure
# ============================================================================
print(f'\nRendering {N_FRAMES} frames...')

plt.rcParams.update({
    'font.family': 'monospace',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': 'white',
    'axes.facecolor': BG,
})

fig = plt.figure(figsize=(16, 9), facecolor=BG)
gs = fig.add_gridspec(4, 3, left=0.03, right=0.99, top=0.94, bottom=0.05,
                      wspace=0.32, hspace=0.55)

# === Cell view (left, large) ===
ax_cell = fig.add_subplot(gs[:, :2])
ax_cell.set_facecolor(BG)
ax_cell.set_xlim(-CELL_RADIUS*1.15, CELL_RADIUS*1.15)
ax_cell.set_ylim(-CELL_RADIUS*1.15, CELL_RADIUS*1.15)
ax_cell.set_aspect('equal')
ax_cell.set_xticks([]); ax_cell.set_yticks([])
for s in ax_cell.spines.values(): s.set_color(MEMBRANE)

ax_cell.add_patch(Circle((0,0), CELL_RADIUS, fill=False, color=MEMBRANE, linewidth=3, zorder=1))
ax_cell.add_patch(Circle((0,0), CELL_RADIUS, fill=True, color='#1c2230', alpha=0.5, zorder=0))

FC_MARKERS = {
    'enzyme':              ('o', 16, '#4dabf7'),
    'ribosomal':           ('D', 14, '#d946ef'),
    'transport':           ('s', 12, '#22d3ee'),
    'rna_processing':      ('v', 10, '#a855f7'),
    'regulatory':          ('P', 10, '#f97316'),
    'structural_division': ('*', 18, '#f59e0b'),
    'unknown':             ('o', 8,  '#64748b'),
    'other':               ('o', 10, '#94a3b8'),
}
ACTIVE_COLOR = '#ffd43b'

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

title_text = ax_cell.text(0, CELL_RADIUS*1.17,
    'JCVI-Syn3A — coupled Layer 2 (events) + Layer 3 (metabolites)',
    ha='center', va='bottom', fontsize=13, weight='bold', color='white')
subtitle_text = ax_cell.text(0, CELL_RADIUS*1.085, '',
    ha='center', va='bottom', fontsize=10, color='#8b9eb3')

# === Metabolite time-series (top right) ===
ax_traj = fig.add_subplot(gs[0:2, 2])
ax_traj.set_facecolor(BG)
for s in ax_traj.spines.values(): s.set_color(MEMBRANE)
ax_traj.set_title('Metabolite concentrations (mM)', color='white', fontsize=11)
ax_traj.set_xlabel('simulated time (ms)', fontsize=9)
ax_traj.tick_params(labelsize=8)
ax_traj.grid(True, color='#30363d', alpha=0.3)

traj_lines = {}
for sid, label, color in WATCHED_METABOLITES:
    line, = ax_traj.plot([], [], color=color, linewidth=2, label=label)
    traj_lines[sid] = line
ax_traj.legend(loc='upper left', fontsize=8, facecolor=BG,
               edgecolor=MEMBRANE, labelcolor='white', ncol=2)
ax_traj.set_xlim(0, SIM_DURATION_S * 1000)

# === ATP/ADP balance + NAD/NADH balance (middle right) ===
ax_bal = fig.add_subplot(gs[2, 2])
ax_bal.set_facecolor(BG)
for s in ax_bal.spines.values(): s.set_color(MEMBRANE)
ax_bal.set_title('Energy & redox balance (mM)', color='white', fontsize=11)
bal_labels = ['ATP', 'ADP', 'NAD+', 'NADH', 'lactate']
bal_colors = ['#ff6b6b', '#ffa94d', '#a1c4fd', '#4dabf7', '#51cf66']
bal_bars = ax_bal.bar(bal_labels, [0]*5, color=bal_colors,
                     edgecolor='white', linewidth=0.3)
ax_bal.tick_params(labelsize=8)

# === Event ticker + top reactions (bottom right) ===
ax_log = fig.add_subplot(gs[3, 2])
ax_log.set_facecolor(BG)
for s in ax_log.spines.values(): s.set_color(MEMBRANE)
ax_log.set_title('Top reactions & recent events', color='white', fontsize=10)
ax_log.set_xticks([]); ax_log.set_yticks([])
log_text = ax_log.text(0.02, 0.97, '', fontsize=7, color='#c9d1d9',
                       family='monospace', va='top', ha='left',
                       transform=ax_log.transAxes)


# Compute max y for trajectory panel (mM range across all watched)
vol_L = state.metabolite_volume_L
traj_mM_max = 0
for sid, _, _ in WATCHED_METABOLITES:
    counts_arr = metabolite_trajectory[sid]
    mM_arr = [count_to_mM(c, vol_L) for c in counts_arr]
    traj_mM_max = max(traj_mM_max, max(mM_arr) if mM_arr else 0)
ax_traj.set_ylim(0, traj_mM_max * 1.15)

# ============================================================================
# Step 4: Render frames
# ============================================================================
writer = FFMpegWriter(
    fps=FPS, codec='libx264', bitrate=3500,
    extra_args=['-pix_fmt', 'yuv420p', '-preset', 'medium'],
)
movie_path = OUTPUT_DIR / 'coupled_syn3a.mp4'

render_t0 = time_module.time()
with writer.saving(fig, str(movie_path), dpi=110):
    for frame_idx in range(N_FRAMES):
        fd = frame_data[frame_idx]

        # Update enzyme colors — mark active enzymes yellow
        active = fd['active_set']
        for fc, (sc, ids) in scatters.items():
            base = FC_MARKERS[fc][2]
            colors = [ACTIVE_COLOR if iid in active else base for iid in ids]
            sc.set_color(colors)

        t_ms = fd['t'] * 1000
        subtitle_text.set_text(
            f"sim t = {t_ms:6.2f} ms   |   "
            f"{fd['n_events_cumulative']:,} events   |   "
            f"{fd['n_complexes']} complexes   |   "
            f"{len(active)} enzymes active this frame"
        )

        # Update trajectory plot
        t_arr_ms = np.array(time_points[:frame_idx + 2]) * 1000
        for sid, _, _ in WATCHED_METABOLITES:
            counts_arr = metabolite_trajectory[sid][:frame_idx + 2]
            mM_arr = [count_to_mM(c, vol_L) for c in counts_arr]
            traj_lines[sid].set_data(t_arr_ms, mM_arr)

        # Update bar chart (current values from per-frame trajectory)
        atp  = count_to_mM(metabolite_trajectory['M_atp_c'][frame_idx + 1], vol_L)
        adp  = count_to_mM(metabolite_trajectory['M_adp_c'][frame_idx + 1], vol_L)
        nad  = count_to_mM(metabolite_trajectory['M_nad_c'][frame_idx + 1], vol_L)
        nadh = count_to_mM(metabolite_trajectory['M_nadh_c'][frame_idx + 1], vol_L)
        lac  = count_to_mM(metabolite_trajectory['M_lac__L_c'][frame_idx + 1], vol_L)

        bal_values = [atp, adp, nad, nadh, lac]
        for bar, v in zip(bal_bars, bal_values):
            bar.set_height(v)
        ax_bal.set_ylim(0, max(max(bal_values) * 1.2, 4))

        # Update log panel
        lines = ['Top 8 reactions (cumulative):']
        for rxn, n in fd['top_reactions']:
            kc = kcats.get(rxn, 0)
            lines.append(f'  {rxn:8s} {n:>6,}  (k_cat {kc:>5.0f}/s)')
        lines.append('')
        lines.append('Recent events:')
        for t, rxn, desc in fd['recent_events'][-3:]:
            lines.append(f'  {t*1e3:5.2f}ms {desc[:38]}')
        log_text.set_text('\n'.join(lines))

        writer.grab_frame()

        if (frame_idx + 1) % 30 == 0:
            elapsed = time_module.time() - render_t0
            rate = (frame_idx + 1) / elapsed
            eta = (N_FRAMES - frame_idx - 1) / rate
            print(f'  Rendered {frame_idx+1}/{N_FRAMES}  {rate:.1f} fps  ETA {eta:.0f}s')

plt.close(fig)
render_time = time_module.time() - render_t0
size_mb = movie_path.stat().st_size / 1e6
print(f'\nSaved: {movie_path}  ({size_mb:.2f} MB, rendered in {render_time:.1f}s)')
