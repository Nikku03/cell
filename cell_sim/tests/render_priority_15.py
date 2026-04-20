"""
Priority 1.5 MP4: reversible MM + medium uptake.

Layout:
  Left column (2 cols): Cell view with proteins colored by compartment activity
  Top right:    Intracellular metabolite concentrations (ATP, ADP, G6P, etc.)
  Middle right: Forward vs Reverse flux bar chart (live) — shows equilibrium
  Bottom right: Transport/uptake counters — cell eating from medium
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

from layer2_field.dynamics import CellState
from layer2_field.fast_dynamics import FastEventSimulator as EventSimulator
from layer2_field.real_syn3a_rules import (
    populate_real_syn3a, make_folding_rule, make_complex_formation_rules,
)
from layer0_genome.syn3a_real import build_real_syn3a_cellspec
from layer3_reactions.sbml_parser import parse_sbml
from layer3_reactions.kinetics import load_all_kinetics, load_medium
from layer3_reactions.reversible import (
    build_reversible_catalysis_rules, initialize_medium,
)
from layer3_reactions.coupled import (
    initialize_metabolites, get_species_count, count_to_mM,
)

# Config
OUTPUT_DIR = Path('/home/claude/cell_sim/data/priority_15_movie')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE_FACTOR = 0.02
MAX_PER_GENE = 10
SIM_DURATION_S = 1.0          # 1 full second of cell life now possible with steady-state
FPS = 24
MOVIE_DURATION_S = 18.0
N_FRAMES = int(FPS * MOVIE_DURATION_S)
N_SNAPSHOTS = N_FRAMES * 2

CELL_RADIUS = 1.0
BG_COLOR = '#0d1117'

# Track these metabolites (cytoplasmic)
WATCHED_METS = [
    ('M_atp_c',    'ATP',    '#60a5fa'),
    ('M_adp_c',    'ADP',    '#3b82f6'),
    ('M_g6p_c',    'G6P',    '#f59e0b'),
    ('M_fdp_c',    'FDP',    '#dc2626'),
    ('M_dhap_c',   'DHAP',   '#a3e635'),
    ('M_g3p_c',    'G3P',    '#84cc16'),
    ('M_pyr_c',    'Pyr',    '#a855f7'),
    ('M_lac__L_c', 'Lac',    '#ec4899'),
    ('M_pi_c',     'Pi',     '#10b981'),
    ('M_nad_c',    'NAD+',   '#f97316'),
    ('M_nadh_c',   'NADH',   '#fbbf24'),
]

FC_MARKERS = {
    'enzyme':         ('o', 14, '#4dabf7'),
    'ribosomal':      ('D', 16, '#d946ef'),
    'transport':      ('s', 14, '#22d3ee'),  # bigger — we emphasize these
    'rna_processing': ('v', 10, '#a855f7'),
    'regulatory':     ('P', 10, '#f97316'),
    'structural_division': ('*', 18, '#f59e0b'),
    'unknown':        ('o', 6, '#64748b'),   # dimmer
    'other':          ('o', 8, '#64748b'),
}


# ============================================================================
# Step 1: Run the simulation
# ============================================================================
print('=' * 60)
print('Priority 1.5 — reversible MM + medium uptake')
print('=' * 60)

print('\nLoading data...')
with contextlib.redirect_stdout(io.StringIO()):
    spec, counts, complexes, _ = build_real_syn3a_cellspec()
sbml_path = Path(__file__).parent.parent / 'data' / 'Minimal_Cell_ComplexFormation' / 'input_data' / 'Syn3A_updated.xml'
sbml = parse_sbml(sbml_path)
kinetics = load_all_kinetics()
medium = load_medium()

rev_rules, _ = build_reversible_catalysis_rules(sbml, kinetics)

state = CellState(spec)
populate_real_syn3a(state, counts, scale_factor=SCALE_FACTOR, max_per_gene=MAX_PER_GENE)
initialize_metabolites(state, sbml, cell_volume_um3=(4/3)*np.pi*0.2**3)
initialize_medium(state, medium)

print(f'  {len(state.proteins):,} proteins, {len(rev_rules)} rules (fwd+rev+transport)')

rules = (
    [make_folding_rule(20.0)]
    + rev_rules
    + make_complex_formation_rules(complexes, base_rate_per_s=0.05)
)

initial_counts = dict(state.metabolite_counts)

# Figure out which rules are transport (for counting later)
rxns_short = sbml.reactions_by_short_name()
transport_rxn_set = set()
for rxn_name in kinetics:
    if rxn_name in rxns_short:
        sbml_rxn = rxns_short[rxn_name]
        if any('_e' in s for s in sbml_rxn.reactants) or any('_e' in s for s in sbml_rxn.products):
            transport_rxn_set.add(rxn_name)

# Run with periodic snapshots
print(f'\nSimulating {SIM_DURATION_S*1000:.0f} ms with {N_SNAPSHOTS} snapshots...')
snapshot_times = np.linspace(0, SIM_DURATION_S, N_SNAPSHOTS + 1)[1:]
met_history = {sid: [] for sid, _, _ in WATCHED_METS}
time_history = []

sim = EventSimulator(state, rules, mode='gillespie', seed=42)
t_start = time_module.time()
for i, t_snap in enumerate(snapshot_times):
    sim.run_until(t_end=t_snap, max_events=1_000_000)
    time_history.append(state.time)
    for sid, _, _ in WATCHED_METS:
        met_history[sid].append(get_species_count(state, sid))
    if (i + 1) % 30 == 0:
        print(f'  Snapshot {i+1}/{N_SNAPSHOTS}  t={state.time*1000:.1f}ms  '
              f'events={len(state.events):,}  wall={time_module.time()-t_start:.1f}s')

wall = time_module.time() - t_start
print(f'\nDone: {len(state.events):,} events in {wall:.1f}s wall')
print(f'  {len(state.events)/wall:.0f} events/sec')

# Convert to mM for plotting
vol_L = state.metabolite_volume_L
met_mM_history = {
    sid: [c / 6.022e23 / vol_L * 1000.0 for c in counts_list]
    for sid, counts_list in met_history.items()
}

# Summary
print(f'\nFinal metabolite changes:')
for sid, label, _ in WATCHED_METS:
    init_c = initial_counts.get(sid, 0)
    fin_c = get_species_count(state, sid)
    d = fin_c - init_c
    dmM = count_to_mM(abs(d), vol_L) * (1 if d >= 0 else -1)
    print(f'  {label:6s}  {init_c:>10,} → {fin_c:>10,}  ({d:>+10,}, {dmM:>+7.3f} mM)')

# ============================================================================
# Step 2: Assign visualization positions
# ============================================================================
print('\nAssigning positions...')
all_ids = sorted(state.proteins.keys())
rng = np.random.default_rng(0)
positions = {}
for i in all_ids:
    while True:
        x = rng.uniform(-CELL_RADIUS, CELL_RADIUS)
        y = rng.uniform(-CELL_RADIUS, CELL_RADIUS)
        # Transporters near the membrane, others distributed
        fc = spec.proteins[state.proteins[i].gene_id].function_class
        r2 = x*x + y*y
        if fc == 'transport':
            # Biased toward outer ~85-95% radius
            if 0.70 < r2 < 0.90:
                positions[i] = (x, y)
                break
        else:
            if r2 < 0.70:
                positions[i] = (x, y)
                break
fc_by_id = {i: spec.proteins[state.proteins[i].gene_id].function_class for i in all_ids}

# ============================================================================
# Step 3: Build per-frame snapshots
# ============================================================================
print(f'Building {N_FRAMES} frame snapshots...')
frame_times = np.linspace(0, state.time, N_FRAMES + 1)[1:]
events_sorted = sorted(state.events, key=lambda e: e.time)
ev_idx = 0
FLASH = 2

current = {
    'flashing': {},
    'recent': deque(maxlen=6),
    'cat_count': 0,
    'cat_by_rxn_fwd': Counter(),
    'cat_by_rxn_rev': Counter(),
    'uptake_count': 0,
    'uptake_by_rxn': Counter(),
}

frame_snaps = []
for f_idx, t_frame in enumerate(frame_times):
    while ev_idx < len(events_sorted) and events_sorted[ev_idx].time <= t_frame:
        ev = events_sorted[ev_idx]
        if ev.rule_name.startswith('catalysis:'):
            pid = ev.participants[0]
            current['flashing'][pid] = ('active', 1)
            current['cat_count'] += 1
            parts = ev.rule_name.split(':')
            rxn = parts[1]
            is_rev = len(parts) > 2 and parts[2] == 'rev'
            if is_rev:
                current['cat_by_rxn_rev'][rxn] += 1
            else:
                current['cat_by_rxn_fwd'][rxn] += 1
            # Track uptake
            if rxn in transport_rxn_set:
                current['uptake_count'] += 1
                current['uptake_by_rxn'][rxn] += 1
        current['recent'].append((ev.time, ev.rule_name, ev.description))
        ev_idx += 1

    # Interpolate metabolite values
    met_mM_now = {
        sid: float(np.interp(t_frame, time_history, vals))
        for sid, vals in met_mM_history.items()
    }

    frame_snaps.append({
        'sim_time': t_frame,
        'flashing': dict(current['flashing']),
        'recent': list(current['recent']),
        'cat_count': current['cat_count'],
        'cat_by_rxn_fwd': dict(current['cat_by_rxn_fwd']),
        'cat_by_rxn_rev': dict(current['cat_by_rxn_rev']),
        'uptake_count': current['uptake_count'],
        'uptake_by_rxn': dict(current['uptake_by_rxn']),
        'met_mM': met_mM_now,
    })
    # Decay flashes
    new_flash = {k: (t, r-1) for k, (t, r) in current['flashing'].items() if r > 1}
    current['flashing'] = new_flash

# ============================================================================
# Step 4: Render
# ============================================================================
print(f'\nRendering {N_FRAMES} frames...')
plt.rcParams.update({
    'font.family': 'monospace', 'text.color': 'white',
    'axes.labelcolor': 'white', 'xtick.color': 'white',
    'ytick.color': 'white', 'axes.edgecolor': 'white',
    'axes.facecolor': BG_COLOR,
})

fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)
gs = fig.add_gridspec(3, 4, left=0.04, right=0.98, top=0.93, bottom=0.06,
                      wspace=0.45, hspace=0.55)

# Cell view
ax_cell = fig.add_subplot(gs[:, :2])
ax_cell.set_facecolor(BG_COLOR)
ax_cell.set_xlim(-CELL_RADIUS*1.2, CELL_RADIUS*1.2)
ax_cell.set_ylim(-CELL_RADIUS*1.2, CELL_RADIUS*1.2)
ax_cell.set_aspect('equal')
ax_cell.set_xticks([]); ax_cell.set_yticks([])
for s in ax_cell.spines.values(): s.set_color('#30363d')

# Outer "medium" ring label
ax_cell.text(0, CELL_RADIUS*1.12, 'MEDIUM (buffered)', ha='center',
              color='#64748b', fontsize=8, style='italic')

# Cell interior
ax_cell.add_patch(Circle((0,0), CELL_RADIUS, fill=True, color='#1c2230',
                          alpha=0.5, zorder=0))
# Membrane — thick
ax_cell.add_patch(Circle((0,0), CELL_RADIUS, fill=False, color='#4a9eff',
                          linewidth=4, zorder=2, alpha=0.6))
# Inner membrane (just for visual depth)
ax_cell.add_patch(Circle((0,0), CELL_RADIUS*0.96, fill=False, color='#2563eb',
                          linewidth=1, zorder=2, alpha=0.4))

scatters = {}
for fc, (marker, size, color) in FC_MARKERS.items():
    ids = [i for i in all_ids if fc_by_id[i] == fc]
    if not ids:
        continue
    xs = [positions[i][0] for i in ids]
    ys = [positions[i][1] for i in ids]
    sc = ax_cell.scatter(xs, ys, c=[color]*len(ids), marker=marker, s=size,
                          zorder=5, edgecolors='white', linewidths=0.2,
                          alpha=0.9)
    scatters[fc] = (sc, ids)

cell_title = ax_cell.text(0, CELL_RADIUS*1.30,
    'JCVI-Syn3A — reversible Michaelis-Menten + real medium uptake',
    ha='center', va='bottom', fontsize=12, weight='bold', color='white')
cell_subtitle = ax_cell.text(0, CELL_RADIUS*1.22,
    'transporters sit on membrane (cyan squares) — uptaking Pi, Mg²⁺, K⁺, amino acids',
    ha='center', va='bottom', fontsize=9, color='#8b9eb3')
cell_time = ax_cell.text(0, -CELL_RADIUS*1.17, '',
    ha='center', va='top', fontsize=10, color='#8b9eb3')

# Metabolite concentrations panel
ax_met = fig.add_subplot(gs[0, 2:])
ax_met.set_facecolor(BG_COLOR)
for s in ax_met.spines.values(): s.set_color('#30363d')
ax_met.set_title('Intracellular metabolite concentrations (mM)',
                  color='white', fontsize=11)
ax_met.set_xlabel('simulated time (ms)', color='white', fontsize=8)
ax_met.set_xlim(0, SIM_DURATION_S * 1000)
met_lines = {}
for sid, label, color in WATCHED_METS:
    ln, = ax_met.plot([], [], color=color, linewidth=1.3, label=label, alpha=0.85)
    met_lines[sid] = ln
ax_met.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
               fontsize=7, frameon=False, labelcolor='white', ncol=1)
ax_met.tick_params(labelsize=8)
ax_met.grid(True, color='#30363d', linewidth=0.3, alpha=0.5)
max_mM = max(max(v) for v in met_mM_history.values())
ax_met.set_ylim(-0.3, max_mM * 1.15)

# Forward vs reverse flux panel
ax_flux = fig.add_subplot(gs[1, 2:])
ax_flux.set_facecolor(BG_COLOR)
for s in ax_flux.spines.values(): s.set_color('#30363d')
ax_flux.set_title('Reversible reactions — fwd vs rev events '
                   '(near-balanced = equilibrium)',
                   color='white', fontsize=10)
ax_flux.tick_params(labelsize=8)
flux_text = ax_flux.text(0.02, 0.95, '', fontsize=8, color='#c9d1d9',
                          family='monospace', va='top', ha='left',
                          transform=ax_flux.transAxes)
ax_flux.set_xticks([]); ax_flux.set_yticks([])

# Transport/uptake panel
ax_up = fig.add_subplot(gs[2, 2:])
ax_up.set_facecolor(BG_COLOR)
for s in ax_up.spines.values(): s.set_color('#30363d')
ax_up.set_title('Medium uptake — cell eating nutrients from outside',
                 color='white', fontsize=10)
ax_up.set_xticks([]); ax_up.set_yticks([])
up_text = ax_up.text(0.02, 0.95, '', fontsize=7.5, color='#c9d1d9',
                      family='monospace', va='top', ha='left',
                      transform=ax_up.transAxes)


def color_for(iid, snap, base):
    if iid in snap['flashing']:
        return '#ffd43b'
    return base


writer = FFMpegWriter(fps=FPS, codec='libx264', bitrate=3800,
                       extra_args=['-pix_fmt', 'yuv420p', '-preset', 'medium'])
movie_path = OUTPUT_DIR / 'priority_15.mp4'

t_render_start = time_module.time()
met_x_running = []
met_y_running = {sid: [] for sid, _, _ in WATCHED_METS}

with writer.saving(fig, str(movie_path), dpi=110):
    for f_idx, snap in enumerate(frame_snaps):
        # Update molecules
        for fc, (sc, ids) in scatters.items():
            base = FC_MARKERS[fc][2]
            sc.set_color([color_for(i, snap, base) for i in ids])

        cell_time.set_text(
            f't = {snap["sim_time"]*1000:6.1f} ms  |  '
            f'{snap["cat_count"]:,} catalysis events  |  '
            f'{snap["uptake_count"]:,} uptake events')

        # Metabolite lines
        met_x_running.append(snap['sim_time'] * 1000)
        for sid, _, _ in WATCHED_METS:
            met_y_running[sid].append(snap['met_mM'][sid])
            met_lines[sid].set_data(met_x_running, met_y_running[sid])

        # Forward/reverse flux table
        all_rxns = set(snap['cat_by_rxn_fwd'].keys()) | set(snap['cat_by_rxn_rev'].keys())
        fluxes = []
        for rxn in all_rxns:
            f = snap['cat_by_rxn_fwd'].get(rxn, 0)
            r = snap['cat_by_rxn_rev'].get(rxn, 0)
            fluxes.append((f + r, rxn, f, r))
        fluxes.sort(reverse=True)
        fluxes = fluxes[:8]
        lines = ['  Reaction    Fwd     Rev     Net']
        for _, rxn, f, r in fluxes:
            net = f - r
            arrow = '→' if net > 0 else ('←' if net < 0 else '=')
            lines.append(f'  {rxn:10s} {f:>6,} {r:>6,} {net:>+6,} {arrow}')
        if not fluxes:
            lines = ['  (no events yet)']
        flux_text.set_text('\n'.join(lines))

        # Uptake table
        top_up = sorted(snap['uptake_by_rxn'].items(), key=lambda x: -x[1])[:10]
        up_lines = []
        for rxn, n in top_up:
            # Look up what is uptaken
            if rxn in rxns_short:
                sb = rxns_short[rxn]
                ext_in = [s.replace('M_', '').replace('_e', '') for s in sb.reactants if '_e' in s]
                ext_label = ','.join(ext_in[:2]) if ext_in else rxn
            else:
                ext_label = rxn
            up_lines.append(f'  {rxn:10s} {n:>5,} events   ({ext_label} uptake)')
        if not up_lines:
            up_lines = ['  (transporters not yet active)']
        # Recent events snippet
        up_lines.append('')
        up_lines.append('Recent:')
        for t, _, desc in snap['recent'][-3:]:
            up_lines.append(f'  {t*1000:6.1f}ms  {desc[:42]}')
        up_text.set_text('\n'.join(up_lines))

        writer.grab_frame()

        if (f_idx + 1) % 30 == 0:
            elapsed = time_module.time() - t_render_start
            rate = (f_idx + 1) / elapsed
            eta = (len(frame_snaps) - f_idx - 1) / rate
            print(f'  Frame {f_idx+1}/{len(frame_snaps)}  '
                  f'{rate:.1f} fps  ETA {eta:.0f}s')

plt.close(fig)
size_mb = movie_path.stat().st_size / 1e6
print(f'\nSaved {movie_path} ({size_mb:.2f} MB)')
