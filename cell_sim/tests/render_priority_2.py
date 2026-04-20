"""
Priority 2 MP4: gene expression layer.

Adds transcription, translation, and mRNA degradation events to the
Priority 1.5 simulation. We simulate 3 seconds of cell life so we can see:
  - mRNA counts changing as genes are transcribed and degraded
  - New proteins appearing (translation events)
  - Proteins turning over via degradation
  - Coupling between GEX and metabolism (NTPs consumed, GTP for translation)

Focus on 40 highly-expressed genes to keep the simulation tractable while
capturing the most visible dynamics.
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

from layer2_field.dynamics import CellState, EventSimulator
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
from layer3_reactions.gene_expression import (
    initialize_gene_expression_state,
    make_transcription_rule, make_translation_rule,
    make_mrna_degradation_rule,
)

# Config
OUTPUT_DIR = Path('/home/claude/cell_sim/data/priority_2_movie')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE_FACTOR = 0.02
MAX_PER_GENE = 10
SIM_DURATION_S = 1.5       # 1.5 seconds — balances tractability with visible GEX
FPS = 24
MOVIE_DURATION_S = 18.0
N_FRAMES = int(FPS * MOVIE_DURATION_S)
N_SNAPSHOTS = N_FRAMES

N_TOP_GENES = 30  # GEX focused on top-expressed genes

CELL_RADIUS = 1.0
BG_COLOR = '#0d1117'

# Metabolites to track (adding GTP, amino acids to show GEX coupling)
WATCHED_METS = [
    ('M_atp_c',    'ATP',    '#60a5fa'),
    ('M_adp_c',    'ADP',    '#3b82f6'),
    ('M_gtp_c',    'GTP',    '#a855f7'),
    ('M_gdp_c',    'GDP',    '#7c3aed'),
    ('M_utp_c',    'UTP',    '#22d3ee'),
    ('M_ctp_c',    'CTP',    '#06b6d4'),
    ('M_ala__L_c', 'Ala',    '#10b981'),
    ('M_gly_c',    'Gly',    '#84cc16'),
    ('M_pyr_c',    'Pyr',    '#f97316'),
    ('M_pi_c',     'Pi',     '#fbbf24'),
]

FC_MARKERS = {
    'enzyme':         ('o', 12, '#4dabf7'),
    'ribosomal':      ('D', 16, '#d946ef'),  # highlight ribosomal for GEX context
    'transport':      ('s', 12, '#22d3ee'),
    'rna_processing': ('v', 11, '#a855f7'),
    'regulatory':     ('P', 10, '#f97316'),
    'structural_division': ('*', 18, '#f59e0b'),
    'unknown':        ('o', 6, '#64748b'),
    'other':          ('o', 8, '#64748b'),
}


# ============================================================================
# Step 1: Run simulation
# ============================================================================
print('=' * 60)
print('Priority 2 — transcription, translation, degradation')
print('=' * 60)

print('\nLoading...')
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
gex_stats = initialize_gene_expression_state(state, scale_factor=0.1)

print(f'  Proteins: {len(state.proteins):,}')
print(f'  Metabolites: {len(state.metabolite_counts)}')
print(f'  mRNAs: {gex_stats["mrnas"]} | RNAPs: {gex_stats["rnap_free"]} | '
      f'Ribosomes: {gex_stats["ribosome_free"]}')

# Select top-expressed genes for GEX rules
top_genes = sorted(state.spec.proteins,
                    key=lambda g: state.mrna_counts.get(g, 0),
                    reverse=True)[:N_TOP_GENES]

gex_rules = []
for g in top_genes:
    p = spec.proteins[g]
    length_aa = max(10, p.length or 100)
    length_nt = length_aa * 3 + 100
    gex_rules.append(make_transcription_rule(g, length_nt))
    gex_rules.append(make_translation_rule(g, length_aa))
    gex_rules.append(make_mrna_degradation_rule(g, length_nt))
    # Skip protein degradation — too slow to matter at these timescales
print(f'  Gene expression rules: {len(gex_rules)} ({N_TOP_GENES} genes × 3)')

rules = (
    [make_folding_rule(20.0)]
    + rev_rules
    + make_complex_formation_rules(complexes, base_rate_per_s=0.05)
    + gex_rules
)
print(f'  Total rules: {len(rules)}')

# Snapshot initial state
initial_counts = dict(state.metabolite_counts)
initial_proteins = {g: len(state.proteins_by_gene.get(g, set())) for g in top_genes}
initial_mrnas = dict(state.mrna_counts)

# Run with snapshots
print(f'\nSimulating {SIM_DURATION_S}s with {N_SNAPSHOTS} snapshots...')
snapshot_times = np.linspace(0, SIM_DURATION_S, N_SNAPSHOTS + 1)[1:]
met_history = {sid: [] for sid, _, _ in WATCHED_METS}
time_history = []
# Track protein counts for top genes
protein_history = {g: [] for g in top_genes[:10]}
mrna_history = {g: [] for g in top_genes[:10]}

sim = EventSimulator(state, rules, mode='gillespie', seed=42)
t_start = time_module.time()
for i, t_snap in enumerate(snapshot_times):
    sim.run_until(t_end=t_snap, max_events=2_000_000)
    time_history.append(state.time)
    for sid, _, _ in WATCHED_METS:
        met_history[sid].append(get_species_count(state, sid))
    for g in top_genes[:10]:
        protein_history[g].append(len(state.proteins_by_gene.get(g, set())))
        mrna_history[g].append(state.mrna_counts.get(g, 0))
    if (i + 1) % 30 == 0:
        print(f'  Snap {i+1}/{N_SNAPSHOTS}  t={state.time*1000:.0f}ms  '
              f'evt={len(state.events):,}  wall={time_module.time()-t_start:.0f}s')

wall = time_module.time() - t_start
print(f'\n  {len(state.events):,} events in {wall:.1f}s wall')

# Event breakdown
by_type = Counter()
for e in state.events:
    parts = e.rule_name.split(':')
    by_type[parts[0]] += 1
print(f'  Event breakdown:')
for k, v in by_type.most_common():
    print(f'    {k}: {v:,}')

# Convert metabolite counts to mM
vol_L = state.metabolite_volume_L
met_mM_history = {
    sid: [c / 6.022e23 / vol_L * 1000.0 for c in cl]
    for sid, cl in met_history.items()
}

# Report gene expression outcomes
tx_events = [e for e in state.events if e.rule_name.startswith('transcribe:')]
tl_events = [e for e in state.events if e.rule_name.startswith('translate:')]
dg_events = [e for e in state.events if e.rule_name.startswith('degrade_mrna:')]
print(f'\n  Transcription events: {len(tx_events)}')
print(f'  Translation events:   {len(tl_events)}')
print(f'  mRNA degradation:     {len(dg_events)}')

print(f'\n  Top 10 genes — protein count changes:')
print(f'  {"Locus":16s}  {"Gene":10s}  {"Init":>5s} {"Final":>5s} {"Δ":>5s}  mRNA(i→f)')
for g in top_genes[:10]:
    init = initial_proteins[g]
    final = len(state.proteins_by_gene.get(g, set()))
    mi = initial_mrnas.get(g, 0)
    mf = state.mrna_counts.get(g, 0)
    gn = spec.proteins[g].annotations.get('gene_name', '')[:10]
    print(f'  {g}  {gn:10s}  {init:>5d} {final:>5d} {final-init:>+5d}  {mi}→{mf}')

# ============================================================================
# Step 2: Positions
# ============================================================================
print('\nAssigning positions...')
all_ids = sorted(state.proteins.keys())
rng = np.random.default_rng(0)
positions = {}
for i in all_ids:
    while True:
        x = rng.uniform(-CELL_RADIUS, CELL_RADIUS)
        y = rng.uniform(-CELL_RADIUS, CELL_RADIUS)
        fc = spec.proteins[state.proteins[i].gene_id].function_class
        r2 = x*x + y*y
        if fc == 'transport':
            if 0.70 < r2 < 0.90:
                positions[i] = (x, y); break
        elif fc == 'ribosomal':
            if 0.05 < r2 < 0.55:  # Ribosomal in inner ring
                positions[i] = (x, y); break
        else:
            if r2 < 0.70:
                positions[i] = (x, y); break
fc_by_id = {i: spec.proteins[state.proteins[i].gene_id].function_class for i in all_ids}

# ============================================================================
# Step 3: Build per-frame snapshots
# ============================================================================
print(f'Building {N_FRAMES} frame snapshots...')
frame_times = np.linspace(0, state.time, N_FRAMES + 1)[1:]
events_sorted = sorted(state.events, key=lambda e: e.time)
ev_idx = 0

current = {
    'flashing': {},        # pid -> (type, frames_left)
    'recent': deque(maxlen=8),
    'cat_count': 0,
    'tx_count': 0,
    'tl_count': 0,
    'dg_count': 0,
    'tx_by_gene': Counter(),
    'tl_by_gene': Counter(),
}

# Track new proteins created during sim for flashing
# This is tricky because new proteins get random positions too
new_protein_positions = {}

frame_snaps = []
for f_idx, t_frame in enumerate(frame_times):
    while ev_idx < len(events_sorted) and events_sorted[ev_idx].time <= t_frame:
        ev = events_sorted[ev_idx]
        rule_prefix = ev.rule_name.split(':')[0]
        if rule_prefix == 'catalysis':
            if ev.participants:
                current['flashing'][ev.participants[0]] = ('active', 1)
            current['cat_count'] += 1
        elif rule_prefix == 'transcribe':
            current['tx_count'] += 1
            gene = ev.rule_name.split(':')[1]
            current['tx_by_gene'][gene] += 1
        elif rule_prefix == 'translate':
            current['tl_count'] += 1
            gene = ev.rule_name.split(':')[1]
            current['tl_by_gene'][gene] += 1
            # Assign position for the newly created protein
            # Latest protein with this gene is the one just made
            for pid in list(state.proteins_by_gene.get(gene, [])):
                if pid not in positions and pid not in new_protein_positions:
                    while True:
                        x = rng.uniform(-CELL_RADIUS, CELL_RADIUS)
                        y = rng.uniform(-CELL_RADIUS, CELL_RADIUS)
                        if x*x + y*y < 0.7:
                            new_protein_positions[pid] = (x, y)
                            positions[pid] = (x, y)
                            fc_by_id[pid] = spec.proteins[gene].function_class
                            break
                    # Flash it brightly
                    current['flashing'][pid] = ('newborn', 10)
        elif rule_prefix == 'degrade_mrna':
            current['dg_count'] += 1
        current['recent'].append((ev.time, ev.rule_name, ev.description))
        ev_idx += 1

    # Interpolate metabolite values
    met_mM_now = {
        sid: float(np.interp(t_frame, time_history, v))
        for sid, v in met_mM_history.items()
    }
    # Interpolate protein/mRNA counts for key genes
    prot_now = {g: float(np.interp(t_frame, time_history, v)) for g, v in protein_history.items()}
    mrna_now = {g: float(np.interp(t_frame, time_history, v)) for g, v in mrna_history.items()}

    frame_snaps.append({
        'sim_time': t_frame,
        'flashing': dict(current['flashing']),
        'recent': list(current['recent']),
        'cat_count': current['cat_count'],
        'tx_count': current['tx_count'],
        'tl_count': current['tl_count'],
        'dg_count': current['dg_count'],
        'tx_by_gene': dict(current['tx_by_gene']),
        'tl_by_gene': dict(current['tl_by_gene']),
        'met_mM': met_mM_now,
        'prot_counts': prot_now,
        'mrna_counts': mrna_now,
    })
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
gs = fig.add_gridspec(3, 4, left=0.04, right=0.98, top=0.92, bottom=0.06,
                      wspace=0.45, hspace=0.55)

# Cell view
ax_cell = fig.add_subplot(gs[:, :2])
ax_cell.set_facecolor(BG_COLOR)
ax_cell.set_xlim(-CELL_RADIUS*1.2, CELL_RADIUS*1.2)
ax_cell.set_ylim(-CELL_RADIUS*1.2, CELL_RADIUS*1.2)
ax_cell.set_aspect('equal')
ax_cell.set_xticks([]); ax_cell.set_yticks([])
for s in ax_cell.spines.values(): s.set_color('#30363d')

ax_cell.text(0, CELL_RADIUS*1.12, 'MEDIUM', ha='center',
              color='#64748b', fontsize=8, style='italic')
ax_cell.add_patch(Circle((0,0), CELL_RADIUS, fill=True, color='#1c2230',
                          alpha=0.5, zorder=0))
ax_cell.add_patch(Circle((0,0), CELL_RADIUS, fill=False, color='#4a9eff',
                          linewidth=4, zorder=2, alpha=0.6))
# Add a "nucleoid-like" region label
ax_cell.text(0, 0, '⊙ nucleoid\n(transcription)', ha='center', va='center',
              color='#444', fontsize=8, zorder=1, alpha=0.5)

scatters = {}
for fc, (marker, size, color) in FC_MARKERS.items():
    ids = [i for i in all_ids if fc_by_id.get(i) == fc]
    if not ids:
        continue
    xs = [positions[i][0] for i in ids]
    ys = [positions[i][1] for i in ids]
    sc = ax_cell.scatter(xs, ys, c=[color]*len(ids), marker=marker, s=size,
                          zorder=5, edgecolors='white', linewidths=0.2, alpha=0.9)
    scatters[fc] = (sc, ids)

# Container for newly-translated proteins (separate scatter, so we can flash big)
newborn_scatter = ax_cell.scatter([], [], c='#ffffff', marker='o', s=100,
                                    zorder=10, edgecolors='#fef08a', linewidths=2)

cell_title = ax_cell.text(0, CELL_RADIUS*1.30,
    'JCVI-Syn3A — Priority 2: central dogma (DNA→RNA→protein)',
    ha='center', va='bottom', fontsize=12, weight='bold', color='white')
cell_subtitle = ax_cell.text(0, CELL_RADIUS*1.22,
    'metabolism + reversibility + transport + transcription + translation + degradation',
    ha='center', va='bottom', fontsize=9, color='#8b9eb3')
cell_time = ax_cell.text(0, -CELL_RADIUS*1.17, '',
    ha='center', va='top', fontsize=10, color='#8b9eb3')

# Metabolite concentrations (NTPs, AAs this time)
ax_met = fig.add_subplot(gs[0, 2:])
ax_met.set_facecolor(BG_COLOR)
for s in ax_met.spines.values(): s.set_color('#30363d')
ax_met.set_title('NTPs and amino acids (mM)', color='white', fontsize=11)
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

# Gene expression activity (middle right)
ax_gex = fig.add_subplot(gs[1, 2:])
ax_gex.set_facecolor(BG_COLOR)
for s in ax_gex.spines.values(): s.set_color('#30363d')
ax_gex.set_title('Gene expression events — top active genes',
                  color='white', fontsize=10)
ax_gex.set_xticks([]); ax_gex.set_yticks([])
gex_text = ax_gex.text(0.02, 0.95, '', fontsize=8, color='#c9d1d9',
                        family='monospace', va='top', ha='left',
                        transform=ax_gex.transAxes)

# Recent events
ax_log = fig.add_subplot(gs[2, 2:])
ax_log.set_facecolor(BG_COLOR)
for s in ax_log.spines.values(): s.set_color('#30363d')
ax_log.set_title('Live events — transcription, translation, catalysis',
                  color='white', fontsize=10)
ax_log.set_xticks([]); ax_log.set_yticks([])
log_text = ax_log.text(0.02, 0.95, '', fontsize=7.5, color='#c9d1d9',
                        family='monospace', va='top', ha='left',
                        transform=ax_log.transAxes)


def color_for(iid, snap, base):
    if iid in snap['flashing']:
        etype, _ = snap['flashing'][iid]
        if etype == 'newborn':
            return '#ffffff'  # bright white for newly translated
        return '#ffd43b'  # yellow for catalyzing
    return base


writer = FFMpegWriter(fps=FPS, codec='libx264', bitrate=3800,
                       extra_args=['-pix_fmt', 'yuv420p', '-preset', 'medium'])
movie_path = OUTPUT_DIR / 'priority_2.mp4'

t_render_start = time_module.time()
met_x_running = []
met_y_running = {sid: [] for sid, _, _ in WATCHED_METS}

with writer.saving(fig, str(movie_path), dpi=110):
    for f_idx, snap in enumerate(frame_snaps):
        # Update molecules (checking against all_ids + any new ones)
        all_current_ids = list(state.proteins.keys())
        # Update scatter plots — we keep the initial ids and check flash
        for fc, (sc, ids) in scatters.items():
            base = FC_MARKERS[fc][2]
            colors = [color_for(i, snap, base) for i in ids]
            sc.set_color(colors)

        # Newborn highlights
        newborn_ids = [i for i, (t, _) in snap['flashing'].items() if t == 'newborn']
        if newborn_ids:
            xs = [positions[i][0] for i in newborn_ids if i in positions]
            ys = [positions[i][1] for i in newborn_ids if i in positions]
            newborn_scatter.set_offsets(np.c_[xs, ys] if xs else np.empty((0, 2)))
        else:
            newborn_scatter.set_offsets(np.empty((0, 2)))

        cell_time.set_text(
            f't = {snap["sim_time"]*1000:6.0f}ms  |  '
            f'cat={snap["cat_count"]:,}  '
            f'tx={snap["tx_count"]}  tl={snap["tl_count"]}  '
            f'deg_mrna={snap["dg_count"]}')

        # Metabolite lines
        met_x_running.append(snap['sim_time'] * 1000)
        for sid, _, _ in WATCHED_METS:
            met_y_running[sid].append(snap['met_mM'][sid])
            met_lines[sid].set_data(met_x_running, met_y_running[sid])

        # GEX activity per gene
        lines = ['  Gene              Tx   Tl  protein(now)  mRNA(now)']
        all_gex_genes = set(snap['tx_by_gene']) | set(snap['tl_by_gene'])
        # Sort by total activity
        sorted_gex = sorted(all_gex_genes,
                              key=lambda g: -(snap['tx_by_gene'].get(g, 0) +
                                               snap['tl_by_gene'].get(g, 0)))
        for g in sorted_gex[:8]:
            gn = spec.proteins[g].annotations.get('gene_name', '')[:8]
            tx = snap['tx_by_gene'].get(g, 0)
            tl = snap['tl_by_gene'].get(g, 0)
            p_now = snap['prot_counts'].get(g, 0)
            m_now = snap['mrna_counts'].get(g, 0)
            lines.append(f'  {gn:8s}/{g[-4:]}   {tx:>3} {tl:>3}   {p_now:>5.1f}        {m_now:>4.1f}')
        if len(sorted_gex) == 0:
            lines = ['  (no gene expression events yet)']
        gex_text.set_text('\n'.join(lines))

        # Recent events — filter to prefer gex events when they're rare
        recent_ev = snap['recent']
        gex_events = [ev for ev in recent_ev if ev[1].startswith(('transcribe', 'translate', 'degrade_mrna'))]
        cat_events = [ev for ev in recent_ev if ev[1].startswith('catalysis')][-2:]
        display_events = (gex_events[-3:] + cat_events)[-5:]
        elines = []
        for t, rule, desc in display_events:
            elines.append(f'{t*1000:6.1f}ms:')
            elines.append(f'  {desc[:60]}')
        log_text.set_text('\n'.join(elines))

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
