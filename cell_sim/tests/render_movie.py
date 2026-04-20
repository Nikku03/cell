"""
Generate an MP4 of the event-driven cell simulation.

We run the simulator, then build a synchronized animation with four panels:
  - Cell view: molecules as dots in 2D, colored by state, with event flashes
  - Population bars: live counts per (gene, state)
  - Event log ticker: the last N events scrolling by
  - Assembly progress: dimer and tetramer counts over time

To give the simulation a nice visual story, we assign each molecule a
random fixed 2D position within the cell circle, then show its state
changing over time. This isn't real spatial motion — it's an identity-
preserving projection, which matches how the simulator actually works
(no positions at atomic level, just compartment + identity).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.animation import FFMpegWriter
from collections import defaultdict, deque, Counter
import time as time_module

from layer0_genome.parser import build_cell_spec, Protein
from layer2_field.dynamics import CellState, EventSimulator, make_example_rules

# ============================================================================
# Config
# ============================================================================
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'event_demo'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIM_DURATION_S = 2.0       # simulated cell time
FPS = 24                    # frames per second in the MP4
MOVIE_DURATION_S = 15.0     # wall-time length of the MP4
N_FRAMES = int(FPS * MOVIE_DURATION_S)

# Visual config
CELL_RADIUS = 1.0           # arbitrary units for the 2D cell circle
BG_COLOR = '#0d1117'        # dark background
CELL_MEMBRANE_COLOR = '#30363d'

# Colors for molecule states
STATE_COLORS = {
    'unfolded': '#ff6b6b',      # red — unfolded
    'native':   '#4dabf7',      # blue — native, monomer
    'bound':    '#51cf66',      # green — in a complex
    'phospho':  '#ffd43b',      # yellow — phosphorylated
}

# Colors per gene (for differentiation)
GENE_SHAPES = {
    'kinase_A':    'o',
    'substrate_B': 's',
    'monomer_C':   '^',
}
GENE_SIZE = {
    'kinase_A':    45,
    'substrate_B': 35,
    'monomer_C':   25,
}


# ============================================================================
# Step 1: Run the simulation and capture everything
# ============================================================================
print("Step 1: Running simulation...")
spec = build_cell_spec(species='syn3a')
spec.proteins['kinase_A']    = Protein(gene_id='kinase_A',    sequence='M'*200, length=200, function_class='enzyme')
spec.proteins['substrate_B'] = Protein(gene_id='substrate_B', sequence='M'*150, length=150, function_class='enzyme')
spec.proteins['monomer_C']   = Protein(gene_id='monomer_C',   sequence='M'*100, length=100, function_class='structural')

state = CellState(spec)
for _ in range(10):  state.new_protein('kinase_A', conformation='unfolded')
for _ in range(50):  state.new_protein('substrate_B', conformation='unfolded')
for _ in range(100): state.new_protein('monomer_C', conformation='unfolded')

all_instance_ids = sorted(state.proteins.keys())
print(f"  {len(all_instance_ids)} molecules created")

sim = EventSimulator(state, make_example_rules(), mode='gillespie', seed=42)
stats = sim.run_until(t_end=SIM_DURATION_S)
print(f"  {stats['n_events']} events in {stats['wall_time_s']:.3f}s wall "
      f"({stats['events_per_wall_sec']:.0f} events/sec)")
print(f"  {len(state.complexes)} complexes formed")

# ============================================================================
# Step 2: Assign fixed 2D positions to each molecule
# ============================================================================
print("Step 2: Assigning positions...")
rng = np.random.default_rng(0)
positions = {}
for inst_id in all_instance_ids:
    # Sample uniformly inside the cell circle
    while True:
        x = rng.uniform(-CELL_RADIUS, CELL_RADIUS)
        y = rng.uniform(-CELL_RADIUS, CELL_RADIUS)
        if x*x + y*y < (CELL_RADIUS * 0.95)**2:
            positions[inst_id] = (x, y)
            break

# ============================================================================
# Step 3: Build per-frame state snapshots by replaying events
# ============================================================================
print(f"Step 3: Building {N_FRAMES} frame snapshots...")

# Initial state (all unfolded, no partners)
def copy_state_minimal():
    return {
        'conformations': {i: 'unfolded' for i in all_instance_ids},
        'phospho':       {i: False for i in all_instance_ids},
        'bound':         {i: False for i in all_instance_ids},
        'n_dimers':      0,
        'n_tetramers':   0,
        'recent_events': deque(maxlen=8),
        'flashing':      {},  # inst_id -> (event_type, frames_remaining)
    }

# Map simulated time onto frame index
sim_times = np.linspace(0, state.time, N_FRAMES + 1)[1:]  # end of each frame
events_sorted = sorted(state.events, key=lambda e: e.time)

# We'll walk through events and checkpoint at each frame time
frame_states = []
ev_idx = 0
current = copy_state_minimal()

# Flash duration in frames
FLASH_DURATION = 3

for frame_idx, t_frame in enumerate(sim_times):
    # Advance events that fire before this frame time
    while ev_idx < len(events_sorted) and events_sorted[ev_idx].time <= t_frame:
        ev = events_sorted[ev_idx]
        # Apply visible state changes
        if ev.rule_name == 'folding':
            pid = ev.participants[0]
            current['conformations'][pid] = 'native'
            current['flashing'][pid] = ('fold', FLASH_DURATION)
        elif ev.rule_name == 'phosphorylation':
            k, s = ev.participants
            current['phospho'][s] = True
            current['flashing'][k] = ('phospho', FLASH_DURATION)
            current['flashing'][s] = ('phospho', FLASH_DURATION)
        elif ev.rule_name == 'dimerization':
            a, b = ev.participants
            current['bound'][a] = True
            current['bound'][b] = True
            current['n_dimers'] += 1
            current['flashing'][a] = ('bind', FLASH_DURATION)
            current['flashing'][b] = ('bind', FLASH_DURATION)
        elif ev.rule_name == 'tetramerization':
            current['n_tetramers'] += 1
            current['n_dimers'] = max(0, current['n_dimers'] - 2)
            for pid in ev.participants:
                current['flashing'][pid] = ('tetra', FLASH_DURATION * 3)
        current['recent_events'].append(
            (ev.time, ev.rule_name, ev.description)
        )
        ev_idx += 1

    # Snapshot
    snapshot = {
        'sim_time': t_frame,
        'conformations': dict(current['conformations']),
        'phospho':       dict(current['phospho']),
        'bound':         dict(current['bound']),
        'n_dimers':      current['n_dimers'],
        'n_tetramers':   current['n_tetramers'],
        'recent_events': list(current['recent_events']),
        'flashing':      {k: v for k, v in current['flashing'].items()},
    }
    frame_states.append(snapshot)

    # Decay flashes for next frame
    new_flashing = {}
    for pid, (etype, remaining) in current['flashing'].items():
        if remaining > 1:
            new_flashing[pid] = (etype, remaining - 1)
    current['flashing'] = new_flashing

print(f"  Built {len(frame_states)} frame snapshots")
print(f"  Final: {frame_states[-1]['n_dimers']} dimers, {frame_states[-1]['n_tetramers']} tetramers")

# ============================================================================
# Step 4: Build the animation
# ============================================================================
print(f"Step 4: Rendering animation at {FPS} fps, {MOVIE_DURATION_S}s total...")

plt.rcParams['font.family'] = 'monospace'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.facecolor'] = BG_COLOR

fig = plt.figure(figsize=(14, 8), facecolor=BG_COLOR)
gs = fig.add_gridspec(3, 3, left=0.05, right=0.98, top=0.93, bottom=0.07,
                      wspace=0.3, hspace=0.4)

# Cell view (big left panel)
ax_cell = fig.add_subplot(gs[:, :2])
ax_cell.set_facecolor(BG_COLOR)
ax_cell.set_xlim(-CELL_RADIUS*1.15, CELL_RADIUS*1.15)
ax_cell.set_ylim(-CELL_RADIUS*1.15, CELL_RADIUS*1.15)
ax_cell.set_aspect('equal')
ax_cell.set_xticks([]); ax_cell.set_yticks([])
for spine in ax_cell.spines.values(): spine.set_color(CELL_MEMBRANE_COLOR)

# Draw cell membrane
membrane = Circle((0, 0), CELL_RADIUS, fill=False,
                   color=CELL_MEMBRANE_COLOR, linewidth=3, zorder=1)
ax_cell.add_patch(membrane)
# Fill the cell with a very slight tint
cell_fill = Circle((0, 0), CELL_RADIUS, fill=True,
                    color='#1c2230', alpha=0.5, zorder=0)
ax_cell.add_patch(cell_fill)

# Prepare scatter plots per gene shape (for efficient updates)
scatters = {}
for gene, marker in GENE_SHAPES.items():
    inst_ids = [i for i in all_instance_ids if state.proteins[i].gene_id == gene]
    xs = [positions[i][0] for i in inst_ids]
    ys = [positions[i][1] for i in inst_ids]
    sc = ax_cell.scatter(xs, ys, c=[STATE_COLORS['unfolded']]*len(inst_ids),
                          marker=marker, s=GENE_SIZE[gene], zorder=5,
                          edgecolors='white', linewidths=0.3)
    scatters[gene] = (sc, inst_ids)

# Title overlay on cell view
cell_title = ax_cell.text(0, CELL_RADIUS*1.14, 'Syn3A cell simulation',
                           ha='center', va='bottom', fontsize=15, color='white',
                           weight='bold')
cell_time_label = ax_cell.text(0, CELL_RADIUS*1.06,
                                'sim t = 0.000 s',
                                ha='center', va='bottom', fontsize=10,
                                color='#8b9eb3')

# Legend (inline text blocks on the cell panel)
legend_y = -CELL_RADIUS*1.08
for i, (label, color) in enumerate([
    ('unfolded', STATE_COLORS['unfolded']),
    ('native',   STATE_COLORS['native']),
    ('bound',    STATE_COLORS['bound']),
    ('phospho+', STATE_COLORS['phospho']),
]):
    x_offset = -CELL_RADIUS*1.13 + i * 0.55
    ax_cell.scatter([x_offset + 0.05], [legend_y], c=[color], s=60,
                    marker='o', edgecolors='white', linewidths=0.3)
    ax_cell.text(x_offset + 0.12, legend_y, label, fontsize=10, color='white',
                 va='center')

# Population bar chart (top right)
ax_bars = fig.add_subplot(gs[0, 2])
ax_bars.set_facecolor(BG_COLOR)
for spine in ax_bars.spines.values(): spine.set_color('#30363d')
ax_bars.set_title('Molecule states', color='white', fontsize=11)
bar_categories = ['unfolded', 'native\n(monomer)', 'bound\n(complex)', 'phospho+']
bar_colors_list = [STATE_COLORS['unfolded'], STATE_COLORS['native'],
                   STATE_COLORS['bound'], STATE_COLORS['phospho']]
bar_patches = ax_bars.bar(bar_categories, [0]*4, color=bar_colors_list,
                           edgecolor='white', linewidth=0.5)
ax_bars.set_ylim(0, 170)
ax_bars.tick_params(labelsize=9)
ax_bars.set_ylabel('count', color='white', fontsize=9)

# Assembly progress (middle right)
ax_asm = fig.add_subplot(gs[1, 2])
ax_asm.set_facecolor(BG_COLOR)
for spine in ax_asm.spines.values(): spine.set_color('#30363d')
ax_asm.set_title('Quaternary assembly', color='white', fontsize=11)
ax_asm.set_xlim(0, SIM_DURATION_S)
ax_asm.set_xlabel('simulated time (s)', color='white', fontsize=9)
ax_asm.set_ylabel('count', color='white', fontsize=9)
ax_asm.tick_params(labelsize=8)
dimer_line, = ax_asm.plot([], [], color='#51cf66', linewidth=2, label='dimers')
tet_line,   = ax_asm.plot([], [], color='#ff9f43', linewidth=2, label='tetramers')
ax_asm.legend(loc='upper left', fontsize=8, facecolor=BG_COLOR,
              edgecolor='#30363d', labelcolor='white')

# Event ticker (bottom right)
ax_log = fig.add_subplot(gs[2, 2])
ax_log.set_facecolor(BG_COLOR)
ax_log.set_xticks([]); ax_log.set_yticks([])
ax_log.set_title('Recent events', color='white', fontsize=11)
for spine in ax_log.spines.values(): spine.set_color('#30363d')
log_text = ax_log.text(0.02, 0.95, '', fontsize=7.5, color='#c9d1d9',
                        family='monospace', va='top', ha='left',
                        transform=ax_log.transAxes)

# Pre-compute assembly history
dimer_history = []
tet_history = []
time_history = []


def molecule_color(inst_id, snapshot):
    """Determine the color for a molecule given its state."""
    if inst_id in snapshot['flashing']:
        etype, _ = snapshot['flashing'][inst_id]
        if etype == 'fold':
            return '#ffffff'
        if etype == 'phospho':
            return '#fff59d'
        if etype == 'bind':
            return '#ffffff'
        if etype == 'tetra':
            return '#ffb347'
    if snapshot['bound'][inst_id]:
        return STATE_COLORS['bound']
    if snapshot['phospho'][inst_id]:
        return STATE_COLORS['phospho']
    if snapshot['conformations'][inst_id] == 'native':
        return STATE_COLORS['native']
    return STATE_COLORS['unfolded']


def molecule_size(inst_id, snapshot, base_size):
    """Size with a flash multiplier."""
    if inst_id in snapshot['flashing']:
        _, remaining = snapshot['flashing'][inst_id]
        return base_size * (1 + 0.5 * remaining / FLASH_DURATION)
    return base_size


# ============================================================================
# Animation writer
# ============================================================================
writer = FFMpegWriter(
    fps=FPS,
    codec='libx264',
    bitrate=3000,
    extra_args=['-pix_fmt', 'yuv420p', '-preset', 'medium'],
)

movie_path = OUTPUT_DIR / 'cell_simulation.mp4'

t_render_start = time_module.time()
with writer.saving(fig, str(movie_path), dpi=120):
    for frame_idx, snap in enumerate(frame_states):
        # --- Update cell view ---
        for gene, (sc, inst_ids) in scatters.items():
            colors = [molecule_color(i, snap) for i in inst_ids]
            sizes = [molecule_size(i, snap, GENE_SIZE[gene]) for i in inst_ids]
            sc.set_color(colors)
            sc.set_sizes(sizes)

        cell_time_label.set_text(f'sim t = {snap["sim_time"]:.3f} s  '
                                  f'({snap["sim_time"]*1e6:.0f} μs)')

        # --- Update bar chart ---
        n_unfolded = sum(1 for i in all_instance_ids
                         if snap['conformations'][i] == 'unfolded'
                         and not snap['bound'][i])
        n_native   = sum(1 for i in all_instance_ids
                         if snap['conformations'][i] == 'native'
                         and not snap['bound'][i])
        n_bound    = sum(1 for i in all_instance_ids if snap['bound'][i])
        n_phospho  = sum(1 for i in all_instance_ids if snap['phospho'][i])
        counts = [n_unfolded, n_native, n_bound, n_phospho]
        for bar, c in zip(bar_patches, counts):
            bar.set_height(c)

        # --- Update assembly plot ---
        time_history.append(snap['sim_time'])
        dimer_history.append(snap['n_dimers'])
        tet_history.append(snap['n_tetramers'])
        dimer_line.set_data(time_history, dimer_history)
        tet_line.set_data(time_history, tet_history)
        max_y = max(max(dimer_history, default=1), max(tet_history, default=1), 5)
        ax_asm.set_ylim(0, max_y * 1.2)

        # --- Update event log ---
        if snap['recent_events']:
            lines = []
            for t, rule, desc in list(snap['recent_events'])[-6:]:
                lines.append(f"{t*1e6:8.1f} μs  [{rule[:12]:12s}]")
                lines.append(f"           {desc[:42]}")
            log_text.set_text('\n'.join(lines))

        writer.grab_frame()

        if (frame_idx + 1) % 30 == 0:
            elapsed = time_module.time() - t_render_start
            rate = (frame_idx + 1) / elapsed
            eta = (len(frame_states) - frame_idx - 1) / rate
            print(f"  Frame {frame_idx+1}/{len(frame_states)}  "
                  f"{rate:.1f} fps  ETA {eta:.0f}s")

plt.close(fig)
render_time = time_module.time() - t_render_start
file_size_mb = movie_path.stat().st_size / 1e6
print(f"\nRendered {len(frame_states)} frames in {render_time:.1f}s")
print(f"Saved: {movie_path}  ({file_size_mb:.2f} MB)")
