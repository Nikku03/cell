"""
Demo: run the event-driven simulator and generate all the visualizations
that would appear in the Colab notebook. This produces real PNG outputs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

from layer0_genome.parser import build_cell_spec, Protein
from layer2_field.dynamics import CellState, EventSimulator, make_example_rules

OUT = Path(__file__).resolve().parent.parent / 'data' / 'event_demo'
OUT.mkdir(parents=True, exist_ok=True)

# ===== Build cell =====
print("Building cell...")
spec = build_cell_spec(species='syn3a')
spec.proteins['kinase_A']    = Protein(gene_id='kinase_A', sequence='M'*200, length=200, function_class='enzyme')
spec.proteins['substrate_B'] = Protein(gene_id='substrate_B', sequence='M'*150, length=150, function_class='enzyme')
spec.proteins['monomer_C']   = Protein(gene_id='monomer_C', sequence='M'*100, length=100, function_class='structural')

state = CellState(spec)
for _ in range(10):  state.new_protein('kinase_A', conformation='unfolded')
for _ in range(50):  state.new_protein('substrate_B', conformation='unfolded')
for _ in range(100): state.new_protein('monomer_C', conformation='unfolded')
print(f"  {len(state.proteins)} molecules created")

# ===== Run =====
print("Running 2 sec simulated time...")
sim = EventSimulator(state, make_example_rules(), mode='gillespie', seed=42)
stats = sim.run_until(t_end=2.0)
print(f"  Wall: {stats['wall_time_s']:.3f}s, events: {stats['n_events']}, rate: {stats['events_per_wall_sec']:.0f}/s")

# ===== Viz 1: Event timeline =====
print("Generating event timeline plot...")
event_types = sorted(set(e.rule_name for e in state.events))
colors = plt.cm.tab10(np.linspace(0, 1, max(len(event_types), 3)))
color_map = dict(zip(event_types, colors))

fig, ax = plt.subplots(figsize=(14, 5))
for etype in event_types:
    times = [e.time for e in state.events if e.rule_name == etype]
    y = [event_types.index(etype)] * len(times)
    ax.scatter(times, y, c=[color_map[etype]], s=15, alpha=0.6, label=f'{etype} ({len(times)})')
ax.set_yticks(range(len(event_types)))
ax.set_yticklabels(event_types)
ax.set_xlabel('Simulated time (s)')
ax.set_title(f'Event timeline: {len(state.events)} events over {state.time:.2f} s of cell life')
ax.grid(alpha=0.3)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(OUT / '01_event_timeline.png', dpi=120)
plt.close()

# ===== Viz 2: Folding dynamics per gene =====
print("Generating folding dynamics plot...")
sampling_dt = 0.01
times = np.arange(0, state.time + sampling_dt, sampling_dt)
trajectories = defaultdict(lambda: np.zeros(len(times)))
current = {'kinase_A:unfolded': 10, 'substrate_B:unfolded': 50, 'monomer_C:unfolded': 100}
for k, v in current.items():
    trajectories[k][0] = v

events_sorted = sorted(state.events, key=lambda e: e.time)
ev_idx = 0
for ti, t in enumerate(times):
    while ev_idx < len(events_sorted) and events_sorted[ev_idx].time <= t:
        ev = events_sorted[ev_idx]
        if ev.rule_name == 'folding':
            pid = ev.participants[0]
            gene = state.proteins[pid].gene_id
            current[f'{gene}:unfolded'] = current.get(f'{gene}:unfolded', 0) - 1
            current[f'{gene}:native'] = current.get(f'{gene}:native', 0) + 1
        ev_idx += 1
    for k, v in current.items():
        trajectories[k][ti] = v

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, gene in zip(axes, ['kinase_A', 'substrate_B', 'monomer_C']):
    for suffix, style, color in [(':unfolded', '--', 'tab:orange'), (':native', '-', 'tab:blue')]:
        key = gene + suffix
        if key in trajectories:
            ax.plot(times, trajectories[key], style, color=color, label=suffix[1:], linewidth=2)
    ax.set_title(gene)
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Molecule count')
plt.suptitle('Folding dynamics — real-time per-species evolution', y=1.02)
plt.tight_layout()
plt.savefig(OUT / '02_folding_dynamics.png', dpi=120)
plt.close()

# ===== Viz 3: Assembly pathway =====
print("Generating assembly pathway plot...")
dimer_times = [c.formation_time for c in state.complexes.values() if c.quaternary_arrangement == 'dimer']
tet_times = [c.formation_time for c in state.complexes.values() if c.quaternary_arrangement == 'tetramer']

fig, ax = plt.subplots(figsize=(12, 4))
if dimer_times:
    ax.hist(dimer_times, bins=30, alpha=0.6, label=f'Dimers ({len(dimer_times)})', color='tab:blue')
if tet_times:
    ax.hist(tet_times, bins=30, alpha=0.6, label=f'Tetramers ({len(tet_times)})', color='tab:red')
ax.set_xlabel('Formation time (s)')
ax.set_ylabel('Count')
ax.set_title('Quaternary structure assembly — dimerization then tetramerization')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / '03_assembly_pathway.png', dpi=120)
plt.close()

# ===== Viz 4: Phosphorylation network activity =====
print("Generating phosphorylation activity plot...")
phospho_events = [e for e in state.events if e.rule_name == 'phosphorylation']
kinase_to_events = defaultdict(list)
for e in phospho_events:
    kinase_to_events[e.participants[0]].append(e.time)

fig, ax = plt.subplots(figsize=(12, 5))
for i, (k, ts) in enumerate(sorted(kinase_to_events.items())):
    ax.scatter(ts, [i]*len(ts), alpha=0.6, s=20)
ax.set_yticks(range(len(kinase_to_events)))
ax.set_yticklabels([f'kinase_A#{k}' for k in sorted(kinase_to_events.keys())])
ax.set_xlabel('Simulated time (s)')
ax.set_title(f'Phosphorylation events per kinase instance ({len(phospho_events)} total)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / '04_phospho_activity.png', dpi=120)
plt.close()

# ===== Viz 5: Single molecule life story =====
print("Generating single molecule life story...")
kinase_ids = list(state.proteins_by_gene['kinase_A'])
inst_id = kinase_ids[0]
p = state.proteins[inst_id]
hist = state.molecule_history(inst_id)

fig, ax = plt.subplots(figsize=(14, 3))
for t, desc in hist:
    if 'folding' in desc:
        c = 'tab:green'
    elif 'phosphorylation' in desc:
        c = 'tab:red'
    elif 'dimerization' in desc:
        c = 'tab:blue'
    else:
        c = 'gray'
    ax.axvline(t, color=c, alpha=0.7, linewidth=2)
ax.set_xlim(0, state.time)
ax.set_yticks([])
ax.set_xlabel('Simulated time (s)')
ax.set_title(f'Life story of kinase_A#{inst_id} — {len(hist)} events '
             f'(green=folding, red=phosphorylation, blue=dimerization)')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(OUT / '05_single_molecule.png', dpi=120)
plt.close()

# ===== Summary =====
print(f"\nAll plots saved to {OUT}")
for f in sorted(OUT.glob('*.png')):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")

# Save a summary text file
with open(OUT / 'SUMMARY.txt', 'w') as f:
    f.write(f"Event-driven simulation summary\n")
    f.write(f"="*60 + "\n")
    f.write(f"Simulated time:   {state.time:.3f} s\n")
    f.write(f"Wall time:        {stats['wall_time_s']:.3f} s\n")
    f.write(f"Speed:            {state.time/stats['wall_time_s']:.1f}x real time\n")
    f.write(f"Events fired:     {len(state.events)}\n")
    f.write(f"Protein molecules: {len(state.proteins)}\n")
    f.write(f"Complexes formed: {len(state.complexes)}\n")
    f.write(f"\nEvents by type:\n")
    for et in event_types:
        n = sum(1 for e in state.events if e.rule_name == et)
        f.write(f"  {et}: {n}\n")
    f.write(f"\nComplexes by type:\n")
    for arr, n in Counter(c.quaternary_arrangement for c in state.complexes.values()).items():
        f.write(f"  {arr}: {n}\n")

print(f"\nDone.")
