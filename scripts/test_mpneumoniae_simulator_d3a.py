"""Phase D3a probe: test whether cell_sim can mechanically run on
M. pneumoniae data via runtime DATA_ROOT monkey-patching.

If this works end-to-end, it produces a partial proof of concept that
the simulator generalizes given the right files in the right schema.
If it fails, the failure mode tells us what the proper Phase D3
refactor needs to handle.

Approach:
  1. Monkey-patch DATA_ROOT in the syn3a_real, kinetics, sbml_parser,
     gene_expression, reversible modules to point at the M. pneumoniae
     input_data dir.
  2. Construct a RealSimulator with sbml_path pointing at the
     M. pneumoniae SBML.
  3. Try to run a single knockout.
  4. Report what worked and what didn't.

This is intentionally hacky. The proper Phase D3 refactor is a separate
session.
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'cell_sim'))
sys.path.insert(0, str(REPO))

MPNE_DATA = REPO / 'cell_sim/data/Mycoplasma_pneumoniae/input_data'
print(f'M. pneumoniae data dir: {MPNE_DATA}')

# 1. Monkey-patch DATA_ROOT in every module that uses it
print('\n[1] monkey-patching DATA_ROOT in cell_sim modules...')
for mod_name in ['layer0_genome.syn3a_real',
                 'layer3_reactions.kinetics',
                 'layer3_reactions.gene_expression']:
    __import__(mod_name)
    mod = sys.modules[mod_name]
    if hasattr(mod, 'DATA_ROOT'):
        old = mod.DATA_ROOT
        mod.DATA_ROOT = MPNE_DATA
        print(f'    patched {mod_name}.DATA_ROOT')
        print(f'         {old}  ->  {MPNE_DATA}')

# 2. Construct RealSimulator config pointed at M. pneumoniae SBML
print('\n[2] constructing RealSimulator with M. pneumoniae SBML...')
from cell_sim.layer6_essentiality.real_simulator import (
    RealSimulator, RealSimulatorConfig,
)

cfg = RealSimulatorConfig(
    scale_factor=0.05,
    seed=42,
    sbml_path=MPNE_DATA / 'M_pneumoniae_iJW145.xml',
    cell_volume_um3=(4.0/3.0) * 3.14159 * (0.4**3),  # M. pneumoniae ~400 nm radius (vs Syn3A 200 nm)
    use_rust_backend=False,
    enable_gene_expression=False,
    enable_imb155_patches=False,
)
sim = RealSimulator(cfg)
print(f'    created RealSimulator with sbml_path={cfg.sbml_path}')

# 3. Try _ensure_setup
print('\n[3] running _ensure_setup() ...')
t0 = time.time()
try:
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        sim._ensure_setup()
    print(f'    _ensure_setup() completed in {time.time()-t0:.1f}s')
    print(f'    spec proteins: {len(sim._spec.proteins)}')
    print(f'    counts entries: {len(sim._counts_template)}')
    print(f'    complexes: {len(sim._complexes)}')
    print(f'    rev_rules: {len(sim._rev_rules)}')
    print(f'    extra_rules: {len(sim._extra_rules)}')
except Exception as e:
    print(f'    FAILED at _ensure_setup: {e}')
    traceback.print_exc()
    sys.exit(1)

# 4. Try a WT run (no knockout)
print('\n[4] running a wild-type trajectory (no KO) for 0.1s sim time...')
t0 = time.time()
try:
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        traj = sim.run([], t_end_s=0.1, sample_dt_s=0.05)
    print(f'    WT run completed in {time.time()-t0:.1f}s wall')
    print(f'    samples: {len(traj.samples)}')
    if traj.samples:
        last = traj.samples[-1]
        print(f'    last sample t_s: {last.t_s}')
        print(f'    last sample TOTAL_EVENTS: {last.pools.get("TOTAL_EVENTS", "missing")}')
        print(f'    pools (first 8): {list(last.pools)[:8]}')
        print(f'    event_counts_by_rule: {len(last.event_counts_by_rule or {})} entries')
except Exception as e:
    print(f'    FAILED at run(): {e}')
    traceback.print_exc()
    sys.exit(1)

# 5. Try a knockout (use first MPN gene from spec)
print('\n[5] running a knockout for one M. pneumoniae gene (0.1s)...')
spec_genes = list(sim._spec.proteins.keys())
test_ko = next((g for g in spec_genes if g.startswith('MPN')), spec_genes[0])
print(f'    KO target: {test_ko}')
t0 = time.time()
try:
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        traj_ko = sim.run([test_ko], t_end_s=0.1, sample_dt_s=0.05)
    print(f'    KO run completed in {time.time()-t0:.1f}s wall')
    print(f'    samples: {len(traj_ko.samples)}')
    if traj_ko.samples:
        last = traj_ko.samples[-1]
        print(f'    last sample TOTAL_EVENTS: {last.pools.get("TOTAL_EVENTS", "missing")}')
except Exception as e:
    print(f'    FAILED at KO run(): {e}')
    traceback.print_exc()
    sys.exit(1)

print('\n[OK] Phase D3a probe SUCCEEDED. cell_sim runs mechanically on')
print('     M. pneumoniae data via DATA_ROOT monkey-patching.')
