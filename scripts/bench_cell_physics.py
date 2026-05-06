"""Benchmark cell_physics — pair-list build + 100 integration steps at
multiple particle counts. Reports wall-time per step and projected time
for 1 μs (100K steps at dt=10 ps) of simulated cell life.

Run: python scripts/bench_cell_physics.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'cell_sim'))

import numpy as np

from cell_sim.atom_engine.cell_physics import (
    CellPhysicsConfig, SpeciesEntry, initial_state, compute_forces,
    brownian_step, _build_pair_list,
)


def make_table(n_species: int = 5) -> list[SpeciesEntry]:
    return [
        SpeciesEntry(name=f's{i}', sigma_nm=2.0, epsilon_kj_per_mol=0.5,
                      mass_da=10000.0, diffusion_nm2_per_ns=0.05)
        for i in range(n_species)
    ]


def benchmark(n_particles: int, n_steps: int = 100, cell_radius_nm: float = 200.0):
    species = make_table(5)
    cfg = CellPhysicsConfig(cell_radius_nm=cell_radius_nm, dt_ns=0.01,
                             pair_rebuild_every=20, seed=42)
    composition = {i: n_particles // 5 for i in range(5)}

    print(f'\n--- N = {n_particles} particles, '
          f'cell radius {cell_radius_nm} nm ---')
    t0 = time.time()
    state = initial_state(species, composition, cfg)
    t_init = time.time() - t0
    print(f'  init state: {t_init*1000:.1f} ms  '
          f'(actual N = {state.n()})')

    # Time pair-list build alone
    t0 = time.time()
    iu, ju = _build_pair_list(state, cfg)
    t_pl = time.time() - t0
    print(f'  pair list:  {t_pl*1000:.1f} ms   '
          f'({iu.size} pairs, ratio {iu.size/state.n():.1f}/particle)')

    # Time force eval alone
    t0 = time.time()
    F = compute_forces(state, cfg)
    t_force = time.time() - t0
    print(f'  forces:     {t_force*1000:.1f} ms')

    # Time full step (force + Brownian)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for _ in range(n_steps):
        F = compute_forces(state, cfg)
        brownian_step(state, cfg, F, rng)
    t_run = time.time() - t0
    per_step = t_run / n_steps
    print(f'  full step:  {per_step*1000:.2f} ms / step  '
          f'({n_steps} steps in {t_run:.2f} s)')

    steps_for_1us = 100_000   # 1 μs / 10 ps per step
    proj = steps_for_1us * per_step
    if proj < 60:
        proj_str = f'{proj:.1f} s'
    elif proj < 3600:
        proj_str = f'{proj/60:.1f} min'
    elif proj < 86400:
        proj_str = f'{proj/3600:.1f} hr'
    else:
        proj_str = f'{proj/86400:.1f} days'
    print(f'  projected for 1 μs (100K steps): {proj_str}')
    return per_step


def main():
    print('=' * 64)
    print('cell_physics benchmark (with cKDTree pair list)')
    print('=' * 64)

    # Scale up the cell radius slightly with particle count so the density
    # stays roughly constant (otherwise more particles = same volume = much
    # higher density = many more pairs).
    radii = {
        1_000:  100.0,
        5_000:  170.0,
        10_000: 215.0,
        50_000: 370.0,
        100_000: 460.0,
    }
    timings = {}
    for n, r in radii.items():
        try:
            timings[n] = benchmark(n, n_steps=20, cell_radius_nm=r)
        except Exception as e:
            print(f'  FAILED at N={n}: {e}')
            timings[n] = None

    print('\n' + '=' * 64)
    print('summary: time per step + projection for 1 μs simulated time')
    print('=' * 64)
    print(f'{"N particles":>12s}  {"ms/step":>10s}  {"steps/s":>10s}  '
          f'{"1 μs proj":>12s}')
    for n, t in timings.items():
        if t is None:
            print(f'{n:>12d}  {"FAIL":>10s}')
            continue
        ms = t * 1000
        sps = 1.0 / t
        proj = 100_000 * t
        if proj < 60:
            ps = f'{proj:.1f} s'
        elif proj < 3600:
            ps = f'{proj/60:.1f} min'
        else:
            ps = f'{proj/3600:.1f} hr'
        print(f'{n:>12d}  {ms:>10.2f}  {sps:>10.1f}  {ps:>12s}')


if __name__ == '__main__':
    main()
