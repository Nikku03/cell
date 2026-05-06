"""Benchmark cell_physics — pair-list build + 100 integration steps at
multiple particle counts. Compares CPU (cKDTree + numpy) vs GPU (cKDTree
pair list + torch on cuda).

Usage:
  python scripts/bench_cell_physics.py            # both backends if cuda
  python scripts/bench_cell_physics.py --no-gpu   # CPU only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'cell_sim'))

import numpy as np

from cell_sim.atom_engine.cell_physics import (
    CellPhysicsConfig, SpeciesEntry, initial_state, compute_forces,
    brownian_step, _build_pair_list, run as cpu_run,
)


def make_table(n_species: int = 5) -> list[SpeciesEntry]:
    return [
        SpeciesEntry(name=f's{i}', sigma_nm=2.0, epsilon_kj_per_mol=0.5,
                      mass_da=10000.0, diffusion_nm2_per_ns=0.05)
        for i in range(n_species)
    ]


def fmt_proj(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.1f} s'
    if seconds < 3600:
        return f'{seconds/60:.1f} min'
    if seconds < 86400:
        return f'{seconds/3600:.1f} hr'
    return f'{seconds/86400:.1f} days'


def benchmark_cpu(n_particles: int, n_steps: int, cell_radius_nm: float
                   ) -> float:
    species = make_table(5)
    cfg = CellPhysicsConfig(cell_radius_nm=cell_radius_nm, dt_ns=0.01,
                             pair_rebuild_every=20, seed=42)
    composition = {i: n_particles // 5 for i in range(5)}
    state = initial_state(species, composition, cfg)
    rng = np.random.default_rng(0)
    # Warmup
    for _ in range(3):
        F = compute_forces(state, cfg)
        brownian_step(state, cfg, F, rng)
    t0 = time.time()
    for _ in range(n_steps):
        F = compute_forces(state, cfg)
        brownian_step(state, cfg, F, rng)
    return (time.time() - t0) / n_steps


def benchmark_gpu(n_particles: int, n_steps: int, cell_radius_nm: float,
                   device: str = 'cuda') -> float:
    import torch
    from cell_sim.atom_engine.cell_physics_gpu import run_gpu
    species = make_table(5)
    cfg = CellPhysicsConfig(cell_radius_nm=cell_radius_nm, dt_ns=0.01,
                             pair_rebuild_every=20, seed=42)
    composition = {i: n_particles // 5 for i in range(5)}
    state = initial_state(species, composition, cfg)
    # Warmup
    _ = run_gpu(state, cfg, species, n_steps=5, save_every=5, device=device)
    # Reset state for clean timing
    state = initial_state(species, composition, cfg)
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    _ = run_gpu(state, cfg, species, n_steps=n_steps,
                 save_every=max(n_steps, 1), device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-gpu', action='store_true')
    args = ap.parse_args()

    print('=' * 72)
    print('cell_physics benchmark — CPU (cKDTree+numpy) vs GPU (cKDTree+torch)')
    print('=' * 72)

    try_gpu = not args.no_gpu
    if try_gpu:
        try:
            import torch
            try_gpu = torch.cuda.is_available()
            if try_gpu:
                print(f'CUDA: {torch.cuda.get_device_name(0)}  '
                      f'(VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
            else:
                print('CUDA not available — CPU only')
        except ImportError:
            try_gpu = False
            print('torch not installed — CPU only')

    radii = {
        1_000:   100.0,
        5_000:   170.0,
        10_000:  215.0,
        50_000:  370.0,
        100_000: 460.0,
        500_000: 790.0,
    }

    print(f'\n{"N":>10s}  {"CPU ms/step":>14s}  {"GPU ms/step":>14s}  '
          f'{"CPU 1μs":>14s}  {"GPU 1μs":>14s}  {"speedup":>10s}')
    print('-' * 80)
    n_steps_per_size = {
        1_000:   200, 5_000: 100, 10_000: 50, 50_000: 20,
        100_000: 10,  500_000: 5,
    }
    for n, r in radii.items():
        ns = n_steps_per_size[n]
        cpu_t = benchmark_cpu(n, ns, r)
        if try_gpu:
            try:
                gpu_t = benchmark_gpu(n, ns, r)
            except Exception as e:
                print(f'{n:>10d}  GPU FAIL: {e}')
                gpu_t = None
        else:
            gpu_t = None
        cpu_proj = 100_000 * cpu_t
        gpu_proj = 100_000 * gpu_t if gpu_t is not None else None
        sup = f'{cpu_t/gpu_t:.1f}×' if gpu_t else '—'
        print(f'{n:>10d}  {cpu_t*1000:>14.2f}  '
              f'{gpu_t*1000 if gpu_t else float("nan"):>14.2f}  '
              f'{fmt_proj(cpu_proj):>14s}  '
              f'{fmt_proj(gpu_proj) if gpu_proj is not None else "—":>14s}  '
              f'{sup:>10s}')


if __name__ == '__main__':
    main()
