"""4D atom/particle tracker — Brownian overlay on v15's well-mixed trajectories.

We do NOT redo any chemistry. We take what cell_sim already produced
(per-species counts at 0.5 s cadence) and *spatially decorate* it: place
that many molecules at random positions inside a syn3A-sized sphere,
integrate Brownian motion at microsecond cadence, and dump a 4D tensor
(t, particle_id, [x, y, z, species_id]).

Same trick the games industry uses: bake the heavy sim, replay as a
point cache. The result is suitable for direct ingestion into a 4D
Gaussian-Splatting pipeline, voxel rendering, or as spatial features
for a downstream essentiality predictor.

Inputs:
  outputs/syn3a_trajectories_t2.0s.npz    — (455 genes, 5 timepoints, 262 features)
                                            We use the 21 chemical species and
                                            average across genes to get a single
                                            "cell" trajectory.

Outputs:
  outputs/syn3a_cell_4d.npz
    positions   (T_frames, N_max, 3)         float32  — nm; NaN for empty slot
    species_id  (T_frames, N_max)            int16    — 0..n_species-1; -1 if empty
    counts      (T_frames, n_species)        int32    — particles of each species per frame
    species     (n_species,)                 str      — species name lookup
    t_us        (T_frames,)                  float32  — μs from start
    cell_radius_nm  scalar
    diffusion_um2_s (n_species,)             float32

Per-frame slots are SORTED BY SPECIES, so positions[t, offset:offset+counts[t,s]]
all belong to species s, where offset = counts[t,:s].sum().
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRAJ_NPZ = ROOT / 'outputs/syn3a_trajectories_t2.0s.npz'
OUT_NPZ = ROOT / 'outputs/syn3a_cell_4d.npz'

# --------------------------------------------------------------------------
# Diffusion coefficients (μm² / s) for the 21 species we have, in cytoplasm.
# Order matters: must match the species names in the npz.
# Sources: Milo & Phillips (Cell Biology by the Numbers), Mika & Poolman 2011.
DIFFUSION_TABLE = {
    'ADP':                3e2,   # ~3e-10 m²/s — small metabolite
    'ATP':                3e2,
    'CTP':                3e2,
    'GTP':                3e2,
    'UTP':                3e2,
    'F6P':                3e2,
    'G6P':                3e2,
    'NAD':                3e2,
    'NADH':               3e2,
    'PYR':                3e2,
    'NTP_TOTAL':          3e2,
    'dATP':               3e2,
    'dCTP':               3e2,
    'dGTP':               3e2,
    'dTTP':               3e2,
    'FOLDED_PROTEINS':    5.0,   # ~5e-12 m²/s — typical 30-50 kDa protein
    'UNFOLDED_PROTEINS':  10.0,  # unfolded chains diffuse slightly faster
    'BOUND_PROTEINS':     1.0,   # tied up in complexes — slow
    'TOTAL_COMPLEXES':    0.5,   # large assemblies — very slow
    'FOLDED_FRACTION':    5.0,   # pseudospecies, treat as protein
    'TOTAL_EVENTS':       0.0,   # event counter, not spatial — pin to centre
}


def load_trajectory():
    d = np.load(TRAJ_NPZ, allow_pickle=True)
    species = [str(s) for s in d['species']]                  # 21 species
    n_species = len(species)
    traj = d['trajectories'][:, :, :n_species]                 # (455, 5, 21)
    avg = traj.mean(axis=0)                                     # (5, 21) per-cell average
    dt_s = float(d['dt_s'])
    t_end = float(d['t_end_s'])
    return species, avg, dt_s, t_end


def species_to_counts(values: np.ndarray, total_budget: int) -> np.ndarray:
    """Convert raw trajectory values (assumed proportional to abundance) into
    integer particle counts that sum to roughly ``total_budget``.

    The npz values look log-scale (range 0..13 with mean ~10 for major
    metabolites) — we treat them as relative weights without trying to recover
    absolute units. Each species gets ``round(weight / total_weight * budget)``
    particles, with a min of 1 if its weight is non-zero. Total may differ
    slightly from ``total_budget`` due to rounding.
    """
    w = np.maximum(values, 0.0).astype(np.float64)
    if w.sum() <= 0:
        return np.zeros_like(values, dtype=np.int64)
    counts = np.round(w / w.sum() * total_budget).astype(np.int64)
    # Ensure nonzero species get at least 1 particle
    counts[(w > 0) & (counts == 0)] = 1
    return counts


def random_in_sphere(n: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform random sample inside a sphere of given radius (nm)."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    # Rejection sampling — simple and exact.
    out = np.empty((n, 3), dtype=np.float32)
    placed = 0
    while placed < n:
        batch = max(n - placed, 64)
        v = rng.uniform(-radius, radius, size=(batch * 2, 3))
        keep = (v ** 2).sum(axis=1) <= radius ** 2
        v = v[keep]
        take = min(len(v), n - placed)
        out[placed:placed + take] = v[:take]
        placed += take
    return out


def brownian_step(pos: np.ndarray, D_um2_per_s: np.ndarray,
                  species_id: np.ndarray, dt_s: float,
                  rng: np.random.Generator) -> np.ndarray:
    """One Brownian-motion step. ``D`` is per-species in μm²/s; positions in nm.

    Standard deviation of displacement per axis: σ = √(2 D dt).
    """
    sigma_nm_per_axis = np.sqrt(2.0 * D_um2_per_s[species_id] * dt_s) * 1000.0
    return pos + rng.normal(0.0, 1.0, size=pos.shape) * sigma_nm_per_axis[:, None]


def reflect_into_sphere(pos: np.ndarray, radius: float) -> np.ndarray:
    """Push any particle that left the sphere back to the boundary along its
    radial direction. Simple "reflect at wall" — not strictly thermodynamic
    but visually correct for confinement."""
    r = np.linalg.norm(pos, axis=1)
    out = pos.copy()
    bad = r > radius
    if bad.any():
        scale = radius / np.maximum(r[bad], 1e-9)
        out[bad] = pos[bad] * scale[:, None] * 0.999
    return out


def adjust_population(positions: list[np.ndarray], target_counts: np.ndarray,
                      radius: float, rng: np.random.Generator) -> list[np.ndarray]:
    """Birth / kill particles per species to match new target counts.
    ``positions`` is a per-species list of (n_s, 3) arrays."""
    new_pos = []
    for s_idx, target in enumerate(target_counts):
        cur = positions[s_idx]
        if cur.shape[0] == target:
            new_pos.append(cur)
        elif cur.shape[0] > target:
            keep = rng.choice(cur.shape[0], size=target, replace=False)
            new_pos.append(cur[keep])
        else:
            extra = random_in_sphere(target - cur.shape[0], radius, rng)
            new_pos.append(np.concatenate([cur, extra], axis=0))
    return new_pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell_radius_nm', type=float, default=200.0,
                    help='syn3A radius in nm (~200 nm = 0.2 μm)')
    p.add_argument('--n_frames', type=int, default=1000,
                    help='number of 4D frames to emit')
    p.add_argument('--dt_us', type=float, default=1.0,
                    help='time between emitted frames in microseconds')
    p.add_argument('--total_particles', type=int, default=5000,
                    help='approximate total particle budget (across all species)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=Path, default=OUT_NPZ)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    radius = args.cell_radius_nm
    dt_s = args.dt_us * 1e-6

    print('[1] loading v15 trajectory...')
    species, traj_avg, coarse_dt_s, t_end_s = load_trajectory()
    n_species = len(species)
    n_coarse = traj_avg.shape[0]
    print(f'  {n_species} species  '
          f'{n_coarse} coarse timepoints (dt={coarse_dt_s}s, t_end={t_end_s}s)')

    # Diffusion table aligned to species order
    D = np.array([DIFFUSION_TABLE.get(s, 1.0) for s in species],
                 dtype=np.float32)
    missing = [s for s in species if s not in DIFFUSION_TABLE]
    if missing:
        print(f'  WARN: no diffusion entry for {missing} — defaulted to 1.0 μm²/s')

    # Total fine-frame duration must fit within v15 trajectory
    total_us = args.n_frames * args.dt_us
    total_s = total_us * 1e-6
    if total_s > t_end_s:
        print(f'  WARN: requested {total_s}s > trajectory {t_end_s}s — '
              f'will hold counts at last value beyond it')

    print(f'\n[2] initializing population at t=0  (radius={radius:.0f} nm)...')
    init_counts = species_to_counts(traj_avg[0], args.total_particles)
    print(f'  total particles at t=0: {int(init_counts.sum())}')
    for i, s in enumerate(species):
        if init_counts[i]:
            print(f'    {s:<20s}  n={int(init_counts[i]):>5d}  '
                  f'D={D[i]:>6.1f} μm²/s')

    # Per-species position lists (variable length)
    positions = [random_in_sphere(int(c), radius, rng) for c in init_counts]

    print(f'\n[3] simulating Brownian motion: '
          f'{args.n_frames} frames × {args.dt_us} μs = {total_us:.1f} μs total')
    # Pre-allocate output. Particles are stored per-frame sorted by species,
    # so a downstream consumer can slice positions[t, offset:offset+counts[t,s]]
    # to recover species s. Empty slots are NaN / -1.
    n_max = max(args.total_particles + 200, int(init_counts.sum()) + 200)
    out_pos = np.full((args.n_frames, n_max, 3), np.nan, dtype=np.float32)
    out_sid = np.full((args.n_frames, n_max), -1, dtype=np.int16)
    counts_log = np.zeros((args.n_frames, n_species), dtype=np.int32)

    # Main loop — fine cadence, population-matched per frame
    for f in range(args.n_frames):
        # Brownian step on every species's particle list
        for s_idx in range(n_species):
            arr = positions[s_idx]
            if arr.shape[0]:
                sigma_nm = np.sqrt(2.0 * D[s_idx] * dt_s) * 1000.0
                arr = arr + rng.normal(0.0, sigma_nm, size=arr.shape).astype(np.float32)
                arr = reflect_into_sphere(arr, radius)
                positions[s_idx] = arr

        # Match population to v15-derived target at this time
        t_now_s = (f + 1) * dt_s
        coarse_idx = min(int(t_now_s / coarse_dt_s), n_coarse - 1)
        target_counts = species_to_counts(traj_avg[coarse_idx], args.total_particles)
        positions = adjust_population(positions, target_counts, radius, rng)

        # Emit a sorted-by-species frame
        write_idx = 0
        for s_idx in range(n_species):
            arr = positions[s_idx]
            n = arr.shape[0]
            if n == 0:
                continue
            if write_idx + n > n_max:
                # Hit the slot ceiling — drop overflow at random
                room = n_max - write_idx
                if room <= 0:
                    break
                keep = rng.choice(n, size=room, replace=False)
                arr = arr[keep]
                n = room
            out_pos[f, write_idx:write_idx + n] = arr
            out_sid[f, write_idx:write_idx + n] = s_idx
            write_idx += n
        counts_log[f] = target_counts.astype(np.int32)

        if (f + 1) % max(args.n_frames // 10, 1) == 0:
            print(f'  frame {f+1}/{args.n_frames}  '
                  f't={(f+1)*args.dt_us:>7.1f} μs  N={write_idx}')

    print(f'\n[4] writing {args.out}...')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        positions=out_pos,
        species_id=out_sid,
        counts=counts_log,
        species=np.array(species),
        t_us=np.arange(args.n_frames, dtype=np.float32) * args.dt_us,
        cell_radius_nm=np.float32(radius),
        diffusion_um2_s=D,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f'  wrote {size_mb:.1f} MB  '
          f'({args.n_frames} frames × {n_max} slots × 3 floats)')
    print(f'  shape: positions={out_pos.shape}  species_id={out_sid.shape}')


if __name__ == '__main__':
    main()
