"""Step B: per-locus spatial features from MinCell_{1..4}.lm files.

For each gene locus in Syn3A, computes per-particle spatial statistics from
the 4 spatial replicates, aggregates to per-locus features, and saves a
parquet for downstream essentiality / model-conditioning use.

Features (per locus):

  Gene (G_LLLL_C1 / G_LLLL_C2):
    gene_dist_to_membrane_mean    mean Euclidean voxel distance to nearest
                                  membrane voxel, averaged across particles
                                  + timesteps + replicates
    gene_dist_to_membrane_std     spread of the above
    gene_n_observations           total particle-timestep observations

  mRNA (R_LLLL):
    rna_dist_to_membrane_mean
    rna_dist_to_degradosome_mean
    rna_frac_in_cytoplasm
    rna_frac_in_outer_cytoplasm
    rna_n_observations

  Protein (P_LLLL + PM_LLLL bundled by locus):
    protein_dist_to_membrane_mean
    protein_frac_in_membrane      fraction of observations where site-type
                                  == membrane (or outer_cytoplasm)
    protein_frac_in_cytoplasm
    protein_n_observations

  Translation complexes (RP_LLLL_C1/C2, RP_LLLL_f_C1/C2, RPM_LLLL):
    translation_dist_to_membrane_mean
    translation_frac_in_membrane
    translation_n_observations

Sampling: every `sample_every` timesteps (default 600 -> 12 samples per
replicate -> 48 total observations per particle across 4 reps).

Site types in MinCell_*.lm:
  0=extracellular, 1=cytoplasm, 2=outer_cytoplasm, 3=ribosomes,
  4=ribo_centers, 5=DNA, 6=membrane

The distance transform is computed once per timestep using
scipy.ndimage.distance_transform_edt against the dynamic membrane mask.
Particle positions are read directly from sim['Lattice'][key] which has
shape (128, 64, 64, 16) and stores up to 16 particle IDs per voxel
(uint32; particle_id == species_index + 1, 0 means empty slot).
"""
from __future__ import annotations
import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage


def _decode_species_names(raw):
    out = []
    for x in raw:
        v = x
        while hasattr(v, '__len__') and not isinstance(v, (bytes, str, np.bytes_)):
            if len(v) == 0:
                v = b''; break
            v = v[0]
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode('utf-8', 'replace').rstrip('\x00')
        else:
            v = str(v)
        out.append(v)
    return out


def _locus_to_species(species_names):
    """Map each 4-digit locus to the list of species names belonging to it,
    grouped by category (gene/rna/protein/translation/cofactor)."""
    _RE = re.compile(r'_(\d{4})(?:_|$)')
    by_locus = defaultdict(lambda: {'gene': [], 'rna': [], 'protein': [],
                                      'translation': [], 'membrane': []})
    for n in species_names:
        m = _RE.search(n)
        if not m:
            continue
        L = m.group(1)
        if n.startswith('G_'):
            by_locus[L]['gene'].append(n)
        elif n.startswith('R_'):
            by_locus[L]['rna'].append(n)
        elif n.startswith('RP_') or n.startswith('RPM_'):
            by_locus[L]['translation'].append(n)
        elif n.startswith('P_'):
            by_locus[L]['protein'].append(n)
        elif n.startswith('PM_'):
            by_locus[L]['membrane'].append(n)
    return dict(by_locus)


def _flatten_particle_positions(lattice, species_ids):
    """Find (x, y, z) coordinates of all voxels containing any particle in
    `species_ids` (a set of int particle IDs = species_index + 1).
    `lattice` is shape (X, Y, Z, slots) uint32. Returns array (N, 3) of voxels."""
    # Any slot in voxel containing one of our species IDs
    mask_per_slot = np.isin(lattice, list(species_ids))  # (X, Y, Z, slots)
    voxel_has_any = mask_per_slot.any(axis=-1)            # (X, Y, Z)
    coords = np.argwhere(voxel_has_any)                   # (N, 3)
    # also count multiplicity per voxel (for the n_observations stat)
    multiplicity = mask_per_slot.sum(axis=-1)             # (X, Y, Z)
    return coords, multiplicity


def _particle_observations(coords, multiplicity, dist_map, site_map,
                              site_id_map):
    """For each voxel in `coords`, look up the membrane distance and the
    site type. Returns dict of aggregates over particles (counted with
    multiplicity)."""
    if coords.size == 0:
        return None
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    n_per_voxel = multiplicity[xs, ys, zs]           # (N,)
    dist_per_voxel = dist_map[xs, ys, zs]            # (N,)
    site_per_voxel = site_map[xs, ys, zs]            # (N,)

    # Weight by multiplicity so a voxel with 5 ribosomes counts 5x
    total_n = int(n_per_voxel.sum())
    weighted_dist = float((dist_per_voxel * n_per_voxel).sum() / total_n)
    site_counts = {}
    for name, sid in site_id_map.items():
        site_counts[name] = int(((site_per_voxel == sid) * n_per_voxel).sum())
    return {
        'n_observations': total_n,
        'dist_to_membrane_mean': weighted_dist,
        'site_counts': site_counts,
    }


def _process_lm(lm_path: Path, sample_every: int = 600, verbose: bool = True):
    """Single .lm file -> per-locus per-category observation stream."""
    print(f'\n=== {lm_path.name} ===')
    rep_t0 = time.time()
    with h5py.File(lm_path, 'r') as f:
        sN = _decode_species_names(f['Parameters']['SpeciesNames'][:])
        site_names = f['Parameters'].attrs['siteTypeNames']
        if isinstance(site_names, bytes):
            site_names = site_names.decode().split(',')
        site_id_map = {name: i for i, name in enumerate(site_names)}
        membrane_id = site_id_map['membrane']

        name_to_pid = {n: i + 1 for i, n in enumerate(sN)}
        locus_map = _locus_to_species(sN)
        if verbose:
            print(f'  species={len(sN)} loci_with_particles={len(locus_map)} '
                  f'site_types={list(site_id_map.keys())}')

        # Degradosome particle ID (for mRNA distance-to-degradosome)
        degradosome_pid = name_to_pid.get('Degradosome', None)
        if degradosome_pid is None:
            print('  WARNING: no Degradosome species; '
                  'rna_dist_to_degradosome will be NaN')

        sim_key = list(f['Simulations'].keys())[0]
        sim = f['Simulations'][sim_key]
        lattice_keys = sorted(sim['Lattice'].keys())
        sample_keys = lattice_keys[::sample_every]
        if verbose:
            print(f'  sampling {len(sample_keys)} of {len(lattice_keys)} '
                  f'timesteps (every {sample_every}s)')

        # Accumulator: per (locus, category), running n + sum(dist) + site_counts
        agg = defaultdict(lambda: defaultdict(lambda: {
            'n': 0,
            'dist_sum': 0.0,
            'site_counts': {s: 0 for s in site_names},
            'degrad_dist_sum': 0.0,
            'degrad_n': 0,
        }))

        for t_idx, key in enumerate(sample_keys):
            t = int(key)
            t_t0 = time.time()
            lattice = np.asarray(sim['Lattice'][key])      # (128, 64, 64, 16)
            sites   = np.asarray(sim['Sites'][key])        # (128, 64, 64)

            # Distance transform from membrane (1 voxel = ~8 nm; report in voxels)
            membrane_mask = (sites == membrane_id)
            # distance_transform_edt computes distance to nearest 0 in input.
            # We want distance to nearest True (membrane), so invert.
            dist_map = ndimage.distance_transform_edt(~membrane_mask)  # (X, Y, Z) float

            # Distance transform from degradosome particles (if exists)
            if degradosome_pid is not None:
                deg_mask = (lattice == degradosome_pid).any(axis=-1)
                if deg_mask.any():
                    deg_dist_map = ndimage.distance_transform_edt(~deg_mask)
                else:
                    deg_dist_map = None
            else:
                deg_dist_map = None

            # For each locus and category, find particles and update aggregates
            for locus, cats in locus_map.items():
                for cat, species_list in cats.items():
                    if not species_list:
                        continue
                    species_ids = {name_to_pid[n] for n in species_list
                                    if n in name_to_pid}
                    if not species_ids:
                        continue
                    coords, mult = _flatten_particle_positions(lattice, species_ids)
                    obs = _particle_observations(
                        coords, mult, dist_map, sites, site_id_map,
                    )
                    if obs is None:
                        continue
                    a = agg[locus][cat]
                    a['n'] += obs['n_observations']
                    a['dist_sum'] += (obs['dist_to_membrane_mean']
                                        * obs['n_observations'])
                    for s_name, s_count in obs['site_counts'].items():
                        a['site_counts'][s_name] += s_count
                    if cat == 'rna' and deg_dist_map is not None and coords.size:
                        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
                        n_per_v = mult[xs, ys, zs]
                        a['degrad_dist_sum'] += float(
                            (deg_dist_map[xs, ys, zs] * n_per_v).sum())
                        a['degrad_n'] += int(n_per_v.sum())

            if verbose and (t_idx + 1) % 5 == 0:
                print(f'    [{t_idx+1}/{len(sample_keys)}] t={t}s '
                      f'wall={time.time()-t_t0:.2f}s '
                      f'cum={time.time()-rep_t0:.1f}s')

    return agg


def _flatten_agg(rep_aggs, site_names):
    """List of per-replicate agg dicts -> per-locus per-category combined
    table."""
    combined = defaultdict(lambda: defaultdict(lambda: {
        'n': 0, 'dist_sum': 0.0,
        'site_counts': {s: 0 for s in site_names},
        'degrad_dist_sum': 0.0, 'degrad_n': 0,
    }))
    for agg in rep_aggs:
        for locus, cats in agg.items():
            for cat, a in cats.items():
                c = combined[locus][cat]
                c['n'] += a['n']
                c['dist_sum'] += a['dist_sum']
                for s in site_names:
                    c['site_counts'][s] += a['site_counts'][s]
                c['degrad_dist_sum'] += a['degrad_dist_sum']
                c['degrad_n'] += a['degrad_n']

    rows = []
    for locus, cats in combined.items():
        row = {'locus': locus}
        for cat in ('gene', 'rna', 'protein', 'membrane', 'translation'):
            a = cats.get(cat)
            prefix = cat
            if a is None or a['n'] == 0:
                row[f'{prefix}_n_obs'] = 0
                row[f'{prefix}_dist_to_membrane_mean'] = float('nan')
                for s in site_names:
                    row[f'{prefix}_frac_in_{s}'] = float('nan')
                if cat == 'rna':
                    row['rna_dist_to_degradosome_mean'] = float('nan')
                continue
            row[f'{prefix}_n_obs'] = a['n']
            row[f'{prefix}_dist_to_membrane_mean'] = a['dist_sum'] / a['n']
            for s in site_names:
                row[f'{prefix}_frac_in_{s}'] = a['site_counts'][s] / a['n']
            if cat == 'rna':
                if a['degrad_n'] > 0:
                    row['rna_dist_to_degradosome_mean'] = (
                        a['degrad_dist_sum'] / a['degrad_n']
                    )
                else:
                    row['rna_dist_to_degradosome_mean'] = float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


def main(lm_dir: str, sample_every: int, out_path: str, replicates: list[int]):
    lm_dir = Path(lm_dir)
    paths = []
    for r in replicates:
        p = lm_dir / f'MinCell_{r}.lm'
        if not p.exists():
            print(f'WARNING: {p} missing, skipping')
            continue
        paths.append(p)
    if not paths:
        raise RuntimeError(f'No replicates found in {lm_dir}')
    print(f'Processing {len(paths)} replicates with sample_every={sample_every}')

    rep_aggs = []
    site_names = None
    for p in paths:
        agg = _process_lm(p, sample_every=sample_every)
        rep_aggs.append(agg)
        if site_names is None:
            with h5py.File(p, 'r') as f:
                sn = f['Parameters'].attrs['siteTypeNames']
                if isinstance(sn, bytes):
                    sn = sn.decode().split(',')
                site_names = sn

    df = _flatten_agg(rep_aggs, site_names)
    df = df.sort_values('locus').reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f'\nsaved {len(df)} loci x {df.shape[1]} cols -> {out_path}')
    print(f'\nFirst 5 rows:')
    print(df.head().to_string(index=False))
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lm-dir',
        default='/content/drive/MyDrive/Minimal_Cell_4DWCM/extracted/Minimal_Cell_4DWCM/DATA/lm_trajectories')
    ap.add_argument('--sample-every', type=int, default=600,
        help='Sample every N timesteps (default 600 -> 12 samples per rep)')
    ap.add_argument('--out',
        default='memory_bank/data/syn3a_spatial_features.parquet')
    ap.add_argument('--replicates', type=int, nargs='+', default=[1, 2, 3, 4])
    args = ap.parse_args()
    main(args.lm_dir, args.sample_every, args.out, args.replicates)
