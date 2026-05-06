"""Run a 4D Brownian-dynamics simulation of a syn3A-flavoured minimal cell.

Uses what we have on hand (no Zenodo download, no Lattice Microbes install):
  - cell_sim/atom_engine/cell_physics.py (Brownian dynamics + soft LJ + FENE
    chain + harmonic complexes + soft sphere wall + proximity reactions)
  - a representative species + reaction set built into this script,
    inspired by v15's syn3A kinetics (real_syn3a_rules.py)

What this is
------------
A real 4D simulation: every molecule is a 3D-positioned particle moving
under physics, reactions firing on actual proximity, output is a trajectory
of (positions, species_id, t) over simulated time.

What this is NOT
----------------
A complete syn3A model. We take a representative subset of reactions —
translation, complex assembly, ATP usage, ribosome cycling — and a static
chromosome polymer + a few complexes. NO replication, NO division, NO
biogenesis (intentional v1 scope per design discussion).

Usage:
  python scripts/run_4d_minimal_cell.py \
      --duration_ns 100 --dt_ns 0.01 --save_every 100

Output:
  outputs/syn3a_4d_minimal_cell.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cell_sim'))

from cell_sim.atom_engine.cell_physics import (
    CellPhysicsConfig, SpeciesEntry, Reaction, CellState,
    initial_state, run,
)


# ──────────────────────────────────────────────────────────────────────
# A representative syn3A-flavoured species table.
# Diffusion in nm²/ns:
#   small metabolite   ~ 0.3   (ATP, ADP, GTP, GDP, Pi)
#   mRNA               ~ 0.05  (~100 nucleotides, hydrodynamic radius ~5 nm)
#   protein 30-50 kDa  ~ 0.005
#   ribosome (70S)     ~ 0.001
#   complex (formed)   ~ 0.001
#   chromosome bead    ~ 0.0005
def species_table() -> list[SpeciesEntry]:
    return [
        # 0..4 metabolites
        SpeciesEntry('ATP', sigma_nm=1.0, epsilon_kj_per_mol=0.5,
                      mass_da=507.0,  diffusion_nm2_per_ns=0.3),
        SpeciesEntry('ADP', sigma_nm=1.0, epsilon_kj_per_mol=0.5,
                      mass_da=427.0,  diffusion_nm2_per_ns=0.3),
        SpeciesEntry('GTP', sigma_nm=1.0, epsilon_kj_per_mol=0.5,
                      mass_da=523.0,  diffusion_nm2_per_ns=0.3),
        SpeciesEntry('GDP', sigma_nm=1.0, epsilon_kj_per_mol=0.5,
                      mass_da=443.0,  diffusion_nm2_per_ns=0.3),
        SpeciesEntry('Pi',  sigma_nm=0.7, epsilon_kj_per_mol=0.3,
                      mass_da=95.0,   diffusion_nm2_per_ns=0.5),
        # 5..6 mRNAs
        SpeciesEntry('mRNA_A', sigma_nm=4.0, epsilon_kj_per_mol=0.7,
                      mass_da=30000.0, diffusion_nm2_per_ns=0.05),
        SpeciesEntry('mRNA_B', sigma_nm=4.0, epsilon_kj_per_mol=0.7,
                      mass_da=30000.0, diffusion_nm2_per_ns=0.05),
        # 7..8 proteins
        SpeciesEntry('Protein_A', sigma_nm=3.0, epsilon_kj_per_mol=1.0,
                      mass_da=30000.0, diffusion_nm2_per_ns=0.005),
        SpeciesEntry('Protein_B', sigma_nm=3.0, epsilon_kj_per_mol=1.0,
                      mass_da=30000.0, diffusion_nm2_per_ns=0.005),
        # 9 ribosome
        SpeciesEntry('Ribosome', sigma_nm=10.0, epsilon_kj_per_mol=1.5,
                      mass_da=2.6e6, diffusion_nm2_per_ns=0.001),
        # 10 complex
        SpeciesEntry('Complex_AB', sigma_nm=5.0, epsilon_kj_per_mol=1.2,
                      mass_da=60000.0, diffusion_nm2_per_ns=0.001),
        # 11 chromosome bead
        SpeciesEntry('DNA_bead', sigma_nm=1.0, epsilon_kj_per_mol=0.3,
                      mass_da=10000.0, diffusion_nm2_per_ns=0.0005),
    ]


SID_ATP, SID_ADP, SID_GTP, SID_GDP, SID_PI = 0, 1, 2, 3, 4
SID_MRNA_A, SID_MRNA_B = 5, 6
SID_PROT_A, SID_PROT_B = 7, 8
SID_RIBO = 9
SID_CPLX_AB = 10
SID_DNA = 11


def reactions() -> list[Reaction]:
    """Bimolecular reactions only (cell_physics v1). Rates expressed as
    per-encounter probability rate per ns (i.e. when the partners are
    within ``reaction_radius_nm`` for one timestep, fire with prob
    ``rate * dt``).

    Rates here are intentionally on the higher side for visible dynamics
    in a short simulation. Real physiological values are typically 10-100x
    slower; we trade realism for visible activity in the v1 demo.
    """
    return [
        # Translation: ribosome encounters mRNA → produces protein
        Reaction(reactant_a=SID_RIBO, reactant_b=SID_MRNA_A,
                 products=(SID_RIBO, SID_PROT_A),    # ribosome released
                 rate_per_ns=0.05, reaction_radius_nm=12.0),
        Reaction(reactant_a=SID_RIBO, reactant_b=SID_MRNA_B,
                 products=(SID_RIBO, SID_PROT_B),
                 rate_per_ns=0.05, reaction_radius_nm=12.0),
        # Complex assembly
        Reaction(reactant_a=SID_PROT_A, reactant_b=SID_PROT_B,
                 products=(SID_CPLX_AB,),
                 rate_per_ns=0.02, reaction_radius_nm=4.0),
        # ATP usage (mock kinase: uses ATP, releases ADP + Pi)
        Reaction(reactant_a=SID_ATP, reactant_b=SID_PROT_A,
                 products=(SID_ADP, SID_PI, SID_PROT_A),
                 rate_per_ns=0.005, reaction_radius_nm=3.0),
        # GTP cycling on ribosome
        Reaction(reactant_a=SID_GTP, reactant_b=SID_RIBO,
                 products=(SID_GDP, SID_PI, SID_RIBO),
                 rate_per_ns=0.01, reaction_radius_nm=10.0),
    ]


def initial_composition() -> dict[int, int]:
    """Approximate copy numbers for syn3A (rough, demo-scale)."""
    return {
        SID_ATP:    600,
        SID_ADP:    150,
        SID_GTP:    400,
        SID_GDP:    100,
        SID_PI:     200,
        SID_MRNA_A: 30,
        SID_MRNA_B: 30,
        SID_PROT_A: 80,
        SID_PROT_B: 80,
        SID_RIBO:   30,
    }


def chromosome_chain(n_beads: int = 64) -> list[int]:
    """A static chromosome polymer of N beads (DNA_bead species)."""
    return [SID_DNA] * n_beads


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--duration_ns', type=float, default=50.0,
                    help='simulated time in ns')
    p.add_argument('--dt_ns', type=float, default=0.01,
                    help='integration step in ns (10 ps default)')
    p.add_argument('--save_every', type=int, default=100,
                    help='save trajectory every N steps')
    p.add_argument('--cell_radius_nm', type=float, default=200.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=Path,
                    default=ROOT / 'outputs/syn3a_4d_minimal_cell.npz')
    p.add_argument('--n_chain_beads', type=int, default=64)
    args = p.parse_args()

    n_steps = int(args.duration_ns / args.dt_ns)
    print(f'[1] config:')
    print(f'  cell_radius = {args.cell_radius_nm} nm')
    print(f'  duration    = {args.duration_ns} ns')
    print(f'  dt          = {args.dt_ns} ns')
    print(f'  n_steps     = {n_steps}')
    print(f'  save_every  = {args.save_every}  '
          f'(save cadence = {args.save_every * args.dt_ns} ns)')

    species = species_table()
    print(f'\n[2] species table: {len(species)} species')
    for i, s in enumerate(species):
        print(f'  {i:>2d}  {s.name:<12s}  sigma={s.sigma_nm:.1f}  '
              f'D={s.diffusion_nm2_per_ns:.4f} nm²/ns')

    config = CellPhysicsConfig(
        cell_radius_nm=args.cell_radius_nm,
        dt_ns=args.dt_ns,
        seed=args.seed,
        pair_rebuild_every=20,
    )

    print(f'\n[3] building initial state...')
    composition = initial_composition()
    chain_species = chromosome_chain(args.n_chain_beads)
    state = initial_state(species, composition, config,
                          chains=[chain_species])
    print(f'  total particles: {state.n()}')
    print(f'  chromosome:      {len(state.chains[0])} beads')

    rxns = reactions()
    print(f'\n[4] reactions: {len(rxns)}')
    for i, r in enumerate(rxns):
        a = species[r.reactant_a].name
        b = species[r.reactant_b].name
        prod = ' + '.join(species[p].name for p in r.products)
        print(f'  R{i}  {a} + {b} -> {prod}  '
              f'(rate={r.rate_per_ns} /ns, r_cut={r.reaction_radius_nm} nm)')

    print(f'\n[5] running {n_steps} steps...')
    t0 = time.time()
    traj = run(state, config, species, reactions=rxns,
                n_steps=n_steps, save_every=args.save_every)
    elapsed = time.time() - t0
    print(f'  done in {elapsed:.1f} s '
          f'({n_steps / elapsed:.0f} steps/s)')

    print(f'\n[6] reaction events: {traj["n_rxn_total"]}')
    counts_initial = traj['counts'][0]
    counts_final = traj['counts'][-1]
    print(f'  species         t=0      t={args.duration_ns}ns   delta')
    for i, name in enumerate(traj['species_names']):
        d = int(counts_final[i]) - int(counts_initial[i])
        print(f'  {name:<12s}  {int(counts_initial[i]):>5d}  '
              f'{int(counts_final[i]):>5d}    {d:+5d}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        positions=traj['positions'],
        species_id=traj['species_id'],
        counts=traj['counts'],
        t_ns=traj['t_ns'],
        rxn_events=traj['rxn_events'],
        species=np.array(traj['species_names']),
        cell_radius_nm=np.float32(args.cell_radius_nm),
    )
    print(f'\nwrote {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)')
    print(f'  shape: positions={traj["positions"].shape}')


if __name__ == '__main__':
    main()
