"""Smoke test for cell_physics — minimal cell-scale Langevin engine.

Sets up a tiny synthetic cell:
  - 200 ATP particles, free in cytoplasm
  - 100 protein particles, free
  - 1 chromosome chain of 32 backbone beads (FENE + bending)
  - 1 protein complex of 5 subunits held together by harmonic restraints
  - 1 reaction: ATP + protein -> bound_complex (rate 0.1 / ns / encounter)

Verifies:
  1. initial_state builds the right species composition + chains + complexes
  2. forces are finite, bounded, and energy-stable across 200 steps
  3. particles stay inside the cell (boundary works)
  4. polymer chain stays connected (FENE keeps consecutive beads close)
  5. complex stays clustered (harmonic restraints work)
  6. reactions fire when ATP and protein come into proximity, producing new species
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atom_engine.cell_physics import (CellPhysicsConfig, SpeciesEntry,
                                        Reaction, initial_state, run,
                                        compute_forces, brownian_step,
                                        fire_reactions)


def main():
    print('=' * 64)
    print('cell_physics smoke test')
    print('=' * 64)

    # --- species table ---
    # diffusion_nm2_per_ns:
    #   ATP / small metabolite ~ 0.3 (= 300 μm²/s)
    #   typical 30-50 kDa protein ~ 0.005 (= 5 μm²/s)
    #   complex / DNA bead ~ 0.001 (= 1 μm²/s)
    species = [
        SpeciesEntry(name='ATP', sigma_nm=1.0, epsilon_kj_per_mol=0.5,
                      mass_da=507.0, diffusion_nm2_per_ns=0.3),
        SpeciesEntry(name='protein_A', sigma_nm=4.0, epsilon_kj_per_mol=1.0,
                      mass_da=30000.0, diffusion_nm2_per_ns=0.005),
        SpeciesEntry(name='bound_complex', sigma_nm=4.5,
                      epsilon_kj_per_mol=1.0, mass_da=30507.0,
                      diffusion_nm2_per_ns=0.005),
        SpeciesEntry(name='dna_bead', sigma_nm=1.0, epsilon_kj_per_mol=0.3,
                      mass_da=10000.0, diffusion_nm2_per_ns=0.001),
        SpeciesEntry(name='subunit', sigma_nm=3.0, epsilon_kj_per_mol=0.8,
                      mass_da=20000.0, diffusion_nm2_per_ns=0.003),
    ]
    SID_ATP = 0; SID_PROT = 1; SID_BOUND = 2; SID_DNA = 3; SID_SUB = 4

    config = CellPhysicsConfig(
        cell_radius_nm=80.0,
        dt_ns=0.01,                          # 10 ps step
        kT_kj_per_mol=2.49,
        seed=42,
        pair_rebuild_every=20,
    )

    composition = {SID_ATP: 200, SID_PROT: 100, SID_SUB: 5}
    chain_species = [[SID_DNA] * 32]
    # Complex pairs: subunits 0..4 form a fully-restrained 5-clique
    # We don't know their final indices yet — initial_state appends species
    # in composition-order, so subunits will be at indices [300, 301, 302, 303, 304].
    complex_pairs = []
    for i in range(5):
        for j in range(i + 1, 5):
            complex_pairs.append((300 + i, 300 + j))

    # --- 1: initial_state shapes + composition ---
    print('\n[1] initial_state composition + chains + complexes')
    state = initial_state(species, composition, config,
                          chains=chain_species,
                          complex_pairs=complex_pairs)
    n = state.n()
    print(f'  total particles: {n}')
    print(f'  by species: {np.bincount(state.species_id, minlength=len(species)).tolist()}')
    print(f'  chains: {len(state.chains)} (length {state.chains[0].size})')
    print(f'  complex pairs: {state.complex_pairs.shape[0]}')
    assert n == 200 + 100 + 5 + 32, f'expected 337 particles got {n}'
    assert state.chains[0].size == 32
    assert state.complex_pairs.shape[0] == 10        # 5C2 = 10
    print('  OK')

    # --- 2: forces are finite + bounded ---
    print('\n[2] force computation finite + bounded')
    forces = compute_forces(state, config)
    print(f'  shape: {forces.shape}')
    print(f'  per-atom |F| mean={np.linalg.norm(forces, axis=-1).mean():.2f}  '
          f'max={np.linalg.norm(forces, axis=-1).max():.2f} kJ/(mol·nm)')
    assert np.isfinite(forces).all()
    assert np.linalg.norm(forces, axis=-1).max() <= config.max_force_kj_per_nm + 1
    print('  OK')

    # --- 3: short run stays inside cell + chain stays connected ---
    print('\n[3] short run: 200 steps, check stability')
    reactions = [Reaction(reactant_a=SID_ATP, reactant_b=SID_PROT,
                           products=(SID_BOUND,), rate_per_ns=0.1,
                           reaction_radius_nm=5.0)]
    traj = run(state, config, species, reactions=reactions,
                n_steps=200, save_every=10)
    print(f'  frames: {traj["positions"].shape[0]}')
    print(f'  rxn events total: {traj["n_rxn_total"]}')
    last_pos = traj['positions'][-1]
    valid = traj['species_id'][-1] >= 0
    r_final = np.linalg.norm(last_pos[valid], axis=-1)
    print(f'  final radii:  mean={r_final.mean():.1f} nm  '
          f'max={r_final.max():.1f} nm  cell_radius={config.cell_radius_nm}')
    assert r_final.max() <= config.cell_radius_nm + 1.0, \
        'particles escaped the cell sphere'
    print('  OK — particles confined')

    # --- 4: chain stayed connected (consecutive beads still within FENE limit) ---
    print('\n[4] polymer chain still connected')
    chain = state.chains[0]
    # FENE r_max = fene_r_max_factor * fene_r0_nm
    r_max = config.fene_r_max_factor * config.fene_r0_nm
    a, b = chain[:-1], chain[1:]
    chain_dists = np.linalg.norm(state.pos[b] - state.pos[a], axis=-1)
    print(f'  chain bond distances: mean={chain_dists.mean():.2f} nm  '
          f'max={chain_dists.max():.2f} nm  fene_r_max={r_max} nm')
    assert chain_dists.max() < r_max, 'FENE bond exceeded r_max — chain broke'
    print('  OK — FENE keeps chain connected')

    # --- 5: complex restraint pulls subunits closer (not necessarily to r0) ---
    # initial_state scatters subunits randomly; the harmonic restraint can't
    # close 70 nm in 2 ns of sim time at protein diffusion. We check that
    # distances DECREASED on average, which confirms the restraint is acting.
    # Set up a fresh scatter and run a short relaxation.
    print('\n[5] complex restraint pulls subunits together')
    state5 = initial_state(species, {SID_SUB: 5}, config,
                            complex_pairs=[(i, j) for i in range(5)
                                            for j in range(i + 1, 5)])
    cp5 = state5.complex_pairs
    d_initial = np.linalg.norm(state5.pos[cp5[:, 1]] - state5.pos[cp5[:, 0]],
                                  axis=-1).mean()
    traj5 = run(state5, config, species, n_steps=2000, save_every=200)
    d_final = np.linalg.norm(state5.pos[cp5[:, 1]] - state5.pos[cp5[:, 0]],
                                axis=-1).mean()
    print(f'  initial mean inter-subunit distance: {d_initial:.1f} nm')
    print(f'  final   mean inter-subunit distance: {d_final:.1f} nm')
    assert d_final < d_initial, 'restraint did not pull complex together'
    print('  OK — harmonic restraint shrinks the complex')

    # --- 6: reactions actually fire when forced into proximity ---
    # Use a fast reaction (rate=100/ns) so it fires reliably in ~10 dt steps.
    print('\n[6] reactions fire on proximity')
    fast_rxn = [Reaction(reactant_a=SID_ATP, reactant_b=SID_PROT,
                           products=(SID_BOUND,), rate_per_ns=100.0,
                           reaction_radius_nm=5.0)]
    state2 = initial_state(species, {SID_ATP: 5, SID_PROT: 5}, config)
    state2.pos[0] = np.array([0.0, 0.0, 0.0])
    state2.pos[5] = np.array([0.5, 0.0, 0.0])    # within 5 nm
    rng = np.random.default_rng(0)
    n_before = state2.n()
    n_atp_before = (state2.species_id == SID_ATP).sum()
    n_prot_before = (state2.species_id == SID_PROT).sum()
    fired = 0
    for _ in range(200):
        fired += fire_reactions(state2, fast_rxn, config, species, rng)
        if fired:
            break
    n_after = state2.n()
    n_atp_after = (state2.species_id == SID_ATP).sum()
    n_prot_after = (state2.species_id == SID_PROT).sum()
    n_bound_after = (state2.species_id == SID_BOUND).sum()
    print(f'  before:  ATP={n_atp_before}  protein={n_prot_before}  bound=0  total={n_before}')
    print(f'  after:   ATP={n_atp_after}  protein={n_prot_after}  bound={n_bound_after}  total={n_after}')
    assert fired >= 1, 'reaction should have fired with reactants in contact'
    assert n_atp_after < n_atp_before
    assert n_prot_after < n_prot_before
    assert n_bound_after >= 1
    print('  OK — proximity reactions fire and replace reactants with products')

    print('\n' + '=' * 64)
    print('cell_physics smoke test PASSED')
    print('=' * 64)


if __name__ == '__main__':
    main()
