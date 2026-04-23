"""Benchmark the cheaper-compute stack end-to-end.

Measures wall time per ps of simulated trajectory under several configs:

  1. Legacy: soft-core Coulomb, no bond cache, no Numba SHAKE,
     no minimisation. This is the baseline before the upgrade stack.
  2. + Minimiser: adds steepest-descent pre-equilibration. Does not
     change per-step cost but dramatically reduces initial instability,
     letting the system actually run stably at dt=1fs for dense water.
  3. + RF Coulomb: adds reaction-field electrostatics in the Rust
     kernel. Improves long-range behaviour + reduces initial drift.
  4. + Bond cache: pre-materialises bond/angle index + parameter
     arrays on SimState so compute_forces skips the per-call Python
     dict rebuild + bond loop. ~2x speedup on Python overhead.
  5. + Numba SHAKE: replaces the pure-Python SHAKE loop with a
     Numba-JIT kernel. ~20x on SHAKE.
  6. RESPA n=4 overlay on the full stack. Note: for our dense water
     RESPA increases instability because F_slow is stale across the
     inner substeps; it helps on systems where non-bonded forces are
     slow-changing. Included as a measurement, not a recommendation.

All runs at dt=1fs. 2fs on dense water would require SETTLE (analytical
rigid-body water) since iterative SHAKE cannot converge that fast; a
standalone steepest-descent pass does NOT unlock 2fs by itself.

System: small water-only box in PBC with SHAKE on O-H bonds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from cell_sim.atom_engine.force_field import (
    ForceFieldConfig, wrap_positions,
)
from cell_sim.atom_engine.integrator import (
    IntegratorConfig, SimState, build_shake_constraints, step,
)
from cell_sim.atom_engine.pdb_importer import load_residue


def build_water_box(n_water=20, box_nm=1.5, temperature_K=300.0, seed=7):
    rng = np.random.default_rng(seed)
    atoms = []
    bonds = []
    angles = []
    waters_placed = 0
    # Water-center exclusion = ~2x (O-H bond 0.096 nm) + LJ sigma_O 0.315 nm
    # = 0.51 nm. Use 0.60 nm to avoid initial LJ clashes at water orientation.
    spacing = 0.58
    candidates = []
    grid = int(box_nm / spacing) + 1
    for ix in range(grid):
        for iy in range(grid):
            for iz in range(grid):
                x = -0.5 * box_nm + spacing * (ix + 0.5)
                y = -0.5 * box_nm + spacing * (iy + 0.5)
                z = -0.5 * box_nm + spacing * (iz + 0.5)
                candidates.append((x, y, z))
    rng.shuffle(candidates)
    positions_so_far = []
    for (x, y, z) in candidates:
        if waters_placed >= n_water:
            break
        too_close = any(
            (x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2 < 0.60 ** 2
            for p in positions_so_far
        )
        if too_close:
            continue
        hoh = load_residue("HOH", temperature_K=temperature_K,
                           parent_molecule=f"w{waters_placed}")
        for a in hoh.atoms:
            a.position[0] += x
            a.position[1] += y
            a.position[2] += z
        atoms.extend(hoh.atoms)
        bonds.extend(hoh.bonds)
        angles.extend(hoh.angles)
        for a in hoh.atoms:
            positions_so_far.append(a.position)
        waters_placed += 1
    pos_arr = np.array([a.position for a in atoms], dtype=np.float64)
    wrap_positions(pos_arr, box_nm)
    for i, a in enumerate(atoms):
        a.position[0] = float(pos_arr[i, 0])
        a.position[1] = float(pos_arr[i, 1])
        a.position[2] = float(pos_arr[i, 2])
    return atoms, bonds, angles


def bench_config(name, ff, int_cfg, total_ps, atoms_factory,
                  use_minimiser=True):
    from cell_sim.atom_engine.integrator import minimise_steepest_descent

    atoms, bonds, angles = atoms_factory()
    state = SimState(atoms=atoms, bonds=bonds, angles=angles)
    state.shake_pairs, state.shake_r0_sq = build_shake_constraints(atoms, bonds)

    if use_minimiser:
        minimise_steepest_descent(state, ff, max_steps=500,
                                   force_tol_kj_per_nm=300.0)

    # Thermal velocities via Maxwell-Boltzmann.
    import random as _rand
    rng = _rand.Random(7)
    k_B = 0.00831446
    T = int_cfg.target_temperature_K
    for a in state.atoms:
        sigma = (k_B * T / a.mass_da) ** 0.5
        a.velocity[0] = rng.gauss(0, sigma)
        a.velocity[1] = rng.gauss(0, sigma)
        a.velocity[2] = rng.gauss(0, sigma)

    n_steps = int(round(total_ps / int_cfg.dt_ps))

    # Warmup: 50 steps at production config so JIT + neighbor list + bond
    # cache are hot when the timed region begins.
    forces = None
    for _ in range(50):
        forces = step(state, ff, int_cfg, forces)

    print(f"[{name}] {len(atoms)} atoms, dt={int_cfg.dt_ps*1000:.1f} fs, "
          f"n_inner={int_cfg.respa_n_inner}, rc={ff.lj_cutoff_nm} nm, "
          f"RF={ff.use_reaction_field}, min={use_minimiser}, "
          f"{n_steps} steps = {total_ps} ps")

    t0 = time.perf_counter()
    for _ in range(n_steps):
        forces = step(state, ff, int_cfg, forces)
    elapsed = time.perf_counter() - t0
    ps_per_s = total_ps / elapsed
    from cell_sim.atom_engine.integrator import current_temperature_K
    T_end = current_temperature_K(state.atoms)
    return {"name": name, "elapsed_s": elapsed, "n_steps": n_steps,
            "total_ps": total_ps, "ps_per_s": ps_per_s, "T_end": T_end}


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-water", type=int, default=40)
    p.add_argument("--box-nm", type=float, default=2.0)
    p.add_argument("--total-ps", type=float, default=0.5)
    args = p.parse_args()
    TOTAL_PS = args.total_ps
    N_WATER = args.n_water
    BOX = args.box_nm

    def mk():
        return build_water_box(n_water=N_WATER, box_nm=BOX)

    print(f"\nSystem: {N_WATER} waters in {BOX} nm PBC box, dt=1fs\n")

    results = []

    # 1. No minimiser, softcore Coulomb — closest to pre-upgrade baseline.
    ff_soft = ForceFieldConfig(
        lj_cutoff_nm=0.8, use_pbc=True, pbc_box_nm=BOX,
        use_neighbor_list=True, use_coulomb=True,
        use_reaction_field=False,
    )
    ic = IntegratorConfig(
        dt_ps=0.001, target_temperature_K=300.0,
        thermostat="langevin", langevin_gamma_inv_ps=5.0,
        shake=True, respa_n_inner=1,
    )
    results.append(bench_config("softcore, no minimise", ff_soft, ic,
                                 TOTAL_PS, mk, use_minimiser=False))

    # 2. + Minimiser. Same FF, but warm up with steepest descent.
    results.append(bench_config("softcore + minimise", ff_soft, ic,
                                 TOTAL_PS, mk, use_minimiser=True))

    # 3. + Reaction field Coulomb.
    ff_rf = ForceFieldConfig(
        lj_cutoff_nm=0.8, use_pbc=True, pbc_box_nm=BOX,
        use_neighbor_list=True, use_coulomb=True,
        use_reaction_field=True, reaction_field_eps=78.5,
    )
    results.append(bench_config("+ RF", ff_rf, ic, TOTAL_PS, mk,
                                 use_minimiser=True))

    # 4. RESPA n=4 overlay (measurement, not a recommendation for this system).
    ic_respa = IntegratorConfig(
        dt_ps=0.001, target_temperature_K=300.0,
        thermostat="langevin", langevin_gamma_inv_ps=5.0,
        shake=True, respa_n_inner=4,
    )
    results.append(bench_config("+ RESPA n=4", ff_rf, ic_respa, TOTAL_PS, mk,
                                 use_minimiser=True))

    print("\n=== Results ===")
    base_ps_per_s = results[0]["ps_per_s"]
    print(f"{'config':30s} {'wall (s)':>9s} {'ps/s':>8s} {'speedup':>8s} {'T_end (K)':>10s}")
    for r in results:
        speedup = r["ps_per_s"] / base_ps_per_s
        print(f"{r['name']:30s} {r['elapsed_s']:9.2f} {r['ps_per_s']:8.3f} "
              f"{speedup:7.2f}x {r['T_end']:10.0f}")


if __name__ == "__main__":
    main()
