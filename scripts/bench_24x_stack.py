"""Benchmark the cheaper-compute stack end-to-end.

Measures wall time per ps of simulated trajectory under four configs:

  1. Legacy: dt=1fs, soft-core Coulomb (Python), rc=0.8 nm, no RESPA.
     Coulomb runs in a separate NumPy pass.
  2. Combined Rust: dt=1fs, soft-core Coulomb (in Rust kernel), rc=0.8,
     no RESPA. Isolates the savings from combining LJ+Coulomb in Rust.
  3. + RF + rc=0.6: dt=1fs, reaction-field Coulomb in Rust, rc=0.6 nm,
     no RESPA. Adds the cutoff shrinkage on top of combined Rust path.
  4. Full stack (RESPA): dt=1fs outer, RESPA n_inner=4 (inner dt=0.25fs),
     RF Coulomb, rc=0.6 nm.

All runs at dt=1fs (2fs requires energy minimization which isn't
implemented yet — see OVERNIGHT_SUMMARY next-workstreams item #2).

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
    spacing = 0.32
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
            (x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2 < 0.34 ** 2
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


def bench_config(name, ff, int_cfg, total_ps, atoms_factory):
    atoms, bonds, angles = atoms_factory()
    state = SimState(atoms=atoms, bonds=bonds, angles=angles)
    state.shake_pairs, state.shake_r0_sq = build_shake_constraints(atoms, bonds)
    n_steps = int(round(total_ps / int_cfg.dt_ps / int_cfg.respa_n_inner))
    # Equivalent simulated time for outer steps; in RESPA the outer dt
    # IS the full step so we don't scale by n_inner. Recompute cleanly:
    n_steps = int(round(total_ps / int_cfg.dt_ps))

    # Equilibrate from the raw PDB geometry at a short dt so the initial
    # potential-energy spike settles (saves benches from blowing up at 2fs).
    # Use the SAME force field (only timestep changes) so thermostat is
    # consistent.
    warm_cfg = IntegratorConfig(
        dt_ps=min(0.0005, int_cfg.dt_ps),
        target_temperature_K=int_cfg.target_temperature_K,
        thermostat=int_cfg.thermostat,
        langevin_gamma_inv_ps=int_cfg.langevin_gamma_inv_ps,
        shake=int_cfg.shake, respa_n_inner=1,
    )
    forces = None
    for _ in range(1000):
        forces = step(state, ff, warm_cfg, forces)
    # Reset RESPA cached slow force since ff params are same but we switched
    # to RESPA config; force a fresh compute.
    state._respa_F_slow = None
    # Tiny warmup at production dt to amortise JIT / neighbor-list build.
    forces = None
    for _ in range(10):
        forces = step(state, ff, int_cfg, forces)

    print(f"[{name}] {len(atoms)} atoms, dt={int_cfg.dt_ps*1000:.1f} fs, "
          f"n_inner={int_cfg.respa_n_inner}, rc={ff.lj_cutoff_nm} nm, "
          f"RF={ff.use_reaction_field}, running {n_steps} steps = "
          f"{total_ps} ps")

    t0 = time.perf_counter()
    for _ in range(n_steps):
        forces = step(state, ff, int_cfg, forces)
    elapsed = time.perf_counter() - t0
    ps_per_s = total_ps / elapsed
    return {"name": name, "elapsed_s": elapsed, "n_steps": n_steps,
            "total_ps": total_ps, "ps_per_s": ps_per_s}


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-water", type=int, default=80)
    p.add_argument("--box-nm", type=float, default=2.3)
    p.add_argument("--total-ps", type=float, default=1.0)
    args = p.parse_args()
    TOTAL_PS = args.total_ps
    N_WATER = args.n_water
    BOX = args.box_nm

    def mk():
        return build_water_box(n_water=N_WATER, box_nm=BOX)

    print(f"\nSystem: {N_WATER} waters in {BOX} nm PBC box\n")
    print("NOTE: all runs at dt=1fs. 2fs unlock requires steepest-descent\n"
          "minimisation (see OVERNIGHT_SUMMARY next-workstreams #2).\n")

    results = []

    # Baseline: soft-core Coulomb, rc=0.8
    ff = ForceFieldConfig(
        lj_cutoff_nm=0.8, use_pbc=True, pbc_box_nm=BOX,
        use_neighbor_list=True, use_coulomb=True,
        use_reaction_field=False,
    )
    ic = IntegratorConfig(
        dt_ps=0.001, target_temperature_K=300.0,
        thermostat="langevin", langevin_gamma_inv_ps=5.0,
        shake=True, respa_n_inner=1,
    )
    results.append(bench_config("baseline (softcore, rc0.8)", ff, ic, TOTAL_PS, mk))

    # + RF + rc=0.6 (reaction field enables smaller cutoff)
    ff2 = ForceFieldConfig(
        lj_cutoff_nm=0.6, use_pbc=True, pbc_box_nm=BOX,
        use_neighbor_list=True, use_coulomb=True,
        use_reaction_field=True, reaction_field_eps=78.5,
    )
    ic2 = IntegratorConfig(
        dt_ps=0.001, target_temperature_K=300.0,
        thermostat="langevin", langevin_gamma_inv_ps=5.0,
        shake=True, respa_n_inner=1,
    )
    results.append(bench_config("+ RF + rc0.6", ff2, ic2, TOTAL_PS, mk))

    # Full stack at 1fs: + RESPA n=4 (inner dt = 0.25fs)
    ic3 = IntegratorConfig(
        dt_ps=0.001, target_temperature_K=300.0,
        thermostat="langevin", langevin_gamma_inv_ps=5.0,
        shake=True, respa_n_inner=4,
    )
    results.append(bench_config("full stack (RESPA n=4)", ff2, ic3, TOTAL_PS, mk))

    print("\n=== Results ===")
    base_ps_per_s = results[0]["ps_per_s"]
    for r in results:
        speedup = r["ps_per_s"] / base_ps_per_s
        print(f"{r['name']:30s}  {r['elapsed_s']:6.2f} s  "
              f"{r['ps_per_s']:7.3f} ps/s  {speedup:5.2f}x")


if __name__ == "__main__":
    main()
