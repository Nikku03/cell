"""Kitchen-sink demo: bio + physics + chemistry in one shot.

Exercises every piece the atom engine has accumulated across the
multi-session plan:

  Physics:  Coulomb (soft-core), angular + dihedral bonds, Langevin
            thermostat, SHAKE bond constraints, PBC with minimum-image
            + wrap, PBC-aware neighbor list, Rust LJ kernel.

  Biology:  Water, Alanine, and Serine loaded via the PDB importer
            (auto bonds / angles / charges).

  Chemistry: dynamic bond formation disabled for stability (would
             interact with SHAKE badly); the reactive-chemistry path
             remains available via reaction_demo / chemistry_demo.

Outputs a JSON summary with per-milestone T, bond-length drift,
angle drift, and H-bond count. Single run, reports honestly when
the trajectory starts heating.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from cell_sim.atom_engine.atom_unit import AtomUnit
from cell_sim.atom_engine.element import Element
from cell_sim.atom_engine.force_field import ForceFieldConfig, wrap_positions
from cell_sim.atom_engine.integrator import (
    IntegratorConfig,
    SimState,
    build_shake_constraints,
    current_temperature_K,
    step,
)
from cell_sim.atom_engine.pdb_importer import load_residue
from cell_sim.atom_engine.water_peptide_demo import count_hbonds


def _offset(structure, shift):
    for a in structure.atoms:
        a.position[0] += shift[0]
        a.position[1] += shift[1]
        a.position[2] += shift[2]


def _bond_drift(atoms, bonds):
    errs = []
    for b in bonds:
        if b.death_time_ps is not None:
            continue
        d = (np.array(b.b.position) - np.array(b.a.position))
        r = float(np.linalg.norm(d))
        errs.append(abs(r - b.equilibrium_length_nm))
    return float(np.mean(errs)) if errs else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-water", type=int, default=30)
    p.add_argument("--pbc-box-nm", type=float, default=1.8)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--dt-ps", type=float, default=0.001)
    p.add_argument("--report-every", type=int, default=1000)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    # Build the system: one ALA + one SER + N waters.
    print(f"loading residues: 1 ALA + 1 SER + {args.n_water} HOH")
    ala = load_residue("ALA", temperature_K=args.temperature,
                       parent_molecule="ala#0")
    ser = load_residue("SER", temperature_K=args.temperature,
                       parent_molecule="ser#1")
    _offset(ala, (-0.4, 0.0, 0.0))
    _offset(ser, (+0.4, 0.0, 0.0))

    atoms: list[AtomUnit] = list(ala.atoms) + list(ser.atoms)
    bonds = list(ala.bonds) + list(ser.bonds)
    angles = list(ala.angles) + list(ser.angles)
    dihedrals = list(ala.dihedrals) + list(ser.dihedrals)

    # Populate waters on a simple cubic grid inside the box, skipping
    # the slab we reserved for the peptides.
    import random as _r
    rng = _r.Random(7)
    waters_placed = 0
    spacing = 0.33
    positions_so_far = [a.position for a in atoms]
    L = args.pbc_box_nm
    grid = int(L / spacing) + 1
    candidates = []
    for ix in range(grid):
        for iy in range(grid):
            for iz in range(grid):
                x = -0.5 * L + spacing * (ix + 0.5)
                y = -0.5 * L + spacing * (iy + 0.5)
                z = -0.5 * L + spacing * (iz + 0.5)
                candidates.append((x, y, z))
    rng.shuffle(candidates)
    # Water exclusion radius needs to cover the template extent
    # (~0.1 nm to the farthest H) plus LJ sigma (~0.3 nm) to avoid
    # overlap at placement time. 0.4 nm is conservative.
    for (x, y, z) in candidates:
        if waters_placed >= args.n_water:
            break
        too_close = False
        for p in positions_so_far:
            if (x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2 < 0.40 ** 2:
                too_close = True
                break
        if too_close:
            continue
        hoh = load_residue("HOH", temperature_K=args.temperature,
                           parent_molecule=f"water#{waters_placed}")
        _offset(hoh, (x, y, z))
        atoms.extend(hoh.atoms)
        bonds.extend(hoh.bonds)
        angles.extend(hoh.angles)
        for ha in hoh.atoms:
            positions_so_far.append(ha.position)
        waters_placed += 1
    print(f"placed {waters_placed} waters (target {args.n_water})")

    # Wrap into PBC box and build the SimState.
    pos_arr = np.array([a.position for a in atoms], dtype=np.float64)
    wrap_positions(pos_arr, args.pbc_box_nm)
    for i, a in enumerate(atoms):
        a.position[0] = float(pos_arr[i, 0])
        a.position[1] = float(pos_arr[i, 1])
        a.position[2] = float(pos_arr[i, 2])

    state = SimState(atoms=atoms, bonds=bonds, angles=angles,
                     dihedrals=dihedrals)
    state.shake_pairs, state.shake_r0_sq = build_shake_constraints(atoms, bonds)

    ff = ForceFieldConfig(
        lj_cutoff_nm=min(0.8, 0.4 * args.pbc_box_nm),
        use_confinement=False,
        use_neighbor_list=True,
        use_coulomb=True,
        use_pbc=True,
        pbc_box_nm=args.pbc_box_nm,
    )
    int_cfg = IntegratorConfig(
        dt_ps=args.dt_ps,
        target_temperature_K=args.temperature,
        thermostat="langevin",
        langevin_gamma_inv_ps=5.0,
        shake=True,
    )

    print(f"\nsystem: {len(atoms)} atoms, {len(bonds)} bonds, "
          f"{len(angles)} angles, {len(dihedrals)} dihedrals, "
          f"SHAKE pairs = {state.shake_pairs.shape[0]}, "
          f"PBC L = {args.pbc_box_nm:.2f} nm")

    traj = {
        "t_ps": [], "temperature_K": [], "bond_drift_nm": [],
        "hbonds_per_water": [], "dead_bonds": [],
    }
    t0 = time.time()
    forces = None
    for k in range(args.steps):
        forces = step(state, ff, int_cfg, forces)
        if (k + 1) % args.report_every == 0 or k == args.steps - 1:
            T = current_temperature_K(state.atoms)
            drift = _bond_drift(state.atoms, state.bonds)
            _, hb = count_hbonds(state.atoms, state.bonds)
            dead = sum(1 for b in state.bonds if b.death_time_ps is not None)
            traj["t_ps"].append(state.t_ps)
            traj["temperature_K"].append(T)
            traj["bond_drift_nm"].append(drift)
            traj["hbonds_per_water"].append(hb)
            traj["dead_bonds"].append(dead)
            print(f"  step {k+1:>6}/{args.steps} t={state.t_ps:.2f} ps "
                  f"T={T:.0f} K  bond_drift={drift*1000:.3f} pm  "
                  f"HB/water={hb:.2f}  dead_bonds={dead}")
    elapsed = time.time() - t0

    # Final summary: how long did we stay stable?
    stable_T_limit = args.temperature * 2.0
    stable_until_ps = None
    for i, T in enumerate(traj["temperature_K"]):
        if T > stable_T_limit:
            break
        stable_until_ps = traj["t_ps"][i]

    print(f"\ndone in {elapsed:.1f} s "
          f"({args.steps / max(elapsed, 1e-6):.0f} steps/s)")
    print(f"stable T-bounded window: up to t = "
          f"{stable_until_ps:.2f} ps" if stable_until_ps
          else "no stable window")
    print(f"final bond drift:       "
          f"{traj['bond_drift_nm'][-1] * 1000:.3f} pm")
    print(f"final HB/water:         {traj['hbonds_per_water'][-1]:.2f}")

    if args.out:
        out = {
            "n_atoms": len(atoms),
            "n_bonds": len(bonds),
            "n_angles": len(angles),
            "n_dihedrals": len(dihedrals),
            "n_shake_pairs": int(state.shake_pairs.shape[0]),
            "pbc_box_nm": args.pbc_box_nm,
            "temperature_K_target": args.temperature,
            "dt_ps": args.dt_ps,
            "steps": args.steps,
            "elapsed_s": elapsed,
            "stable_until_ps": stable_until_ps,
            "final_bond_drift_pm": traj["bond_drift_nm"][-1] * 1000,
            "final_hbonds_per_water": traj["hbonds_per_water"][-1],
            "trajectory": traj,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"summary: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
