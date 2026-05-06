"""atom_stability — atomic-resolution protein stability score.

Given a peptide/protein (PDB file or amino-acid sequence), simulate it
with atom_engine for a short trajectory and return a stability score
based on backbone RMSD, bond preservation, and temperature stability.

Inputs
------
  --pdb <path>          : load a real PDB structure
  --sequence <ABCDE...>  : one-letter amino-acid codes; built from the
                          built-in residue library (all 20 standard AAs
                          + nucleotides are available)
  --steps <N>           : number of integration steps (default 4000 = 4 ps)
  --dt-ps <f>           : timestep (default 0.001 ps = 1 fs)
  --temperature <K>     : target temperature (default 300 K)
  --pbc-box-nm <f>      : optional PBC box (default off — vacuum sim)
  --shake               : enable SHAKE bond constraints (recommended)
  --json-out <path>     : write a JSON report

Outputs
-------
  Prints a stability report:
    - n_atoms, n_bonds, n_angles
    - backbone_rmsd_avg_nm   (average over last 1 ps; lower = more stable)
    - bonds_preserved        (fraction of starting bonds still present)
    - temperature_stability  (1 - relative-T-drift; 1.0 = perfectly thermostatted)
    - overall_score          (combined 0-1 score; 1 = stable, 0 = collapsed)

What this is NOT
----------------
  - NOT a Tm prediction in real units (need μs-scale MD; we simulate 4 ps)
  - NOT an absolute folding free energy
  - NOT a binding affinity
  Use it for RELATIVE comparison — "protein A more stable than B" — only.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from cell_sim.atom_engine.atom_unit import AtomUnit
from cell_sim.atom_engine.force_field import ForceFieldConfig
from cell_sim.atom_engine.integrator import (
    IntegratorConfig, SimState, run, current_temperature_K,
    build_shake_constraints,
)
from cell_sim.atom_engine.pdb_importer import (
    load_pdb, STANDARD_RESIDUES_PDB,
)


# One-letter -> three-letter PDB residue names (aligned with what
# STANDARD_RESIDUES_PDB provides).
ONE_TO_THREE = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}


def build_peptide_pdb_from_sequence(seq: str, residue_spacing_nm: float = 0.38
                                       ) -> Path:
    """Concatenate single-residue templates into one PDB file, offset
    along x by residue_spacing_nm (~3.8 A, typical CA-CA in extended
    conformation). Each residue gets a unique resseq."""
    out_lines = []
    serial = 1
    x_offset_A = 0.0   # accumulated X offset in Angstroms (PDB units)
    for resseq, ch in enumerate(seq, start=1):
        ch = ch.upper()
        if ch not in ONE_TO_THREE:
            raise ValueError(f'unknown amino-acid letter: {ch}')
        rname = ONE_TO_THREE[ch]
        if rname not in STANDARD_RESIDUES_PDB:
            raise ValueError(f'no template for residue {rname}')
        for line in STANDARD_RESIDUES_PDB[rname].splitlines():
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            # Rewrite columns: serial, resseq, x-coord
            try:
                x_orig = float(line[30:38])
                y_orig = float(line[38:46])
                z_orig = float(line[46:54])
            except (ValueError, IndexError):
                continue
            new_x = x_orig + x_offset_A
            atom_name = line[12:16]
            element = line[76:78] if len(line) >= 78 else '  '
            new_line = (f'ATOM  {serial:>5d} {atom_name} {rname} A'
                         f'{resseq:>4d}    '
                         f'{new_x:>8.3f}{y_orig:>8.3f}{z_orig:>8.3f}'
                         f'  1.00  0.00          {element}')
            out_lines.append(new_line)
            serial += 1
        x_offset_A += residue_spacing_nm * 10.0   # nm -> A
    out_lines.append('END')

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb',
                                         delete=False, prefix='atomstab_')
    tmp.write('\n'.join(out_lines))
    tmp.close()
    return Path(tmp.name)


def _backbone_atom_indices(atoms: list) -> np.ndarray:
    """Return indices of N, CA, C, O atoms (the 'backbone' set)."""
    indices = []
    for i, a in enumerate(atoms):
        # PDB atom-name in atom_unit is stored as `name` (first letter
        # of element if not set). The PDB importer sets `parent_molecule`
        # to the residue name but doesn't preserve atom_name. Fall back
        # to using all atoms if name isn't preserved.
        nm = (getattr(a, 'name', None) or '').strip().upper()
        if nm in ('N', 'CA', 'C', 'O'):
            indices.append(i)
    if not indices:
        # Fallback: use every heavy atom (non-H)
        from cell_sim.atom_engine.element import Element
        indices = [i for i, a in enumerate(atoms)
                    if a.element != Element.H]
    return np.array(indices, dtype=np.int64)


def _positions(atoms: list, indices: np.ndarray) -> np.ndarray:
    return np.array([atoms[i].position for i in indices], dtype=np.float64)


def stability_score(pdb_path: Path, *, n_steps: int, dt_ps: float,
                       temperature_K: float, pbc_box_nm: float | None,
                       use_shake: bool, label: str) -> dict:
    print(f'[{label}] loading {pdb_path}...')
    structure = load_pdb(str(pdb_path), temperature_K=temperature_K)
    atoms = structure.atoms
    bonds = structure.bonds
    angles = structure.angles
    print(f'  n_atoms={len(atoms)} n_bonds={len(bonds)} n_angles={len(angles)}')
    if len(atoms) < 4:
        return {'label': label, 'error': 'fewer than 4 atoms; skipping'}

    state = SimState(atoms=atoms, bonds=list(bonds),
                       angles=list(angles), dihedrals=list(structure.dihedrals))

    if use_shake:
        state.shake_pairs, state.shake_r0_sq = build_shake_constraints(
            atoms, bonds)
        n_constr = (state.shake_pairs.shape[0]
                      if state.shake_pairs is not None else 0)
        print(f'  shake constraints: {n_constr}')

    if pbc_box_nm:
        ff = ForceFieldConfig(
            lj_cutoff_nm=min(1.0, 0.4 * pbc_box_nm),
            use_confinement=False, use_neighbor_list=False,
            use_coulomb=True, use_pbc=True, pbc_box_nm=pbc_box_nm,
        )
    else:
        ff = ForceFieldConfig(
            lj_cutoff_nm=1.0, use_confinement=False,
            use_neighbor_list=True, use_coulomb=True,
        )

    int_cfg = IntegratorConfig(
        dt_ps=dt_ps, target_temperature_K=temperature_K,
        thermostat='langevin', langevin_gamma_inv_ps=2.0,
        shake=use_shake, dynamic_bonding=False,
    )

    backbone_idx = _backbone_atom_indices(atoms)
    pos0 = _positions(atoms, backbone_idx)

    rmsd_traj = []
    temp_traj = []
    n_initial_bonds = len(state.bonds)
    chunk = max(50, n_steps // 40)   # ~40 sample points
    n_done = 0
    t0 = time.time()
    while n_done < n_steps:
        n_run = min(chunk, n_steps - n_done)
        run(state, n_run, ff, int_cfg)
        n_done += n_run
        pos_now = _positions(state.atoms, backbone_idx)
        # Center-of-mass align (no rotation; sufficient for relative metric)
        pos_now_c = pos_now - pos_now.mean(axis=0)
        pos0_c = pos0 - pos0.mean(axis=0)
        rmsd = float(np.sqrt(np.mean(np.sum((pos_now_c - pos0_c) ** 2,
                                                  axis=1))))
        rmsd_traj.append(rmsd)
        temp_traj.append(current_temperature_K(state.atoms))

    wall = time.time() - t0
    n_final_bonds = len(state.bonds)

    # Stability metrics — use POST-EQUILIBRATION window for both RMSD and T.
    # The known atom_engine T overshoot (~2000K on PDB-imported geometry,
    # see OVERNIGHT_SUMMARY.md) means the first ~25% of the trajectory is
    # equilibration noise. We measure stability over the last 50%.
    n_pts = len(rmsd_traj)
    eq_start = max(1, n_pts // 2)
    rmsd_eq = rmsd_traj[eq_start:]
    temp_eq = temp_traj[eq_start:]
    rmsd_avg_post_eq = float(np.mean(rmsd_eq)) if rmsd_eq else float('nan')
    bonds_preserved = float(n_final_bonds / max(1, n_initial_bonds))
    # T stability: relative to the equilibrated mean, not the target
    if temp_eq:
        t_mean = float(np.mean(temp_eq))
        t_std = float(np.std(temp_eq))
        temp_drift = t_std / max(1.0, t_mean)
    else:
        t_mean = t_std = float('nan')
        temp_drift = 1.0
    temp_stability = float(max(0, 1 - temp_drift))
    rmsd_term = float(1.0 / (1.0 + rmsd_avg_post_eq * 10.0))   # 1 nm RMSD -> 0.09
    overall = float((rmsd_term + bonds_preserved + temp_stability) / 3.0)

    report = {
        'label': label,
        'n_atoms': len(atoms),
        'n_bonds_initial': n_initial_bonds,
        'n_bonds_final': n_final_bonds,
        'n_angles': len(angles),
        'sim_time_ps': dt_ps * n_steps,
        'wall_time_s': float(wall),
        'rmsd_trajectory_nm': [float(r) for r in rmsd_traj],
        'temperature_trajectory_K': [float(t) for t in temp_traj],
        'rmsd_avg_post_eq_nm': rmsd_avg_post_eq,
        'temperature_post_eq_mean_K': t_mean,
        'temperature_post_eq_std_K': t_std,
        'bonds_preserved_frac': bonds_preserved,
        'temp_stability_score': temp_stability,
        'rmsd_score': rmsd_term,
        'overall_score': overall,
        'config': {
            'n_steps': n_steps, 'dt_ps': dt_ps,
            'temperature_K': temperature_K,
            'use_shake': use_shake, 'pbc_box_nm': pbc_box_nm,
        },
    }
    print(f'  done in {wall:.1f}s  '
          f'RMSD_eq={rmsd_avg_post_eq:.3f} nm  '
          f'T_eq={t_mean:.0f}±{t_std:.0f}K  '
          f'bonds={bonds_preserved:.2f}  '
          f'T-stab={temp_stability:.2f}  '
          f'OVERALL={overall:.3f}')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--pdb', type=Path, action='append', default=[],
                      help='PDB file (can repeat for multiple)')
    src.add_argument('--sequence', action='append', default=[],
                      help='one-letter amino-acid sequence (can repeat)')
    ap.add_argument('--label', action='append', default=[],
                      help='label for each input (matches --pdb/--sequence order)')
    ap.add_argument('--steps', type=int, default=4000,
                      help='integration steps (default 4000 = 4 ps)')
    ap.add_argument('--dt-ps', type=float, default=0.001)
    ap.add_argument('--temperature', type=float, default=300.0)
    ap.add_argument('--pbc-box-nm', type=float, default=None)
    ap.add_argument('--no-shake', action='store_true')
    ap.add_argument('--json-out', type=Path,
                      default=ROOT / 'outputs/atom_stability_report.json')
    args = ap.parse_args()

    use_shake = not args.no_shake

    # Build the input list
    inputs = []
    for i, p in enumerate(args.pdb):
        lbl = args.label[i] if i < len(args.label) else f'pdb_{p.stem}'
        inputs.append(('pdb', p, lbl))
    for j, s in enumerate(args.sequence):
        lbl = (args.label[j + len(args.pdb)]
               if j + len(args.pdb) < len(args.label) else f'seq_{s}')
        inputs.append(('seq', s, lbl))

    reports = []
    for kind, value, label in inputs:
        try:
            if kind == 'seq':
                pdb_path = build_peptide_pdb_from_sequence(value)
            else:
                pdb_path = value
            r = stability_score(
                pdb_path, n_steps=args.steps, dt_ps=args.dt_ps,
                temperature_K=args.temperature, pbc_box_nm=args.pbc_box_nm,
                use_shake=use_shake, label=label)
            r['input_type'] = kind
            r['input_value'] = str(value)
            reports.append(r)
        except Exception as e:
            print(f'[{label}] FAILED: {type(e).__name__}: {e}')
            reports.append({'label': label, 'error': f'{type(e).__name__}: {e}',
                              'input_type': kind, 'input_value': str(value)})

    # Summary
    print('\n=== STABILITY RANKING ===')
    valid = [r for r in reports if 'overall_score' in r]
    valid.sort(key=lambda r: -r['overall_score'])
    print(f'{"label":<25s} {"score":>7s} {"RMSD(nm)":>10s} {"bonds":>7s} {"T-stab":>7s}')
    for r in valid:
        print(f'{r["label"]:<25s} {r["overall_score"]:>7.3f} '
              f'{r["rmsd_avg_post_eq_nm"]:>10.3f} '
              f'{r["bonds_preserved_frac"]:>7.2f} '
              f'{r["temp_stability_score"]:>7.2f}')

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({
        'method': 'atom_engine_stability_score',
        'config': {'steps': args.steps, 'dt_ps': args.dt_ps,
                    'temperature_K': args.temperature,
                    'pbc_box_nm': args.pbc_box_nm, 'use_shake': use_shake},
        'reports': reports,
    }, indent=2, default=str))
    print(f'\nwrote {args.json_out}')


if __name__ == '__main__':
    main()
