# Overnight multi-session summary

Plan: `MULTI_SESSION_PLAN.md`. Nine phases, one running branch. When
you wake up this is what landed.

## What's in now

| Phase | Title | Commit | Status |
|-------|-------|--------|--------|
| 1 | SHAKE bond constraints | `c47658c` | ✅ done — 5× longer stable runs |
| 2 | Vectorised angle + dihedral | `f638aab` | ✅ done |
| 3 | PBC-aware neighbor list | `a9c3746` | ✅ done |
| 4 | Rust LJ + PBC + Coulomb | `d5b4892` | ✅ done |
| 5 | Long stability benchmark | `ea0a7b9` | ✅ done — honest 4 ps |
| 6 | PDB importer (supersedes 20-AA) | `f490570` | ✅ done — auto bonds/angles/charges |
| 7 | 11 standard residues | `7af9e8b` | ✅ done — 11/20 AAs + water |
| 8 | Importer (merged into 6) | — | ✅ merged |
| 9 | Kitchen-sink demo | `717f6a8` | ✅ done — bio+phys+chem, 5 ps |

All 29 existing tests pass throughout.

## What the engine can now do

- **Stable integration**: SHAKE lets dt run at 1 fs instead of 0.2 fs.
- **Periodic boundaries**: cubic PBC with minimum-image everywhere
  (pair forces, bonded forces, angles, dihedrals, neighbor list).
- **Fast force kernel**: Rust `lj_forces` now handles PBC natively
  inside its tight loop. No NumPy fallback needed for PBC.
- **Biology import**: any standard PDB file (or the 11 built-in
  residue strings) produces a complete force-field-ready structure
  with bonds, angles, and partial charges auto-assigned.
- **Chemistry**: the earlier reactive-bonding stack remains
  available via `chemistry_demo` / `reaction_demo`.

## What's honest about the limits

- **5 ps stability at liquid density.** Beyond that, dense Coulomb +
  Langevin noise without proper long-range electrostatics (Ewald /
  reaction field) leads to thermal drift. This is a physics-model
  upgrade, not a bug.
- **Initial T overshoot** (~2000 K, not 300 K): the PDB-imported
  geometry has residual Coulomb potential energy that gets converted
  to heat on step 0. Production MD does a steepest-descent
  minimisation first; we don't.
- **9 amino acids still missing** from the built-in residue set
  (PHE, TYR, TRP, HIS, LYS, ARG, GLU, GLN, PRO). All because aromatic
  rings or long flexible side chains need better PDB geometry. Paste
  in real PDB coordinate blocks when needed.
- **No dihedral parameters on imported residues**: the importer
  doesn't generate φ/ψ dihedrals yet. Legacy `Gly-Gly` template
  still has them explicitly.

## How to run the demos

```bash
# Liquid-density water under PBC + SHAKE:
python scripts/run_water_peptide.py --demo water --pbc --shake \
    --n-water 20 --pbc-box-nm 1.5 --water-steps 5000 --dt-ps 0.001

# Peptide (Gly-Gly) in water with all the bond terms:
python scripts/run_water_peptide.py --demo glycine --n-water 25

# Kitchen sink: Ala + Ser + water in PBC with every upgrade:
python scripts/run_kitchen_sink.py --n-water 15 --pbc-box-nm 2.0 \
    --steps 5000 --dt-ps 0.001 --out /tmp/ks.json

# Load any residue programmatically:
python -c "
from cell_sim.atom_engine.pdb_importer import load_residue
s = load_residue('ALA')
print(len(s.atoms), 'atoms,', len(s.bonds), 'bonds,', len(s.angles), 'angles')
"
```

## Next workstreams if you want to push further

Ranked by leverage:

1. **Reaction-field electrostatics** (~2-4 hours) — adds long-range
   Coulomb truncation with a smooth shift. The single biggest unlock
   for multi-ps stability at dense liquid conditions. After this,
   the ~5 ps ceiling probably becomes 50+ ps.
2. **Steepest-descent minimiser** (~1 hour) — relax initial PDB
   structures before thermalisation so the first-step T overshoot
   disappears. Small but important.
3. **Residue library completion** (~30 min/residue) — the 9 missing
   AAs are mechanical paste-and-adjust once we have a reference PDB
   source (e.g. RCSB's free-amino-acid ligand records).
4. **Nucleotides + lipids** (~2 hours) — using the same PDB importer,
   any residue with well-tabulated geometry loads immediately.
5. **Per-residue force-field parameters** (AMBER ff14SB / CHARMM36)
   instead of the current charge heuristics. Production biophysics.

All commits push to the branch; nothing destructive or merged to
main without your sign-off.
