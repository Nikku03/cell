# Atom Engine: Multi-Session Plan

Target: an atom engine that can simulate complex chemistry and biology
for multi-picosecond to nanosecond trajectories with periodic boundary
conditions, without blowing up. Everything commits to this branch so
there are no lost checkpoints; each phase is independently useful.

## Phase 0 — this doc (done)
Written, committed.

## Phase 1 — SHAKE bond constraints
**Goal:** eliminate the 10 fs bond vibration that currently forces dt ≤
0.2 fs. With bonds held at their equilibrium lengths via SHAKE (the
standard MD trick: one Newton iteration per constrained bond per step),
dt = 2 fs becomes feasible. That's **10× longer simulations for the
same wall clock**.

Scope:
 - `IntegratorConfig.shake_tolerance` + `shake_max_iter`
 - Build a list of "constrained bonds" from molecule templates
   (everything tagged `COVALENT_SINGLE` / `COVALENT_DOUBLE` on H-containing pairs)
 - Project positions after the velocity-Verlet step so each constrained
   bond's r matches its equilibrium length
 - Remove the same-bond's harmonic force (SHAKE replaces it)
 - Tests: two-atom H-O pair stays at r0 under SHAKE even with dt = 5 fs
 - Rerun water demo at dt = 1 fs, show 10 ps stable trajectory

## Phase 2 — Vectorised angle + dihedral kernels
Current `_compute_angle_forces` / `_compute_dihedral_forces` iterate
each term in Python. For proteins with hundreds of angle terms this is
~10% of step time. Vectorised NumPy rewrite: gather all (i, j, k)
index tuples, compute all bends in one pass. Expected 5-10×.

## Phase 3 — PBC-aware neighbor list
Spatial hash with cell wrap-around. Eliminates the current O(N²)
fallback under PBC. At N = 10 000, saves ~99% of pair work.

## Phase 4 — Rust LJ kernel with PBC + minimum-image + Coulomb
Extend the existing `cell_sim_rust::lj_forces` to accept
`box_l` and do the minimum-image wrap inside. Combined with Phase 3
this makes large PBC systems tractable.

## Phase 5 — Long stability demo
Water (50+) in a cubic box at liquid density for 10 ps. Check: no
runaway heating, HB/water climbs toward the literature value. Glycine
in same box for 10 ps: backbone geometry preserved, no broken bonds.
This is the "can it actually run overnight" validation.

## Phase 6 — 20 amino-acid templates
Add alanine (have), arginine, asparagine, aspartate, cysteine,
glutamate (skip glutamine — similar to asp), glycine (have), histidine,
isoleucine, leucine, lysine, methionine, phenylalanine, proline,
serine (have), threonine, tryptophan, tyrosine, valine. Each:
geometry + partial charges + angles + backbone φ/ψ dihedrals. Mostly
mechanical.

## Phase 7 — Nucleotide templates (ATGCU)
Purines (A, G) and pyrimidines (C, T, U) with ribose + phosphate.
Enough to build short oligomers.

## Phase 8 — Template importer
Either SMILES (via rdkit, if available) or a simple PDB parser so
larger biomolecules can be loaded instead of hand-coded. Punt SMILES
if rdkit not installable; PDB parser is ~150 lines of Python.

## Phase 9 — Kitchen-sink demo
Final demo: a small peptide (GlyGlyGly or short Ala-helix seed)
solvated in PBC water at 300 K for 10 ps. Report bond/angle
preservation, HB count, peptide RMSD. Single JSON + plain-English
writeup.

## Abort conditions
- If any phase takes > 2 hours of interactive work, commit what's
  there and move on; mark the remainder as "phase N.1 TODO".
- If a phase breaks earlier tests, revert that phase's force-field
  changes and keep its template/tooling additions. Never leave the
  main line broken.
- If compute time blows up (any single bench > 30 min), reduce scale
  and record the honest number.

## Success criteria
- All 25+ existing tests still pass at the end.
- New demos: water 10 ps stable; small peptide 10 ps stable; Rust PBC
  path matches NumPy PBC path numerically.
- One "kitchen sink" JSON sample at the end that aggregates everything.
