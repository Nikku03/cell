# atom_engine — atomic-scale physics simulator

A self-contained classical molecular dynamics engine that lives alongside
the kinetics-based cell-sim layer stack. **It is not a competing
essentiality predictor.** It operates 9–12 orders of magnitude below
the timescale at which Syn3A gene essentiality manifests, and direct
prediction of Syn3A essentiality from atomic dynamics is infeasible.

## What it is

| | atom_engine | cell-sim layer stack (v15) |
|---|---|---|
| Scale | atoms in a reactive soup or PDB structure | metabolites/proteins as discrete species |
| Step | 1 femtosecond (1e-15 s) | sub-microsecond Gillespie events |
| Trajectory length | ~picoseconds (10 ps stable) | ~seconds biological time |
| Dynamics | velocity-Verlet + SHAKE + LJ + Coulomb | stochastic mass-action kinetics |
| Question answered | "does this chemistry happen?" | "does this gene KO collapse metabolism?" |

## Capability summary (per OVERNIGHT_SUMMARY.md, all 9 phases done)

- SHAKE bond constraints (5× longer stable runs at dt=1 fs)
- Vectorised angle + dihedral kernels
- PBC-aware neighbor list with minimum-image
- Rust LJ + Coulomb kernel (`cell_sim_rust::lj_forces`)
- 11 standard amino acid templates + water
- PDB importer with auto bonds/angles/charges
- Reactive bonding stack (chemistry + fission/fusion demos)

## Headline results

**On its own task** (chemistry-pair essentiality in H/C/N/O/P/S soup):
```
PerRule detector    MCC = 1.000   (TP=6 FP=0 TN=11 FN=0)  perfect
```
17 candidate pair-genes, full panel, 8 ps simulation.
See `outputs/atom_engine_full_panel_results.json`.

**On Syn3A gene essentiality via metabolite-bond bridge:**
```
chemistry-pair vote MCC = 0.190   (TP=107 FP=4 TN=68 FN=276)  weak
```
Confirms the architectural separation: atom-scale physics has
insufficient discriminative power for cell-scale gene essentiality.
See `outputs/atom_engine_on_syn3a_results.json`.

## What it's actually for

These are tasks where atom-scale physics has unique signal that the
kinetic simulator cannot give:

| Task | Why atom_engine fits |
|---|---|
| Drug-target binding affinity | Direct atomic interactions at ps–ns scale |
| Protein structural stability under mutation | Bond/angle force field handles point mutations |
| Membrane permeability for novel metabolites | LJ + Coulomb in PBC = direct bilayer crossing |
| Enzyme transition-state characterization | Reactive force field models bond rearrangement |
| Cofactor binding pocket validation | Sub-nanometer geometry is the natural domain |
| Predicting novel chemistries a cell can perform | Reactive bonding in arbitrary soup |

## How it relates to the cell-sim layer stack

**One-way validator, not a feature provider.** The intended workflow:

1. v15 simulator + cross-org cascade predict gene essentiality (MCC ≈ 0.53)
2. For predictions you want to validate at higher physical resolution,
   `scripts/atom_validator.py` looks up the gene's catalysed chemistry
   and checks whether the corresponding bond rearrangements are
   plausible in atom_engine's reactive soup.
3. The validator returns a "chemistry plausibility" score per gene —
   not a competing essentiality call.

The single integration point is `essentiality_bridge.py`, which adapts
atom_engine output to the Layer-6 detector framework so the same
detectors (`PerRuleDetector`, `ShortWindowDetector`, etc.) can be
reused. This bridge is **valid for atom_engine's own task** (chemistry
essentiality, MCC=1.000) and **not valid for Syn3A gene essentiality**
(MCC=0.190 via the metabolite-bond bridge).

## Entry points

```bash
# Run the full atom_engine essentiality sweep (17 chemistry pairs):
python scripts/run_atom_essentiality.py --steps 8000 --temperature 3000

# Validate v15 essentiality predictions chemistry-plausibility:
python scripts/atom_validator.py --gene JCVISYN3A_0445
python scripts/atom_validator.py --predictions outputs/<v15_csv> --top 50

# Other demos (water, peptide, vesicle, kitchen-sink):
python scripts/run_water_peptide.py --demo water --pbc --shake
python scripts/run_kitchen_sink.py
python scripts/run_fusion_demo.py
```

## Honest limits

- **5 ps stability** at liquid density (long-range Coulomb / Ewald not
  implemented; thermal drift past that)
- **9 amino acids missing** from the built-in residue set (PHE, TYR,
  TRP, HIS, LYS, ARG, GLU, GLN, PRO) — paste real PDB blocks for these
- **No dihedral parameters on imported residues** — backbone φ/ψ
  generation is on the residue templates, not the importer
- **Initial T overshoot** (~2000 K not 300 K on PDB-imported geometry)
  — production MD does steepest-descent minimisation first; we don't
- **Cannot meaningfully predict gene-level essentiality** — see
  `atom_engine_on_syn3a_results.json` for the empirical confirmation
