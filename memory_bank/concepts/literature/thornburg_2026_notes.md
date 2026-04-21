# Thornburg 2026 (Cell) — deep analysis

_Session 11 deep analysis assembled by a research subagent running ~10 targeted WebSearches. Direct fulltext access to Cell.com, bioRxiv, and news sites was blocked by the sandbox proxy; GitHub and PMC were reachable. Where a specific number came from an unambiguous verbatim quote in a news article, confidence is high; where it came from paraphrased summary, confidence is hedged in the relevant fact file._

## 1. Paper provenance

- **Title:** Bringing the genetically minimal cell to life on a computer in 4D
- **Journal:** Cell, 2026, DOI `10.1016/j.cell.2026.02.009`, article PII S0092-8674(26)00174-1.
- **Preprint:** bioRxiv `10.1101/2025.06.10.658899v1` (deposit June 2025).
- **Authors** (17): Thornburg ZR, Maytin A, Kwon J, Brier TA, Gilbert BR, Fu E, Gao Y-L, Quenneville J, Wu T, Li H, Long T, Pezeshkian W, Sun L, Glass JI, Mehta AP, Ha T, Luthey-Schulten Z.
- **Code:** https://github.com/Luthey-Schulten-Lab/Minimal_Cell_4DWCM — publicly reachable, main entry point `Whole_Cell_Minimal_Cell.py`. Registered as `sources/luthey_schulten_minimal_cell_4dwcm_repo.json`.
- **Data mirror:** Zenodo `10.5281/zenodo.15579158` — particle counts for all 50 simulated cells + 4 RDME trajectory files.
- **License:** Not yet inspected; default to "research-share; do not redistribute".

## 2. Scientific claims (the five things they found)

1. **First full-cell-cycle 4D whole-cell simulation.** The 4DWCM runs the entire ~105-minute cycle end-to-end across all 493 genes, the full metabolic network, ribosome biogenesis, chromosome replication + segregation, and morphological growth + division.
2. **Simulated doubling time matches experiment to ±2 min.** Across 50 replicates, mean cycle length ≈ 105 min. Growth is limited by lipid + membrane-protein synthesis driving surface-area expansion.
3. **DNA replication and symmetric division are tightly coordinated.** The model reproduces the experimental origin:quarter:terminus DNA-copy-number ratios measured by qPCR, and recapitulates symmetric division (prolate → dumbbell → fission morphology matches JCVI-Syn3B imaging).
4. **Quantitative RNAP utilisation prediction.** Spatial simulation predicts that ~34 % of the 187 RNA polymerases (i.e. 63 RNAPs) are actively transcribing on average early in the cell cycle; the rest are transiently binding / unbinding genes. This is an emergent number the 0D 2022 model couldn't produce.
5. **Emergent mRNA dynamics.** Median simulated mRNA half-life across 452 protein-coding genes is **3.63 min** — a substantial revision of the ~2 min (M. gallisepticum-extrapolated) value used in Thornburg 2022. Per-gene distribution available on Zenodo.

## 3. Methodological novelty vs Thornburg 2022

Thornburg 2022 was a **well-stirred CME–ODE** 0D model covering only the first ~20 min. The 2026 paper adds:

- **Hybrid 4-method architecture:** RDME (spatial stochastic gene expression + diffusion) + CME (fast reactions) + ODE (metabolism) + Brownian Dynamics (chromosome) running together via "hooks" in Lattice Microbes.
- **Spatial discretisation:** RDME on a 3D lattice inside a cell-shaped boundary. Exact voxel size not reported; Lattice Microbes RDME typically uses 8–32 nm subvolumes.
- **Chromosome polymer model:** worm-like-chain at 10 bp / bead → ~54,300 beads for 543 kbp. Excluded-volume interactions, SMC loop extrusion, topoisomerase action alternating loop updates + energy minimisation + BD steps. Simulated in LAMMPS + Kokkos GPU (btree_chromo_gpu). Inherited from Gilbert 2023.
- **Membrane model:** Dynamically Triangulated Surface via FreeDTS (Pezeshkian 2024); membrane shape updated each cycle step from lipid-flux and membrane-protein-insertion rates computed by the ODE metabolic module.
- **Protein diffusion:** RDME voxel-to-voxel hops with species-specific D; complexes form via spatially localized reactions, not well-mixed mass action.
- **Growth + division:** surface area from synthesised lipids + inserted membrane proteins; morphology transitions match Syn3B imaging.
- **DnaA kinetics refined:** on-rate 1.0–1.4 × 10⁵ M⁻¹ s⁻¹, off-rate 0.55–0.42 s⁻¹, tuned to smFRET of DnaA-box / ssDNA filaments.

## 4. Compute architecture

- **GPUs:** 2 × NVIDIA A100 per replicate — one dedicated to chromosome BD (LAMMPS + Kokkos), the other for Lattice Microbes RDME + CME + ODE hooks.
- **Infrastructure:** NCSA Delta supercomputer, 200 GPU nodes used for the ensemble (each node 64 CPU cores / 256 GB RAM / 200 Gbps fabric).
- **Wall time:** 4–6 days per cell cycle.
- **Data per run:** ~420 GB.
- **Replicates:** 50 simulated cells for the main results.
- **CPU-only fallback:** None.

**Cost to cross-validate one knockout against their simulator:** ~10–60 A100-days per gene. Out of scope for this project.

## 5. Validation approach

Experimental targets, with simulated-vs-measured comparisons where snippets exposed them:

| Target | Experimental value | Simulated value | Method |
|---|---|---|---|
| Doubling time | 105 min | 105 ± 2 min over 50 cells | Fluorescence / bulk growth |
| Ori:quarter:terminus ratio | qPCR of Syn3A | Matches qPCR | [Mendeley `nprw2h5tx6`](https://data.mendeley.com/datasets/nprw2h5tx6/1) |
| mRNA half-life | ~2 min (extrapolated) | **3.63 min emergent** | pending genome-wide Syn3A measurement |
| Ribosome count | ~500 at birth (cryo-ET) | 500 initial | Bianchi 2022 + cryo-ET |
| RNAP count | 187 per cell | 34 % active early cycle | Bianchi 2022 proteomics |
| Per-gene protein abundance | Bianchi 2022 (~452 genes) | Distribution recovered | Bianchi 2022 |
| Morphology | Prolate → dumbbell (Syn3B imaging) | Matches | fluorescence microscopy |

**Not validated:** single-cell ATP flux, genome-wide experimental mRNA half-life map, protein turnover rates, any knockout phenotypes.

## 6. Gene-level findings

- All **452 protein-coding + 41 RNA = 493 genes** are active in the simulation (our parse gives 458 CDS + 38 RNA = 496 — see `syn3a_gene_count_thornburg2026_discrepancy`).
- Per-gene mRNA half-lives available on Zenodo — pending download.
- Per-gene protein copy-number trajectories across the cycle available.
- **No MCC-style gene-deletion / essentiality analysis** is reported. This confirms our Layer 6 MCC work is novel territory.
- DnaA is highlighted as a gene whose kinetics had to be re-tuned from smFRET; SMC + topoisomerase gene products drive chromosome organisation.

## 7. Stated limitations (from the Discussion)

- Not atomistic — molecules are coarse-grained.
- Growth medium concentrations held fixed (no extracellular depletion).
- FreeDTS backmapping covers ~55 % of the metabolome (CG-Martini topologies available).
- mRNA half-life calibration is indirect; pending Syn3A-specific genome-wide measurement.
- Only 50 replicates → limited statistical power for rare events.
- 6-day / 2-A100 compute cost precludes routine perturbation sweeps.

## 8. Integration with prior Luthey-Schulten lab work (registered as separate sources)

| Paper | Role | Source ID |
|---|---|---|
| Thornburg 2022, Cell 185:345 | CME–ODE engine + genetic-info parameters | `thornburg_2022_cell` |
| Breuer 2019, eLife 8:e36842 | iMB155 metabolism (338 rxn / 304 met / 155 genes) | `breuer_2019_elife` |
| Hutchison 2016, Science | Syn3A design + transposon essentiality ground truth | `hutchison_2016_science` |
| Fu 2026, JPC-B 130(1):11–32 | Complex assembly kinetics (21 complexes) | `fu_2026_jpcb_complex_assembly` |
| Gilbert 2023, Front Cell Dev Biol 11:1214962 | Chromosome 10-bp/bead WLC + SMC + topo model | `gilbert_2023_frontiers_chromosome` |
| Bianchi 2022, JPC-B | Per-gene proteome reference | `bianchi_2022_jpcb_proteomics` |
| Pezeshkian 2024, Nat Commun | FreeDTS membrane simulator | `pezeshkian_2024_natcomm_freedts` |

Other papers worth acquiring in later sessions: Stevens 2023 Front Chem (CG-MD Martini cell), Pelletier 2021 Cell (division genetics), Bock 2024 Nat Commun (Syn3A lipidome minimality), Rees-Garbutt 2020 Nat Commun (genome design via whole-cell models).

## 9. Implications for our CPU-only essentiality predictor

1. **Per-gene data now available on Zenodo** — mRNA half-life trajectories + protein copy-number trajectories for 452 genes across 50 cells. Could sharpen Layer 1/3 validation targets once ingested. Future-session task.
2. **No knockout benchmark in Thornburg 2026** — our Layer 6 MCC work remains the only effort pointed directly at essentiality prediction. Not duplicated by this paper.
3. **4DWCM repo input_data is reachable** from the sandbox via GitHub raw. `kinetic_params.xlsx` is 85 KB vs our staged 59 KB — refined parameters we could ingest. Opportunity flagged in `NEXT_SESSION.md` for a later session.
4. **Cross-validating our predictions against Thornburg 2026 is infeasible** (~10–60 A100-days per knockout). Our approach is to approximate their model using their published rate tables, not to invoke their simulator.
5. **High-confidence numbers extracted into memory-bank facts** (all with Thornburg-2026-inferred caveats):
   - `syn3a_rnap_count_per_cell` = 187
   - `syn3a_active_rnap_fraction_early_cycle` = 0.34
   - `syn3a_ribosome_count_at_birth` = 500
   - `syn3a_mrna_halflife_mean_4dwcm` = 218 s (3.63 min)
   - `syn3a_chromosome_bead_model` = 10 bp / bead, ~54,300 beads
   - `syn3a_doubling_time` updated 7,200 s → 6,300 s (105 min)
6. **Gene-count discrepancy resolved** into `syn3a_gene_count_thornburg2026_discrepancy`: they filter ~6 CDS that we keep (likely pseudogenes that Breuer also excludes) and count 3 more RNA features than we do (classification difference on ncRNA / misc_RNA / CP016816.1 vs .2). Our Layer 6 panel uses Breuer-labelled genes which are a subset of both sets, so the discrepancy does not affect our MCC work.

## 10. Open questions + what the user could paste

Full fulltext still blocked. If the user can paste the bioRxiv preprint PDF text (or specific tables) into chat, I can promote the remaining "not reported in accessible sources" items into proper facts:

- Exact voxel resolution (nm).
- Per-gene mRNA half-life distribution shape (variance, outliers).
- Per-complex assembly rates (from Fu 2026's 21 complexes).
- License file content of the 4DWCM repo.
- Per-kernel wall-time split (spatial vs reaction vs BD).
- Replicate-to-replicate variance on each validation metric.
- Any explicit per-gene "sensitivity" analysis (sensitivity is not essentiality, but a useful secondary signal).
