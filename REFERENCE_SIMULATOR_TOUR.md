# Luthey-Schulten Minimal Cell — what it does, how it gets accuracy

A tour of the upstream simulator that produced the 50 parquet trajectories the
LGNN emulator (`colab_cell_emulator.py`) is learning from.

**Primary source**: Thornburg et al., 2026, *Cell* 189, 2582–2597,
["Bringing the genetically minimal cell to life on a computer in 4D"](https://doi.org/10.1016/j.cell.2026.02.009)
([PDF on Drive](https://drive.google.com/file/d/1eYItG18BFsJWgH8kX4guC8tiNb3r-P3R/view)).

**Code**:
- [`Luthey-Schulten-Lab/Minimal_Cell_4DWCM`](https://github.com/Luthey-Schulten-Lab/Minimal_Cell_4DWCM) — the 4D spatial model (this paper)
- [`Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation`](https://github.com/Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation) — the predecessor well-mixed model (Zhou et al., *JPCB* 2025; Thornburg et al., *Cell* 2022)

This tour was rewritten after reading the Thornburg 2026 paper directly; the
earlier version cited the well-mixed 2022 model, but the parquets we use are
actually from the 2026 4D model. Numbers below come from the paper.

## What the 4DWCM is

A whole-cell simulator of JCVI-Syn3A over its complete cell cycle (~105 min),
in 4D (3 spatial + time). It hybridizes **five** computational methods:

| Subsystem | Method | Time-step |
|---|---|---|
| Spatially-localized gene expression (RNAP↔gene binding, mRNA degradation, translation, ribosome diffusion) | RDME (reaction-diffusion master equation) on 10 nm lattice | 50 μs |
| Whole-cell stochastic reactions (transcription elongation, tRNA charging) | CME (chemical master equation) — global Gillespie | hooked every 1 s |
| Metabolism (glycolysis, nucleotide/lipid/cofactor synthesis, transporters) | ODE — SciPy LSODA | hooked every 1 s |
| Chromosome dynamics (SMC loop extrusion, topoisomerases, replication) | Brownian dynamics — LAMMPS on a 2nd GPU | hooked every 4 s |
| Cell morphology (growth, division) | Geometric overlapping-spheres model from membrane SA/V | hooked every 4 s |

The RDME parent solver (Lattice Microbes) drives the time loop; everything
else is invoked via the `hookSimulation` callback at 12.5 ms intervals.

**Compute cost**: 4–6 days per cell cycle per replicate; ~250 GPU hours per
replicate. The 50 parquets in your Drive represent **~15,000 GPU hours** of
compute.

## What each component does

### Genome / proteome ingest
- `syn3A.gb` (NCBI GenBank CP016816.2) — 493 genes, 455 proteins
- `initial_concentration.xlsx`:
  - "Comparative Proteomics" → 455 protein counts (mass-spec from real Syn3A)
  - "Intracellular Metabolites" → 140 metabolite mM concentrations (scaled from E. coli)
  - "Simulation Medium" → 52 medium-component mM (defined-medium formulation)
  - "mRNA Count" → 455 mRNA initial counts (Poisson-sampled per replicate, mean = 2× the 2022 model's averages)
- `Syn3A_updated.xml` — SBML L3 + FBC: 308 species, 356 reactions, gene-protein-reaction associations
- `complex_formation.xlsx` — 24 multi-subunit complex assembly stoichiometries

### Metabolism (356 reactions)
Random-order bi-bi rate law per reaction (`rxns_ODE.py`):

$$v = E \cdot \frac{k_{cat}^{fwd}\prod_i S_i/K_{m,Si} - k_{cat}^{rev}\prod_j P_j/K_{m,Pj}}{\prod_i(1 + S_i/K_{m,Si}) + \prod_j(1 + P_j/K_{m,Pj}) - 1}$$

All k_cat / K_m from `kinetic_params.xlsx` (1,138 rows across Central / Nucleotide / Lipid / Cofactor / Transport sheets) — measured values from Breuer 2019 + Thornburg 2022. **160 reactions have complete bi-bi data**; the remaining 196 are stoichiometric-only.

### Gene expression
- **Transcription**: per-gene initiation rate scaled by **promoter strength** `S_g = Init_Ptn_Cnt_g / 180` (180 = average across 455 mRNA-coding genes). Elongation 20 nt/s (avg) scaled by `S_g`, capped at 85 nt/s. **No polycistronic operons** — each gene independent.
- **Translation**: 12 aa/s, Hofmeyr rate law depending on charged-tRNA pool. **No polysomes** — one ribosome per mRNA at a time. Authors note this is the main cause of slight protein under-production.
- **mRNA degradation**: degradosome (RNase Y + J1) binds mRNA at periphery, degrades at 88 nt/s. mRNA half-life median ~1.97 min, range <1 to 20 min.
- **Ribosome biogenesis**: 30S has 19-intermediate linear assembly chain (reduced from 145-step hierarchical map). 50S similar, with "strong" (10⁶ M⁻¹s⁻¹) and "weak" (10⁴ M⁻¹s⁻¹) binders.

### DNA replication
- DnaA forms a filament at oriC by sequential binding to one high-affinity (9/9 consensus) and two low-affinity (7/9) dsDNA sites, then polymerizes on ssDNA (140 mM⁻¹s⁻¹ on, 0.42 s⁻¹ off).
- Replisome (DnaB helicase) loads once filament ≥ 20 DnaA.
- Replication elongation: 100 bp/s (Syn3A; from M. capricolum measurement, slower than E. coli's 600 bp/s).
- Train-track model: leading + lagging strands replicate simultaneously.

### Chromosome dynamics
- Polymer model in LAMMPS (10 bp per bead, σ=3.4 nm, persistence length 45 nm).
- SMC loops: one-sided extrusion, ~200 bp per step every 0.4 s, 4 s dwell time.
- Topoisomerase: soft-then-hard potential switching to allow strand crossing.
- **Daughter chromosome partitioning**: a non-physical 12 pN repulsive force is added to drive segregation (explicitly noted as a kludge — Syn3A lacks ParABS, MinDE, MukBEF, polysome-based mechanisms).

### Cell growth + division
- Surface area grows from `Σ membrane_components × per_component_SA_contribution`.
- Volume held constant during division; cells grow as overlapping spheres.
- 200 nm radius initial → 250 nm radius at end of growth (≈98% volume increase).
- **Initial cytoplasmic volume = 3.35 × 10⁻¹⁷ L** (sphere of 200 nm radius).

### Stochasticity
- Exact Gillespie SSA in the RDME — not tau-leaping.
- Each of the 50 replicate cells starts from Poisson-sampled initial mRNA counts; everything diverges from there.

## Where the accuracy comes from

Three sources, ranked by impact:

**1. Measured parameters.** k_cat, K_m, initial counts, medium composition — every number comes from Breuer 2019, Thornburg 2022, BRENDA, or mass-spec on actual Syn3A. The model is *parameterised*, not *fit*.

**2. Mechanism-aware coupling.** NTP pools appear directly in the transcription rate law; charged-tRNA pools in translation; dNTPs in replication. When ATP crashes, gene expression slows down *by chemistry*, not by a learned correlation.

**3. Exact SSA on real parameters.** Stochastic timing of individual reaction events is correct (per Gillespie); fluctuations match the true CTMC.

**Mass conservation across CME/ODE boundary** is enforced by `payAfterODE`: when CME events consume metabolites the ODE hasn't yet produced, the CME logs a deficit, and the next ODE step reconciles it from updated pools. So counts never go genuinely negative.

## Validation — paper vs experiment

The paper validates the simulator against real Syn3A measurements:

| Quantity | Simulated | Experimental |
|---|---|---|
| Doubling time | 105 min | 105 min ✓ |
| ori:ter ratio | 1.28 | 1.21 (DNA sequencing) ✓ |
| Ribosome count at division | 881 | matches literature (~800) ✓ |
| mRNA half-life median | ~2 min | matches B. subtilis distribution ✓ |
| Cell morphology fractions | ~80% spherical, 12% prolate, 5% dividing | matches fluorescence imaging ✓ |
| Cell radius range | 200–250 nm | matches cryo-ET |

The validation is **population-level** — they don't compare individual simulated trajectories to anything because each cell is unique. This matters for our emulator: trajectory-level R² isn't what the paper claims.

## Acknowledged limitations (the authors' own caveats)

Direct quotes / summaries from the paper:

1. *"We are likely missing some balancing effects between NTP and dNTP pools in the metabolic rates"* — uptake rates had to be hand-tuned to compensate.
2. Most proteins reach only **1.25–1.5× initial counts** at division instead of doubling — attributed to no polysomes.
3. No polycistronic transcription. No coupled transcription-translation. No FtsZ kinetic model.
4. Chromosome partitioning uses an artificial 12 pN force (no biological mechanism known in Syn3A).
5. Some assembly rate constants (LSU) are estimated from order-of-magnitude analogy to SSU.

These matter for our emulator: the simulator itself isn't trustworthy on every transient. "Sense-1 better than the simulator" is only credible where we have an INDEPENDENT physical law (mass conservation, ΔG° sign, count ≥ 0) — not where we just don't trust the literature parameters.

## What our emulator captures vs misses

| Reference subsystem | Our emulator (v14.7) |
|---|---|
| SBML stoichiometry | ✓ MetabolismCore (115 reactions wired) + PINN head |
| Bi-bi metabolism kinetics | ✓ MetabolismCore (115 reactions with measured k_cat/K_m) |
| ΔG° sign constraint | ✓ MetabolismCore clamps reverse k_cat to 0 for ΔG° < −10 kJ/mol |
| Dynamic cell volume | ✓ VolumeCore (lipid-sum proxy) |
| Central dogma (per-gene tx/tl/decay) | ✓ CentralDogmaCore (455 genes wired, per-gene rate calibration from initial counts) |
| Shared ribosome pool | ✓ CD scales translation by ribo_total / ribo_total(t=0) |
| ATP balance | ✓ Soft penalty when net ATP rate < calibrated maintenance floor |
| Complex assembly | partial — AssemblyCore exists but only 2/24 wired on this data; currently disabled |
| 50S biogenesis chain | partial — code in AssemblyCore, currently disabled |
| Transport reactions | ✓ (subsumed in MetabolismCore) |
| tRNA charging | ✓ (subsumed in MetabolismCore where data exists) |
| DNA replication kinetics | ✗ — no ReplicationCore; LGNN handles gene replication implicitly |
| Chromosome dynamics (SMC, topoisomerases) | ✗ |
| 3D spatial diffusion | ✗ — single forward pass, no spatial component |
| Per-cell stochasticity | partial — stochastic head outputs σ; rollout currently deterministic |

## Where the emulator could meaningfully exceed the simulator

Three legitimate paths:

1. **Refuse violations of universal laws.** Mass conservation and ΔG° sign hold regardless of whether the simulator's k_cat values are right. We can claim "where the simulator violates these by numerical accident, our constrained model produces the physically admissible trajectory instead." Limited but defensible (see `test_conservation_violations.py`).

2. **Massive stochastic sampling for tail statistics.** The simulator can't afford to run 10,000 replicates; we can in seconds. Useful for rare-event probabilities — bounded by our Gaussian σ approximation of the true Poisson noise.

3. **Data fusion.** Fine-tune our model against experimental observations (Breuer 2019 essentiality, doubling-time distributions, fluorescence imaging morphologies) on top of the simulator-derived base. The moment we train on real data the simulator doesn't capture, we exceed it on those quantities. This is the most defensible "better than simulator" claim and is the natural sequel to the calibration work in v14.3–v14.7.
