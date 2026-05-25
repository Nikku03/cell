# Equations to wire into the emulator — build list

Strategy: take the rate laws the upstream simulator uses (Luthey-Schulten Minimal_Cell_ComplexFormation), bake them directly into PyTorch modules, replace the LGNN's neural prediction for those reactions with the actual formula. The LGNN keeps the **residual** job — capture whatever the equations don't model (allostery, growth coupling, edge cases).

Each module is gated on a verifiable improvement before we proceed to the next. If a module doesn't move rollout R² meaningfully, we pause and diagnose before adding the next one.

## Baseline (v13.8.2, stride=30)
- Rollout R² (mean): **0.625**
- Honest R² (top-200 variable): **+0.340**
- KO MCC: +0.029 (random, but 6/12 top-essential are confirmed)
- Element drift: 80–208%
- PINN coverage: 137 / 5,940 species

## Build order

| # | Module | Source data | Replaces | Expected rollout R² | KO MCC | Status |
|---|--------|-------------|----------|---------------------|--------|--------|
| 1 | **MetabolismCore** — bi-bi rate law, 356 SBML reactions | `kinetic_params.xlsx` (k_cat_fwd/rev, K_m per substrate/product), SBML stoichiometry | PINN head's neural `v_log` prediction | 0.70–0.85 | unchanged | **in progress** |
| 2 | **VolumeCore** — dynamic volume per timestep | `SA_i.csv` or analytical `V₀·exp(t·ln2/2h)` | constant-volume assumption | +0.05–0.15 | – | not started |
| 3 | **CentralDogmaCore** — per-gene tx / tl / mRNA-deg / prot-deg | gene lengths, NTP/aa pools, literature k_init + half-lives | LGNN's prediction for ~5,000 per-gene species | +0.10–0.20 | – | not started |
| 4 | **AssemblyCore** — complex assembly + 50S biogenesis (mass-action) | `complex_formation.xlsx`, `LargeSubunit.xlsx` | LGNN's prediction for complex species | +0.05–0.10 | – | not started |
| 5 | **TransportCore** — 58 transport reactions (bi-bi) | `kinetic_params.xlsx` Transport sheet | LGNN's prediction for transport species | +0.02–0.05 | – | not started |
| 6 | **tRNAChargingCore** — 20 aa charging (bi-bi) | `kinetic_params.xlsx` Cofactor sheet | LGNN's prediction for tRNA species | +0.02 | – | not started |
| 7 | **KnockoutAugmentation** — random species-zero per training batch | – | model trained only on unperturbed data | unchanged | **+0.20–0.40** | not started |
| 8 | **ReplicationCore** — DnaA filament + replisome | upstream `replication.py`, would need new data parse | LGNN's prediction for `chromosome` + `ori_rep*` species | +0.05–0.10 | – | not started |
| 9 | **Residual LGNN + Stochastic head** — keep existing, shrink | – | – | – | – | retained |

**Net expected final state:** rollout R² 0.80–0.95, KO MCC 0.40–0.70, element drift near 0%, structural OOD.

## Conventions

- Each module is a `nn.Module` with frozen-by-default parameter buffers (k_cat, K_m, …).
  Optionally a `learnable_rates: bool` flag lets us fine-tune the buffers if needed.
- Forward signature: `(state, volume) → (delta_state, fluxes)`.
- Modules compose additively: `delta_total = Σ delta_module + λ · residual_LGNN(state)`.
- Each module owns a `coverage_mask`: a bool per species indicating "I am responsible for this species' Δ." The LGNN's prediction is overridden for masked species.
- Missing data fallbacks: if a reaction is missing k_cat or K_m, mark it as not-wired and let the LGNN handle it. Track coverage stats and report.

## Gating per module

Before moving to the next module:
1. Module passes a standalone unit test (forward pass produces finite values, correct shapes).
2. Module's predicted fluxes are within ~10× of upstream's actual fluxes (where available).
3. End-to-end run shows rollout R² did not regress; ideally improved.
4. Commit + push, document the result before starting next.

## Open data questions per module

- **MetabolismCore**: which columns in `kinetic_params.xlsx`? Need to map "Parameter Type" values to (k_cat_fwd, k_cat_rev, K_m_substrate, K_m_product). Currently only "Eff Enzyme Count" is read.
- **VolumeCore**: is `SA_i.csv` in our parquet trajectories or a separate output? If absent, use analytical doubling.
- **CentralDogmaCore**: NTP-dependence functional form — assume MM on each NTP, k_init values from Thornburg 2022 paper.
- **AssemblyCore**: are association rate constants k_on actually in `complex_formation.xlsx`, or do they need to be inferred?
- **ReplicationCore**: need to parse upstream's `replication.py` to extract the DnaA binding kinetics + replisome loading constants.

## Tracking

Each module gets a section below as it lands.

### Module 1 — MetabolismCore

- **Status**: foundation landed (commit v13.9 part 1); not yet wired into DynamicsModel
- **Data extraction**: `parse_kinetics` now extracts the full per-reaction table:
  k_cat_fwd, k_cat_rev, K_m per substrate/product, enzyme, GPR rule.
  Coverage in real file: 160 of 160 reactions have complete k_cat + K_m + enzyme.
  Other 196 SBML reactions have no kinetic params in the xlsx (will fall back to LGNN).
- **Module implementation**: `class MetabolismCore(nn.Module)` — frozen-buffer
  bi-bi rate law on the wired reactions, forward signature
  `(state[B,S] counts) → (delta_state[B,S] counts/s, fluxes[B,R] mM/s)`.
  Internal unit conversion via `NA_AVOGADRO` and `SYN3A_VOLUME_L=2e-16` (constant
  for now; VolumeCore will make this dynamic).
- **Standalone test** (`/tmp/test_metabcore.py`): 7/7 checks pass:
  - flux/delta finite
  - isolated forward direction holds for 20/20 reactions tested
  - knockout (enzyme=0) zeros all reactions using that enzyme — by construction
  - flux magnitudes physically reasonable (< 1 mM/s peak)
  - element mass-balance drift < 1 atom/s (worst: H at 0.5/s, from K_m fallbacks)
  - gradients flow through learnable_rates path
  - PGI spot-check: enzyme P_0445, k_cat 804/s fwd 650/s rev — biologically sane
- **Wired into model**: NOT YET (next commit)
- **Headline result**: NOT YET (next commit)

#### Next step (commit 2)
Wire MetabolismCore into DynamicsModel as a parallel branch to the PINN head.
The PINN head still runs (for reactions MetabolismCore doesn't cover); for the
160 wired reactions, MetabolismCore's prediction takes over. Then a Colab run
shows whether rollout R² jumps above stride=30 baseline of 0.625.
