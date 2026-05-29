# Equations to wire into the emulator — build list

---

## v15.2 — diagnostic suite + framing audit

Nine post-hoc diagnostics from the "How Nature Reaches Outcomes Without
Computing Them" technical survey (Freidlin–Wentzell, JKO, Barato–Seifert,
Wang landscape–flux, Maximum Caliber, Putnam–Chalmers–Piccinini computation
criterion).  Each is grounded in a specific peer-reviewed theorem; none
requires retraining or architecture changes.

| # | Diagnostic | Theorem | What it catches | Falsifier? |
|---|---|---|---|---|
| 1 | TUR per reaction | Barato–Seifert *PRL* 2015 | reactions where σ·Var(J)/⟨J⟩² < 2k_B | **yes** — hard physics |
| 2 | Counterfactual-robustness | Putnam–Chalmers–Piccinini (computation criterion) | computing vs memorising per species | soft (relative) |
| 3 | Friedlin–Wentzell action | Freidlin–Wentzell 1984; report §10 | predicted trajectories that are improbable under the inferred SDE | soft (relative) |
| 4 | Stop "optimisation/finds" framing | — | language hygiene (Maximum Caliber, not MEPP) | pedagogical |
| 5 | Drop FEP / grand-framework appeals | Biehl 2021, Aguilera 2022 (FEP refutations) | unfalsifiable scaffolding in our justifications | pedagogical |
| 6 | Constraint provenance audit | (debugging tool, no theorem) | which data source loaded the wrong constraint | diagnostic |
| 7 | Kramers residence times in lipid bins | Kramers 1940; Hänggi review | wells where model spends wrong amount of time → doubling-time mechanism | **yes** if mismatch |
| 8 | Sliced W₂ per species | Jordan–Kinderlehrer–Otto 1998 | distribution mismatch (the metric per-trajectory R² should have been) | replaces wrong metric |
| 9 | Helmholtz curl detection (2D proj) | Wang 2015; Friston–Da Costa 2023 | model collapsed to pure-gradient (equilibrium-like) when cell is NESS | **yes** if Q ≈ 0 |

All wired into the eval block in `main()` after `paper_validation_metrics`.

### #4 (framing audit): use Maximum Caliber, not MEPP / "finds the optimum"

**Why**: MEPP (Maximum Entropy Production Principle) has been formally refuted
(Bruers 2007, Paquette 2010, Grinstein–Linsker 2007 explicit counterexamples).
Maximum Caliber (Jaynes 1980; Pressé–Ghosh–Lee–Dill *Rev. Mod. Phys.* 85:1115,
2013) is the rigorous path-space MaxEnt and IS a theorem.

**Default vocabulary** for describing what our model does:

| Bad framing | Good framing |
|---|---|
| "the model finds the trajectory" | "the model produces a distribution over trajectories whose mode is X" |
| "the model optimises the objective" | "the model minimises a training loss with respect to its parameters" |
| "the cell selects the optimal path" | "the cell's stochastic dynamics concentrate on trajectories with low Freidlin–Wentzell action" |
| "evolution optimises fitness" | "selection acts on heritable variation; differential reproduction shifts allele frequencies" |
| "the simulator computes the cell" | "the simulator integrates an SDE that samples from a stochastic path measure" |
| "constraint-shaped local dynamics" (used as a *law*) | "constraint-shaped local dynamics + filtering" used as a *question-generator* — where did the constraint get loaded? |

Loss functions are objectives we wrote down.  Cells don't have objectives.
Trajectories aren't "found"; they're sampled.

### #5 (grand-framework audit): no FEP / constructor-theoretic justifications

**Why**: as of 2026, the original Friston "free-energy lemma" is refuted by
counterexample (Biehl–Pollock–Kanai *Entropy* 23:293, 2021), Markov-blanket
existence is restrictive (Aguilera–Millidge–Tschantz–Buckley *Entropy* 24:18,
2022), and constructor-theoretic predictions of life/thermodynamics
remain arXiv-only since 2016 with no novel falsifier (Plesnik–Violaris
arXiv:2404.07786, 2024).  Treating these as "established theory" smuggles
unfalsifiable scaffolding into our reasoning.

**Allowed**: cite FEP / constructor-theory as a *lens* or *charitable
reformulation* where the math directly applies (e.g., when our model has
an explicit Gaussian NESS and Heins–Da Costa conditions are checkable).

**Not allowed**: justify a model choice by "this is what FEP / constructor
theory says we should do" without showing the specific math derivation
that supports the choice in our setting.

---

## v15.1 — refinement-pass training + reverted transformer

v15.0 Colab result: 1-step R² **0.814** (beats persistence 0.781), rollout R² **0.586**
(within noise of v14.9's 0.578), KO MCC +0.100 (flat), σ over-confident by 10×.
The transformer added 1.1M params and 2× wall-clock for ~0 metric movement.
Diagnosis: history buffer fills with model predictions, transformer attends over
its own errors and compounds them at long rollouts (the K=40 training-loss flip
from -1.32 to +1.89 was the smoking gun).

**The bottleneck is no longer architecture — it's rollout self-consistency.**
The 0.22 R² gap between 1-step (0.814) and rollout (0.586) is pure compounding
error.  Adding more parameters cannot close it; we need a training signal that
punishes the model for being unstable under its own outputs.

### Three changes:

1. **Disable transformer** — `USE_TEMPORAL_CONTEXT=False`.  Code retained for future
   re-enable if combined with teacher-forced history during training.
2. **Bump σ-anchor** — `LAMBDA_SIGMA_ANCHOR` 0.05 → 0.15.  v15.0's σ was 10× over-confident.
3. **Refinement-pass loss** in the last 10% of training (`USE_REFINEMENT=True`,
   `LAMBDA_REFINE=0.15`, `REFINE_START_FRAC=0.9`):

       pred_1 = model(state(t))           # normal forward, predicts state(t+1)
       pred_2 = model(pred_1.detach())    # feed prediction back → predicts state(t+2)
       loss += λ · MSE(pred_2, true(t+2))

   Gradient only flows through the SECOND forward (pred_1 is detached).  Teaches:
   "if your input is your own imperfect prediction, still produce a good next-step
   prediction" — directly attacks the rollout-vs-1step gap.

   Cost: 1 extra forward per rollout step in the last 10% of training = ~3% extra
   wall-clock.  Activates at step 2700 of 3000.

Expected effect: rollout R² closer to 1-step R² (0.81 ceiling) by some fraction.
Headline result: pending Colab run.

---

## v15.0 — LGNN + Transformer hybrid (two-cortex architecture, disabled in v15.1)

The LGNN handles local species-graph dynamics; a small transformer (`TemporalContext`)
handles global trajectory shape via attention over the last 8 states.  Brain-cortex
analogy: graph layer = local circuits, transformer = "where in the cell cycle are we?"

| Component | Params | What it adds |
|---|---|---|
| `TemporalContext` module | ~178k | Linear(S→32) embed + learned positional encoding + 2-layer encoder (4 heads, FFN dim 64) + Linear projection → context vector |
| `ctx_to_hidden` projection | 2k | Adds the context vector as a per-batch bias to the LGNN's hidden state at every species (broadcast) |

Wired into `DynamicsModel.forward(x, state_history=...)`; rolling history buffer
maintained in `train_model`, `full_rollout`, `full_rollout_batched`, `_ko_rollout`.
History buffer shape `(B, T_CTX_WINDOW, S)` is constant throughout a rollout
(torch.compile-friendly).  Output projection zero-init → starts as v14.9
behaviour, transformer wakes up gradually during training.

Configuration knobs at the top of the file:
- `USE_TEMPORAL_CONTEXT=True` — gate flag
- `T_CTX_WINDOW=8` — past states attended over (W=8 is half the cell cycle at K=120, stride=30)
- `T_CTX_HIDDEN=32` — per-token embed dim
- `T_CTX_LAYERS=2`, `T_CTX_HEADS=4`, `T_CTX_FF=64` — transformer shape

Also: `STEPS` bumped 1500 → 3000 to give the larger architecture room to converge.

What this attacks (theory):
1. **Mode collapse on long rollouts** — attention back to t=0 keeps the model
   anchored to the seed; pure LGNN drifts toward mean predictions as horizon grows.
2. **Phase awareness** — positional encoding within the window tells the model
   "this is step N of the cell cycle"; LGNN's CfC cells encode tau but not
   absolute phase.
3. **Global drift detection** — the transformer sees pooled state summaries
   and can correct LGNN predictions when they diverge from a plausible trajectory.

Standalone tests pass (`/tmp/test_temporal_ctx.py` style): forward path correct
for short/exact/long histories, gradients flow, output near zero at init.

Headline result: pending Colab run.

---

## v14 — 5-day physics list (consolidated)

Independent of the v13.9 module plan below.  This is the user's
physics-grounded list (8 items, see chat history for full framing); each
constraint is implemented as a `nn.Module` extension or a training-loop loss term.

| # | Item | Day | Status | Notes |
|---|------|-----|--------|-------|
| #1 | Stoichiometric conservation | – | inherited from v13.9 (cores 1, 3, 4 push updates through S·v) | – |
| **#2** | ΔG° sign clamp | day 1 | **landed** | k_cat_rev=0 when ΔG° < -10 kJ/mol |
| **#7** | Ribosome pool cap | day 2 | **landed** | shared-pool MM on translation |
| **#3** | ATP energy ledger | days 3-4 | **landed** | deficit penalty when net ATP < NGAM floor |
| **#8-lite** | σ-calibration anchor | day 5 | **landed** | log σ pulled toward empirical std, eval-time calibration metric |
| #4, #5 | Terminal manifold + doubling | – | deferred — needs full-cell-cycle rollouts in training | – |
| #6 | Dissipation Lyapunov | – | deferred — proxied by multi-step rollout loss already | – |
| #8 full | Full FDT coupling | – | deferred — calibration anchor is the lite version | – |

Expected post-Day-5 outcome (untested on Colab yet):
rollout R² 0.79–0.89, KO MCC 0.30–0.50, element drift 20–30%, wall-clock ~30 min.

---

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
| 1 | **MetabolismCore** — bi-bi rate law, 356 SBML reactions | `kinetic_params.xlsx` (k_cat_fwd/rev, K_m per substrate/product), SBML stoichiometry | PINN head's neural `v_log` prediction | 0.70–0.85 | unchanged | **wired, awaiting Colab** |
| 2 | **VolumeCore** — dynamic volume per timestep | membrane-lipid count proxy (`LIPID_PREFIXES`) | constant-volume assumption | +0.05–0.15 | – | **wired** |
| 3 | **CentralDogmaCore** — per-gene tx / tl / mRNA-deg / prot-deg | per-gene G/R/P species; literature k_tx, k_tl, half-lives | LGNN's prediction for mRNA + protein species | +0.10–0.20 | – | **wired** |
| 4 | **AssemblyCore** — complex assembly + 50S biogenesis (mass-action) | `complex_formation.xlsx`, `LargeSubunit.xlsx` | LGNN's prediction for complex species | +0.05–0.10 | – | **wired (complex_formation only; LargeSubunit deferred)** |
| 5 | **TransportCore** — 58 transport reactions (bi-bi) | `kinetic_params.xlsx` Transport sheet | – | – | – | **subsumed by Module 1** (MetabolismCore reads all 5 sheets) |
| 6 | **tRNAChargingCore** — 20 aa charging (bi-bi) | `kinetic_params.xlsx` Cofactor sheet | – | – | – | **subsumed by Module 1** (same — all kinetics share the bi-bi rate law) |
| 7 | **KnockoutAugmentation** — random species-zero per training batch | – | model trained only on unperturbed data | unchanged | **+0.20–0.40** | **wired** |
| 8 | **ReplicationCore** — DnaA filament + replisome | upstream `replication.py`, would need new data parse | LGNN's prediction for `chromosome` + `ori_rep*` species | +0.05–0.10 | – | deferred (state-machine complexity) |
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

### Module 7 — KnockoutAugmentation

- **Status**: wired (commit v13.9 module 7)
- **Approach**: input perturbation during training, persistent through the K-step
  rollout, target reflects the same knockdown.  No new module class — modifies
  `train_model` directly.
- **Mechanism**:
  - Per training step, ~30% of batch elements (`KO_AUG_PROB=0.3`) have one
    random species replaced by its `zero_norm` value (the normalised count=0
    point per species) in both the input state and the per-step target.
  - The knockdown is re-applied to `nxt` after each K-step so it persists
    (matches the eval-time `knockout_sweep` permanent-knockdown semantics).
  - The equation cores then naturally propagate the perturbation downstream
    (zero enzyme → zero flux → cascading deltas through the stoichiometry).
- **What it teaches the model**:
  1. A zeroed species stays zeroed (target-side knockdown).
  2. Other species follow trajectory dynamics — the model has to find the
     RIGHT response without supervision (the equation cores carry the
     downstream-propagation load).
- **Cost**: ~10% slower per training step (extra clone + masked write); no
  separate forward pass.
- **Expected**: KO MCC −0.04 → 0.2–0.4 when combined with the equation cores
  doing the heavy lifting on enzyme-driven flux.
- **Headline result**: pending Colab run.

### Modules 5, 6 — subsumed by Module 1

`parse_kinetics` already reads all five sheets (Central, Nucleotide, Lipid,
Cofactor, Transport).  `build_metabolism_tensors` then wires every SBML
reaction with full kinetic data — that covers transport reactions and aa
charging reactions automatically, since they share the same bi-bi rate
law and stoichiometric matrix.  No separate `TransportCore` or
`tRNAChargingCore` needed.

### Module 8 — ReplicationCore: deferred

Upstream's `replication.py` is a state machine: DnaA filament forms by
sequential binding to high- and low-affinity oriC sites; once filament
length ≥ 20 the replisome (P_0044) loads; replication proceeds gene by
gene at position- and size-dependent rates; termination at JCVISYN3A_0421
doubles the chromosome.  To wire this correctly would need parsing the
upstream state machine + finding the DnaA / replisome / oriC species in
our trajectory.  Expected gain is small (+0.05–0.10 rollout R²), the
implementation is large.  Deferred until other modules' Colab gain
plateaus and replication-related species become the bottleneck.

### Module 4 — AssemblyCore

- **Status**: wired (commit v13.9 module 4); LargeSubunit chain deferred.
- **Approach**: mass-action with safety cap.
  - rate = `k_on · Π subunit_i^stoich_i`
  - Per-reaction rate cap: cannot drain > 50% of any subunit pool per step
    (`ASSEMBLY_SAFETY_FRAC = 0.5`).  Protects against Euler overshoot on
    fast reactions over a long 30s step.
- **Coverage**: complexes from `complex_formation.xlsx` where every subunit
  resolves to a `P_<locus>` species AND the complex name resolves to a
  trajectory species (tries bare name + `C_<name>` variants).
- **Default k_on**: 1e-5 /s/molecule^stoich.  Will need calibration against
  upstream when we know how many complexes wire on real data.
- **Standalone test** (`/tmp/test_asmcore.py`): 11/11 pass — stoichiometry
  perfect, safety cap holds at exactly 50%, KO P_0001 zeros C_RNAP rate
  without affecting C_RIBO, gradients flow.
- **Bug caught + fixed**: padding slots in the rate-cap calculation
  were dragging max_per_rxn → 0.  Fixed by setting padding-slot max to +inf
  via `torch.where(stoich > 0, ratio, inf)`.
- **Headline result**: pending Colab run.
- **Deferred**: LargeSubunit.xlsx (32-step 50S assembly chain) — would
  need a sequential-step graph representation, not the simple "subunits →
  complex" pattern here.

### Module 2 — VolumeCore

- **Status**: wired (commit v13.9 module 2)
- **Approach**: lipid-sum proxy.  `V_L(t) = V_0 · (Σ lipid_count(t) / Σ lipid_count(0))`
  where lipid species are matched by `LIPID_PREFIXES` heuristic (PE, PG, PC, PS,
  cardiolipins, cholesterol, glycerol-P intermediates).
- **Standalone test** (`/tmp/test_volumecore.py`): V doubles when lipid count
  doubles; no-lipid fallback returns None.
- **Plumbed into**: `MetabolismCore.forward(volume_l=...)` and
  `DynamicsModel.forward()` (called once per step, broadcast to all reactions).
- **Headline result**: pending Colab run.

### Module 3 — CentralDogmaCore

- **Status**: wired (commit v13.9 module 3)
- **Approach**: first-order tx/tl/decay per gene with literature constants.
  - `k_tx = 0.06 /s`, `k_tl = 2e-3 /s`, `T_half_mRNA = 120s`, `T_half_prot = 36000s`
  - Tuned to give plausible steady states: R_ss ~ 10 per gene, P_ss ~ 1000 per mRNA
- **Coverage**: every gene locus with all three of G/R/P species in the
  trajectory (prefers `_C1` chromosome-copy variant).  ~490 genes expected on
  real data → ~980 species covered (mRNA + protein).
- **Standalone test** (`/tmp/test_cdcore.py`): 8/8 pass — steady-state math
  exact to 1e-6, KO causes mRNA decay, genes are NOT modelled (left to LGNN),
  gradients flow.
- **Headline result**: pending Colab run.
- **Limitations** (deferred to future modules):
  - No NTP/aa/ribosome pool coupling — using literature-average rates.
  - No per-gene rate constants — would need gene-length parse from `syn3A.gb`.
  - Genes themselves left to LGNN (no DNA replication module yet).

### Module 1 — MetabolismCore

- **Status**: foundation landed (commit v13.9 part 1); wired into DynamicsModel (commit v13.9 part 2)
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
- **Wired into model**: yes (DynamicsModel takes priority over PINN for covered species)
- **Headline result**: pending Colab run with modules 1+2+3 all active
