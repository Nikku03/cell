# REM-Cell: what the architecture asks for, and what this repo can supply

Written 25 August 2026, after 200 loops. Every number below was read from a file in this
repository and is cited with its path and loop, or is marked as arithmetic over cited counts, or
is marked as measured in the red-team session and **not yet committed as a loop output**. Where a
number could not be found, the entry says *not measured* rather than guessing.

This file is a design review, not a proposal. The conclusion is in section 7 and it is negative.

---

## 1. What REM-Cell is

REM-Cell is a proposed whole-cell simulator built on four ideas.

**A hierarchical biological graph.** The cell is a tree of clusters — 10–20 strongly-related nodes
per cluster, communicating with the rest of the model through a narrow boundary message rather
than through all-to-all coupling. This is the design's answer to the expander failure of loops
171–172: do not wire everything to everything, and do not spend design effort on which sparse
wiring, because at fixed density the topology did not matter.

**Equation-based state nodes.** Each node holds a small state vector `z_i` and an explicit update
law `z_i^{t+1} = F_i(z_i^t, N(i), g^t)` — an ODE, a Markov chain, a stoichiometric constraint, a
learned operator — rather than an opaque embedding.

**Dynamic global fields.** A small number of cell-wide scalars `g_k^{t+1} = G_k(g_k^t, Σ_i
W_ki(z_i^t))` — growth, energy, crowding, ribosome capacity — written by every node and read back
by every node, so that long-range coupling costs O(n) rather than O(n²).

**Adaptive Stage-6 representation selection.** Each node's mathematical class is chosen from its
own observable features rather than fixed by fiat: ODE where copy numbers are large, stochastic
simulation where they are small, spectral where the operator is low-rank, learned where the
relation is unknown.

**An asynchronous multiscale scheduler.** Nodes wake on their own clocks, from milliseconds to
hours, with staleness bounds on the inputs they read between wakes.

The design principle behind all five: **never pay exponential cost merely because the theoretical
state space is exponential.** A cell has 10^13 possible states and roughly one trajectory; the
model should cost the trajectory.

---

## 2. What this repo can actually supply

Coverage denominators are the 16,492 genes in `colab/data/cell_complete.json.gz`, the 12,931
reactions in `HumanGEM.xml`, or the layer's own denominator where it differs. "Parameters" means
**rate constants the update law needs**, not state variables.

| subsystem | state available | coverage | parameters have / need | verdict |
|---|---|---|---|---|
| Loop extrusion + CTCF barriers | cohesin legs (2×321 ints), site occupancy bits, chr21 1,926 bins | 40,790 CTCF peaks genome-wide; 359/404 oriented on chr21 (88.9%) | 4 / 4, **all literature, none fitted to the scored map** (`outputs/loop_extrusion.json` `parameters_literature`: v 0.75 kb/s, residence 900 s, one per 150 kb; τ_CTCF 120 s in `colab/loop_insulation.py`) | instantiable — and refuted as a chromosome (§4) |
| 3D conformation (spectral polymer) | Laplacian, mode amplitudes | chr21 + replicate, chr22 dense; genome-wide only ±1.125 Mb bands | 2 calibrated from geometry, 2 unidentifiable (κ gives persistence 25.0 kb at every value in {0,4,8,16}) | instantiable, engine certified, model refuted |
| Per-gene chromatin position | (pc1, ins, dens) | 16,216 / 16,492 (98.3%) | no update law exists, so no rate constants | static (F_i = identity) |
| Replication timing / fork | RT per gene, 20 tracks | 16,325 / 16,492 (99.0%) | 0 of 1 decisive parameter — **no measured origin map anywhere on this disk** | not present as a mechanism |
| DNA torsion | σ̂ per gene | 16,348 genes; 10,266 joined to a rate | 0 of 2 kinetic constants (Lk flux, topoisomerase turnover) | static field only |
| mRNA pool | M_g | loss for 13,105 / 16,492 (79.5%); production for 4,190 (25.4%, mouse) | loss real, production **93% abundance-degenerate** (ρ +0.9333, partial −0.9137) | instantiable at steady state |
| Protein pool | P_g | loss for 5,915 / 16,492 (35.9%), 79.2% of abundance mass | k_sp back-solved as P·b/M — circular to 3.1e-14 (`outputs/loop_integrator.json` `i2_circular_excluded`) | instantiable, one constant is not a measurement |
| mRNA decay emitter (sequence → k_deg) | k̂_deg | 8,556 K562 genes scored held out | learned, fully specified | instantiable |
| Promoter accessibility | A_g(t) | peak at some timepoint for 18.9% of genes | **no F_i was ever fitted** | replayed boundary condition only |
| Nascent transcription | PRO-seq gene body | 11,001 genes, 3 timepoints | 0 of 3 (k_init, k_release, v_elong). 120 kb/h is hard-coded at `colab/loop_rates.py:102` and never gated | not present |
| Promoter bursting (telegraph) | (s_g, m_g) | 1,420 / 16,492 (8.6%) | 3 per gene, **in units of k_deg — no absolute clock** | instantiable open-loop, uncontrollable |
| Nuclear/cytoplasmic mRNA | (M_nuc, M_cyt) | **0 genes parameterised**; solver verified to 1.9e-14 | 0 of 2 (k_exp, γ_nuc) | not present |
| TF regulatory operator | 612,133 directed edges | 16,492 genes | **0 of 612,133 gains**; 8.8% signed; 0 self-edges | not present |
| Metabolic flux (FBA) | v ∈ R^12931 | 12,931 / 12,931; S is 8,461×12,931, 55,198 nonzeros | LP complete; **0 of 12,931 measured fluxes to score against** | instantiable, unfalsifiable here |
| Metabolite concentrations | c ∈ R^8461 | **0 of 8,461**; 0 of 9 compartment volumes | none | not present |
| k_cat | 8,184 values | 2,549 genes | values exist and **lose to a constant** (§4) | present and refuted |
| Ribosome / proteasome budgets | (R_tot, D, U) | R_tot from 73 ribosomal proteins; N_26S from 34 subunits | complete, nothing fitted | instantiable |
| Growth dilution μ | scalar | all genes | measured, error-bounded to 3.97% | instantiable |
| Complex assembly | C_k, free pools | 2,039 complexes, 3,257 genes | **0 of 4,078 rates, and the stoichiometry coefficients are absent too** | not present |
| Protein localisation | (P_g, f_cyt, comp) | 8,710 proteins measured; 7,202 join the model (43.7%) | 0 of 17,420 transport rates | static partition only |
| Kinase → substrate signalling | 74,862 sites, 5,136 edges | 12,883 proteins | 0 kinase k_cat (checked: CDK1, CDK2, MAPK1, AKT1, SRC, EGFR, PLK1, AURKB, GSK3B all return None in `colab/data/kinetics_bundle.json.gz`); **no phosphatase layer at all** | not present |
| Cell-cycle phase | θ | 153 phase-labelled genes (0.93%); **S phase is not among them** | clock has its one parameter; coupling has none and is falsified | open-loop clock only |
| Element → gene links | score per pair | 11,933 pairs, 199 evaluable genes | learned, R@1 0.6734 vs distance 0.5930 | instantiable, static |

The cascade that matters for a coupled model is in `outputs/cell_run.json` `c4`, read today:

```
in the model                              16,492
+ full dynamical state                     3,363
+ a metabolic reaction                       721
+ a signed TF regulator                      440
+ a measured kcat surviving loop 127          87     core_mass_fraction 0.0249
```

**87 genes, 2.49% of proteome mass.** That is the intersection REM-Cell's cross-subsystem
propagation would run on.

---

## 3. The architecture as designed

### 3.1 The graph and its clustering

Four clustering criteria were considered and each was tested against the repo.

- **Shared stoichiometry.** Human-GEM's 147 subsystems have median 29 reactions and 52 species —
  but 60.0% of species are shared across subsystems, median boundary fraction 0.821, falling only
  to 0.765 once the 33 currency metabolites at degree > 200 are removed. Refining to 405
  (subsystem × compartment) cells raises the cut fraction to 0.849 and puts only 73 cells in the
  10–20 reaction band, holding 8.3% of reactions. **Finer partitions have proportionally more
  boundary, not less.**
- **Obligate complexes.** 2,039 complexes, median size 3, only 106 in the 10–20 range, and no
  rates.
- **Co-timing** — the criterion a dynamical cluster actually needs. **Refuted.** Loop 194: 1,210
  enzyme pairs coupled by shared chemistry, observed mean |Δt| 143.4 min against a
  1,000-permutation null 139.8 ± 4.6, z −0.8, stable at hub thresholds 25/50/100. Curated pathway
  membership: z +1.1.
- **Compartment** — the only criterion that ever bought anything. Loop 170 took metabolite
  completion hit@1 from 0.7266 to 0.8506 (+0.1240 ± 0.0047) on the strength of
  `same_comp_as_seed` and `log_transport_pairs`. That gain came from **naming the partition and
  counting its crossing edges**, not from wiring inside it.

Biology hands three natural grains — gene (1), complex (median 3), subsystem (median 29–52) — and
none is "10–20 nodes with a narrow interface." The only clustering this repo can defend is **by
shared solver**: chromosome mechanics (dt 33.3 s), gene expression (dt 0.25 h), the metabolic LP,
the budgets, the static emitters, and a declared-uninstantiable remainder. That is a bookkeeping
choice and should be labelled one.

**The hierarchy is not a tree.** `outputs/loop_cell_census.json` `pairwise`, read today: 44 genes
bridge the regulatory network and the reaction layer; **2 genes** bridge motif and reaction. A
rank-2 channel between a 723-gene module and a 2,568-gene module is a coincidence, not a summary.
The honest object is a forest of weakly-coupled layers with named seams.

### 3.2 Boundary messages, and one new measurement

Widths, with the ill-posed one marked:

| message | width | note |
|---|---|---|
| loop set {(l_k, r_k)} | 642 ints (or 4 summary statistics) | **no subscribers** — see §4 |
| metabolism: (μ, J_ex) | 1,659 | solver-invariant |
| metabolism: v | 12,931 | **must never be emitted** |
| medium bounds consumed | 54 | all 53 named exchanges resolve |
| gene expression (M_g, P_g) | 8,380 | 4,190 genes |
| budgets (U_ribo, U_prot, μ) | 3 | |
| regulation → metabolism | 18 genes × 2 bits | a substrate-class tag, not a rate |

The metabolic entry was measured in the red-team session, not previously recorded: rebuilding the
model from `outputs/orphan/cell_reactions.json` alone reproduces loop 72's growth to 16 digits
(0.020359220115642406 /h), and solving it with two different LP backends gives the identical
optimum with interior fluxes differing by up to **1.597 mmol/gDW/h** (support 1,056 vs 980 at
|v|>1e-6, pFBA 699) while **all 1,658 exchange fluxes agree to 6.25e-15**. The optimum and the
boundary are functions of the inputs; the interior is a function of the solver basis. Typing the
metabolic boundary message as (μ, J_ex) and keeping v private is therefore not a style preference
— it is the difference between a reproducible module and one that changes answer between wakes.

### 3.3 Global fields

The test a field must pass to be an information channel rather than a rank-1 all-to-all coupling:
**shuffle the identities of the writers, verify the shuffle is capable of moving the statistic,
and require |z| > 2 on the field.** Both fields this repo tested fail it.

| field | writer | reader | status |
|---|---|---|---|
| μ (growth / dilution) | protein budget, biomass flux | every loss rate, 16,492 genes | **measured as bookkeeping, refuted as a channel.** Discrete-vs-continuous error bounded at median 3.89% / max 3.97% over 4,821 genes against an analytic limit matched to 6.5e-07 (`outputs/loop_division.json` d2). But `outputs/loop_growth_loop.json` g7: protein-identity shuffle capable at 54.6% of v_max, real μ* 0.1258163 vs null 0.1035270 ± 0.0261927, **z 0.851, survival UNDEFINED** |
| ribosome capacity | codon demand | shared cap | **measured**, every constant counted, nothing fitted: U 0.2951 growth-only → 0.4166 with degradation, 29.18% replacement (`outputs/loop_lifetime.json`). One failure: 12.38% of genes exceed the 672/h initiation cap against a <5% bar |
| crowding | proteome mass | diffusion coefficient only | **measured** (163.32 mg/mL cytosol), physically calibrated, and **write-only** — no rate constant in this repo depends on it |
| energy / ATP | translation, phosphorylation, biomass | **nothing** | **refuted twice over.** `outputs/loop_signalling_cost.json` y5: site-count shuffle capable at 93.1%, **null ρ +0.9255 exceeds real +0.9218, z −3.003** — the ATP bill is abundance with extra steps. And both non-growth maintenance reactions are bounded exactly [0.0, 0.0] (MAR09931, MAR09932), so at the optimum growth is the only ATP sink and energy is 45·μ, with no degrees of freedom |
| compartment composition | every gene | nothing | measured and rank-1 (PC1 79.8% vs null 12.5% ± 0.2%, z +363) — but over **1,517 DepMap cell lines, not timepoints**. 0 of 12 relaxation times exist |
| cell-cycle phase | nothing | would gate b_g, k_sp | posited; the coupling the expression ODE would produce is falsified (§4) |
| redox, proteotoxic stress, pH / ΔΨ | — | — | **posited with no value anywhere.** 1,921 redox reactions and 4,070 proton-moving reactions are enumerated structurally; zero ratios, zero pH, zero volumes |

Nine candidate fields; three measured with a working control, one measured and refuted, one on the
wrong axis, four with no value. **None of the three survivors passes the writer-identity test.**
Keep μ, ribosome capacity and crowding as shared scalars with bookkeeping semantics; delete the
other six; and do not present any of the three as a channel.

### 3.4 The Stage-6 selector

The design's table maps subsystem → mathematical class. Every measured head-to-head in this repo
says that is the wrong axis, because in each case the mathematically-correct class was
instantiated and lost to a parameter-free sibling:

| rich representation | its cheap sibling | source |
|---|---|---|
| polymer, genome-scored: **0.8229** | distance-only null **0.8283** | `outputs/loop_genome_reselect.json` (loop 90) |
| per-gene k_cat: **12.95×** median fold error | global constant **9.42×**, p 0.5535, n 66 | `outputs/loop_kcat_audit.json` k2 (loop 124) |
| accessibility-informed ridge: **−0.0520** | persistence **−0.0295** | `outputs/loop_capability_4d.json` (loop 198) |
| rate routed through the ODE: ρ **0.4649** | raw Θ, no ODE, ρ **0.4960**, gain −0.0308 vs a +0.010 bar | `outputs/loop_integrator.json` `i5` (loop 112) |
| fork PDE: R² **0.2291** | Gaussian blur of the same track **0.3201** | replication-timing loop |
| chromatin-emitted k_sm: **1.800×**, ρ +0.343 | constant k_sm = 1.8401: **1.827×**, ρ +0.326 | `outputs/loop_chromatin_to_rate.json` `x7` (loop 114) |

So the selector's primary axis is not "what shape are the dynamics" but **"do the parameters that
carry the derivative exist, and does the richer representation beat its own cheapest sibling on
held-out data."** Structure features only break ties among representations that already passed
both.

The rule is lexicographic and **climbs** from the cheapest rung rather than descending to it:

- **G0 admissibility** (ledger lookup, no compute). Reject if any term carrying a time derivative
  has an unmeasured constant, or a constant back-solved from the quantity being predicted, or a
  swept constant whose verdict flips inside the sweep. G0 alone eliminates the metabolite ODE
  (0/8,461 concentrations, 0/9 volumes), the regulatory operator (0/612,133 gains), the complex
  graph (0/4,078 rates and no stoichiometry), the localisation flow (0/17,420), the
  nascent-transcription cascade (0/3), the cohesin-pool ODE (0/2), and every HMM (no phase axis).
- **G1 identifiability.** Sweep each surviving parameter; if the score-elasticity is below the
  score's own replicate noise, freeze the parameter at its population constant and demote. This
  rule mechanically reproduces the repo's own answers: τ_coh ×10 leaves half-recovery at exactly
  20.0 min; κ ∈ {0,4,8,16} gives persistence 25.0 kb at every value; fork speed ×56 moves ρ by
  0.0222.
- **G2 earning.** Beat the immediate cheaper rung on a held-out split whose axis matches the claim
  — time for dynamics, chromosome for genomic features, homology cluster for sequence — against
  `max(persistence, cheap structural null, fame floor)`, with the null verified capable first and
  survival returned as UNDEFINED when |z| < 2.

The **fame floor** is not a diagnostic printed afterwards; it is part of the bar. Publication count
scores ρ +0.3911 on the loop-114 target where the chromatin emitter scores +0.0159
(`outputs/loop_chromatin_to_rate.json` x3, x4).

**Switching cost.** Exactly one representation bridge in this repo is verified with an error bound
(dense inverse ↔ Woodbury, max rel err 1.982e-12, 171.87 ms vs 272.90 ms) and one more is
analytically bounded (discrete division ↔ continuous dilution, max 3.97%). The ODE → stochastic
lift is **UNDEFINED for metabolites** because `N = c·V·N_A` needs volumes that do not exist. The
Markov → stationary-field collapse must be forbidden by default whenever any consumer reads a
residence time: the CTCF chain's stationary law is exact by construction, but the measured
mechanism is the dwell time — mean stall 2.913 min over 1,520 encounters against 1.058 min for the
memoryless coin it replaced, worth +0.064 insulation (loop 43).

**Runtime failure detection.** Six monitors, five with a measured precedent: refinement-vs-replicate
(§3.5), invariant residual (9,831 of 9,855 checkable internal reactions balance on all 22 elements,
97.93% on mass and charge; 26.7 s for a full sweep), moment consistency (the birth-death ODE
implies a CV-vs-N log-log slope of −0.5 and the measurement is **−0.2055** on 2,582 genes, so the
correct action is to refuse to report a variance, not to upgrade the representation), iteration
spectral radius (§3.5), budget saturation, and coverage drift.

### 3.5 The scheduler

**Wake conditions.** A module wakes on its own clock, or when a subscribed channel has moved by
more than `eps_c` — and `eps_c` is read off the measuring instrument's replicate, never tuned. A
message a replicate could have produced is not news. Measured floors: Hi-C same-dt replicate ρ
0.9692 (`outputs/loop_second.json`), P(s) replicate spread 0.005, k_cat replicate 0.0607 log10,
A549 interval-change reliability 0.6007, metabolic exchange flux 6.25e-15.

**The clock is adequate where it was tested, and refinement is never justified there.**
`outputs/loop_second.json` swept dt over {33.33, 10, 3, 1} s: P(s) spread 0.016370 across the whole
sweep, ρ against the 1 s run 0.9713 at dt = 33.33 s, against a **same-dt replicate ρ of 0.9692**.
The margin is 0.0021. Refining 33× buys less than the estimator's own noise.

**Staleness bounds.** For a contractive linear node the bound is analytic and needs no new
measurement: holding an input stale for τ while it drifts by δ gives output error ≤ (1 −
e^{−bτ})·δ ≤ δ. At the median protein b = ln2/71.5455 + 0.0252 = 0.0349 /h, a 20%-stale input
stays inside 1% for 1.46 h. Consistent with the measured relaxation: median t95 is 31.0 h for mRNA
and 83.25 h for protein, and **only 0.18% of genes (5.05e-05 of mass) equilibrate within one 24 h
cycle** (`outputs/loop_integrator.json` `i7`). For the LP there is no global bound on disk, only a
local elasticity, dlog(μ*)/dlog(k_cat) = +0.363.

**Consistency between clocks — and this is measured, not hypothetical.** The one closed
multi-module feedback in this repo does not converge under naive relaxation.
`outputs/loop_growth_loop.json` g2: F is monotone on a 9-point grid, |F'(μ*)| = 1.1187, **the fixed
point repels**; plain Picard runs 25 iterations without converging, oscillating between 1.6733 and
0.0017, while bisection reaches 0.12581633 and damped Picard 0.12581888 in 7. Two modules on
different clocks reading and writing a shared field *are* plain Picard with an arbitrary ordering.
So: any field with |F'| ≥ 1 is solved on a single designated clock by bisection and broadcast, or
damped with λ < 2/(1 − F') = 0.944 — a bound derived from a measured derivative rather than chosen.

**The admission rule.** On wake, a module's default action is **persistence**. It executes F_i only
with a recorded held-out-in-time score above the persistence bar for the channel it writes.
Applied today, on the only channel where that bar exists, **no node among the 58 in the subsystem
specifications qualifies**.

### 3.6 Cost

Timed in the red-team session or read from stored runtimes:

| operation | cost | source |
|---|---|---|
| build Human-GEM from JSON | 2.9 s | timed this session |
| cold FBA solve | 8.82 s | timed this session |
| warm `slim_optimize` | 41.2 ms (mean of 20) | timed this session |
| growth fixed point | 235 solves, 340 s total in loop 113; ~0.5 s at the warm rate | `outputs/loop_growth_loop.json` |
| extrusion KMC | 0.4166 s per configuration; 4 conditions in 1.095 min on 4 CPUs | `outputs/loop_extrusion.json` |
| 1,400-bead spectral solve | 0.4648 s for all 979,300 pair distances | `outputs/loop_polymer.json` |
| Woodbury loop update | 171.87 ms vs 272.90 ms fresh, err 1.982e-12 | `outputs/loop_second.json` |
| 4,190-gene ODE, 8,000 steps | seconds; agrees with independent RK4 to 5.14e-12 | `outputs/loop_integrator.json` `i1` |
| whole-network invariant check | 26.7 s | `outputs/loop_rem_chemistry.json` |
| metabolite re-ranker cold fit | 199 s (no persisted artefact on disk) | `outputs/loop_spatial_rerank.json` |

Scaling: the LP is O(nnz) and does not decompose (55,198 nonzeros, density 5.0e-04, but not
separable). The polymer is O(n³) time and O(n²) memory — chr1 at 9,971 bins is 139× chr21 per
inverse at 0.795 GB per dense matrix and the summed genome is 743× chr21, so genome-wide 3D is a
different algorithm, not a longer run. Fields cost O(k·n) per tick, which is negligible — **which
is exactly why the field count must be argued from evidence and not from cost.**

Total: roughly **1.4 s of wall time per simulated cell-hour**, about 60% of it in the chromosome
module. `df` reports 2.4 GB free. **Compute is not the constraint anywhere in this design.**

---

## 4. What the red team established

### 4.1 The persistence objection, in full

Any node claiming to step state forward must beat **held-out R² −0.02953459**. That is
`outputs/loop_capability_4d.json` `step_forward`, loop 198: the A549 dexamethasone grid
[30, 60, 120, 180, 240, 420, 480, 600, 720] min, fit on the first 6 intervals (6,680 rows), scored
on intervals ending 480/600/720 min (4,008 rows), over 1,336 genes.

```
persistence (predict no change)     -0.02953459
training-set mean change            -0.03034305
accessibility-informed ridge        -0.05204663
```

All three are negative, so the test window's own mean would score exactly 0.0. The informed model
is **76% worse than doing nothing.** Coverage is 1,336 of 58,735 rows — 2.3%.

The red team then ran the arms REM-Cell would actually use, on the same harness. All lose:

```
momentum (previous velocity)                       -1.91155
per-gene mean velocity fitted on train             -0.85106
relaxation to a train-window-extrapolated plateau  -0.33855
ridge on m_prev * dt                               -0.09581
```

And it found the one positive that changes how this failure should be read. **Relaxation to the
true per-gene plateau, with a single fitted global rate, scores held-out-in-time R² +0.36290**
(λ +0.003766/min) against a reproducibility ceiling of +0.5225 derived from the three replicates.
So the *form* REM-Cell wants — first-order relaxation to a set point — captures roughly 70% of all
reproducible signal on this course. **The form is right and the set point is missing.** Loop 198's
result should be read as "the map cannot compute a set point," not "dynamics do not work here."

The red team then measured what a set point must be worth. Degrading the oracle with calibrated
noise puts the crossover at **Pearson r ≥ 0.889** against the measured plateau (r 0.858 → R²
−0.0725; r 0.895 → −0.0205; r 0.912 → +0.0129), against a target measurement ceiling of r 0.989.
The bar is reachable in principle, which is what makes it a decidable test rather than a noise
wall. And with the best upstream predictor that exists anywhere — nine ChIP/DNase tracks measured
**in the same cells, in the same experiment**, 27 features, gene-held-out ridge — the set point
reaches **r +0.4104**, giving a full-model R² of **−0.02977**. It fails by 0.00024, and it fails by
collapsing: the fitted rate goes to λ = −0.000036/min, i.e. the optimiser correctly learns to do
nothing.

These set-point numbers were measured in the red-team session and are **not yet committed as a loop
output.** Committing them is the job of §5.

### 4.2 The parameter ratio

Arithmetic over the counts cited in §2, restricted to **rate constants** (state variables excluded,
duplicates across subsystems collapsed):

```
NEEDED
  birth-death constants, 4 x 16,492                65,968
  TF edge gains                                   612,133
  complex assembly/disassembly, 2 x 2,039           4,078
  compartment transport, 2 x 8,710                 17,420
  kinase-substrate + site dephosphorylation        79,998
  k_cat                                            12,931
  nuclear export k_exp                             16,492
  burst parameters, 3 x 16,492                     49,476
  transporter rate laws                               714
  regulated-degradation r_g                           748
  globals (elongation, NGAM, 9 volumes, ...)       about 23
                                                  --------
                                                  ~859,981

MEASURED AND USABLE
  mRNA half-lives                                  13,105
  protein half-lives                                5,915
  k_sm (mouse, 93% abundance-degenerate)            4,190
  burst parameters (dimensionless)                  4,260
  globals                                               4
                                                  --------
                                                  ~27,474   = 3.19%
  excluding k_sm as degenerate                    ~23,280   = 2.71%
```

k_sp counts as 0 independent — it is back-solved as P·b/M, circular to 3.1e-14, and costs −0.0308
when routed through the ODE. k_cat counts as 0 usable — 8,184 values exist and lose to a constant.

**71% of the requirement is one block, the 612,133 regulatory gains, at exactly zero** — and the
design's entire cross-subsystem propagation runs through it. The 3% that exists is not spread
thinly; it is concentrated entirely in the *loss* half of every equation, which is the half that
makes a node relax to a set point it cannot compute.

### 4.3 The remaining fatal objections

**The propagation chain breaks at its second link, on a null with demonstrated power.**
`outputs/orphan/ptm_sites.json`: kinase → TF-substrate co-dependency ratio 0.9713, CI
[0.8716, 1.0839], 0/20 draws p < 0.05, minimum detectable ratio 1.146 at 80% power — and the sham
control on random pairs scores 0.9959, so the real edges perform *below* random. Power is
demonstrated in the same file: BioPlex 3.0 through identical machinery scores 1.0884 on 75,639
pairs. Link 4 (TF → transcription) is worse than a coin: real edge signs score AUC 0.5465 against
shuffled signs 0.5494 ± 0.0079, with publication count at 0.5536 beating both (loop 120). Link 5
(chromatin → rate) is z 0.805, survival UNDEFINED. Link 7 (enzyme → capacity) is k_cat×[E] AUC
0.5201 against abundance alone 0.5415. **Four of seven links are at or below their own nulls.**

**The global fields are the thing that already failed.** `g_k = G_k(g_k, Σ_i W_ki(z_i))` is a
permutation-invariant sum read back by every node — a rank-1 all-to-all coupling by construction.
The two fields this repo tested for writer identity both failed, one of them with the null beating
the real value (§3.3). For comparison, the expander result itself is narrower than it is usually
quoted: `outputs/loop_ramanujan_sparse.json` a3 is Ramanujan against **random sparse at identical
density**, −0.003404 ± 0.002383, on 24 tabular features, DEV only. The measured finding is that at
fixed density the topology was immaterial — not that biologically-named clustering would do better,
which was never tested.

**Stage 6 assigns chromatin a polymer, and the polymer scores below a distance-only null.**
`outputs/loop_genome_reselect.json`: genome-best ρ 0.8229 against the distance null 0.8283, with
0 of 108 configurations inside one genome sd on all four decay bands. This module is ~60% of the
cost model's wall time, and its boundary message has **no subscribers**, because the one forward
wire ever built from it fails its shuffled-position null at z 0.805 while publication count scores
+0.3911 on the same target. (The chr21-scored pass — ρ 0.8424 vs distance null 0.8280 — is real but
target-specific and does not license genome use.)

**The scheduler's cross-layer wake signal is unusable in either direction.** The accessibility lead
reads +47.67, +153.94 and +101.60 min for the same statistic on the same data depending on grid and
baseline (`NOTES_accessibility_clock_status.md`); four timepoints reverse its sign; and under the
only forced perturbation on disk the arrow runs the other way, −0.0840 with CI [−0.1123, −0.0547]
on 11,001 genes, from an estimator calibrated to +0.1179 on A549 *before* the answer was read.
Wake on clocks, never on a cross-layer event.

**The model cannot be validated even if built.** The persistence bar exists on one channel covering
2.3% of rows. Loop 197 queried 5,792 released human ENCODE accessibility/RNA experiments, recovered
A549 at 11 matched points as a positive control before believing any absence, and found nothing
above 4 matched points elsewhere. There is **no protein-side and no flux-side persistence bar at
all** — the entire time-resolved perturbation-protein holding is 4 surface proteins at one endpoint,
mean r² 0.2493. Four of the six modules are exempt from the admission rule only because no bar
exists for them.

**The design refutes itself, and that is the honest output.** Under its own admission rule, nothing
steps: the expression module holds, the emitters are identity by construction, the LP and the
budgets are quasi-static solves, and the chromosome module writes to a scorer with no subscribers.
The end state is the repository as it already exists, plus a type system.

**Red-team verdict: would not beat persistence.**

---

## 5. The minimum viable test

One loop. It is the smallest object that contains REM-Cell's whole claim and nothing else, and its
decisive arm has already been run once informally and failed — so the loop's job is to commit that
measurement under gates, and to run the two arms that were not run.

```python
"""Loop 201 -- can the set point be computed?

Loop 198 measured that persistence beats every dynamic rule this project knows
(-0.02953 vs -0.05205). That was read as "dynamics do not work here". This loop
separates the two halves of that claim, because they are not the same claim.

The law is first-order relaxation to a set point, which is dM/dt = k_sm - a*M
rewritten and is the ONLY law that appears in more than one subsystem here
(chromatin node 10, transcription nodes 1 and 3, proteome node 1, signalling C2
are one law over the same 4,190 genes):

    delta_m(g,j) = lam * dt_j * (S_g - m(g,j-1))

with ONE global rate lam fitted on the training window and S_g a per-gene SET
POINT. Split over TIME exactly as loop 198: train on intervals ending
60-240 min (6,680 rows), score on intervals ending 480/600/720 min (4,008 rows),
1,336 genes, A549 + dexamethasone, scratchpad/grtc/rna.npz.

One free dynamic parameter, so the law cannot win by fitting. If it wins, it wins
because the set point is right.

GATES

S1  FORM.  Relaxation to the ORACLE set point (the measured 3-replicate plateau)
    beats persistence, held out in TIME.
    statistic: held-out R2.  bar: > -0.02953.
    FAILURE MEANS: the relaxation form is wrong and REM-Cell's whole ODE class
    is refuted, not merely unparameterised. Stop.
    (Measured informally at +0.36290, lam +0.003766/min. This gate commits it.)

S2  CEILING.  The set point is a real target, not noise.
    statistic: Spearman-Brown reliability of the plateau across the 3 replicates.
    bar: implied predictor ceiling r > 0.889, the crossover from S4.
    FAILURE MEANS: the benchmark is noise-limited and cannot decide anything.
    VOID S3-S6 if S2 fails.
    (Measured informally: pairwise r 0.9483/0.9332/0.9354, SB 0.9788, ceiling
    r 0.989.)

S3  CROSSOVER.  Establish, by degrading the oracle set point with calibrated
    Gaussian noise, the set-point accuracy at which relaxation crosses
    persistence. This is a DESIGN SPEC, not a score: it converts REM-Cell's
    architecture question into a single number any future layer must reach.
    statistic: the r at which held-out R2 crosses -0.02953.
    bar: none -- this gate reports.
    (Measured informally at r = 0.889.)

S4  THE GATE THAT DECIDES REM-CELL.  A set point predicted from upstream layers,
    on gene-held-out folds, reaches the S3 crossover.
    statistic: held-out-in-TIME R2 of relaxation to the predicted set point.
    bar: > -0.02953, AND > -0.03034 (the training-mean arm), AND the set point
    must beat publication count as a predictor of the same plateau (the standing
    fame floor; pubs scores rho +0.3911 on the analogous loop 114 target).
    FAILURE MEANS: REM-Cell's propagation buys nothing on the only channel that
    can score it. The architecture is refuted on its own central claim.
    (The 9-track same-cell arm was measured informally at r +0.4104 -> R2
    -0.02977, i.e. FAIL by 0.00024, with lam collapsing to -0.000036/min. This
    gate commits that and adds the two arms below.)

S5  ALL LAYERS.  Add every remaining per-gene layer this repo holds -- the 27
    chromatin/RT/torsion features from _chromatin_features.json and
    _rt_matrix.npy, the 191-factor K562 binding block, CollecTRI in-degree, the
    sequence features behind the ESM decay emitter -- and re-measure the set-point
    r on the same folds.
    statistic: pearson r of predicted vs measured plateau.
    bar: r >= 0.889.
    FAILURE MEANS: no layer combination on this disk computes a set point. Report
    where it plateaus; a plateau below 0.6 refutes the architecture with a
    measurement rather than an argument.

S6  PER-GENE RATES.  Replace the single global lam with per-gene rates from the
    13,105 measured mRNA half-lives -- the one place REM-Cell's real parameters
    could help -- and score at the ORACLE set point.
    statistic: held-out-in-time R2, per-gene lam vs global lam.
    bar: > the S1 value.
    FAILURE MEANS: the half-life layer, the best-parameterised object in this
    repo, is decorative for dynamics too.

CONTROLS.  CTCF and RAD21 promoter tracks as inert negatives (measured
informally at r -0.0017 and +0.0504). Publication count as the fame floor. All
nulls verified capable via gate_guard.null_can_move before survival is read;
survival via gate_guard.survival(z_min=2.0), returning UNDEFINED rather than a
percentage when |z| < 2.

DATA.  All on disk, verified: scratchpad/grtc/rna.npz (3.79 MB, hash
5496b7c455b82c23 in outputs/loop_capability_4d.json manifest); scratchpad/grtc/
{NR3C1,EP300,JUN,JUNB,CEBPB,FOSL2,DNase,CTCF,RAD21}/*.bed.gz; _tss_hg38.bed;
colab/data/cell_complete.json.gz. Harness: colab/loop_capability_4d.py:194-262
and colab/loop_response_timing_d.promoter_track. The expensive promoter parse is
already cached to scratchpad/_rt_setpointX.npy and _rt_setpointy.npy.

COST.  Under an hour.

DECISION RULE, STATED BEFORE RUNNING.  If S5 reaches r >= 0.889, REM-Cell is
worth building -- and its architecture is a STATIC set-point regressor with one
relaxation constant bolted on, not a hierarchy, not global fields, not a
scheduler. If S5 plateaus below r 0.6, the architecture is refuted on its central
claim and the correct output of this whole effort is the ledger and the type
system of section 6.
"""
```

The baseline that can beat it is persistence, and on the arm already run, it does.

---

## 6. The staged plan

Only stages the red team did not kill. Each has an entry condition and an exit gate.

**Stage A — the admission ledger and type system.** *Entry:* none; blocked by nothing. *Build:*
`NOT_MEASURED` as a legal return value; `UNDEFINED` instead of a survival percentage when |z| < 2;
provenance tags making back-solved and swept parameters inadmissible for the quantity they were
solved from; a `step()` that refuses to feed an unvalidated derivative into another node; a
boundary-message type that rejects solver-dependent interiors. *Exit gate:* it retroactively flags
k_sp's circularity (3.1e-14), the k_cat bundle losing to a constant (12.95× vs 9.42×), and the
twelve survival percentages computed from two indistinguishable numbers that `colab/gate_guard.py`
already documents. *This is the only stage with no missing parameter.*

**Stage B — loop 201, the set-point test.** *Entry:* Stage A's `gate_guard` in place (it is).
*Exit gate:* S4 or S5 passes at r ≥ 0.889 → proceed to Stage C. S5 plateaus below 0.6 → stop, and
the project's answer is Stage A plus a recorded refutation.

**Stage C — a static set-point regressor with one relaxation constant.** *Entry:* Stage B passed.
*Build:* the winning layer combination as a static predictor, plus per-gene λ from the 13,105
measured half-lives. Not a hierarchy, not global fields, not a scheduler. *Exit gate:* held-out-in-
time R² > 0 on the A549 course, i.e. beating the test window's own mean and not merely persistence.

**Stage D — a second persistence bar.** *Entry:* none; this can run in parallel with B. *Build:* a
protein-side or flux-side persistence baseline on any public time course. *Exit gate:* the bar
exists and is a number. **Currently the single highest-value cheap fetch**, because four of six
modules are exempt from the admission rule solely for want of it, and because it would say whether
−0.0295 is a property of A549 or of biology.

**Stage E — metabolism as a constraint solver with two ports.** *Entry:* none; instantiable today.
*Build:* the LP perturbed only through the Boolean GPR (gene state) and the 53 medium bounds,
emitting (μ, J_ex) and a structural viability score. *Exit gate:* **blocked** — 0 of 12,931
reactions have a measured flux, so the module is solvable and unfalsifiable. **Fetchable:** one
published ¹³C-MFA dataset would make it testable. This is the highest-value fetch for that
subsystem.

**Stage F — static completion, already working.** *Entry:* none. *Build:* persist the two fitted
rankers (metabolite completion, four-block enzyme merge) — no `.pkl` exists for either, so both are
re-fit at 199 s and 57 s per cold start. Report TEST-split numbers, not DEV. *Exit gate:*
engineering only; nothing is blocked.

**Blocked and not fetchable, listed so nobody re-plans them.** ~~Regulatory edge gains (612,133, a
curation and biology gap, not a download).~~ **CORRECTED 25 August 2026 by loop 208 — this line was
wrong.** Genome-scale Perturb-seq (Replogle et al. 2022, figshare 20029387) measures exactly this
quantity: knock a gene down, read the transcriptome, and the change *is* the gain. Downloaded in
this session as 470 MB of pseudobulk — K562 11,258 × 8,248 and RPE1 2,679 × 8,749, 81,363,282
measured perturbation–response values. **360,540 of the 612,133 edges (58.9%) now carry a measured
value** (`outputs/loop_perturbseq.json` A2). It was a download, not a curation gap.

What loop 208 also measured, and it does not rescue the plan: those gains reach **r 0.2785** as a
set-point predictor against the 0.9081 requirement, losing to the nine same-cell ChIP tracks at
0.2932; and the sharpest available test failed — NR3C1 was perturbed in the dataset and
dexamethasone acts through NR3C1, yet its measured knockdown signature scores |r| 0.0513 against a
null of 1,000 *other real perturbations* whose 95th percentile is 0.1533. Gains do transfer between
cell types (same gene K562↔RPE1 median r +0.1677 against a within-line different-gene null of
+0.0327), so the failure is not a cell-type artefact. **The block was never the availability of the
gains. It is that having 360,540 of them does not move the set point.** Complex assembly rates and compartment transport rates
(do not exist at proteome scale). Per-protein degradation in two cell-cycle states (a new
experiment). Intracellular compartment-resolved metabolite concentrations (the fetchable proxy,
HMDB biofluid, is known to be the wrong object). A second densely-sampled matched time course
(loop 197 established the absence in ENCODE with a positive control). Two layers measured in the
same cells at the same time (the experiment does not exist).

**Blocked and fetchable, cheap.** An NGAM constant — one literature number, and without it the
model cannot represent a resting cell. Complex stoichiometry coefficients (Complex Portal / CORUM).
Per-cell FUCCI pseudotime (public; the three `.h5ad` files that produced the only continuous θ in
this project are no longer on disk).

---

## 7. The honest paragraph

Built today with what exists, REM-Cell would be a compartment-resolved static map of 16,492 genes
with a linear program attached: a metabolic constraint solver perturbed through two ports and
answering with one scalar and a structural viability score, an mRNA and protein layer that computes
steady states at roughly two-fold error and whose per-gene regulatory input is worth about 0.02
Spearman over a single constant, three resource budgets in which every constant is counted and
nothing is fitted, a dilution term error-bounded at 3.97%, four genuinely good static predictors,
and a scheduler whose correct action on every wake is to hold the state it already has. It would
not be a simulator. Under its own admission rule none of its 58 node types is licensed to step
forward, because on the only channel carrying a held-out-in-time bar, predicting no change scores
−0.0295 and everything this project knows scores −0.0520; it needs roughly 860,000 rate constants
and about 27,000 exist, 71% of the shortfall being the regulatory gains its cross-layer propagation
would run through; four of the seven links in that propagation chain are measured at or below their
own nulls; its global fields are rank-1 couplings whose writers were shown to be interchangeable in
both cases where anyone checked; and its chromosome module, which would consume most of the compute,
scores below a distance-only null and has nothing licensed to read its output. The one new positive
is worth more than the architecture: relaxation to the true set point, with a single global rate,
recovers about 70% of all reproducible signal on the one course that can be scored — so the
equation REM-Cell wants is right, and what is missing is not dynamics but a per-gene set point that
this repo, using nine layers measured in the same cells, can predict at r 0.41 against a
requirement of 0.89.
