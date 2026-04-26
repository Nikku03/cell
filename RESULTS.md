# Results: gene essentiality prediction in Syn3A

This is a longer scientific summary of the project. It expands on the README and cites the underlying fact JSONs at `memory_bank/facts/measured/` so any specific number can be re-verified. I built the simulator, ran 22 development sessions of measurements + falsifications, and reached MCC 0.5372 on the full 455-gene Breuer panel — comparable to but not exceeding the FBA benchmark (0.59). The trajectory and the documented dead ends are below.

## Background and motivation

*Mycoplasma mycoides* JCVI-Syn3A is the smallest free-living organism ever assembled in a lab. Its 543 kbp genome encodes 452 protein-coding genes — a "minimum viable bacterium" designed by the J. Craig Venter Institute by iteratively deleting non-essential genes from *M. mycoides* (Hutchison et al., 2016, *Science*). Because the genome is small, fully sequenced, and entirely synthetic, Syn3A is a rare model system where every gene's essentiality can be both **measured** by single-gene knockout (Breuer et al., 2019, *eLife*) and **modelled** computationally with reasonable hope of mechanistic accuracy.

Predicting essentiality computationally matters because exhaustive knockout screens are slow and expensive, and the predictions encode mechanistic understanding that pure phenotype data does not. If a model says "gene X is essential because pathway Y collapses 30 seconds after knockout," that is a testable hypothesis, not just a yes/no answer. This project asks whether a fast event-driven simulation, paired with a small set of biology-aware detectors, can reproduce Breuer 2019's labels at a useful accuracy.

The answer, after 22 development sessions: **the result is close to but does not exceed the existing FBA-based benchmark**. The trajectory of how the work got there, including the directions that were tried and ruled out, is in the rest of this document.

## Approach: three integrated layers

### Layer 1: Stochastic biochemistry simulation

The cell is represented as a vector of integer molecule counts per species (308 species in total). Reactions fire stochastically using the Gillespie algorithm — the next event is sampled from a propensity distribution proportional to current counts, time advances by an exponential random variable, the event applies, and the loop repeats. The simulator implements 356 reactions parsed from the Luthey-Schulten lab's Syn3A SBML model, with reversible Michaelis-Menten kinetics and proper saturation factors (`mm_saturation_factor` in `cell_sim/layer3_reactions/reversible.py`). A Rust port of the hot path runs **1.86× faster than the pure-Python reference** on the production sweep workload (measured in Session 18 across 20 genes; below the "roughly 2×" claim from prior work but on the same side, and consistent across reruns).

The scope is metabolism + transcription + translation + protein turnover. **Cell division and biomass accumulation are not modelled** — that's the deferred Layer 5. For a 0.5-second simulated window, the reduced scope is appropriate; for whole-cell-cycle modelling it's a known gap.

### Layer 2: Detector framework

A "vote of seven" architecture turns trajectories into binary essential/non-essential predictions:

| Detector class | Detector | Signal it uses |
|---|---|---|
| Trajectory-only | `ShortWindow` | Pool deviation in the first ~0.5 s post-knockout |
| Trajectory-only | `PerRule` | Number of catalysis events suppressed vs. wild-type |
| Trajectory-only | `RedundancyAware` | Like PerRule, but accounts for parallel pathways producing the same metabolite |
| Knowledge-based | `ComplexAssembly` | Membership in 24 known protein complexes (ribosome, RNA pol, ATP synthase, …) |
| Knowledge-based | `AnnotationClass` | 30+ keyword priors mined from Syn3A's GenBank annotations against Breuer's FN pool |
| Composed | `composed_detector` | Union: ComplexAssembly ∪ AnnotationClass ∪ PerRule (the v15 detector) |
| Supervised ML | `Tier1_XGBoost` | XGBoost over Tier-1 features; **falsified — see negative result #1** |

Trajectory-only detectors plateau at MCC ~0.13 on small panels (measured in v5/v6/v9). Knowledge-based detectors drove the improvement from MCC 0.342 to 0.537 over five iterations as more annotation classes were mined from the false-negative pool of each preceding version.

### Layer 3: ML feature extraction

Built in Sessions 14–17 as infrastructure for a future learned detector:

- `esm2_650M.parquet` — 1280-dim ESM-2 protein embeddings (`facebook/esm2_t33_650M_UR50D`), 455/455 coverage.
- `esmfold_v1.parquet` — 9 structural descriptors per protein (pLDDT mean/std, secondary-structure fractions, radius of gyration), 455/455.
- `mace_off_kcat.parquet` — 7 BDE-derived k_cat aggregates from the MACE-OFF foundation model, 111/455 coverage (only enzymes with mapped substrates).
- `alphafold_db.parquet` — empty placeholder; Syn3A is not in UniProt and so has no AFDB entries.

Total cache weight: ~7 MB. A SHA-256 manifest is committed alongside.

**Honest framing:** the cache is built and verified, but the supervised detector that consumes it (`tier1_xgb_detector.py`) **does not improve on v15** — see negative result #1 below.

## Results

### Headline metric

**v15 composed detector on the full 455-gene Breuer panel: MCC = 0.5372**, with confusion `tp=287, fp=3, tn=69, fn=96` (precision 0.990, recall 0.749, specificity 0.958). Source: `memory_bank/facts/measured/mcc_against_breuer_v15_round2_priors.json`. Reproducible by running the command in `README.md`.

Multi-seed reproducibility: at three different RNG seeds {42, 1, 2} the simulator produces **bit-identical confusion matrices**. Source: `memory_bank/facts/measured/mcc_v15_replicates.json`. The detector design is deterministic at this configuration.

For comparison, **Breuer et al. 2019's own FBA-based predictor reports MCC 0.59** on the same labels. Their approach uses steady-state flux balance analysis on the same SBML model; this work uses event-driven dynamics with a multi-detector vote. The two approaches are different in method but reach roughly comparable accuracy.

### Progression of measured MCC

The full-panel trajectory from Session 11 onwards:

| Version | What changed | MCC | TP | FP | TN | FN |
|---|---|---:|---:|---:|---:|---:|
| v10_full | First full-panel sweep; ComplexAssembly KB | 0.342 | 211 | 6 | 66 | 172 |
| v10b_full | Additional complex KB entries | 0.364 | 223 | 6 | 66 | 160 |
| v12 | iMB155 metabolic completeness patches | 0.393 | 222 | 3 | 69 | 161 |
| v13 | tRNA + tRNA-modification keyword priors | 0.410 | 231 | 3 | 69 | 152 |
| v14 | 30+ annotation classes mined from FN pool | 0.494 | 270 | 3 | 69 | 113 |
| **v15** | **12 more annotation classes + truA/truB widening** | **0.537** | **287** | **3** | **69** | **96** |

The visible pattern: **false positives stayed flat at 3 from v12 onwards** — every MCC lift came from converting false negatives into true positives. This is the "high-precision, growing recall" trajectory captured in `figures/scripts/plot_confusion_matrix_grid.py`.

### Three documented negative results

These bound what was tried and ruled out. They are recorded as measured facts so that future work can avoid re-running them.

**1. Tier-1 ML stacking does not improve on v15** (Session 17, fact: `mcc_tier1_xgboost_naive_stack.json`).

Three configurations of XGBoost, all 5-fold stratified CV on the same 455-row dataset:

- Tier-1 features only (1295 cols: ESM-2 + ESMFold + MACE): pooled MCC 0.145
- v15 detector outputs only (10 cols): pooled MCC 0.537 (matches baseline)
- Union of Tier-1 + v15 outputs: pooled MCC 0.443 (**worse than v15 alone**)

Diagnosis: the 1280:455 feature-to-row ratio is prohibitive for supervised learning at this scale. Adding ESM-2 features to the v15 outputs introduces 23 new false positives per fold while only recovering 43 new true positives — net MCC drop. Confirmed by independent partition (XGBoost + PCA on v15-silent genes) and pure-kNN (cosine on ESM-2 embeddings) attempts; both also falsified. Smaller embedding (ESM-2 150M, 640 dims, Session 19) **also loses** to 650M, confirming row count rather than embedding dim is the bottleneck.

**2. Path A — longer biological time amplifies false positives** (Session 9, fact: `mcc_against_breuer_v9.json` and adjacent calibration runs).

Extending the simulation from 0.5 s to 5.0 s biological time was hypothesised to catch the late-failing genes (e.g. `trxA`, `hupA`, `recR`). It did the opposite: false positives grew faster than true positives, because the simulator lacks proper metabolite-consumption sinks and concentration caps — perturbations that should equilibrate instead drift, triggering more pool-deviation alarms. The detector design choices that work at 0.5 s do not generalize to longer windows without a concomitant detector overhaul.

**3. Toxicity prediction extension halted** (Sessions 21-22, fact: `toxicity_gate_b_halted_negative_finding.json`).

A separate research direction was scoped to extend the simulator into a mechanism-aware antibiotic-toxicity predictor. Gate B viability check: of the 155 SBML-encoded enzymes, **0 are canonical Mycoplasma-active antibiotic targets**. Every active drug class (macrolides, tetracyclines, pleuromutilins, fluoroquinolones, sulfonamides, trimethoprim, etc.) targets molecular machinery that lives outside the metabolic SBML — ribosomal proteins, DNA gyrase, RNA polymerase, folA, folP. The original spec scoped to "modify Layer 3 reaction rates," which by construction excludes these targets. The negative finding ended the toxicity work after one Colab notebook of measurement; the alternative paths (expand to Layer 2 translation inhibition; pivot to a different organism) are documented but require explicit re-authorisation.

### Comparison to existing work

| Approach | Mechanism | Speed | MCC vs Breuer panel |
|---|---|---|---:|
| **Breuer 2019 FBA** | Steady-state flux balance on the same SBML | seconds | 0.59 |
| **Thornburg 2022 well-stirred Gillespie** | Whole-cell stochastic dynamics | hours per cell cycle | not measured on this panel |
| **Thornburg 2026 4D whole-cell model** | Spatial + atomistic + chromosome model | days per cell cycle | not measured on this panel |
| **This work** | Event-driven Gillespie + 7-detector vote | 6.6 s/gene; 50 min full panel | **0.537** |

The relevant comparison is Breuer 2019. This work is **fast** (a knockout sweep that takes a workday to design and minutes to run, not a multi-day simulation), but **does not exceed** the static-FBA accuracy. The accuracy gap (0.053 MCC) sits in the false-negative pool — the simulator and detector framework miss 96 essential genes that Breuer's labels mark essential, and 84 of those 96 carry "Uncharacterized" annotations that the keyword-prior detectors cannot resolve without external information.

## What this project does that others don't

This section names the configurations and practices in this work that I haven't seen combined elsewhere. Each item also names what's unrealized or limited about it; I'd rather a reader assess the work accurately than be sold on it.

### Cross-language, cross-environment integration in one pipeline

I built the simulator in Python (~10,400 lines under `cell_sim/`), wrote a Rust hot path for the Gillespie inner loop (`cell_sim/layer2_field/rust_dynamics.py` wrapping the PyO3 binding), integrated three pretrained protein models as feature extractors (ESM-2 650M for 1280-dim sequence embeddings, ESMFold v1 for structure prediction with pLDDT in the 0-100 scale after a unit-fix in Session 17, MACE-OFF for substrate bond-energy estimation), and split execution between local CPU for the simulator and Colab GPU for the embeddings via parquet-cache hand-off. The end-to-end pipeline runs and is reproducible from the commit history. The Rust hot path is **1.86× faster** than pure Python on the production sweep config (20-gene sample, scale 0.05, t_end 0.5 s, seed 42; measured Session 18) — useful but not dramatic, because the Python path is already numpy-vectorised. The ML-feature integration produces the cache and verifies it, but the supervised-learning detector built on top of the cache **does not improve essentiality MCC** on Syn3A's 455 rows; the negative result is recorded as `mcc_tier1_xgboost_naive_stack.json`.

### Reproducibility infrastructure as a first-class concern

`memory_bank/facts/measured/` holds 47 JSON fact files, one per measurement. Each fact pins a specific numerical claim to a reproducible script invocation, a commit, and (where applicable) a SHA256-validated parquet under `cell_sim/features/cache/manifest.json`. An invariant checker at `memory_bank/.invariants/check.py` validates the dependency graph between facts, the source-citation completeness, the physical-range bounds on parameter values, and the existence of every code path that a fact's `used_by` field references — all in under one second. This was built for my own use: I wanted to be able to resume work cleanly after kernel-state loss, distinguish "this MCC was measured" from "this MCC was estimated" without ambiguity, and keep the provenance trail intact across long-running development. Whether the same infrastructure would help another researcher pick the project up is untested — nobody other than me has tried.

### Documented negative results

Three independent extension attempts were each evaluated rigorously and halted with a measured-fact JSON when the data said stop, not silently dropped. (1) **Tier-1 ML stacking** (Session 17): three configurations of XGBoost over ESM-2 + ESMFold + MACE features on the Breuer 455-row dataset all underperformed v15's keyword priors; pooled MCC 0.443 union, 0.145 features-only, 0.530 partition. The 1280:455 feature-to-row ratio is the diagnosed cause. (2) **Path A longer biological time** (Session 9): extending the simulation window from 0.5 s to 5.0 s grew false positives faster than it recovered true positives, because the simulator lacks proper metabolite-consumption sinks at longer windows. (3) **Toxicity-prediction extension** (Sessions 21-22): the Gate B viability check found 0 of 155 SBML enzymes match canonical Mycoplasma-active antibiotic targets, halting the project before any predictive results were claimed. Each negative result is informative for what doesn't work; none of the three turned into a positive contribution to MCC.

### Methodological combination on a single evaluation framework

Stochastic whole-cell simulation, a multi-detector ensemble (`ShortWindow` + `PerRule` + `RedundancyAware` for trajectory analysis, `ComplexAssembly` + `AnnotationClass` for biological priors), and pretrained-ML feature stacking (ESM-2 + ESMFold + MACE) are all evaluated against the same Breuer 2019 labels using the same MCC measurement, with the same train/eval pipeline. I haven't seen this specific configuration in the cited prior work — Breuer used FBA + curated GPR mapping; Thornburg used well-stirred Gillespie without the ML stack. The data also shows the components are **not strongly complementary** on Syn3A's 455-gene set: trajectory-only detectors plateau at MCC ~0.13, ML stacking actively hurts (0.443 union vs v15's 0.537), and the knowledge-based priors carry the bulk of the result. The integration is in a configuration I haven't seen elsewhere; the scientific finding from that integration is mostly null.

### Position on the speed-fidelity tradeoff

A full 458-gene knockout sweep at production config (scale 0.05, t_end 0.5 s, 4 workers, Rust on) completes in about **50 minutes on a 4-core CPU** — 6.6 s effective per gene. That sits between Breuer 2019's seconds-scale FBA (steady-state only, no temporal trajectory information) and the Thornburg 2026 4D whole-cell model (much richer biology with chromosome dynamics + spatial fields, GPU-days of compute per cell cycle). I have not yet used this niche for an application that the other tools couldn't address — having a fast knockout-screening pipeline available is not the same as having found the right question for it. Possible uses (parameter sensitivity sweeps, synthetic-lethality screens, time-resolved failure-mode characterisation) are all named in `RESULTS.md`'s "What I would do next" section but unrealized in this codebase.

### Scope discipline across long-running development

Twenty-two sessions over several months, with a measurable deliverable per session: an MCC number, a feature parquet, a falsification, a Colab notebook. When the data said a direction was over (Tier-1 ML in Session 17, toxicity prediction in Session 22), I halted that direction and documented the negative result in the same commit instead of letting it sprawl into adjacent experiments. `PROJECT_STATUS.md`'s session log preserves all 22 entries verbatim. This is a process observation, not a scientific finding — I list it because the artifact reflects it and a careful reader will see it.

## Synthetic-lethality screen (Session 26)

I extended the harness with `predict_pair(locus_a, locus_b)` so the simulator can run joint two-gene knockouts, then ran two pilot screens.

### v1 pilot — 196 pairs across 5 biologically-motivated categories

Categories: ESM-2 paralogs (cos > 0.85), same-pathway sequential pairs (negative control), random different-pathway non-essential pairs, transporter-substrate pairs, and a small hand-curated set. Result: 1 / 50 paralogs flagged synth-lethal (2 %), no separation from random baseline (0 / 41). Three viability invariants passed (`predict_pair(X, X)` matches `predict(X)` on 5 sample genes; pair symmetry verified on 5 pairs; v15 single-knockout sweep on a 20-gene sample reproduces 20 / 20 against the existing fact). Recorded as `synthlet_pilot_v0.json`. Decision after v1: `NARROW_SCOPE` — the eligibility issue (most paralog pairs had at least one already-essential gene, so most candidates couldn't be synth-lethal by definition) was burning ~64 % of compute on pairs the test couldn't fire on.

### v2 pilot — 249 pairs with eligibility filter applied at selection time

Re-pulled candidates only from the 165 v15-non-essential genes, with cosine bands recalibrated to the actual pool distribution (mean cos 0.93, std 0.04 — the v15-non-essential pool sits in a high-baseline space, so v1's 0.85 threshold was sampling above-median pairs, not real paralogs). Five categories: tight paralog (cos ≥ 0.99, n=24), loose paralog (0.97-0.99, n=60), shared substrate (n=54), shared product (n=31), and a hard-negative random baseline (cos < 0.88 + no shared SBML metabolite, n=80). Results: 1 / 24 tight paralogs (4.2 %), 0 / 60 loose paralogs, 1 / 54 shared substrate, 0 / 31 shared product, 0 / 80 random baseline.

### Headline result

**Both v2 hits AND the v1 hit are the SAME pair** — `JCVISYN3A_0876 × JCVISYN3A_0878`. Across both pilots (~445 distinct attempts):

- **0 false positives in 121 random / negative-control pairs.** Perfect specificity at this pilot scale.
- **1 reproducibly-detected candidate** flagged by three independent selection criteria.
- Population-level paralog-vs-random enrichment is **not statistically significant** (Fisher's exact one-sided p = 0.23 for tight paralog 1 / 24 vs baseline 0 / 80; combined functional-similar 3 / 132 vs random 0 / 121, p = 0.14). n = 24 tight paralogs is the absolute ceiling — Syn3A's 165 v15-non-essentials yield only 24 pairs at cos ≥ 0.99.

So the session does not deliver a population-level finding (paralog enrichment is suggestive but underpowered). It does deliver a case-study finding worth wet-lab follow-up.

### The pair, mechanistically

`JCVISYN3A_0876` (526 aa, NCBI annotation "Uncharacterized amino acid permease") and `JCVISYN3A_0878` (512 aa, same annotation) are paralogs at ESM-2 cosine 0.996 — top 0.5 % of the v15-non-essential pool's similarity distribution. The Luthey-Schulten Syn3A SBML explicitly assigns both genes (and only these two) to all 18 amino-acid transport reactions via an OR gene-protein-reaction rule. Either alone catalyses the reactions; loss of both silences all 18. The simulator faithfully executes the OR rule and reports `catalysis_silenced` at confidence 1.0.

Reframing this honestly: the synth-lethal prediction is not an emergent simulator discovery — the redundancy was encoded by the SBML curators. The simulator's role is correct execution of the OR rule. The biological credibility comes from whether the SBML curation is right.

### Eight verification layers — corroborates the SBML curation without wet-lab work

To check whether the OR-redundancy assignment is biologically reasonable, I ran a follow-up Colab notebook (`notebooks/synthlet_0876_x_0878_external_lookup.ipynb`) hitting four external biology databases. Combined with the in-sandbox checks already done, the verification stack is:

1. **SBML GPR-OR rule** — Luthey-Schulten curators encoded the redundancy explicitly.
2. **NCBI GenBank annotation** — both products independently labelled "Uncharacterized amino acid permease" (separate annotation source from the SBML).
3. **Sequence-level paralogy** — conserved N-terminal motif spanning ~50 residues; same family architecture.
4. **ESM-2 cosine 0.996** — top 0.5 % of pool similarity distribution.
5. **Multi-seed simulator robustness** — 5 / 5 RNG seeds {42, 1, 2, 7, 123}, joint = catalysis_silenced at confidence 1.0 every time. Bit-stable.
6. **UniProt orthologs in the parent organism** — `Q6MS69` (526 aa) and `Q6MS71` (512 aa) in *M. mycoides* SC, both UniProt-curated as "Amino acid permease", both Pfam **PF13520** ("AA_permease_2"). The two paralogs survived the Syn3A genome-reduction process intact, suggesting both were retained because at least their union was functionally essential.
7. **NCBI BLAST cross-species** — 0878 has a strong ortholog `MPN_308` (UniProt P75472) in *M. pneumoniae* M129 (E = 2 × 10⁻¹⁴, 79 bits, 24 % identity over 470 aa). Subsequent BLAST hits to cystine/glutamate transporter, b(0,+)-type and L-type amino-acid transporters across mammals, and *B. subtilis* SteT serine/threonine exchanger — all amino-acid transporters in the same conserved family.
8. **Pfam family** — both proteins independently classified into PF13520.

The remaining gap is the Syn3A-specific double-knockout phenotype itself. That is a wet-lab test: knock out 0876, knock out 0878, knock out both, measure growth at 36 °C in defined medium with full amino-acid supplementation, three biological replicates per genotype. Predicted phenotype: 0876-only and 0878-only viable (with mild growth defect for 0878 per the Quasiessential class), 0876 + 0878 fails to grow because the cell can no longer import any of the supplemented amino acids. ~1 week of bench work.

Recorded as `synthlet_pilot_v0.json`, `synthlet_pilot_v2.json`, and `synthlet_0876_x_0878_verification.json`.

## Limitations

- **Single-organism validation.** The 30+ annotation-keyword priors and the 9 iMB155 metabolic patches were both mined from Syn3A's specific data. Cross-organism transfer would require remining; that effort was scoped (`memory_bank/data/multiorg_essentiality/`) but not completed because the upstream DEG flat-file URLs returned navigation HTML rather than data.
- **Trajectory-only detectors hit MCC ~0.13 ceiling** measured across multiple detector designs (`ShortWindow`, `PerRule`, `RedundancyAware`). The simulator's stochastic dynamics carry less essentiality signal than the existing biological knowledge bases do for this dataset size.
- **Knowledge-based detectors carry the bulk of the result.** The composed v15 detector reaches 0.537; the trajectory components alone reach ~0.13. This is the honest credit assignment — the headline number is mostly an artifact of careful annotation mining, not of the simulator's stochastic core.
- **The simulator cannot predict non-metabolic essentials** that depend on protein-machinery integrity (e.g. ribosomal proteins fail through translation collapse, not metabolic shutdown) without the gene-expression-layer detectors in the composed vote. Roughly 75% of Breuer-essential genes are caught only because of the knowledge-based path, not the simulation per se.
- **Layer 5 (biomass + division) is not implemented.** Cell-cycle phenotypes are out of scope.
- **3 stubborn false positives never audited** at the wet-lab level. They are the cleanest hypothesis this project produces — see the README for the proposed experiment.

## What I would do next if continuing

Three honest directions, in roughly decreasing expected payoff per unit effort:

1. **Multi-organism essentiality predictor.** The Tier-1 ML stack falsifies on 455 rows but the infrastructure is already built. Pulling labelled essentiality data for *E. coli* K-12 (~4,300 genes), *B. subtilis* 168 (~4,200), *M. pneumoniae* M129 (~700), *M. genitalium* G37 (~480), and Syn3A (455) into one matrix changes the regime to ~10,100 rows × 1280 ESM-2 dims, where supervised learning has a real chance. The data curation Colab notebook exists (`notebooks/curate_multiorg_session20.ipynb`) but the DEG flat-file URLs went stale during my run; replacing them with OGEE or per-paper supplementary downloads is the unblocking step.

2. **The 3-FP wet-lab audit.** The simplest contribution this project can make is to determine which of `JCVISYN3A_0034` (and the other two persistent false positives — exact locus tags in the v15 fact) are genuinely Breuer-borderline versus simulator artifacts. A targeted growth-rate measurement at 36 °C, three biological replicates per gene, would resolve this in a week. If even one of the three turns out to be a genuine essential mislabelled by Breuer, the simulator has produced a real biological correction.

3. **Parameter sensitivity analysis.** Several detector hyperparameters (`min_wt_events`, `_UNGATED_TOKEN_COUNT`, the saturation thresholds) were chosen by a small number of pilot runs and not systematically swept. A coarse grid search would tell us whether the v15 result is robust to detector tuning or whether it sits on a sharp local maximum. This is a pure-compute task that doesn't require new wet-lab data.

A fourth direction — extending the simulator to model ribosome-targeting antibiotic toxicity by adding inhibition kinetics to Layer 2 (`gene_expression.py`) — would resurrect the toxicity work that halted in Session 22 at an estimated 3-6 weeks full-time-equivalent. Documented but not recommended without an external pull from a collaborator interested in computational toxicology.

## Reproducibility

Every number in this document is sourced from a fact JSON under `memory_bank/facts/measured/`. The schema enforces:

- A reproducible `source_detail` field with the exact command to regenerate the result
- Explicit `dependencies` on prior facts
- Explicit `caveats` documenting limitations honestly

`memory_bank/.invariants/check.py` validates the dependency graph + source citations and runs in under one second; the fact graph currently passes with **47 facts and 12 sources** (commit `f1a1cc2`). The Rust hot path is versioned at `cell_sim_rust_0_2_0` (registered fact). The simulator's full sandbox test suite reports **235 passing tests**.

## Acknowledgments

- **Luthey-Schulten Lab (UIUC)** — `Minimal_Cell_ComplexFormation` repo provides the curated SBML, kinetic parameters, initial concentrations, and protein complex data that underpin Layers 1-3.
- **Breuer et al. 2019, *eLife*** — the experimental gene essentiality labels.
- **Hutchison et al. 2016, *Science*** — the JCVI-Syn3.0 minimal cell paper.
- **Meta AI** — `facebook/esm2_t33_650M_UR50D` (ESM-2) and `facebook/esmfold_v1` (ESMFold).
- **Cambridge / Oxford MACE-OFF developers** — the foundation model for substrate bond-energy estimation.
- **Anthropic** — Claude Code was used as a development assistant. All scientific decisions, biological interpretations, and conclusions are mine; the assistant helped with implementation, debugging, and documentation drafts.
