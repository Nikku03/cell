# Cellformer v4 — a multicomponent, multimodal cellular-state transformer

This is the design specification. It supersedes `CELLFORMER.md`, `CELLFORMER2.md` and `CELLFORMER3.md`,
none of which described the architecture that was actually intended: v1 and v2 were AlphaFold-shaped and
v3 documented only the six blocks that happened to get *tested*, which made a partially-built design read
as a complete one.

The model is no longer "a transformer over a gene network". It is trained in stages to learn three
separate things:

    what biological entities are
    what state a cell is currently in
    how that state changes after an intervention

**Two of this document's own admission gates have already been run**, and their results are recorded
inline rather than left as open questions. See §16 and §18.

---

## 1. The prediction problem

A human cell is a **partially observed state**:

    S[c,t] = { RNA, protein, chromatin, PTM, complexes, metabolites, morphology, environment }

and the model predicts

    p( S[c, t+Δ] | S[c,t], P, E, Δt )

with `c` the cell context, `P` a genetic/chemical/environmental perturbation, `E` the extracellular
environment, `Δt` elapsed time.

The important change: **the model predicts a distribution over possible future states, not one complete
deterministic cell.** Every output must declare itself as one of:

| label | meaning |
|---|---|
| directly supervised | a measurement of this modality existed for this condition |
| cross-modally inferred | reconstructed from another modality |
| mechanistically constrained | produced by a solver/constraint layer, not learned |
| unsupported / out of domain | the model is extrapolating and says so |

---

## 2. The data representation — six canonical tables

**Do not feed `cell_complete.json.gz` to the transformer.** Convert the project into six tables.

**A. `entities`** — one record per biological object.
`entity_id, entity_type, ensembl_gene_id, transcript_id, uniprot_id, metabolite_id, reaction_id,
genomic_coordinates, sequence_embedding, static_annotations`

Entity types: gene, transcript, protein, enhancer, PTM site, complex, metabolite, reaction, compartment,
drug, phenotype. The existing `entity_registry` (172,539 typed entities, declared-rate joins) is the backbone.

**B. `relations`** — one row per possible biological relation.
`source_id, target_id, relation_type, direction, sign, strength, confidence, evidence_type, cell_context,
provenance, validation_status`

Relation types: regulatory, PPI, complex membership, kinase–substrate, enhancer–gene, enzyme–reaction,
reaction–metabolite, ligand–receptor, codependency, coexpression. **They must never be collapsed into one
undifferentiated edge list.**

**C. `experiments`** — `experiment_id, cell_type, cell_line, donor, study, perturbation, perturbation_type,
dose, time, medium, environment, assay, batch`. This prevents cell identity, study and assay from being
silently fused.

**D. `measurements`** — long form. `experiment_id, entity_id, modality, value, uncertainty, observed_mask,
measurement_power, processing_method`.

**A missing value must never become zero.** At minimum distinguish: measured zero · below detection ·
not measured · censored · failed join · inferred.

> This table is the direct response to the largest error in the project's history. The `nlz_*` benches
> stored a median of 250 genes per knockout out of 7,223 — 349,999 recorded values in a 10,112,200-cell
> matrix — and `eval_harness` consumed the unrecorded 96.5% as real zeros. Every transformer in the arc
> trained against that. `observed_mask` exists so it cannot recur.

**E. `states`** — a state packet grouping measurements from one biological context.
`state_id, cell_context, time, RNA_state, protein_state, chromatin_state, PTM_state, metabolite_state,
morphology_state, environment`

**F. `contradictions`** — machine-readable, not prose.
`claim_id, relation_or_feature, status, failed_test, applicability, do_not_use_for`

This prevents rejected claims from re-entering through later preprocessing. The registry currently holds
18 contradictions and 8 method errors; as prose it cannot gate a training pipeline.

---

## 3. Architecture — eight blocks

    Universal entity knowledge
              │
              ▼
    Baseline cell-state measurements
              │
              ▼
    Context-conditioned entity tokens
              │
              ▼
    Real biological partner-set attention
              │
              ▼
    Perturbation + dose + time + environment
              │
              ▼
    Shared biological response  μ
              +
    Cell-specific residual      δ
              │
              ▼
    Latent future cell state
              │
     ┌────────┼────────┬─────────┬──────────┐
    RNA    Protein   ATAC/PTM  Metabolism  Morphology

1. Entity foundation encoder
2. Cell-state encoder
3. Biological partner-set encoder
4. Perturbation and environment encoder
5. Shared transition model
6. Context-residual transition model
7. Mechanistic modules
8. Modality-specific observation heads

---

## 4. Block 1 — Universal entity foundation encoder

Learns what every entity *is*, independently of any one cell.

    u_i = f_entity( identity, sequence, function, relations, structure, pathways )

**Inputs from this project:** gene identity; GO; pathway membership; protein sequence and structural
annotations; `darkfn` **only as predicted evidence**; complexes; PPI; regulatory edges; reaction
assignments; enhancer annotations; PTM sites; localization priors.

**Public dataset roles**

| dataset | role |
|---|---|
| Human Cell Atlas | which genes and programmes co-occur across many normal and diseased human contexts; tens of millions of cells, hundreds of projects. Observational, not perturbational |
| OpenCell | aligns protein identity with localization, experimentally observed interaction partners, and imaging-derived architecture, in a common cellular context |
| HumanGEM | enzyme identity, reaction participation, metabolite transformations, compartment-specific metabolic roles. **Constraints, not response labels** |

**Pretraining objectives**

- *Masked entity-feature prediction* — hide GO terms, pathways, complex membership, localization,
  reaction role, selected neighbours; predict from the rest.
- *Cross-source agreement* — one representation must reconstruct consistent information from sequence,
  interaction partners, expression programmes, perturbation signatures and metabolism.
- *Evidence-aware relation reconstruction* — predict whether a relation is curated-only, observational,
  perturbationally supported, or contradicted. **Do not train all relations as equally true.**

Output: `u_i ∈ R^d_u`, shared across every cell.

---

## 5. Block 2 — Cell-state encoder

Block 1 says what a gene *can be*. This says what it *is currently doing*.

    s[i,c] = [ RNA, Protein, ATAC, PTM, Localization, MetabolicState, MeasurementStatus ]
    h⁰[i,c] = u_i + W_s·s[i,c] + W_m·m[i,c]

`m[i,c]` encodes measured · inferred · missing · borrowed prior · uncertainty.

**Dataset roles.** The validated four-line baseline RNA (K562, RPE1, HepG2, Jurkat — audited as four
distinct states, pairwise r 0.805–0.884) is the initial context test. HCA pretrains the state encoder so
it does not simply learn four cell-line identities. Multiome Perturb-seq aligns transcription with
chromatin in the same experiment. Cell-line proteomics and OpenCell teach when RNA is a poor proxy for
protein — measured here at **mRNA explaining 15.9%** of protein variance. The genome-wide morphology
atlas (>20,000 genes, >30M cells) constrains latent state.

**Objectives.** Masked modality reconstruction (RNA+ATAC → protein priors; RNA+morphology → pathway state;
ATAC+protein → RNA), each reconstructed value still labelled *inferred*. Cross-modal contrastive alignment.
Batch/study adversarial loss — **handled carefully, because some cell lines occur in only one study, making
study and biology partly inseparable.**

Output: global `z_c` and local `h[i,c]`.

---

## 6. Block 3 — Biological partner-set encoder

**This block preserves the only robust transformer result the project has.** Measured:

| finding | number |
|---|---|
| real partners add substantial information | **+0.0101** over self-only, dense target |
| random partners *corrupt* the embedding | **−0.0085** below self-only |
| relation-specific attention bias | +0.0034, **not detected** |
| internal wiring vs shuffled wiring, same density | +0.005 ± 0.008, **not detected** |

Therefore it is a **typed biological partner-set encoder, not a causal graph simulator.**

    N(p) = N_codep ∪ N_complex ∪ N_PPI ∪ N_causal-reg ∪ N_metabolic
    n[j,c] = [ u_j, s[j,c], relation_type, evidence_strength ]
    z_partner[p,c] = SetTransformer({ n[j,c] : j ∈ N(p) })

Permutation-invariant set attention. Relation sources: `causal_reg`, `bound_causal`, `reliable_edges`,
complexes, PPI, codependency, HumanGEM enzyme/reaction neighbours, calibrated E–G partners.

**What stays out:** the sign channel (unless dense-target testing revives it — not detected four times);
relation-specific bias; message-passing topology; shuffled or low-evidence relations; rejected competition
features.

**Required controls, every training stage:**

1. real partners · 2. degree-matched random partners · 3. shuffled identities · 4. wrong-perturbation
partners · 5. self-only · 6. **simple weighted mean** · 7. **Deep Sets** · 8. **Set Transformer**

> **The transformer remains only if it beats simpler set encoders.** Arms 6–8 have never been run, and
> this is now the highest-priority unrun test in the whole programme. Block B's existing result — real
> wiring indistinguishable from shuffled wiring at the same density — predicts that a mean-pool baseline
> may match the attention layer. If it does, the validated object is a *bag of typed partners* and the
> attention is not earning its parameters. See §18.

---

## 7. Block 4 — Perturbation, time and environment encoder

A perturbation is not a target-gene ID.

    q_P = [ u_target, intervention_type, strength, dose, efficiency, onset, duration ]
    e_E = [ medium, nutrients, oxygen, serum, drug, dose, cell_density, stimulus ]
    τ(Δt)   — continuous time embedding, not categorical labels

**Dataset roles.** The corrected dense tensor `Y[c,p,g]` is the main supervised genetic-response target
for the four shared lines. Replogle K562/RPE1 for genome-scale genetic programmes. LINCS for many cell
contexts × small molecules × doses × times (with L1000's assay limitations declared). Sci-Plex for
single-cell chemical dose-response. Frangieh for perturbation × environmental condition. PerturbSci-Kinetics
for **real RNA synthesis and degradation rates rather than pseudotime**.

---

## 8. Block 5 — Shared biological transition model

    Y[c,p,g,t] = μ[p,g,t] + δ[c,p,g,t] + η[study,g] + ε

The shared branch predicts `μ̂[p,g,t]` from perturbation identity, universal entity embeddings, the real
partner set, perturbation type and time. **It does not receive cell identity as its main source of signal.**
It answers: *what generally happens when this molecular programme is perturbed?*

The validated relational-attention result belongs here.

**Losses**

- **Explicit tide removal** — `Ŷ[p,g] = tide_g + R̂[p,g]`, so predicting the per-gene marginal earns no
  credit. This construction is validated: it is what prevented the collapse `neural_ko` documented, where
  a regression net landed *below* the tide floor.
- Response probability, against the per-gene response marginal.
- Direction, against the per-gene direction marginal.
- Ranking — perturbation-specific responders above generally responsive genes.
- Effect distribution — **heavy-tailed likelihood, not plain MSE.**
- Measurement model — uncertainty conditioned on cell count, baseline expression, guide efficiency,
  sequencing depth, measurement power. **Measurement power influences the likelihood; it is never a
  biological explanatory input.** (Measured reliability ceiling: R² 0.477.)

---

## 9. Block 6 — Context-residual transition model

**The most important new block.** Train and freeze the shared branch, then

    δ[c,p,g] = Y[c,p,g] − μ̂[p,g]^(train-only)
    δ̂[c,p,g] = f_context( z_c, h[p,c], h[g,c], z_partner[p,c] )

This forces the model to explain what *differs* between cells.

**Inputs.** Initially: global baseline RNA `z_c`; baseline expression of the perturbed gene; of the
response gene; of the real partner genes. Later, **only if RNA context passes**: protein abundance, ATAC,
selected pathway activity, environment.

**Objective components.** Cell-specific residual loss · pairwise contrast loss on `D[a,b,p,g] = Y[a,p,g] −
Y[b,p,g]` · **context-swap loss** `max(0, m + L_correct − L_wrong_context)` · context-ordering loss
(which cell responds more strongly to the same perturbation–gene pair) · study-adversarial loss where cell
and study are not fully confounded.

**Required result.** Correct cell context must beat swapped context, shuffled gene context, cell-ID only,
global state only, zero residual, and cell-average residual. **Otherwise this branch does not enter the stack.**

> **Status: measured, and it does not currently pass.** See §18. This branch is *blocked*, not refuted —
> the blocking constraint is the number of cell contexts, not the objective.

---

## 10. Block 7 — Mechanistic modules

The transformer should not relearn known physical constraints.

**A. Metabolism.** Transformer predicts reaction bounds `ℓ_r ≤ v_r ≤ u_r` from predicted enzyme
abundance/activity, HumanGEM reaction mapping, medium composition, metabolite measurements and required
complex availability. HumanGEM imposes `Sv = 0`. Outputs: feasible flux ranges, bottlenecks, growth,
metabolic uncertainty. **HumanGEM stays a solver/constraint layer, never a generic graph neighbourhood.**

**B. Complexes.** `C_k ≈ min_i (P_i / ν_i)`, modified by localization, perturbation, PTM state where
available, interaction evidence. CORUM/OpenCell/BioPlex as priors.

**C. Compartments.** Mask impossible interactions: `M[ij,c] = 0` when entities cannot colocalize.
Localization priors are **context-specific, not universal truths**.

**D. Regulatory.** Accessibility, calibrated E–G edges, TF presence/activity, promoter state. **Do not
include unsupported Hi-C or polymer terms merely because they are mechanistically attractive.**

**E. RNA kinetics.** `dR_g/dt = k_syn,g − k_decay,g · R_g`, constrained by the validated half-life work
and PerturbSci-Kinetics.

---

## 11. Block 8 — Modality-specific output heads

**There is no single universal output head.**

| head | predicts | primary supervision |
|---|---|---|
| RNA | response probability, direction, rank, effect distribution, uncertainty | dense Perturb-seq; LINCS/Sci-Plex; kinetics |
| Chromatin | ATAC changes, accessible-region probability, peak state, regulatory programme changes | Multiome Perturb-seq; paired RNA–ATAC |
| Protein | abundance, selected surface proteins, translation/degradation-consistent changes, uncertainty | cell-line proteomics; ECCITE/CITE; Papalexi; OpenCell priors |
| PTM / signalling | site-level direction, kinase activity, pathway activity — **only where data exist** | iPTMnet-derived, gated |
| Morphology | morphology embeddings, organelle features, cell-shape responses | genome-wide morphology atlas; Perturb-Multimodal; Cell Painting |
| Metabolic | enzyme constraints, flux intervals, growth effects | linked to the HumanGEM solver |
| Viability | dependency, drug response | DepMap; CRISPRGeneEffect; PRISM; metabolic predictions; codependency |

**Codependency belongs in the viability head, not in generic transcription prediction.** This is measured:
`depmap_codep` predicts viability but not transcription.

**Do not claim universal PTM prediction.** The kinase–substrate layer measured 0.992× a decile-matched
control at n=2,810, and a compartment-gated rescue fell from 1.139× to 1.037× once PPI/co-complex pairs
were removed.

---

## 12. Dataset-to-block map

| dataset / artifact | main model role | what it must **not** be used as |
|---|---|---|
| `dense_response.npz` | primary dense genetic-response supervision | baseline cell-state atlas |
| `baseline_state` | cell-context input | causal response label |
| K562 / RPE1 / HepG2 / Jurkat | initial cross-cell residual tests | universal biology proof |
| HCT116 | fifth independent context | automatically comparable without a processing audit |
| Frangieh | context × immune-condition response | generic baseline state |
| Sci-Plex | drug, dose, chemical-response training | genetic-perturbation equivalent without perturbation-type encoding |
| Shifrut | primary T-cell transfer | same distribution as cancer lines |
| Papalexi / ECCITE | RNA–protein multimodal bridge | full-proteome supervision |
| HCA | baseline cell-state pretraining | causal perturbation transitions |
| LINCS | broad context, drug, dose, time | high-fidelity whole-transcriptome ground truth for every gene |
| Multiome Perturb-seq | paired RNA–ATAC perturbation response | universal context diversity |
| PerturbSci-Kinetics | RNA synthesis/decay dynamics | full cell dynamics |
| OpenCell | protein localization and interaction priors | K562-specific active PPI map |
| morphology atlas | genotype → morphology head | RNA → morphology causal chain unless paired |
| HumanGEM | mechanistic metabolic constraint | generic neighbour list |
| DepMap | viability and context dependency | direct RNA-response labels |
| `causal_reg` | partner selection, perturbational edge pretraining | universal active regulatory network |
| `reliable_edges` | reliability-filtered partner set | guaranteed causal truth |
| E–G model | proximal enhancer–gene prior | distal causal regulation or effect magnitude |
| `contradictions` | hard exclusion / gating | ordinary training labels |

---

## 13. Training curriculum — the order matters more than the model size

**Stage 0 — Data integrity.** Distinguish missing from measured zero; canonicalize identifiers; preserve
assay and study; attach uncertainty; audit density; compare independent reconstructions; remove
label-derived and leaked fields; quarantine predictions. **No architecture testing proceeds until this passes.**

**Stage 1 — Entity pretraining.** Universal embeddings from sequence, GO, PPI, complexes, reactions,
pathways, regulatory information, perturbation signatures. No cell-response claims.

**Stage 2 — Baseline cell-state pretraining.** HCA, baseline RNA, multiome, protein, morphology,
localization. Masked reconstruction, cross-modal alignment, batch robustness, cell-state discrimination.

**Stage 3 — Dense shared perturbation training.** First re-test: real vs random partners; **set
transformer vs simple aggregators**; all heads vs their marginals; dense reliability ceiling. Then train μ.

**Stage 4 — Context-residual training.** Freeze μ. Train δ with residuals, pairwise cross-cell contrasts,
context swaps, correct-vs-wrong context ranking. **This is where cell context must prove itself.**

**Stage 5 — Chemical / time training.** Sci-Plex, LINCS, Frangieh, PerturbSci-Kinetics. Dose, time,
environmental conditioning.

**Stage 6 — Multimodal heads, one at a time:** chromatin → protein → morphology → metabolism →
PTM/signalling. Each must beat its modality marginal, cell-type marginal, a simple translator, shuffled
modality, and a missingness baseline.

**Stage 7 — Mechanistic integration.** HumanGEM, complex capacity, compartments, RNA kinetics. Each tested
on its own relevant endpoint.

**Stage 8 — Joint fine-tuning.** Only admitted blocks. Balanced losses so the massive RNA dataset does not
overwhelm smaller protein or PTM datasets.

---

## 14. The latent-cell-state objective

    z = Encoder( RNA, ATAC, Protein, Morphology, Localization, Metabolism )
    RNA ≈ D_RNA(z),  Protein ≈ D_Protein(z),  ATAC ≈ D_ATAC(z),  Morphology ≈ D_Morph(z)

The same latent state should explain multiple biological "languages".

**But there must not be only one bottleneck vector.** Use a global cell-state vector, entity-specific
latent states, pathway/module tokens, and modality-specific residuals — otherwise a single embedding
averages away important local information.

    genes / proteins / metabolites
              ↓
        functional modules
              ↓
    pathway and complex states
              ↓
        global cell state
              ↓
    morphology / function / phenotype

The transformer may discover modules; known pathways and complexes provide weak supervision.

---

## 15. Evaluation suite

**Reconstruction:** held-out genes · modalities · samples · studies.

**Perturbation:** unseen perturbation · unseen cell · unseen perturbation *and* cell · unseen pathway ·
unseen study · unseen perturbation type.

**Context:** correct vs swapped cell state · vs shuffled context · vs cell-ID only · vs nearest-cell baseline.

**Relation:** real vs random partners · vs degree-matched shuffled · partner set vs topology ·
evidence-filtered vs all edges.

**Cross-modal:** RNA from morphology · protein from RNA+ATAC · ATAC from RNA+perturbation · morphology from
perturbation+latent state. **Every imputation is evaluated only where the true modality was actually measured.**

**Mechanistic:** HumanGEM vs abundance-only · complex model vs protein-only · compartment gating vs shuffled
compartments · kinetic model vs static response.

**Calibration:** every prediction returns expected error, evidence coverage, missing-modality burden,
out-of-distribution score, prediction interval.

---

## 16. Initial implementation size

**Do not begin with a billion-parameter model.**

v4 prototype: 6,000–8,000 shared genes · 128–256-dim entity embeddings · 4–6 set-attention layers ·
separate shared and context branches · **RNA head only** · ~50–150M parameters.

First admissions only: entity identity, baseline RNA, real biological partner sets, perturbation strength,
tide, measurement uncertainty.

    First question:  Does the real-partner advantage survive the dense target?
    Then:            Can a frozen-shared residual objective make cell context readable?

**No protein, ATAC, morphology or HumanGEM enters before those are answered.** Both have now been answered —
§18.

---

## 17. What counts as success

Not "90% accuracy". Nine criteria:

1. real partners beat random partners on the dense target
2. **a simple set encoder cannot match the transformer**
3. every head beats its marginal by more than its MDE
4. correct context beats swapped context
5. the context residual improves held-out-cell prediction
6. the improvement survives held-out perturbations
7. adding a modality improves only its relevant endpoint
8. uncertainty is calibrated
9. the model knows when it is out of domain

Ultimate evaluation: **new cell state + new perturbation → future multimodal state**, with both the cell
and the perturbation absent from response training.

---

## 18. Status of v4's own gates, as already measured

This section exists so the specification cannot be read as though nothing has been tried.

| criterion (§17) | status | number |
|---|---|---|
| 1 real > random partners, dense | **PASSED** | +0.0186 gap vs a 0.0012 MDE, 10,337 knockouts. Honest decomposition: **+0.0101** real gain, +0.0085 control dilution |
| 2 simple set encoder cannot match it | **NEVER TESTED** | mean-pool / Deep Sets arms not run. Block B evidence (real vs shuffled wiring +0.005 ± 0.008) predicts this may fail |
| 3 every head beats its marginal | **in progress** | `dense_heads.py`, 3 partitions. Smoke: responds 0.5004 vs marginal 0.5039; direction 0.4948 vs 0.5056 — both at chance and *below* marginal |
| 4 correct > swapped context | **FAILED** | swap gap **−0.0375**, negative in 3 of 4 cells |
| 5 context residual improves held-out cell | **FAILED** | 0/7 predeclared criteria; a trivial nearest-cell copy (+0.0674, t=+3.13) beats the model (+0.0622, t=+1.10) |
| 6–9 | not reached | — |

**And the dense ceiling.** The ORACLE on an untruncated target is **0.0526**; the partner-set encoder
reaches **67%** of it. The 0.607 ceiling quoted throughout the earlier arc is **void** — it was a ceiling
of the truncation.

### Why criterion 4/5 failing does not refute Block 6

The contrast objective was implemented exactly as §9 specifies, with `μ` the mean over the **three
training cells**, which makes the training residual zero-mean by construction — every cell-blind feature
then has an optimal prediction of exactly zero. There was no shortcut left to take, and it still failed.

But the cross-cell minimum detectable increment is **0.170 with n = 4 contexts**, while the measured
effect is **+0.062 with a 95% CI of −0.118 to +0.242**. The interval spans zero. Propagating the
uncertainty on the cross-cell sd (itself estimated from 3 degrees of freedom):

| assumption | contexts needed, 3σ | contexts needed, 2σ |
|---|---:|---:|
| optimistic | 10 | 4 |
| central | **30** | 13 |
| pessimistic | 415 | 184 |

**The binding constraint is the number of independent cell contexts, and it is not architectural.** The
dense rebuild delivered 1,186× more usable cross-cell observations and moved this result by nothing,
because its limiting axis was never observations. Separately, leave-one-cell-out with four contexts fits
a cell encoder from **three training points** — that is an identifiability failure no sample size per cell
can repair, and crossing roughly n ≈ 10 is what turns Block 6 from degenerate into an actual regression.

### The two things worth doing before any of §13 stage 5 onward

1. **Run controls 6–8 of §6.** If mean-pooling matches the Set Transformer, criterion 2 fails and the
   validated object is a bag, not a transformer. This is cheap and it is decisive.
2. **Acquire cell contexts — they cannot be recovered from disk.** This was audited and the result is
   negative. What a fifth cell buys is *shared perturbations* against the dense tensor's 1,763:

   | candidate | perturbations | shared with dense set | usable? |
   |---|---:|---:|---|
   | K562 (reference) | 1,400 | 1,007 | — |
   | HCT116 | 1,400 | 125 | **provenance unresolved** |
   | Frangieh / melanoma | 249 | 41 | no — below the ≥100 bar |
   | Shifrut / primary T | 33 | 0 | no |
   | Papalexi / THP-1 | 107 | 0 | no |

   HCT116 and Frangieh **were** already processed (`nlz_HCT116.pkl`, `nlz_Melanoma.pkl`, written
   2026-07-21) — an earlier claim in this project that they were "unprocessed" was wrong. But a profile
   is not a join. Frangieh's 41 shared perturbations fall below the pre-declared ≥100 bar that already
   excluded Shifrut and Papalexi. And `hct116.h5ad` is (17,768 × 16,380) with an `obs` containing only
   `idx` — no perturbation column — while `nlz_HCT116.pkl` carries 1,400 KO labels intersecting the K562
   bench's 1,400 by just **113**. §12 of this document already says HCT116 is "not automatically
   comparable without a processing audit"; that line was right and the audit has not been done.

   **Net: n = 4 stands, and nothing on disk moves it.** New contexts must be acquired. That is precisely
   the case for X-Atlas/Orion (HEK293T as a genuinely new context, plus an independent HCT116 screen that
   would settle the provenance question by replication), genome-scale primary CD4⁺ T cells (Shifrut's 33
   perturbations cannot carry a primary-cell context; a genome-scale screen can), and the donor
   perturbation map (a within-cell-type genetic-background axis this project has no access to at all).

---

## 19. Final architecture in one equation

For entity `g`, context `c`, perturbation `p`, environment `E`, time `t`:

    Ŝ[g,c,p,t] = D_g [ F_shared(u_p, N(p), t)                                    ← conserved perturbation biology
                     + F_context(z_c, h[p,c], h[g,c], N_c(p), E, t)              ← cell-specific modulation
                     + F_mechanistic(complexes, compartments, kinetics, metabolism) ]  ← hard constraints

followed by an assay-specific observation model:

    ŷ_observed ~ p( y | Ŝ, assay, power, uncertainty )

A universal entity foundation, a multimodal cell-state encoder, a validated biological partner-set
transformer, separate shared and context-specific transition branches, mechanistic constraint modules, and
modality-specific probabilistic heads — trained from dense perturbation targets and heterogeneous partially
observed experiments **without ever treating missing biology as zero.**
