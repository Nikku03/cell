# PHASE 2 DESIGN — make sure it's the best we can do, *before* building

Status: design review + feasibility MEASURED. The four gating numbers were run
against feba.db (`scripts/phase2_feasibility.py`,
`outputs/phase2_feasibility.json`). Results below changed the plan.

## FEASIBILITY RESULTS (measured 2026-06; 8-organism sample incl. 3 Shewanella)

| # | number | value | consequence |
|---|--------|-------|-------------|
| Q1a | replicate corr, full | median 0.60 | dominated by neutral mass — not the target |
| **Q1b** | **replicate corr, TAIL** | **median 0.34, p25 −0.18** | **continuous fit value does NOT reproduce — regression target was wrong** |
| **Q1c** | **strong-hit reproducibility** | **median 0.78, mean 0.62** | **binary "is it a hit" DOES reproduce — this is the real target + the ceiling** |
| Q2a | interaction (residual) variance | 0.64 | the headroom; additive captures only main-effects |
| Q2b | additive tail-recall | mean 0.155, ~0.01 in MR1/SB2B | the floor to beat; ~0 in our litmus orgs |
| Q2c | bacitracin litmus vs additive | missed by 3.24 fit-units avg (envZ/SB2B: actual −5.2, pred +0.20) | proof the signal is pure interaction |
| Q3 | compounds with CAS / in ≥3 orgs | 97% have CAS; 153 in ≥3 orgs | **LOCO in scope** via CAS→structure→MoA features |
| Q4 | org×compound cells / w/ replicates | 2,425 / 1,815 | **split by compound, not experiment** |

### THE decision the data forced: target = strong-hit, not continuous fit
The original brief said two-tower → *predicted fit* (a regression). Q1 kills
that: the continuous tail value reproduces at only ~0.34 (p25 negative), so a
perfect model is bounded by measurement noise on that target. But strong-hit
*identity* reproduces at 0.62–0.78. So Phase 2 predicts **P(strong conditional
vulnerability | gene, condition)** — binary/ranking — which is (a) the
reproducible signal and (b) exactly the adjuvant-ranking output we want. The
continuous fit becomes an auxiliary/secondary head, not the headline target.

### Ceiling, floor, and the win condition (now quantitative)
- **Ceiling** = strong-hit reproducibility ≈ 0.62–0.78. A held-out model whose
  precision/recall approaches this is "as good as the measurement allows."
  Report everything relative to this, never to 1.0.
- **Floor** = additive tail-recall ≈ 0.155 (and ≈0.01 in MR1/SB2B). Beat this.
- **Litmus** = recover envZ/ompR/pspB × bacitracin (additive misses by 3.24).

### Two more locked decisions from the results
- **Feed leak-free additive main-effects as FEATURES.** gene-mean-fit and
  compound-mean-fit (computed per-fold, excluding held-out org/compound) go in
  as inputs, so the model spends its capacity on the 0.64 interaction residual
  instead of re-learning main-effects. The model predicts the part additive
  can't.
- **Lean on cross-organism agreement + calibrated abstention for precision.**
  Single-measurement reproducibility is 0.62–0.78; requiring agreement across
  ≥2 organisms (as phase 1 did) lifts precision toward the high-confidence
  subset — the AlphaFold-paradigm move, carried forward.

---

Original review (still valid) below.

## TL;DR

1. **The task is interaction prediction, not regression.** Variance
   decomposition already showed the conditional zone is organism 7.5% /
   gene 13.6% / **residual (env×gene) 79%**. The 79% IS the product. A model
   that nails gene main-effects and condition main-effects and misses the
   interaction is worthless here. So the baseline we must beat is the
   *additive* model, and the metric must isolate interaction capture.
2. **Do NOT default to a dot-product two-tower.** That architecture is built
   for retrieval at scale (millions of items, sub-linear lookup) — we don't
   have that problem. Its weakness is exactly our signal: a dot product is a
   thin interaction model. The honest plan is a 3-way bake-off and let the
   data pick the simplest model that reaches the ceiling.
3. **Four numbers gate the whole design.** Measure them first
   (`scripts/phase2_feasibility.py`), because each one changes the
   architecture or kills a promise:
   - noise ceiling (replicate agreement) — the max achievable
   - additive-baseline tail-recall — the floor we must beat
   - condition-feature transferability — whether leave-one-condition-out is
     even possible
   - holdout group structure — how to split without replicate leakage
4. **Global MSE/R² is a trap.** Fitness is zero-inflated; a model that
   predicts ~0 everywhere wins on MSE and is useless. Eval must be
   tail-focused (recall of fit<−3 at fixed precision, tail Spearman, AUPRC).
5. **The validated bacitracin cluster is the litmus test.** If the model
   can't recover envZ/ompR/pspB × bacitracin when bacitracin is held out as a
   condition (and Shewanella held out as organisms), it isn't ready.

---

## 1. The task, stated precisely

Predict `fit(gene, condition)` — the RB-TnSeq fitness of a gene knockout under
a specific experimental condition — for combinations not seen in training.
Two honest holdouts, which are *different difficulties*:

- **LOO-organism** (seen compound, unseen organism): generalize across
  genomes. Feasible — ESM + leak-free family_frac already transfer (proven in
  phase 0).
- **LOCO** (unseen compound, seen organism): generalize across chemistry/
  mechanism. Feasible **only if** condition features transfer. This is the
  promise that can die at question 3.
- **Double-blind** (unseen both): the real adjuvant-discovery setting. Hardest.

We will report all three separately. Pooling them hides the LOCO difficulty.

## 2. Why this is interaction prediction (and the baseline that proves it)

The additive model is:

    fit(g,c) ≈ μ + α_gene[g] + β_cond[c]

It captures "generally fragile genes" and "generally toxic conditions" with
zero interaction. Our entire product — adjuvant targets — lives in the
residual `fit(g,c) − (μ + α_g + β_c)`. The bacitracin lead is the canonical
case: envZ/ompR are neutral in 156–188 of ~190 conditions (α_gene ≈ 0) and
bacitracin is mild for most genes (β_cond small, only 0.9–2.3% of genes
drop), yet the *pair* is lethal. Additive predicts ~0 for that cell. So:

- **The additive baseline must be built and reported.** Model value =
  interaction variance captured *above additive*, under holdout.
- A neural two-tower with a dot-product head is, after centering, close to a
  bilinear interaction term on top of additive — i.e. only marginally more
  expressive than the baseline unless the head is richer (concat-MLP, FiLM,
  cross-attention). That's the core architecture argument.

## 3. The four gating numbers (measure before architecture)

### Q1 — Noise ceiling (the single most important number)
RB-TnSeq fitness is noisy. Two runs of the same (organism, compound) give two
fit values that disagree. The correlation between replicate experiments is the
**upper bound** on any model's achievable accuracy. Measure:
- replicate fit correlation across shared genes (overall + in the tail where
  |fit|>2),
- reproducibility of strong hits: of pairs at fit<−3 in replicate A, what
  fraction are <−2 in replicate B?

If the tail reproduces at r≈0.6, chasing r≈0.9 is chasing noise. We saw one
beautiful example in phase 1 (envZ/SB2B bacitracin: −5.20 vs −5.18 across two
experiments) — Q1 measures whether that holds in general.

### Q2 — Additive-baseline tail-recall (the floor)
Fit μ + α_gene + β_cond (per-organism, leak-free) and measure how many fit<−3
pairs it recovers and its tail Spearman. The gap from Q2 to Q1 is the
**interaction headroom** — the only thing the model is actually fighting for.
If headroom is small, a complex model is not worth it.

### Q3 — Condition-feature transferability (gates LOCO entirely)
For leave-one-compound-out to be possible, conditions need features that
transfer to never-seen compounds:
- mechanism-of-action / drug class (cell-wall, DNA, protein-synth, membrane,
  oxidative, metal, carbon-source, nitrogen-source, …),
- chemical structure for compounds (feba.db `Compounds` table → PubChem CID →
  Morgan fingerprint / descriptors),
- physical context: concentration, units, media, pH, temperature, aerobic.

Measure: of the distinct compounds in `Experiment`, how many map to a known
structure/MoA? If conditions are effectively one-hot (no transferable
features), LOCO is a cold-start and impossible — and we'd honestly restrict
the Phase 2 claim to LOO-organism only. **This question decides scope.**

### Q4 — Holdout group structure (leakage discipline)
- Hold out **compounds, not experiments.** set4H29 and set4H30 are both
  bacitracin/SB2B — splitting them across train/test is replicate leakage.
  Group by normalized compound (norm_cond + `Compounds`).
- LOO-organism must recompute family_frac excluding the held-out org
  (machinery exists: per-fold family_frac columns).
- Count (organism × compound) cells and per-compound organism coverage so the
  triple-stratified eval (seen-cpd/unseen-org, unseen-cpd/seen-org,
  unseen-both) has enough cells to be meaningful.

## 4. Architecture decision — the bake-off, not a bet

Ranked by what to actually run, simplest first:

| candidate | interaction model | leak-audit | data hunger | when it wins |
|-----------|-------------------|------------|-------------|--------------|
| Additive (μ+α+β, ALS) | none (reference) | trivial | none | — (floor) |
| **XGBoost on [gene⊕cond features]** | native (tree splits) | established in this repo | low | default workhorse; messy/missing features; tabular condition metadata |
| Neural FiLM / bilinear two-tower | condition modulates gene repr | harder | high | ESM carries continuous gene signal trees underuse AND data is enough |
| Dot-product two-tower | thin (dot product) | medium | medium | retrieval scale we don't have — **not our case** |
| Cross-attention transformer | rich | hardest | very high | only if all above plateau below ceiling |

Recommendation: **start with additive (reference) + XGBoost-on-concat
(workhorse).** XGBoost natively models the interaction via splits
(`split on condition_class → split on gene_pathway`), reuses our leak-free LOO
machinery, eats missing/mixed condition metadata, and is leak-auditable —
all the project's existing strengths. Add a neural **FiLM** head (condition
representation gates the gene representation — biologically apt: the condition
"switches on" the gene's relevance) ONLY if (a) Q-analysis shows ESM
embeddings carry continuous gene signal trees can't exploit, and (b) there's
enough data. Note: the binary-essentiality task found ESM/codon got **absorbed
by family_frac** — but that was baseline essentiality; the conditional task
may differ, and that's an explicit experiment, not an assumption.

Carry forward the **AlphaFold-paradigm confidence/abstention** that worked in
phase 0 (95% accuracy on the confident 39%): the model outputs calibrated
uncertainty and flags low-confidence pairs, so the confident subset is
high-precision for downstream wet-lab triage.

## 5. Evaluation protocol (locked)

- **Metrics (tail-focused):** recall of fit<−3 pairs at fixed precision
  (0.3, 0.5); Spearman within the negative tail; AUPRC for the binary
  "strong-negative" label. Report global R² only as a sanity check, never as
  the headline.
- **Three holdout regimes reported separately:** seen-cpd/unseen-org,
  unseen-cpd/seen-org, unseen-both.
- **Litmus:** held-out recovery of envZ/ompR/pspB × bacitracin in Shewanella.
  Concrete pass/fail on our own validated finding.
- **Ceiling-relative reporting:** every number stated as a fraction of the Q1
  noise ceiling. "0.55 tail-Spearman" is meaningless alone; "0.55 against a
  0.62 ceiling" is the truth.

## 6. Leakage discipline (the project's spine, extended to two axes)

- gene tower / features: no held-out-organism labels; family_frac recomputed
  per fold excluding the held-out org.
- condition tower / features: no held-out-compound identity; features derived
  only from chemistry/physics, not from the held-out fitness.
- group splits by (organism, compound), never by experiment (replicate
  leakage).
- runtime leak audit: assert no (organism×compound) cell appears in both train
  and test of any fold.

## 7. The sequence

1. **`scripts/phase2_feasibility.py`** (Colab, against feba.db) → answers
   Q1–Q4. Output: `outputs/phase2_feasibility.json` + console verdict.
2. Read the verdict. It dictates: LOCO in-scope or not (Q3); headroom worth
   chasing or not (Q1–Q2 gap); compound-level split design (Q4).
3. **Build the training frame**: full GeneFitness (not the fit<−3 atlas — the
   neutral mass is needed) joined to gene features (ESM, family_frac leak-free,
   orthology, codon) and condition features (MoA, structure, physical).
4. **Bake-off harness**: additive vs XGBoost-concat vs (conditionally) FiLM,
   same leak-free triple-stratified holdout, tail metrics, bacitracin litmus.
5. Pick the simplest model within noise of the ceiling. Ship with calibrated
   confidence + abstention.

## 8. What "best we can do" means, quantitatively

Not "highest R²." It means:
- the chosen model's tail-recall is within ~10% of the Q1 noise ceiling
  (you cannot beat the measurement noise — pretending to is overfit),
- it beats the additive baseline on interaction capture by a margin larger
  than fold-to-fold variance,
- it recovers the bacitracin litmus under double-blind holdout,
- and we are honest about which holdout regime each claim lives in.

If feasibility shows the noise ceiling is low or condition features don't
transfer, the *best we can do* might be a strong LOO-organism model with no
LOCO claim — and saying so plainly is the right outcome, not a failure.
