# Cellformer v2 — the whole cell through a transformer, and the plan that makes it falsifiable

v1 mapped AlphaFold's *architecture* onto the cell. It scored **0.0508** against a frequency baseline of
**0.2824**. This document plans v2, which maps AlphaFold's *method* — the templates, the curriculum, the
self-distillation, the staged losses — and, more importantly, fixes the thing that made v1's failure hard to
interpret.

Everything below is grounded in a number measured in this project. Where a component cannot be built from data
that exists, that is stated rather than designed around.

---

## 0. What v1 got wrong, in three measurements

| measured | what it means for v2 |
|---|---|
| full model **0.0508** vs frequency **0.2824** | the model was not competitive; ablating its parts would have measured noise |
| negatives drawn uniformly, evaluated against frequency-ranked candidates | it was trained on easy negatives and tested on the hardest — a sampling bug, fixed, worth 2.3× |
| the benchmark rewards the **shared stress core** (37.4% of held-out variance; residual transfer *loses* to frequency while raw transfer *beats* it) | **the metric itself is the problem** — see §6 |

The third is the deep one. A model can only "win" on mover-recall by reproducing the genes that move for
*everything*. That is why frequency has beaten seven mechanistic methods here. v2 is scored on a second metric
that cannot be won that way.

---

## 1. Four input channels — the whole cell, not just the matrix

v1 used two (response matrix, gene-pair network). v2 uses four. The two new ones are where the kinetics enter.

```
A  RESPONSE      m[p, g]     5,120 perturbations x 8,246 genes of z-scores      the "MSA"
B  PAIR          z[g, g']    5 measured relation channels                       the pair representation
                             reaction 6,369 | complex 16,078 | PPI 186,314 | coexpr 123,090 | signed-reg 23,340
C  TEMPLATES     t[k, g]     partial answers from OTHER modalities              <- new, §2
D  KINETICS      e[r]        reaction -> kcat, 8,184 reactions, tiered          <- new, §3
```

Channel D ships in the repo as `colab/data/kinetics_bundle.json.gz` (0.17 MB), built by
`colab/build_kinetics_bundle.py` with the hierarchy that `kcat_headtohead.py` measured on 915 common
leave-one-out labels:

| tier | source | median fold error | reactions |
|---|---|---:|---:|
| 1 | human-EC median | **2.62×** | needs `dlkcat.tsv` (currently missing) |
| 2 | CatPred | 8.38× | 7,592 |
| 3 | any-organism EC | — | needs `dlkcat.tsv` |
| 4 | global median | 14.23× | 592 |

**Deliberately excluded: ecHumanGEM (66.3×, bias +1.71) and EC-max (40.7×, bias +1.61).** Both are `kcat_MAX`
— upper bounds by design, not estimators. Feeding them in as point estimates inflates every capacity by ~1.7
log units. This is the one place the "use ECMpy 2.0" instinct is actively wrong, and it is wrong for a reason
that only shows up when you score them on common labels.

---

## 2. The template stack — the AlphaFold idea, mapped properly

AlphaFold's templates are **structures of homologs**: a different modality that hands the model a partial
answer directly, through a dedicated stack rather than mixed into the MSA. The cell has two genuine analogues,
and one of them is much stronger than anything v1 used.

### Template 1 — the same knockout in a different cell line

We hold Perturb-seq readouts for **six lines**: K562 (5,120 perturbations), RPE1, HepG2, Jurkat, HCT116,
Melanoma. Knocking out gene *g* in RPE1 is, structurally, exactly what a homologous crystal structure is to
AlphaFold: *the same object measured in a different context*. It is a partial answer from a different
experiment, not a re-reading of the same one.

This is the single highest-value addition in v2, and it is also the honest generalisation test (§6).

### Template 2 — the physics prediction

For a knockout inside the metabolic model, the enzyme-constrained FBA gives a *predicted* flux response:

```
knockout -> GPR -> capacity from kinetics bundle (channel D) -> pFBA -> ΔF -> Δmetabolite turnover -> prior over genes
```

Build 6 established this chain works and is strictly ordered (845/845 knockouts change flux, 0 ordering
violations), and that it beats both its own shuffled control and a gene-annotation predictor — while losing to
frequency. As a **template** rather than a standalone predictor, that is exactly the right role: a partial
answer the transformer may attend to or ignore.

Coverage is stated up front, not discovered later: **845 of 5,120 perturbations (16.5%)** are genes the
metabolic model contains. Template 2 is absent for the other 83.5%, and the template stack must handle absence
as a first-class case — AlphaFold's does too.

---

## 3. Training curriculum — three stages, because the expensive losses only matter once the model is roughly right

AlphaFold trains at crop 256 without violation losses, then fine-tunes at crop 384 with them on. The reason is
practical: a structurally-nonsensical early model gains nothing from a clash penalty. The same logic applies
here.

| stage | crop | channels | losses | purpose |
|---|---|---|---|---|
| **A** | 128 | A + B | response + masked-gene | learn the co-response structure |
| **B** | 256 | A + B + C | + template loss, + confidence | learn when to trust a partial answer |
| **C** | 256 | all four | + violation losses (§5) | make the physics bind |

Random recycling count (uniform 0–3) throughout, as AlphaFold does, so inference works at any depth.

---

## 4. Self-distillation — the 4,282 perturbations currently thrown away

AlphaFold predicts structures for 350k unlabelled sequences, filters by pLDDT, and trains on its own
high-confidence output. The cell equivalent is sitting unused: **4,282 of 5,120 perturbations produce fewer
than 5 specific movers** and are excluded from scoring entirely. Plus every perturbation in the five non-K562
lines that has no K562 counterpart.

Procedure: predict, keep only where the confidence head is high, add to training with a down-weighted loss.

**The control this needs, because self-training amplifies its own bias.** A model that has learned the
frequency prior will confidently predict the frequency prior on unlabelled data, and training on that makes it
*more* of a frequency predictor. So self-distillation is only kept if it improves the **specific-arm metric**
(§6), not merely the legacy one. If it improves legacy recall while flat or worse on specific-arm, it is
confirmation bias and gets reported as such.

---

## 5. Losses, and which stage each is enabled in

```
L = L_response                              stage A   BCE on specific movers
  + λ_mask · L_masked_gene                  stage A   BERT-style; forces the response repr to mean something
  + λ_conf · L_confidence                   stage B   predicts its own error -> abstention (build 2)
  + λ_tmpl · L_template                     stage B   agreement with templates, where present
  + λ_phys · ‖S·v‖²                         stage C   mass balance
  + λ_dose · L_monotonicity                 stage C   response must be monotone in knockdown strength
  + λ_core · L_decomposition                stage C   predicted response must split core/arm as build 3 measured
```

`L_dose` and `L_decomposition` are new and specific to this problem. Build 5 established dose-monotonicity is a
real property of the system (80/80 genes monotone). Build 3 measured the core/arm split (residual coherence
z = +93.4). Both are things the model can be *required* to respect rather than hoped to learn.

Honest note on `‖S·v‖²`: the design wanted a hard projection onto `null(S)`; that is not tractable for an
8,000 × 13,000 stoichiometric matrix here, so it enters as a penalty. The model is *encouraged* toward mass
balance, not confined to it.

---

## 6. Evaluation — two metrics, and the second is the point

**Metric 1 — legacy mover-recall.** Tide-removed specific movers, top-50, held-out perturbations, frequency
baseline in the table. Comparable to every number this project has published.

**Metric 2 — specific-arm recall.** The identical computation on the **residual after the shared core is
projected out**, using build 3's decomposition (core fitted on training perturbations only, k=20, 37.4% of
held-out variance).

Why metric 2 decides this: builds 3 and 4 measured that the residual carries real mechanism (complex-partner
coherence z = +93.4, permuted control z = +1.5) but that predicting it *loses* to frequency on metric 1 —
because metric 1 rewards naming genes that move for everything. **A model cannot win metric 2 by learning the
frequency prior**, since the prior is precisely what was subtracted. Metric 2 is the one that measures
"which gene was damaged" rather than "was it damaged".

**Metric 3 — cross-cell-line transfer.** Train on K562, test on RPE1. No overlap in measurement, only in
biology. This is the test AlphaFold's CASP is: held-out targets, not a held-out split of the same set.

---

## 7. The ablation plan

Stage-level and channel-level, because v2's claims are about *method*, not just architecture.

| # | ablation | tests |
|---|---|---|
| 0 | full v2 | — |
| 1 | stage A only (no curriculum) | does staged training buy anything? |
| 2 | stages A+B (no violation losses) | do the physics losses bind? |
| 3 | no template stack | do partial answers from other modalities help? |
| 4 | template 1 only (cross-line, no FBA) | which template carries it? |
| 5 | template 2 only (FBA, no cross-line) | " |
| 6 | no kinetics channel (uniform kcat) | does kcat matter *at all* once it is a feature not a constraint? |
| 7 | no self-distillation | does the unlabelled majority help, or amplify bias? |
| 8 | no decomposition loss | does forcing the core/arm split help? |
| 9 | shuffled labels | the floor; must collapse |
| 10 | frequency baseline | the bar |

**Pre-registered criteria, set now:**

1. v2 must beat frequency on **metric 2** with a paired-bootstrap CI excluding zero. Metric 1 is reported but
   is not the criterion, for the reason in §6.
2. Ablation 6 is the direct test of the kcat question. If a uniform kcat scores the same as the measured
   hierarchy, then kcat does not matter as a *feature* either — which, given builds 1/7/8/9 showed it does not
   matter as a *constraint*, would close the question for good.
3. If v2 beats frequency but ablations 1, 2, 3 and 6 all cost nothing, the report is **"scale and capacity beat
   the baseline; none of the biology or the method mattered"** — in those words.

---

## 8. What this plan does not promise

- It will not learn the one-off tail. 40.6% of responsive genes move for exactly one perturbation.
- It will not fix FBA essentiality. That ceiling is 0.561 and structural (build 10), and no representation
  learning changes which reactions carry flux.
- Tiers 1 and 3 of the kinetics bundle are **currently unavailable** — `dlkcat.tsv` was destroyed in a sandbox
  rollback. The bundle records this. Rebuilding it restores the 2.62× tier for 1,730 reactions; until then the
  kinetics channel is CatPred-dominated at 8.38×.
- Six cell lines is not 350k Uniclust sequences. Self-distillation here operates on thousands of examples, not
  hundreds of thousands, and should be expected to give a correspondingly smaller gain — or none.
