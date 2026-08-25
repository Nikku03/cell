# CellOS — the complete record

**24 April – 25 August 2026. 2,026 commits, 199 numbered loops, 196 loop modules, 307 recorded runs.**

This file is the honest account of what was built, what was used and why, what came out, and what
is still missing. It is assembled from the committed record — the loop modules, their JSON outputs
and their commit messages — rather than from recollection. Where a number here disagrees with
something said earlier in the project, the number here is the one that survived.

---

## 1. What the project was trying to do, and the method it used

The goal was a **whole-cell model of a human cell**: a representation in which a question about one
layer — a mutation, a reaction, a regulator, a chromosome fold — could be traced through to a cell
outcome.

The method was fixed early and did not change:

- **One module per idea**, in `colab/`, named for the question it asks.
- **Gates predeclared in the docstring before any number is computed.** A gate names the statistic,
  the bar, and what a failure would mean. This is the single most important rule in the project and
  most of what follows exists because of it.
- **Source committed before it runs.** So a result cannot be reverse-engineered into a gate that
  would have passed.
- **Every claim measured on held-out data, with controls.** A control is only evidence if it could
  have changed the answer.
- **Negatives recorded as findings.** Roughly half of the 199 loops returned a negative, and the
  negatives are the more useful half.
- **Overstatements corrected in the repo record**, not quietly dropped. There are explicit
  retractions in loops 7, 29, 31, 32, 45, 55, 70, 83, 84, 129, 144, 146, 152, 178, 185.

A count of gates passed is *not* a measure of progress here. Loops that scored 2/6 include some of
the most useful results, because a gate designed to be failable is doing its job when it fails.

---

## 2. The arcs, in order

### Arc I — foundations and the capability statement (loops 1–32)

**What was used and why.** A minimal-cell metabolic model as the starting substrate, because
metabolism is the only layer with a complete stoichiometry. Later: ESM-2 protein language model
embeddings, structural contact graphs, drug-response panels.

**What came out.**
- Loop 1: the growth feedback **did not earn its place** — the cycle closed but the feedback term
  added nothing. The project's first negative, and it set the tone.
- Loop 3: the lookup table that beat the mechanistic model **knew nothing about metabolism.**
- Loop 4: the open medium *was* holding flux balance back — a real fix, found by asking the model
  what it needed rather than hand-writing the medium.
- Loop 8: **measured DNA torsion predicts transcription, and the sign came from the physics.** One
  of the arc's cleanest positives.
- Loop 10, 20, 26, 30: repeated capability statements assembled from recorded runs. Loop 30 is the
  first time the count went **down** — the audit learned to un-believe a result.
- Loop 14 → 29: "the model can say what is NOT a cancer driver" — later **withdrawn**; the pooled
  anti-correlation was a broken control, though the role split that explained it survived.
- Loop 15: a 92% figure turned out to be **an off-by-a-bisection-tolerance error in my own gate.**
- Loop 25: of seven layers, **two carry the model and five are passengers.**

### Arc II — the 4D chromatin model (loops 33–45, 77–90)

**What was used and why.** Rao 2014 in-situ Hi-C (GM12878 and K562) at 25 kb and 5 kb, because it
is the highest-resolution public contact map with matched CTCF ChIP; CTCF peak orientation, because
loop extrusion makes a directional prediction nothing else does; a Langevin polymer engine with one
eigendecomposition giving both the 3D map and the 4D dynamics.

**What came out.**
- Loop 33: the target is **real and falsifiable**, and loop extrusion is visible in it.
- Loop 35–36: the mechanism is causal; the defaults were most of the gap.
- Loop 37: the time axis works.
- Loops 39–40, 45: compartmental affinity **collapsed the chromosome**; excluded volume made it
  worse; a cost premise of mine was wrong. Three mechanisms proposed and rejected on their own
  stopping rules.
- Loop 43: a barrier re-rolled every timestep is not a barrier — worth **+0.064** once fixed.
- Loop 83: **loop 82's mechanism claim retracted** — the compaction did the work, not stiffness.
- Loop 84: the orientation control had been **wrong for seven loops.**
- Loop 85: four preprocessing defects attributed; loop 33 reproduced exactly; **one of the four
  defects I named did nothing.**
- Loop 86: **chr21 is atypical** — 91st percentile on P(s), 96th on the long band. The arc had been
  calibrated on an outlier.
- Loop 88: across **108 configurations, ZERO** satisfy the joint criterion. The trade-off is
  structural (rank correlation −0.7574; four of five axes opposed). The entire admissible grid spans
  map ρ 0.7773–0.8555 while **a distance-only null scores 0.8283.**
- Loop 89: the model's loops equilibrate in under 20 minutes; **the real cell takes over three
  hours.** Loop 92 explained it: RAD21 re-synthesis.
- Loop 90 (the last loop run): re-scoring loop 88's grid against loop 86's genome targets —
  **0 of 108 configurations land within one genome standard deviation on all four bands
  simultaneously**, and the genome-best configuration scores **0.8229 on the map, below the
  distance-only null's 0.8283.**

**Verdict on the arc: the chromatin model does not work, and the failure is structural rather than
a tuning failure.** Eleven loops of parameter search, four physical mechanisms.

### Arc III — the cell as a typed, budgeted object (loops 46–75, 91–159)

**What was used and why.** Human-GEM (12,931 reactions) for stoichiometry; UniProt for annotation;
CORUM/complex data for machines; measured half-lives, kcat values (BRENDA/SABIO), ribosome profiling,
cell-cycle proteomics and imaging, the Human Protein Atlas spatial proteome. Each was fetched because
a specific slot in the state vector was empty and a loop had named it.

**What came out.**
- Loop 61: **26 of 28 layers earn their fill; the chromosome does not.**
- Loop 64: **the chromosome is disconnected**, and it is not a size effect.
- Loop 68–71: the type system closes in *E. coli* — "the machinery works; it was the data" — and the
  human transplant closes structurally but **loses to fame** (how well-studied a gene is).
- Loop 72: the cell model **contains** the metabolism instead of describing it.
- Loop 74: the lifetime slot goes 0 → 13,329; **29% of the ribosome budget is replacement.**
- Loop 92: self-consistent abundance closes the ribosome budget; `ppm` is not a cell.
- Loop 101: **doubling time predicted at 13.8 h against a measured 24 h, from a protein budget with
  nothing fitted to it.** One of the strongest results in the project.
- Loop 117: compartments **move**, on one axis carrying 79.8% of the variance.
- Loop 119: **the cell cycle is real, and this model's protein equation cannot make it.**
- Loops 120–123, 145–146, 153: five candidate mechanisms for cell-cycle protein timing — TF control,
  translation control, geometry, phosphodegrons, D-box quality — **all eliminated**, each on its own
  predeclared control. Loop 146 **overturned loop 145's positive** by redoing it on curated motifs.
- Loop 129: **1/6 — the validation expansion was illusory and the claim was withdrawn.**
- Loop 144: the 526-protein list **withdrawn on its own rule** when a control failed.
- Loop 156: **ESM-2 8M predicts a measured degradation rate at ρ +0.324, homology-aware split.**
- Loop 131–133: a model beats a constant on kcat by 9.5% of the available distance — and **the
  sequence adds nothing beyond the EC number.**

### Arc IV — metabolite completion (loops 160–172)

**What was used and why.** The Human-GEM bipartite reaction graph; RDKit molecular descriptors;
ESM-2 embeddings; AlphaFold structures; compartment and transport topology. The question: given a
reaction with a missing participant, name it.

**What came out.**
- Loop 160: the walk **loses to counting** on the whole graph, **beats it by +0.1254** on a shortlist.
- Loop 163b–d: **structure loses to sequence by 22 sem** — then loop 163c found that C5's negative
  was about *concatenation*, and with the right merge rule **structure does add**. Electrostatics is
  the weakest block alone and **load-bearing in the merge.**
- Loop 167: a learned re-ranker takes hit@1 from **0.4982 → 0.7266.**
- Loop 168: calibration and abstention work — **precision 0.9986 at 50% coverage.**
- Loop 169: **confidence measures uniqueness, not strength of evidence.**
- Loop 170: spatial features take hit@1 from 0.7266 → **0.8506**, and every bucket improves.
- Loops 171–172: a Ramanujan expander network and 14 RDKit descriptors — **both add nothing.**

### Arc V — the enhancer, the census, and time (loops 173–199)

**What was used and why.** The EP CRISPR benchmark (EngreitzLab, K562 arm) as ground truth, because
it is the only large set of *experimentally validated* element–gene links; JASPAR motifs with
DNAshapeR pentamer shape/electrostatics; hg38→hg19 liftover; ENCODE K562 ChIP for 191 factors;
ENCODE H3K4me3 and WGBS; Rao Hi-C loops and domains; CollecTRI/DoRothEA regulatory network; ENCODE
A549 dexamethasone and dendritic-cell LPS time courses; GEO GSE148175 PRO-seq + ATAC.

**What came out.**
- Loop 173: **3/11** — the sequence chain loses to four columns of base composition.
- Loop 174: **8/9** — the same chain works on the question the CRISPR benchmark could not ask.
- Loop 177: stage one 0.6807 → **AUC 0.8506**, and **it was motif identity all along**; enhancers
  are motif-*poor* (0.914× enrichment).
- Loop 178: **P1 overturns the ceiling I reported from loop 176** — the leave-one-gene-out oracle is
  0.4422, not 0.8844.
- Loop 181, 186: **Hi-C did not close stage two**; three contact instruments all fail a stranger-swap.
  Sub-kb anchors scored *worse* than coarse loops, refuting my own prediction.
- Loop 184: **8/9 — an enhancer is a crowded open place, and the factor's own motif is a minor term.**
  Co-binding 0.8455 > accessibility 0.7902 > H3K27ac 0.7510 > motif 0.6228. AlphaFold DBD geometry
  explains **0%** of the spread (all q > 0.36).
- Loop 185: the best stage-two numbers in the arc, **R@1 0.6734 vs distance 0.5930** — and Z6 caught
  a reasoning error of mine.
- Loop 187: feedforward loops **z +1.3, not enriched**; two-cycles **z +43.8**; coherence is
  composition (z +0.3); autoregulation **z +4.0, 2.2×**.
- Loop 188b: **7/13** — the epigenetic layer, fetched in full, adds **+0.0086 AUPRC** over measured
  binding. 5mC is strongly directional (p 3.55e-16) but nearly useless as a feature.
- Loop 190: the census — 16,492 genes across nine layers; **10% in no mechanistic layer; only 44
  genes bridge mechanism and regulation; only 2 bridge motif and reaction.**
- Loop 191d: **accessibility leads mRNA by 48 min** (p 6.4e-58, all three magnitude terciles, two
  passing negative controls). Feedback sign does **not** order timing; occupancy does **not** carry
  timing once size is controlled.
- Loops 192/196/197: **the clock cannot be replicated with public data.** Four timepoints reverse
  the sign; four estimators chosen to fail differently all fail together; ENCODE has exactly one
  dense matched series.
- Loop 193: the 44-gene seam is **load-bearing** — writers' targets are enriched **13×** for genes
  handling the writer's own substrate (z +10.1), and the swap collapses it.
- Loop 194: metabolic timing coordination is **absent** (z −0.8, stable across hub thresholds).
- Loop 195: the census holes are mostly **not what they looked like** — 5,149 orphan reactions is
  2,315 category errors + 2,834 curation jobs; 2,078 unmodelled enzymes is 1,149 annotation joins +
  260 genuinely new; **1,643 dark genes is 104.**
- Loop 198: **persistence beats everything.** Predicting no change scores −0.0295 held-out R²;
  everything the project knows scores −0.0520. **The map does not step state forward.**
- Loop 199: under a **forced** chromatin perturbation, **transcription precedes accessibility**
  (−0.0840, CI [−0.1123, −0.0547]) — the reverse of the observational clock.

---

## 3. What stands

| question | answer | loop |
|---|---|---|
| which metabolite completes a reaction | hit@1 **0.8506** | 170 |
| …when it abstains | precision **0.9982** at **10%** coverage | 168 |
| which enzyme catalyses a reaction | **0.8065** held out; oracle ceiling 0.8217 | 163d |
| is this DNA an enhancer | AUC **0.8506** | 177 |
| which element does a gene use | R@1 **0.6734** (distance 0.5930) | 185 |
| what makes a TF bind an enhancer | co-binding > accessibility > H3K27ac > motif | 184 |
| does a protein's sequence predict its degradation rate | ρ **+0.324**, homology-aware | 156 |
| doubling time from a protein budget | **13.3 h** vs measured 24 h, nothing fitted | 101 |
| do feedback two-cycles exist beyond chance | **z +43.8** | 187 |
| is autoregulation above chance | **z +4.0**, 2.2× | 187 |
| is the metabolism/regulation seam load-bearing | **yes**, z +10.1, substrate-specific | 193 |

**Every one is single-system. None has been independently replicated.**

**Corrections applied to this table on 25 August 2026**, on re-reading the stored JSON rather than
the commit messages. All three were overstatements in the first draft of this file:

- Loop 168 read **precision 0.9986 at 50% coverage**. The stored risk–coverage curve says
  **0.9982 at 10% coverage**; at 50% coverage precision is **0.9534**. Gate X3 **failed** on its own
  bar — "90% precision is not reachable above 10% coverage; the rule is too selective to be worth
  having." The abstention machinery is calibrated and works; the operating point is not a useful one.
- Loop 163d read **0.825**. The held-out four-block merge is **0.8065** (folds 0.8199 / 0.7932),
  against loop 163c's two-block merge at 0.8001. **0.8217 is the four-arm oracle** — it uses the
  answer and is a ceiling, not a score.
- Loop 101 read **13.8 h**. The stored value is **13.28 h** (bootstrap median 13.21,
  CI 11.24–15.57).

Loops 170 and 177 both land on 0.8506 on unrelated tasks. That is a coincidence to four decimals,
not a transcription error: 170 is hit@1 0.85059, 177 is mean gbm AUC 0.85056.

## 4. What was refuted — including my own predictions

Feedforward loop enrichment · sign coherence as logic · epigenetics adding over measured binding ·
metabolic timing coordination · Hi-C closing stage two · sub-kb resolution being the constraint ·
the 0.8844 ceiling · H3K27me3 as a repressive predictor · my coverage excuse for that failure ·
the chromatin model producing a typical chromosome · five mechanisms for cell-cycle protein timing ·
buffering being buffering · the expander property · molecular descriptors · 650M over 35M ESM.

## 5. What is missing

**Data that does not exist publicly.** A densely-sampled time course with matched accessibility and
nascent transcription in one cell system. ENCODE has one dense matched series (A549); GEO's nascent
time courses sample two or three points. This blocks replication of the only 4D result.

**Coupling.** Only 44 genes bridge mechanism and regulation; only 2 bridge motif and reaction. The
layers coexist and barely touch.

**Dynamics.** Nothing here runs forward. Persistence beats the model.

**Coverage that is real rather than apparent.** 260 genuinely unmodelled reactions, 2,834 orphan
reactions needing curation, 104 genuinely dark genes. Smaller and harder than the totals suggested.

**One cell.** K562 for enhancers, A549 for timing, mixed lines for the network. No two layers were
ever measured in the same cells at the same time.

## 6. What was learned about method

Four gate defects recurred, each patched then rewritten a loop later. They are now structurally
impossible (`colab/gate_guard.py`, `colab/test_gate_guard.py`, 15/15):

- **VOID** as a first-class outcome — a test that did not apply is not a test that failed.
- **Lazy narration** — a message can no longer kill a verdict that was decided correctly.
- **Declared preconditions** — a confirmatory gate voids when its upstream fails.
- **Magnitude-based controls** — a gate must not assume the sign of its own answer.

The running record of gate-design defects is `NOTES_e9_gate_wrong_denominator.md`: E9 got the
denominator wrong, R5 the null, B6 the file, G2 the missing values. Each is the same failure at a
different depth — *a gate written from the idea, without looking at the data it would touch.*

The gates caught things reading output never would have: a batch discontinuity at 25→30 min that
invalidated three rebuilds; a target that was reproducible and degenerate; a control that could not
move; a search that needed its own positive control.

## 7. The honest one-paragraph statement

This project built a detailed, well-documented **static map** of a human cell — nine layers over
16,492 genes, with genuinely useful predictors for metabolite completion, enzyme assignment,
enhancer identification and protein degradation, and one strong unfitted result (doubling time from
a protein budget). It has **one timing relation that does not replicate with any public data and
whose causal version points the opposite way**, and **no demonstrated ability to run anything
forward** — persistence beats every dynamic model tried. Roughly half of what was attempted failed
and is recorded as having failed. The map is real; the cell is not.
