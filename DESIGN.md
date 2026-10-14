# Design for a cell model that actually works

> **STATUS — all seven buildable items have now been built and scored against their own pre-registered tests.
> Three passed, four failed, and the failures are more informative than the passes. Section 10 holds the results
> table and the corrections; the original design reasoning below is left intact, with `MEASURED:` notes marking
> every claim the measurements changed. The single biggest correction: §3 named quantity "the single
> highest-value missing piece" and it is not — see §10.**


Every design decision below is justified by a number measured in this project, and every component has an
acceptance test it must pass before it is believed. Where the honest answer is "this cannot be built with data that
exists", that is stated rather than designed around.

---

## 0. Why the current architecture cannot work, in four measurements

The present system maps `gene -> ranked gene list` in one shot, from a static graph. Four measured facts make that
architecture unable to succeed regardless of how well it is implemented.

| measured | consequence for the architecture |
|---|---|
| **58.2%** of perturbations (2,982/5,120) produce **zero** specific movers | a model that always emits a list is wrong by construction on the majority of inputs |
| **313 genes carry 63.2%** of all mover-calls; top-20 carry 12.6% | most of the answer is a shared stress core, which a trivial frequency baseline already captures (0.17–0.21) |
| **1,425 genes (40.6%)** move for exactly **one** perturbation | the specific tail is unlearnable from response data; it can only come from mechanism |
| direction known for **6%** of edges; **91%** of the regulation layer is binding that overlaps real regulation at chance (fold 0.96, p=0.68) | you cannot propagate an effect along the graph as it stands |

The system also has **no state**. A knockout is applied to a graph, not to a cell that is in some condition. That is
why it cannot answer "what if the cell is also starved", "what if this is a point mutation rather than a deletion",
or "what happens after six hours rather than at steady state".

---

## 1. The core change: a state vector, not a lookup

A working model must be a **simulator**. The object being perturbed is a cell **state**, and the readout is a
projection of that state after it evolves.

```
    S = ( M , P , F , A , C )

    M   metabolite concentrations            ~8,000 species
    P   protein abundances                   16,492 genes      <- HAVE: PaxDb, 95.4% coverage
    F   reaction fluxes                      28,528 reactions  <- HAVE for 12,931 (stoichiometric)
    A   TF / signalling activities           ~1,600 TFs        <- BLOCKED, see §5
    C   chromatin / accessibility state      genome-wide       <- HAVE for K562 (ATAC, H3K27ac)
```

A perturbation is an **operator on S**, not a lookup key. That single change is what lets one model answer
knockout, knockdown, point mutation, overexpression, drug, nutrient shift, and combinations — because all of them
are different operators on the same state.

```
    knockout        P[g] := 0
    knockdown       P[g] := (1-k)·P[g]                     k from the guide's measured efficiency
    point mutation  P[g] unchanged, but kcat/Km/binding of g altered   <- needs §3
    overexpression  P[g] := c·P[g]                         c > 1
    drug            competitive inhibition on target's kcat
    nutrient        bound on an exchange reaction
    combination     compose the operators — no new machinery
```

**Acceptance test:** the same code path must handle a CRISPR knockout and a nutrient withdrawal, and reproduce the
measured direction of both. Today's `cell_sim.py` already passes the nutrient half (anaerobic → growth 87.7%,
lactate export up).

---

## 2. Three layers, three time constants — and the readout is the slowest

The single deepest reason the current model fails at transcription while succeeding at essentiality: it treats one
graph as if all edges act on the same timescale. They do not.

| layer | timescale | mechanism | status |
|---|---|---|---|
| **metabolic flux** | seconds | mass balance, stoichiometry | **works** — AUC 0.656 vs K562, Warburg reproduced |
| **signalling / post-translational** | minutes | directed, sign-carrying edges | **blocked** — 94% of edges undirected |
| **transcription** | hours | TF activity → mRNA | **blocked** — regulation layer is binding, not regulation |

mRNA — the thing every benchmark here scores against — is the **output of the slowest layer**, three steps
downstream of the layer that works. Predicting it from a static graph skips two layers that carry all the
specificity.

**Design consequence:** simulate the fast layer to steady state, use its *deviation* to drive the medium layer,
and only then read the slow layer. Not one propagation — a cascade of solvers with an explicit ordering.

**Acceptance test:** perturbing a purely metabolic gene must change F before it changes A, and the model must
produce a mRNA response only through A. If mRNA changes without A changing, the layering is not real.

---

## 3. Quantities everywhere — the single highest-value missing piece

Measured evidence that this is the binding constraint:

- FBA **recall 0.203**: it declares a gene tolerated whenever *any* alternative route exists
- the ablation's `BUFFERED` verdict means "an alternative is annotated", never "enough flux gets through"
- `reaction_ablation.py` states this limitation explicitly and could not resolve it

What we have and what is missing:

| quantity | coverage | source |
|---|---:|---|
| protein abundance E₀ | **95.4%** | PaxDb — have it |
| essentiality floor | 96.5% | DepMap — have it |
| dosage floor/ceiling | 87.1% | Collins pHaplo/pTriplo — have it |
| **kcat** | **0%** | **missing** — no table in the repo |
| **Km** | **0%** | **missing** |
| guide knockdown efficiency | 0% | missing |

With E₀ + kcat, `BUFFERED` becomes a **number**: alternative capacity = Σ(E₀ᵢ · kcatᵢ) over remaining routes,
compared against the flux the cell needs. That converts a boolean into a dose, which is what the E₀/E_min/E_max
band model was for and never got to use.

**Obtainable:** BRENDA and SABIO-RK give measured kcat for a few thousand human enzymes; DLKcat/TurNuP predict it
sequence-wide. This is the most tractable large win available and is **not blocked** — it simply was not done.

**Acceptance test:** FBA recall must rise from 0.203 without precision falling below 0.6. If enzyme capacity
constraints don't recover missed essentials, the hypothesis that quantity is the blocker is wrong.

> **MEASURED: the hypothesis is wrong.** `colab/ec_capacity.py` built exactly this -- real DLKcat kcat, GPR-aware
> capacity (isozymes add, complexes take the min), ceilings recomputed per deletion, one global scale fitted to
> the measured 24 h doubling time. Recall did not move: 0.203 -> 0.203, not one lethal call changed, AUC fell
> 0.676 -> 0.594, and permuting the quantities reproduced it exactly.
>
> `colab/kapp.py` then measured WHY, and the answer is not kcat accuracy. Of 1,651 reactions carrying a ceiling,
> **1,578 carry no flux at all** at the optimum; 5 are at their ceiling; 19 are within two orders of it. A
> ceiling on an unused reaction constrains nothing however accurate its kcat, so improving in-vitro kcat
> (log10 RMSE ~1.0) to proteome-calibrated k_app (~0.4) cannot move a bound sitting six orders above the flux.
> Bayesian shrinkage of the kcat spread confirmed it: lambda 0 and lambda 0.25 both leave exactly 5 reactions
> at a ceiling.
>
> The defect is the CONSTRAINT'S SHAPE, not its parameters. Per-reaction ceilings do not couple reactions.
> `colab/gecko.py` tests the constraint that does -- a shared proteome budget, sum(v_i/kcat_i) <= P_total, where
> every reaction competes for one pool and kcat acts as a PRICE rather than a limit.

---

## 4. Predict magnitude before content — and be allowed to say "nothing"

58.2% of perturbations do nothing measurable. Any model that always emits a gene list is structurally wrong more
often than it is right.

```
    stage 1   MAGNITUDE     will this perturbation do anything at all?      binary, calibrated
    stage 2   CLASS         shared stress core, or specific arm, or both?
    stage 3   CONTENT       which genes
```

Stage 1 is learnable from things we have: cascade breadth (validated, z = −6.88), essentiality, abundance,
participation. Stage 2 is forced by the measured structure — 63.2% of signal is the shared core and 40.6% of
responsive genes are one-off; these need *different predictors* and currently share one.

- **shared core** → a small number of stress axes driven by *damage magnitude*, not identity. The 89-gene sensor
  menu was the right idea applied to the wrong fraction: it should predict the 313-gene core and nothing else.
- **specific arm** → only from direct mechanism: the knockout's own TF regulon, its direct substrates, its complex
  partners. Never from a shared menu, because it is one-off by measurement.

**Acceptance test:** calibration. Among perturbations the model says are silent, ≥90% must really have <5 movers.
A model that abstains correctly is more useful than one that guesses.

> **MEASURED: passes the letter, fails the spirit.** `colab/abstain.py`, 24 a priori features, two-way hash split
> run both ways. On the pre-registered label (<5 movers) the base rate is 0.835, so the 90% bar sits 6.5 points
> above answering "nothing" every time; 90% precision holds to 82.1% coverage -- a near-free pass. On the honest
> label (exactly zero movers, base rate 0.578, which is what this section's own 58.2% refers to) precision at
> 50% coverage is 0.764 and the 90% bar is unreachable at any usable coverage. AUC 0.753 vs shuffled 0.487; lift
> over always-silent +0.187 with both folds' CI excluding zero. Real information, no certification of inertness.
>
> **This section's claim that cascade breadth is the feature to use is wrong.** Cascade breadth is validated at
> z = -6.88 for a different question and is at CHANCE here: AUC 0.5016 alone, as are n_stops, n_rxn, n_buffered
> and n_redundant. What carries the gate is DepMap dependency (0.769 alone) -- but not only that: dropping both
> DepMap columns still leaves AUC 0.721, and dropping expression too leaves 0.689.

---

## 5. The regulation layer must be rebuilt, and this is the real blocker

The specific arm needs TF → target edges that are **causal, signed, and directional**. What exists:

- 91% of the layer is ChIP **binding**, measured against real regulation *in K562* and overlapping **at chance**
  (fold 0.96, p = 0.68)
- the direct-regulon layer contributed **1 hit in 39** in the sensor model

Binding is not regulation. This is not a modelling shortfall — the data is wrong for the purpose.

**What would fix it:** the perturbation data itself. 5,120 measured knockouts, of which ~1,600 are TFs, *is* an
interventional regulatory map — knock out a TF, whatever moves is its functional regulon. That is causal by
construction, unlike ChIP.

**The catch, stated honestly:** ~40% of what moves is the shared stress core, which appears for every knockout and
is not that TF's regulon. So the map must be built on the **specific** residual after removing the shared core —
which is exactly the decomposition §4 requires. The two designs depend on each other.

**Acceptance test:** a regulon derived this way must predict held-out TF knockouts better than the ChIP layer,
which is a low bar (chance), and better than the shared core alone, which is not.

> **MEASURED: clears the low bar by 80x, loses the real one.** `colab/regulon.py`, 439 held-out perturbations,
> transfer from network neighbours' regulons (never the gene's own measurement). Residual+neighbours 0.2308 vs
> ChIP 0.0029 -- and ChIP returns nothing at all for 384 of 439. But frequency scores 0.3502 and wins by 0.119
> with a CI nowhere near zero.
>
> The informative part is which arm wins. Transfer on the RAW matrix beats frequency (+0.0292, CI excludes zero);
> transfer on the residual loses badly. Removing the shared core removes exactly what this metric rewards,
> because the metric asks "name the genes that moved" and the genes that move are mostly the shared core.
> Build 3 showed the specific arm is real (permutation z +93). Build 4 shows it is real and not what the
> benchmark scores. **The benchmark credits predicting DAMAGE, not predicting WHICH GENE was damaged.**

---

## 6. Learned vocabulary, not hand-written

Settled by measurement today. Hand-written 89-gene menu: oracle ceiling **4.4%**, actual 0.027/0.034. Learned NMF
basis over 8,243 genes on identical reasoning: **0.187/0.201**, ~7×.

Keep the learned basis. Do not reintroduce hand-written programmes as the output space — they belong (at most) as
*priors on which component fires*, not as the vocabulary itself.

---

## 7. What the model must be tested against — the harness is the product

Today produced six silent data failures, each of which made a broken pipeline look like a working one: 0 of 191,447
PPI edges read, 0 of 1,148 catalysts, controls scored as knockouts, Ensembl ids joined to gene symbols, infinities
passing the `|z|≥1` mover test, and a matrix rebuilt from rolled-back code. Every one produced plausible output.

The harness is therefore not a nicety. Required properties:

1. **Every join asserts non-empty.** A zero-row intersection fails loudly instead of scoring 0.
2. **Every matrix asserts finite on load**, with the count printed — silent sanitisation is how the last one got in.
3. **Every method ships with a shuffled twin**, and the twin's score is reported next to it always.
4. **Two disjoint cohorts**, one never used for development. A rule fitted on cohort 1 must be scored on cohort 2
   before it is believed — this caught a fabricated +0.005 "improvement" today.
5. **A trivial baseline is always in the table.** Frequency beat seven mechanistic methods; a number without it is
   uninterpretable.
6. **Provenance and measured accuracy at the point of use**, as `cellos2.py boot` now does.

---

## 8. Build order, by expected value per unit of work

| # | component | blocked? | expected gain |
|---|---|---|---|
| 1 | **kcat/Km → enzyme-capacity constraints** | no — BRENDA/SABIO + DLKcat | converts BUFFERED into a dose; directly targets FBA's recall 0.203 |
| 2 | **magnitude-first predictor with abstention** | no — data in hand | fixes being wrong on 58% by construction |
| 3 | **shared-core / specific-arm decomposition** | no | prerequisite for both §5 and §4 |
| 4 | **interventional regulon from the 5,120 knockouts** | no, but needs §3 | the only route to the specific arm |
| 5 | **state vector + operator algebra** | no | one model answers mutation, drug, nutrient, combination |
| 6 | **layered solver with time constants** | needs §1, §4 | mRNA becomes an output of mechanism rather than a lookup |
| 7 | measurement-noise ceiling | **blocked** — the screen's only replicates are its weak tail | without it we cannot say how much residual error is even reducible |

**MEASURED (see §10):** 1 FAIL, 2 PARTIAL, 3 PASS, 4 FAIL, 5 PASS (4 of 5 tests). Two further items were added
in response to what the failures showed: `kapp.py` (in-vivo apparent turnover — FAIL, and predicted in advance by
its own diagnostic) and `gecko.py` (shared proteome budget — the coupling that item 1 lacked).

Items 1–5 are buildable now with data that exists. Item 7 is genuinely blocked and bounds how well anything can be
known to work.

---

## 10. What happened when it was built — measured outcomes

Every item was implemented and scored against the acceptance test written for it *before* the run. Nothing below
was scored twice with a different bar.

| # | component | module | verdict | the number that decided it |
|---|---|---|---|---|
| 1 | enzyme-capacity constraints | `ec_capacity.py` | **FAIL** | recall 0.203 → 0.203, AUC 0.676 → 0.594, shuffled quantities identical |
| 2 | magnitude-first abstention | `abstain.py` | **PARTIAL** | passes on the <5-mover label (base rate 0.835); fails on the zero-mover label (0.764 at 50% coverage) |
| 3 | shared-core / specific-arm split | `decompose.py` | **PASS** | residual complex-partner coherence +0.0708, permutation z +93.4, permuted-membership control z +1.5 |
| 4 | interventional regulon | `regulon.py` | **FAIL** | 0.2308 vs ChIP 0.0029 (clears), vs frequency 0.3502 (loses, CI [−0.148, −0.092]) |
| 5 | state vector + operator algebra | `cell_state.py` | **PASS (4/5 tests)** | T1–T4 pass; T5 fails — epistasis is exactly 0.00000 for real and control SL pairs alike |
| 7 | in-vivo k_app / kcat tuning | `kapp.py` | **FAIL, predicted in advance** | 1,578 of 1,651 ceilinged reactions carry no flux; 5 are at a ceiling |

### The three findings that matter more than the verdicts

**1. Quantity was not the blocker, and the reason is structural.** §3's argument was that `BUFFERED` needed to
become a number. It became a number and nothing happened. `kapp.py` shows why: per-reaction ceilings do not
couple reactions, so a ceiling on an unused reaction is inert regardless of its kcat. This is not an argument
against enzyme constraints — it is an argument against *this* enzyme constraint. The coupling version (a shared
proteome budget, where kcat is a price rather than a limit) is `gecko.py`.

**2. The benchmark rewards predicting damage, not predicting which gene was damaged.** Builds 3 and 4 together
are conclusive. The specific arm is real: complex partners leave similar residuals at permutation z +93, with a
flat permuted-membership control. But it retains only **20%** of the raw functional coherence (+0.0708 of
+0.3561), and on the mover-recall benchmark the residual *loses* to frequency while the raw matrix *beats* it.
Every score this project has published on that benchmark — including the ones it is proudest of — is
substantially a measurement of how well a method reproduces the shared stress core.

**3. Calibrating to the real doubling time makes the model bottleneck-dominated.** One fact explains four
separate results. Fitting the enzyme ceiling to 24 h doubling leaves growth set by a single limiting step, and
therefore: build 1's essentiality ranking flattens (AUC 0.676 → 0.594); nutrients stop mattering (O2 or glucose
withdrawal moves growth <0.1%, and the LP becomes degenerate); only 1 of 80 genes behind active ceilings is
dose-sensitive (GARS1, exactly linear in remaining abundance); and epistasis vanishes entirely, because
knockouts are lethal or neutral and combinations are exactly multiplicative.

### Corrections to claims made in this document

- **§3 "the single highest-value missing piece"** — wrong. kcat was obtained (362 EC numbers, 2,437 human
  records) and applied correctly, and it moved nothing.
- **§4 "Stage 1 is learnable from … cascade breadth (validated, z = −6.88)"** — wrong. Cascade breadth is at
  chance for this task (AUC 0.5016), as are every other ablation-derived feature. Validated for one question
  does not mean informative for another.
- **§4's 58.2% and its acceptance test disagree.** The motivating statistic counts *zero*-mover perturbations;
  the test says "<5 movers", whose base rate is 83.5%. Both are now reported, and the honest one fails.
- **§0's "313 genes carry 63.2% of all mover-calls"** counts mover-*calls*; by variance the shared core is
  37.4% of the held-out response at k=20. The two are not interchangeable and were being used as if they were.
- **§1's acceptance test is met** — `cell_state.py` runs a knockout and a nutrient withdrawal through one
  `solve()`, reproducing `cell_sim.py`'s independently measured 87.7% anaerobic growth, with lactate *yield*
  rising 25.23 → 27.06 per unit biomass. Warburg must be scored as yield: raw export falls under anaerobiosis
  simply because the cell grows less, which reads as the wrong direction.

### Bugs found, all silent, all caught by asserting join sizes

- the donor/test split in build 3 was the parity of a *padding zero* for any gene symbol under 8 characters,
  producing a 5,027 / 93 "half" split. Visible only because a downstream module printed its test-set size.
- `cell_complete`'s `coexpr` values are `[index, correlation]` pairs, not bare indices — read as ints they gave
  a silently empty co-expression layer.
- build 1's first version anchored viability to 5% of *unconstrained* growth — a 2-minute doubling time already
  on record as a ~700× error — and rejected the correct answer on that basis.
- build 1's first version imposed a **static** ceiling and called `single_gene_deletion`, which asks the GPR only
  whether a reaction survives, never how much capacity survivors have left. The ceilings never moved.
- `pfba()`'s `objective_value` is minimised total flux, not growth; printing it as growth put a number three
  orders of magnitude wrong on screen.
- `kapp.py`'s first utilisation report clipped zero-flux reactions to 1e-30 and reported a "median 30 orders
  below ceiling", which is the clip, not a measurement. The real statement is that the median capacity reaction
  carries no flux at all.

---

## 9. What this design does not promise

It will not predict the one-off tail well. 40.6% of responsive genes move for a single perturbation, and no model
learns a rule from one instance — those require the mechanism to be right in detail, gene by gene, which no current
database supports.

The realistic target is: **correct abstention on the 58% that do nothing, the shared core predicted from damage
magnitude, and the specific arm predicted only where a direct mechanistic route exists** — with honest coverage
reporting on the rest. That is a smaller claim than "predicts any knockout", and it is the one the measurements
support.
