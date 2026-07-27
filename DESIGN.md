# Design for a cell model that actually works

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

Items 1–5 are buildable now with data that exists. Item 7 is genuinely blocked and bounds how well anything can be
known to work.

---

## 9. What this design does not promise

It will not predict the one-off tail well. 40.6% of responsive genes move for a single perturbation, and no model
learns a rule from one instance — those require the mechanism to be right in detail, gene by gene, which no current
database supports.

The realistic target is: **correct abstention on the 58% that do nothing, the shared core predicted from damage
magnitude, and the specific arm predicted only where a direct mechanistic route exists** — with honest coverage
reporting on the rest. That is a smaller claim than "predicts any knockout", and it is the one the measurements
support.
