# Resource competition and cooperativity — Xiao-Jun Tian (Arizona State)

**Their papers.** According to PubMed:

- Melendez-Alvarez JR, Zhang R, Tian X-J, "Growth Feedback Confers Cooperativity in
  Resource-Competing Synthetic Gene Circuits", *Chaos, Solitons & Fractals* 173 (2023),
  PMID 37485435, [DOI](https://doi.org/10.1016/j.chaos.2023.113713).
- Stone A, Youssef A, Rijal S, Zhang R, Tian X-J, "Context-dependent redesign of robust synthetic
  gene circuits", *Trends in Biotechnology* 42:895–909 (2024), PMID 38320912,
  [DOI](https://doi.org/10.1016/j.tibtech.2024.01.003).

**Their framing of the field's central problem, in their own words:**

> "modular components often do not function as expected when assembled into larger circuits. One of
> the major issues is caused by resource competition"

**Their finding:**

> "Our results suggest a cooperative behavior between resource-competing gene circuits under growth
> feedback."

---

## 1. The question, and it is a question

They observe cooperativity between resource-competing circuits and attribute it to **growth
feedback**. REM's shared-pool model produces **positive correlation between two genes on one pool
with no growth term anywhere in the model.**

The model is three coupled species — pool level, gene 1, gene 2. Pool refills in bursts at a fixed
rate and decays linearly. Each gene produces at a rate **linear in the current pool level** and
decays linearly in its own copy number. There is no growth rate, no dilution coupling, no metabolic
burden term, and no feedback from circuit expression back onto the host. (`joint_two_genes` in
`rem/atlas/pool.py`; the generator is short enough to read in full.)

Measured, from the exact stationary joint distribution:

| pool supply | product-of-marginals ÷ exact | corr(gene1, gene2) | sign |
|---|---|---|---|
| 30.0 | 0.665 | 0.0895 | POSITIVE |
| 6.0 | 0.302 | 0.3256 | POSITIVE |
| 1.5 | 0.184 | 0.6498 | POSITIVE |

**Positive at every supply, rising steeply as the pool tightens.** The mechanism available in this
model is the only one present: both genes are driven by the same fluctuating pool level, so they
rise and fall together. A shared driver induces positive correlation regardless of the fact that
the genes are competing for it.

**So the question for them is a decomposition, not a correction:** how much of the cooperativity
they measure is growth feedback, and how much is the shared-driver term that is present even
without it? Their framework can answer that — set the growth coupling to zero and see what
survives. That is a control internal to their own model, and it is the whole offer.

## 2. A separate number, larger than expected

Composing two genes by **multiplying their individual behaviours** — the operation modularity
assumes — is off by:

- **1.5×** at a loose pool (ratio 0.665)
- **3.3×** at moderate supply (0.302)
- **5.4×** at a tight pool (0.184)

The error grows as the resource tightens, which is the regime their work is about.

**Recorded honestly:** the specification this was built against predicted 0.725 / 0.586 / 0.480 —
i.e. about 2.1× at the tight end. The measured value is **0.184, a 5.4× error, 2.6× worse than
predicted.** The measurement is what is reported.

## 3. What else is available, priced by friction

**Frictionless — a number, no adoption required.** A memory element's flip rate as an exact mean
first-passage time. Measured in `rem/switching.py`: MFPT **7.8434 × 10⁶ generations**, i.e.
p = 1.275 × 10⁻⁷ per generation, computed in **0.13 s** — in a regime where a matched 30-second
Gillespie run saw **zero** switching events. Cost is near-independent of rarity: across a sweep
holding the state space fixed at 25,921 states, the switching probability spanned **7.48 × 10⁵×**
while wall-clock spanned **2.73×**.

**Honest limits on that.** A positive control was run first — at a shallow barrier where Gillespie
*can* see events, exact MFPT 9.9790 vs Gillespie 10.2722 ± 0.2328 generations, difference 0.2933
against 3 s.e. of 0.6984, PASS. And a gate **failed**: the closed-form-versus-linear-solve check
missed its predeclared 10⁻¹⁰ bar (errors 6.3 × 10⁻¹¹, 1.8 × 10⁻⁹, 1.0 × 10⁻⁹). It passes only
against a condition-number-derived bar, because the problem is genuinely ill-conditioned —
κ up to 2.1 × 10⁹. The right reading is that these MFPTs are good to about 9 significant figures,
not 10, and that the limit is the problem's conditioning rather than the solver.

**Higher friction, higher value.** Before a circuit is built: the fraction of cells in the wrong
state, the flip rate, and how both move under resource competition. That needs parameters from them.

---

## What is NOT claimed

- **Not that their attribution is wrong.** Growth feedback may well account for most or all of what
  they see. The claim is only that a shared driver produces positive correlation without it, so the
  decomposition is worth doing.
- **Novelty is not established.** They may already have run the zero-growth-feedback control.
- The model here is a linear-production shared pool, not their nonlinear circuit-host system. It
  shows a mechanism is *available*, not that it is *operating* in their system.

## UNRETRIEVED

No parameters from their model were retrieved; nothing here is fitted to their system.

## The question, as a question

*In your framework with the growth-feedback coupling set to zero, does any cooperativity survive?
A shared fluctuating resource pool induces positive correlation between competing genes on its own
— we measure corr = 0.65 at tight supply with no growth term in the model at all — so part of the
effect may be structural. Is that decomposition something you have already looked at?*

---
*Computation: `rem/atlas/pool.py` gates P4/P5, output `rem/atlas/RESULTS_pool.txt`;
`rem/switching.py`, output `rem/RESULTS_switching.txt`.*
