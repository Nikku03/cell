# Residence times and the memoryless assumption — Marr & Theis (Helmholtz Munich)

**Their paper.** According to PubMed, Strasser M, Theis FJ, Marr C, "Stability and multiattractor
dynamics of a toggle switch based on a two-stage model of stochastic gene expression",
*Biophysical Journal* 102:19–29 (2012), PMID 22225794,
[DOI](https://doi.org/10.1016/j.bpj.2011.11.4000).

**What they found, in their own words:**

> "Contrary to the expectation from a deterministic description, this switch shows complex
> multiattractor dynamics without autoactivation and cooperativity."

> "the residence times of the system in one of the committed attractors are geometrically
> distributed"

— with an analytical expression for the parameter of that distribution.

**Why this matters beyond their paper.** Geometric residence means **memoryless**: how long a
circuit has already held its state tells you nothing about how much longer it will hold. That is
the assumption under every circuit-reliability figure, because it is what licenses summarising a
switch by a single rate.

---

## 1. The question that was asked

Their derivation is for a model in which every elementary step has exponential waiting. Real
circuits contain multi-step gating reactions — assembly, maturation, multimerisation,
translocation — whose waiting times are Erlang, not exponential.

**Does the memoryless result survive that?**

Tested on an autoactivating bistable gene by replacing exponential waiting on the production step
with Erlang-*k*. Each substep fires at rate *k*·birth(*n*), so the **mean production flux is held
exactly fixed** and only the *shape* of the waiting time moves. Any movement is then attributable
to shape, never to flux.

**Instrument validated first.** Started from the quasi-stationary distribution, the exit time is
exponential by construction, so its coefficient of variation must be 1. Measured: 1.000000000 at
every *k*, worst |CV − 1| = **3.8 × 10⁻¹⁵**. Only then is any other row allowed to mean anything.

## 2. The answer: it survives, and it gets stronger

Started from where the circuit actually sits — the stationary law restricted and renormalised to
the basin:

| k | mean residence (generations) | exit-time CV | \|CV − 1\| |
|---|---|---|---|
| 1 (exponential) | 1.866 × 10³ | 1.000330 | 3.3 × 10⁻⁴ |
| 2 | 1.261 × 10⁴ | 1.000052 | 5.2 × 10⁻⁵ |
| 4 | 6.988 × 10⁴ | 1.000010 | 1.0 × 10⁻⁵ |
| 8 | 2.456 × 10⁵ | 1.000003 | 3.0 × 10⁻⁶ |

**Memorylessness is not fragile here — it is more exact with a multi-step gate than without one.**
The predeclared gate (broken if any departure exceeded 3× the exponential-case deviation) is met in
the surviving direction, worst ratio 1.00×.

That is a **null result on the question as posed**, and it is reported as the answer rather than
retired in favour of a better-sounding one. It is also, for them, good news: the geometric result
appears more robust than its derivation requires.

## 3. But the parameter moves two orders

The distribution stays exponential. **Its rate does not.**

> Mean residence spans **1.866 × 10³ → 2.456 × 10⁵ generations** across k = 1 to 8 —
> **2.1193 orders, a factor of 132** — at *identical* mean production flux.

So: the *form* is robust to waiting-time shape and the *rate* is not. A reliability spec computed
from an all-exponential model can be wrong by two orders **while the residence-time histogram looks
perfect** — the very diagnostic one would use to check it is the one that cannot see the error.

The direction is unsurprising once stated: more substeps means less production noise, so fewer
excursions over the barrier and longer residence. **The contribution is the magnitude, not the
sign**, and the magnitude is not something a shape-blind fit recovers.

## 4. A correction to a claim that was being carried

An earlier framing of this offer asserted that non-memoryless gating moves the tail by **2.04 to
2.80 orders**, and that the effect is confined to the gating step. Both parts are wrong.

Those were **specification values that had never been measured** — the module that was supposed to
measure them (`rem/atlas/gapdetect.py`) contained an unterminated string literal and had never once
parsed. Its results file held only a traceback. Fixed and run for the first time:

- measured **1.61 orders** on the identified gating step and **1.43** on the other
- separation **1.1×** against a predeclared bar of 5× — **gate GD2 FAILED**

The reason is structural: in a linear cascade **every** step lies on the causal path to the
observable, so there is no off-path step whose shape can be safely assumed exponential. The
actionable rule is not "model the gating step's shape" but "find out whether your circuit has any
off-path steps at all — in a linear cascade it has none, and then every shape assumption is
load-bearing."

---

## What is NOT claimed

- This is one autoactivating gene, **not their two-stage toggle**. It tests whether the memoryless
  *property* is robust to waiting-time shape — not whether their analytical expression is right.
  It is right, for the model they state it for.
- **Novelty is not established.** They are computational biologists at a major institute and may
  well have this already. It is sent as a question, not a finding.
- The 132× figure is for this circuit at these parameters. The transferable claim is that the
  gap between form-robustness and rate-sensitivity exists, not its size in any other system.

## The question, as a question

*Your geometric residence-time result appears to survive a multi-step gating reaction — in our test
it becomes more exact, not less. But the rate of that geometric distribution moved by a factor of
132 at identical mean production flux, purely from waiting-time shape. Is the robustness of the
form already known, and is the sensitivity of the parameter something you would expect?*

---
*Computation: `rem/atlas/residence.py`, output `rem/atlas/RESULTS_residence.txt`;
`rem/atlas/gapdetect.py`, output `rem/atlas/RESULTS_gapdetect.txt`.*
