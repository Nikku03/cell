# Field-equation test on real E. coli data

Test: for each of 60 TFs with ≥10 RegulonDB-mapped targets, extract 250 bp
upstream of every real target and an equal pool of random non-target genes.
Score each window with two equations. AUC = "can the equation rank real
targets above random non-targets?"

| version | mean AUC | global TFs (n=9) | local TFs (n=51) |
|---|---|---|---|
| **Original** (Φ·Ψ · W · C, λ tuned per TF, charitably) | **0.658** | 0.535 | **0.679** |
| **Factored** (affinity only, abundance outside) | 0.505 | 0.497 | 0.506 |
| **Sequence-only** baseline (Φ·Ψ alone, no W, no abundance) | 0.505 | 0.497 | 0.507 |

The original wins on paper. Let me explain why, because **the reason is not
field physics — it's the W(x) term smuggling in genomic adjacency.**

## Diagnosis

**1. Φ·Ψ alone (the field-overlap core) is essentially at random.** Sequence-only
AUC = 0.505 — the family-consensus-rendered-as-a-Φ-field plus the DNAshape-style
Ψ track barely beats coin flip. Reason: family consensus is **too degenerate**
to identify real binding sites without a measured PWM. We knew this; the
test confirms it on real data. The "continuous biochemistry field" idea is
correct in spirit but needs actual measured affinity matrices to do work.

**2. W(x) carries the entire win.** Best λ values landed mostly at 5 kbp –
500 kbp; for many local TFs the optimal λ collapses to ~5 kbp, which is
**literally adjacency**. The original integral, with the tuned λ, isn't
finding sequence-driven binding sites — it's finding **the same divergent-pair
adjacency** our previous wheel found at precision 0.51. Compare:
- original equation on local TFs: AUC 0.679
- our pure adjacency predictor on local TFs: precision ~0.51, recall 0.12

The "win" of the original is the adjacency signal we already had. The field-
overlap term contributed nothing.

**3. The factored version reduces to seq-only by construction here.** When
abundance is a scalar per TF and accessibility C is constant, both terms add
the *same* shift to every score for that TF — and AUC is invariant to
per-TF shifts. So per-TF AUC literally cannot distinguish factored from
seq-only here. This is honest, not a bug: it means the layers I peeled
into occupancy don't change ranking *within one TF*, only ranking *across
conditions*. To see their effect you'd need a condition-resolved test.

## What the test actually decided

| claim | verdict |
|---|---|
| "Field overlap Φ·Ψ filters out decoys without string matching" | **failed** — without real PWMs, family-level Φ is too soft (AUC 0.505) |
| "Wave-field W(x) helps via 'local concentration after translation'" | **half-right** — W(x) does boost AUC, but the optimal λ is 5–50 kbp, i.e. **structural adjacency**, not the millisecond diffusion field the original physically posits |
| "Putting W and C inside ΔG is wrong" | **upheld in the formal sense** — the per-TF AUC can't distinguish factored from seq-only, but the original equation's win comes entirely from W's adjacency role, which is a *structural* property of the genome, not an *affinity* property of the site. Both are improved by moving W out of ΔG and replacing it with adjacency-as-a-prior |

## The honest synthesis

Both equations as written **rediscover the adjacency signal we already had**
(0.51 precision / ~0.68 AUC). Neither demonstrates that the continuous
biochemistry field is doing the work it's supposed to do. To test the field
idea **as field physics** rather than as a smuggled adjacency, we'd need:

1. **Real measured PWMs** for the TFs (CollecTF/PRODORIC) so Φ encodes actual
   specificity, not family consensus.
2. **Cross-condition tests** to expose where the factored version *should*
   win — when [TF] or C(x) genuinely differ across conditions, the original
   (which bakes them into ΔG) gets stuck on the training condition, while
   the factored version generalizes. We can't do this without condition data.

## What this means for plans (1) and (2)

- **(1) DNAshape features**: justified — adds physical content to Ψ_DNA. But
  on this test it would lift seq-only from 0.505 toward maybe 0.55–0.60 *only*
  if combined with real PWMs. As a standalone fix it won't move the headline.
- **(2) SE(3)-equivariant pair predictor**: still the right architecture for
  spacer geometry, but the test says **the binding payoff is gated by having
  real per-TF specificity data**, not by the spacer enforcement. We could
  build it cleanly and prove the equivariance works mathematically, but it
  won't beat adjacency without measured affinity inputs.

So: the equation test is the **most honest negative result on this
direction**. The intuition that "continuous fields beat string matching" is
right *in principle*, wrong *in practice without per-TF affinity data*. The
real lever is still the same one we've been seeing — measured per-TF data —
just now reached from the field-physics direction instead of the
condition-fitness direction.
