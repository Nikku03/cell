# P15 — Null result notes

Date: 2026-04-19
Status: null result, filed — THIRD in a row on modular-factorization mechanisms

## Claim tested
A mixture-of-K-rules architecture with a soft-assignment network should
outperform a single-rule baseline on a task with spatially-switching
reaction dynamics, by clustering behaviors and assigning the right rule
per region.

## Setup
* Target: 2D reaction-diffusion with 3 regions, each using a different
  reaction (logistic, cubic, sign-flipped). Task defined on a 32×32 grid
  with stripe and quadrant region patterns.
* Baseline: single conv PDE network (like P11). 18,849 parameters.
* Mixture: assignment network + 3 shared rule heads + weighted combination.
  10,022 parameters.

## Result
Rollout error at step 20: baseline 27.66%, mixture 26.74%. Mixture
improved by 3.3% relative — within noise, not meaningful.

Mixture diagnostics:
- Rule-to-region agreement: 38.26% (chance = 33% for 3-way arbitrary labels)
- Mean peak weight: 0.334 (uniform = 0.333)
- Rule usage: [0.186, 0.044, 0.770] — mode-collapsed: rule 2 does most work,
  rule 1 barely used
- Diversity loss was active but did not prevent collapse

## Diagnosis
The same pattern as P12 and P14: the mechanism was mathematically present
but never received gradient signal to use itself. Three contributing
factors:

1. The baseline's 7×7 receptive field covers enough spatial extent to
   see region boundaries directly and learn region-specific responses
   implicitly.
2. The diversity loss I added was too weak to prevent collapse against
   a main loss that's easily minimized by one dominant rule.
3. The reaction differences between regions (logistic vs cubic vs
   sign-flipped) are local — the baseline only needs to see the current
   u value and position to infer which region it's in, because the
   gradient `∂du/∂t / ∂u` differs between regions.

Even with multi-region dynamics, the local convolutional baseline
absorbed the structure. Rule separation as an explicit mechanism
received no gradient pressure to do work.

## The three-null pattern

This is the third consecutive null result on modular-factorization
mechanisms:

| Prototype | Mechanism proposed             | Result | Mechanism status |
|-----------|--------------------------------|--------|------------------|
| P12       | Explicit gauge/connection field| NULL   | Collapsed to constant A |
| P14       | Retrieval-augmented memory     | NULL   | Attention collapsed to uniform |
| P15       | Mixture-of-K rules             | NULL   | Mode collapse to 1 effective rule |

Contrast with the two passes:

| Prototype | Mechanism proposed             | Result | Mechanism structure |
|-----------|--------------------------------|--------|---------------------|
| P11       | Local field evolution          | PASS   | Applied monolithically to every layer |
| P13       | Structural unitary evolution   | PASS   | Applied monolithically to every step |

**The pattern:** monolithic structural invariants (P11, P13) are load-
bearing. Modular factorizations that are "sometimes active, sometimes
not" (P12, P14, P15) get absorbed by the baseline's implicit capacity.

## What this means
Three matched-design experiments with the same failure mode is no longer
coincidence. For neural PDE tasks, **explicit factorization of
computation is rarely load-bearing against a competent conv baseline,
even when the target has structure that would seem to demand it.**

This does not falsify the Dark Manifold design entirely. It narrows the
load-bearing claims to monolithic constraints (unitarity, conservation
laws, perhaps local field evolution itself). Modular pieces (gauge
decomposition, memory banks, rule banks) either:
- Are redundant on PDE targets, OR
- Need task designs where the baseline *cannot* succeed at all, which
  is a hard design problem and may not exist in the neural-PDE space.

## Action taken
Accept as null. Document the three-null pattern as a substantive
negative result in the consolidated findings. Recommend returning to the
cell-simulator track where monolithic architectural primitives (P11
locality, P13 unitarity) can be applied, rather than pursuing more
modular-mechanism prototypes that are likely to null similarly.
