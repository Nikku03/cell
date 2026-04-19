# P12 — Null result notes

Date: 2026-04-19
Status: null result, filed

## Claim tested
Adding an explicit learned gauge-connection field A(x) to a neural PDE should
improve rollout accuracy when the ground-truth dynamics contain genuine gauge
structure (Abelian-Higgs / gauged complex Ginzburg-Landau).

## Setup
* Ground truth: gauged complex Ginzburg-Landau with a nonzero background
  A(x). Three A styles used (swirl, uniform, sinusoidal).
* Model A baseline: conv stack predicts dψ/dt from ψ only.
  Architecture: 3 layers × 3×3 conv, hidden=32, 19,170 parameters.
* Model B gauge: two networks. NetA(ψ) → learned A(x). NetPsi(ψ, A_learned)
  → dψ/dt. Trained jointly. 30,180 parameters (1.57× baseline).

## Result
Both models fit the dynamics essentially perfectly. Rollout error at step 30:
* Baseline: 0.57%
* Gauge:    0.63%

Gauge model lost by ~10%. The learned A(x) collapsed to a near-constant
(std 0.003 vs ground-truth std 0.286) — the gauge network was effectively
unused and its extra parameters hurt slightly via overfitting.

## Diagnosis
The test was not discriminative. The baseline's effective receptive field
(3 layers of 3×3 conv = 7×7 neighborhood) is large enough to implicitly
absorb any smooth gauge-coupled local operator. With the baseline already
able to represent the gauged covariant Laplacian across its receptive
field, the extra factorization of computation into (ψ-network + A-network)
had no function to perform.

## What this does and doesn't mean
What it means:
* On this target at this capacity, the gauge factorization is redundant.
* A network with enough capacity to represent a gauge-coupled operator
  will do so implicitly rather than factoring it into an explicit gauge channel.

What it does NOT mean:
* "Gauge structure in neural networks is useless." Different setups
  (smaller capacity, sharper locality constraint, or gauge structure that
  must be INFERRED from multi-trajectory statistics rather than memorized
  per-trajectory) could yet show an advantage.
* "The Dark Manifold architecture's gauge field idea is wrong." It could
  become load-bearing when combined with constraints that come from later
  prototypes — complex unitary evolution, or retrieval/memory.

## Action taken
Accept as null result. Move to P13 (complex Ψ + unitary evolution). If a
future prototype needs explicit gauge structure to work, revisit P12 with
a better experimental design.

## Implementation lesson
When designing a prototype that's supposed to discriminate architectural
variants, **check that the baseline lacks the capacity/structure to
implicitly do what the variant explicitly does.** Otherwise the test
can't distinguish them. I didn't do this check; next time I will.
