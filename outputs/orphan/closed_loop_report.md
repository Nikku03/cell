# Closing the loop: environment → TF activity → expression → flux → essentiality

The full circuit, end to end, on the textbook cAMP/CRP catabolite-repression
system, wired to iJO1366 FBA.

**Loop:** carbon source (environment) → cAMP level → CRP active fraction →
catabolic operon expression (Gillespie) → OFF genes knocked out in iJO1366 →
growth + conditional essentiality.

## Growth by condition — regulation-aware vs regulation-blind FBA

| environment | CRP | operons ON | growth (loop) | growth (blind FBA) |
|---|---|---|---|---|
| glucose | off | — | 0.98 | 0.98 |
| arabinose | on | araBAD | 0.81 | 0.81 |
| glycerol | on | glpFK | 0.56 | 0.56 |
| maltose | on | malPQ | 1.98 | 1.98 |
| **arabinose + glucose** | off | — | **0.98** | **1.80** |

The diauxie row is the first loop signature: with glucose present, CRP is
repressed, the arabinose operon stays OFF, so the loop grows on glucose only
(0.98) — **regulation-blind FBA wrongly co-consumes both carbons (1.80).** The
loop reproduces glucose preference; pure FBA can't.

## The decisive case — sole carbon = arabinose

| model | growth |
|---|---|
| CRP active (loop) | **0.81** |
| CRP inactive (loop; crp mutant / catabolite repression) | **0.00** |
| regulation-blind FBA | 0.81 |

This is the loop in one line: the cell is **metabolically capable** of eating
arabinose, but if CRP doesn't turn the genes on, it **dies** (0.00). Pure FBA
says it grows (0.81) — it misses the regulatory block. This matches real biology:
*crp* mutants cannot grow on most alternative carbon sources.

## Conditional essentiality (deletion → growth lost?)

| condition | araA | araB | glpK | malQ |
|---|---|---|---|---|
| glucose | ne | ne | ne | ne |
| arabinose | **ESS** | **ESS** | ne | ne |
| glycerol | ne | ne | ne | ne |
| maltose | ne | ne | ne | **ESS** |

Essentiality is **conditional and gated by both layers**: araA/araB are essential
*only* on arabinose, malQ *only* on maltose — and only because regulation turned
them on there. The same gene is dispensable in every other condition.

## Why this matters for the cell model
This is the complete conditional engine the wheels were missing:
- **Wheel 2 (FBA)** answers "is the gene metabolically necessary in this medium?"
- **Regulatory layer** answers "is the gene even ON in this condition?"
- A gene is **conditionally essential ⇔ metabolically necessary AND expressed.**
  Either layer alone gets it wrong (FBA over-predicts capability; regulation
  alone ignores necessity).

Inputs needed are only what we can actually get: the **edge** (RegulonDB /
co-expression-inferred) + the effector→TF logic + the metabolic model — never the
unpredictable binding site. Rate parameters are illustrative; the
feasibility/essentiality logic is exact.

Files: colab/closed_loop.py, outputs/orphan/closed_loop.{png,json}.
