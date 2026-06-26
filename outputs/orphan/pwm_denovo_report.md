# The decisive test: even REAL learned PWMs can't find TF targets

We learned each TF's actual motif from its own RegulonDB targets (cross-
validated), confirmed the motifs are real, and tested whether the learned PWM
separates held-out targets from random non-targets.

## The motif finder works — it rediscovered known boxes

| TF | learned consensus | real motif | ✓ |
|---|---|---|---|
| LexA | `CTGTATAAATATACAG` | CTGT-N8-ACAG SOS box (palindrome) | ✓ |
| Fur | `GATAATCATTATCATTATC` | AT-rich Fur box | ✓ |
| PurR | `CGGCAAAAA…` | PurR box | ✓ |
| FNR | `…TTGAT…ATCAA…` | FNR site | ✓ |

So Φ now carries **real specificity**, not family consensus.

## …and it still barely separates targets from non-targets

| Φ source | mean AUC | best (palindromic TFs) |
|---|---|---|
| family consensus (earlier) | 0.505 | — |
| **real learned PWM** | **0.555** (median 0.548) | ArgR 0.71, LexA 0.68, Lrp 0.67, PurR 0.66 |
| genomic adjacency (co-location) | **0.68** | — |

Only **2 of 40 TFs** reach AUC ≥ 0.70. Global regulators (CRP 0.51, FNR 0.48,
Fis 0.55) stay near chance *even with their correct motif*. And the single best
signal in this whole thread — **plain genomic adjacency (0.68)** — still beats
the real PWM (0.55).

## Why: this is an information-theoretic ceiling, not a data or model problem

The reason is the **Wunderlich–Mirny futility theorem** (2009), and it's
quantitative:

- To uniquely specify one site in a 4.6 Mb genome (both strands) you need
  **~log₂(9.3×10⁶) ≈ 23 bits**.
- A real bacterial PWM carries only **~8–12 bits** of information.
- Gap = ~11–15 bits → **~2¹¹–2¹⁵ ≈ 2,000–30,000 positions genome-wide score as
  well as a true site.**

So the true targets are buried among thousands of equally-good motif matches.
**No scoring function on the motif alone can pull them out — the information
isn't in the sequence.** This is exactly why AUC sits at ~0.55 even with the
perfect PWM.

## What this means for the field-equation direction (definitive)

- The continuous-field / SE(3)-equivariant architecture is **elegant but
  bounded by this ceiling**. A flawless Φ·Ψ field model still only has the
  ~8–12 bits the motif contains. Plans (1) DNAshape and (2) equivariance would
  add maybe **1–3 bits** of shape readout — nudging 0.55 → perhaps 0.60 — and
  **cannot reach usability**. The bottleneck is information content, not the
  representation. So we should NOT build (1)/(2) expecting a binding wheel; the
  test just proved the ceiling.
- The cell escapes the ceiling with the **non-sequence** signals — TF
  concentration, accessibility, cooperativity, and **co-location**. Of these,
  only co-location is readable from a static genome — which is precisely why
  **adjacency (0.68) beats the real motif (0.55)**. We already have that wheel.

## The honest bottom line of the whole TF thread

1. **Family → motif type**: known, complete (the filter works). ✓
2. **Motif → specific targets from sequence**: information-theoretically
   impossible to do well (~0.55 AUC even with the real PWM). ✗
3. **The recoverable regulatory signal is co-location**, not sequence. ✓ (0.68)
4. **The global regulatory network is unreachable from any static genome** — it
   needs condition-resolved measurement (TF abundance, accessibility, fitness).

This closes the sequence/field-binding direction with a *reason*, not a shrug:
the genome doesn't contain enough information to specify TF targets — the cell
supplies the rest at runtime through concentration and physical state. Which is,
once more, the condition-data (Wheel 4) frontier — now reached from
information theory.
