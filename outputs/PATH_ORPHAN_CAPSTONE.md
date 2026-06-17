# Path-Orphan — capstone (paper-ready scaffold)

## One-line claim
Under leakage-controlled evaluation, *no $0 feature class — conservation, dN/dS,
co-fitness, full-dimension protein-language-model embeddings — detects
conditional or orphan bacterial gene essentiality across organisms*; the channel
that does (FB phenotype, leak-free conservation, structure where available)
defines a small, honestly-bounded usable region of the genome, and the residual
is the part interventional data must answer.

## Anchors (all measured, in this repo)

| anchor | result |
|---|---|
| Paper 1 — κ-floor of Tn-seq self-agreement | binary κ 0.39, continuous ρ 0.38; **47.7% technical / 45.4% biology**; matched-quality ceiling ρ≈0.77 |
| Conditional-flip decomposition (plan_assessment.md) | org=7.5% + gene=13.6% + **org×gene=79%** |
| Conservation R@P30, rogue zone (cons<0.1) | **0.000** — conservation has zero purchase on rogue essentials by construction |
| dN/dS (Nei-Gojobori) on 3,207 GMI1000 genes | rogue essentials ω 0.158 vs non-essentials 0.160 — **negative** |
| ESM-2 (650M), within-org 5-fold CV | rogue R@P30 = 0.843 |
| ESM-2, leave-one-CLADE-out | rogue R@P30 = **0.000** (with full 1280-d + L2) |
| Composition baseline (length + AA), LOO-org | rogue R@P30 = 0.129 |
| Live-ghost flag enrichment (kill-gate) | orth-breadth 7.1 vs 3.8; ω 0.187 vs 0.445 — **PASS** |
| DEG1057 cross-grade (independent screen) | overall R 0.842, **rogue R 0.343** — independent confirmation of the conditional zone |
| Function-recovery validator (held-out annotations) | context channel **20% vs 1.79% null, p < 0.001** — **PASS** |

## The three novelties (each validated, all $0)

### Novelty 1 — Acute × evolutionary 2×2 lens
Crossing FB-measured essentiality (acute) with ortholog breadth (evolutionary
retention) partitions every gene; all three **non-circular** predictions hold:

| | retained (≥24 orgs) | labile |
|---|---|---|
| essential | **core** n=1050, 0.53 paralog, ω 0.113 | **conditional** n=311, **0.14 paralog**, **0.36 dark** |
| non-essential | **buffered** n=1390, **0.56 paralog** | accessory n=1694, ω 0.185 |

- Buffered paralog-rich vs conditional (0.56 vs 0.14) → redundancy explains retained-but-dispensable. **PASS**
- Conditional darker than core (0.36 vs 0.00) → novel niche genes. **PASS**
- Core under strongest purifying selection (0.113 < 0.185). **PASS**

This is the *structural* reading of the 79% residual.

### Novelty 2 — Fused atlas + live-ghost hit-list
Per-gene integration across all built channels, with confidence tiers + an
honest unknown bucket. On GMI1000 (channels currently merged: dnds, cofit,
coinherit; foldhit and GEM land from the queued Colab steps):

| tier | n | % | meaning |
|---|---|---|---|
| annotated | 3,893 | 88 | known function (control) |
| live_ghost | 306 | 7 | dark **but functional** — prioritized for experiment |
| context_inferred | 49 | 1 | dark, but confound-guarded co-inheritance gives a characterized partner |
| unknown | 197 | 4 | flagged unknowns |

**Kill-gate PASS** — the live-ghost flag enriches for ortholog breadth (7.1 vs 3.8) and stronger purifying selection (ω 0.187 vs 0.445); the validator confirms the integration carries real functional signal at 11× over a shuffled-pair null.

### Novelty 3 — leakage-controlled negative for sequence-based prediction
A clean, audited demonstration that the field's strongest sequence tool (ESM-2,
650M) collapses under honest evaluation: 0.843 within-organism → **0.000
leave-one-clade-out**, with the audit identifying the within-organism result as
homology leak (paralogs of test genes in the training fold) and an interim
"all zero" reading as a PCA artifact (top-variance PCs discard the
discriminative directions). The fix (full 1280-d + L2) gave the honest number.

## Why it works the way it does

Conditional essentiality depends on the **organism's environment and network
state**, not on the gene's sequence. ESM computes f(sequence). For rogue
essentials, the same protein is essential in one organism and dispensable in
another — so no function of sequence separates them. The information required
isn't in the input. Four orthogonal $0 features (conservation, selection,
cofitness, protein-language-model) hit ~0 on the rogue zone cross-clade for the
same reason.

The atlas works precisely because it stops trying to *predict* conditional
essentiality and starts *reading* what's already been measured (FB), corroborated
by structure and context, with an honest "we don't know" bucket for the rest.

## Live-ghost hit-list — wet-lab targets

355 prioritized genes (dark in current annotation, but provably functional:
essential, under purifying selection, or broadly conserved). Top entries are
conserved-DUF and hypothetical proteins essential across 10–26 organisms — the
highest-value targets for single-experiment de-orphaning that propagates to
their entire family. Plus a separate, validated wet-lab lead already in hand:
EnvZ/OmpR + PspB → bacitracin sensitization in *Shewanella* (one MIC plate from
confirmation).

## What this earns

- **Paper 1** (κ-floor) reframed: the corrected reproducibility ceiling sets
  the standard against which any future predictor must be read.
- **Novel benchmark for bacterial foundation models**: the leakage-controlled
  cross-clade test (this work) — within-org 0.843 → cross-clade 0.000 — is the
  honest evaluation the field has been skipping.
- **Multi-organism extension**: the same pipeline, GMI1000-tested, runs on any
  organism with a clean locus-tag bridge to its labels. The atlas generalizes.
- **CRISPRi argument earned, not asserted**: every $0 channel has been honestly
  exhausted on the conditional/orphan slice. The case for interventional data
  is now a measurement, not a hope.

## Honest residual

The deepest orphans — 19 singletons (no relatives) and ~71 virgin conserved
families (no characterized member anywhere) — remain dark from every
computational channel by construction. These are the genes where one wet
experiment per family demasks the entire family computationally. That's the
correct division of labor: computation maps and triages; perturbation reveals.
