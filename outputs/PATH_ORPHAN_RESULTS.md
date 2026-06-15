# Path-Orphan — results (live)

Detecting conditional/orphan essentials (essential & conservation<0.1) in
*Ralstonia solanacearum* GMI1000. Org chosen because Keio's label namespace
joins the cached genome at 0% (disjoint-data wall); GMI1000 joins at 99% and has
the independent DEG1057 screen. 140 rogue essentials / 1,694 rogue-zone genes.

## The bar
Conservation (leak-free family_frac) rogue-zone **R@P30 = 0.000**. Conservation
has zero purchase on rogue essentials by construction. Gate is absolute: any
P≥0.30 head beating the permutation null.

## Feature-by-feature (rogue zone, GMI1000)

| feature | rogue R@P30 | verdict |
|---|---|---|
| conservation (bar) | 0.000 | — |
| **dN/dS** (Nei-Gojobori, 3,207 genes, real) | ~0 | **NEGATIVE** — ess ω 0.158 vs non-ess 0.160; selection ≠ conditional lethality |
| **cofitness** (GeneFitness-derived, leak-free) | 0.000 | **NULL** — only 7 FB experiments for Ralstonia; too thin |
| composition control (length + 20 AA, no GPU) | 0.093 | weak-but-real floor |
| **ESM-2** (combined, within-org 5-fold CV) | **0.843** | **PASS** (provisional) — null p=0.000 |

## What we've ruled out (the result is NOT an artifact)
1. **Not capacity/overfitting** — permutation null (shuffle labels, refit) median
   0.000, p=0.000. The signal is label-dependent.
2. **Not compositional** — length+AA-composition baseline gets 0.093; ESM gets
   0.843 (9× gap). ESM captures real fold/function, not length/AA statistics.
3. **Not ribosome re-detection** — of the 140 rogue essentials, only **3** are
   core translation/replication. Breakdown: 29 transport, 26 metabolic enzyme,
   26 dark (hypothetical/DUF), 5 regulatory, 51 other. Functionally diverse,
   incl. genuine orphans — the conditional-essential profile, not the trivial
   intrinsic-essential one.

## The one threat still standing: cross-organism generalization
The 0.843 is **within-organism 5-fold CV**. The project's gold standard (Paper 1,
HANDOFF leak rules) is leave-one-ORGANISM-out — and the stricter leave-one-CLADE-
out. Within-org CV lets ESM exploit GMI1000-specific proteome structure. This is
exactly the gap that killed the rogue specialist and the coupling model
(high-CV mirages that collapsed cross-org).

**Decisive test built (`orphan_loo.py`):** train on OTHER organisms' ESM+
conservation, predict GMI1000 cold. Smoke-verified to PASS on transferable
signal (0.881) and COLLAPSE on org-specific/memorized signal (0.000) — so its
verdict is trustworthy. Two modes:
- **LOO-organism** (sisters BSBF1503/PSI07 stay in training) — lenient.
- **LOO-clade** (`--clade_regex Ralstonia`, NO Ralstonia in training) — the
  strict ceiling, 7 distant orgs train, predict GMI1000.

Needs `esm_all.parquet` (all orgs embedded together, global PCA) via
`orphan_esm.py --multi` (one GPU pass; 32,392 proteins / 10 orgs available in-repo).

## Verdict logic
- LOO-org/clade R@P30 holds (≥~0.1, well above the 0.0 bar) → ESM is a $0
  detector of rogue/orphan essentials that transfers to novel organisms. Result.
- Collapses to ~0 → ESM memorized org structure; conditional essentiality stays
  environment-intrinsic (only a screen decides). Clean negative.

## Still optional
- **DEG1057 cross-grade** — do ESM's top rogue calls agree with the independent
  Ralstonia screen (not just the RB-TnSeq label we trained on)?
- **Foldseek** (step 4) — not yet run; would add structural-homology evidence,
  but ESM already carries the structural signal.
