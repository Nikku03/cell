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

## LOO result + AUDIT (the negative was PREMATURE — corrected)
First LOO run (ESM reduced to 16 PCs): rogue R@P30 = **0.000** cross-org (both
LOO-org and LOO-clade) → looked like a clean negative.

**Sanity-check audit (real multi-org data, sandbox, no GPU) overturned it:**
| LOO feature (GMI1000 held out) | rogue R@P30 | rogue AUPRC (base 0.083) | pred std |
|---|---|---|---|
| conservation (positive control) | 0.000 | 0.094 | 0.29 |
| **composition (len+20 AA)** | **0.129** | **0.154 (1.85×)** | 0.16 |
| ESM **16-PC** (our 1st run) | 0.000 | — | — |

Findings:
1. **Harness is sound** — conservation positive control gives cross-org WHOLE
   AUPRC 0.786, predictions vary (std 0.29). Not degenerate. The 10→8 org drop
   is legit (mtub/saur lack orthology).
2. **A weak but REAL transferable signal exists** — raw AA composition gets rogue
   R@P30 0.129 cross-org (above the 0.0 bar). The "everything is 0" conclusion
   was wrong.
3. **ESM-16-PC = 0 is a PCA ARTIFACT.** ESM ⊋ composition in information, so ESM
   should be ≥ composition cross-org, not 0. Reducing 1280-d → 16 top-variance
   PCs kept protein-family structure and discarded the essentiality-
   discriminative directions. Composition kept its signal because it was never
   reduced.

**Fix applied:** `orphan_esm.py --multi` now saves the FULL 1280-d embedding (no
PCA); `orphan_loo.py` gained `--l2`. Re-embed (one GPU pass) and re-run LOO with
full dims + L2 to get ESM's TRUE cross-org number.

## Verdict logic (after the full-dim re-run)
- ESM-full LOO rogue R@P30 clearly > composition's 0.13 → ESM is a real $0
  cross-org detector of rogue/orphan essentials. Result.
- ESM-full ≈ composition (~0.13) → the transferable signal is weak/compositional,
  not deep protein understanding; honest to report as a weak effect.
- ESM-full ≈ 0 even at full dims → genuinely no transferable signal; the clean
  negative stands and conditional essentiality is environment-intrinsic.

## Still optional
- **DEG1057 cross-grade** — do ESM's top rogue calls agree with the independent
  Ralstonia screen (not just the RB-TnSeq label we trained on)?
- **Foldseek** (step 4) — not yet run; would add structural-homology evidence,
  but ESM already carries the structural signal.
