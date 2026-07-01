# Epigenetic regulatory architecture (the missing layer)

Added the epigenetic layer: CpG-island promoter status + ENCODE cCRE counts
(promoter-like/enhancer-like/CTCF, 2.35M elements) near each gene's TSS, integrated with
essentiality, TFs, and disease. Code: `colab/epigenetics.py`.

## Result

| group | CpG-island promoter | enhancer cCREs | promoter cCREs | CTCF |
|---|---|---|---|---|
| **essential (CEG)** | **0.95** | 99 | 6.7 | 3.0 |
| non-essential (NEG) | 0.24 | 53 | 2.7 | 5.0 |
| all genes | 0.58 | 85 | 4.8 | 3.6 |
| TFs | 0.86 | 103 | 5.9 | 3.4 |
| **disease TFs** | 0.87 | **105** | 5.6 | 3.6 |

## Two clean findings that tie the whole story together

**1. Essential/housekeeping genes have CpG-island promoters — 95% vs 24% for non-essential.**
This is a strong, *orthogonal* epigenetic essentiality signal: the vital core is run from
CpG-island promoters (constitutively open, methylation-controlled), the dispensable
periphery is not. It matches every other axis we found (constraint, conservation, structure)
pointing at the same core.

**2. Developmental TFs have the most complex enhancer landscapes.** TFs and disease TFs sit
at the top for enhancer density, and the single most enhancer-dense genes are the **HOX
clusters** (HOXA/HOXB) plus developmental lncRNAs/miRNAs — the classic super-enhancer,
bivalent-domain, body-plan-patterning genes. 17 of the top-50 enhancer-dense genes are TFs.
This is the **epigenetic basis of differentiation**: cell-type/developmental genes are
wired to many enhancers so each cell type can switch them independently — the same-genome →
many-cell-types mechanism, now with its regulatory hardware.

## How it closes gaps we flagged
- The **non-coding disease** we showed is under-counted lives in exactly these elements —
  enhancers/promoters around disease TFs and their targets are the search space for the
  regulatory variants exome sequencing misses.
- The **differentiation** finding (toggle switches) now has its lock-in mechanism: complex
  enhancer landscapes + CpG/methylation set and hold the cell-type state.

## Honest caveats
- ENCODE cCREs are a **cell-type-agnostic union** (SCREEN registry across many cell types),
  so "enhancer count" = total regulatory *potential*, not the active set in one cell. A true
  per-cell-type state needs ChromHMM/Roadmap per tissue.
- Enhancer count is confounded by locus activity/accessibility and gene density (HOX clusters
  are gene-dense), which is why active essential genes also score high — the *specific*
  essentiality signal is the CpG-island promoter (0.95 vs 0.24), not raw enhancer count.
- CpG-island *presence* ≠ methylation *state*; actual methylation is cell-type/disease-specific
  (would need WGBS/TCGA to add).
