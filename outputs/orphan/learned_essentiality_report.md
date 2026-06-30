# Learned essentiality: how far the full stack gets (night work)

5-fold OOF on E. coli (2295-3038 genes, vs Keio truth). Replaces the naive
OR-fusion (precision 0.51 / recall 0.36) with a learned classifier.

| model | AUC | coverage @90% precision |
|---|---|---|
| conservation only | 0.694 | 49% |
| GENOME-ONLY (conservation+FBA+network, no fitness) | 0.688 | 36% |
| panel-wide universal (conservation, 48 bacteria, leave-clade-out) | 0.785 | 70% |
| NON-ESM measured fusion (cons+FBA+fitness+network) | **0.841** | **99%** |
| ESM-8M only | 0.641 | 0% |
| FULL (ESM + all) | 0.783 | 86% |

## Findings (honest)
1. Learned fusion of MEASURED signals -> AUC 0.841, ~99% genome callable at >=90%
   precision. Huge jump over the OR-fusion. The driver is feba FITNESS (a measured
   essentiality-like signal); it lifts coverage 36% -> 99%.
2. GENOME-ONLY ceiling (no measured fitness): AUC ~0.69, 36% coverage (E. coli);
   universal cross-bacterial (conservation, leave-clade-out): 0.785 / 70%.
3. ESM-8M HURTS within-organism (0.841 -> 0.783): 320 dims overfit on ~2300 genes
   with the tiny model, diluting strong tabular signals. ESM's value is
   cross-ORGANISM transfer (mtub 0.733), where no organism-specific data exists.

## Takeaway
Data is the lever, not a bigger sequence model (at 8M). A data-rich cell -> near-
complete essentiality (0.84/99%); genome-only -> conservation ceiling (0.69-0.79,
36-70% coverage). The system should fuse whatever measured data exists per organism;
ESM contributes specifically in the cross-organism, data-poor regime.

Files: colab/ecoli_esm_embed.py, colab/ecoli_learned_essentiality.py,
outputs/orphan/ecoli_learned_essentiality.json (ecoli_esm.npz gitignored, ~4MB).
