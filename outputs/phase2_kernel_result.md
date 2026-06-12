# Phase 2 kernel feature (phylo x MoA) — fast-pass result

Combined Naresh's stepping-stone (phylogenetic distance weighting) idea with
the chemistry MoA gap the cascade scorer exposed, in ONE feature on top of
og_cpd. Built `kernel_ogcpd_features.parquet` (78.7M rows) and re-ran the
4-fold fast bake-off WITH it.

## General task — biggest single-feature lift so far

| metric (4-fold macro, TEST at true prevalence) | before kernel | with kernel |
|------------------------------------------------|---------------|-------------|
| AUPRC                                          | 0.434         | **0.494** (+0.060) |
| recall@P30                                     | 0.664         | **0.694** (+0.030) |
| de-novo (og_cpd_n==0) recall@P30               | 0.357         | 0.316 (-0.041) |

AUPRC lift is the largest of any feature added. The de-novo dip is within
fold noise and needs the full 43-fold to interpret.

## Bacitracin litmus (HELD-OUT rows only = SB2B fold)

| gene/org      | before kernel | with kernel |
|---------------|---------------|-------------|
| ompR / SB2B   | top 9.5%      | **top 2.6%** |
| envZ / SB2B   | top 11.5%     | top 5.9%    |
| pspB / SB2B   | top 35%       | top 35%     |

ompR/SB2B at top 2.6% held-out is a real, meaningful improvement and the
MoA mechanism is visibly responsible: bacitracin (cell_wall MoA) borrowed
strength from vancomycin / D-cycloserine / cefoxitin observations in
phylo-close relatives, which single-compound og_cpd could not do.

## IMPORTANT honesty caveat (do not over-read)

The litmus block ALSO printed envZ/PV4 (top 1.55%), ompR/PV4 (top 2.6%),
pspB/MR1 (top 2.4%) -- but PV4 and MR1 were in TRAINING in this fast run
(held-out folds were only SB2B, Keio, Phaeo, pseudo5). So those PV4/MR1
percentiles are TRAINING rankings, not held-out evidence, and must NOT be
cited as model performance. Only the SB2B rows are honest here.

The decisive test is the full 43-fold LOO where PV4 and MR1 each become
held-out folds -- only then are their litmus numbers real. Plus a kernel
leak audit (the leave-own-org-out routing for train rows needs verifying:
a train row's own-org kernel still includes phylo-close siblings, which
could smuggle cross-boundary signal in a way the og_cpd leave-own-org-out
fix did not have to handle).

## MoA coverage (from the build)

47-antibiotic curated MoA table assigned a class to a meaningful slice:
ribosome 293K, cell_wall 178K, dna_damage 113K, membrane_detergent 65K,
protein_synthesis 58K, membrane 38K, transcription 27K, folate 7.8K
(og x compound) cells. Carbon/nitrogen-source conditions are NOT yet
MoA-classed -- a natural extension (amino-acid family, sugar class).

## Next

1. Full 43-fold LOO with the kernel feature -> honest macro + honest PV4/MR1
   litmus.
2. Kernel leak audit: train vs test recall delta; confirm leave-own-org-out
   routing doesn't leak phylo-sibling signal across the boundary.
3. If clean and it holds: this is the combined stepping-stone + MoA result,
   and the next feature is functional-module coverage (the bucket-network idea).
