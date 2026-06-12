# Essential vs non-essential genes — separation analysis

All 52,945 essential genes vs 52,945 randomly sampled non-essential genes (59 organisms, balanced).

Cohen's d: standardized mean difference (|d|>0.8 large, >0.5 medium, >0.2 small). AUC: single-feature essentiality discrimination (0.5 useless, 1.0 perfect). Positive d = higher in essential genes.

| feature | ess mean | non mean | Cohen's d | AUC | MWU p |
|---------|---------:|---------:|----------:|----:|------:|
| family_frac_essential_leakfree | 0.605 | 0.133 | +1.625 | 0.868 | 0.0e+00 |
| family_n_organisms | 35.936 | 23.024 | +0.836 | 0.731 | 0.0e+00 |
| cai | 0.624 | 0.535 | +0.702 | 0.689 | 0.0e+00 |
| family_size_total | 217.410 | 78.460 | +0.385 | 0.678 | 0.0e+00 |
| fba_essential | 0.463 | 0.108 | +0.836 | 0.677 | 0.0e+00 |
| gc3 | 0.731 | 0.644 | +0.369 | 0.622 | 0.0e+00 |
| gc | 0.605 | 0.560 | +0.348 | 0.599 | 0.0e+00 |
| n_paralogs_in_genome | 6.179 | 1.785 | +0.338 | 0.577 | 0.0e+00 |
| same_strand_next | 0.764 | 0.690 | +0.164 | 0.537 | 5.0e-07 |
| same_strand_prev | 0.765 | 0.695 | +0.157 | 0.535 | 1.7e-06 |
| intergenic_prev | 117.916 | 137.732 | -0.073 | 0.466 | 2.5e-06 |
| intergenic_next | 125.620 | 135.689 | -0.035 | 0.466 | 3.5e-06 |
| is_regulator | 0.119 | 0.070 | +0.166 | 0.524 | 0.0e+00 |
| is_transporter | 0.082 | 0.129 | -0.153 | 0.477 | 0.0e+00 |
| fba_n_rxns | 2.605 | 3.321 | -0.132 | 0.485 | 3.4e-01 |
| cds_length | 965.098 | 958.106 | +0.010 | 0.512 | 9.2e-02 |
| is_conditional | 0.207 | 0.217 | -0.024 | 0.495 | 1.0e-02 |
| is_signaling | 0.023 | 0.030 | -0.044 | 0.497 | 6.6e-02 |

## Within single organisms (conservation features removed)

Cross-organism conservation (family_frac) dominates the pooled view. Within ONE organism every gene shares the same phylogenetic context, so this isolates INTRINSIC gene properties (length, codon usage, GC, genomic neighbourhood).

Organisms examined: ['beril_RalstoniaGMI1000', 'mtub', 'beril_SynE', 'beril_MR1']

| feature | beril_RalstoniaGMI1000 | mtub | beril_SynE | beril_MR1 |  (AUC) 
|---------|---:|---:|---:|---:|
| cds_length | 0.395 | 0.623 |   -   |   -   |
| cai | 0.591 | 0.607 |   -   |   -   |
| gc | 0.437 | 0.512 |   -   |   -   |
| gc3 | 0.481 | 0.590 |   -   |   -   |
| intergenic_prev | 0.496 | 0.437 |   -   |   -   |
| intergenic_next | 0.495 | 0.443 |   -   |   -   |
| n_paralogs_in_genome | 0.591 |   -   | 0.542 | 0.551 |
| is_transporter | 0.491 |   -   | 0.464 | 0.475 |
| is_regulator | 0.547 |   -   | 0.500 | 0.525 |
| fba_essential |   -   | 0.720 |   -   |   -   |

## What separates them (plain reading)

- **family_frac_essential_leakfree**: AUC 0.868, HIGHER in essential genes (d=+1.62)
- **family_n_organisms**: AUC 0.731, HIGHER in essential genes (d=+0.84)
- **cai**: AUC 0.689, HIGHER in essential genes (d=+0.70)
- **family_size_total**: AUC 0.678, HIGHER in essential genes (d=+0.38)
- **fba_essential**: AUC 0.677, HIGHER in essential genes (d=+0.84)
- **gc3**: AUC 0.622, HIGHER in essential genes (d=+0.37)