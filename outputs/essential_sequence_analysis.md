# Per-protein sequence analysis: essential vs non-essential

**455 proteins** from 1 DEG-benchmark organisms (syn3a), 383 essential vs 72 non-essential.

All features computed DIRECTLY FROM THE AMINO-ACID SEQUENCE, no annotation or conservation used.

## Pooled comparison

| feature | essential mean | non-essential mean | Cohen's d | AUC |
|---|---:|---:|---:|---:|
| cys_frac | 0.007 | 0.011 | -0.380 | 0.378 |
| met_frac | 0.020 | 0.016 | +0.364 | 0.604 |
| aromaticity | 0.092 | 0.101 | -0.268 | 0.417 |
| length | 360.151 | 290.139 | +0.274 | 0.581 |
| pI_approx | 8.200 | 8.077 | +0.059 | 0.539 |
| n_TM_helices_pred | 1.146 | 1.014 | +0.050 | 0.466 |
| has_signal_pep | 0.031 | 0.083 | -0.267 | 0.474 |
| low_complexity_frac | 0.007 | 0.009 | -0.135 | 0.479 |
| charge_pH7 | 7.107 | 5.750 | +0.113 | 0.520 |
| trp_frac | 0.008 | 0.008 | +0.013 | 0.512 |
| gravy | -0.231 | -0.245 | +0.034 | 0.496 |

Reading: |d| > 0.2 small, > 0.5 medium, > 0.8 large. AUC = single-feature essentiality discrimination (0.5 = useless, 1.0 = perfect).
