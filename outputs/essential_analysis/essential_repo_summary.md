# Essential-gene patterns across 3 minimal Mycoplasma genomes

**Organisms analyzed:** syn3a (JCVI-syn3.0), mgen (M. genitalium G37), mpne (M. pneumoniae M129)  
**Total labeled genes:** 986  
**Pooled essential rate:** 68.2%

## Per-organism counts

| organism | n_genes | n_essential | % essential |
|---|---|---|---|
| syn3a | 455 | 270 | 59.3% |
| mgen | 476 | 375 | 78.8% |
| mpne | 55 | 27 | 49.1% |

## Top features distinguishing essential from non-essential (pooled)

Sorted by |Cohen's d| (effect size). AUC = univariate classifier.

| feature | ess_mean | non_mean | Cohen's d | AUC | p (MW) |
|---|---|---|---|---|---|
| aa_R | 0.034 | 0.021 | +0.59 | 0.681 | 7.7e-20 |
| aa_V | 0.059 | 0.045 | +0.55 | 0.622 | 3.6e-10 |
| aa_K | 0.097 | 0.073 | +0.55 | 0.614 | 5.0e-09 |
| aa_P | 0.027 | 0.019 | +0.51 | 0.635 | 4.5e-12 |
| aa_T | 0.050 | 0.039 | +0.50 | 0.622 | 1.5e-09 |
| aa_G | 0.046 | 0.033 | +0.49 | 0.639 | 2.7e-12 |
| aa_A | 0.053 | 0.040 | +0.47 | 0.629 | 2.5e-10 |
| aa_Q | 0.041 | 0.031 | +0.46 | 0.615 | 5.3e-09 |
| aa_E | 0.054 | 0.041 | +0.44 | 0.618 | 7.8e-09 |
| aa_I | 0.083 | 0.068 | +0.41 | 0.576 | 3.4e-04 |
| aa_M | 0.018 | 0.013 | +0.40 | 0.628 | 7.7e-11 |
| aa_L | 0.094 | 0.079 | +0.38 | 0.554 | 6.2e-03 |
| aa_D | 0.047 | 0.038 | +0.36 | 0.590 | 8.3e-06 |
| aa_S | 0.057 | 0.047 | +0.36 | 0.577 | 1.0e-04 |
| aa_H | 0.016 | 0.012 | +0.34 | 0.617 | 5.8e-09 |
| aa_F | 0.050 | 0.041 | +0.31 | 0.582 | 1.1e-05 |
| aa_N | 0.065 | 0.056 | +0.29 | 0.546 | 2.5e-02 |
| operon_prev | 0.631 | 0.516 | +0.24 | 0.529 | 2.1e-01 |
| aa_Y | 0.030 | 0.027 | +0.21 | 0.547 | 2.4e-02 |
| same_strand_next | 0.875 | 0.799 | +0.21 | 0.605 | 1.3e-20 |

## syn3a: essentiality by function class

| function class | n | n_essential | % essential |
|---|---|---|---|
| Metabolism | 167 | 113 | 67.7% |
| Genetic Information Processing | 192 | 126 | 65.6% |
| Unclear | 88 | 29 | 33.0% |
| Cellular Processes | 7 | 2 | 28.6% |
| Human Diseases | 1 | 0 | 0.0% |

## Words enriched in essential gene product descriptions

| word | n_ess_genes | n_non_genes | ess% | non% | enrichment |
|---|---|---|---|---|---|
| synthetase | 45 | 0 | 6.7% | 0.0% | 66965.29x |
| ligase | 23 | 0 | 3.4% | 0.0% | 34227.19x |
| translation | 13 | 0 | 1.9% | 0.0% | 19346.24x |
| f0f1 | 8 | 0 | 1.2% | 0.0% | 11905.76x |
| glutamyl-trna | 8 | 0 | 1.2% | 0.0% | 11905.76x |
| initiation | 7 | 0 | 1.0% | 0.0% | 10417.67x |
| topoisomerase | 6 | 0 | 0.9% | 0.0% | 8929.57x |
| amidotransferase | 6 | 0 | 0.9% | 0.0% | 8929.57x |
| ion | 6 | 0 | 0.9% | 0.0% | 8929.57x |
| metal | 6 | 0 | 0.9% | 0.0% | 8929.57x |
| beta | 18 | 1 | 2.7% | 0.3% | 8.41x |
| polymerase | 29 | 2 | 4.3% | 0.6% | 6.77x |
| alpha | 17 | 1 | 2.5% | 0.3% | 7.94x |
| atp | 16 | 1 | 2.4% | 0.3% | 7.47x |
| aspartyl-trna | 5 | 0 | 0.7% | 0.0% | 7441.48x |
| ribosomal | 95 | 11 | 14.1% | 3.5% | 4.04x |
| 30s | 19 | 2 | 2.8% | 0.6% | 4.44x |
| dna-directed | 11 | 1 | 1.6% | 0.3% | 5.14x |
| iii | 15 | 2 | 2.2% | 0.6% | 3.50x |
| membrane | 14 | 2 | 2.1% | 0.6% | 3.27x |

## Words depleted in essential (enriched in non-essential)

| word | n_ess_genes | n_non_genes | ess% | non% | enrichment |
|---|---|---|---|---|---|
| oligopeptide | 4 | 5 | 0.6% | 1.6% | 0.37x |
| pts | 3 | 4 | 0.4% | 1.3% | 0.35x |
| serine | 2 | 3 | 0.3% | 1.0% | 0.31x |
| substrate-binding | 2 | 3 | 0.3% | 1.0% | 0.31x |
| helicase | 4 | 6 | 0.6% | 1.9% | 0.31x |
| division | 2 | 4 | 0.3% | 1.3% | 0.23x |
| acid | 3 | 6 | 0.4% | 1.9% | 0.23x |
| rrna | 4 | 8 | 0.6% | 2.5% | 0.23x |
| -methyltransferase | 4 | 8 | 0.6% | 2.5% | 0.23x |
| cell | 1 | 4 | 0.1% | 1.3% | 0.12x |
| pseudouridine | 2 | 7 | 0.3% | 2.2% | 0.13x |
| 16s | 1 | 5 | 0.1% | 1.6% | 0.09x |
| methyltransferase | 1 | 6 | 0.1% | 1.9% | 0.08x |
| peptidase | 1 | 6 | 0.1% | 1.9% | 0.08x |
| amino | 0 | 6 | 0.0% | 1.9% | 0.00x |
