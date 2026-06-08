# Flanking-region analysis: anything unique to essential genes?

**Genes:** 32,427 (8,748 essential) across 10 organisms  
**Window:** 300 bp each side; k-mer scan on closest 50 bp

## Flank features (essential vs non-essential)

| feature | ess_mean | non_mean | Cohen's d | AUC | p |
|---|---|---|---|---|---|
| first_in_operon | 0.4603 | 0.5260 | -0.13 | 0.462 | 5.9e-27 |
| up_gc | 0.5832 | 0.5667 | +0.12 | 0.523 | 2.0e-10 |
| up_at_run | 6.9611 | 7.4864 | -0.11 | 0.485 | 9.9e-04 |
| down_gc | 0.5951 | 0.5815 | +0.10 | 0.515 | 2.7e-05 |
| intergenic_prev | 111.9346 | 129.2194 | -0.07 | 0.481 | 6.1e-07 |
| sd_score | 0.6481 | 0.6589 | -0.07 | 0.487 | 2.3e-01 |
| minus35_score | 0.6371 | 0.6416 | -0.04 | 0.470 | 1.7e-40 |
| polyT_run | 2.5947 | 2.5813 | +0.01 | 0.508 | 2.5e-07 |
| sd_spacing | 7.9086 | 7.8991 | +0.00 | 0.502 | 9.0e-01 |
| has_hairpin | 0.7770 | 0.7776 | -0.00 | 0.490 | 1.8e-84 |
| minus10_score | 0.5914 | 0.5911 | +0.00 | 0.501 | 4.2e-84 |

## Top upstream k-mers enriched in essential flanks

| kmer | ess% | non% | enrichment | z |
|---|---|---|---|---|
| AGGTGA | 3.1% | 2.1% | 1.48x | +5.3 |
| CGAAG | 5.7% | 4.3% | 1.32x | +5.2 |
| TCTCC | 3.8% | 2.7% | 1.41x | +5.2 |
| AAGGTG | 2.7% | 1.8% | 1.51x | +5.1 |
| CCTTCC | 2.2% | 1.4% | 1.57x | +5.1 |
| CTAAG | 2.4% | 1.6% | 1.53x | +5.1 |
| TCCGAT | 1.8% | 1.1% | 1.63x | +5.0 |
| TCCTCC | 1.4% | 0.8% | 1.77x | +5.0 |
| TAAGGT | 1.6% | 0.9% | 1.70x | +4.9 |
| GAAGC | 6.0% | 4.7% | 1.28x | +4.8 |
| CTTCC | 6.1% | 4.7% | 1.28x | +4.8 |
| CCAAGC | 2.3% | 1.5% | 1.51x | +4.8 |
| CAAGC | 7.3% | 5.9% | 1.24x | +4.7 |
| GTGCTC | 1.0% | 0.5% | 1.83x | +4.4 |
| CGAAGC | 1.7% | 1.1% | 1.56x | +4.4 |
| AAGCC | 6.0% | 4.8% | 1.25x | +4.3 |
| CTCTCC | 1.2% | 0.7% | 1.70x | +4.3 |
| CTCCTC | 1.0% | 0.5% | 1.79x | +4.3 |
| TCTGGA | 1.4% | 0.9% | 1.59x | +4.2 |
| CGCCT | 8.5% | 7.1% | 1.19x | +4.2 |
