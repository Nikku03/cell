# VUS pathogenicity classifier — fuse everything we built

Scores Variants of Uncertain Significance by fusing the signals established across the
project: ESM variant-effect + AlphaFold burial/pLDDT + UniProt functional-site proximity +
gnomAD gene constraint (LOEUF). Trained on ClinVar pathogenic vs benign, applied to VUS.
16 disease genes. Code: `colab/vus_classifier.py`.

## Result
- 17,122 missense variants: **3,706 labeled** (3,020 pathogenic, 686 benign), **13,416 VUS**.
- **Fusion classifier 5-fold OOF AUC = 0.886.**

| feature | single-feature AUC | weight |
|---|---|---|
| burial (contacts) | **0.840** | +0.79 |
| pLDDT | 0.788 | +0.62 |
| gene LOEUF | 0.710 | +0.58 |
| dist to functional site | 0.680 | ~0 |
| ESM variant-effect | 0.649 | −0.83 |
| at functional site | 0.576 | +0.16 |

**Structural burial alone predicts pathogenicity at 0.84** — a buried change is the single
biggest red flag; the fusion adds constraint + ESM severity on top.

## VUS scoring
13,416 VUS scored; top hits are classic buried destabilizers — LDLR W490S/W533G/W577C
(losing buried tryptophans), STK11 L183P, PTEN I122N/H123P (buried, ESM −8 to −13). Output
in `vus_scored.csv`.

## Honest caveat
Training is 81% pathogenic (disease-gene ascertainment), so the 0.5 threshold over-calls
pathogenic (67% of VUS flagged). The **ranking** (AUC 0.886) is the solid deliverable; the
absolute threshold needs calibration to a realistic prior before clinical use. Not a
diagnostic — a triage/prioritization score.
