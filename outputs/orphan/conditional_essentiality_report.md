# Genome-wide conditional essentiality across growth conditions (iJO1366)

Single-gene-deletion FBA across 7 sole-carbon minimal media. Partition the
genome by how condition-dependent each gene's essentiality is.

## Result (1367 model genes, media: glucose, arabinose, glycerol, maltose, galactose, succinate, acetate)
| class | genes |
|---|---|
| CORE-essential (essential in ALL media) | 207 |
| CONDITIONAL (essential in some, not others) | 26 |
| never-essential | 1134 |

**11% of ever-essential genes are conditional** — their essentiality exists only
in specific environments.

## Validation vs Keio (Baba) truth (n=1036 mapped, 199 essential)
| call | precision | recall |
|---|---|---|
| CORE-essential | 0.455 | 0.432 |
| CORE + CONDITIONAL | 0.460 | **0.487** |
Adding the conditional set raises recall (they are genuine essentials, just
condition-specific) at equal precision.

## Example conditional-essential genes (biologically correct)
- succinate: sdhA, sdhC (succinate dehydrogenase), tpiA
- maltose: malQ, malF, malG, malK (maltose uptake/catabolism)
- arabinose: araB ; galactose: galE
- acetate: atpA, atpD, atpF (ATP synthase - oxidative phosphorylation needed on
  a poor carbon source), tpiA

## Why this matters
This quantifies the magnitude of the conditional-essentiality phenomenon the
regulatory + closed-loop layer exists to capture: a fixed metabolic core plus a
condition-dependent shell. The metabolic layer (FBA + medium) captures the
condition-dependence directly; the regulatory layer adds the gating (a gene must
also be EXPRESSED to be essential), as shown in the closed-loop demo.

Files: colab/conditional_essentiality.py, outputs/orphan/conditional_essentiality.json.
