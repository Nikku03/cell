# Stage 2 audit: conditional signal is real but weak; strong defects at chance

Aggregate R was Bayes-dominated by gene-mean. The correct gate is the residual
(fit - gene_mean) + strong-defect detection AUC.

| org | model | R_resid | defect_AUC |
|---|---|---|---|
| Btheta | MLP | 0.237 | 0.447 |
| Btheta | XATTN | 0.213 | 0.418 |
| Keio | MLP | 0.088 | 0.224 |
| Keio | XATTN | 0.174 | 0.535 |
| Putida | MLP | 0.107 | 0.524 |
| Putida | XATTN | 0.134 | 0.491 |
| mean | MLP | 0.144 | 0.398 |
| mean | XATTN | 0.174 | 0.481 |

## Findings
1. Conditional signal REAL but WEAK: R_resid 0.14-0.17 (>0). Framing not dead.
2. Cross-attention HELPS modestly (XATTN 0.174 > MLP 0.144) -- invisible under
   aggregate R; the original gate metric was misleading.
3. Strong-defect detection AT CHANCE (defect_AUC ~0.48-0.50). The rare strong
   condition-specific essentials -- the soup-crackers -- are NOT detectable.
4. CONFOUND: residual still contains condition-global harshness, so the truly
   gene-specific signal is weaker than 0.17; defect AUC confirms ~0.

## Diagnosis
Bottleneck = the GENE ENCODER. Classical AAC/dipeptide/Pfam gave held-out-gene
R ~0.03-0.17 (leak-free). Can't predict WHEN a gene matters without capturing
WHAT it does. ESM is the fix.

## Stage 3 sharpened gate
- double-centered residual (subtract gene-mean AND condition-mean)
- defect AUC > 0.6 on strong condition-specific defects
If ESM + genome context can't move those, gene-specific conditional
essentiality is a feature/data wall, not a model wall.
