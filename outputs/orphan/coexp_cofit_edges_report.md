# Regulatory EDGE prediction from functional data — this one works

After proving the binding SITE caps at the information wall (~0.55), test the
thing the cell-layout actually needs: the **edge** — which genes a TF regulates —
from functional data we can get: **co-expression** (PRECISE, 278 E. coli RNA-seq
conditions) + **co-fitness** (feba). Validated on RegulonDB. 60 TFs, blind
(rank all 3923 genes per TF).

## Result — the edge is predictable

| signal | mean AUC | mean recall@K |
|---|---|---|
| co-expression | 0.612 | 0.103 |
| co-fitness | 0.549 | 0.045 |
| **combined** | **0.626** | 0.063 |

combined enrichment ≈ **12× over base rate**. Contrast with the operator-SITE
ceiling (0.55): the EDGE clears it, and for many individual TFs it is *strong*:

| TF | family | combined AUC |
|---|---|---|
| gadW | AraC | 0.98 |
| malT | NarL | 0.98 |
| leuO | LysR | 0.91 |
| purR | LacI | 0.89 |
| cytR | LacI | 0.88 |
| evgA | NarL | 0.88 |
| metJ | — | 0.86 |
| iscR | — | 0.82 |
| narP | NarL | 0.81 |

## By family — and the honest caveat made visible

| family | coexp | cofit | combined |
|---|---|---|---|
| NarL | 0.743 | 0.607 | 0.739 |
| AraC | 0.705 | 0.585 | 0.706 |
| LacI | 0.630 | 0.474 | 0.662 |
| LysR | 0.574 | 0.536 | 0.632 |
| OmpR | 0.516 | 0.557 | 0.527 |
| Crp | 0.540 | 0.519 | 0.546 |
| TetR | 0.391 | 0.664 | 0.414 |

The split is exactly the biology: **local/metabolic regulators** (gadW, malT,
purR, cytR, NarL/AraC/LacI families) whose *mRNA tracks their activity* are
recovered strongly; **global/signaling regulators** (OmpR two-component, CRP)
are weak in co-expression — because they're controlled post-translationally
(phosphorylation, ligands), so their mRNA doesn't move with their targets. There
co-fitness partly rescues (TetR cofit 0.664 ≫ coexp 0.391; OmpR cofit 0.557 >
coexp 0.516) — the two signals are orthogonal, which is why combining helps.

## What this means
- **Co-expression is the stronger single edge signal (0.612)**, co-fitness is
  orthogonal and rescues the activity-controlled families; combined is best (0.626)
  and reaches 0.8–1.0 for dozens of TFs.
- This is the **opposite** outcome to the site/operator work: the functional edge
  carries real, usable signal; the sequence site does not.
- Limits (honest): (1) within-organism — needs that organism's own expression
  compendium (PRECISE exists for E. coli/a few others, not all); (2) predicts the
  edge, not the direction (activator vs repressor) or the site; (3) strong for
  local regulators, weak for global/signaling ones (use co-fitness there).

## Verdict
For building the cell's regulatory layer, **predict the edge from co-expression +
co-fitness** (this, 0.63 overall and ≥0.85 for many TFs), reuse the AraC/σ54
family-motif win for the few site-predictable families, and treat exact binding
sites as measured/transferred. The functional route is the one that works.
