# Combined activator->target: naive fusion HURTS; adjacency alone is best

Tested adjacency (positional) + co-fitness (functional) + family-effector,
head-to-head vs each alone, on Keio RegulonDB activators (>=5 in-Keio targets).

## Overall precision@K
| signal | prec@K |
|---|---|
| co-fitness | 0.071 |
| adjacency | 0.226 |
| combined (z-sum) | 0.155 |

Combining HURT: the noisier cofit signal dilutes the clean adjacency signal.
The signals are REDUNDANT (both pick up local metabolic regulators), not
complementary.

## Per-family precision@K (which signal wins)
| family | cofit | adj | combined |
|---|---|---|---|
| AraC | 0.00 | 0.60 | 0.23 |
| IclR | 0.00 | 0.67 | 0.33 |
| LacI | 0.00 | 0.40 | 0.20 |
| DeoR | 0.71 | 0.43 | 0.71 |
| LysR | 0.11 | 0.19 | 0.16 |
| Sigma | 0.02 | 0.30 | 0.15 |
| Crp/Fur/ArcA/SlyA/Rob (global) | 0 | 0 | 0 |

## Verdict
- Adjacency is the single best activator->target matcher (0.226 prec@K; AraC/
  IclR/LacI 0.4-0.67) -- the classic "regulator next to its operon".
- Co-fitness complements ONLY specific families (DeoR 0.71).
- Naive fusion is wrong (averages the strong signal down).
- Better: ROUTE BY FAMILY -- adjacency default, co-fitness for DeoR-type.
  (Per-family learned weights = overfit on 1-17 activators/family.)
- Global/signaling regulators unrecoverable by ALL signals -- same wall.

The "combining lifts performance" hypothesis fails for precision: signals are
redundant; adjacency alone is best; family-routing is the real refinement.
