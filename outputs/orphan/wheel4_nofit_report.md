# Wheel 4 for an organism WITHOUT its own fitness data

Test: mtub as stand-in (independent DeJesus truth). Pretend mtub has no fitness;
project fitness-essentiality from relatives via feba orthologs.

| signal | AUC | neCov@P0.9 | on conservation-blind genes |
|---|---|---|---|
| conservation (ortholog labels) | 0.725 | 0.00 | — |
| transferred fitness (relatives via orthologs) | 0.755 | 0.13 | AUC 0.558 (~chance) |
| organism's OWN fitness | (0.819 combined) | 0.75 | AUC 0.717 |

corr(conservation, transferred-fitness) = +0.84.

## Verdict
Transferred fitness COLLAPSES onto conservation. On the genes conservation
misses, transferred fitness is near chance (0.558) vs 0.717 with own data --
the orthogonal power evaporates in transfer. Same pattern as metabolic_transfer
(native orthogonal, OG-transferred redundant). Universal: anything transferred
via orthology becomes conservation, because orthology IS conservation's substrate.

Small positive: transferred fitness edges conservation (0.725->0.755) only because
relatives' direct fitness-essentiality is a cleaner per-gene label than our noisy
'essential_families' labels -- not orthogonality.

## Complete picture
  has own fitness (48 orgs):       breakthrough (mtub +0.10 AUC, neCov->0.75)
  no fitness, has relatives:       ~conservation ceiling (~0.50), marginally cleaner
  no fitness, isolated:            conservation fails too -> near floor

Fitness's orthogonal coverage cannot be borrowed -- it must be measured (~3
TnSeq conditions/organism). Neighbors cannot substitute.
