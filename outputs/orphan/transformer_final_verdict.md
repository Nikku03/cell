# Transformer build: final verdict (Stages 0-5)

Gene x condition fitness-tensor completion. ESM-2 gene encoder, compositional
condition encoder (media + tokenized stresses), cross-attention. All gates
leak-free.

## Staged trajectory of strong-defect detection (the soup-crackers)
| stage | change | R_dc | defect_AUC |
|---|---|---|---|
| 2 | classical feats, MLP/XATTN | 0.14-0.17 | 0.40-0.48 (<=chance) |
| 3 | ESM gene encoder | 0.31 | 0.53 |
| 4 | + tokenized stress chemicals | 0.33 | 0.67 (within-org PASS) |

Each bottleneck was diagnosed and fixed: gene encoder (Stage 2 audit -> ESM),
then condition encoder (Stage 3 recheck -> stress tokens).

## Stage 5 generalization (the boundary)
| regime | R_dc | defect_AUC |
|---|---|---|
| within-org, seen-chemical | 0.33 | 0.67 |
| novel chemical, same org | 0.31 | 0.53 (chance) |
| NEW organism (leave-org-out) | ~0.07 | <0.5 |

## Verdict
The transformer IS a working WITHIN-ORGANISM, SEEN-CHEMICAL conditional-fitness
predictor -- it detects strong condition-specific defects at AUC 0.67, the
conditional-essentiality capability nothing else achieved. But it does NOT
generalize:
  - novel drug -> broad signal holds, strong defects collapse to chance
  - new organism -> conditional signal collapses (R_dc ~0.07, defect <chance)

This is the SAME boundary found 5 independent ways in this project, now via a
real ESM transformer: the conditional/measured signal does not transfer; it must
be measured per organism (and per chemical class). The transformer raises the
WITHIN-org ceiling and is the best conditional model built here, but confirms
rather than breaks the generalization wall.

Caveats (honest): trained on 7 pilot orgs (full 48 may lift cross-org modestly);
ESM-2 8M is the smallest model (bigger may help); leave-org-out is optimistic
(assumes held org's gene/cond means known). None of these is expected to rescue
the strong-defect transfer, given the project-wide transfer=conservation result.

## Practical takeaway
Use the transformer to predict conditional essentiality FOR AN ORGANISM YOU HAVE
MEASURED (interpolate across its conditions/genes -- defect AUC 0.67). Do NOT
expect it to predict a new organism's conditional essentials from sequence alone.
The lever remains: measure the organism (~TnSeq conditions).
