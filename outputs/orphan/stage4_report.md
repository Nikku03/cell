# Stage 4: compositional condition encoder -- PASSES the defect gate

Tokenized condition_1..4 stress/nutrient chemicals (265 distinct) as embeddable
tokens alongside media compounds; ESM-XATTN attends over media+stress.

| condition encoder | R_dc | defect_AUC |
|---|---|---|
| media-only | 0.289 | 0.618 |
| +stress | 0.333 | 0.670 |

Per-org +stress lift: Keio defect 0.651->0.722, Putida 0.599->0.687, Btheta R_dc
0.312->0.354. GATE defect_AUC>0.6: PASSED (0.670).

Trajectory of strong-defect detection (the soup-crackers):
  classical feats : 0.42 (below chance)
  ESM gene encoder: 0.53 (above chance)
  +stress tokens  : 0.67 (reliable)

## Recheck caveat
Held-out-condition split can place the SAME chemical in train and eval
(different concentration/replicate). So 0.67 = "seen-chemical, held-out-
experiment" generalization, NOT "novel chemical". Stricter tests (hold out whole
chemicals; leave-one-ORGANISM-out; independent DeJesus essentiality) are Stage 5.
