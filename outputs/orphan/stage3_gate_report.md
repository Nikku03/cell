# Stage 3: ESM gene encoder -- real win, plus the next bottleneck found

| encoder | R_dc (gene x cond) | defect_AUC | heldgene_R |
|---|---|---|---|
| classical-MLP | 0.228 | 0.422 | 0.110 |
| ESM-MLP | 0.310 | 0.523 | 0.176 |
| ESM-XATTN | 0.305 | 0.530 | 0.199 |

ESM lifts EVERY metric -- the Stage 2 audit diagnosis (gene encoder = bottleneck)
was correct:
  - held-gene R 0.11 -> 0.20 (nearly doubled: sequence->fitness generalization)
  - defect AUC 0.42 -> 0.53 (crosses below-chance to above-chance)
  - gene x condition residual 0.23 -> 0.31
XATTN ~= MLP (cross-attention adds a sliver on defect/heldgene).

GATE: defect AUC > 0.6 -> MISSED (0.53). Strong condition-specific defects now
detectable above chance but not reliably.

## Recheck found the next bottleneck: the CONDITION encoder
Btheta: 357/542 experiments are STRESS with 118 distinct chemicals.
Keio: 55 stress, 80 distinct condition_1 chemicals.
But the encoder collapsed ALL stresses into a single "has stress" binary bit.
The model cannot distinguish Dimetridazole from Antimycin A -> cannot predict
drug-specific defects -> defect AUC capped near chance.

## Stage 4 = compositional condition encoder
Tokenize the actual stress/condition chemicals (condition_1..4) as embeddable
tokens alongside media compounds. Target: defect AUC > 0.6 via learned
gene<->stress couplings (e.g. efflux gene <-> drug).
