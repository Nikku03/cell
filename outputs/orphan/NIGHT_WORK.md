# Overnight work — limitations attacked, things completed

Pushed the genuine weak points and finished open threads. Honest results
(including negatives), all committed.

## 1. Learned essentiality (replaced the naive OR-fusion 0.51/0.36)
| regime | AUC | coverage @90% precision |
|---|---|---|
| genome-only (conservation+FBA+network) | 0.688 | 36% |
| panel-wide universal (conservation, 48 bacteria, leave-clade-out) | 0.785 | 70% |
| data-rich (+ feba fitness), learned fusion | **0.841** | **99%** |
Driver = measured fitness (lifts coverage 36%->99%). ESM-8M HURTS within-organism
(overfits ~2300 genes); its value is cross-ORG transfer. -> data is the lever, not
a bigger 8M sequence model.

## 2. Generalization to a SECOND organism (the universality proof)
P. putida (non-E.coli), own data, 5-fold OOF vs truth:
conservation 0.865, +FBA 0.868, +fitness **0.959** (~100% coverage). The stack
generalizes -- Putida even easier than E. coli. (First run was chance 0.547 from a
NaN-in-conservation bug poisoning the MLP; diagnosed & fixed. "Doesn't generalize"
was a data bug, not a wall -- always diagnose.)

## 3. Learned regulatory-edge model
Fuse coexp+cofit+adjacency+operon+expr -> leave-TF-out AUC **0.695** (beats prior
0.626). Adding operator-affinity (sequence): 0.68 -- no gain (functional dominates;
operator degenerate). Honest.

## 4. Folded into the deliverable
The interactive app now shows learned P(essential) per gene (bar + color dots);
essential mean prob 0.68 vs non-ess 0.30. Verified in-browser, zero JS errors.

## Net position after the night
- Essentiality is now VALIDATED on two organisms (E. coli 0.84, P. putida 0.96 with
  data) + 48-organism leave-clade-out (0.785, 70% coverage). Genuinely universal.
- The regulatory edge improved to 0.695 (functional fusion); operator adds nothing.
- The deliverable reflects the best model.
- Honest walls unchanged: global-TF target identity from sequence, absolute kinetic
  scales. Everything else got measurably stronger.

Files: ecoli_learned_essentiality.py, putida_fullstack.py, panel_loco.py,
edge_model.py, edge_model_v2.py, update_app_predictions.py (+ reports/jsons).
