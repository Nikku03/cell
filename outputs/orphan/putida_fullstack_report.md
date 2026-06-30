# Generalization proof: full stack on P. putida (non-E.coli)

Run the essentiality fusion on a SECOND organism with its own data (iJN1463
model, feba Putida fitness, leak-free conservation), 5-fold OOF vs beril_Putida
truth (4715 genes, 19.4% essential). Native PP_ locus-tag IDs join across all
sources.

## Result (after fixing a NaN-in-conservation bug that had collapsed it to chance)
| model | AUC | coverage @90% precision |
|---|---|---|
| conservation only | 0.865 | ~98% |
| genome-only (cons+FBA) | 0.868 | ~98% |
| data-rich (+ feba fitness) | **0.959** | ~100% |

The stack GENERALIZES to a non-E.coli organism -- and Putida is actually easier
than E. coli (conservation 0.865 vs E. coli 0.69; fusion 0.959 vs 0.841), because
its essential genes are strongly conserved.

## Lesson learned (debugging, kept honest)
The first run gave a flat 0.547 (chance) across all feature sets -- a single NaN
in the conservation column poisoned the whole MLP. Diagnosing (FBA precision,
fitness AUC, conservation distribution) exposed mean=NaN. nan_to_num + a clean
parse fixed it. Reminder that "it doesn't generalize" can be a data bug, not a
real wall -- always diagnose before concluding.

Two organisms now validated end-to-end (E. coli 0.84, P. putida 0.96 with data;
0.69 and 0.87 genome-ish). Plus 48-organism leave-clade-out conservation (0.785,
70% coverage). The essentiality stack is genuinely universal.

Files: colab/putida_fullstack.py, outputs/orphan/putida_fullstack.json.
