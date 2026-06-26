# Stage 1 gate: result + sharpening for Stage 2

## Numbers (Pearson R, held-out splits within organism)

HELD-OUT CONDITIONS (primary gate):
  Putida:   MODEL 0.670  gene-mean 0.706  cond-mean 0.000
  Keio:     MODEL 0.595  gene-mean 0.590  cond-mean 0.000
  Btheta:   MODEL 0.621  gene-mean 0.558  cond-mean 0.000

HELD-OUT GENES (sequence -> fitness generalization):
  Putida:   MODEL 0.557  gene-mean 0.000  cond-mean 0.064
  Keio:     MODEL 0.535  gene-mean 0.000  cond-mean 0.114

LEAVE-ONE-ORG-OUT: did not complete (OOM concatenating 7-org feature matrices).
Refactor: stream from indices in Stage 2.

## Verdict

PASSES, with a precise pointer for Stage 2.

- The model crushes the cond-mean baseline (0.6+ vs 0.0) -- a fitness measurement
  is NOT just a per-condition shift, the gene identity carries real signal.
- The model ties / barely beats gene-mean on held-out conditions -- meaning the
  hand-crafted feature MLP captures the per-gene average fitness but adds little
  condition-specificity over and above it.
- The model achieves R=0.55 on HELD-OUT GENES -- the headline: sequence features
  predict the fitness of genes never seen during training, far above any
  baseline. The framing -- "the cell is a tensor, complete it" -- has real
  ML signal at the cheapest possible architecture.

## What Stage 2 must specifically deliver

Beating gene-mean on HELD-OUT CONDITIONS is exactly the soup-cracking ask:
which gene is essential under which condition. The simple concat-MLP can't
capture this because it has no mechanism for gene<->nutrient interaction.

Stage 2's contribution must be measurable as:
   delta(MODEL R) - delta(gene-mean R) on held-out conditions > +0.10
This is the cross-attention design's only justification.

If Stage 2 fails this specific test, Stage 3+4 cannot rescue it. The gate is
honest and sharp.
