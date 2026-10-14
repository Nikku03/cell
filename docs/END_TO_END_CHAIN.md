# End-to-end quantitative chain — mutation → phenotype in one pipeline

Every rung was validated in isolation; this connects them so a single point mutation propagates all the way to
a quantitative cell-level readout:

```
mutation → ΔΔG (structure) → folded/active fraction → enzyme capacity → ec-flux (Human-GEM)
         → pathway flux / growth → phenotype severity
```

`colab/end_to_end_chain.py` (`Chain.run(gene, uniprot, pos, wt, mut)`) returns the full trace with a value +
carried confidence at **every** rung, so error compounds visibly rather than hiding.

## What it does, honestly

For a mutation it computes ΔΔG from the AlphaFold structure, converts it to an active-enzyme fraction via the
two-state folding equilibrium, maps the gene to its Human-GEM reaction(s), reduces their capacity, and re-solves
FBA for growth/flux. The output is a mechanistic **what-if**: *if this mutation destabilizes the fold, here is
the quantitative flux/growth consequence.*

## Validation — and the honest negative

Tested on **180 ClinVar-labeled missense variants** (90 pathogenic / 90 benign) across 15 metabolic enzymes,
blind to the label (`colab/validate_chain.py`). Two results, both reported:

**❌ As a general pathogenicity classifier it is at chance — AUC 0.52.** The deployed ΔΔG (biophysical, no
ProteinMPNN) under-calls destabilization (ΔΔG compressed to ±2 kcal/mol), and — more fundamentally — **most
pathogenic missense is not destabilization-mediated** (active-site, catalytic, splicing). A stability chain
cannot see those.

**✅ As the ONE mechanism it models, it is high-precision:** when it fires (ΔΔG > 1), **75% are pathogenic vs a
50% base rate (1.5× lift)**; when it fires strongly (ΔΔG > 1.5), 3/3. But **recall is ~10%** — it only catches
the destabilizing fraction.

So the honest verdict: **a high-precision, low-recall detector of the destabilization→loss-of-function→flux
mechanism — not a pathogenicity oracle.** Scorecard axis `chain_mechanism` gates on the precision-lift claim,
with the AUC≈0.5 negative recorded in the same record.

## The identified bottleneck (and next upgrade)

The chain's ceiling is its **front rung**: ΔΔG is a *stability* predictor (r=0.47), and stability explains only
part of pathogenicity. Swapping in a dedicated variant-effect predictor (AlphaMissense / ESM1v, AUC ~0.85–0.9
on pathogenicity) would lift the whole chain — the composition and the flux propagation are already correct;
only the input rung is weak. That is the clean next step.

## Why it still matters

It is the difference between a database and a model: the pieces now **compose** into an outcome. For a
destabilizing metabolic-enzyme variant it gives a *number* (activity → flux → growth), with every rung's
confidence carried forward and an explicit abstain when a rung can't run.
