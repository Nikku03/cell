# Can metabolic necessity reach the organisms with no FBA model?

**The question (yours):** only 15 / 60 of our organisms have (or can borrow) a
genome-scale metabolic model. What about the other 45 — they're stuck on two
wheels. Metabolic necessity is a property of the *reaction / orthogroup*, not the
organism, so in principle: if gene G is FBA-essential in the modeled organisms
that carry orthogroup OG, then a gene in OG in an *unmodeled* organism should
inherit a metabolic-necessity prior.

**What we built & ran.** Single-gene-deletion FBA (rich medium) on 4 *distinct*
models covering 4 distinct organisms, then aggregated FBA-essentiality per
orthogroup and propagated it to every organism:

| model | organism | bridge | FBA-essential / genes |
|---|---|---|---|
| iJN1463 | P. putida | native | 197 / 1462 |
| iML1515 | E. coli (Keio) | b-number | 105 / 1516 |
| iEK1008 | M. tuberculosis | native (OG) | 204 / 1008 |
| iYL1228 | K. oxytoca | gene-name | 58 / 1229 |

→ `og_fba_rate` for **1,578 orthogroups**. Validated on the **35 unmodeled
organisms** with ≥30 labelled essentials *and* ≥30 genes hitting those OGs.

## Result: it does not help. Conservation already owns this signal.

| signal | mean AUC (35 unmodeled orgs) |
|---|---|
| FBA-transfer (OG-propagated) | **0.653** |
| cross-organism conservation (Wheel 2) | **0.871** |
| combined (max of the two) | 0.853 |

**Combining is even slightly worse: −1.8pp vs conservation alone.** FBA-transfer
loses to conservation in **35 / 35** organisms (deltas −0.13 to −0.32).

## Why — and it's the same lesson as the gap-fill report, sharper

The orthogroups where transferred FBA fires are **exactly the conserved
core-metabolic OGs** — heme, riboflavin, NAD, peptidoglycan, central carbon.
Those are the genes conservation *already* calls best, because they're conserved
and essential across the whole panel. So the FBA-transfer score is a **noisy
subset** of what conservation sees:
- conservation is built from **real essentiality labels across ~55 organisms**;
- FBA-transfer is a **model prediction from only 4 organisms**.

Predicting a conserved signal from 4 models will never beat measuring it across
55 genomes. Taking `max(fba, cons)` just injects FBA's false positives into
conservation's cleaner ranking → the small negative lift.

## The real takeaway

Metabolic necessity is genuinely orthogonal **only when you have the organism's
own model** (native FBA), where it catches organism-specific holes conservation
mistransfers — that's the Putida pyrimidine-pathway win from the gap-fill report.
Once you try to *transfer* that necessity across orthogroups to an unmodeled
organism, you collapse onto the conserved core, which conservation already
measures directly and better.

**Conclusion:** for the 45 organisms without a model, do **not** add an
OG-transferred FBA prior — it's redundant and slightly harmful. The path to a
genuinely new signal for them is **condition-specific fitness data** (feba.db,
Wheel 4), not a transferred metabolic prior. Negative result, cleanly measured.
