# Human essentiality from POPULATION CONSTRAINT — the signal that works

The FBA metabolic wheel failed on human (17% coverage, 0/26 drug targets). The fix,
proposed and confirmed: read essentiality off **human population variation** — genes
depleted of mutations across many people can't afford to be broken, i.e. are essential.
This is the human-population analog of the bacterial cross-species conservation signal.

Data: gnomAD v2.1.1 constraint (~141,000 people, diverse ancestries), 19,658 genes.
Truth: Hart CEGv2 / NEGv1. Code: `colab/human_constraint_benchmark.py`.

## Result — it works, and it works on the part FBA can't see

| gene set | LOEUF AUC | missense-z AUC | pLI AUC | constraint coverage |
|---|---|---|---|---|
| **genome-wide** (all essential vs non) | **0.864** | 0.849 | 0.708 | 96% |
| metabolic essentials (the 17% FBA sees) | 0.849 | 0.841 | 0.704 | 95% |
| **non-metabolic essentials (the 83% FBA CANNOT see)** | **0.866** | 0.850 | 0.709 | **97%** |

Constraint predicts the non-metabolic 83% (ribosome, spliceosome, cell cycle, proteasome)
just as well as the metabolic 17% — because purifying selection doesn't care what pathway
a gene is in, only whether losing it costs fitness.

## FBA vs constraint on human — the contrast

| | FBA metabolic wheel | gnomAD constraint |
|---|---|---|
| coverage of essential genome | 17% | **96–97%** |
| AUC on essentials it covers | precise but recall 0.25 | **LOEUF 0.86** |
| recovers non-metabolic essentials | no (structurally blind) | **yes (0.87)** |
| known drug targets flagged | 0/26 | (constraint ranks, not KO-tests) |

## Why this is the same lesson as the whole project

Bacteria: **conservation (cross-species) + measured fitness** → essentiality (~0.84).
Human: **constraint (cross-population) + [DepMap fitness]** → essentiality (LOEUF 0.86).
Same recipe — an *observed selection/fitness signal*, not a mechanistic model. The
mechanistic FBA model is the wrong tool for a single, richly-sequenced species; the
data-driven constraint signal is the right one. This is the identity-vs-quantity
principle again: essentiality is read most reliably off *observed evolutionary
constraint*, whichever timescale (species or population) the data gives you.

## The sub-gene and co-mutation extensions (the user's deeper points)

- **"Important regions that can't take mutations"** = **regional missense constraint** —
  sub-gene windows depleted of variation map the essential domains (active sites,
  interfaces) *within* a gene. gnomAD/Chen 2022 provide this; it's the natural next layer.
- **"Co-mutation"** = **epistasis / compensatory variation** — positions or genes whose
  variants co-occur (one tolerated only with another). The population-genetics analog of
  the coevolution signal used for protein contacts and bacterial operator coevolution.

## The human stack that would actually work (from this result)

1. **population constraint (gnomAD LOEUF/missense-z)** — genome-wide, 0.86 AUC — the base.
2. **W1 protein essentiality (ESM + cross-species conservation)** — orthogonal sequence
   signal, covers the non-metabolic machinery by fold/family.
3. **DepMap CRISPR dependencies** — the measured-fitness layer (lineage-selective targets,
   tumor-vs-normal selectivity) — the human analog of the bacterial fitness screens.
4. FBA only as a precise-but-narrow metabolic sub-module, not the backbone.

Constraint + W1 + DepMap is the real human essentiality/target engine. The bacterial FBA
backbone does not port; the *data-driven selection signals* do.
