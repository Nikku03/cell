# Human essentiality FUSION — constraint + conservation + sequence

The human analog of the bacterial essentiality stack: fuse **population constraint**
(gnomAD, 6 metrics) + **W1** (cross-species conservation via BLAST to yeast + E. coli,
and ESM-2 sequence embedding) into one score. 5-fold cross-validated (OOF), with
ablations. Truth: Hart CEGv2 / NEGv1 (1,479 genes: 636 essential, 843 non). Code:
`colab/human_fusion.py`.

## Result

| model | 5-fold OOF AUC |
|---|---|
| LOEUF alone (reference) | 0.823 |
| constraint (6 gnomAD metrics) | 0.886 |
| conservation (yeast + E. coli BLAST) | 0.860 |
| ESM-2 8M alone | 0.953 |
| constraint + conservation | **0.932** |
| **constraint + conservation + ESM (FUSION)** | **0.982** |

The fusion clears the 0.90 target (0.982). Constraint and conservation — the two
*evolutionary* signals — combine to 0.932 on their own; ESM lifts it to 0.982.

## Honest caveat: the benchmark is favorable, and ESM's solo 0.953 needs context

This contradicts our **bacterial** finding, where ESM-8M *hurt* within-organism
essentiality (overfit). The difference is the benchmark, not a reversal of physics:

- **CEG/NEG are curated *extreme* sets** — clear housekeeping essentials (ribosome,
  spliceosome, proteasome) vs clear dispensables (olfactory receptors, tissue-specific,
  late-onset). These differ by **protein family / type**, which ESM separates trivially
  from sequence (composition, membrane vs soluble, family). So ESM's 0.953 largely
  reflects *functional-family separability of the curated sets*, not hard,
  genome-wide, continuous essentiality prediction.
- The bacterial test scored essential-vs-non among **all** genes of one organism (a
  harder, less family-separable problem), where ESM overfit.

So the robust, transferable claim is the **evolutionary fusion: constraint +
conservation = 0.932** — two independent "can't take mutations" signals (population +
deep phylogeny) that don't depend on benchmark easiness. ESM adds real family-level
signal on top, but the honest acid test is a harder target.

## The honest next validation

Benchmark on **DepMap CRISPR dependency** (1000+ cell lines, continuous gene-effect
across the *whole* genome including ambiguous genes) rather than curated CEG/NEG. That
removes the family-separability shortcut and tests genome-wide, and would also give the
oncology-relevant axis (lineage-selective / tumor-vs-normal). Expectation: constraint +
conservation hold up; ESM's margin shrinks toward its bacterial behavior.

## Where this leaves the human engine

- **Constraint + conservation (0.93)** is the robust, data-driven backbone — the human
  analog of bacterial conservation + fitness, and it works genome-wide (unlike FBA's 17%).
- **ESM/W1** adds sequence-family signal (strong on curated sets; validate on DepMap).
- **FBA** is a narrow, precise metabolic sub-module, not the backbone.
- Next layers (per the plan): **DepMap** measured fitness, **regional missense
  constraint** (sub-gene essential domains), **co-mutation/epistasis**.

Bottom line: a cross-validated human essentiality predictor at 0.93 (evolutionary,
robust) to 0.98 (with sequence, on curated sets), built from public data — where the
mechanistic FBA wheel gave 17% coverage. The data-driven selection signals are the
human engine; validate the ESM margin on DepMap before trusting it genome-wide.
