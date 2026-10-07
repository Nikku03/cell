# Cell-type conditioning — the novel dimension

The stated goal asks what happens "in the cell" — but *the cell* is **200 different cell types**, and a gene
that is a hub in a hepatocyte can be silent in a T cell. This layer conditions the whole-cell model's queries
on a **cell type**, so the same question returns a **different, cell-type-appropriate answer**. An interaction
or regulatory edge is only *active* in cell type `t` if both partners are **expressed** there.

`colab/celltype_conditioning.py` (`CellTypeConditioner`), gated by `colab/validate_celltype_conditioning.py`
as a recovery-scorecard axis. Wired into **CellQA**: `what_binds(X, cell_type=…)` and `knockout(X, cell_type=…)`
gate their answers by expression.

## The data

The populated **`emask`** — a 200-cell-type expression **bitmask** over 7,496 genes (bit `t` set ⇔ gene
expressed in cell type `t`). It is the compact encoding of a healthy cross-tissue single-cell census.

## What conditioning does (live)

```
GATA1 binds — in erythroid?          -> HNRNPK, KLF1, LMO2, TAL1 (the erythroid GATA1/TAL1/LMO2 complex) …
GATA1 binds — in regulatory T cell?  -> ABSTAIN: GATA1 not expressed in regulatory T cell
```

Same protein, same question — the erythroid answer surfaces the real erythroid partners; the T-cell answer
correctly abstains because GATA1 isn't there. `knockout(X, cell_type=t)` likewise only propagates through
edges whose endpoints are expressed in `t`.

## Validation — the honest split

### ✓ What is robust (the scorecard axis): the gate is cell-type-specific

Using **external, textbook** lineage markers (chosen independently of the model), each marker set lands in its
own cell type far above background and near-zero elsewhere:

| lineage | native expr | control mean | × background | specificity |
|---|---|---|---|---|
| erythroid | 0.83 | 0.00 | 9.9× | +0.83 |
| monocyte | 0.83 | 0.03 | 9.8× | +0.80 |
| B cell | 0.80 | 0.00 | 11.3× | +0.80 |
| Treg | 1.00 | 0.00 | 11.0× | +1.00 |
| NK | 0.60 | 0.00 | 6.6× | +0.60 |
| hepatocyte | 1.00 | 0.03 | 7.7× | +0.97 |

**Min specificity +0.60, mean +0.83, ~9.4× over background across 6 diverse lineages.** The gate correctly
turns marker genes ON in their own lineage and OFF everywhere else — so cell-type-conditioned answers are
built on **correct** cell-type biology. Curated regulatory programs also show it: FOXP3 targets are **2.84×**
enriched in Treg, PAX5 **1.94×** in B cell, SPI1 **1.29×** in monocyte.

### ✗ What is NOT a predictive lift (documented, not gated)

Conditioning is a **correct gate**, not a way to beat the global predictors. Three honest negatives, all
recorded in the validation JSON:

- **Co-expression is a weak PPI predictor.** Shared cell types give PPI AUC ~0.71 with random negatives, but
  collapse to **~0.57** once negatives are **expression-degree-matched** — broadly-expressed proteins both
  interact more and share more cell types (a hub artifact), so the apparent signal is mostly confound.
- **Reachability specificity saturates.** On the dense directed reg/sig graph, a random TF reaches the same
  lineage markers within 3 hops as the true lineage master TF — the graph is too connected for cell-type
  reachability to be specific (the same density that limited the cause-finder).
- **Bulk-inferred regulatory targets aren't cell-type-resolved.** Only small **curated** programs (FOXP3-tier)
  show native-lineage enrichment; large ChIP/inferred target sets (GATA1 689, SPI1 2267, CEBPA 5013) sit at
  the background rate, and erythroid's narrow expression breadth even inverts the raw signal.

So the claim is deliberately narrow and true: **conditioning gives correct, cell-type-specific answers on a
validated gate — it does not claim to out-predict the global model.** This is the abstain-when-unsure
philosophy applied to a whole new axis: where the gate is trustworthy (marker/expression level) we condition;
where it would only add noise (reachability, bulk targets) we don't pretend it helps.

## Why it matters for the goal

"If a protein is removed, what is the downstream effect?" and "what does X bind?" are **cell-type-dependent**
questions. This layer makes the whole-cell model answer them per cell type instead of as one averaged blob —
the 200-way dimension the goal implicitly requires, added with an honest account of exactly how far the
current data lets it go.

## Next

- Fold cytokine→receptor **PPI edges** into the directed cascade so receptor bottlenecks become reachable
  (also fixes the SLE/IFNAR1 miss in the disease→target pipeline).
- A **cell-type-resolved** interaction/perturbation dataset (e.g. a lineage-specific Perturb-seq) would let
  conditioning be validated as a *predictive* lift, not only as a correct gate.
