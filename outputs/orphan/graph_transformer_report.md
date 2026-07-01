# Transformer+GNN on the integrated cell — measured, honest

You asked whether a transformer combined with a GNN/LGNN helps. We built it and benchmarked
head-to-head on a task we can label (predict essentiality from the integrated cell, 5-fold
OOF, 30,093 genes, 291,034 regulatory+PPI edges, 1,610 labeled). Code:
`colab/graph_transformer.py`.

## Result — the graph and attention both HURT

| model | what it adds | OOF AUC |
|---|---|---|
| **MLP** | integrated features only (no graph) | **0.973** |
| GCN | + graph message-passing | 0.947 |
| GAT | + attention on the graph (= transformer+GNN) | 0.943 |

**Features-only MLP wins.** Adding graph message-passing (GCN) *and* graph attention (GAT)
both **lowered** accuracy. The neighbours-mixing smooths in non-essential genes and dilutes
the already-strong per-gene signal.

## Why this is the honest, expected answer

This is the **third** time in this project the same lesson has appeared:
1. bacterial conditional gate: **MLP beat cross-attention**;
2. LGNN was a better dynamics surrogate but a **worse essentiality predictor**;
3. now: **MLP beats GCN/GAT** on human essentiality.

And it matches the single-cell foundation-model literature (Kedzierska 2023): fancy
architectures **often don't beat simple baselines** on well-featured tasks. When the features
already encode the answer, structure/attention add noise, not signal.

## When transformer+GNN *would* help (and where it wouldn't here)

Graph/attention pays off only when **structure carries information the features don't** — which
is *not* this task (our features already include PPI degree, regulon size, constraint…). It
would help for:
- **the dark genome** — genes with *no* features, inferred purely from network position;
- **dynamics/trajectories** — where the **LGNN/liquid** layer models state over time (needs
  single-cell trajectory data we don't have);
- **cell-type-specific state** — better handled by a single-cell foundation model (Geneformer/
  scGPT) than a GNN on an aggregate network.

## Still useful: dark-genome scoring

The winning MLP, applied to uncharacterized genes (no disease, ≤1 pathway), flags plausible
hidden-essential genes: **PELO** (ribosome rescue), **EIF1**/**NACA** (translation), **MFAP1**/
**SERBP1** (splicing/RNA), **PA2G4**/**ETS2**/**MEIS1** — mostly translation/splicing-adjacent,
consistent with the essential core.

## Bottom line for the ML question
- **Transformer+GNN does not beat a simple model here** — we measured it, we didn't assume it.
- The architecture is the *easy* part; the missing ingredient is **data** (single-cell/
  trajectory), not a fancier network.
- Keep the interpretable map + simple models for static prediction; reserve
  transformer/GNN/LGNN for **dynamics and cell-type state**, and only after benchmarking vs a
  baseline — because it may tie, and a black box that ties isn't worth losing interpretability.
