# Training data plan (fine-tune / ensemble)

## The three data pieces
| piece | dataset | access | role |
|---|---|---|---|
| **priors (have)** | integrated_cell_human.csv (30k genes x 9 layers) | in repo | our features + edge |
| **essentiality labels** | Hart CEG/NEG (have) -> upgrade to **DepMap CRISPR** (measured, ~1100 cell lines) | figshare | supervised target |
| **substrate/cells (keystone)** | **Tabula Sapiens** (~500k human cells, 24 tissues) | cellxgene-census | cell-type state, phase 2 |
| **perturbation (phase 3)** | Replogle 2022 genome-wide Perturb-seq | GEO | KO->effect head |
| **foundation embeddings** | Geneformer [+scGPT] gene/cell embeddings (frozen) | HuggingFace | ensemble features |

## Phase 1 (runnable now on Colab GPU): gene-level essentiality ensemble
Test the core question cheaply: do frozen foundation-model **gene embeddings** ADD over our
integrated features for essentiality? Baseline to beat = our features-only MLP **AUC 0.973**
(and we already showed graph GNN/GAT *hurt*: 0.947/0.943 — so combiner = MLP first).
Notebook: `colab/finetune_ensemble_colab.py`.

## Phase 2: cell-type state (the keystone) — Tabula Sapiens + Geneformer cell embeddings.
## Phase 3: perturbation head — Perturb-seq.

## Discipline (non-negotiable)
Adopt the foundation-model ensemble ONLY where it beats the simple baseline, validated
leave-one-tissue-out. Our own results + the field's benchmarks warn it may only tie.
