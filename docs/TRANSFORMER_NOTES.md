# Transformer notes — running log (how each phase feeds a cell transformer)

Goal: after the phases, assemble a transformer that ingests single-cell expression **and** the
knowledge graph to predict perturbation responses and generalize to unseen genes — moving past
classic statistics toward a model that generalizes. One entry per cleared phase; at the end these
notes become the data spec + architecture.

## Design thesis (updated as phases land)
A hybrid **graph-aware expression transformer**:
- **Tokens** = genes; per-cell input = genes ranked/binned by expression (Geneformer/scGPT style).
- **Gene identity embedding** = a learned vector per gene *initialized/augmented* with our static
  multi-omic features (compartment, process, network degrees, essentiality, pathway one-hots) — this
  is how the knowledge graph enters, and it's what lets the model generalize to genes with few data.
- **Attention bias / graph** = our convergence edges (P2) and co-expression edges (P1) as a
  structural prior (relative-position / graph-attention bias), so attention respects known relationships.
- **Objectives** = masked-gene expression prediction (self-supervised, pretrained on ARCHS4) + a
  supervised perturbation head (predict the Perturb-seq KO signature) fine-tuned on measured perturbations.
- **Why this beats Model 4:** Model 4 mapped *static* features → signature and failed (neg R²). A
  transformer conditions on the *actual expression state of the cell* + graph structure, which is the
  information static features lack.

## Per-phase contributions

### P1 — ARCHS4 co-expression → PRETRAINING CORPUS + graph prior
- ARCHS4 itself (~1M samples) is the **self-supervised pretraining corpus** (masked-gene-expression).
- The co-expression network (`coexpr_neighbors.json`) → **graph-attention edges** / a gene-gene
  affinity prior. Also each gene's co-expression profile is a ready **gene embedding init**.
- Data spec: expression matrix (genes×samples), rank or bin per sample; store gene-symbol vocab.

### P2 — Convergence → EDGE STRUCTURE + supervised edge labels
- Convergence edges (multi-lens agreement) → the **adjacency/attention-bias graph** the transformer
  attends over; convergence score = edge weight.
- Known-complex control pairs → **positive edge labels**; novel high-score pairs → candidate edges to
  test the model's link-prediction generalization.
- Feeds a **link-prediction auxiliary head** (does the model rediscover convergent links from expression?).

_(P3+ entries appended as phases clear.)_
