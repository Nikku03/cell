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

### P3 — Causal regulome → the DIRECTED, causal backbone (Tower A's highest-value edges)
- `causal_reg.tsv` (signed TF→target from binding × response) → the **directed causal edges** Tower A
  needs; pure-expression FMs capture co-expression, not this. These are the edges most likely to make
  the perturbation embedding `p` extrapolate correctly.
- `perturbseq_targets.json` (per-KO responders) → direct **Stage-3 supervision** (measured Δ per gene).

### P4 — GTEx + DoRothEA/TRRUST union → more regulatory edges + a genetic-regulation edge type
- DoRothEA/TRRUST/CollecTRI union → denser TF→target edges (Tower A regulatory backbone).
- GTEx trans-eQTL edges → a distinct **genetic-regulation edge type** (orthogonal to ChIP/curated).

### P5 — ncRNA → a new node/edge type (miRNA & lncRNA regulators)
- `ncrna_targets.json` → ncRNA→gene edges. In the transformer, ncRNAs become **new node types** with
  their own (sequence-derived) embeddings, extending the graph beyond the 16,492 proteins.

## Tower A — BUILT + TESTED (sandbox prototype)
`build_kg_edges.py` assembles the heterogeneous typed edge list: **16,492 nodes, 533k typed edges**
(reg 278k, ppi 158k, codep 77k, complex 19k, convergence) + 35-dim node features.
`train_tower_a.py` (SVD graph-embedding + feature-fusion link-prediction, leakage-free hold-out) tests
the core claim on real data:
| model | held-out link AUC |
|---|---|
| KG embedding (structure) | 0.944 |
| KG embedding + node features | 0.946 |
| Adamic-Adar baseline (honest) | 0.888 |
| **HYBRID (embedding + features + structure)** | **0.954** |

**Verdict: the multi-omic KG is highly predictive; the learned model beats the AA heuristic; the hybrid
modification beats both.** This validates Tower A before spending GPU. Production version:
`train_tower_a_gnn.py` (torch GraphSAGE, Colab) → `tower_a_embeddings.npz` (the g_i that feed Tower C).
Reasoning-derived edges (5,422 confirmed transitive links) and convergence edges are available to add as
extra typed edges — expected to lift AUC further. **Next upgrade:** typed R-GCN (per-edge-type weights)
+ ESM2 node init (unseen-gene generalization).

## Consolidated: the transformer's typed edge set is now assembled across P1-P5
co-expression (P1) · convergence high-confidence (P2) · causal TF→target signed (P3) · curated
regulatory union + GTEx genetic (P4) · ncRNA→target (P5) · plus the pre-existing PPI / complex /
co-essentiality / pathway / metabolic edges. Next concrete step (post-phases): a `build_kg_edges.py`
that emits one heterogeneous typed edge list — the direct input to Tower A. See `TRANSFORMER_PLAN.md`.
