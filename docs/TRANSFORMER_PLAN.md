# Transformer plan — a KG-conditioned perturbation transformer

Grounded in a literature review (Geneformer, scGPT, scFoundation, CellPLM, UCE, GEARS, STATE, CPA;
see `docs/HOMEWORK_BRIEF.md` for citations). This is the endpoint the phase work builds toward.

## The uncomfortable truth that shapes the design
Benchmarks (Ahlmann-Eltze et al., *Nature Methods* 2025, DOI 10.1038/s41592-025-02772-6; Csendes
"One PCA still rules them all", arXiv:2410.13956) show pure-expression foundation models **do not beat
mean/additive/PCA baselines** for unseen perturbations, because pretraining data is *observational, not
perturbational*, and FM attention captures **co-expression, not causal signal**. **This independently
explains our Model 4 result (9.2× recall but negative R²) — it's a known-hard problem, not just our bug.**

The lever that demonstrably helps (GEARS, *Nat Biotech* 2024): a **structured knowledge-graph prior over
genes** + **predicting the change (Δ), not the absolute state**. Our multi-omic KG is exactly the asset
the pure-expression models lack — so we build *toward the KG*, not toward a bigger expression model.

## Architecture — two towers + a transition operator

### Tower A — Gene tower (KG encoder; the source of unseen-gene generalization)
One embedding `g_i` per gene for all 16,492 proteins **and genes with no Perturb-seq data**.
- **Node init (no cold-start):** ESM2 protein-sequence embedding (UCE's trick) ⊕ our static features
  (compartment, process, essentiality, PTMs, pathway one-hots). A brand-new gene enters via its ESM2
  vector + KG edges — no learned ID needed.
- **Encoder:** heterogeneous/relational GNN (R-GCN / Heterogeneous Graph Transformer) over typed edges:
  TF→target (directed, causal backbone) · PPI / same-complex / shared-PTM (physical) · **co-essentiality
  + co-expression (functional modules)** · shared-pathway/reaction/compartment (context) · drug→target.
- **Pretrain (Stage 1):** self-supervised link prediction + node-feature reconstruction — bakes KG
  geometry into `g_i` before any expression is seen. (GEARS lesson, generalized to a full multi-omic graph.)

### Tower B — Cell/expression tower
- **Per-gene token = `g_i` (Tower A) ⊕ binned-expression ⊕ condition/compartment embedding** (scGPT-style,
  but reusing Tower A embeddings so unseen genes participate).
- **Pretrain (Stage 2):** masked-expression modeling on ARCHS4 + our atlas; keep scFoundation's
  read-depth-aware variant if depths vary.

### Tower C — Transition operator (the perturbation predictor)
- **Perturbation embedding `p` = MLP(pool of target-gene `g_i`, direction, dose).** Built from KG
  embeddings ⇒ an **unseen target still yields a meaningful `p`** from its neighborhood (GEARS extrapolation).
- **Set-level bidirectional transformer** over a *population* of control cells + `p` → predicts per-cell
  **Δ (post − pre)**, not the absolute profile (STATE + CPA lessons). Two deliberate choices from the
  failure analysis: predict the residual; model the distribution shift, not one cell.
- **Output head tied to Tower-A `g_i`** ⇒ any gene with an embedding (incl. unseen readouts) gets a head.

### Losses / evaluation (bake in honesty)
- Δ-reconstruction (NB/Poisson or MSE-on-log-Δ) **weighted to DE genes**; distributional MMD/Sinkhorn
  between predicted vs true perturbed populations; contrastive alignment of the two towers' geometries.
- **Always report vs mean-of-training / additive / PCA baselines**; split by unseen-single /
  unseen-combination / unseen-context; metrics = DE-direction accuracy, Spearman-of-Δ on top-DE genes,
  energy distance (Virtual Cell Challenge protocol). Hold out whole genes to force KG-based extrapolation.

## How each phase's data maps to the model
| our data | tower | role |
|---|---|---|
| ARCHS4 co-expression (P1) | A edges + B pretrain corpus | functional-module edges; masked-expression data |
| convergence graph (P2) | A edges | high-confidence typed edges + link-prediction labels |
| causal regulome (P3, TF→target signed) | A edges | the directed, causal backbone (highest-value edges) |
| DoRothEA/TRRUST/CollecTRI, SIGNOR | A edges | regulatory + signaling edges |
| PPI (STRING/BioPlex/OpenCell/HuRI), complexes | A edges | physical edges |
| co-essentiality, synthetic-lethal | A edges | genetic-dependency edges |
| Perturb-seq signatures (~9k KOs) | C supervision | Stage-3 fine-tune target (measured Δ) |
| static features (compartment/process/essentiality/PTM) | A node init | cold-start-free gene features |
| ESM2 of each protein (TODO: fetch) | A node init | unseen-gene / cross-species generalization |
| pathways/metabolism | A context edges | mechanistic context |

## Honest expectation
Wins are most attainable for **in-distribution genes, ranking/prioritization, and cell-context transfer**;
exact magnitudes for genuinely novel perturbations stay hard. The KG (co-essentiality + TF→target +
complex/pathway) is the primary lever and our best shot at beating the additive baseline on unseen
combinations. This is a **build-toward**, not a claim that it will "solve" perturbation prediction.

## Realistic staging (post phase-work)
1. Assemble the **typed KG edge list** from P1-P4 outputs (a script: all layers → one heterogeneous graph).
2. Fetch **ESM2 embeddings** per protein (one-time; the unseen-gene enabler).
3. Build + pretrain **Tower A** (GNN link-prediction) — testable in the sandbox on a subgraph.
4. Tower B + C need GPU + the expression corpora → Colab.
5. Evaluate against the trivial baselines *first* — if the KG-GNN perturbation embedding (Tower A + C,
   GEARS-style) doesn't beat additive, that's the honest answer before investing in B.
