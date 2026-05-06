# Sparse-LNN Cascade — gene essentiality predictor across bacteria

A learned predictor that takes a bacterial gene and outputs the probability it is essential. Trained on 8 organisms with leave-one-out cross-validation; deployed as a stacker over multiple feature modalities.

This document covers what the system is, what it does, how it scores, and how to reproduce it.

---

## 1. The architecture in one diagram

```
                  ┌──────────────────────────────────────────────────────┐
                  │  Stage 2: MultimodalSparseConceptLNN                 │
                  │                                                      │
       gene  ──►  │   ┌────────┐   ┌────────┐   ┌────────┐               │
                  │   │ static │   │ traj   │   │ know   │               │
                  │   │ branch │   │ branch │   │ branch │               │
                  │   └───┬────┘   └───┬────┘   └───┬────┘               │
                  │       │            │            │                    │
                  │       ▼            ▼            ▼                    │
                  │   ────────  K=128 sparse-pattern bottleneck ────     │
                  │       │            │            │                    │
                  │       ▼            ▼            ▼                    │
                  │     128            128          128                  │
                  │   activations    activations  activations            │
                  └───────┴────────────┴────────────┴──────┬─────────────┘
                                                            │
       scalar gene features                                 │
       (length, GC, kw_*, ortholog priors, v15)  ──────►    │
                                                            ▼
                                            ┌────────────────────────────┐
                                            │  Cascade stacker           │
                                            │  (logistic regression LOO) │
                                            └────────────┬───────────────┘
                                                         ▼
                                                  P(essential)
```

Two stages. Stage 2 produces sparse pattern activations from three modalities. The cascade stacker takes those plus scalar features and outputs the final probability.

---

## 2. Stage 1: SparseConceptLNN (single-modality)

`cell_sim/layer_ml/sparse_concept_lnn.py`

A liquid neural network with a **K = 128 sparse-pattern bottleneck**. For each gene the model:
1. Encodes scalar+keyword features into a hidden state via an LNN block.
2. Projects to 128 pattern logits.
3. Selects the **top-5** patterns (sparse selection).
4. Decodes essentiality probability from those 5 activations.

Diversity loss prevents all genes from collapsing onto the same pattern. Reconstruction loss makes the bottleneck preserve enough information to recover the input.

**Honest result**: Stage 1 alone gets cross-org mean MCC of **0.150** — barely above chance. The static features are too thin to carry the prediction by themselves. Stage 2 fixes that by adding two more modalities.

---

## 3. Stage 2: MultimodalSparseConceptLNN

`cell_sim/layer_ml/multimodal_sparse_concept_lnn.py`

Three branches that all share the same K = 128 patterns:

| Branch | Input | Encoder |
|---|---|---|
| **Static** | scalar + keyword features per gene | LNN block (Stage 1) |
| **Trajectory** | per-gene v15 simulator output (T=5 timepoints × 262 features) | LSTM |
| **Knowledge** | ortholog prior, n_orthologs, PPI degree, hub flag, conservation, ... (6 dims) | small MLP |

All three branches project into the same 128-pattern space. Top-5 selected per gene per modality. The shared bottleneck means **the same pattern that lights up in the static branch should also light up in the trajectory branch when the gene is functioning the same way** — that's enforced via a cosine-alignment loss.

**Loss components** (compute_loss, line 286):

```python
w_main_bce          = 1.0    # binary essentiality cross-entropy
w_per_step_bce      = 0.3    # auxiliary per-timepoint loss on traj branch
w_align_static_traj = 0.5    # cosine alignment of static & traj activations
w_align_static_know = 0.3    # static & knowledge alignment
w_diversity         = 0.1    # prevent pattern collapse
w_reconstruction    = 0.1    # bottleneck preserves inputs
```

**Honest result**: Stage 2 gets cross-org mean MCC **0.334** — that's +0.184 over Stage 1, real lift, all 5 LOO folds improved.

| held-out org | Stage 1 MCC | Stage 2 MCC | Δ |
|---|---|---|---|
| styphimurium | 0.112 | 0.386 | +0.274 |
| ccrescentus | 0.252 | 0.401 | +0.149 |
| saureus | 0.063 | 0.152 | +0.089 |
| mtuberculosis | 0.220 | 0.379 | +0.159 |
| abaylyi | 0.103 | 0.352 | +0.249 |
| **mean** | **0.150** | **0.334** | **+0.184** |

`outputs/multimodal_sparse_concept_results.json` has the per-org details.

---

## 4. Cascade stacker — the deployed predictor

`scripts/sparse_lnn_cascade_stacker.py` produces the final, headline prediction.

A logistic regression over a 676-dimensional feature vector per gene:

| Block | Dim | Source |
|---|---|---|
| **Scalar** | 30 | log_length_bp, gc, upstream_gc, upstream_at_skew, position_norm, operon_run_length, 11 keyword flags (kw_replication, kw_transcription, kw_translation, kw_atp, kw_membrane, kw_synthase, kw_kinase, kw_dehydrogenase, kw_protease, kw_rrna, kw_chaperone), 3 hypothetical/uncharacterised/putative flags, n_orthologs, ortholog_prior, has_lnn, has_v15, v15_call, v15_conf, lnn_prob, has_pattern, has_traj_signal, has_traj_pool |
| **Static patterns** | 128 | activations from the static branch of Stage 2 |
| **Trajectory patterns** | 128 | activations from the trajectory branch |
| **Knowledge patterns** | 128 | activations from the knowledge branch |
| **Trajectory mean-pool** | 262 | mean over 5 timepoints of v15 raw species/event counts (Syn3A only; padded zero elsewhere) |

**Important: PPI features are deliberately excluded** in the Path-B variant. This is to test whether the ortholog + simulator + sparse-LNN cascade alone can carry the cross-org signal without relying on protein-protein interaction data from STRING. The earlier `meta_lnn_stacker.py` *does* include PPI for comparison.

Trained leave-one-out across **6 organisms** (5 cross-org + Syn3A in-domain).

---

## 5. Headline numbers

`outputs/sparse_lnn_cascade_stacker_results.json`

| Metric | Value |
|---|---|
| **Pooled MCC across 6 LOO orgs** | **+0.5510** |
| **Mean per-org MCC (5 cross-org)** | **+0.4442** |
| Mean per-org MCC (all 6, includes syn3a) | +0.4206 |
| Syn3A in-domain MCC | +0.3027 |

Per-organism breakdown:

| held-out | n | MCC | precision | recall |
|---|---|---|---|---|
| styphimurium | 3,291 | +0.529 | 0.438 | 0.713 |
| ccrescentus | 3,614 | +0.540 | 0.701 | 0.498 |
| saureus | 402 | +0.126 | 0.912 | 0.561 |
| mtuberculosis | 3,497 | +0.503 | 0.730 | 0.419 |
| abaylyi | 2,887 | +0.523 | 0.620 | 0.519 |
| syn3a | 455 | +0.303 | 0.955 | 0.554 |

**Comparisons:**

| Method | Mean cross-org MCC |
|---|---|
| meta_lnn_stacker (with PPI) | +0.460 |
| **sparse_lnn_cascade_stacker (this, no PPI)** | **+0.444** |
| rule-based router (mean) | +0.443 |
| ortholog prior alone (mean) | +0.420 |
| cross-org LNN on M. tb alone | +0.260 |

**Cost of removing PPI:** ‒0.016 MCC. Substantially smaller than the gain from including the sparse-LNN pattern features. The pattern-cascade head replaces most of what PPI contributed.

S. aureus is the weak fold (87% of its 402 genes are essential — class imbalance). Same structural issue affects Syn3A in-domain at 84% essential. Both are heavily-essential minimal genomes; the model is calibrated for typical bacterial genomes (~12% essential) and undercalls in those regimes.

---

## 6. Pattern atlas — interpretability

After Stage 2 trains, `cell_sim/layer_ml/multimodal_atlas.py` post-processes the 128 patterns into:

* **`outputs/multimodal_sparse_concept_pattern_atlas.csv`** — one row per pattern. For each, the top-N most-activating genes per modality, the essential-fraction within those genes, and three cross-modal correlation scores (corr_static_traj, corr_static_know, corr_traj_know). Sorted by mean alignment.
* **`outputs/multimodal_sparse_concept_pattern_connections.csv`** — K×K co-activation graph. Long-format: which patterns light up together in the same gene, per modality, with a normalized weight.
* **`outputs/multimodal_sparse_concept_per_gene_attributions.csv`** — one row per gene. Top-3 patterns per modality, normalized weights, plus the model's predicted probability.

**Headline patterns** (from atlas, sorted by alignment):

| pattern | static top genes | label | essential_frac |
|---|---|---|---|
| **96** | trmK, yqfN, rpsS, rplF, rpmC, ribF, rplR, dnaA | **essential housekeeping core** (ribosomal proteins + replication) | 1.00 static, 0.80 traj, 1.00 know |
| 89 | rplO, rplR, rpsE, rplP, rplQ, rplF, rplP, secY | **ribosome / membrane core** | 1.00 / 0.80 / 0.87 |
| ... | ... | ... | ... |
| 119 | (hypothetical / unlabeled gene cluster) | **non-essential / hypothetical** | low |
| 28 | similar | **non-essential** | low |

The cascade stacker's strongest **negative** coefficients are exactly patterns 119 and 28; the strongest **positive** is pattern 96. The model learned a meaningful internal language: *"if the gene activates the housekeeping pattern, predict essential. If it activates the hypothetical pattern, predict non-essential."*

---

## 7. How to reproduce

```bash
# 1. Train Stage 1 (static-only LNN, 5-fold LOO)
#    Inside notebooks/multimodal_sparse_concept_lnn.ipynb cell 2

# 2. Generate v15 trajectories for all genes (this feeds Stage 2's traj branch)
#    Inside the same notebook cell 3
#    Or pull from outputs/syn3a_trajectories_t2.0s.npz

# 3. Train Stage 2 (multimodal, shares K=128 patterns)
#    Inside the notebook cell 4

# 4. Build the pattern atlas + per-gene attributions
#    Inside the notebook cell 5  →  writes the 3 CSVs to outputs/

# 5. Run the cascade stacker
python scripts/sparse_lnn_cascade_stacker.py
#    →  outputs/sparse_lnn_cascade_stacker_results.json
#    →  outputs/sparse_lnn_cascade_stacker_per_gene.csv
```

The notebook is `notebooks/multimodal_sparse_concept_lnn.ipynb`. Trained checkpoints live at `cell_sim/layer_ml/checkpoints/sparse_concept_lnn_static_loo.pt` and `sparse_concept_lnn_multimodal.pt`.

---

## 8. What this is good for / not good for

**Good for**:
- Cross-organism essentiality prediction across 5+ bacteria with one model
- Interpretable predictions — every gene comes with its top-3 patterns per modality
- Working without PPI data (which not all organisms have)
- A feature block downstream models can plug into

**Not good for**:
- Outperforming the rule-based router or the meta-LNN stacker by a wide margin — we're around the same headline MCC. The contribution is interpretability and the freedom from PPI, not raw accuracy.
- In-domain Syn3A (MCC 0.303) — v15 alone hits ~0.537 in-domain; for syn3A specifically, use v15.
- Heavily essential-biased small genomes (S. aureus, syn3a). Class-imbalance calibration is off.

---

## 9. Files at a glance

| File | What it holds |
|---|---|
| `cell_sim/layer_ml/sparse_concept_lnn.py` | Stage 1 model |
| `cell_sim/layer_ml/multimodal_sparse_concept_lnn.py` | Stage 2 model |
| `cell_sim/layer_ml/multimodal_atlas.py` | atlas + connections + attributions |
| `scripts/sparse_lnn_cascade_stacker.py` | the deployed stacker |
| `notebooks/multimodal_sparse_concept_lnn.ipynb` | end-to-end training pipeline |
| `cell_sim/layer_ml/checkpoints/sparse_concept_lnn_*.pt` | trained weights |
| `outputs/multimodal_sparse_concept_*.csv` | atlas, connections, attributions |
| `outputs/multimodal_sparse_concept_results.json` | Stage 1 + Stage 2 cross-org table |
| `outputs/sparse_lnn_cascade_stacker_results.json` | final headline numbers |
| `outputs/sparse_lnn_cascade_stacker_per_gene.csv` | per-gene predictions |
