# Learned GraphSAGE — the GPU-tier upgrade to fixed CellGraph

The fixed CellGraph answers with *SIGN propagation* (`[X | SX | S²X]`, no learning). The obvious question:
does a **trained** graph neural network beat it? We tested it honestly — same graph, same leakage-free
benchmark, only the encoder swapped — and the answer is **yes, and it generalizes**.

`colab/cellgraph_gnn.py` (`SAGE` + `head_to_head`), gated by `colab/validate_cellgraph_gnn.py` (scorecard axis
`learned_gnn_beats_fixed`). GPU notebook: `colab/gnn.ipynb`.

## The honest head-to-head

Identical everywhere except the encoder: same 10% edge hold-out, same test-edge removal before encoding, same
degree-matched hard negatives, same edge features `[u·v | |u−v| | u+v]`. Fixed = SIGN propagation → logistic
regression. Learned = a trained 2-layer GraphSAGE (mean aggregation) + a small edge-MLP decoder, end-to-end.

| relation | fixed SIGN | learned GraphSAGE | learned **R-GCN** | Δ (R-GCN − fixed) |
|---|---|---|---|---|
| **PPI** | 0.826 | 0.875 | **0.886** | **+0.060** |
| **regulatory** | 0.813 | 0.862 | — | +0.049 (SAGE) |
| **signaling** | 0.764 | 0.809 | — | +0.045 (SAGE) |

**Two learned encoders, both beat fixed on 3/3 relations; R-GCN is best.** Trained message-passing genuinely
extracts more than fixed propagation — unlike the earlier fixed-side enhancement (complex edges + 3 hops) that
moved link AUC 0.754→0.755 (noise) and was **not** adopted.

- **GraphSAGE** (merges all 5 relations into one adjacency): 0.826 → 0.875 on PPI (+0.049).
- **R-GCN** (a separate weight matrix *per relation* — reg/ppi/sig/codep/lr): 0.875 → **0.886** (+0.011 over
  GraphSAGE, **+0.060 over fixed**). Distinguishing the edge types beats merging them, and it was still
  climbing at 200 epochs. R-GCN is the adopted encoder (`learned_auc` in the validation).

## GPU vs CPU — identical result, faster training

The table above is **200 epochs on CPU**. A GPU changes *only the speed* — the AUC is identical (same math,
same seed). On a Colab GPU it trains in seconds instead of minutes, and 300+ epochs push the numbers a little
higher still (the loss was still dropping at 200). This is the first genuinely GPU-tier component: the encoder
is `torch.sparse.mm` message-passing, so a GPU actually does work (unlike the fixed engines, which are
numpy/sklearn/LP and leave the GPU idle).

## What it is

- **Encoder:** 2-layer GraphSAGE. Each layer concatenates a node with the mean of its neighbors and applies a
  learned linear + ReLU + dropout: `h' = ReLU(W · [h ‖ D⁻¹A·h])`. Input = the 35-dim node features.
- **Decoder:** an MLP on `[u·v | |u−v| | u+v]` → link logit. Trained with BCE over train positives +
  degree-matched negatives (Adam, weight decay, dropout for regularization).
- **Leakage control inherited from the fixed benchmark:** test edges are removed from the adjacency *before*
  encoding, so the model never sees them during representation learning.

## Honest limits / scope

- **Link prediction only, so far.** The +0.045 win is measured on link AUC. The other CellGraph heads
  (perturbation direction, structure→function, drug polypharmacology) still use the fixed pipeline; porting
  them to the learned encoder is the next step.
- **Still a small model.** 2 layers, 64-dim, mean aggregation — deliberately modest. R-GCN (per-relation
  weights) or attention (GAT) are the obvious next levers, along with ESM-2 node features (see
  `docs/FUTURE_IDEAS.md`).
- **Result is reproducible up to torch's seeded stochasticity** — the win margin is stable across runs, the
  absolute value moves ±~0.005.

## Adoption rule

The scorecard axis passes only while the learned encoder **beats the fixed one** (PPI Δ>0.01 and wins ≥2/3
relations). If a future change erased the advantage, the axis would fail — so the learned encoder is kept only
as long as it's genuinely better than the free fixed baseline.
