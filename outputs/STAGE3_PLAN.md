# Stage 3 — rich condition-content features (run plan)

**Status (built tonight, validated, ready):** `scripts/stage3_condition_features.py`
+ `--cond3` wired into `scripts/step2_train.py`. All smoke/positive-control
tests pass. Nothing here needs a GPU to *start* — the feature build reads the
feba.db Experiment table on CPU.

## Where Stage 2 left us
LOCO across 29 clades, AUPRC-essential (base rate ~0.015):

| model | AUPRC | gain |
|---|---|---|
| conservation | 0.041 | — |
| + gene features | 0.157 | +0.115 |
| + condition stats | 0.193 | +0.036 |
| **+ ESM-2 (640d)** | **0.220** | **+0.026 ← Stage 2 PASS** |

## The Stage 3 bet
The condition axis is impoverished: each of 1,296 `condition_cluster`s is
encoded as 4 scalars (aerobic, size, target-encoded mean_fit, ess_rate). That
tells the model *how essential genes are on average here*, not *what the
perturbation IS chemically*. Stage 3 gives the condition a real content vector
(pH, temp, log-concentration, expGroup one-hots, and a 32-d hashed
bag-of-words over the free-text description) so the GBM can learn
**gene-function × perturbation-content** interactions (efflux pump × drug;
siderophore × metal limitation).

Source: feba `Experiment` table (already downloaded). Pure metadata, nothing
fitness-derived → **leak-free by construction** (a property of the condition,
identical in train and held-out clades). Key = `media|aerobic|condition_1`,
reconstructed exactly as `ces_consensus.py` builds it → 1:1 join to the frame.

## THE CAVEAT (learned tonight from positive controls)
`cond_mean_fit` (the leak-free target encoding already in `M_full`) is itself a
**near-optimal 1-D learned condition embedding**. Any per-cluster main effect,
and any single-scalar gene×condition interaction, is reconstructable from it.
**cond3 only beats it if condition content is genuinely multi-dimensional** —
different chemistry hitting different genes in different directions — *and* the
GBM can learn that interaction. This is a steeper hill than Stage 2 (which added
a brand-new axis). Do not expect a free +0.02. Treat the kill-gate as a real
test of whether condition *content* > condition *summary*.

## Kill-gate
`M_full_esm_cond3` must add **≥ +0.02 AUPRC-essential over `M_full_esm`**
(0.220). The trainer prints this line automatically. Also watch
`M_full_cond3` vs `M_full` to isolate cond3's contribution without ESM.

## Run order (Colab, tomorrow)
```bash
cd /content/cell && git pull origin claude/vectorize-gex-propensity-NRqBW

# 1. build condition-content features from feba.db (CPU; downloads feba.db if
#    not cached — ~2.3 GB, same source as the consensus step)
python scripts/stage3_condition_features.py --real
cp outputs/condition_features.parquet /content/drive/MyDrive/path_b_stage2/

# 2. train with ESM + condition-content (GPU runtime for xgboost device=cuda)
python scripts/step2_train.py --real \
    --frame /content/drive/MyDrive/step2_training_frame.parquet \
    --esm   /content/drive/MyDrive/path_b_stage2/esm_embeddings.parquet \
    --cond3 outputs/condition_features.parquet \
    --subsample 800000
```
Read the **STAGE-3 KILL-GATE** line.

## Decision tree
- **PASS (≥ +0.02):** condition content is a real lever. Proceed to Stage 4
  (calibrated abstention curve — finally apply abstention to AUPRC and measure
  precision on the retained high-`total_weight` subset; this is the actual
  "better than Tn-seq" test). Cumulative AUPRC would be ~0.24+.
- **FAIL/marginal:** the target encoding already saturates the condition axis.
  That's a clean, publishable negative ("learned condition summary ≥ explicit
  metadata content"). Skip straight to Stage 4 — the unmeasured abstention
  curve is then the main remaining lever before the Stage 6 longer shot
  (interventional bacterial DepMap).

## If feba metadata is thin
Inspect `outputs/condition_features_summary.json` and the printed column list.
If pH/temp/concentration coverage is low and expGroup is sparse, the hashed
free-text vector carries most of the signal — that's expected and fine. If even
the text is degenerate, the honest read is "feba condition annotation is too
coarse for content features" → the lever becomes external condition embeddings
(iModulon/Hawkins), a bigger build deferred to a later stage.
```
