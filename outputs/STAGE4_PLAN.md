# Stage 4 — selective prediction / abstention (the product decision)

**Status (built, validated, ready):** `scripts/stage4_abstention.py` +
`--dump_oof`/`--models` added to `scripts/step2_train.py`. Smoke + integration
tests pass. Stage 4 analysis is CPU-only and runs in seconds; only the OOF
dump needs the (GPU) LOCO pass.

## Why Stage 4 is the real test
Feature engineering plateaued. Full-population AUPRC-essential by stage:

| | AUPRC | lift |
|---|---|---|
| conservation | 0.041 | — |
| + gene | 0.157 | +0.115 |
| + condition stats | 0.193 | +0.036 |
| + ESM | 0.220 | +0.026 |
| + condition-content (cond3) | **0.234** | +0.014 |

Lifts halve each stage → another feature ≈ +0.007. **More features won't break
through.** But a product never runs at full population: it reports only its
confident calls and ABSTAINS on the rest. The question that decides "better
than Tn-seq" is **precision at low coverage**, which no full-population AUPRC
number tells you. That is what Stage 4 measures.

## What it computes
From out-of-fold (leave-one-clade-out) per-cell predictions, the
**risk-coverage curve**:
- rank held-out cells by predicted essentiality (= -predicted fitness)
- at coverage c (report top c% most confident): PRECISION + RECALL
- full model vs conservation baseline (the only other zero-wet-lab option)
- on ALL labels and on RELIABLE labels (high `total_weight` — trustworthy
  ground truth; the honest precision estimate, cf. Paper 1's matched-quality
  ceiling)
- the coverage at which precision crosses 0.5 / 0.7 / 0.9 (+ how many genes
  that is, and the recall there)

## The bar
Tn-seq needs a transposon library + sequencing per organism×condition, and its
own replicate calls agree only at binary kappa ~0.39 (Paper 1). Our predictor
needs zero wet lab. **If the confident subset reaches high precision at usable
coverage, it's a deliverable shortlist for free** — which conservation alone
(flat near base rate in the synthetic control) does not provide.

## Run (Colab; reuses cached frame/ESM/cond3 — no rebuild)
```bash
cd /content/cell && git pull origin claude/vectorize-gex-propensity-NRqBW

# 1) dump out-of-fold predictions for baseline + best model (GPU runtime).
#    Only 2 models (1 heavy) -> ~30 min, won't OOM (test_cap on).
python scripts/step2_train.py --real \
    --frame outputs/step2_training_frame.parquet \
    --esm   outputs/esm_embeddings.parquet \
    --cond3 outputs/condition_features.parquet \
    --models M_cons,M_full_esm_cond3 \
    --dump_oof outputs/stage4_oof.parquet \
    --subsample 800000
cp outputs/stage4_oof.parquet /content/drive/MyDrive/path_b_stage2/

# 2) analyze (CPU, seconds). Prints the precision@coverage table + crossings,
#    writes curve parquet, summary json, and stage4_abstention.png
python scripts/stage4_abstention.py --real --oof outputs/stage4_oof.parquet
cp outputs/stage4_abstention.* /content/drive/MyDrive/path_b_stage2/
```

## How to read the result
The decisive lines are the **COVERAGE AT WHICH PRECISION IS REACHED**:
- **precision >= 0.7 at a non-trivial coverage** (say top few %, covering
  thousands of genes) ⇒ usable product: a high-precision conditional-essential
  shortlist with zero wet lab. Proceed to Stage 5 (3-way honest eval +
  calibrated abstention head + write-up).
- **precision tops out near base rate even at tiny coverage** ⇒ the model can't
  separate a confident head; conditional vulnerability needs interventional
  data. Pivot to Stage 6 (bacterial DepMap design) and bank Paper 1 + the
  clean Path-B negative as the publishable result.
- The conservation curve quantifies the value-add: the gap between the model
  and conservation at each coverage is what ESM+cond3+CES bought us.

## No projection
Per standing commitment: I am not predicting the precision@coverage numbers.
The synthetic smoke shows the machinery (0.9 prec @ top 0.8%, 0.7 @ 2.2%, 0.5 @
5.2% for a *decent* ranker) but real numbers come only from the run.
