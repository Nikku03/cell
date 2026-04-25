# Figures

Plot-ready data + matplotlib scripts. **Run the scripts locally to produce PNG/PDF outputs**; nothing is pre-rendered in this repo.

## What's here

```
figures/
├── README.md                    (this file)
├── data/
│   ├── mcc_progression.csv      version-by-version MCC + panel size
│   ├── detector_contributions.csv   isolated MCC per detector + role
│   └── confusion_matrices.csv   tp/fp/tn/fn per full-panel version
└── scripts/
    ├── plot_mcc_progression.py
    ├── plot_detector_contributions.py
    └── plot_confusion_matrix_grid.py
```

## How to generate the figures

From the repo root, with `matplotlib`, `pandas`, and `numpy` installed:

```bash
python figures/scripts/plot_mcc_progression.py
python figures/scripts/plot_detector_contributions.py
python figures/scripts/plot_confusion_matrix_grid.py
```

Each script writes a `.png` (150 dpi) and a `.pdf` (vector) into `figures/output/`. The output directory is created on first run and is gitignored.

## What each plot shows

### `mcc_progression.png`

Shows how the Matthews Correlation Coefficient evolved across detector versions on the full 455-gene Breuer panel. Two lines:

- **Blue (n=455)** — full-panel measurements from v10 onwards. Trajectory: 0.342 → 0.537 over five iterations.
- **Grey dashed (n=40)** — small-panel calibration runs from v5/v6/v9. These plateau at MCC ~0.13, which is the ceiling for trajectory-only detectors.

A red dotted reference line marks Breuer 2019's own FBA benchmark (MCC 0.59) so the reader sees the gap.

**The story:** trajectory detection alone caps out around 0.13. Knowledge-based priors (complex membership, annotation keywords) take the result from 0.342 to 0.537 over five rounds of FN-pool mining.

### `detector_contributions.png`

Bar chart of isolated MCC per detector, color-coded by role:

- **Grey** — trajectory-only detectors (`PerRule`, `RedundancyAware`)
- **Blue** — knowledge-based detectors (`ComplexAssembly`, `AnnotationClass`); shown as part of v15 composed since they aren't measured in isolation against the full panel
- **Green** — `Composed_v15` (the union: ComplexAssembly ∪ AnnotationClass ∪ PerRule)
- **Red** — `Tier1_XGBoost` (supervised-ML stack, falsified at 0.145 features-only / 0.443 union)

Two horizontal reference lines: v15 composed at 0.537 and Breuer FBA at 0.59.

**The story:** the bulk of v15's MCC comes from knowledge-based priors, not from learned features. The Tier-1 ML detector (red bar) was tested rigorously and falsified — adding ESM-2/ESMFold/MACE features on 455 rows hurts MCC instead of helping.

### `confusion_matrices.png`

A 2×3 grid of heatmaps showing the confusion matrix at each full-panel version (v10 → v15). Cells: `[[TN, FP], [FN, TP]]`. Each subplot title carries the version, MCC, precision, recall.

**The story:** false positives stayed flat at **3** from v12 onwards; the entire MCC lift came from converting false negatives into true positives (FN: 161 → 96, TP: 222 → 287). v15 has near-perfect precision (0.990) — only 3 false positives across 290 positive predictions.

## Why scripts and not pre-rendered PNGs

Two reasons:

1. **Reproducibility** — anyone with `pandas` + `matplotlib` can regenerate the figures from the CSVs in 10 seconds. The CSVs come from `memory_bank/facts/measured/mcc_against_breuer_v*.json`.
2. **Customizability** — tweak the script's color palette, font size, or axes for the specific application (slide deck, poster, paper, README image) without re-querying the underlying facts.

If you want pre-rendered images committed for a specific use case (e.g. the README's "Visual summary" section), generate them locally and either inline-embed via base64 or commit them under `figures/output/` and unignore that path in `.gitignore`.

## Source provenance

Every number in the CSVs comes from a fact JSON in `memory_bank/facts/measured/`:

| CSV column / row | Source fact |
|---|---|
| `mcc_progression.csv` v15 row | `mcc_against_breuer_v15_round2_priors.json` |
| `mcc_progression.csv` v14 row | `mcc_against_breuer_v14_annotation_expansion.json` |
| `mcc_progression.csv` v13 row | `mcc_against_breuer_v13_trna_priors.json` |
| `mcc_progression.csv` v12 row | `mcc_against_breuer_v12_imb155_patches.json` |
| `mcc_progression.csv` v10/v10b rows | `mcc_against_breuer_v10_full.json`, `..._v10b_full.json` |
| `mcc_progression.csv` v9 row | `mcc_against_breuer_v9.json` + `mcc_v9_robustness.json` |
| `detector_contributions.csv` Tier1_XGBoost row | `mcc_tier1_xgboost_naive_stack.json` |
| `confusion_matrices.csv` all rows | the same `mcc_against_breuer_v*.json` files above |
| `confusion_matrices.csv` Breuer FBA reference | Breuer 2019 paper (cited; not in repo) |

If a fact JSON updates, regenerate the relevant CSV row and re-run the scripts.
