# A100 Colab bundle

Self-contained run for the cross-organism gene-essentiality project: a GPU
transformer + smart cooccurrence rescue + a self-learning loop + a thesis
production loop. Open **`essentiality_a100.ipynb`** in Colab (A100 runtime) and
run top to bottom, or run the modules directly from the repo root.

```bash
python colab/af_torch.py --big          # 1. transformer LOCO (GPU) -> af_torch_preds.npz
python colab/smart_cooccur.py           # 2. calibrated cooccur rescue @ P>=0.90
python colab/active_learning.py         # 3. active-learning curves
python colab/thesis_loop.py             # 4. splice results into CAPSTONE_PAPER.md
```

## Modules

| file | what it does | output |
|---|---|---|
| `af_common.py` | paths, cache loader, leak-clean MSA masking, ranking metrics | — |
| `af_torch.py` | PyTorch port of the ortholog-MSA attention model + confidence head; LOCO over 5 clades; `--big` scales to multi-head/2-block | `af_torch_results.json`, `af_torch_preds.npz` |
| `smart_cooccur.py` | continuous, leak-clean cooccur channels (backup/presence/co-ess/phyletic/**dN/dS**) → cross-clade calibrated logistic stacker → threshold that maximises rescued coverage at a 90% precision floor; dN/dS ablation | `smart_cooccur_results.json` |
| `active_learning.py` | leave-one-clade-out AL simulation; reveal held-clade labels by random / uncertainty / low-brightness; AUC-vs-labels curves | `active_learning_results.json`, `.png` |
| `build_presence.py` | materialise the OG presence matrix; optional `--extra_presence_csv` to union OG-assigned extra genomes | `presence_matrix.npz` |
| `thesis_loop.py` | render results table + splice into `CAPSTONE_PAPER.md` between AUTOGEN markers | `RESULTS_AUTOGEN.md` |
| `fetch_proteomes.py` | NCBI efetch (fasta_cds_aa) for DEG / GTDB genomes; stdlib only | `work/proteomes/*.faa`, `*.genes.csv`, `deg_manifest.csv` |
| `assign_ogs.py` | mmseqs2 easy-search proteomes vs `og_reps.faa`; best-hit OG per gene | `work/og_assignments.csv` |
| `integrate_deg.py` | match DEG-essential gene names to proteomes, append DEG orgs to the driver CSVs (with match-rate report + drop filter) | `data/drive_import/labels_aug/` |
| `build_cache.py` | parameterised, bit-faithful rebuild of `af_msa_cache.npz` from any labels dir | `af_msa_cache_aug.npz` |

## Scaling up ("cover all species") — path A vs B

- **Path A (DEG labels) — the real lever.** Adds ~40 labelled organisms (new
  clades + more orgs in existing clades). Run notebook section 7. The cache is
  rebuilt by *appending rows to three CSVs* — `build_cache.py` reproduces the
  original cache bit-for-bit, so the augmentation is the only change.
- **Path B (GTDB presence-only) — structurally a no-op here.** Every cooccur
  channel is gated on *labelled* organisms; presence-only genomes are excluded
  from all correlations, and `phyl` is prebuilt. So more genomes ≠ more signal
  under the current design. The fetch/assign/presence scripts exist for a future
  redesign (phyletic-profile partner search), but today **C collapses to A.**

## Design notes

- **Why a stacker, not transformer features.** Adding cooccur as focal features
  was flat — the MSA already encodes ranking signal. The orthogonal value of
  cooccur is on the *abstention bucket* (brightness < 0.85). The stacker is fit
  **cross-clade** (score clade *c* with a model trained on the other 4) so the
  operating point is honest.
- **Most coverage, least precision drop.** The acceptance threshold is chosen to
  maximise rescued genes subject to *combined* precision ≥ 0.90 (configurable via
  `PRECISION_FLOOR`).
- **dN/dS** is already in `af_msa_cache.npz` (focal cols 5–6) and is added as a
  stacker channel; the run reports its marginal coverage via the `--no_dnds`
  ablation. Expect a small effect (~46% gene coverage, overlaps the core).
- **Data scope.** No 8,300-genome set lives in the repo; that earlier depth
  feature was flat. Use all in-repo orgs for presence; expand only with
  OG-assigned extra genomes.

## Requirements

Colab already has `torch`, `numpy`, `matplotlib`. No extra installs needed for
the default run.
