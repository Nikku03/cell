# Data acquisition — fetching the remaining external datasets to Drive

The big genomics datasets can't live on GitHub (100 MB/file limit) or be streamed from an ephemeral
sandbox — acquisition runs **on Colab with Drive mounted** (ample disk, in-place reads), exactly like the
existing `download_all_data.ipynb`. This runbook covers the datasets that unblock the Better/Diverse
improvements. Run `colab/fetch_external_data.ipynb`; everything lands in `MyDrive/virtual_cell_data/`.

**Already on Drive (skipped):** ARCHS4 (`expression_geo/archs4_human_gene.h5`, 59 GB) and `lincs_train.npz`.

All URLs below were reachability-checked 2026-07-06.

| # | dataset | unblocks | access (verified) | size | license | Drive path |
|---|---|---|---|---|---|---|
| 1 | **Perturb-seq** (scPerturb: Norman 2019, Replogle 2022) | trained perturbation model (Better 1), causal edges (Better 2) | Zenodo record `13350497`, direct h5ad ✅ | ~1–10 GB | CC-BY | `perturbseq/` |
| 2 | **Cell-type expression** (CELLxGENE census / Tabula Sapiens) | context-specific networks (Diverse 1), cell↔tissue (Better 3) — **populates the empty `emask`/`abund`** | `cellxgene-census` Python API ✅ | ~2 GB | CC-BY | `celltype_expression/` |
| 3 | **Disease-state expression** (TCGA via UCSC Xena) + dev/disease scRNA (census) | beyond cancer/healthy (Diverse 3), disease attractors | Xena GDC hub direct ✅; census API | ~1–20 GB | open / CC-BY | `atlases/` |
| 4 | **Spatial transcriptomics** (10x Xenium, Vizgen MERFISH) | real tissue geometry (Diverse 2) | 10x/Vizgen direct S3 | ~1–30 GB/sample | CC-BY | `spatial/` |
| 5 | **Cross-species conservation** (UCSC phyloP100way) | conservation prior for dark genes (Diverse 4) | `hgdownload.soe.ucsc.edu` direct ✅ | ~9 GB bigWig → tiny per-gene TSV | open | `conservation/` |
| 6 | **Tahoe-100M** (Arc Institute) | large drug-perturbation model (Better 1) | HuggingFace `arcinstitute/Tahoe-100M` 🔑 token | ~100s GB (fetch a subset) | CC-BY | `tahoe100m/` |
| 7 | **AlphaFold** (per-accession) | structure/fold layer (fixes the 404) | AFDB API `alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb` ✅ | ~KB/protein | CC-BY 4.0 | `alphafold/` |

## Notes / access quirks found during verification
- **AlphaFold** — the EBI *proteome tar* path (`.../latest/UP000005640_9606_HUMAN_v4.tar`) returns **404**;
  the per-accession AFDB API is stable, so the notebook fetches only the accessions the model uses (from
  `D['acc']`) rather than the whole proteome. Bulk alternative: `gs://public-datasets-deepmind-alphafold-v4/`.
- **Tahoe-100M** — HuggingFace returns **401** without a token; `huggingface-cli login` first, and use
  `allow_patterns` to pull one plate/shard (the full corpus is hundreds of GB).
- **CELLxGENE census** — not a plain URL; use the `cellxgene_census` API. Iterate tissues and mean per
  `cell_type` to avoid loading the whole atlas into memory.
- **CORUM** — host was unreachable (SSL/down); it's **optional** — Complex Portal (already integrated)
  covers named complexes. Skipped rather than block on a mirror.
- **figshare** (an alternative Perturb-seq host) bot-challenges direct downloads — the notebook uses the
  Zenodo mirror instead, which serves the files cleanly.

## Wiring the new data into the model
After the fetch, set the env vars the build already reads and re-run `build_complete_cell.ipynb`:
- `PERTURBSEQ_NORMAN_URL` / `PERTURBSEQ_RPE1_URL` → the local h5ad paths (`compute_perturbseq.py`, Model 4).
- `celltype_expression/` → feeds `emask`/`abund` (Model 2) — the piece that was empty in the explorer export.
- `atlases/tcga/` → disease signatures for the reversal engine's disease attractors.
- `conservation/phyloP_pergene.tsv` → a new orthogonal prior (small enough to commit to the repo).

**Discipline:** each new layer must keep `colab/recovery_scorecard.py` at all-PASS — add data only where it
measurably improves a recovery/calibration number, or record the negative (the repo's standing rule).
