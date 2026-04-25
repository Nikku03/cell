# Multi-organism essentiality dataset

This directory holds the curated `(organism, locus_tag, essential, source)` labels that drive Session 20's multi-organism XGBoost predictor. The labels are public-domain or open-access; the original papers are cited per-row in `labels.csv`.

## Why we need this

Session 17 falsified the Tier-1 XGBoost stack on 455 Syn3A rows. Session 19 confirmed that switching to a smaller embedding (ESM-2 150M, 640 dims) does NOT recover the loss — the bottleneck is row count, not embedding dim. Pulling in ~10k labeled `(protein sequence, essentiality)` pairs from four other bacteria changes the regime: 1280:455 → 1280:~10,000, the ratio at which supervised learning on dense embeddings actually starts to work.

## Data sources (all open-access)

| Organism | Genome accession | Essentiality source | ~Genes labeled |
|---|---|---|---|
| *E. coli* K-12 MG1655 | NC_000913.3 | Baba 2006 (Keio collection); aggregated by DEG / OGEE | ~4,300 |
| *B. subtilis* 168 | NC_000964.3 | Commichau 2013 / DEG / OGEE | ~4,200 |
| *M. pneumoniae* M129 | NC_000912.1 | Lluch-Senar 2015 (eLife 4:e09943, Table S1) | ~700 |
| *M. genitalium* G37 | NC_000908.2 | Glass 2006 (PNAS) / Hutchison 1999 | ~480 |
| *J. craig venter syn3.0* (Syn3A) | CP016816.2 | Breuer 2019 (eLife) | 455 |

Total expected: **~10,135 rows** post-deduplication.

## How the labels are produced

`notebooks/embed_multiorg_session20.ipynb` (Cells 4-7) downloads each organism's essentiality table on Colab (where outbound HTTP is unrestricted), normalizes to a binary `essential` column, joins to the protein sequence pulled from NCBI GenBank, and writes:

- `memory_bank/data/multiorg_essentiality/labels.csv` — the joined `(organism, locus_tag, gene_name, essential, source)` table
- `cell_sim/features/cache/esm2_650M_multiorg.parquet` — the ESM-2 650M embeddings, indexed by `(organism, locus_tag)` composite key

The notebook is the single source of truth for how the labels are derived; this README is just the human-readable summary. Re-running the notebook regenerates both files end-to-end.

## Column contract

`labels.csv` schema:

| column | dtype | notes |
|---|---|---|
| `organism` | str | one of `ecoli`, `bsub`, `mpne`, `mgen`, `syn3a` |
| `locus_tag` | str | NCBI locus_tag from the source GenBank |
| `gene_name` | str | optional, may be empty |
| `essential` | int | 0 = nonessential, 1 = essential / quasi-essential |
| `source` | str | citation string (e.g. `"breuer_2019_elife"`) |

Composite key `(organism, locus_tag)` is unique across the table.

## Licensing

All upstream sources are either public-domain (NCBI GenBank), CC-BY (eLife papers), or research-use-only (DEG flat files). The merged CSV is shipped under the same terms as the rest of this repository. Cite the original papers — they're listed in `labels.csv:source`, not just here.

## Provenance

Created in Session 20. Access date is recorded inside the CSV header. If you re-run the notebook against newer data dumps the access date will update; the label flips that result are the responsibility of the runner to inspect.
