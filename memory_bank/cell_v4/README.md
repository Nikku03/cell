# cell_v4: all the measured knockdown data, compact and fast

Written by `rem/atlas/cellbank4.py build`. Read with `cellbank4.Bank(path)`.

## What it holds

| File | Contents |
|---|---|
| `records.npz` | `codes` (int8, one row of 128 per record), `scale` (float32, one per record), `basis` (float16, 8,248 genes × 128) |
| `records_meta.json` | per record: gene, Ensembl id, screen (`A`, `B`, `C`), split (`TRAIN` / `TEST`, same SHA-256 rule as the repo), energy-test p |
| `genes.json` | per knocked-down gene: interaction partners (HPA consensus) and the 20 most co-essential genes (DepMap, K562 removed) with r |
| `index.sqlite` | table `records(id, gene, ensembl, screen, split, energy_p)`, indexed by gene |
| `journal.jsonl` | append-only log for later writes (`Bank.add`) |
| `build_report.json` | record counts, storage, query speed, compression check H1 |

A record is one knockdown's specific response:
1. Take the screen's normalised bulk expression.
2. Remove the screen's own mean response over TRAIN genes.
3. Map onto screen A's 8,248 genes; genes the screen did not measure count as 0.
4. Project onto the 128 components fitted on screen A's TRAIN responses.
5. Store as int8 codes plus one scale.

Only knockdowns with a phenotype (energy-test p < 0.05) are stored. So are only genes that have screen-A features; the build report says how many were dropped.

To decode a record: `codes[i] * scale[i] / 127 @ basis.T`.

## Sources (all CC BY 4.0)

| Screen / table | Source |
|---|---|
| A | Replogle et al. 2022, K562 genome-wide Perturb-seq, day 8 (figshare 20029387) |
| B | Replogle et al. 2022, K562 essential-gene Perturb-seq, day 6 |
| C | Replogle et al. 2022, RPE1 essential-gene Perturb-seq |
| Partners | Human Protein Atlas v25 interaction consensus |
| Co-essentiality | DepMap 24Q4 CRISPRGeneEffect, K562 (ACH-000551) removed |

## Rules for users

- The bank stores TEST records too, so that it holds all the data. Any evaluation must hide:
  - all screen-A TEST records;
  - for each query gene, that gene's own records in every screen.

  `cellbank4.py` does both.
- Screen B's record of a gene is a replicate of its screen-A answer (the noise ceiling, 0.315). Never use it to predict that same gene.
