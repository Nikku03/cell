# cellbrain: both data groups in the brain's memory bank

Built by `rem/atlas/cellbrain_bank.py build`. It uses the brain's own bank (Nikku03/money `brain/bank.py`; brain commit in `outputs/cellbrain/bank_report.json`):
- append-only;
- bitemporal;
- every fact names its source and tier;
- derived facts cite their inputs.

The bank itself lives in `rem/atlas/_cache/cellbrain/bank`, which is local and git-ignored, because some inputs are CC BY-NC-SA or of unclear licence. This directory documents it.

## What is in it (2026-11-20)

- **75 layer facts.** One per dataset, from both groups:
  - 42 pointer layers: files as pulled;
  - 33 derived layers: gene-indexed matrices built here, with rows as Ensembl ids.

  Each layer fact gives the path, sha256, shape, rows and columns, units, transform, group, source and licence. `layer(name)` refuses a file whose sha changed.
- **Genes (about 45,000 entities):**
  - identity: GENCODE 46 and HGNC;
  - memberships: Complex Portal complexes, Reactome pathways, HPA locations, Open Targets drugs;
  - which layers hold the gene.
- **Other entities:**
  - 17 screens;
  - 496 ENCODE biosample terms;
  - 883 CELLxGENE cell types and 64 tissues;
  - 2,143 complexes;
  - 2,180 pathways;
  - 13,128 drugs.
- **The connections:** added by `rem/atlas/cellbrain.py` as derived facts named `link/<family>/<other entity>`, each citing the layers it used.

## Commands

| Command | What it does |
|---|---|
| `python3 -P rem/atlas/cellbrain_bank.py check` | checks every layer's sha and verifies 500 entities |
| `python3 -P rem/atlas/cellbrain_bank.py card ENSG00000102145` | everything the bank holds about a gene (here GATA1), connections included |

## Connections the brain made (cellbrain C2/C3, 2026-11-20; `rem/atlas/RESULTS_TWOPROBLEMS.txt`)

| Family | Confirmed links | Verdict |
|---|---|---|
| F1 data <-> data | 507 view pairs | PASS |
| F2 gene <-> gene | 9,074 pairs, >= 3 kinds of data | PASS (C3) |
| F3 cell type <-> cell type | 100,440 among 4,512 units | PASS |
| F4 disease <-> cell type | 7,426 trait-biosample links, 959 of 1,507 traits | PASS |
| F5 disease <-> gene <-> tissue | 171,854 shared-variant links | descriptive |
| F6 knockdown <-> cell identity | 107,872 (75,514 genome-wide) | PASS |
| F7 complex / pathway <-> cell type | 10,623 | FAIL (planted 87% < 90%) |
| F8 drug <-> disease | 35,810 untested candidates; OR 1.24 | PASS (weak) |
| F9 disease <-> disease | 9,398 | PASS |

The strongest 20,000 public links per family are in `outputs/cellbrain/F*_connections.json`. Every link is in the bank: `python3 -P rem/atlas/cellbrain.py card ENSG00000102145`.

## Added from Biohub (2026-11-20; `rem/atlas/biohubbank.py`, `outputs/cellbrain/biohub_report.json`)

- **OpenCell** (CC BY-SA 4.0), human proteins tagged in HEK293T cells:
  - where 1,310 proteins sit in the cell (17 compartments);
  - 29,922 protein-protein interactions from 1,260 pull-downs;
  - RNA and absolute protein amounts for 19,255 genes;
  - library metadata.

  Gene facts: `opencell/localization`, `opencell/interactors`, `opencell/abundance_hek293t`. Layers: `SETS_OPENCELL_LOCATION`, `PPI_OPENCELL`, `EXPR_OPENCELL_HEK293T`.
- **Psoriasis CRISPR screen** (Zhao, Shih et al. 2026; CC BY-NC-ND, **local only**):
  - genome-wide knockout in primary human skin cells (keratinocytes), sorted on IL17RA, with results for 18,785 genes;
  - protein profiles of the ALOX5 and OXTR knockouts;
  - mouse skin proteomics under three treatments.

  Gene fact: `screen/psoriasis_il17ra`. Entity: `SCREEN-PSORIASIS_IL17RA`.

In total, 44,082 facts and 16 layers were added.

## Connections re-run with the Biohub data (C4, 2026-11-20)

- **Data ↔ data:** PASS. 551 view pairs; OpenCell interactions agree with HPA interactions, knockdowns and complexes.
- **Gene ↔ gene:** PASS. 9,408 gene pairs.
- **New F10, psoriasis screen ↔ the bank:** FAIL. No link to skin-cell expression, psoriasis genetics or IL17RA's network.
  - ALOX5 and IL17RA (brodalumab) were found.
  - OXTR missed the cutoff.
  - Gene rows stay local (CC BY-NC-ND).
