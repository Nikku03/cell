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
