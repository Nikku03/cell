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

## Active genes per cell type and gene influence (genereg G1, 2026-11-20)

- **Active genes per cell type** (layer `ACTIVE_GENES_CELLTYPE`): 841 cell types × 49,451 genes, with share of cells, level and specificity. Gene facts `role/housekeeping` and `role/marker_of`; per-cell-type facts `active/summary`.
- **Who switches whom in K562** (layer `GENE_INFLUENCE_K562`): 110,034 links confirmed in two independent screens (32,558 activations, 77,476 repressions). Gene facts `influence/activates`, `represses`, `activated_by`, `repressed_by`.
- **Co-expression switch map** (layer `GENE_COEXPRESSION_TF`): 71,131 transcription factor → gene links across cell types. Gene fact `influence/coexpressed_targets`.
- **Concepts per regulator** (Reactome): gene facts `concepts/activates`, `concepts/represses`, `concepts/coexpression`.
- **Tests:** all 3 FAIL as declared.
  - The textbook K562 examples were too weak in this data.
  - Activations are only slightly co-expressed across the body.
  - The co-expression map does not predict K562 knockdowns.

  The per-cell-type regulators and programmes read like the textbook.

## Influence across cell types, and its level (generegx G2, 2026-11-21)

- **New data** (`xcellbank.py`): the same ~2,400 knockdowns read out gene by gene in K562, RPE1, HepG2 and Jurkat, plus the K562 genome-wide screen (Nadig 2025, CC BY 4.0). Layers `XCELL_K562ESS`, `XCELL_RPE1ESS`, `XCELL_HEPG2ESS`, `XCELL_JURKATESS`, `XCELL_K562GW`. CollecTRI TF → target (`REG_COLLECTRI`, local only).
- **Results:**
  - **Influence changes with cell type:** PASS. Agreement with K562: second K562 experiment 1.12, Jurkat 0.78, HepG2 0.49, RPE1 0.41.
  - **Follows the gene's own expression:** FAIL.
  - **Effect per unit of knockdown as the level:** PASS, but weak (raw effect is still the steadier measure).
  - **Direct vs through another gene:** FAIL (does not match curated TF targets).
- **Gene facts:**
  - `xcell/influence_by_cell_type` (reach and expression per line, agreement with K562);
  - `xcell/targets_*` and `xcell/regulators_*` (everywhere, flips_sign, only_<line>, with log2 fold change and effect per unit of knockdown);
  - `xcell/k562_direct_targets`, `k562_indirect_targets` (with the middle gene);
  - `xcell/tf_hierarchy_k562`.
- **Monte Carlo check (generegmc, 2026-11-22):** all four G2 verdicts are stable.
  - Chance runs: 10,000 / 10,000 / 200 / 1,000 scrambled-data runs. Chance never matched the two passes (p 0.0001 and 0.005) and easily matched the two fails (p 0.97 and 0.83).
  - Re-drawn runs: each verdict was the same in 200 of 200 runs (fresh random seeds plus re-sampled genes).
  - Bank facts: `montecarlo` on `CLAIM-G2-X1` … `CLAIM-G2-L2`.

## Prediction (generegp G3, 2026-11-22)

- **Predicting a knockdown in another cell type:**
  - Using the other three cell types beats the simple guess in K562, HepG2 and Jurkat. RPE1 misses the bar, so the declared claim FAILS.
  - In K562, the prediction is better than repeating the experiment.
- **Predicting from K562 alone:** works only for Jurkat (the other blood cell). Predictions for 1,149 knockdowns are stored as gene facts `predict/knockdown_effect_Jurkat`, marked as predictions, not measurements.
- **Predicting the next step (chain reaction):** FAIL. Effects do not chain in this simple linear way.
- **Claim facts:** `prediction_test` on `CLAIM-G3-T1`, `CLAIM-G3-T2`, `CLAIM-G3-C1`.

## The interaction-room rule (generegppi G4, 2026-11-22)

- **Rule:** if two proteins interact in one cell, they work together in another cell when both are made there and share a compartment.
- **Results:**
  - **Same compartment matters:** PASS in all five screens.
  - **Interactions seen in HEK293T carry over:** PASS in K562, RPE1, HepG2 and Jurkat, beyond simply sharing a compartment. Monte Carlo floor: none of 1,000 random draws matched.
  - **Not made → no partnership:** right direction, under the bar → FAIL.
- **Prediction from partners:**
  - Partners beat the earlier transfer model in K562, HepG2 and Jurkat. RPE1 misses the bar → declared FAIL.
  - In all four cell types, 200 random "partners" never matched (p 0.005).
  - In K562, partners alone (r 0.30) beat the other cell types (0.26) and a repeat experiment (0.24).
- **Facts:** `ppi_rule_test` on `CLAIM-G4-R1` … `CLAIM-G4-P2`. Details: `outputs/genereg/ppi_rule_summary.json`.

## Substitute proteins (paralogbank + generegpara G5, 2026-11-22)

- **New layers:**
  - `PARALOGS_ENSEMBL` (Ensembl Compara 116);
  - `EXPR_DEPMAP` (DepMap 24Q4 RNA, 1,673 lines; CC BY 4.0).
- **Results:** substitutes are real but small on average.
  - **Cover across cancer lines:** beats random genes, p 0.001, but under the bar → FAIL.
  - **Cover across the four screened cell types:** right direction, under the bar → FAIL.
  - **Stepping up:** the substitute rises a little, p 0.001, under the bar → FAIL.
- **Prediction:** the substitute's level helps predict which cancer lines need a gene → PASS.
- **Gene fact:** `substitute/closest_paralog` (substitute, identity, DepMap cover rho).

## One combined predictor (generegc G6, 2026-11-22)

- **Cancer lines (DepMap):** PASS. Combining line strength, partners (made, same room) and the substitute predicts which lines need a gene: r 0.05 (own RNA alone) → 0.25.
  - Partners add the most.
  - The substitute adds little: +0.005 among genes that have one.
- **The four screened cell types:** FAIL. The combined predictor beats the generic guess (0.34 → 0.41 pooled) but adds only +0.007 over strength alone. Clear gains in K562 and Jurkat.
- **Claim facts:** `combined_predictor_test` on `CLAIM-G6-*`.

## Cell maps down to single proteins (cellmap, 2026-11-22)

- **Layers:** `CELLMAP_BODY` (841 cell types), `CELLMAP_LINES` (4), `CELLMAP_CANCER` (1,673).
- **Per protein and cell:** made or not, room, number of active partners (interacting + both made + same room), estimated log2 amount.
- **Per cell:** complex completeness.
- **Fact `cellmap/summary`** on `CELL-<family>-<id>`: proteins per room, complete and nearly complete machines (with the missing parts), core and distinctive jobs, distinctive proteins, hubs, backups.
- **Also banked:** the G1 layers, and the Complex Portal human file.
- **Test G7-M1 (a complete machine needs its parts):** FAIL. RNA presence is too coarse for this in general. Some machines show it, for example cFLIP-CASP8-FADD.
