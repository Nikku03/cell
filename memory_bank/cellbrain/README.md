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

## Limits fixed where possible (fixbank, cellfix G8, cellmap2; 2026-11-23)

- **New layers:**
  - Protein: `PROT_PROCAN`, `PROT_CCLE` (local), `PROT_HPA_IHC`.
  - Deeper RNA: `EXPR_HPA_SCT`.
  - Interactions and pathways: `PPI_BIOPLEX` (local), `PATH_PROGENY` (local).
  - Timing: `TIMING_HPA_CCD`.
  - Maps: `CELLMAP_DEEP`, `CELLMAP_CANCERPROT`, `PROTEIN_OFFSET_PROCAN`.
- **Tests:**
  - **PASS:** protein from RNA in cancer lines (0.46 → 0.84); deeper RNA finds real proteins; jobs of unknown proteins from neighbours (AUROC 0.77).
  - **FAIL:** protein offsets in tissues; partners by cell line (just under the bar); TF on/off in general (39 TFs confirmed); machine tests with protein.
- **New facts:**
  - Cell summaries: `CELL-DEEP-*` (with `tf_on`, `pathways`, `protein_stain`, `cell_cycle_varying`) and `CELL-CANCERPROT-*`.
  - Gene facts: `protein/per_rna_offset_cancer_lines`, `predict/job_from_neighbours`, `partners/bioplex_by_cell_line`.
  - Complex facts: `COMPLEX-*` `machine/protein_comovement`.
- **The completed list of limits:** see `rem/atlas/RESULTS_TWOPROBLEMS.txt`.

## Address tags, partner finder, RNA versions (addressbank, addresstest G9; 2026-11-24)

- **New layers:** `ADDR_UNIPROT` (signal / transit peptides, transmembrane segments, NLS, ER-retention, peroxisome signal, GPI, Pfam domains, isoforms whose alternative sequence touches the tag), `ADDR_MITOCARTA` (local), `RNA_FRACTIONS_ENCODE` (K562 and HepG2 cell fractions), `ISOFORM_HPA_TISSUE` (each RNA version in 40 tissues).
- **Tests:**
  - **PASS:** address tag → room (signal OR 5.7, mito transit OR 88, NLS OR 6.3); the common-sense partner finder (fit → same destination → made together) finds a known partner in the first 10 tries for 18.3% of held-out proteins (random 1.2%) and beats fit alone (median +0.38 log2).
  - **FAIL:** RNA in the membrane fraction (the ENCODE fraction is not an ER read-out); RNA versions that change the tag vs dual-room proteins (imaging cannot tell isoforms apart).
- **New facts:** `CLAIM-G9-*`; gene facts `isoform/tag_changing_by_tissue` (189 genes: where the tag-changing version is used, and splicing proteins that follow it; correlation, not cause).
- **Outputs:** `outputs/address/address_tests.json`, `outputs/address/isoform_when_and_governors.json`.

## RNA at the ER and mitochondria, measured in place (erlocbank, erloctest G12; 2026-11-24)

- **New layer:** `APEX_SEQ_HEK293T` (local; Fazal et al. 2019 APEX-seq: RNA enrichment at the ER membrane, ER lumen, mitochondrial surface and matrix, cytosol and four nuclear places; mitochondrial surface under puromycin / cycloheximide / CCCP).
- **Tests (all FAIL as declared):** ER-made proteins' RNA at the ER membrane AUROC 0.69 (bar 0.70; strongly real, +0.66 vs −0.08 log2); mitochondrial proteins' RNA at the mitochondrial surface 0.42 (no); puromycin sensitivity 0.57 (real, under the bar).
- **New facts:** `CLAIM-G12-*`; gene facts `rna_location/er_membrane_without_er_tag` (289 RNAs at the ER whose protein has no annotated ER tag). Lists stay local (source terms).
- **Meaning:** the RNA of ER proteins travels to the ER while being made; for mitochondria in these cells it mostly does not.

## ESM C protein language model (esmcbundle, colab/esmc G10; 2026-11-24)

- **Run:** on the user's Colab L4 from a fixed bundle (`colab/esmc/`, sha256 in `manifest.json`) and declared code; results checked against the declared code id and recorded with `esmcbundle.py record`.
- **Tests:**
  - **PASS:** address from the whole sequence beats tags alone (61% of held-out proteins better; e.g. nuclear AUROC 0.65 → 0.76); partner finder + ESM C: partner within 10 tries 40.5% → 51.0%, median tries 21 → 10.
  - **FAIL (real, just under the bars):** hidden addresses (AUROC 0.687 vs 0.70); isoform moves (0.57 vs 0.60).
- **New facts:** `CLAIM-G10-*`. Fingerprints were not kept (re-run the notebook with `KEEP_FINGERPRINTS_IN_DRIVE = True` to bank them).

## What governs RNA versions (splicebank, splicetest G11; 2026-11-24)

- **New layer:** `SPLICE_KD_ENCODE` (ENCODE shRNA knockdowns of 451 RNA-binding proteins in K562 and HepG2, summed per UniProt isoform; TPM and counts per replicate).
- **Tests:** **PASS** — splicing-protein knockdowns shift the address-changing RNA versions more than other RNA-binding-protein knockdowns (72% of genes; also true for all other versions). **FAIL** — the tissue correlation does not pick the right governor.
- **New facts:** `CLAIM-G11-*`; gene facts `isoform/governors_measured` (205 genes; noisy — single hits are weak).
- **Disclosure:** G9's descriptive "candidate governors" used sums that skipped most splicing proteins; prefer `isoform/governors_measured`.
- **Page (2026-11-25):** the cell-map page now has a Proteins view (address tags, likely partners from the tested finder, RNA versions with ESM C predictions, tissue use and measured controllers) and a Tests view. Built by `rem/atlas/proteinpage.py`; partner suggestions in `outputs/address/partner_suggestions.json.gz`.

## Partner-list checks (partnercheck G13; 2026-11-25)

- **PASS:** ESM C helps outside big machines (contacts only: within 10 tries 25.8% → 36.4%).
- **PASS:** the page's lists find BioPlex partners never used to build them (9.9% within 10 tries, 9.8× chance; ESM C beats the finder alone).
- Output: `outputs/address/partner_checks.json` (summary; BioPlex itself stays local).

## Ribosomes in 3D pictures of human cells (cryoetbundle, colab/cryoet G14; 2026-11-25)

- **Run** on the user's Colab L4 (CryoET Data Portal; local only, no data licence stated).
- **R1 FAIL (under the bar):** template-matching finder F1 0.46 on held-out hand-labelled HSC-3 tomograms (bar 0.50; scrambled control 0.13). The depositors' own picks reach 0.74.
- **R2 VOID:** Jurkat membrane-near share 0.35 vs predicted 0.19, but random positions give 0.25 and scrambled picks 0.43 — not a specific measure in cryo-sections.
- **Lesson:** counting machines in cells needs a better finder (finer voxels, CTF-aware search, or a model trained on the portal's many automated picks) and whole-cell lamellae rather than old cryo-sections.

## Predicting the next state (lagtagbank, lagtagtest G15; 2026-11-27)

- **New layer:** `LAGTAG_GATA4` (local; Takei et al. 2026 GEO GSE305653: mouse ES cells before / after Gata4, RNA Pol II per gene, with human orthologs).
- **FAIL:** CollecTRI GATA4 targets do not specifically switch on (OR 1.65, p 0.17); two-step targets enrich (OR 1.71, p 8e-5) but no more than any regulator's two-step set.
- **Meaning:** tissue-derived regulator lists do not carry over to an early fate switch in another species; the next-step problem needs cell-type-specific regulator data (e.g. measured before/after pairs in the cell type of interest).

## 3D chromosome map vs gene expression over time (k3dbank, k3dtest G16; 2026-11-27)

- **New layer:** `K562_PMA_3D` (local; Bond et al. 2023 GEO GSE213909): K562 + PMA, per gene RNA TPM and intron signal at 8 times x 2 replicates; open/closed compartment value per 100-kb bin at 0, 6, 72 h; 33,914 loops with counts in 30 libraries. Hi-C read remotely with `hicremote.py` (no 6 GB downloads).
- **C1 PASS:** genes in the 10% most-shifted bins move with the shift (median +0.38 log2, 67% agree; p_MC 0.001).
- **L1 PASS:** genes at gained loops go up, at lost loops go down (median +0.25 log2, 62%; p_MC 0.001).
- **T1 FAIL (reverse):** at 6 h transcription has made 67% of its 72 h change, loops 31% — genes move first, loops follow.
- **Meaning:** 3D shape and expression are tightly linked, but in this switch loops lock in activity changes rather than start them. Next: repeat on THP-1 (8 matched time points) and A549 (ENCODE) to see if the order holds.

## Same test in THP-1 macrophages, 8 matched time points (t3dbank, t3dtest G17; 2026-11-27)

- **New layer:** `THP1_LPIF_3D` (local; Reed et al. 2022 GEO GSE201376; files on GRCh37 -> genes placed with GENCODE 46 lift37): RNA TPM + intron signal and compartment E1 at 0, 30, 60, 90, 120, 240, 360 min, 24 h; 42,690 loops with counts in 5 replicate columns.
- **C1 PASS (repeats, stronger):** genes in the most-shifted bins move with the shift (median +0.64 log2, 78%; already +0.16 at 2 h).
- **L1 FAIL (does not repeat):** changing loops at the gene start do not predict direction (median +0.06, 51%).
- **T1 PASS (repeats G16's finding):** transcription reaches 46% / 68% / 74% of its 24 h change at 2 / 4 / 6 h, loops 32% / 39% / 42%; mature mRNA lags about as much as the loops.
- **Meaning (K562 + THP-1):** the open/closed neighbourhood is a robust 3D partner of gene activity in both; loops track genes in the slow K562 switch but not in the fast immune switch; in both, gene activity changes first and the 3D map follows.

## Same test in A549 + dexamethasone, ENCODE (a3dbank, a3dtest G18; 2026-11-27)

- **New layer:** `A549_DEX_3D` (ENCODE, free use): RNA TPM + intron signal (4 replicates) and compartment E1 at 0, 1, 4, 8, 12 h; 60,457 joined HiCCUPS loops with counts in 20 replicate maps. `hicremote.pixels()` fetches only the blocks around given pixels.
- **The steroid barely changes the 3D map:** compartments r >= 0.98 to 0 h; 4 of 60,457 loops change at q < 0.1.
- **C1 PASS (third cell type):** the most-shifted bins' genes follow (median +0.27 log2, 79%).
- **L1 FAIL:** most-changed loops lean the right way (60%) but no more than shuffled labels (p_MC 0.23).
- **Timing not measurable** (no loop change to time) -- genes respond without the 3D map moving.
- **Across G16-G18:** open/closed neighbourhood tracks genes in all 3 cell types; loops only in K562; gene activity leads and the 3D map follows (or does not need to move).

## The folding rules (loopcodebank, loopcodetest G19, loopdyntest G20; 2026-11-28)

- **New layers:** `LOOPCODE_K562` (local; K562 CTCF sites with motif direction from the hg38 sequence, cohesin / H3K27ac / JUN / ATAC site counts at 0, 6, 72 h, six ENCODE histone marks per 100-kb bin) and `LOOPCODE_HCT116` (ENCODE; loop and control pixel counts with cohesin present, destroyed, and returning; E1 with and without cohesin). `hicremote.py` now reads .hic v7-9.
- **Loops = motor + stop signs (G19, all PASS on unseen chromosomes):** 77% of loops with a clear CTCF site at each end are convergent (chance 25%); the CTCF map with directions ranks loop pairs 13x above chance (AP 0.08 distance -> 0.18 -> 0.32 with direction); destroying cohesin removes 89% of loop signal, and it returns within an hour of cohesin coming back.
- **Zones = like with like (G19):** unchanged without cohesin (r 0.98); predicted from chromatin marks at r 0.84.
- **Change follows activity (G20, all PASS):** loop change r 0.44 from CTCF + cohesin, 0.57 with activity added; anchors' H3K27ac reaches 52% of its change by 6 h, loops 37%, zones 33%; zone change predicted at r 0.48 from activity change.
- **Loops act by contact (G20):** a gene follows a looped enhancer's activity change (r 0.22) better than an equally distant unlooped one (0.12).
- **Open:** CTCF explains about half of the loops (19% have no clear CTCF site at either end); no physical 3D simulation yet; cause shown only for cohesin.

## The folding simulator (foldsim, foldsimbank, foldsimtest G21, foldpage; 2026-11-28)

- **`rem/atlas/foldsim.py`:** motors + directional stop signs on a line of 5-kb sites -> loops and contact maps; four numbers fitted once on DEV (motor reels ~200 kb, one per 150 kb, capture 1.0, contact ~ distance^-0.75), optional landing preference for active DNA. Never sees a measured contact. `python3 -I rem/atlas/foldsim.py demo`.
- **New layers:** `FOLDSIM_K562_MAPS` (local; 88 fixed 3-Mb regions at 25 kb) and `FOLDSIM_HCT116` (ENCODE; the same regions with / without cohesin + HCT116 CTCF sites with motif direction). `hicremote.region()` reads a map window.
- **S1 PASS:** simulated vs measured K562 maps 0.38 (random stop signs 0.13). **S2 PASS:** finds loops, AP 0.175 vs 0.081 for distance (fitted formulas: 0.18 / 0.32). **S3 PASS:** predicts loop change over 72 h, r 0.24, mostly from motors landing where activity is. **S4 VOID:** zones r 0.49 but its planted control failed. **S5 PASS:** transfers to HCT116 (0.40) and the match drops without cohesin (0.23).
- **Final page:** "How a Chromosome Folds" (rules, in-browser simulator, measured vs simulated maps, evidence), built by `foldpage.py` from the results files; copy in `outputs/foldsim/page/`.
- **Open:** a 3D physical model; loops made without CTCF; a properly powered zone test; perturbation tests (single stop-sign deletions / inversions).
