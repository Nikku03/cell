# NEXT_SESSION — queued work

_Read this file FIRST, immediately after running the invariant checker._

## Pre-flight

1. `python memory_bank/.invariants/check.py` — must print `OK`.
2. Read `PROJECT_STATUS.md`.
3. Read `memory_bank/concepts/essentiality/REPORT.md` for full MCC history (v0–v9).
4. Read `memory_bank/facts/measured/mcc_against_breuer_v9.json` for the latest honest result + diagnosis.
5. Read this file.

## Where Layer 6 stands (end of Session 11)

**Detector-design space: EXHAUSTED.**

| Version | Detector | MCC (n=40, 5 seeds) | Notes |
|---|---|---|---|
| v1 / v4 / v6b | ShortWindow variants | 0.064 ± 0.088 | High-variance pool noise. |
| v5 / v6a | PerRule / Ensemble pool_confirm | 0.112 ± 0.029 | Structural; medium variance. |
| **v9** | **RedundancyAwareDetector** | **0.125 ± 0.000** | **Deterministic; current best.** |

All detector variants plateau at the same 3 FPs (0034, deoC, lpdA) and 15 FNs (ribosomal / tRNA / translation). Session 11's v9 confirmed that the remaining MCC gap is **not a detection problem** — it is a **simulator biology incompleteness problem**.

## Session-15 top priority — close the two Tier-1 data gaps, then measure

Session 14's populate pass landed ESM-2 (real 1280-dim embeddings, all 455 CDS) plus two empty-by-design parquets for AlphaFold-DB and MACE-OFF. Commit `ebbfdff` carries the SHAs; `memory_bank/facts/measured/session_14_populate.json` records the outcome. The two empty parquets came from *data-availability gaps*, not extraction bugs:

### 0a. Structural features — PIVOT from AlphaFold-DB to ESMFold

**Session-15 finding (verified):** the Luthey-Schulten 4DWCM upstream repo does NOT contain a UniProt cross-reference CSV — its `input_data/` directory lists 14 files (`syn3A.gb`, `kinetic_params.xlsx`, `Syn3A_updated.xml`, `LargeSubunit.xlsx`, `iMB155_noUnqATP_lipdiomics_wPUNP5_noNBtransport.xml`, `protein_metabolites.xlsx`, etc.) and none maps the `AVX*` NCBI accessions to UniProt entries. The 4DWCM authors used their own local structure pipeline, not AFDB. Additionally the sandbox's network allowlist blocks `rest.uniprot.org` and `www.ebi.ac.uk`, so live queries can only run from Colab. Combined with the Session-14 finding that 4 different UniProt queries for taxid `2144189` all returned zero rows, the original AFDB path is doubly blocked.

**Recommended pivot**: replace `alphafold_extractor.py` with an `esmfold_extractor.py` that calls `transformers.EsmForProteinFolding` directly on the CDS amino-acid sequence. Same 9-feature schema (pLDDT mean/std, disorder fraction, helix/sheet/coil fractions, length, Rg, has_structure). No UniProt mapping needed; ESMFold takes sequence in, returns structure+confidence.

Steps:

1. New module `cell_sim/features/extractors/esmfold_extractor.py` — mirror ESM2Extractor's lazy-load pattern; use `EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")`. Input: CDS AA sequences. Output: same 9 cols as the AlphaFold schema so downstream consumers (Tier1 bundle, XGB detector) need no changes.
2. Retire `alphafold_extractor.py` OR keep it as a (working) path for when a UniProt map is later provided; add a new `esmfold_v1` source to the default registry.
3. Update `notebooks/populate_tier1_cache.ipynb` cell 6: replace the UniProt/AFDB block with a single ESMFold run. VRAM footprint on ESMFold v1 is ~24 GB at full precision, ~12 GB at fp16 — the Blackwell GPU the user ran Session 14 on has 95 GB, so fp32 is fine.
4. Delete `memory_bank/facts/structural/alphafold_extractor.json` in favour of `esmfold_extractor.json` (same schema, different source citation).
5. Re-run notebook; expect 100% coverage (every CDS has a sequence), ~10 s/protein on GPU → 455 × 10 s ≈ 75 min wall. Push refilled parquet.

Expected lift on MCC: replaces 455 rows of all-NaN with 455 rows of real structural features. The Tier-1 XGBoost `stacked` slice then has meaningful data; Session-15's v11 fact records that XGBoost was below v10b largely because the 9-dim AlphaFold block was all NaN (featureless). With real ESMFold features, the honest prediction is ~+0.05 on the balanced-40 panel, unclear on FULL-455. Needs a v13 measurement.

### 0b. Fix MACE-OFF SMILES map

Syn3A's SBML (`Syn3A_updated.xml`) encodes metabolites as BiGG-style species IDs (`M_atp_c`, `M_h2o_c`, ...). MACE-OFF needs real SMILES. Notebook cell 7 carries a `HAVE_CURATED_SMILES_MAP=False` guard that skips the backend entirely today. Fix:

1. Build `memory_bank/data/syn3a_species_smiles.csv` mapping every species ID in the SBML to a canonical SMILES. Primary source: KEGG COMPOUND (free bulk download + SBML annotation has `bqbiol:is` cross-refs to `kegg.compound:Cxxxxx` that can be followed). Fallback for metabolites KEGG doesn't cover: ChEBI.
2. Register the CSV via a pointer fact in `memory_bank/facts/structural/`.
3. Edit notebook cell 7 to load `SUBSTRATE_SMILES_MAP = dict(zip(csv["species_id"], csv["smiles"]))` and flip `HAVE_CURATED_SMILES_MAP = True`.
4. Re-run cell 7. Expected ≥ 60% of the 356 SBML reactions will have SMILES for every substrate; MACE-OFF writes real BDE-derived k_cat aggregates for those enzymes.

### 0c. Baseline Tier-1 MCC (XGBoost on cached features)  — **CLOSED in Session 15 (negative result)**

Landed as `cell_sim/layer6_essentiality/tier1_xgb_detector.py` + `scripts/run_tier1_sweep.py` + `memory_bank/facts/measured/mcc_against_breuer_v11_tier1_xgb.json`. Honest finding: **no Tier-1 XGBoost slice beats the v10b rule baseline on either set.** The FULL-455 and BALANCED-40 sweep outputs are persisted in `outputs/tier1_sweep_v11.json`. Best XGB aggregated MCC: 0.241 (esm2_only) on FULL vs v10b 0.364; 0.603 (esm2_plus_priors) on BALANCED vs v10b 0.800. Diagnosis: 5.3:1 class imbalance + 455 labels vs 1280-dim ESM-2 + priors already cover the easy signal -> ML overfits or collapses. Re-visit this fact **after** 0a and 0b land — the hypothesis is that structural + kinetic features are what address the 15 translation-machinery FNs that ESM-2 alone cannot. The null hypothesis (learned features dominate rules on this benchmark) is now falsified; a positive re-run would need to beat 0.364 FULL / 0.800 BALANCED to be worth publishing.

### 1. iMB155 pathway patches — **LANDED in Session 15 (Item 1)**

Patch shipped as `cell_sim/layer3_reactions/imb155_patches.py` + RealSimulatorConfig flag `enable_imb155_patches`. Clears three over-assigned loci (`JCVISYN3A_0034` placeholder, `JCVISYN3A_0228` lpdA, `JCVISYN3A_0732` deoC) from their `enzyme_loci` lists. When the sole catalyser was the patched locus, the rule is replaced with a python-closure variant that fires at `kcat × saturation` without enzyme gating. The PerRuleDetector's `gene_to_rules[<patched_locus>]` becomes empty, returning `no_catalytic_rules` for the KO — FP is closed by removing the detector signal, not by manipulating event counts. 11 unit tests plus fact `memory_bank/facts/parameters/imb155_pathway_patches.json`. Measurement lands as `mcc_against_breuer_v12_imb155_patches.json`.

### 2. Ingest Thornburg 2026's refined `kinetic_params.xlsx` (4DWCM repo)

Their 85 KB version vs our staged 59 KB 2022 version has refined k_cat / Km for ~30 reactions. Pull via:

```bash
curl -sS -o cell_sim/data/Minimal_Cell_ComplexFormation/input_data/kinetic_params_4dwcm.xlsx \
    https://raw.githubusercontent.com/Luthey-Schulten-Lab/Minimal_Cell_4DWCM/main/input_data/kinetic_params.xlsx
```

Diff sheet-by-sheet against our staged file. For each changed parameter, promote to a proper `facts/parameters/*.json` with Thornburg 2026 citation. Sharpens metabolic timing without changing detector behaviour.

### 3. Ribosome complex — **SUPERSEDED by FN re-audit (Session 15)**

The "15 FN wall" claim from Session 11 is **outdated**. Running the FN categorisation against `outputs/predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4_composed_all455_v10a.csv` (v10a = v10b's trajectory base) gives:

| FN category | Count | Mechanism today |
|---|---|---|
| Ribosomal proteins (rps*/rpm*/rpl*) | **1** (prp/ysxB, maturation protease only) | already caught by `ComplexAssemblyDetector` via complex_formation.xlsx membership; `make_complex_formation_rules` gates Ribosome assembly on all 58 subunits |
| tRNA / aminoacyl / modification enzymes | 11 (tilS, tsaB/C/D/E, trmD, mnmA, gatA/B/C, ribF) | missed by all three priors + PerRule |
| Translation factors (EF-Tu, IF, RF, etc.) | 0 | all caught by `translation_factor` annotation prior |
| Uncharacterised / membrane / metabolic | 86 | no catalysis signal; no complex membership; no annotation match |

**Implication**: the existing complex-formation machinery already implements the ribosome gate. Adding an "explicit ribosome rule" duplicates what's shipping. The real remaining tractable lever is the 11 tRNA-modification FNs.

**New Item 3 proposal — add tRNA-modification complex priors:**

- Three annotated multi-subunit complexes in this FN cluster:
  - **tsaBCDE** (JCVISYN3A_0079 / 0144 / 0270 / 0271) — tRNA threonylcarbamoyltransferase complex
  - **gatABC** (JCVISYN3A_0687 / 0688 / 0689) — glutamyl-tRNA amidotransferase
  - Standalone essentials: **tilS** (0040), **trmD** (0364), **mnmA** (0387), **ribF** (0291)
- Add these as new `ComplexAssemblyKB` entries (for the 2 multi-subunit ones) and as new annotation keywords (`trna_threonylcarbamoylation`, `trna_amidation`, `trna_lysidine`, `trna_methylation`, `trna_thiolation`, `fad_biosynthesis`) for the standalone ones.
- Expected lift: closes 8-11 of 98 FNs without introducing FPs (all 11 are Breuer-Essential). Honest prediction: +0.03 to +0.05 on the full 455 MCC.
- Implementation: add rows to `cell_sim/layer6_essentiality/complex_assembly_detector.py::_BUILTIN_COMPLEX_KB` and new keyword classes to `annotation_class_detector.py::_CLASS_RULES`. ~40 lines of code, ~8 tests, no new simulator module. A single session.

Estimate: 1 session (not 3+). Measurable via existing sweep script.

### 4. Multi-seed replicates as the default

Session 10's Block B found single-seed MCC numbers are unreliable. The `--panel-seed` flag now exists. Sweep-level default should be `--seeds 42 1 2 3 4` with aggregated fact output. One-line change to `run_sweep_parallel.py`.

## The honest ceiling bound

Path A tried (v7): falsified. Higher scale tried (v8): no help. All detector variants tried (v0–v9): plateau at 0.125.

**Landed so far:**
- v10b composed stack: MCC 0.364 (full 455) / 0.800 (balanced 40)
- v12 iMB155 patches: MCC 0.393 (+0.029 over v10b, three target FPs closed)
- v13 tRNA-mod priors: MCC 0.410 (+0.017 over v12, 9 TPs, zero FPs)
- v14 annotation expansion + NNATr fix: **MCC 0.494** (+0.084 over v13, **+0.130 cumulative over v10b**, 39 TPs, zero FPs, zero regressions)

**Reachable with remaining fixes:**
- Tier-1 XGBoost with populated ESMFold + MACE parquets (items 0a/0b Colab-side run pending) + stacked slice: likely +0.02–0.05 on balanced-40, unclear on full-455; re-measure the v11 negative result with real features
- Mining the 84 uncharacterized-protein FNs via structural + coevolutionary signals (requires ESMFold features); ceiling likely another +0.03–0.08 if half of them are truly essential and ESMFold pLDDT + disorder separate them from random Nonessentials
- Multi-seed v14 replicates to put variance bounds on the 0.494 result

**Not reachable without one of:**
- Full Thornburg 2026 simulator (2 A100 × 6 days per replicate — infeasible)
- ML surrogate trained on their 50 simulated cells + Breuer labels — forbidden by brief without new justification
- Rewriting iMB155 as a complete SBML (multi-month project; probably needs a metabolic modeller)

**Realistic ceiling for this project: MCC ≈ 0.35**. If you want 0.59, the brief goal needs revision (the current v2 proposal: "MCC > 0.40 with interpretability and sub-2s/gene inference").

## Deferred / done-elsewhere

- `cell_sim_rust` extension (Session 6), `PerRuleDetector` (Session 7), `EnsembleDetector` (Session 8), Path-A attempt (Session 9), Colab replicates (Session 10), `RedundancyAwareDetector` + metabolite sink (Session 11) — all shipped.
- Layer 5 (biomass + division) — still not the bottleneck; detection is.
- Hutchison 2016 secondary labels — cheap to add but doesn't lift MCC.

## Git

Session 1–11 commits on `origin/claude/syn3a-whole-cell-simulator-REjHC`. PAT in `/tmp/.gh_tkn`. Revoke when done.
