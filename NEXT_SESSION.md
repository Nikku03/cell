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

### 0a. Fix AlphaFold-DB coverage (taxid 2144189 not in UniProt)

JCVI-Syn3A is a synthetic organism and UniProt has not imported its proteome — four query routes all returned zero rows in the Session-14 run. AFDB keys on UniProt accessions, so without a mapping there's nothing to fetch. Fix:

1. Download the Luthey-Schulten 4DWCM supplement — the authors had to solve the same mapping problem for their own work. Likely candidate path in that repo: a CSV or JSON cross-referencing Syn3A NCBI protein accessions (`AVX*`) to UniProt accessions from the parent *M. mycoides capri* proteome.
2. Alternative: manually construct the mapping via NCBI BioProject `PRJNA331011` cross-references + UniProt's *M. mycoides capri* entries.
3. Persist the mapping as `memory_bank/data/syn3a_ncbi_to_uniprot.csv` + a pointer fact in `memory_bank/facts/structural/`.
4. Re-run `notebooks/populate_tier1_cache.ipynb` cell 6 — it already calls a UniProt stream and will populate real pLDDT / SS / Rg features once the map lands. Expected coverage: ≥ 80% of the 455 CDS if the parent-proteome mapping is reused.
5. `git add -f` the refilled `alphafold_db.parquet` + updated `manifest.json` + update the `populated_yet_content_status` in the AlphaFold fact.

### 0b. Fix MACE-OFF SMILES map

Syn3A's SBML (`Syn3A_updated.xml`) encodes metabolites as BiGG-style species IDs (`M_atp_c`, `M_h2o_c`, ...). MACE-OFF needs real SMILES. Notebook cell 7 carries a `HAVE_CURATED_SMILES_MAP=False` guard that skips the backend entirely today. Fix:

1. Build `memory_bank/data/syn3a_species_smiles.csv` mapping every species ID in the SBML to a canonical SMILES. Primary source: KEGG COMPOUND (free bulk download + SBML annotation has `bqbiol:is` cross-refs to `kegg.compound:Cxxxxx` that can be followed). Fallback for metabolites KEGG doesn't cover: ChEBI.
2. Register the CSV via a pointer fact in `memory_bank/facts/structural/`.
3. Edit notebook cell 7 to load `SUBSTRATE_SMILES_MAP = dict(zip(csv["species_id"], csv["smiles"]))` and flip `HAVE_CURATED_SMILES_MAP = True`.
4. Re-run cell 7. Expected ≥ 60% of the 356 SBML reactions will have SMILES for every substrate; MACE-OFF writes real BDE-derived k_cat aggregates for those enzymes.

### 0c. Baseline Tier-1 MCC (XGBoost on cached features)  — **CLOSED in Session 15 (negative result)**

Landed as `cell_sim/layer6_essentiality/tier1_xgb_detector.py` + `scripts/run_tier1_sweep.py` + `memory_bank/facts/measured/mcc_against_breuer_v11_tier1_xgb.json`. Honest finding: **no Tier-1 XGBoost slice beats the v10b rule baseline on either set.** The FULL-455 and BALANCED-40 sweep outputs are persisted in `outputs/tier1_sweep_v11.json`. Best XGB aggregated MCC: 0.241 (esm2_only) on FULL vs v10b 0.364; 0.603 (esm2_plus_priors) on BALANCED vs v10b 0.800. Diagnosis: 5.3:1 class imbalance + 455 labels vs 1280-dim ESM-2 + priors already cover the easy signal -> ML overfits or collapses. Re-visit this fact **after** 0a and 0b land — the hypothesis is that structural + kinetic features are what address the 15 translation-machinery FNs that ESM-2 alone cannot. The null hypothesis (learned features dominate rules on this benchmark) is now falsified; a positive re-run would need to beat 0.364 FULL / 0.800 BALANCED to be worth publishing.

### 1. Fix iMB155 pathway incompleteness (the real bottleneck)

**Three specific FPs diagnosed in v9:**

| Gene | Breuer call | Simulator issue | Fix |
|---|---|---|---|
| **JCVISYN3A_0034** | Nonessential | iMB155 makes `M_chsterol_c` only via 0034's transport reactions. | Either: (a) add `chsterol_scavenge` or `chsterol_medium_passive_import` pseudo-reaction, (b) mark cholesterol as `metabolite_infinite` in state (sourced from medium), OR (c) remove cholesterol from the biomass equation entirely — Syn3A doesn't synthesise cholesterol. |
| **JCVISYN3A_0732 (deoC)** | Nonessential | iMB155's DRPA is the sole acetaldehyde source. | Either add alternate acetaldehyde sources (PDH byproduct is one) or mark acald as conditionally infinite. |
| **JCVISYN3A_0228 (lpdA)** | Nonessential | iMB155 has only PDH_E3 producing lipoylated PdhC. | Add `lipA` / `lipB` / `lplA` alternate lipoylation rules. These genes exist in Syn3A (check `syn3a_gene_table`) but may lack catalysis rules in the 2022-era `kinetic_params.xlsx`. |

Each fix is a SBML/kinetic-params annotation, not new simulator code. Expected: drops all 3 FPs → MCC ≈ 0.19–0.22 ± 0.03 on the same panel. Honest prediction (not a promise).

**Implementation sketch:**
- Add a new fact `facts/parameters/imb155_pathway_patches.json` listing each added reaction with its rationale and source citation.
- Add a helper `cell_sim/layer3_reactions/imb155_patches.py` that builds the patched rules and folds them into `_extra_rules`.
- Re-run v9 sweep → record as v10.

### 2. Ingest Thornburg 2026's refined `kinetic_params.xlsx` (4DWCM repo)

Their 85 KB version vs our staged 59 KB 2022 version has refined k_cat / Km for ~30 reactions. Pull via:

```bash
curl -sS -o cell_sim/data/Minimal_Cell_ComplexFormation/input_data/kinetic_params_4dwcm.xlsx \
    https://raw.githubusercontent.com/Luthey-Schulten-Lab/Minimal_Cell_4DWCM/main/input_data/kinetic_params.xlsx
```

Diff sheet-by-sheet against our staged file. For each changed parameter, promote to a proper `facts/parameters/*.json` with Thornburg 2026 citation. Sharpens metabolic timing without changing detector behaviour.

### 3. Explicit ribosome complex (addresses the 15 FN wall)

Still the biggest remaining MCC lever. Currently blocked by needing to compose the 50S subunits from `LargeSubunit.xlsx` (4DWCM repo) and the 30S subunits from `complex_formation.xlsx`. Add a complex-formation rule that requires all ~55 subunits to be present; KO of any subunit halts new ribosome assembly.

Estimate: +2 sessions of code, +1 session of tuning. Would catch 5–7 of 15 ribosomal FNs → MCC toward 0.25–0.30.

### 4. Multi-seed replicates as the default

Session 10's Block B found single-seed MCC numbers are unreliable. The `--panel-seed` flag now exists. Sweep-level default should be `--seeds 42 1 2 3 4` with aggregated fact output. One-line change to `run_sweep_parallel.py`.

## The honest ceiling bound

Path A tried (v7): falsified. Higher scale tried (v8): no help. All detector variants tried (v0–v9): plateau at 0.125.

**Reachable with simulator-biology fixes (Sessions 12–14):**
- #1 iMB155 patches: MCC ≈ 0.19–0.22
- #1 + #3 ribosome complex: MCC ≈ 0.30–0.35

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
