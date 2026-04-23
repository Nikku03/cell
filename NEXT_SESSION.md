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

## Session-15 top priority — run the Tier-1 populate notebook, then measure

### 0. Run `notebooks/populate_tier1_cache.ipynb` on Colab Pro

**New top-priority item, refreshed in Session 14.** Session 14 landed all three extractor subclasses (`ESM2Extractor`, `AlphaFoldExtractor`, `MaceOffExtractor`), their sandbox-safe tests (17), and the populate notebook itself — but the cache is still empty. Session 15 is the user-side execution + verification pass:

1. Open `notebooks/populate_tier1_cache.ipynb` on Colab Pro (A100 preferred, T4 fallback).
2. Run cells 1-8 to install deps, clone the repo at the current HEAD, load Syn3A CDS sequences, and execute the three extractors. Expected wall time: ~20-40 minutes total (ESM-2 ~2 min, AlphaFold ~15-30 min network-bound, MACE-OFF ~5 min).
3. Pick an `OUTPUT_MODE` in cell 9:
   - `drive` — Google Drive mount
   - `download` — four browser download prompts
   - `github_pat` — fully automated `git add -f` + commit + push via `$GITHUB_PAT`
4. Verify cell 10's summary block: three parquets tracked in `manifest.json`, SHAs match, row counts sane. Paste the summary back for review.
5. Once the parquets are on `origin`, open a follow-up PR that flips each of the three `populated_yet: false` flags in `memory_bank/facts/structural/{esm2,alphafold,mace_off}_extractor.json` and adds the real `sha256` fields.
6. Only after the cache is populated, measure the baseline ensemble MCC with an XGBoost classifier on the cached feature join — that's the first honest Tier-1 MCC number. Record as `mcc_against_breuer_v11_tier1_xgb.json`.

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
