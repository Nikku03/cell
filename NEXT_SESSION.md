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

## Session-14 top priority — populate the feature cache

### 0. Populate feature cache with ESM-2 (650M) embeddings for all 452 Syn3A CDS

**New top-priority item, added in Session 13.** The feature-cache infrastructure built this session (`cell_sim/features/`) is ready to consume parquet files but has nothing cached yet. The highest-leverage populater is ESM-2 650M embeddings, one 1280-dim vector per CDS:

- Separate session; requires `torch` + `transformers` install. NOT attempted this session — the plumbing session's hard limits forbid pulling in heavy ML deps.
- Subclass `BatchedFeatureExtractor`, name = `"esm2_650M"`, version = `"0.1.0"`, feature_cols = `[f"esm2_650M_dim_{i}" for i in range(1280)]`.
- Inputs: DataFrame with `locus_tag` + `sequence` columns sourced from `cell_sim/layer0_genome/genome.py` (protein sequences already parsed from CP016816.2).
- Wall-time estimate: 452 CDS at batch_size=32 on a single L4 GPU → ~2 minutes.
- Output parquet: ~2.2 MB. Cache hit: instant (200 ms join with the sweep).
- Once cached, a follow-up detector session can compare essentials vs nonessentials in ESM-2 embedding space (e.g. cosine similarity to known-essential ortholog embeddings, or a trivial linear classifier on the balanced-40 panel) without re-running the LM.

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
