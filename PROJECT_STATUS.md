# PROJECT_STATUS — SYN3A Whole-Cell Simulator

_This file is the authoritative snapshot of project state. Updated at the end of every session, read at the start. If out of date, reconcile before doing work._

## The Goal (unchanging)

Build a biologically accurate, computationally cheap Syn3A whole-cell simulator that predicts **time-dependent gene essentiality** with **Matthews correlation coefficient > 0.59** against Breuer 2019 experimental labels, for all 452 Syn3A genes.

## Layer Progress

| Layer | Name | Phase | Status |
|-------|------|-------|--------|
| 0 | Genome | complete (A-E) | Genome API + 12 validation tests passing; facts cited and stamped. |
| 1 | Transcription machinery | partial | existing `cell_sim/` code + kinetic data covers it at Thornburg-lumped level; no memory-bank citation trail yet. |
| 2 | Translation machinery | partial | same as Layer 1; ribosome is a pool, not a tracked complex. |
| 3 | Protein folding + complex assembly | partial | complex_formation.xlsx loaded by existing rules; 24 complexes defined with stoichiometry. |
| 4 | Metabolism | partial | Syn3A_updated.xml + kinetic_params.xlsx loaded by existing rules; 6 transporter k_cats patched without citation yet. |
| 5 | Biomass + division | not started | no biomass accumulation / division logic anywhere. |
| 6 | Essentiality analysis | RealSimulator wired (Python + Rust), parallel sweep, 10 MCC measurements (v0-v9) + replicates + v9 robustness. **Best honest MCC = 0.123 ± 0.020** (v9 RedundancyAwareDetector across 2 different 40-gene panels, 10 total runs via Colab). Panel 42 reproduces exactly (0.125, std=0.000); panel 400 introduces boundary-gene variance (0.119 ± 0.041). Detector-design space exhausted; iMB155 pathway-completeness is the bottleneck. |

Phase codes (for the layers we gate): A = Literature survey, B = Design, C = Implementation, D = Validation, E = Layer report.

## Memory Bank

- Facts: **26**
  - structural (9): doubling time, chromosome length, gene count, gene table, oriC, Breuer 2019 labels, RNAP count per cell, ribosome count at birth, chromosome bead model.
  - parameters (2): active RNAP fraction, mRNA half-life mean.
  - measured (12): `mcc_against_breuer_v0..v9` + `mcc_replicates_summary` + `mcc_v9_robustness`.
  - resolved uncertainty (3): `syn3a_gene_count_dispute`, `syn3a_chromosome_length_pending`, `syn3a_gene_count_thornburg2026_discrepancy`.
- Sources: **11** (Thornburg 2022 + 2026 Cell, Hutchison 2016, Breuer 2019, GenBank CP016816, Luthey-Schulten ComplexFormation + 4DWCM repos, Fu 2026 JPC-B, Gilbert 2023 Frontiers, Bianchi 2022 JPC-B, Pezeshkian 2024 Nat Commun).
- Sources: **5** (`thornburg_2022_cell`, `hutchison_2016_science`, `breuer_2019_elife`, `genbank_cp016816`, `luthey_schulten_minimal_cell_complex_formation_repo`).
- Invariant checker: `OK`.
- Data files (tracked): `memory_bank/data/syn3a_gene_table.csv` (496 rows), `memory_bank/data/syn3a_essentiality_breuer2019.csv` (455 rows).
- Data files (local only, gitignored): 5 Luthey-Schulten input files under `cell_sim/data/Minimal_Cell_ComplexFormation/input_data/` — SHAs recorded in `memory_bank/data/STAGING.md`.

## Validation Targets (reference)

- Layer 0-3: measured steady-state protein counts (Thornburg 2022) within 2x for 90% of genes.
- Layer 4: central-carbon metabolite concentrations within 2x.
- Layer 5: biomass doubling in 2 +/- 0.5 h.
- Layer 6: **MCC > 0.59** vs Breuer 2019. Measurements (all at scale=0.05 + t_end=0.5 s unless noted): v0=0.333 (n=4), v1=0.160 (n=40 ShortWindow+cal), v2=0.229 (n=20 scale=0.10), v3=0.229 (n=20 scale=0.25+rust), v4=0.229 (n=20 +non-metabolic pools), v5=0.125 (n=40 PerRule), **v6=0.125 (n=40 ensemble-per-rule-with-pool-confirm) / 0.160 (n=40 ensemble-AND-+-rule-necessity-only)**. All three detector families (pool-deviation, per-rule event counts, ensembles of the two with or without rule-necessity filtering) have now hit the same short-window ceiling. Path to 0.59 requires Path A (longer bio-time runs, committed compute) — detector-side optimisation at 0.5 s bio-time is exhausted.

## Performance Targets

- ≥ 10x real-time on one CPU core — **not met**: current is ~0.06× realtime at scale=0.05 in pure Python, ~0.13× with Rust.
- ≥ 100x real-time with Rust hot paths — **not met** (see above).
- No GPU required for normal operation — **met**.
- Practical throughput: **1.9 s/gene effective wall** at scale=0.05 with Rust + 4-worker parallel (v4 config). 458-gene sweep at that config ≈ 15 min wall.

## Session Log

### Session 14 (populate pass) — 2026-04-23 — Colab GPU run of populate_tier1_cache.ipynb
- **Scope**: user-side run of the Session-14 notebook on a Colab A100-class GPU (NVIDIA RTX PRO 6000 Blackwell, 95 GB VRAM), then `git add -f` + push of the three parquets. Commit `ebbfdff` on `claude/syn3a-whole-cell-simulator-REjHC`. No sandbox-side MCC measurement.
- **ESM-2 (650M)** landed fully populated: 455 CDS × 1280 dims, zero NaN, `sha256=0d5726…`, ~18 s of inference. This is the real deliverable.
- **AlphaFold-DB** landed empty by design. JCVI-Syn3A (taxid `2144189`, "synthetic bacterium JCVI-Syn3A") is not indexed in UniProt — the populate notebook verified this by trying four different query routes (direct search, `xref:embl-CP016816`, proteome-name match, `organism_id:2144189` stream) and getting zero rows each time. The parquet has 455 rows of NaN + `af_has_structure=0.0`; `sha256=6ebc5c…`. Session 15 TODO: wire in a curated NCBI-accession → UniProt map (candidate: Luthey-Schulten 4DWCM supplement).
- **MACE-OFF** landed empty by design. Syn3A's SBML encodes metabolites as BiGG-style species IDs (`M_atp_c`, etc.), not SMILES. Notebook cell 7 now carries a `HAVE_CURATED_SMILES_MAP=False` guard that skips the backend entirely rather than pass species IDs as SMILES. Parquet: 455 rows of NaN + `mace_has_estimate=0.0`, `sha256=9db2b8…`. Session 15 TODO: wire in a curated species → SMILES map (KEGG or ChEBI) and flip the guard.
- **Manifest integrity**: sandbox-side FeatureRegistry.load SHA-verified all three sources; `join_features` on a 3-locus query returned the expected (3, 1296) DataFrame with correct NaN handling for an unknown locus.
- **Notebook stability work**: commits `a0252ea`, `9626eef`, `214d04e`, `6540f49` all landed mid-run to fix (respectively) the mace-torch × e3nn pin conflict, upstream-data staging, UniProt REST fallback, and finally the error-free end-to-end rewrite that the user triggered after hitting the `GITHUB_PAT` RuntimeError and the UniProt-empty-result confusion. `172/172` tests still pass.
- **New memory-bank entries**: 3 structural facts flipped `populated_yet: true` with real SHAs + content-status callouts + Session-15 TODOs; 1 new measured fact `session_14_populate.json` records the commit, parquet SHAs, FeatureRegistry contract check, and downstream caveats.

### Session 14 — 2026-04-23 — Tier-1 extractor classes + Colab populate notebook (no MCC)
- **Scope**: pure plumbing on the sandbox side. Subclassed the Session-13 `BatchedFeatureExtractor` three times and shipped a Colab notebook that populates the cache on GPU. No pretrained model was executed in the sandbox. No detector in `cell_sim/layer6_essentiality/` was modified.
- **New module `cell_sim/features/extractors/`** (~450 lines impl + ~300 lines tests):
  - `esm2_extractor.py` — `ESM2Extractor` for `facebook/esm2_t33_650M_UR50D`. 1280 `esm2_650M_dim_{i}` float columns per CDS. Lazy-imports torch + transformers inside `_ensure_loaded`; empty input short-circuits before any heavy import (verified by `test_empty_input_returns_empty_frame`).
  - `alphafold_extractor.py` — `AlphaFoldExtractor` fetches `AF-{UNIPROT}-F1-model_v4.pdb` from EBI-EMBL, parses with biopython to emit 9 descriptors (pLDDT mean/std, disorder fraction, helix/sheet/coil fractions via torsion bins, length, Rg, has_structure). Lazy biopython import; 3× retry + backoff on transient network failures; 404 → no-structure NaN row.
  - `mace_off_extractor.py` — `MaceOffExtractor` wraps the existing `cell_sim.layer1_atomic.engine.MACEBackend` without re-implementing any BDE / Eyring / Hammond math. Aggregates per-substrate k_cat estimates into 7 per-locus summary stats so the output matches the `FeatureRegistry.locus_tag` index contract.
- **Colab notebook `notebooks/populate_tier1_cache.ipynb`** (10 cells, ~420 lines of JSON):
  - Cells install GPU-side deps, clone the repo, load CDS sequences from the GenBank file, run each extractor, refresh the manifest, and offer three output paths (Drive / direct download / GitHub PAT push — user picks via `OUTPUT_MODE` at top of cell 9). The PAT path uses `git add -f` on the parquets so the gitignore rule stays general.
- **Tests**: 17 extractor tests (6 + 6 + 5) + 8 notebook-validation tests = **25 new, all pass in < 1 s**. Notebook tests cover: valid JSON, correct cell count, code/markdown ordering, no saved outputs at commit time, references to the three extractor class names, three output modes exposed, and no embedded model weights.
- **Memory bank**: 3 new structural facts (`esm2_extractor`, `alphafold_extractor`, `mace_off_extractor`). Each declares the module path, feature columns, expected parquet path / size, `populated_yet: false`, and a caveat that the cache is empty until the notebook runs. `confidence: measured` (verifiable by reading the module + running the tests). `source: internal_infrastructure` (registered in Session 13).
- **Dependencies**: no new line in `cell_sim/requirements.txt`. Added a 6-line comment there pointing at the notebook for the GPU-side stack. Explicitly did NOT add torch / transformers / mace-torch / e3nn / biopython to the repo requirements — those live in the Colab install cell only.
- **Session totals**: 147 (pre-session) + 17 (extractor) + 8 (notebook) = **172 tests passing**. 33 facts / 12 sources / invariant checker OK.
- **Explicit non-claims**: no MCC measurement, no detector change. The notebook is ready for user execution; parquets will be pushed in a separate follow-up commit once the notebook runs.

### Session 13 — 2026-04-23 — Feature-cache infrastructure (plumbing, no MCC)
- **Scope**: pure infrastructure for per-gene pretrained-model feature caching. No pretrained models were downloaded or run. No MCC measurement made. No detector in `cell_sim/layer6_essentiality/` was modified.
- **New module `cell_sim/features/`** (~500 lines of impl, 350 lines of tests):
  - `feature_registry.py` — `FeatureRegistry` / `FeatureSource`: declare parquet-backed feature tables, check cached state, load with SHA-256 validation, join multiple sources as a single DataFrame indexed by `locus_tag`. `join_features` is pure — missing sources become NaN columns; never triggers an extraction.
  - `cache_manifest.py` — `CachedFeatureManifest`: SHA-256 authority for the cache dir. `add()` computes + records, `verify()` recomputes + compares, `save()` writes atomically via sibling `.tmp`. Handles missing / malformed manifest files gracefully (empty manifest, clear `ValueError` respectively).
  - `batched_inference.py` — abstract `BatchedFeatureExtractor` base class + `BatchedInferenceConfig` dataclass. Zero model imports at module load; subclasses (future sessions) import `torch`/`transformers`/etc. inside their own `extract()` method, mirroring the `MACEBackend._ensure_loaded` pattern in `cell_sim/layer1_atomic/engine.py`. Default `write_cache()` writes the parquet + updates the manifest with the producer's version tag.
  - `cache/manifest.json` committed empty (`{"sources": {}}`); `cache/.gitkeep` committed; all other cache-dir contents gitignored.
- **Tests**: 24 new (11 manifest + 13 registry), all pass in ~0.5s. Baseline was 123 passing; post-session total **147 passing**. One pre-existing collection error on `test_end_to_end.py` (matplotlib missing) is unchanged by this work.
- **Invariant checker**: 30 facts / 12 sources / OK. Added a new `internal_infrastructure` source to cover the new `feature_cache_infrastructure` structural fact; without it the brief's suggested `source: "internal_infrastructure"` value would have failed the sources-must-be-registered check.
- **Dependencies**: added `pyarrow>=14.0` to `cell_sim/requirements.txt` (one line, comment says why). `pandas.to_parquet` / `read_parquet` delegate to pyarrow; this was the only missing piece needed to run the new tests.
- **Explicit non-claims**: this session produces no improvement against Breuer 2019. MCC remains at the v10b-session number. No ESM-2, AlphaFold, or MACE-OFF weights were touched.

### Session 12 — 2026-04-22 — v9 Colab reproduction + panel-seed robustness
- Added `notebooks/colab_v9_run.ipynb` — reproduces v9 (5 sim seeds × panel=42) and runs a panel-seed robustness block (up to 5 panels × 3 seeds). GitHub push-protection correctly caught a literal PAT I left in notebook markdown; amended commit without the token.
- **Colab run (L4, 8 vCPU, partial — user interrupted after panel 400)**: 10 total runs across 2 panels.
  - **Panel 42 (7 runs)**: MCC = 0.125 ± 0.000 — reproduces sandbox v9 exactly. 5 TP / 3 FP / 17 TN / 15 FN across every run.
  - **Panel 400 (3 runs)**: MCC = 0.119 ± 0.041 — different gene draw exposes boundary-gene variance. Mean TP/FP/TN/FN = 2.3 / 1.0 / 19.0 / 17.7 (non-integer means confirm per-seed variation).
  - **Overall (10 runs, 2 panels): MCC = 0.123 ± 0.020** — the honest project-wide v9 number.
- **New scientific finding**: v9 is deterministic at a fixed panel but introduces real variance across panels when gene samples include catalytic genes whose product-collapse ratios sit near `drop_threshold=0.30`. Panel 400's seed-to-seed MCC std of 0.041 came from one or two boundary genes flipping call direction. This nuances the Session-11 "seed-invariant" claim.
- Recorded as `mcc_v9_robustness` measured fact. PROJECT_STATUS and REPORT updated with the 0.123 ± 0.020 number.
- Commits: `a32ccf0` (notebook, rewritten after secret-scan catch → `916b548`) + `6561e72` (Colab results). Pushed.

### Session 11 — 2026-04-21 — Rule-alternates detector + metabolite sink + v9
- **Deliverable 1 — `RedundancyAwareDetector`**: new detector that trips only when a silenced gene's products actually lose production capacity (summing events × stoichiometry across ALL catalysing rules, not just the gene's own). Addresses the v5/v6 FP mechanism structurally. Code: `cell_sim/layer6_essentiality/redundancy_aware_detector.py` (150 lines) + helpers `build_metabolite_producers` / `build_rule_products` in `gene_rule_map.py`. 9 new unit tests (total 58 passing).
- **Deliverable 2 — Metabolite sink**: first-order drain rules that fire above `tolerance × initial_count`. Addresses the v7 transporter-KO pool-blowup mechanism. Code: `cell_sim/layer6_essentiality/metabolite_sink.py` + `RealSimulatorConfig.enable_metabolite_sinks` flag. Off by default.
- **v9 measurement** — 5-seed replicates on the same balanced n=40 panel at scale=0.05 + t_end=0.5 s:
  - **MCC = 0.125 ± 0.000** — **identical across all 5 seeds**. Structural (production-collapse) signal is deterministic at n=40.
  - Same 5 TPs as v5 (pgi, tpiA, plsX-area, 0813, 0729) but with quantitative production-collapse evidence (e.g. pgi's F6P goes from 7894 → 8 events).
  - Same 3 FPs as v5/v6: **0034** (cholesterol, no alternate source in iMB155), **deoC / 0732** (acetaldehyde, unique DRPA source in iMB155), **lpdA / 0228** (PdhC lipoylation, lipA/lipB alternates missing from iMB155).
  - Same 15 FNs — all non-catalytic (ribosomal, tRNA, translation factors). Architecturally uncatchable without explicit complex-assembly dynamics.
- **Scientific finding**: the detector-design space is confirmed exhausted. The remaining gap to MCC > 0.59 localises to **iMB155 reconstruction incompleteness** (missing alternate pathways for cholesterol / acetaldehyde / lipoylation). Closing it requires simulator-biology work (add missing reactions to iMB155 or substitute an updated SBML), not detector changes.
- Best-ever balanced-panel MCC updated from 0.112 ± 0.029 (v5/v6a) to **0.125 ± 0.000** (v9). Predicted 0.22 ± 0.04 in the Session 10 planning doc was too optimistic; the detector worked but the simulator limits what can be caught.

### Session 10 — 2026-04-21 — Colab multi-seed replicates + v8 higher-scale + Thornburg 2026 integration
- Wrote `scripts/run_colab_bc.py` and `notebooks/colab_bc_sweep.ipynb` to offload Block B (5-seed replicates × 5 detector configs on balanced n=40) and Block C (scale=0.5, t_end=1.0, n=40) to a Colab L4 VM. Gene panel held fixed via new `--panel-seed` flag; only simulator RNG varies across replicates.
- Fixed over-broad root `.gitignore` that was hiding `memory_bank/data/*.csv` from commits; CSVs now tracked.
- Ran the notebook on Colab L4: 25 replicate sweeps + 1 higher-scale sweep in ~90 min wall.
- **v8 (scale=0.5, t_end=1.0, ensemble pool_confirm)**: MCC = **0.060** (TP=5, FP=4, TN=16, FN=15). Higher scale added one more FP (4 vs 3) without recovering any essentials. Confirms Session-9 diagnosis at yet another config.
- **Replicates summary (`mcc_replicates_summary`)** with gene panel fixed at seed=42, 5 simulator seeds {42,1,2,3,4}:
  - `v5_per_rule` / `v6a_ensemble_pool_confirm`: MCC = **0.112 ± 0.029** (tightest, structural signal)
  - `v1_shortwindow_cal` / `v4_shortwindow_nonmetabolic` / `v6b_ensemble_and_unique`: MCC = **0.064 ± 0.088** (wide variance)
  - Previously-reported single-seed 0.160 for v1 / v6b was cherry-picked: the across-seed mean is 0.064.
- **Key measurement-hygiene insight**: per-rule detectors are much lower variance than pool-based ones (0.029 vs 0.088). That's because per-rule is a structural signal, pool-based rides stochastic metabolite fluctuations.
- In parallel: integrated Thornburg et al. 2026 Cell paper via WebSearch + deep-research subagent. 5 new sources registered (4DWCM repo, Zenodo 15579158, Fu 2026 JPC-B, Gilbert 2023 Frontiers chromosome, Bianchi 2022 JPC-B, Pezeshkian 2024 Nat Commun). 5 new parameter/structural facts (RNAP count 187, active RNAP fraction 0.34, ribosome count at birth ~500, mRNA half-life mean 3.63 min, chromosome 10 bp/bead model). `syn3a_doubling_time` updated 7200 s → 6300 s (105 min measured). Gene-count discrepancy resolved: Thornburg 2026 uses 452 CDS + 41 RNA = 493, our parse gave 458 + 38 = 496 — does not affect Layer 6 because Breuer labels are the intersection.
- Confirmed via subagent: **Thornburg 2026 does NOT report knockout MCC**. Our Layer 6 work remains unique territory.
- Commits: `8f19f5d` (notebook), `5bc078f` (gitignore fix), `0ff69a1` + `8a38488` (Thornburg 2026), `f6bb7c2` (Colab sweep results). Pushed to origin.

### Session 9 — 2026-04-21 — Path A attempt; longer-window falsified
- Ran the go/no-go reference panel at t_end=5.0 s (ensemble per_rule_with_pool_confirm, min_pool_dev=0.05). Result: pool deviations DO strengthen for the 2 catchable TPs (pgi 0.50→0.59, ptsG caught at 0.17), n=4 MCC=0.577. Decision: commit to n=20 balanced at the same config.
- **v7 measurement**: n=20 balanced at t_end=5.0 s, ensemble per_rule_with_pool_confirm min_pool_dev=0.10 → MCC = **0.000** (TP=3, FP=3, TN=7, FN=7). Longer bio-time *does not help* — it lets FP pool deviations grow too (0034 transporter-KO max_pool_dev went from ~1.0 at 0.5 s to 13.3 at 5.0 s, because the upstream metabolite accumulates unboundedly in the simulator without the biological consumers / diffusion equilibrium that would cap it in a real cell).
- **Path A falsified** at scale=0.05. Session 8 predicted 0.2–0.35 honestly; measured 0.000. Not running the full 458-gene sweep at t_end=5.0 s — no justification for the 3-hour compute commitment.
- Diagnosis for MCC > 0.59: the simulator-biology gap is the real bottleneck, not any detector-side variable. Pathway redundancy, proper translation dynamics, or per-pathway dilution modelling is required in the simulator before detector changes can close the remaining distance.
- 1 commit: TBD. Pushed to origin.

### Session 8 — 2026-04-21 — Ensemble detector + rule-necessity filter + path-A diagnosis
- **Deliverable**: `cell_sim/layer6_essentiality/ensemble_detector.py` with three policies (`AND`, `OR_HIGH_CONFIDENCE`, `PER_RULE_WITH_POOL_CONFIRM`). Composes `PerRuleDetector` + `ShortWindowDetector`.
- Added `unique_rules_per_gene` / `invert_to_rule_catalysers` helpers in `gene_rule_map.py` for rule-necessity weighting.
- Wired `--detector ensemble`, `--ensemble-policy`, `--min-confidence`, `--min-pool-dev`, `--rule-necessity-only` into `run_sweep_parallel.py`. Per-rule/ensemble workers receive the gene-to-rules map via `Pool.initargs`.
- 7 new tests (`test_layer6_ensemble_detector.py`): rule-necessity helpers (inverse map, unique filter), ensemble AND policy (fires + refuses), ensemble per_rule_with_pool_confirm (fires + abstains on flat pools + abstains when PerRule abstains). Total: **49 passing**.
- **v6a measurement**: ensemble per_rule_with_pool_confirm + min_pool_dev=0.02, n=40 balanced → MCC = 0.125. Identical to v5 PerRule alone; 2% pool floor is trivially exceeded by stochastic noise at scale=0.05.
- **v6b measurement**: ensemble AND + rule-necessity-only, n=40 balanced → MCC = 0.160. Collapses to pgi-only (same as v1/v4) because ShortWindow trips only on pgi.
- **Diagnostic finding (the real Session-8 result)**: FP catalytic KOs (0034, 0228, 0732) and TP catalytic KOs (0445, 0727, 0419, 0813, 0729) show max_pool_dev in the same 0.167–1.00 range. No short-window pool-confirm gate can separate them. The v5 FP mechanism is not fixable by detector composition in the ≤0.5 s bio-time regime — the simulator lacks the biological pathway redundancy that would let Breuer's nonessentials actually compensate for the KO.
- v6 fact recorded both variants honestly; the path to MCC > 0.59 is now unambiguously Path A (longer bio-time runs), not further detector composition.

### Session 7 — 2026-04-21 — Per-rule event-count detection + Session-6 reconciliation
- **Deliverable 1**: reconciled `PROJECT_STATUS.md` and `NEXT_SESSION.md` with the actual Session-6 state (both files had drifted to Session-4 content with duplicate NEXT_SESSION headings). One commit: `010255c`.
- **Deliverable 2**: per-rule event-count detection shipped.
  - `cell_sim/layer6_essentiality/gene_rule_map.py` — extracts `{locus_tag: {rule_name, ...}}` from rule objects' `compiled_spec.enzyme_loci`.
  - `cell_sim/layer6_essentiality/per_rule_detector.py` — `PerRuleDetector(wt, gene_to_rules, min_wt_events)`. Trips `CATALYSIS_SILENCED` iff every rule in a gene's set has ≥`min_wt_events` in WT and 0 in KO. Safe refusals on WT-under-threshold and partial-silence cases.
  - `Sample` dataclass now carries `event_counts_by_rule: dict[str, int] | None`; `RealSimulator._snapshot` populates it via a single `Counter` pass over `state.events`.
  - `FailureMode.CATALYSIS_SILENCED` added.
  - `scripts/run_sweep_parallel.py --detector {short-window|per-rule} --min-wt-events N`. Per-rule detector builds the gene-to-rules map in the main process and ships it to workers via `initargs`. Calibration is skipped for per-rule (no thresholds to tune).
  - 9 new unit tests in `test_layer6_per_rule_detector.py`; total 42 passing.
- **v5 measurement**: n=40 balanced → **MCC=0.125** (TP=5, FP=3, TN=17, FN=15). Below v4 (0.229). The 3 FPs are Breuer-nonessential catalytic genes (JCVISYN3A_0034 transport system, lpdA/0228 PDH_E3, deoC/0732 DRPA) whose rules the simulator runs but Breuer labels as nonessential due to pathway redundancy the simulator doesn't model. 15 FN are non-catalytic essentials (ribosomal, tRNA, replication) which have zero rules in `gene_to_rules` and the detector correctly refuses to call. Side-result: on the 4-gene reference panel MCC=0.577, but n=4 is sample-size noise.
- Infrastructure works as designed. Mismatch with Breuer labels is real biology, not a bug.
- Gene-to-rules map covers 114 / 458 CDS (~25%), avg 3.7 rules/gene, max 19 rules/gene.
- Sweep effective wall: 1.7 s/gene (Rust + 4-worker). 67.6 s total for n=40.

### Session 6 — 2026-04-21 — Rust hot path + non-metabolic pool signals + diagnostic ceiling
- Built `cell_sim_rust` wheel from source via `maturin build --release`. Installed and wired into `RealSimulator` via `RealSimulatorConfig.use_rust_backend` + `--use-rust` flag on `run_sweep_parallel.py`. ~2× speedup at scale=0.05.
- Added 6 non-metabolic pool signals to `RealSimulator._snapshot`: `TOTAL_COMPLEXES`, `FOLDED_PROTEINS`, `UNFOLDED_PROTEINS`, `FOLDED_FRACTION`, `BOUND_PROTEINS`, `TOTAL_EVENTS`. All plumbed into `SHORT_WINDOW_POOLS`.
- v3: scale=0.25 + Rust + calibration → MCC = 0.229 on n=20 balanced.
- v4: scale=0.05 + Rust + full expanded pool set + calibration → MCC = 0.229 on n=20 balanced. Effective wall 1.9 s/gene (~7× speedup vs v0 baseline).
- **Diagnosis confirmed**: MCC is invariant across scale {0.05, 0.10, 0.25}, t_end {0.5, 1.0, 2.0}, pool set {12, 17, 18 pools}, threshold {0.03–0.10}, and sample size {4, 20, 40}. Only pgi (central glycolysis) trips. Ceiling is architectural, not tuning.
- Scale=0.5 sweep attempted but timed out at 9 min wall in-session.
- 1 commit: `dbc1e07`. Pushed to `origin/claude/syn3a-whole-cell-simulator-REjHC`.

### Session 5 — 2026-04-21 — parallel sweep + noise-floor calibration + diagnostic MCC measurements
- Built `scripts/run_sweep_parallel.py` using `multiprocessing.Pool` with `--workers N`. 4-worker fan-out gives ~4× speedup (process-safe; FastEventSimulator is not thread-safe).
- Added `--calibrate K` + `--safety-factor S` flags to `run_full_sweep_real.py`. Calibration runs K non-essential KOs, computes per-pool max|dev| noise floor, sets per-pool thresholds = floor × safety_factor (fallback to `--threshold`).
- Extended `ShortWindowDetector.deviation_threshold` to accept `dict[str, float]` for per-pool thresholds.
- v1: n=40 balanced, scale=0.05, cal=10, sf=2.5 → MCC = 0.160. Specificity=1.0; recall=0.05; 1 TP (pgi) / 0 FP / 20 TN / 19 FN.
- v2: n=20 balanced, scale=0.10, t_end=1.0, cal=5, sf=2.5 → MCC = 0.229.
- Attempted scale=0.5 but timed out at 9 min wall.
- Also ran diagnostic: t_end=2.0 at scale=0.05 gives MCC=0.333 (identical to v0; longer window doesn't help at small scale).
- 2 commits: `7216586`, `c0e34e3`. Pushed.

### Session 4 — 2026-04-21 — Layer 6 real-simulator wiring + first MCC
- Wrote `cell_sim/layer6_essentiality/real_simulator.py` wrapping `FastEventSimulator + populate_real_syn3a` stack behind the `Simulator` Protocol. Heavy setup cached across knockouts.
- Wrote `cell_sim/layer6_essentiality/short_window_detector.py` — bidirectional `|ko/wt - 1|` deviation detector with two-consecutive-sample confirmation.
- Wrote `scripts/run_full_sweep_real.py` single-process orchestrator with `--reference-panel`, `--max-genes N --balanced`, and `--all` modes.
- 8 new tests in `test_layer6_short_window_detector.py` (incl. RealSimulator smoke).
- **v0: MCC = 0.333** on 4-gene reference panel (pgi, ptsG, ftsZ, JCVISYN3A_0305) at scale=0.05, t_end=0.5 s, threshold=0.10. TP=1, FP=0, TN=1, FN=2.
- Confirmed low thresholds invite FP from dATP stochastic noise; per-pool calibration needed.

### Session 3 — 2026-04-21 — Layer 0 complete + Layer 6 skeleton
- Staged the five Luthey-Schulten input files (syn3A.gb, kinetic_params.xlsx, initial_concentrations.xlsx, complex_formation.xlsx, Syn3A_updated.xml) from GitHub. SHAs in `STAGING.md`.
- Parsed CP016816.2: 543,379 bp circular, 496 gene features (458 CDS + 29 tRNA + 6 rRNA + 2 ncRNA + 1 tmRNA), oriC at position 1.
- Built Layer 0 `Genome` API + 12 validation tests. DESIGN.md + REPORT.md written.
- Extracted Breuer 2019 essentiality labels (270 Essential / 113 Quasi / 72 Nonessential = 455 labeled CDS).
- Layers 1-5 TRIAGE doc written (no re-implementation).
- Built Layer 6 skeleton: labels loader, MCC metrics, `KnockoutHarness + FailureDetector` with 7 failure modes, 13 unit tests on MockSimulator.
- Autonomously resolved 4 Phase A open questions; recorded in `memory_bank/concepts/dna/DECISIONS.md`.

### Session 2 — 2026-04-21 — Layer 0 Phase A
- Inventoried `cell_sim/` and `cell_sim_rust/` (see `EXISTING_CODE_INVENTORY.md`). 15 keep-asis, 6 adapt, 4 skip, 0 replace.
- Registered canonical sources; flagged gene-count dispute and chromosome-length as uncertainty facts pending GenBank staging.

### Session 1 — 2026-04-21 — scaffolding
- memory_bank tree + invariant checker + ranges.json + example fact + example source + session tracking files.

## Next

See `NEXT_SESSION.md`.
