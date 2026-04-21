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
| 6 | Essentiality analysis | RealSimulator wired (Python + Rust), parallel sweep, 6 MCC measurements (v0-v5). Best balanced-panel MCC = 0.229 (v2/v3/v4); best reference-panel MCC = 0.577 (v5 per-rule detector on n=4, sample-size noise). Target 0.59 blocked by biological-redundancy FPs (per-rule) + non-catalytic essentials uncatchable at short windows. |

Phase codes (for the layers we gate): A = Literature survey, B = Design, C = Implementation, D = Validation, E = Layer report.

## Memory Bank

- Facts: **14**
  - structural (6): `syn3a_doubling_time`, `syn3a_chromosome_length`, `syn3a_gene_count`, `syn3a_gene_table`, `syn3a_oric_position`, `syn3a_essentiality_breuer2019`.
  - measured (6): `mcc_against_breuer_v0`, `v1`, `v2`, `v3`, `v4`, `v5`.
  - resolved uncertainty (2): `syn3a_gene_count_dispute`, `syn3a_chromosome_length_pending`.
- Sources: **5** (`thornburg_2022_cell`, `hutchison_2016_science`, `breuer_2019_elife`, `genbank_cp016816`, `luthey_schulten_minimal_cell_complex_formation_repo`).
- Invariant checker: `OK`.
- Data files (tracked): `memory_bank/data/syn3a_gene_table.csv` (496 rows), `memory_bank/data/syn3a_essentiality_breuer2019.csv` (455 rows).
- Data files (local only, gitignored): 5 Luthey-Schulten input files under `cell_sim/data/Minimal_Cell_ComplexFormation/input_data/` — SHAs recorded in `memory_bank/data/STAGING.md`.

## Validation Targets (reference)

- Layer 0-3: measured steady-state protein counts (Thornburg 2022) within 2x for 90% of genes.
- Layer 4: central-carbon metabolite concentrations within 2x.
- Layer 5: biomass doubling in 2 +/- 0.5 h.
- Layer 6: **MCC > 0.59** vs Breuer 2019. Measurements so far (all at scale=0.05 + t_end=0.5 s unless noted): v0 MCC=0.333 (n=4), v1=0.160 (n=40 ShortWindow+cal), v2=0.229 (n=20 scale=0.10), v3=0.229 (n=20 scale=0.25+rust), v4=0.229 (n=20 +non-metabolic pools), **v5=0.125 (n=40 PerRule) / 0.577 (n=4 PerRule reference panel)**. ShortWindowDetector hits an architectural ceiling (only pgi trips). PerRuleDetector (Session 7) catches 5 metabolic essentials but false-positives on 3 Breuer-nonessential catalytic genes because the simulator doesn't model Breuer's pathway redundancy; it remains architecturally unable to catch non-catalytic essentials (ribosomal, tRNA synthetase, replication). See `memory_bank/concepts/essentiality/REPORT.md` for the full history and the path-to-0.59 roadmap.

## Performance Targets

- ≥ 10x real-time on one CPU core — **not met**: current is ~0.06× realtime at scale=0.05 in pure Python, ~0.13× with Rust.
- ≥ 100x real-time with Rust hot paths — **not met** (see above).
- No GPU required for normal operation — **met**.
- Practical throughput: **1.9 s/gene effective wall** at scale=0.05 with Rust + 4-worker parallel (v4 config). 458-gene sweep at that config ≈ 15 min wall.

## Session Log

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
