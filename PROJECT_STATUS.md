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
| 3 | Protein folding + complex assembly | partial | complex_formation.xlsx loaded by existing rules; 25 complexes defined with stoichiometry. |
| 4 | Metabolism | partial | Syn3A_updated.xml + kinetic_params.xlsx loaded by existing rules; 6 transporter k_cats patched without citation yet. |
| 5 | Biomass + division | not started | no biomass accumulation / division logic anywhere. |
| 6 | Essentiality analysis | design + labels + MCC + harness skeleton with MockSimulator; real simulator backend not yet wired | |

Phase codes (for the layers we gate): A = Literature survey, B = Design, C = Implementation, D = Validation, E = Layer report.

## Memory Bank

- Facts: **8**
  - `syn3a_doubling_time`, `syn3a_chromosome_length`, `syn3a_gene_count`, `syn3a_gene_table`, `syn3a_oric_position`, `syn3a_essentiality_breuer2019`, `syn3a_gene_count_dispute` (resolved), `syn3a_chromosome_length_pending` (resolved).
- Sources: **5** (`thornburg_2022_cell`, `hutchison_2016_science`, `breuer_2019_elife`, `genbank_cp016816`, `luthey_schulten_minimal_cell_complex_formation_repo`).
- Invariant checker: `OK`.
- Data files (tracked): `memory_bank/data/syn3a_gene_table.csv` (496 rows), `memory_bank/data/syn3a_essentiality_breuer2019.csv` (455 rows).
- Data files (local only, gitignored): 5 Luthey-Schulten input files under `cell_sim/data/Minimal_Cell_ComplexFormation/input_data/` — SHAs recorded in `memory_bank/data/STAGING.md`.

## New code in this session (Session 3)

- `cell_sim/layer0_genome/genome.py` — Layer 0 API (frozen `Genome` + `Gene`; memory-bank-backed).
- `cell_sim/layer6_essentiality/` — `labels.py`, `metrics.py`, `harness.py`, `sweep.py`.
- `cell_sim/tests/test_layer0_genome.py` (12 tests) + `test_layer6_essentiality.py` (13 tests). All 25 pass in 0.23 s.

Existing `cell_sim/layer0_genome/parser.py`, `syn3a_real.py`, `cell_sim/layer2_field/*`, `cell_sim/layer3_reactions/*`, and `cell_sim_rust/` are untouched.

## Validation Targets (reference)

- Layer 0-3: measured steady-state protein counts (Thornburg 2022) within 2x for 90% of genes.
- Layer 4: central-carbon metabolite concentrations within 2x.
- Layer 5: biomass doubling in 2 +/- 0.5 h.
- Layer 6: **MCC > 0.59** vs Breuer 2019. **Not yet measured** — blocked on wiring the real simulator into `KnockoutHarness`.

## Performance Targets

- >= 10x real-time on one CPU core.
- >= 100x real-time with Rust hot paths.
- No GPU required for normal operation.

## Session Log

### Session 3 — 2026-04-21 — Layer 0 complete + Layer 6 skeleton
- Staged the five Luthey-Schulten input files (syn3A.gb, kinetic_params.xlsx, initial_concentrations.xlsx, complex_formation.xlsx, Syn3A_updated.xml) from the public GitHub repo into `cell_sim/data/Minimal_Cell_ComplexFormation/input_data/` (gitignored). SHAs in `STAGING.md`.
- Parsed CP016816.2: **543,379 bp circular, 496 gene features (458 CDS + 29 tRNA + 6 rRNA + 2 ncRNA + 1 tmRNA), oriC at position 1**. Published 4 structural facts and a `syn3a_gene_table.csv`.
- Built Layer 0 `Genome` API + 12 validation tests. All pass. DESIGN.md + REPORT.md written.
- Extracted Breuer 2019 essentiality labels from `Comparative Proteomics` sheet (270 Essential / 113 Quasi / 72 Nonessential = 455 labeled CDS).
- Wrote Layers 1-5 TRIAGE doc (no re-implementation this session).
- Built Layer 6: labels loader, MCC metrics (pure Python), KnockoutHarness + FailureDetector with 7 failure modes and two-consecutive-sample confirmation, run_sweep orchestrator, and 13 unit tests exercising the full logic against a MockSimulator. Real simulator backend deferred.
- Committed 4 session-3 commits onto `claude/syn3a-whole-cell-simulator-REjHC`.
- Autonomously resolved the four Phase A open questions with my recommended defaults; recorded in `memory_bank/concepts/dna/DECISIONS.md` for user reversal if desired.
- Total tests: 25 pass.

### Session 2 — 2026-04-21 — Layer 0 Phase A
- Inventoried `cell_sim/` and `cell_sim_rust/` (see `EXISTING_CODE_INVENTORY.md`). 15 keep-asis, 6 adapt, 4 skip, 0 replace.
- Registered canonical sources; flagged gene-count dispute and chromosome-length as uncertainty facts pending GenBank staging.

### Session 1 — 2026-04-21 — scaffolding
- memory_bank tree + invariant checker + ranges.json + example fact + example source + session tracking files.

## Next

See `NEXT_SESSION.md`.
