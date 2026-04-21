# PROJECT_STATUS — SYN3A Whole-Cell Simulator

_This file is the authoritative snapshot of project state. It is updated at the end of every session and read at the start of every session. If this file is out of date, no work proceeds until it is reconciled._

## The Goal (unchanging)

Build a biologically accurate, computationally cheap Syn3A whole-cell simulator that predicts **time-dependent gene essentiality** with **Matthews correlation coefficient > 0.59** against Breuer 2019 experimental labels, for all 452 Syn3A genes.

## Layer Progress

| Layer | Name | Phase | Status |
|-------|------|-------|--------|
| 0 | Genome | Phase A complete (awaiting user sign-off for Phase B) | summary in memory_bank/concepts/dna/PHASE_A_SUMMARY.md |
| 1 | Transcription machinery | not started | — |
| 2 | Translation machinery | not started | — |
| 3 | Protein folding + complex assembly | not started | — |
| 4 | Metabolism | not started | — |
| 5 | Biomass + division | not started | — |
| 6 | Knockout + essentiality analysis | not started | — |

Phase codes: A = Literature survey, B = Design, C = Implementation, D = Validation, E = Layer report.

## Memory Bank

- Directory tree: created (`memory_bank/facts`, `memory_bank/concepts`, `memory_bank/sources`, `memory_bank/index`, `memory_bank/.invariants`).
- Invariant checker: `memory_bank/.invariants/check.py` (runs from repo root, rebuilds `memory_bank/index/facts.sqlite` on success).
- Range limits config: `memory_bank/.invariants/ranges.json`.
- Facts: 3 (`syn3a_doubling_time`, `syn3a_gene_count_dispute`, `syn3a_chromosome_length_pending`).
- Sources: 4 (`thornburg_2022_cell`, `hutchison_2016_science`, `breuer_2019_elife`, `genbank_cp016816`).

## Existing Code (pre-brief)

The repository already contains prior work in `cell_sim/` (including the event-driven engine at `cell_sim/layer2_field/fast_dynamics.py`) and `cell_sim_rust/`. Per the brief section 8, we reuse these rather than rewrite. A detailed inventory of which components we will carry forward vs. replace is scheduled for early in Layer 0 Phase A.

## Validation Targets (repeated here for convenience)

- Layer 0-3: measured steady-state protein counts (Thornburg 2022) within 2x for 90% of genes.
- Layer 4: measured central-carbon metabolite concentrations within 2x.
- Layer 5: biomass doubling in 2 +/- 0.5 h.
- Layer 6: MCC > 0.59 vs Breuer 2019.

## Performance Targets

- >= 10x real-time on one CPU core (pure Python).
- >= 100x real-time with Rust hot paths.
- No GPU required for normal operation.

## Session Log

### Session 2 — 2026-04-21 — Layer 0 Phase A
- Ran pre-flight invariant check (`OK`).
- Inventoried all of `cell_sim/` and `cell_sim_rust/` via an Explore subagent. Result: `memory_bank/concepts/dna/EXISTING_CODE_INVENTORY.md`. Verdicts: 15 keep-asis, 6 adapt, 4 skip, 0 replace. No file needs to be thrown away.
- Registered three canonical sources: `genbank_cp016816`, `hutchison_2016_science`, `breuer_2019_elife`.
- Found gene-count disagreement across sources (452 / 493 / 496 / 473). Recorded as `facts/uncertainty/syn3a_gene_count_dispute.json` with all four candidate numbers and their provenance. Recorded chromosome length as pending (`facts/uncertainty/syn3a_chromosome_length_pending.json`) because the GenBank file is not on disk.
- Identified the main Phase B blocker: the Luthey-Schulten data files (`syn3A.gb`, kinetics / proteomics / complexes Excel files) that the existing `cell_sim/layer0_genome/syn3a_real.py` expects are not committed. Four open questions for the user are listed at the bottom of `PHASE_A_SUMMARY.md`.
- Final invariant check `OK` (3 facts, 4 sources). Commit pending; push still blocked on GitHub auth from previous sessions.

### Session 1 — 2026-04-21 — scaffolding
- Read the project brief; confirmed working branch `claude/syn3a-whole-cell-simulator-REjHC`.
- Created `memory_bank/` directory tree per brief section 4.1.
- Wrote `memory_bank/.invariants/check.py` implementing all seven invariants from brief section 4.3 (required fields, source resolvability, dependency existence, parameter-range check, contradiction detection, `used_by` file existence, staleness warning). Checker rebuilds `memory_bank/index/facts.sqlite` on success.
- Wrote `memory_bank/.invariants/ranges.json` with physical ranges covering the parameter types we expect in Layers 0-6.
- Wrote example source `memory_bank/sources/thornburg_2022_cell.json` and example fact `memory_bank/facts/parameters/syn3a_doubling_time.json` to exercise the fact format end-to-end.
- Wrote this file, `NEXT_SESSION.md`, and `FUTURE_WORK.md`.
- Ran invariant checker; confirmed it passes on the example fact.
- Committed to `claude/syn3a-whole-cell-simulator-REjHC`.
- Did **not** read any biology papers (brief section 10 rule).

## Next

See `NEXT_SESSION.md`.
