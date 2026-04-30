# FUTURE_WORK — parked ideas

Anything noticed mid-session but not on the current layer goes here. This file is append-only. Items get pulled out into real work only with explicit user go-ahead.

## Phase 2: vectorized-propensity gex compilation

**Status:** queued. Phase 1 landed in commits `0df33ff` + `b67f12f` on `claude/syn3a-whole-cell-simulator-REjHC`. Wiring works mechanically; measured slowdown 3.60× per-run at python-closure granularity (`memory_bank/facts/measured/phase1_gex_wiring_wall_measurement.json`). Phase 2 cuts that to <1.3× by moving the high-frequency gex rules out of the python-closure cache path and into the same vectorized propensity vector as the SBML metabolic core.

**First deliverable (before any architectural change):** regression test that pins gex-off bit-identity pre-/post-change. The v15 MCC 0.5372 confusion matrix `tp=287, fp=3, tn=69, fn=96` must be exactly reproducible after every commit in phase 2; if it shifts, the phase-2 commit broke the gex-off code path and has to be reverted. Reference fact: `memory_bank/facts/measured/mcc_v15_replicates.json`.

**Scope:**
- Extend the state vector in `fast_dynamics.py` with mRNA / RNAP / ribosome / degradosome pseudo-species so they can sit alongside metabolites in the compiled propensity calc.
- Extend the `compiled_spec` schema with a `machinery_loci` path that scales propensity by a scalar machinery count (analogous to `enzyme_loci` for enzymes).
- Move `make_transcription_rule`, `make_translation_rule`, `make_mrna_degradation_rule` from python closures to compiled stoichiometric rules. Keep `make_protein_degradation_rule` as python (low rate, touches protein-instance buckets — not worth compiling).
- Mirror the same schema in the Rust backend (`cell_sim/layer2_field/rust_dynamics.py` + the PyO3 binding) so `--use-rust` works with gex on.

**Success criterion:** at flag-on, full-panel sweep wall is within 1.3× of flag-off (target: ~65 min vs current 50 min on Rust + 4 workers). At flag-off, every fact JSON that the v15 measurement chain depends on remains bit-identical.

**Resume in a fresh session.** Phase 2 is ~600-800 lines across `fast_dynamics.py` + Rust mirror + tests, and benefits from a fresh context budget covering the compiled-spec schema, the Rust backend, and the fast-equivalence test suite. The pause is intentional — phase 1 is a clean landing point and the work doesn't compound by being chained to phase 1 in one context.

