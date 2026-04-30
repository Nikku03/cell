# FUTURE_WORK — parked ideas

Anything noticed mid-session but not on the current layer goes here. This file is append-only. Items get pulled out into real work only with explicit user go-ahead.

## Phase 2: vectorized-propensity gex compilation

**Status:** done in session 28. Commits `9273d57` (step 2 vectorised gex partition) + `d1de4fe` (step 3 factories emit compiled_spec) + `2dad436` (step 4 rust backend mirror) + `1346476` (step 5 wall measurement + optimisation chain) on `claude/syn3a-whole-cell-simulator-REjHC`.

**Wall-time outcome:** at python-only, gex-on slowdown drops from 3.60× (phase 1) to 1.29× (phase 2), meeting the <1.30× criterion. Recorded in `memory_bank/facts/measured/phase2_gex_compiled_wall_measurement.json`. Under the production Rust + 4-worker config the slowdown is 1.76× (the Rust backend accelerates MM but leaves gex-prop in Python); a phase 3 optimisation could move gex-prop into Rust to close that gap.

**MCC outcome:** v16 (gex-on) MCC = 0.535 vs v15 (gex-off) MCC = 0.5372 — a 0.0027 drop. One prediction flipped TP→FN; FP and TN counts are unchanged. Recorded in `memory_bank/facts/measured/mcc_against_breuer_v16_gex_compiled.json`. The v15 detector stack ignores gex events, so this is a robustness check, not a quality improvement.

**Phase 3 candidates (parked, not active work):**
- Move the gex propensity calc into the Rust extension to close the python-only-to-Rust slowdown gap (1.29× → 1.76× under Rust).
- Compile `make_protein_degradation_rule` into a `'gex'` subkind so its 458 python-closure can_fire calls per python-cache rebuild stop dominating the gex-on profile (~7% of cumulative time).
- A new gex-aware detector (e.g. TranscriptionStallDetector watching transcribe-event rate per gene) that actually USES the gex events the simulator now produces. The current v15 detector treats gex as decoration.
- Multi-seed replicates of the v16 measurement to confirm whether bit-identity across seeds {1, 2, 42} holds under gex-on the way it does for v15.

