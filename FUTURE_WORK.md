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

## Phase R1 — Regulation layer infrastructure (done, no biology)

**Status:** landed in session 29 as a single commit on `claude/syn3a-whole-cell-simulator-REjHC`. Adds `cell_sim/layer4_regulation/` with rule factory scaffolding for TF binding, sigma factor competition, riboswitch folding, and two-component signaling, plus `cell_sim/data/regulation_network_syn3a.yaml` (schema-only, zero entries) and 13 new tests. Recorded as `memory_bank/facts/structural/regulation_layer_phase_r1.json`.

**Explicit non-deliverables (do NOT claim otherwise):**

- No biological accuracy. Rate constants are placeholders; calling a factory without `source_citation` raises `BiologyNotCurated`.
- No production integration. `real_simulator.py` does not import the regulation layer; v15 / v16 MCC values are unchanged. The Phase 2 gex-off bit-identity regression test still passes.
- No MCC measurement. Phase R1 produces no detector evaluation.

## Phase R2 — Regulation curation (next session)

Phase R1 (single session-29 commit on `claude/syn3a-whole-cell-simulator-REjHC`, see git log for SHA) produced regulation layer infrastructure. The next phase is biological curation, which is independent from infrastructure and was deliberately deferred.

Curation tasks before regulation can be wired into the production sweep:

1. **Identify Syn3A sigma factors.** Sequence homology against B. subtilis SigA / SigH or M. genitalium homologs. Probably 1–3 sigma factors total in Syn3A given its minimal genome. Source: literature review + BLAST.
2. **Identify transcription factors.** M. genitalium has a small TF complement (~5–10). Sequence-homology starting points exist in public databases. Source: NCBI HMM searches against TF Pfam families.
3. **Map TF target genes.** This is the bottleneck. Direct ChIP-seq data does not exist for Syn3A. Closest analog: M. pneumoniae regulon studies (Lluch-Senar 2015 supplementary data may have inferred regulatory targets). Most TF–target relationships will be tagged `inferred` rather than `measured`.
4. **Identify riboswitches.** RNA structure prediction on UTRs of Syn3A genes. Tools: Infernal/Rfam. Likely 0–3 riboswitches given minimal genome.
5. **Identify two-component systems.** Genome scan for sensor kinase + response regulator gene pairs. M. genitalium has limited two-component signaling.

Once curation is complete and entered into `regulation_network_syn3a.yaml` with proper source citations:

6. **Wire into `real_simulator.py`** behind `enable_regulation: bool = False` flag.
7. **Bit-identity test** at flag-off (must equal v16 baseline).
8. **Measure v17 MCC** at flag-on.
9. **Document the result** as a new fact, regardless of whether MCC moves.

Realistic time estimate for Phase R2 (curation + integration + measurement): 4–8 weeks of focused work. The curation phase is the bottleneck and is research work, not engineering.

