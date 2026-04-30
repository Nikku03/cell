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

## Phase R2 — Regulation curation (split into R2a acquisition + R2b review)

Phase R2 was deliberately split into two passes so machine work and human work happen on separate days.

### Phase R2a — Regulation candidate acquisition (done, session 30)

**Status:** landed in session 30 as a single commit on `claude/syn3a-whole-cell-simulator-REjHC`. Adds `memory_bank/staging/regulation_curation/` (5 files: README, acquisition log, 4 candidate YAML files) plus a reproducible acquisition script `scripts/acquire_regulation_candidates.py` and a structural fact `memory_bank/facts/structural/regulation_curation_phase_r2a.json`.

**What was acquired:**

| Class | Count | Note |
|---|---|---|
| Sigma factors | 1 | rpoD / JCVISYN3A_0407 |
| Transcription factors | 3 | mraZ + 2 uncharacterized regulators |
| Riboswitches | 0 | channel limitation (RNA structure, not in CDS annotation) |
| Two-component systems | 0 | consistent with minimal-Mycoplasma expectation |

**Acquisition channel was a fallback.** The session brief specified Pfam HMM (PF00140 / PF00309 / PF08281 sigma; PF00126 / PF00165 / PF01047 / PF13411 / PF03466 TF; PF07568 / PF00512 / PF02518 / PF00072 two-component) and Rfam Infernal (riboswitch). Neither path was reachable from the acquisition sandbox: `hmmscan` / `cmscan` / `blastp` not installed; Pfam-A.hmm / Rfam.cm not on disk; EBI / Pfam / Rfam / NCBI / UniProt all blocked at the proxy with HTTP 403 `host_not_allowed`. The fallback channel is local GenBank annotation parsing of `CP016816.2` `/product=` strings, with provenance carried as `(provenance_channel: genbank_annotation, protein_id, inference_xref, product_annotation)`. Every staged candidate has full provenance; none have HMM E-values.

**Production YAML is unchanged.** `cell_sim/data/regulation_network_syn3a.yaml` still ships with all four lists empty.

**Explicit non-deliverables:**

- Zero curation decisions made. Every entry carries `confidence: candidate`.
- No production integration; `real_simulator.py` is untouched.
- No MCC measurement.

### Phase R2b — Regulation curation review (next session, human-driven)

Phase R2a (single session-30 commit, see git log) staged regulation candidates in `memory_bank/staging/regulation_curation/`. Phase R2b is the human review session.

For each candidate file, the reviewer must:

1. **Read the candidate's source provenance.** What is the JCVI annotator's product string? What does the RefSeq / SwissProt xref say about the closest characterized homolog? Is it a Mycoplasma sequence or a distant homolog?
2. **Apply biological judgment.** Is this candidate credible for Syn3A specifically? Genome reduction can leave pseudogenes that match Pfam HMMs but no longer function.
3. **Decide promotion or rejection.** Promoted entries move from staging to `cell_sim/data/regulation_network_syn3a.yaml` with `confidence` labels appropriate to the evidence (`measured` for direct experimental evidence on Syn3A or a very close homolog; `inferred` for sequence-based inference with a credible reference; otherwise reject).
4. **Document the decision.** Each promotion or rejection logged with reasoning. A reviewer six months from now must be able to reconstruct the call.

This is research work, not engineering. Allow 4–8 hours of focused review per candidate file. Don't rush; rejected candidates are better than incorrectly promoted ones.

**Phase R2b candidate priorities (from R2a output):**

- `rpoD / JCVISYN3A_0407` — canonical housekeeping sigma 70. Strongest promotion candidate; conserved across all bacteria; JCVI's annotation is unambiguous.
- `mraZ / JCVISYN3A_0525` — well-characterized cell-division-cluster repressor. Probably promotable as `inferred` based on SwissProt xref.
- `JCVISYN3A_0042` and `JCVISYN3A_0620` — uncharacterized regulators. Promotion would require reading the RefSeq xref's source organism characterization; probably best deferred.
- Two-component systems: zero acquired. The reviewer may want to re-run the strict Pfam HMM channel from a future session with EBI access before concluding "really absent vs. annotation-channel limitation."

**Phase R3 — wiring + measurement (after R2b completes):**

Once at least the most credible 1–3 entries are in the production YAML with `measured` / `inferred` confidence:

1. **Wire into `real_simulator.py`** behind `enable_regulation: bool = False` flag.
2. **Bit-identity test** at flag-off (must equal v16 baseline).
3. **Measure v17 MCC** at flag-on.
4. **Document the result** as a new fact, regardless of whether MCC moves.

Realistic time estimate for R2b + R3 (curation + integration + measurement): 4–8 weeks of focused work. The curation phase is the bottleneck and is research work, not engineering.

