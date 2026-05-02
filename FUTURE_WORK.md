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

### Phase R2b — DONE (session 31)

Reviewed 4 candidates from R2a staging. Promoted `rpoD / JCVISYN3A_0407` (sigma factor, housekeeping) and `mraZ / JCVISYN3A_0525` (TF, dcw cluster) to the production YAML at confidence: `inferred`. Deferred `JCVISYN3A_0042` and `JCVISYN3A_0620` (uncharacterized HTH-containing TFs) — both have `deferral_decision` blocks appended in staging with `next_review_trigger: target_gene_mapping_session`.

Production YAML now has 2 entries (1 sigma, 1 TF). Riboswitches and two-component systems remain `[]`. Decision audit trail in `memory_bank/staging/regulation_curation/r2b_review_log.md`. Fact: `memory_bank/facts/structural/regulation_curation_phase_r2b.json`.

**Why no `measured`:** both promotions rest on cross-species sequence homology + the JCVI annotator's gene name, not direct ChIP-seq / biochemical pull-down on Syn3A. `inferred` is the honest ceiling.

### Phase R3 — Optional regulation wiring (future session)

With 2 inferred-confidence entries in the production YAML, regulation can be wired into `real_simulator.py` behind an `enable_regulation: bool = False` flag.

1. **Bit-identity test** at flag-off (must equal v16 baseline; reuses the existing Phase 2 regression test machinery).
2. **Measure v17 MCC** at flag-on. Realistic expectation: v17 MCC ≈ v16 MCC (~0.535) ± stochastic noise, because:
   - A single housekeeping sigma factor with no competition produces no differential transcription dynamics.
   - A single TF with no defined targets has no regulatory effect.
   - The v15 detector stack ignores regulatory state entirely.
3. **Document the result** as a new fact, regardless of whether MCC moves.

Phase R3 is **optional**. The honest scientific result is that Syn3A's minimal regulation does not contribute substantial essentiality signal at this network density, and demonstrating this through measurement is reasonable but not strictly necessary. The Phase R3 wiring will also need to decide what `parameters_status: not_specified` means at runtime — either supply default kinetics or treat absent kinetics as a no-op.

### Phase R2a-strict — Optional (resolves the two-component zero)

Re-run R2a candidate acquisition with strict Pfam HMM search rather than the keyword channel that R2a fell back to. Specifically:

- Sigma: `PF00140` / `PF00309` / `PF08281`.
- TF: `PF00126` / `PF00165` / `PF01047` / `PF13411` / `PF03466`.
- Two-component: `PF07568` (HisKA) / `PF00072` (Response_reg) — the priority case.
- Riboswitch: Rfam Infernal `cmscan` against riboswitch families.

Requires a network-enabled environment with EBI / Pfam / Rfam reachable (the R2a sandbox had all these blocked at the proxy with HTTP 403 `host_not_allowed`; see `memory_bank/staging/regulation_curation/acquisition_log.md`). If a strict search returns zero two-component candidates the absence in Syn3A is biologically supported. If candidates appear, they go into staging for an R2b-style review pass.

## Dark Manifold salvage — followup status (session 32)

A followup smoke test (single session-32 commit on `claude/syn3a-whole-cell-simulator-REjHC`, see git log) re-evaluated SIREN and HNN at increased training data densities. Results: SIREN's 500-point salvage failure was data-sparsity (SIREN-ω30 wins at 500 000 points by 2.1× test MSE); HNN's damped-oscillator failure persists 1 500–8 400× across all four data conditions tested, confirming a structural failure (Hamiltonian dynamics conserve energy by construction; cannot represent dissipation). Per-run measurements at `experiments/dark_manifold_salvage_followup/results/{siren,hnn}_results.csv`. Calibrated writeup at `experiments/dark_manifold_salvage_followup/FINDINGS.md`. Memory bank fact: `dark_manifold_salvage_followup`.

The full Dark Manifold concept (4D spacetime field with dark matter coupling, quantum fluctuations as sampling, superposition collapse, cognitive scaffold) **remains untested**. Building and testing it would be a multi-month research project, not a smoke test, and is not currently planned.

## Bioelectric layer (session 33+)

Following the synthesis at `experiments/quantum_biology_speculation/SYNTHESIS.md` — which concluded that classical bioelectric (Levin lab) is the published-evidence-supported framework for cellular computation, while quantum frameworks (Orch OR, cellular quantum coprocessor) are not — the project is taking a small, calibrated step in that direction. Same R1 → R2 → R3 staging pattern as the regulation track.

### Phase B1 — DONE (session 33)

**Status:** landed in session 33 as a single commit on `claude/syn3a-whole-cell-simulator-REjHC`. Adds `cell_sim/layer5_bioelectric/` with a Goldman-Hodgkin-Katz (GHK) Vmem estimator and a small feature extractor, plus a single conditional injection of `pools["VMEM_MV"]` into trajectory snapshots when `RealSimulatorConfig.enable_bioelectric` is True.

**Explicit non-deliverables:**

- No new dynamics rules. The simulator's `_step` / `_apply` paths are not touched.
- No production wiring beyond the snapshot pool injection. The v15 / v16 detector stack does not consume `VMEM_MV`; that's Phase B3.
- No MCC measurement. Phase B1 is observable-only.
- `enable_bioelectric` defaults False; flag-off behavior is bit-identical to v15 / v16 (verified by `test_phase2_gex_off_bit_identity`).

**Caveat worth flagging in any downstream work:** Syn3A's medium has K⁺ nearly equal inside (10 mM) and outside (12.67 mM), unlike typical bacteria (200 mM in / 5 mM out). The GHK Vmem at the simulator's resting state is therefore small (single-digit positive mV) rather than the canonical bacterial −100 to −180 mV. This is a feature of JCVI-Syn3A's engineered minimal medium, not a bug. Bacterial-default permeability ratios (P_K=1, P_Na=0.04, P_Cl=0.45) are placeholders; Syn3A has no measured ion permeabilities.

### Phase B2 — DONE (session 33)

Single voltage-gated K+ leak rule wired behind the same `enable_bioelectric` flag (commit `178b8bc`). Each fire moves one K+ ion in the direction that reduces |Vmem - E_K|; token count proportional to driving force, capped at 100. Bit-identity at flag-off preserved (the rule is added to the rule list only when the flag is on; rule import lives inside the conditional). 6 new tests; 302 sandbox tests passing.

**Honest correction surfaced by B2 tests:** the Phase B1 docs claimed Vmem at Syn3A defaults is "small positive (single-digit positive mV)". That was wrong. Including the Cl- gradient (19.9 mM out / 0 in) in the GHK denominator with P_Cl=0.45 flips the sign — the actual Vmem is around −10 mV (small negative). The B1 fact, B1 module docstrings still say "small positive"; treating that as known-stale narrative.

### Phase B3 — DONE (session 33)

**v17 = v16 exactly.** Full 455-gene Breuer 2019 sweep at scale=0.05, t_end=0.5, seed=42 with `--enable-gene-expression --enable-bioelectric` produces TP=286, FP=3, TN=69, FN=97, MCC=0.5345578057087662 — bit-identical to v16's confusion matrix and MCC to all 16 decimal places. The K+ leak rule did not change a single per-gene prediction. Recorded in `memory_bank/facts/measured/mcc_against_breuer_v17_bioelectric.json`.

This is a stronger null than the realistic-expected "v17 ≈ v16 ± noise"; it's "v17 = v16 exactly." The K+ leak rule perturbs ion counts (M_k_c drifts by hundreds of ions per 0.5 s window) but the v15 detector's decision surface is dominated by features that don't depend on K+ count or Vmem (annotation-class priors + complex-assembly priors + per-rule silenced-rule decisions). The perturbation flows through the simulator but doesn't reach the predictions.

**This does not refute classical bioelectric.** Three things would change the picture:

1. A **bioelectric-aware detector** that explicitly consumes VMEM_MV-derived features (Phase B3 used the v15 detector unmodified; VMEM_MV is in pools but not in the detector's watched set). Phase B4 candidate.
2. A **multi-cell setup** where gap-junction-mediated bioelectric communication matters. Levin's published work is on collective cellular intelligence; Syn3A is single-compartment.
3. **More aggressive bioelectric dynamics**: Na+ / Cl- channels (Phase B2 was K+-only) + proton motive force pumping (the F1F0 H+ ATPase is not modeled in cell_sim at all). Phase B2.5 candidate.

**Wall cost:** v17 is ~10% slower than v16 (5827 s vs 5294 s on 4 workers Rust). The K+ leak rule fires ~16 tokens per can_fire at resting Vmem and stays in the python-closure path; vectorizing it (Phase B2-vectorize) could recover most of that.

**The Phase R/B arc summary:**

| Phase | What | MCC outcome |
|---|---|---|
| R1 | regulation infrastructure (4 rule factories) | no measurement |
| R2a | regulation candidate acquisition (4 candidates staged) | no measurement |
| R2b | regulation curation review (rpoD + mraZ promoted) | no measurement |
| B1 | bioelectric Vmem observable | no measurement |
| B2 | voltage-gated K+ leak dynamics | no measurement |
| **B3** | **v17 sweep at gex+bio on** | **v17 = v16 = 0.5346 (zero shift)** |

Both regulation and bioelectric layers now have working infrastructure but make no detectable contribution to v15-stack MCC at the Syn3A single-cell scale. This is the calibrated honest outcome — neither layer is refuted as biology, but neither buys signal that the v15 biology-first detector wasn't already extracting from annotation priors and per-rule silencing.

## Cross-organism validation (session 33-34)

The "real held-out test" the project named in past sessions: train on one organism, predict on another. Required pulling cross-organism data into the repo by hand because the sandbox proxy blocks NCBI/EBI/eLife.

**Data integrated:**
- M. pneumoniae M129 (Lluch-Senar 2015 MSB) → `labels.csv` (737 rows) + `sequences.fasta` (593 sequences after RefSeq reannotation losses + smORF gaps; 80.5% coverage)
- ESM-2 650M embeddings re-run on Colab T4 → `cell_sim/features/cache/esm2_650M_multiorg.parquet` (1048 × 1280)

**v18 measurement (new fact: `mcc_against_breuer_v18_multiorg_xgb_loocv`):**

Leave-one-organism-out CV on (Syn3A + M. pneumoniae) using ESM-2 650M + XGBoost:

| Held-out organism | n_train | n_test | MCC |
|---|---|---|---|
| M. pneumoniae | 455 | 593 | +0.1070 |
| **Syn3A** | **593** | **455** | **+0.1376** |

vs v15 baseline 0.5372 → **delta −0.3996**. Decision band per script: `FALSIFIED` (its `<0.45` threshold for "look elsewhere").

**Honest reading:** ESM-2 embeddings + XGBoost do not transfer essentiality predictions across organisms at this data scale. The v15 detector's MCC=0.5372 on Syn3A is achieved by biology-first priors (annotation-class, complex-assembly, iMB155 patches) that DO encode organism-specific information. Sequence embeddings alone discard that signal. Both held-out MCCs are barely above chance.

This does NOT mean cross-organism essentiality prediction is impossible — it means ESM-2 + XGBoost at ~500 train rows isn't the right tool. Richer features (Pfam family, GO category, pathway membership, complex membership), more organisms, or a model that explicitly transfers biology priors across organisms are all candidate next steps. None of them are session-sized; each is its own research question.

**E. coli / B. subtilis / M. genitalium remain to be integrated** (per the original Session 20 dataset spec). Same pattern as M. pneumoniae: paper-supplementary table → `labels.csv` rows; NCBI GenBank → `sequences.fasta` sequences; re-run embed notebook on Colab. Each adds ~500–4000 training rows. With all five organisms, the LOOCV would have ~2000–8000 rows per fold — a more robust test than the two-organism case.

Until then, the v18 fact is the honest answer to the cross-organism question for the data we have: at this scale, with this feature set, the v15 detector does not generalize.





