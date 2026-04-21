# Layer 6 — Essentiality Analysis — REPORT

Layer 6 is the scientific answer-generator. This report documents what was delivered in Session 3 and exactly what remains before the brief's MCC > 0.59 target can be measured.

## Delivered in Session 3

Module: `cell_sim/layer6_essentiality/`

- `labels.py` — Loader for the 455-row Breuer 2019 essentiality table (`memory_bank/data/syn3a_essentiality_breuer2019.csv`). Exposes `load_breuer2019_labels()`, `binary_labels(quasi_as_positive=True|False)`, and `select()`.
- `metrics.py` — Pure-Python `evaluate_binary(y_true, y_pred) -> Metrics` with MCC, precision, recall, specificity, accuracy, confusion matrix. No sklearn dependency.
- `harness.py` — `FailureDetector` (seven failure modes, two-consecutive-sample confirmation, ratio-to-WT thresholds), `KnockoutHarness` (WT caching, per-gene prediction), `Simulator` protocol, and `MockSimulator` for tests.
- `sweep.py` — `run_sweep(...)` that iterates genes, writes predictions CSV, and returns `list[Prediction]`. Uses `Genome.load()` so it always sees all 458 CDS.

Tests (`cell_sim/tests/test_layer6_essentiality.py`): **13/13 pass** and run in <200 ms. Coverage:

- Label loading (counts match the fact, binary mapping is correct).
- MCC hand-calculation sanity check; degenerate one-class input returns 0.
- Failure detection on hand-built trajectories for ATP crash, essential-metabolite depletion, translation stall, and a noisy-WT-like knockout that correctly does NOT trip.
- End-to-end sweep on a 2-gene subset produces the expected CSV.
- Sanity check: a trivial all-positive predictor does NOT beat the 0.59 MCC target — confirming the target is non-trivial.

Memory bank: `facts/structural/syn3a_essentiality_breuer2019.json` points at the labels CSV; its `used_by` references `labels.py` and the tests.

## NOT delivered (deferred to next sessions)

1. **Real simulator backend.** The harness accepts any `Simulator` conforming to the `run(knockout, t_end_s, sample_dt_s) -> Trajectory` protocol. The production backend, wrapping `cell_sim.layer2_field.FastEventSimulator` + `populate_real_syn3a`, is not wired in this session. Plumbing it requires:
   - Resolving the sample-grid mismatch between the event-driven sim (continuous-time jumps) and the periodic-sample trajectory expected by the detector (build a sampling adapter).
   - Exposing the required pool keys (`ATP`, `CHARGED_TRNA_FRACTION`, `NTP_TOTAL`, essential metabolites) from the sim state.
   - A way to remove transition rules for knocked-out genes' reactions (currently the existing `test_knockouts.py` uses copy+filter; we lift that into `KnockoutHarness`).
2. **The actual genome-wide sweep.** Needs the real backend. Estimated wall time at 50% scale: ~150 CPU-hours for 458 CDS at `t_end_s = 7200`. Should be run on a cluster with embarrassingly parallel per-gene workers, not from an interactive session.
3. **MCC measurement against 0.59 target.** Blocked on the above.
4. **Threshold tuning.** Current thresholds in `harness.THRESHOLDS` are conservative first-pass numbers. They WILL be tuned against a held-out subset of the Breuer labels once the sweep produces real predictions.
5. **Non-CDS gene handling.** The sweep defaults to CDS only (458 genes). The 38 RNA genes (tRNA, rRNA, ncRNA, tmRNA) are not in the current sweep scope; whether to include them is a scientific decision deferred to the first real evaluation run.

## Why this split

Layer 6's *logic* is fully testable without running the full simulator — the detector math doesn't care whether the trajectories are real or synthetic. Delivering the harness + labels + metrics + mock simulator gives us an auditable MCC pipeline today. Adding the real simulator is pure plumbing, not new science; it fits one or two sessions whenever the simulator is reliable enough to produce real trajectories.

Separating the two also means **improvements to Layers 1-5 are evaluable.** Any session that improves the simulator can run the same sweep against the same labels and compare MCC deltas — this gives us a quantitative regression test for biological fidelity, which is exactly the brief's gate for "the simulator works."

## Known limitations

1. **Three unlabeled CDS.** 458 CDS exist in CP016816.2; 455 have Breuer labels. The sweep predicts all 458, but MCC is only evaluated on the 455 labeled ones.
2. **Breuer 2019 is not the only essentiality source.** Hutchison 2016 has transposon-based labels that may disagree for some genes (brief section 7 lists both). We compute MCC against Breuer to match the brief's 0.59 benchmark; a secondary comparison against Hutchison is FUTURE_WORK.
3. **Binary collapse loses information.** `time_to_failure` and `failure_mode` are produced per gene and written to the CSV but not currently scored. A future metric (e.g. "predicts correct failure mode for essential genes in >X%") is a richer target and belongs on the roadmap.

## Validation target from the brief (Layer 6)

- MCC > 0.59 against Breuer 2019 with `{Essential, Quasiessential}` as positive. **Not yet measured** (blocked on real-simulator wiring).

## Handoff

See `NEXT_SESSION.md` for the ordered to-do list. The first Layer 6 follow-up is wiring the real `Simulator` backend — a focused single-session task with a clear acceptance test: running `run_sweep` on a 4-gene subset and confirming the two Breuer-essential genes (e.g. pgi, ptsG) come back with `essential=True` and the two Breuer-nonessential genes (e.g. ftsZ, an uncharacterized peptidase) come back with `essential=False`.
