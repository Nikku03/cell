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

- **Target:** MCC > 0.59 against Breuer 2019 with `{Essential, Quasiessential}` as positive.
- **Best measurement so far: MCC = 0.333** on the 4-gene reference panel (Session 4, v0). Larger samples produce lower MCC because more essential genes expose a fundamental sensitivity floor.

### Session-by-session MCC history

| Version | Session | n | Config | MCC | Confusion | Notes |
|---|---|---|---|---|---|---|
| v0 | 4 | 4 | scale=0.05, t_end=0.5, thr=0.10 | **0.333** | 1 TP / 0 FP / 1 TN / 2 FN | pgi caught via F6P -11%. |
| (v0b) | 4 | 4 | scale=0.10, t_end=1.0, thr=0.05 | -0.577 | 1 TP / 1 FP / 0 TN / 2 FN | Low threshold tripped FP from dATP noise. |
| (v0c) | 5 | 4 | scale=0.05, t_end=2.0, thr=0.10 | 0.333 | Identical to v0 | Longer window did not help. |
| v1 | 5 | 40 | scale=0.05, t_end=0.5, cal=10, sf=2.5 | **0.160** | 1 TP / 0 FP / 20 TN / 19 FN | Parallel 4-worker. Specificity=1; recall=0.05. |
| v2 | 5 | 20 | scale=0.10, t_end=1.0, cal=5, sf=2.5 | **0.229** | 1 TP / 0 FP / 10 TN / 9 FN | Doubled scale + window: no improvement over v1. |

### The one thing the detector catches

Every config catches exactly `JCVISYN3A_0445` (pgi) via F6P depletion of 6–11 %. That's the only Essential-class gene whose KO produces a detectable metabolite signature in the short-window real simulator at the tried scales.

### What gets missed and why

- **Ribosomal proteins** (rpmI, rpsI, rpsN, rpsR, rpsE, rpsB, rpmF): KO stops ribosome assembly and translation. Effect on metabolite pools is slow (minutes). The detector watches central-carbon + NTP + dNTP pools only; it does not watch ribosome assembly state, protein synthesis rate, or charged-tRNA fraction.
- **tRNA synthetases / RNases** (rnjB): RNA-processing defects take many bio-seconds to manifest in downstream pools.
- **Transporters / PTS components** (ptsG, crr, ywjA): upstream of central carbon; pool depletion takes longer than 0.5–1 s of bio-time.
- **Replication machinery, lipid synthesis** (plsX), **cell division** (ftsZ — correctly predicted non-essential at these short windows).

### Three orthogonal paths to MCC > 0.59 (for next session)

1. **Non-metabolic signals in the detector.** Add a pool for charged-tRNA fraction and a pool for ribosome-assembly-complex count, sourced from `CellState.proteins_by_state`. Ribosomal-protein KOs would then show up via the charged-tRNA lump freezing (no ribosome turnover to unload it) or via a drop in intact ribosome count. This requires ~100 lines in `real_simulator.py::_snapshot`.
2. **Trajectory-divergence integral instead of threshold.** Rather than `|ko/wt - 1| > X at two samples`, sum the deviations across the whole window and compare to a distribution of WT-replicate integrals. Captures slow drifts that never cross a hard threshold. Requires a few WT replicates (multiple seeds) to build the null distribution.
3. **Larger-scale long-window runs on real hardware.** The knockout_test.py in the existing cell_sim reports clear signal at scale=0.5, t_end=0.5 s — that run takes ~25 min per gene. The full 458-gene sweep at that scale on a cluster (128 cores) fits in ~2 hours wall. At that budget the `ShortWindowDetector` as-is probably hits 0.4–0.5; with signal (1) added as well, 0.59 is plausible.

## Session 4 additions

- `cell_sim/layer6_essentiality/real_simulator.py` — `RealSimulator` Protocol implementation wrapping `FastEventSimulator` + `populate_real_syn3a` + reversible MM rules + nutrient-uptake patches. Caches heavy setup (SBML parse, kinetics, base CellSpec) across knockouts.
- `cell_sim/layer6_essentiality/short_window_detector.py` — `ShortWindowDetector` purpose-built for sub-second runs: bidirectional `|ko/wt - 1| > X` with two-consecutive confirmation. Catches both substrate buildup AND product depletion.
- `scripts/run_full_sweep_real.py` — runnable sweep script with `--reference-panel`, `--max-genes N --balanced`, and `--all` modes; emits `outputs/predictions_*.csv` + `outputs/metrics_*.json`.
- `cell_sim/tests/test_layer6_short_window_detector.py` — 8 new tests including a `RealSimulator` smoke test (skipped when data not staged).

## Handoff

See `NEXT_SESSION.md` for the ordered to-do list. The first Layer 6 follow-up is wiring the real `Simulator` backend — a focused single-session task with a clear acceptance test: running `run_sweep` on a 4-gene subset and confirming the two Breuer-essential genes (e.g. pgi, ptsG) come back with `essential=True` and the two Breuer-nonessential genes (e.g. ftsZ, an uncharacterized peptidase) come back with `essential=False`.
