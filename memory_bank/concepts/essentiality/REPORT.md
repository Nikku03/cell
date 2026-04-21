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

| Version | Session | n | Detector / config | MCC | Wall (s/gene) | Notes |
|---|---|---|---|---|---|---|
| v0 | 4 | 4 | ShortWindow scale=0.05, t_end=0.5, thr=0.10 | **0.333** | 13 | pgi via F6P -11%. |
| (v0b) | 4 | 4 | ShortWindow scale=0.10, t_end=1.0, thr=0.05 | -0.577 | 40 | FP from dATP noise. |
| (v0c) | 5 | 4 | ShortWindow scale=0.05, t_end=2.0, thr=0.10 | 0.333 | 50 | Longer window didn't help. |
| v1 | 5 | 40 | ShortWindow cal=10, sf=2.5 | **0.160** | 4.9 | Specificity=1; recall=0.05. |
| v2 | 5 | 20 | ShortWindow scale=0.10, t_end=1.0, cal=5 | 0.229 | 12.7 | No improvement over v1. |
| v3 | 6 | 20 | ShortWindow scale=0.25, rust, cal=5 | 0.229 | 7.1 | pgi F6P grows to -15%. |
| v4 | 6 | 20 | ShortWindow +non-metabolic +TOTAL_EVENTS, rust | 0.229 | **1.9** | Full ShortWindow stack. |
| v5 | 7 | 40 | **PerRule** min_wt=20, rust | **0.125** | 1.7 | 5 TP / 3 FP / 17 TN / 15 FN. |
| v5-ref | 7 | 4 | PerRule (reference panel) | 0.577 | 2.5 | pgi + ptsG caught; sample-size noise. |

### What each detector family catches

**ShortWindowDetector** (v0–v4): catches exactly `JCVISYN3A_0445` (pgi) via F6P depletion of 5–15 %, regardless of scale / window / pool set / threshold. The only Essential-class gene whose KO produces a detectable metabolite / non-metabolic signature in ≤0.5 s bio-time at tractable scales.

**PerRuleDetector** (v5, Session 7): catches 5 of 20 balanced-panel essentials at n=40 — the metabolic subset (pgi, tpiA, plsX area, 0813, 0729) — via direct "catalysis rule stops firing" signal. **Also produces 3 FPs** on Breuer-nonessential catalytic genes (lpdA, deoC, 0034 transport system) because the simulator lacks the pathway redundancy that Breuer's nonessentiality labels account for. Architecturally misses all 15 non-catalytic essentials in the panel (ribosomal proteins, tRNA factors, replication). Net MCC=0.125 at n=40, worse than v4 ShortWindow = 0.229.

### The Session-6 infrastructure stack

Session 6 built the plumbing that makes long-running / large-scale sweeps tractable, even though the MCC number didn't move:

- **Rust hot path** (`cell_sim_rust`): wheel built, installed, exposed via `RealSimulatorConfig(use_rust_backend=True)` and `--use-rust` on both sweep scripts. ~2× speedup on a single core at scale=0.05.
- **Multiprocess fan-out**: `scripts/run_sweep_parallel.py` with `--workers N`. ~4× on this 4-core sandbox.
- **Combined speedup vs baseline (v0): ~7×** (1.9 s/gene effective vs 13 s in v0).
- **Non-metabolic pool signals** added to `RealSimulator._snapshot`: `TOTAL_COMPLEXES`, `FOLDED_PROTEINS`, `UNFOLDED_PROTEINS`, `FOLDED_FRACTION`, `BOUND_PROTEINS`, `TOTAL_EVENTS`. All plumbed into `SHORT_WINDOW_POOLS` and calibrated. None trip at 0.5 s bio-time, but the framework is ready.

### What gets missed and why — updated diagnosis

Across v0-v4 (5 configs spanning scale 0.05-0.25 and t_end 0.5-2.0s with/without non-metabolic pools), the detector trips **only** on pgi. All other essentials (ribosomal proteins, tRNA synthetases, transporters, RNase, replication, lipid synthesis) produce sub-threshold deviations across every watched pool. **This is architectural**, not a threshold-tuning failure:

- At 0.5 s bio-time, 140 k events fire across ~160 reactions. Knocking out one protein typically removes a few reaction-pathways; the remaining ~155 reactions still fire normally, so aggregate pool levels and event counts barely move.
- Only central-glycolysis disruption (which has a high flux relative to other pathways and short feedback loops) produces detectable perturbation in 0.5 s.

### Three paths to MCC > 0.59 (updated after Session 7)

Path #1 has been tried (v5). The remaining paths:

1. **Ensemble: require both detectors to agree** (Session 8 top lever). AND the per-rule signal with a weak pool-deviation signal. Drops the 3 FPs from v5 (they catalyse rules but don't perturb central carbon because Breuer redundancy provides alternate paths). Keeps the 5 TPs (they do perturb pools). Predicted lift to MCC ~0.3–0.4 on n=40 balanced; still capped by non-catalytic essentials being invisible at short windows.
2. **Longer bio-time run** (Path A, committed compute). At 1.7 s/gene in Rust+parallel, a 458-gene sweep at t_end=5.0 s is ~3 h wall. At that window, transporter and translation-adjacent KOs may start producing pool signatures. Pair with ensemble.
3. **Rule-necessity weighting.** For each rule, count how many genes catalyse it. `PerRuleDetector` should only trip on genes whose silenced rules have NO alternate catalysers (the "uniquely-required" set). Addresses the v5 FP mechanism directly. Orthogonal to detector fusion; can combine.

Architectural limit that no short-window detector can cross: **non-catalytic essentials (ribosomal proteins, tRNA synthetases, translation factors, replication machinery) do not perturb pools or catalysis events in ≤0.5 s**. Reaching those requires Path #2's longer bio-time, not better detectors.

## Session 4 additions

- `cell_sim/layer6_essentiality/real_simulator.py` — `RealSimulator` Protocol implementation wrapping `FastEventSimulator` + `populate_real_syn3a` + reversible MM rules + nutrient-uptake patches. Caches heavy setup (SBML parse, kinetics, base CellSpec) across knockouts.
- `cell_sim/layer6_essentiality/short_window_detector.py` — `ShortWindowDetector` purpose-built for sub-second runs: bidirectional `|ko/wt - 1| > X` with two-consecutive confirmation. Catches both substrate buildup AND product depletion.
- `scripts/run_full_sweep_real.py` — runnable sweep script with `--reference-panel`, `--max-genes N --balanced`, and `--all` modes; emits `outputs/predictions_*.csv` + `outputs/metrics_*.json`.
- `cell_sim/tests/test_layer6_short_window_detector.py` — 8 new tests including a `RealSimulator` smoke test (skipped when data not staged).

## Handoff

See `NEXT_SESSION.md` for the ordered to-do list. The first Layer 6 follow-up is wiring the real `Simulator` backend — a focused single-session task with a clear acceptance test: running `run_sweep` on a 4-gene subset and confirming the two Breuer-essential genes (e.g. pgi, ptsG) come back with `essential=True` and the two Breuer-nonessential genes (e.g. ftsZ, an uncharacterized peptidase) come back with `essential=False`.
