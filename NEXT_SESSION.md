# NEXT_SESSION — queued work

_Read this file FIRST, immediately after running the invariant checker._

## Pre-flight

1. `python memory_bank/.invariants/check.py` — must print `OK`.
2. Read `PROJECT_STATUS.md`.
3. Read `memory_bank/concepts/essentiality/REPORT.md` (Session 4 added the real-sim section).
4. Read this file.

## Where Layer 6 stands

- Real simulator wired (`RealSimulator`, `ShortWindowDetector`, `run_full_sweep_real.py`, `run_sweep_parallel.py` with 4-worker fan-out).
- Calibration mode + per-pool thresholds in `ShortWindowDetector` shipped; 5–10 non-essential KOs produce a noise floor used to set tight per-pool thresholds.
- Three MCC measurements recorded as measured facts (`mcc_against_breuer_v0` / `v1` / `v2`):
  - v0, n=4: 0.333 (scale=0.05, t_end=0.5)
  - v1, n=40 balanced with calibration: 0.160 (scale=0.05, t_end=0.5)
  - v2, n=20 balanced with calibration: 0.229 (scale=0.10, t_end=1.0)
- **Conclusion: at tractable scales (<=0.1) and windows (<=2 s), the detector catches exactly one gene — `pgi` — via F6P depletion.** Scaling the knobs does not help. The limit is not a threshold-tuning problem; it's that most essential-gene KOs don't produce detectable *metabolite* signatures within this runtime budget.
- Brief target: **MCC > 0.59**. We are at 0.16–0.33 depending on sample size.

## Highest-priority queue (in order)

### 1. Push MCC toward 0.59 — three orthogonal improvements

Based on Session 5's diagnostic measurements, the **high-leverage** move is #1a below. Tuning (#1b, #1c) was tried in Session 5 and does not fix the underlying signal problem.

**1a. Add non-metabolic detection signals (HIGH leverage, medium cost).** Extend `RealSimulator._snapshot` to also emit:
  - `RIBOSOME_COUNT`: number of fully-assembled 70S ribosomes (from `CellState.complexes`, filter by complex name == "ribosome" or by matching the 24 complex definitions).
  - `CHARGED_TRNA_FRACTION`: aggregate across the 20 aa-tRNA synthetase outputs. Needs some spelunking in the existing `real_syn3a_rules.py`; the tRNA-charging pools are on `kinetic_params.xlsx` sheet "tRNA Charging".
  - `TOTAL_PROTEIN_COUNT`: sum across all `state.proteins` — drops when folding / translation is halted.
  With these pools populated, ribosomal-protein KOs and tRNA-synthetase KOs get detected because the ribosome pool stops replenishing. Estimated: +15 essentials caught out of 20 in the balanced sample → MCC lift to ~0.4-0.5 without any other change.

**1b. Trajectory-divergence integral instead of threshold (MEDIUM leverage).** Run N WT replicates with different seeds, build a per-pool null distribution of `integral_|ratio-1| dt`, then flag a KO as essential if its integral exceeds the 99th percentile of WT-to-WT integrals. Catches slow drifts that never cross any hard threshold. Cost: N × WT wall (modest). Needs to be paired with 1a to add to 0.59; alone probably a small lift only.

**1c. Larger-scale long-window runs (costs compute).** The existing `cell_sim/tests/test_knockouts.py` reports 20% deviations on ATP/G6P at scale=0.5 / t_end=0.5 s — but at ~25 min wall per gene. That's 192 CPU-hours for the 458-CDS sweep; tractable on a cluster but not interactive. **Only do this after 1a and 1b.**

### 2. Layer 5 — biomass + division

Without it the only "viability" proxy in Layer 6 is "did metabolites crash within t_end". A real biomass + division check would let us answer "does the cell double in 2 ± 0.5 h", which the brief asks for. Phase A: register the biomass composition vector from Thornburg 2022.

### 3. The six patched-transporter k_cat fact files

Already noted in the Session-3 NEXT_SESSION; still pending. Pure paper-trail work.

### 4. Hutchison 2016 secondary essentiality labels

Layer 6 currently benchmarks against Breuer only. Hutchison 2016 transposon labels are a secondary ground-truth; adding them lets us report MCC against both and flag genes where the two sources disagree.

## Lower-priority queue

### 5. Drop the unused `torch` import from `cell_sim/layer3_reactions/network.py`

One-liner; no approval needed. Was deferred from Session 3.

### 6. Build `cell_sim_rust` extension

Last-resort optimisation. The Python `FastEventSimulator` runs at ~0.06x realtime at scale=0.05. The Rust `compute_propensities` would speed up the inner propensity loop, but profiling first to confirm that loop is actually the bottleneck — at small scale the `populate_real_syn3a` setup may dominate.

## Deferred (not for next session)

- Multi-gene knockouts.
- Synthetic lethality.
- Neural-net anything (brief section 2 forbids it without justification).

## Git state

All Session 1-4 commits pushed to `origin/claude/syn3a-whole-cell-simulator-REjHC` (HEAD = `7216586`). Local `origin` URL still points at the dead local proxy; future pushes from this sandbox use the stashed PAT in `/tmp/.gh_tkn` (gitignored, chmod 600). Token must be revoked when work concludes.

## How to actually run the next sweep

```bash
# Reference panel only (~1 min):
python scripts/run_full_sweep_real.py --reference-panel

# Calibrate + balanced 50-gene sweep at the longer window (~10 min wall once 1b is implemented):
python scripts/run_full_sweep_real.py --max-genes 50 --balanced --t-end-s 5.0

# Full sweep, multi-process (after 1c is implemented):
python scripts/run_sweep_parallel.py --workers 8 --all
```
