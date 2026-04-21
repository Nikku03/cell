# NEXT_SESSION — queued work

_Read this file FIRST, immediately after running the invariant checker._

## Pre-flight

1. `python memory_bank/.invariants/check.py` — must print `OK`.
2. Read `PROJECT_STATUS.md`.
3. Read `memory_bank/concepts/essentiality/REPORT.md` (Session 4 added the real-sim section).
4. Read this file.

## Where Layer 6 stands

- Real simulator wired (`RealSimulator`, `ShortWindowDetector`, `run_full_sweep_real.py`).
- First MCC measurement: **0.333** on the 4-gene reference panel at scale=0.05, t_end=0.5 s.
- Brief target: **MCC > 0.59**.
- Bottleneck identified: the 0.5 s simulation window is too short for upstream-of-glycolysis knockouts (e.g. transporters like `ptsG`) to propagate to detectable metabolite changes. Lower thresholds invite false positives from dATP / NADH stochastic noise.

## Highest-priority queue (in order)

### 1. Push MCC toward 0.59 — three orthogonal improvements

Pick whichever is cheapest first; combine if needed.

**1a. Longer simulation window (single-process; only requires patience).** Re-run the reference panel at t_end_s = 5.0 s with the same scale=0.05. Wall: ~2.5 min/gene → ~10 min for the panel. If `ptsG` now shows F6P depletion, the longer window is enough. If it does, run the full sweep at the same config (~38 hours single-process — needs multiprocessing, see §1c).

**1b. Better detection with a noise-floor calibration step.** Add a `--calibrate K` mode to `run_full_sweep_real.py` that:
  - Picks K Breuer-non-essential genes at random.
  - Runs them as a calibration set.
  - Computes `noise_floor[pool] = max(|ko/wt - 1|)` per pool from those K runs (helper already exists: `short_window_detector.calibrate_noise_floor`).
  - Sets per-pool thresholds = `noise_floor[pool] * safety_factor` (start with `safety_factor = 2.0`).
  - Then runs the actual sweep with those thresholds.
This costs K × 13 s upfront and should improve specificity dramatically.

**1c. Multiprocess fan-out.** The existing `RealSimulator` is not thread-safe (the `FastEventSimulator` shares numpy buffers). Multiprocess is fine: each worker holds its own `RealSimulator` instance. Add `scripts/run_sweep_parallel.py` that uses `multiprocessing.Pool` to run N workers, each handling 1/N of the gene list, then merge the CSVs and compute MCC. On a 16-core CPU the 458-gene full sweep at scale=0.05 / t_end=0.5 s drops from ~100 min to ~7 min wall.

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
