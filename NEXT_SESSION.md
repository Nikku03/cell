# NEXT_SESSION — queued work

_Read this file FIRST, immediately after running the invariant checker._

## Pre-flight

1. `python memory_bank/.invariants/check.py` — must print `OK`.
2. Read `PROJECT_STATUS.md`.
3. Read `memory_bank/concepts/essentiality/REPORT.md` (Session 4 added the real-sim section).
4. Read this file.

## Where Layer 6 stands

**Infrastructure** (v0 → v4 cumulative):
- Real simulator wired (`RealSimulator`, `ShortWindowDetector`).
- `scripts/run_full_sweep_real.py` — single-process with `--calibrate K`.
- `scripts/run_sweep_parallel.py` — 4-worker multiprocess + `--use-rust`.
- `cell_sim_rust` wheel built and installed; `RealSimulatorConfig.use_rust_backend` toggles it.
- **7× speedup vs Session-4 baseline** (1.9 s/gene effective at scale=0.05 with Rust + 4-worker parallel).
- Expanded pool set in `_snapshot`: metabolites + `TOTAL_COMPLEXES`, `FOLDED_PROTEINS`, `UNFOLDED_PROTEINS`, `FOLDED_FRACTION`, `BOUND_PROTEINS`, `TOTAL_EVENTS`.

**MCC measurements** (all recorded as `facts/measured/mcc_against_breuer_v0..v4.json`):
- v0 (n=4): 0.333
- v1 (n=40): 0.160
- v2 (n=20, scale=0.10): 0.229
- v3 (n=20, scale=0.25, rust): 0.229
- v4 (n=20, scale=0.05, rust, + non-metabolic pools + TOTAL_EVENTS): 0.229

**Diagnosis (definitive across v0-v4):** the detector catches exactly **one gene** — `pgi` — at any tested short-window (≤0.5 s) config. The short-window essentiality-detection floor is **architectural**, not threshold-tuning. Most essential-gene KOs don't perturb aggregate pool levels by more than 1-3% within 0.5 s because only a handful of the ~160 reactions stop firing — the other 155 dilute the signal. Brief target **MCC > 0.59** unreachable at these windows.

## Highest-priority queue (updated after Session-6 measurements)

### 1. Per-rule event-count detection (HIGH leverage, cheap — THE move)

Session 6 proved that pool-based detection hits a hard ceiling regardless of pool choice, scale, or threshold. The remaining high-leverage lever is a **different detection signal**: count how many times each `catalysis:REACTION` rule fires in KO vs WT.

Mechanism: for each gene, obtain the set of rule names its product catalyses (structural, from `real_syn3a_rules.py`). A KO of gene X should cause `events_per_rule["catalysis:Y"]` for all Y in that set to drop to ~0 in KO while remaining >threshold in WT. This is:

- **Direct causal**: not a downstream propagation that takes bio-time — the direct effect is immediate.
- **Gene-specific**: each gene has a known rule set, so the detector can be asked "did the right rules stop?".

Implementation sketch:
- Add `event_counts_by_rule: dict[str, int]` to `Sample`. Populate in `_snapshot` by iterating `state.events` (already exposed) and counting by `event.rule_name`.
- Extend `ShortWindowDetector` (or add a sibling) that takes a `gene_to_rules: dict[str, set[str]]` map and trips when all rules in a gene's set have zero events in KO but non-zero in WT.
- The gene-to-rules map is produced by walking the rules list during setup.

Expected lift: **from 1/10 to 6-8/10 essentials caught** (all the metabolic ones — roughly Breuer's FBA coverage). Ribosomal / tRNA-synthetase KOs are still missed because their rules are in the catalysis-free Gene-Expression sheet, not `catalysis:*`. MCC to ~0.45-0.55 at n=40-100.

### 2. Longer bio-time at Rust+parallel budget (MEDIUM leverage)

Now that effective wall is 1.9 s/gene (scale=0.05 + Rust + 4 workers), a **full 458-CDS sweep at t_end=5.0 s** fits in ~3 hours on this sandbox (vs >30 h in v0). At that window the transporter KOs (ptsG, crr) probably show. Worth running once #1 is in place.

### 3. Bigger calibration sample (cheap side-improvement)

Increasing `--calibrate` from 5 to 20-30 tightens the noise floors on `TOTAL_EVENTS` (from 30 % to <10 %) and on `TOTAL_COMPLEXES` / `BOUND_PROTEINS` (from 30-40 % to ~10 %). Makes the new Session-6 pools actually useful. Combine with #1 for additional recall.

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
