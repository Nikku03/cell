# NEXT_SESSION — queued work

_Read this file FIRST, immediately after running the invariant checker._

## Pre-flight (every session)

1. `python memory_bank/.invariants/check.py` — must print `OK`.
2. Read `PROJECT_STATUS.md` — confirm Session 7 state is current.
3. Read `memory_bank/concepts/essentiality/REPORT.md` for the MCC history.
4. Read `memory_bank/facts/measured/mcc_against_breuer_v5.json` for the latest honest result.
5. Read this file.

## Where Layer 6 stands (end of Session 7)

**Infrastructure (cumulative v0 → v5):**
- `RealSimulator` (Python + Rust via `--use-rust`) wraps the existing `FastEventSimulator` stack.
- Two detectors shipped:
  - `ShortWindowDetector` — bidirectional pool-deviation with calibration.
  - `PerRuleDetector` — direct causal, watches per-rule event counts.
- `scripts/run_sweep_parallel.py` — 4-worker multiprocess; `--use-rust`; `--detector {short-window|per-rule}`; `--calibrate K`.
- Sample dataclass carries `event_counts_by_rule` alongside the pool dict.
- ~7× speedup over v0 baseline (1.7-1.9 s/gene effective).

**MCC measurements** (`facts/measured/mcc_against_breuer_v0..v5.json`):

| Version | n | Detector | MCC | Notes |
|---|---|---|---|---|
| v0 | 4 | ShortWindow | 0.333 | pgi via F6P. |
| v1 | 40 | ShortWindow+cal10 | 0.160 | 1 TP. |
| v2 | 20 | ShortWindow scale=0.1 | 0.229 | 1 TP. |
| v3 | 20 | ShortWindow scale=0.25+rust | 0.229 | 1 TP. |
| v4 | 20 | ShortWindow +non-metabolic +TOTAL_EVENTS | 0.229 | 1 TP. |
| v5 | 40 | **PerRule** | **0.125** | 5 TP, 3 FP, 17 TN, 15 FN. |
| v5 ref | 4 | PerRule | 0.577 | 2 TP / 1 TN / 1 FN; sample-size noise. |

**Two architectural walls, now both measured:**
1. ShortWindowDetector ceiling: only pgi-class central-glycolysis KOs perturb pools in ≤0.5 s at any tractable scale. 1 TP max.
2. PerRuleDetector ceiling: catches 5 metabolic essentials but (a) false-positives on Breuer-nonessential catalytic genes whose apparent essentiality in the simulator comes from the simulator's lack of biological pathway redundancy, and (b) architecturally cannot catch non-catalytic essentials (ribosomal proteins, tRNA synthetases, replication machinery).

**Brief target: MCC > 0.59.** Not reached. The diagnosis is now **two-part**: short-window pool ceiling AND per-rule biological-redundancy mismatch.

## Session-8 queue (in execution order)

### 1. Combine the two detectors into a high-precision ensemble

Run both detectors on each KO. Only flag a gene as essential when **both** fire OR when one fires with high confidence AND the other isn't in refusal state. A simple "AND" rule would drop most of v5's FPs (the nonessential catalytic genes) because those don't perturb pool levels; it would also keep the TP set (pgi, tpiA, PGM, etc. all perturb F6P / G6P / PYR in addition to silencing rules).

Implementation sketch:
- New `cell_sim/layer6_essentiality/ensemble_detector.py` with `EnsembleDetector(short_window, per_rule)`.
- `detect_for_gene(locus_tag, ko)` — return `(mode, t, conf, evidence)` from whichever detector agrees with the other. Policy to decide (start with AND, measure, then try OR-of-high-confidence).
- Add `--detector ensemble` to `run_sweep_parallel.py`.

Expected: drop v5's 3 FPs, keep the 5 TPs (since the TPs also perturb pools a bit — confirm). MCC should rise from 0.125 to ~0.3–0.4 on n=40 balanced. Still below 0.59 because non-catalytic essentials stay missed.

Honest cap: the ceiling for ANY short-window detector family is bounded by what the simulator exposes in ≤0.5 s. 0.59 needs either longer bio-time (#2) or simulator biology upgrades.

### 2. Longer bio-time run with the Rust+parallel stack (Path A from REPORT)

At 1.7 s/gene effective, a full 458-CDS sweep at t_end=5.0 s is ~3 h wall (with Rust + 4 workers). At that window, transporter and translation KOs may start showing metabolic signatures.

Run both detectors at t_end=5.0 s on a n=40 balanced panel first (~8 min), check whether the signal strengthens, then commit compute for the full sweep if it does. Record as v6.

### 3. Gene-to-rule weighting by necessity

For each rule, precompute the set of genes that catalyse it. Rules with only 1 gene are "unique"; rules with >1 gene are "redundant". `PerRuleDetector` could weight silencing by uniqueness: a gene that silences only rules with ≥2 catalysers should NOT trip, because in reality another gene covers that reaction. This directly addresses v5's FP mechanism (Breuer's pathway-redundancy labels). Should drop FPs without code elsewhere.

## Deferred (not this session)

- **Layer 5 (biomass + division)** — still not the current bottleneck.
- **Hutchison 2016 secondary labels** — secondary validation.
- **Multi-gene knockouts / synthetic lethality** — brief is single-gene only.
- **Neural-net anything** — brief section 2.

## Done-elsewhere (do NOT re-do)

- `cell_sim_rust` extension: built in Session 6, integrated via `--use-rust`.
- `PerRuleDetector` + gene-to-rules map: shipped in Session 7.
- Session-3/4/5/6 items that were stale in prior NEXT_SESSIONs.

## Git state

Session 1–7 commits on `origin/claude/syn3a-whole-cell-simulator-REjHC`. PAT still in `/tmp/.gh_tkn` for pushing from the sandbox; revoke when the work is handed off.

## How to run the planned v6 sweep (after ensemble ships)

```bash
# Ensemble on balanced n=40 (~2 min wall at short window):
python scripts/run_sweep_parallel.py \
    --max-genes 40 --balanced \
    --calibrate 10 \
    --workers 4 --use-rust \
    --scale 0.05 --t-end-s 0.5 \
    --detector ensemble

# Then compare: ensemble vs per-rule vs short-window on the same panel.
```
