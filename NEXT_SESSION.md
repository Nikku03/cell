# NEXT_SESSION — queued work

_Read this file FIRST, immediately after running the invariant checker._

## Pre-flight (every session)

1. `python memory_bank/.invariants/check.py` — must print `OK`.
2. Read `PROJECT_STATUS.md` — confirm Session 6 state is current.
3. Read `memory_bank/concepts/essentiality/REPORT.md` for the MCC history and diagnosis.
4. Read `memory_bank/facts/measured/mcc_against_breuer_v4.json` for the latest measurement.
5. Read this file.

## Where Layer 6 stands (summary)

**Infrastructure, v0 → v4 cumulative:**
- `RealSimulator` (Python + Rust via `--use-rust`) wraps the existing `FastEventSimulator` stack behind the `Simulator` Protocol.
- `ShortWindowDetector` with bidirectional `|ko/wt - 1|` + two-consecutive confirmation + per-pool threshold support.
- `scripts/run_sweep_parallel.py` — 4-worker multiprocess; `--calibrate K` per-pool noise-floor calibration; `--use-rust`.
- **7× speedup over v0** (1.9 s/gene effective at scale=0.05 + Rust + 4-worker).
- Pool set (17): 12 metabolites + `TOTAL_COMPLEXES`, `FOLDED_PROTEINS`, `UNFOLDED_PROTEINS`, `FOLDED_FRACTION`, `BOUND_PROTEINS`, `TOTAL_EVENTS`.

**MCC measurements** (facts/measured/mcc_against_breuer_v0..v4.json):

| Version | n | Config | MCC |
|---|---|---|---|
| v0 | 4 | scale=0.05, t_end=0.5 | **0.333** (best) |
| v1 | 40 | scale=0.05, t_end=0.5, cal=10 | 0.160 |
| v2 | 20 | scale=0.10, t_end=1.0, cal=5 | 0.229 |
| v3 | 20 | scale=0.25, Rust, cal=5 | 0.229 |
| v4 | 20 | scale=0.05, Rust, +non-met pools, +TOTAL_EVENTS, cal=5 | 0.229 |

**Diagnosis (settled across v0–v4):** MCC is invariant across scale, t_end, pool set, threshold, and sample size. The detector trips on exactly one gene — `pgi` — via F6P depletion. Pool-based short-window detection has hit an architectural ceiling, not a tuning one.

Brief target: **MCC > 0.59**. Unreachable at these short windows with the current detector family.

## Session-7 queue (in execution order)

### 1. Per-rule event-count detection (THIS session's main deliverable)

Session 6 proved pool-based detection is exhausted. The next signal to try is **how often each catalysis rule fires in KO vs WT** — a direct causal signal that doesn't depend on downstream bio-time propagation.

Implementation:
- Add `event_counts_by_rule: dict[str, int]` to `Sample` in `cell_sim/layer6_essentiality/harness.py`. Populate in `RealSimulator._snapshot` by counting `state.events` by `event.rule_name`.
- New module `cell_sim/layer6_essentiality/gene_rule_map.py`: given a list of rule objects produced by the existing `build_reversible_catalysis_rules`, return `dict[locus_tag, set[rule_name]]`. Inspect `rule.enzyme_loci` (or whatever attribute the existing code uses) to extract the mapping.
- New module `cell_sim/layer6_essentiality/per_rule_detector.py`: `PerRuleDetector(wt, gene_to_rules, min_wt_events=20)`. Trips when every rule in `gene_to_rules[locus_tag]` has ≥`min_wt_events` events in WT but 0 in KO. Returns `(FailureMode.CATALYSIS_SILENCED, t_first, confidence, evidence)`.
- Add `FailureMode.CATALYSIS_SILENCED` to the enum in `harness.py`.
- Add `--detector per-rule` flag to `scripts/run_sweep_parallel.py`.

Honest scope:
- This detector is tautologically true for direct KOs: remove the enzyme, its rule stops firing. The *value* of the detector is getting this signal uniformly across all catalytic genes without per-pool threshold tuning.
- **Catches**: ~50-80 Syn3A genes whose product has a measured k_cat in `kinetic_params.xlsx`. These are the metabolic enzymes.
- **Misses**: ribosomal proteins, tRNA synthetases, replication genes — not in `catalysis:*` rules at all. Their `gene_to_rules` set is empty, so the detector returns `NONE`.
- **Predicted MCC**: 0.35–0.45 on a balanced n=40 panel. If measured > 0.45, audit for false positives before reporting.

Measurement protocol:

```bash
python scripts/run_sweep_parallel.py \
    --max-genes 40 --balanced \
    --calibrate 10 \
    --workers 4 --use-rust \
    --scale 0.05 --t-end-s 0.5 \
    --detector per-rule
```

Record the result as `memory_bank/facts/measured/mcc_against_breuer_v5.json` with:
- Full confusion matrix.
- `gene_to_rules` map size (genes with ≥1 rule; avg rules per gene).
- How many of the caught essentials were pgi-class (also caught by v4) vs new catches.
- Honest caveats listing what this detector cannot catch and why.

Do NOT claim this hits > 0.59. It won't. The eventual path to 0.59 is Path A in the REPORT (longer bio-time runs), which is a compute commitment the user will make later.

### 2. Longer bio-time full sweep (MEDIUM leverage, compute-committed)

After #1 is landed, a 458-gene sweep at t_end=5.0 s in Rust + 4-worker parallel fits in ~3 hours on the sandbox. At that window, transporter KOs (ptsG, crr) should begin to show metabolite signatures. Worth running once #1 establishes a baseline.

### 3. Bigger calibration sample

`TOTAL_EVENTS` and non-metabolic pool noise floors shrink with more calibration samples (30% at K=5 → <10% at K=20-30). Cheap at Rust+parallel speeds.

## Deferred (not this session)

- **Layer 5 (biomass + division)** — not the current bottleneck; the short-window detector plateau is the bottleneck.
- **Hutchison 2016 secondary essentiality labels** — secondary validation; adds comparison but doesn't lift the MCC ceiling.
- **Multi-gene knockouts / synthetic lethality** — brief is explicitly single-gene only.
- **Neural-net anything** — forbidden by brief section 2 without justification.

## Done-elsewhere items (do NOT re-do)

- `cell_sim_rust` extension: built and installed in Session 6. Wheel in `cell_sim_rust/target/wheels/`. Integrated via `RealSimulatorConfig.use_rust_backend=True`.
- `torch` import in `cell_sim/layer3_reactions/network.py`: still present. Not on the Layer-6 critical path; leave for a cleanup session. (Logged as deferred rather than queued to stop it appearing in every NEXT_SESSION.)

## Git state

All Session 1-6 commits on `origin/claude/syn3a-whole-cell-simulator-REjHC` (HEAD = `dbc1e07`). Local `origin` URL points at a dead local-proxy port; push uses the stashed PAT at `/tmp/.gh_tkn` (gitignored, chmod 600). Token must be revoked when work concludes.

## How to run the planned v5 sweep

```bash
# After implementing the per-rule detector + --detector flag:
python scripts/run_sweep_parallel.py \
    --max-genes 40 --balanced \
    --calibrate 10 \
    --workers 4 --use-rust \
    --scale 0.05 --t-end-s 0.5 \
    --detector per-rule \
    --out-dir outputs
```

Expected wall: ~10 min. Check `outputs/metrics_parallel_*_per-rule*.json` for the MCC.
