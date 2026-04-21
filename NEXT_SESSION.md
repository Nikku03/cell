# NEXT_SESSION — queued work

_Read this file FIRST, immediately after running the invariant checker._

## Pre-flight (every session)

1. `python memory_bank/.invariants/check.py` — must print `OK`.
2. Read `PROJECT_STATUS.md` — confirm Session 8 state is current.
3. Read `memory_bank/concepts/essentiality/REPORT.md` for the MCC history + updated paths.
4. Read `memory_bank/facts/measured/mcc_against_breuer_v6.json` for the latest honest result.
5. Read this file.

## Where Layer 6 stands (end of Session 8)

**Infrastructure (v0 → v6 cumulative):**
- `RealSimulator` (Python + Rust via `--use-rust`) wraps the existing `FastEventSimulator`.
- Three detectors shipped:
  - `ShortWindowDetector` — bidirectional pool deviation with calibration.
  - `PerRuleDetector` — direct causal, watches per-rule event counts.
  - `EnsembleDetector` — composes both with 3 policies (AND, OR_HIGH_CONFIDENCE, PER_RULE_WITH_POOL_CONFIRM).
- `scripts/run_sweep_parallel.py` — 4-worker multiprocess, `--use-rust`, `--detector {short-window|per-rule|ensemble}`, `--ensemble-policy`, `--min-confidence`, `--min-pool-dev`, `--rule-necessity-only`, `--calibrate K`.
- Gene-to-rules inversion (`invert_to_rule_catalysers`) and rule-necessity filter (`unique_rules_per_gene`) in `gene_rule_map.py`.

**MCC measurements** (`facts/measured/mcc_against_breuer_v0..v6.json`):

| Version | n | Detector / config | MCC |
|---|---|---|---|
| v0 | 4 | ShortWindow | **0.333** (best) |
| v1 | 40 | ShortWindow + cal10 | 0.160 |
| v2 | 20 | ShortWindow scale=0.10 | 0.229 |
| v3 | 20 | ShortWindow scale=0.25, rust | 0.229 |
| v4 | 20 | ShortWindow + non-metabolic + TOTAL_EVENTS | 0.229 |
| v5 | 40 | PerRule | 0.125 |
| v6a | 40 | Ensemble per_rule_with_pool_confirm | 0.125 |
| v6b | 40 | Ensemble AND + rule-necessity-only | 0.160 |

**Three architectural walls, all measured:**
1. ShortWindow ceiling: only pgi-class central glycolysis visible at ≤0.5 s.
2. PerRule FP floor: Breuer-nonessential catalytic genes (0034, lpdA, deoC) have their rules go silent in KO — biologically they're nonessential because of pathway redundancy the simulator doesn't model.
3. Ensemble-fusion cannot fix (2): TP and FP catalytic KOs produce overlapping pool deviations at short windows.

**Session 8 conclusion**: detector-side optimisation at 0.5 s bio-time is exhausted. Path to MCC > 0.59 requires longer bio-time runs (Path A).

## Session-9 queue (in execution order)

### 1. Longer-window reference panel — confirm the signal strengthens

Before committing compute for a full sweep, measure whether t_end=5.0 s actually produces stronger discrimination on the 4-gene panel (pgi, ptsG, ftsZ, 0305). Expected wall: 4 genes × ~2 min = 8 min serial (single WT + 4 KO). With Rust + 4-worker, ~3 min parallel.

```bash
python scripts/run_sweep_parallel.py \
    --reference-panel \
    --workers 4 --use-rust \
    --scale 0.05 --t-end-s 5.0 \
    --detector ensemble --ensemble-policy per_rule_with_pool_confirm \
    --min-pool-dev 0.05 \
    --out-dir outputs
```

**Go/no-go criterion**: if `max_pool_dev` on the 3 non-essentials drops below 0.05 while on the essentials stays above 0.05, the longer window is working. If it doesn't separate, don't commit the full sweep — the simulator's pool dynamics may saturate even at 5 s.

### 2. IF #1 succeeds: balanced n=40 at t_end=5.0 s

```bash
python scripts/run_sweep_parallel.py \
    --max-genes 40 --balanced \
    --calibrate 10 \
    --workers 4 --use-rust \
    --scale 0.05 --t-end-s 5.0 \
    --detector ensemble --ensemble-policy per_rule_with_pool_confirm \
    --min-pool-dev 0.05 \
    --out-dir outputs
```

Wall budget: ~25 min (51 KOs × ~30 s each / 4 workers). Record as v7 fact. **Honest prediction**: MCC 0.2–0.35 — better than v6 because some transporter KOs (ptsG, crr) become visible, but still below 0.59 because non-catalytic essentials (ribosomal proteins) probably remain invisible until Layer 1/2 get proper translation dynamics.

### 3. IF #1 fails (pool saturation at 5 s): try scale=0.25

Same structure at scale=0.25 t_end=1.0 s. Higher copy numbers reduce stochastic noise on nonessentials, so min_pool_dev can be tightened. Compute cost: ~10x the scale=0.05 baseline.

### 4. Layers 1/2 citations (ground work for future MCC lift)

Independent of the MCC work, start putting memory-bank citations on the Layer 1/2 parameters already in `cell_sim/layer3_reactions/gene_expression.py` + the tRNA charging sheet. These are prerequisites if we eventually need proper ribosome/translation dynamics to catch the non-catalytic essentials.

## Deferred (not for session 9)

- Layer 5 (biomass + division).
- Hutchison 2016 secondary labels.
- Multi-gene / synthetic-lethal knockouts.
- Neural-net anything.

## Done-elsewhere (do NOT re-do)

- `cell_sim_rust` extension (Session 6).
- `PerRuleDetector` + gene-to-rules map (Session 7).
- `EnsembleDetector` + `unique_rules_per_gene` + `--rule-necessity-only` (Session 8).

## Git state

Session 1–8 commits on `origin/claude/syn3a-whole-cell-simulator-REjHC`. PAT in `/tmp/.gh_tkn`; revoke when work concludes.
