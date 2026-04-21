# NEXT_SESSION — queued work

_Read this file FIRST, immediately after running the invariant checker._

## Pre-flight

1. `python memory_bank/.invariants/check.py` — must print `OK`.
2. Read `PROJECT_STATUS.md`.
3. Read `memory_bank/concepts/essentiality/REPORT.md` for the full MCC history (v0–v7).
4. Read `memory_bank/facts/measured/mcc_against_breuer_v7.json` for the latest honest result + the Path-A falsification.
5. Read this file.

## Where Layer 6 stands (end of Session 9)

**MCC history (final, after the detector + bio-time exploration):**

| Version | Detector | t_end | MCC | Note |
|---|---|---|---|---|
| v0 | ShortWindow | 0.5 | **0.333** (best balanced-panel MCC ever) | n=4; pgi. |
| v1–v4 | ShortWindow variants | 0.5 | 0.160 – 0.229 | pgi only; ceiling. |
| v5 | PerRule | 0.5 | 0.125 | 5 TP + 3 FP on catalytic nonessentials. |
| v6a/b | Ensemble | 0.5 | 0.125 / 0.160 | Composition can't separate FP/TP. |
| v7-ref | Ensemble pool_confirm | **5.0** | 0.577 | n=4 only; sample-size optimism. |
| v7 | Ensemble pool_confirm | **5.0** | **0.000** | n=20 balanced. Path A falsified. |

**Status of the brief target (MCC > 0.59):** unreached. The detector-side and bio-time-side levers are both now measured and exhausted. Further MCC improvement requires **simulator-biology upgrades**, not detector composition.

**Why v7 falsified Path A:** at t_end=5.0 s the simulator's pool deviations grow for BOTH TPs and FPs. Transport-KO 0034's `max_pool_dev` rose from ~1.0 at 0.5 s to **13.3** at 5.0 s because the upstream metabolite accumulates unboundedly in the simulator — a real cell would either consume it via alternate enzymes or cap it via diffusion equilibrium, neither of which the simulator models.

## Session-10 queue

Detector + bio-time work is **done**. The remaining options are deeper:

### 1. Layers 1/2 — citation + parameter audit (PREREQUISITE for any MCC lift)

Before any simulator-biology work, establish the memory-bank citation trail on what already exists. Walk `cell_sim/layer3_reactions/gene_expression.py` + the `tRNA Charging` and `Gene Expression` sheets in `kinetic_params.xlsx`; convert each parameter into a `facts/parameters/*.json` with the Thornburg 2022 or Breuer 2019 citation it came from.

Why first: any change to translation dynamics needs to know which parameter values are measured vs. lumped vs. estimated. Currently those distinctions aren't recorded.

### 2. Pathway-redundancy annotation in the rule set

For each catalysis rule in `build_reversible_catalysis_rules`, annotate (or compute from SBML gene-association) the full set of alternate-enzyme genes. Emit a `rule_to_alternates: dict[str, set[str]]` map alongside the existing `gene_to_rules`. Then modify `PerRuleDetector` to only trip on a gene's rule if (a) the rule goes silent AND (b) no alternate-enzyme gene is present and active. This addresses the v5/v6 FP mechanism structurally.

Orthogonal to #3; can combine.

### 3. Explicit sink for accumulating pools

Add a configurable drain/sink term to `_build_state_and_rules` that caps the upper bound on any pool at `k_sink * (pool - medium_equilibrium)`. Tuned against the non-essential calibration run. This addresses the v7 FP mechanism (transporter-KO metabolite blow-up) without requiring a full pathway-redundancy model.

Less principled than #2 but much cheaper. Pairs with #2.

### 4. Secondary ground-truth: Hutchison 2016 transposon labels

Register Hutchison 2016's transposon-derived essentiality classes as a second labels source. Report MCC against both; flag genes where the two sources disagree. Does not lift MCC against Breuer but gives an orthogonal validation signal for simulator-biology work.

### Not recommended

- More detector variants at the current simulator fidelity. Eight measurements across five detector families hit the same ceiling. More variants are unlikely to help and risk theatre.
- A full 458-CDS sweep at t_end=5.0 s. v7's n=20 result makes the 3-hour compute commitment unjustified.
- Pivoting the project. Brief section 11 forbids it; diagnosis is solid; the remaining work is simulator-biology, not a rework.

## Deferred (not for session 10)

- Layer 5 (biomass + division).
- Multi-gene / synthetic-lethal knockouts.
- Neural-net anything.
- Rust-side simulator changes (Python side has the needed surface area).

## Done-elsewhere (do NOT re-do)

- `cell_sim_rust` extension (Session 6).
- `PerRuleDetector` + gene-to-rules map (Session 7).
- `EnsembleDetector` + `unique_rules_per_gene` + `--rule-necessity-only` (Session 8).
- Longer-window sweeps at scale=0.05 (Session 9 — falsified).

## Git state

Session 1–9 commits on `origin/claude/syn3a-whole-cell-simulator-REjHC`. PAT at `/tmp/.gh_tkn` for pushing from the sandbox. Revoke when done.
