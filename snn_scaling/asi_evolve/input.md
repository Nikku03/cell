# Task: Improve the snn_scaling reversal-learning benchmark

## Background

`snn_scaling` is a 7-trick spiking-neural-network architecture (vectorised LIF + sparse synapses + sparse activation + recurrent depth + reservoir + modular routing + memory + distillation + dendrites). The benchmark in `snn_scaling.asi_evolve.initial_program` tests three RL agents on a reversal task:

- **Naive Q-learning**: linear Q + TD updates over the reservoir's centred features
- **MemRec**: a single recency-weighted episodic memory (the prior reversal winner)
- **CLS**: complementary learning systems (fast hippocampus + slow cortical Q-table)

All three agents see identical reservoir features. The task is 5 stimulus classes × 3 actions, with the correct class→action mapping **reshuffled every 150 trials** (4 phases × 150 = 600 trials total). The agent must detect each shift from the reward stream alone.

**Goal**: maximise the BEST integrated agent's mean cumulative reward (whichever of MemRec, CLS, or EvolvedAgent wins), without letting the naive baseline beat it.

## Baseline values (3 seeds, N=2000, T=200, defaults with TASK_NOISE_STD=0.25)

| Agent | Mean cumulative reward (max +600) |
|---|---|
| Naive Q-learning | ~−40 |
| MemRec | ~+30 |
| CLS | ~+25 |
| EvolvedAgent (default) | ~+30 |

The defaults already include the confirmed `TASK_NOISE_STD=0.25` win (eval_score ≈ +28 from the first ASI-Evolve run). The job now is to push `best_integrated_cum` past ~+30 — most likely through a structural mutation of the `EvolvedAgent` class, since the numerical-knob landscape is mostly mapped.

## Score formula (lower is better)

```
composite = -best_integrated_cum + 0.1 * (wall_time / 60)
          + 200 * (1 if naive > best_integrated else 0)   # sanity penalty
```

ASI-Evolve sees `eval_score = -composite` (so higher score = better). Each +10 cumulative reward improvement = +10 score. Each minute of wall-time = −0.1 score.

## What you can change (the MUTABLE SECTION)

### Reservoir (affect feature richness + speed)
- `N_RESERVOIR` (default 2000): bigger = more features but O(N²) edges in build
- `P_RECURRENT` (default 0.10): connection density
- `G_EXC`, `G_INH` (default 0.04, 0.30): synaptic conductances
- `TAU_SYN` (default 5.0): synaptic time constant
- `RES_NOISE_STD` (default 0.2): per-neuron noise
- `EXC_FRACTION` (default 0.8): E/I balance

### Task / feature extraction
- `T_TIMESTEPS` (default 200): how long the reservoir sees each input
- `TASK_NOISE_STD` (default 0.25 — already optimized; do not re-tune): additive noise on inputs
- `N_PER_CLASS_POOL` (default 30): pool of distinct stimuli per class

### MemRec hyperparameters
- `MEMREC_TAU` (default 50.0): recency time constant; lower = faster forgetting
- `MEMREC_TOP_K` (default 10): k-NN retrieval breadth
- `MEMREC_EPS` (default 0.10): epsilon-greedy exploration

### CLS hyperparameters
- `CLS_HIPPO_TAU` (default 20.0): hippocampus recency (faster than MemRec)
- `CLS_HIPPO_TOP_K` (default 10), `CLS_HIPPO_CAPACITY` (default 200)
- `CLS_CORTEX_THRESHOLD` (default 0.6): cosine threshold for cluster joining
- `CLS_CORTEX_LR` (default 0.05): EMA learning rate for cortex Q-tables
- `CLS_ALPHA` (default 0.7): blend weight (alpha * hippo + (1-alpha) * cortex)
- `CLS_EPS` (default 0.10)

### Naive baseline
- `NAIVE_LR` (default 0.05), `NAIVE_EPS` (default 0.10)

### Structural canvases (the real "evolve the core" surface)
- `EvolvedAgent` class — a full mutable agent (4th competitor). Rewrite `.act()`/`.update()`/`__init__` to try new memory/learning algorithms.
- `evolved_feature_transform(traces)` — the feature extractor the EvolvedAgent sees. **Highest-leverage mutation**: the fixed (mean, std, range) representation capped every prior experiment. A better transform (temporal bins / FFT / derivatives) is the only path to a large gain.

## Constraints

- **Must complete within 15 minutes** wall-clock (eval.sh timeout).
- **Must preserve harness footer** — the `main()` pipeline and `===EVAL_METRICS===` block cannot change. Only modify the MUTABLE SECTION.
- **Naive must NOT outperform best integrated** (sanity penalty +200 if it does). Mutations that nuke MemRec/CLS while leaving Naive intact will be heavily penalised.
- `N_RESERVOIR` should stay in `[300, 3000]` (smaller = unstable features; larger = builds slowly).
- `T_TIMESTEPS` should stay in `[100, 400]`.
- All `EPS` values in `[0.02, 0.30]`.
- All `TAU` values in `[5, 200]`.

## Known calibration (from prior hand-tuning, in cognition seeds)

- The reservoir's centered (mean, std, range) features have weak class separability (centroid pairwise cosine ~0.51 after centering). This caps how good any architecture can do.
- `MEMREC_TAU=50` is the prior local optimum for pure reversal — the hand-tuned MemRec at this τ hit +91 cumulative over naive across 4/5 seeds at full scale (5 seeds, 600 trials).
- `CLS_HIPPO_TAU=20` is intentionally smaller than `MEMREC_TAU` because the cortex provides stability. Too fast (<10) and CLS gets noisy; too slow (>40) and CLS just becomes another MemRec.
- `CLS_ALPHA<0.5` makes stale cortex Q-values dominate, hurting reversal performance.
- `RES_NOISE_STD>0.5` makes reservoir features dominated by noise, hurting all architectures equally.
- `T_TIMESTEPS<100` doesn't let the reservoir settle.

## Promising directions

1. **Tune `MEMREC_TAU`** in the range [30, 80] — the prior best (50) might not be optimal for this exact reservoir.
2. **Lower `RES_NOISE_STD`** from 0.2 toward 0.1 — less reservoir noise might improve feature separability.
3. **`TASK_NOISE_STD` is already set to its confirmed optimum (0.25)** — do not re-tune it; lower values stack worse.
4. **Tune `CLS_ALPHA`** in [0.5, 0.9] to find the sweet spot between hippo (adaptive) and cortex (stable).
5. **Cut `T_TIMESTEPS`** from 200 toward 120 — most reservoir dynamics settle within ~5τ_m = 100ms, so trimming costs little but speeds up.
6. **Tune `CLS_CORTEX_THRESHOLD`** — too low → all stimuli collapse into one cluster; too high → no consolidation.

## Output format

The candidate MUST print this exact block at the end of stdout:

```
===EVAL_METRICS===
{"naive_cum_mean": ..., "memrec_cum_mean": ..., "cls_cum_mean": ..., "best_integrated_cum": ..., "wall_time_sec": ..., "composite": ..., ...}
===END_EVAL_METRICS===
```

The harness footer already does this — don't modify it.
