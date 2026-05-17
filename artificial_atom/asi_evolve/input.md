# Task: Improve the artificial_atom v3 molecular force-field surrogate

## Background

`MultiModePotentialV3` is a ~75k-parameter neural network that predicts molecular energies, forces, and pose-quality scores from atomic coordinates and chemistry features. It's trained on MMFF94-labeled drug-like molecules generated from RDKit.

The current demo (`artificial_atom.demo_v3`) achieves **8/8 checks passed** on the verification contract. Your goal is to **maintain 8/8 pass rate while reducing wall-clock time and/or improving the metric values**.

## Verification contract (these are the 8 checks)

| Check | Threshold | Direction |
|-------|-----------|-----------|
| Energy R² vs MMFF | > 0.85 | higher better |
| Force MAE | < 200 | lower better |
| Translation invariance | < 0.01 | lower better |
| Rotation invariance | < 0.01 | lower better |
| Newton 3rd law (sum forces = 0) | < 0.01 | lower better |
| Bad-pose rejection rate | > 0.95 | higher better |
| Near-native retention rate | > 0.85 | higher better |
| Ensemble OOD std / in-dist std | > 2.0 | higher better |

Baseline values (from the 8/8 PASS run):
- Energy R² = 0.9743
- Force MAE = 32.85
- Bad-pose rejection = 0.9792
- Near-native retention = 0.8962
- Ensemble OOD ratio = 7.35

## Score formula (lower is better)

```
composite = 10.0 * n_failed + 0.1 * (wall_time / 60.0)
```

- **Each failed check costs 10 points** — keep all 8 passing.
- **Wall time has mild pressure** (0.1 point per minute). Cutting train time helps marginally.

The system reports `eval_score = -composite` so ASI-Evolve maximizes it (higher score = lower composite = better).

## What you can change

The MUTABLE SECTION of `initial_program.py` exposes these knobs:

**Architecture** (affect capacity + speed):
- `D_H` (default 40): hidden dim; bigger = more capacity, slower
- `D_RBF` (default 20): radial basis function count
- `D_BOND_EXTRA` (default 8): extra bond-feature dim
- `N_LAYERS` (default 2): message-passing layers
- `R_CUTOFF` (default 6.0): non-bonded cutoff in Å
- `R_CUTOFF_ELECTRO` (default 8.0): electrostatic cutoff in Å

**Training** (affect convergence + speed):
- `EPOCHS_PRIMARY` (default 18), `EPOCHS_ENSEMBLE` (default 12)
- `LR_PRIMARY` (default 3e-4)
- `BATCH_SIZE` (default 8)
- `ENERGY_WEIGHT`, `FORCE_WEIGHT`, `POSE_WEIGHT`, `BAD_POSE_WEIGHT` (loss term weights)
- `GRAD_CLIP` (default 5.0)
- `SEED_PRIMARY`, `SEED_ENSEMBLE`

**Data split** (affect train/val/test sizes):
- `TRAIN_FRAC` (default 0.70), `VAL_FRAC` (default 0.15)

## Constraints

- **Must complete within 15 minutes** wall-clock (eval.sh timeout).
- **Must preserve the harness footer** — the `main()` function pipeline cannot change. Only modify the MUTABLE SECTION.
- **Must keep 8/8 passing**. Reducing epochs too aggressively will tank `Energy R²` below 0.85 (one failed check = 10 points penalty, likely outweighs any time saving).
- `LR_PRIMARY` must stay in `[1e-4, 1e-3]`. Outside this range the loss diverges.

## Promising directions

1. **Lower N_LAYERS from 2 to 1** if pass rate holds — halves train time.
2. **Increase BATCH_SIZE from 8 to 16 or 32** — better hardware utilization, possibly faster convergence.
3. **Tune FORCE_WEIGHT** — currently 0.02 (low because MMFF forces are O(100)); 0.05 might give better Force MAE without hurting energy.
4. **Reduce EPOCHS_PRIMARY from 18 to 12** if val loss plateaus by then — direct wall-clock win.
5. **Increase POSE_WEIGHT** — better classifier accuracy on bad-pose rejection and near-native retention.
6. **R_CUTOFF tuning** — wider cutoff = more atom neighbors per message = slower but potentially better physics.

## Output format

The candidate MUST print this exact block at the end of stdout:

```
===EVAL_METRICS===
{"checks_passed": 8, "checks_total": 8, "pass_rate": 1.0, "wall_time_sec": 421.3, "composite": 0.702, ...}
===END_EVAL_METRICS===
```

The harness footer already does this — don't modify it.
