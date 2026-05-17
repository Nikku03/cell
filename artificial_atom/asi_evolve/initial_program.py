"""ASI-Evolve candidate program for artificial_atom v3.

Trains MultiModePotentialV3 + 2nd ensemble member, runs the 8-check
verification suite, and prints metrics. The ASI-Evolve Researcher
mutates THIS file via diff blocks (loose 4-10 angle brackets).

================================================================
== STRUCTURE: THREE sections                                   ==
==   1. HARNESS HEADER  - imports, paths. DO NOT MODIFY.       ==
==   2. MUTABLE SECTION - the Researcher's only target.        ==
==   3. HARNESS FOOTER  - pipeline + metrics. DO NOT MODIFY.   ==
==                                                              ==
== Stay between BEGIN/END MUTABLE markers. Don't import         ==
== anything other than what's in the header. Don't call the     ==
== demo_v3.main() function — that's a CLI tool.                ==
================================================================
"""
# ================================================================
# === HARNESS HEADER — DO NOT MODIFY ===
# ================================================================
import json
import sys
import time

sys.path.insert(0, '/content/cell')

import torch

from artificial_atom.mmff_dataset import (
    chem_dataset_stats,
    generate_chem_dataset,
)
from artificial_atom.multimode_v3 import MultiModePotentialV3, MultiModeV3Config
from artificial_atom.train_v3 import TrainV3Config, train_v3
from artificial_atom.verify_v3 import run_all_checks, format_results
# ================================================================
# === END HARNESS HEADER ===
# ================================================================


# ================================================================
# === BEGIN MUTABLE SECTION ===
# Researcher: this is the ONLY block you should change.
# ================================================================

# ---- Architecture (MultiModeV3Config) ----
D_H = 40
D_RBF = 20
D_BOND_EXTRA = 8
N_LAYERS = 2
R_CUTOFF = 6.0
R_CUTOFF_ELECTRO = 8.0

# ---- Training (TrainV3Config) - primary member ----
EPOCHS_PRIMARY = 18
LR_PRIMARY = 3e-4
BATCH_SIZE = 8
ENERGY_WEIGHT = 1.0
FORCE_WEIGHT = 0.02
POSE_WEIGHT = 0.5
BAD_POSE_WEIGHT = 4.0
GRAD_CLIP = 5.0

# ---- Training (TrainV3Config) - 2nd ensemble member ----
EPOCHS_ENSEMBLE = 12
SEED_PRIMARY = 0
SEED_ENSEMBLE = 13

# ---- Data split ----
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Notes (do NOT include these comment lines in SEARCH blocks):
# - EPOCHS_PRIMARY: 18 is the demo default; 8-30 is safe range
# - LR_PRIMARY: safe [1e-4, 1e-3]; outside risks divergence
# - D_H, D_RBF, N_LAYERS: bigger = more capacity but slower train
# - FORCE_WEIGHT: MMFF forces are O(100); too high dominates loss
# - POSE_WEIGHT: balances classifier loss against energy/force
# - BAD_POSE_WEIGHT: class-imbalance reweighting; 1.0 = no rebalance
# - R_CUTOFF / R_CUTOFF_ELECTRO: angstroms; affects which neighbors interact

# ================================================================
# === END MUTABLE SECTION ===
# ================================================================


# ================================================================
# === HARNESS FOOTER — DO NOT MODIFY ===
# ================================================================
def main():
    torch.manual_seed(0)
    t0 = time.time()

    # Build dataset
    print('[candidate] Building MMFF dataset...')
    all_data = generate_chem_dataset(seed=0)
    s = chem_dataset_stats(all_data)
    print(f'  {len(all_data)} samples, '
          f'unique_smiles={s["unique_smiles"]}, '
          f'near={s["n_near_native"]}, intermediate={s["n_intermediate"]}, '
          f'bad={s["n_bad"]}')

    # Split
    rng = torch.Generator().manual_seed(1)
    idx = torch.randperm(len(all_data), generator=rng).tolist()
    n_train = int(TRAIN_FRAC * len(all_data))
    n_val = int(VAL_FRAC * len(all_data))
    train_set = [all_data[i] for i in idx[:n_train]]
    val_set = [all_data[i] for i in idx[n_train:n_train + n_val]]
    test_set = [all_data[i] for i in idx[n_train + n_val:]]
    print(f'  split: train={len(train_set)} val={len(val_set)} test={len(test_set)}')

    # Build model
    cfg = MultiModeV3Config(
        d_h=D_H, d_rbf=D_RBF, d_bond_extra=D_BOND_EXTRA,
        n_layers=N_LAYERS,
        r_cutoff=R_CUTOFF, r_cutoff_electro=R_CUTOFF_ELECTRO,
    )
    model = MultiModePotentialV3(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  model: d_h={D_H} d_rbf={D_RBF} n_layers={N_LAYERS} '
          f'params={n_params:,}')

    # Train primary
    print(f'\n[candidate] Training primary ({EPOCHS_PRIMARY} epochs)...')
    train_cfg = TrainV3Config(
        epochs=EPOCHS_PRIMARY, lr=LR_PRIMARY, batch_size=BATCH_SIZE,
        energy_weight=ENERGY_WEIGHT, force_weight=FORCE_WEIGHT,
        pose_weight=POSE_WEIGHT, bad_pose_weight=BAD_POSE_WEIGHT,
        grad_clip=GRAD_CLIP, log_every=max(EPOCHS_PRIMARY // 4, 1),
    )
    train_v3(model, train_set, val_set, train_cfg, seed=SEED_PRIMARY)

    # Train ensemble member
    print(f'\n[candidate] Training ensemble member ({EPOCHS_ENSEMBLE} epochs)...')
    model_b = MultiModePotentialV3(cfg)
    torch.manual_seed(SEED_ENSEMBLE)
    train_cfg_b = TrainV3Config(
        epochs=EPOCHS_ENSEMBLE, lr=LR_PRIMARY, batch_size=BATCH_SIZE,
        energy_weight=ENERGY_WEIGHT, force_weight=FORCE_WEIGHT,
        pose_weight=POSE_WEIGHT, bad_pose_weight=BAD_POSE_WEIGHT,
        grad_clip=GRAD_CLIP, log_every=max(EPOCHS_ENSEMBLE // 4, 1),
    )
    train_v3(model_b, train_set, val_set, train_cfg_b, seed=SEED_ENSEMBLE)

    # Verify
    print('\n[candidate] Running 8-check verification...')
    results = run_all_checks(
        model, test_set,
        ensemble_members=[model, model_b],
        ood_pool=val_set,
    )
    print(format_results(results))
    n_passed = sum(r.passed for r in results)
    print(f'  -> {n_passed}/{len(results)} checks passed')

    # Compose metrics. Scoring favors:
    #   1. number of checks passed (most important)
    #   2. energy R^2 (higher is better)
    #   3. lower force MAE (lower is better)
    #   4. lower wall-clock (we add a mild time penalty)
    wall_time = time.time() - t0
    metrics = {
        'checks_passed': int(n_passed),
        'checks_total': len(results),
        'pass_rate': n_passed / len(results),
        'wall_time_sec': float(wall_time),
        'n_params': int(n_params),
    }
    for r in results:
        key = r.name.lower().replace(' ', '_').replace('-', '_')
        # collapse to ascii
        key = ''.join(c if c.isalnum() or c == '_' else '_' for c in key)[:60]
        metrics[f'check_{key}_value'] = float(r.value)
        metrics[f'check_{key}_passed'] = bool(r.passed)

    # Composite score: lower is better.
    #   Penalty for each failed check: 10.0
    #   Plus 0.1 * wall_time_min (mild time pressure)
    n_failed = len(results) - n_passed
    composite = 10.0 * n_failed + 0.1 * (wall_time / 60.0)
    metrics['composite'] = float(composite)

    print('\n===EVAL_METRICS===')
    print(json.dumps(metrics))
    print('===END_EVAL_METRICS===')


if __name__ == '__main__':
    main()
# ================================================================
# === END HARNESS FOOTER ===
# ================================================================
