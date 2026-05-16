"""M7 self-improvement candidate program — ASI-Evolve target.

The ASI-Evolve Researcher mutates this file via diff blocks. Each
mutation produces a candidate that the Engineer runs; the resulting
metrics are scored by evaluator.py and stored in the trial database.

Mutable surface (the Researcher may change ANY of these, or rewrite
the entire fine-tune loop):
  - HYPERPARAMETERS section (lr, bias_strength, weights, etc.)
  - target_category filter logic
  - The fine_tune call's signature / behaviour
  - Add new loss terms by wrapping the fine_tune function
  - Replace the whole pipeline with a different approach

After running, the script MUST print a JSON metrics dict between the
markers '===EVAL_METRICS===' and '===END_EVAL_METRICS===' for the
evaluator to parse.
"""
import os
import sys
import json

import torch

# ============================================================
# MUTABLE HYPERPARAMETERS — Researcher may change these freely
# ============================================================
FINE_TUNE_STEPS  = 2000
BATCH_SIZE       = 256
LEARNING_RATE    = 1e-4
WEIGHT_DECAY     = 1e-4
BIAS_STRENGTH    = 0.70

# Per-row-type loss weights (count/flux/cum)
WEIGHT_COUNT     = 0.05
WEIGHT_FLUX      = 1.0
WEIGHT_CUM       = 0.5

# Restrict weakness-targeting to specific row-prefixes
# Options: 'all', 'count', 'flux', 'cum', 'transport', 'ribo'
TARGET_CATEGORY  = 'all'

# Skip first N timesteps (simulator init artifact)
T_SKIP_INITIAL   = 10

# ============================================================
# PIPELINE — Researcher may replace any part of this
# ============================================================
sys.path.insert(0, '/content/cell')

from cell_sim.lgnn.data.species_graph import load_species_graph
from cell_sim.lgnn.self_improve.iterate import (
    load_m7_with_state,
    analyze_model_weaknesses,
    fine_tune_on_weaknesses,
    compute_full_val_metrics,
)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DRIVE_DIR = os.environ.get(
    'M7_DRIVE_DIR', '/content/drive/MyDrive/cell_count_dynamics',
)
INITIAL_CKPT = os.environ.get('M7_INITIAL_CKPT', f'{DRIVE_DIR}/m7_7_best.pt')
SPECIES_GRAPH = os.environ.get('M7_SPECIES_GRAPH',
                                f'{DRIVE_DIR}/species_graph_v7.pt')
TRAIN_CACHE = os.environ.get('M7_TRAIN_CACHE', f'{DRIVE_DIR}/preload_train.pt')
VAL_CACHE   = os.environ.get('M7_VAL_CACHE',   f'{DRIVE_DIR}/preload_val.pt')
SIGMA_PATH  = os.environ.get('M7_SIGMA',
                              f'{DRIVE_DIR}/sigma_train_reps_1to49.pt')
OUTPUT_CKPT = os.environ.get('M7_OUTPUT_CKPT', '/tmp/m7_candidate.pt')


def filter_weakness_by_category(weakness, row_names, category):
    """Keep only worst-species indices whose names match the category."""
    if category in (None, 'all'):
        return weakness
    prefix_map = {
        'count':     ('G_', 'R_', 'P_', 'M_'),
        'flux':      ('F_',),
        'cum':       ('C_',),
        'transport': ('M_',),     # transport species (extracellular)
        'ribo':      ('P_', 'R_'),
    }
    prefixes = prefix_map.get(category, ())
    if not prefixes:
        return weakness
    new_idx = [
        i for i in weakness['worst_species_idx']
        if any(row_names[i].startswith(p) for p in prefixes)
    ]
    if new_idx:
        weakness['worst_species_idx'] = new_idx[:50]
    return weakness


def run_candidate():
    print(f'[m7-candidate] device={DEVICE}  initial={INITIAL_CKPT}')
    sg = load_species_graph(SPECIES_GRAPH)
    row_names = list(sg.row_names)
    train_data = torch.load(TRAIN_CACHE, map_location=DEVICE,
                             weights_only=False)
    val_data = torch.load(VAL_CACHE, map_location=DEVICE,
                           weights_only=False)
    sigma = torch.load(SIGMA_PATH, map_location=DEVICE,
                        weights_only=False)

    model, cfg, fidx = load_m7_with_state(
        INITIAL_CKPT, sg, row_names, DEVICE,
    )
    weakness = analyze_model_weaknesses(
        model, val_data, sigma, row_names, t_step_stride=100,
    )
    weakness = filter_weakness_by_category(weakness, row_names,
                                            TARGET_CATEGORY)
    model = fine_tune_on_weaknesses(
        model, train_data, sigma, val_data, row_names, fidx, weakness,
        n_steps=FINE_TUNE_STEPS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        bias_strength=BIAS_STRENGTH,
        weight_count=WEIGHT_COUNT,
        weight_flux=WEIGHT_FLUX,
        weight_cumulative=WEIGHT_CUM,
        log_every=500,
    )
    metrics = compute_full_val_metrics(
        model, val_data, sigma, row_names,
    )

    torch.save({
        'state_dict': model.state_dict(),
        'cfg': cfg,
        'architecture': 'M7_hybrid',
        'flux_indices': fidx.cpu().tolist(),
        'metrics': metrics,
    }, OUTPUT_CKPT)

    print('===EVAL_METRICS===')
    print(json.dumps(metrics))
    print('===END_EVAL_METRICS===')


if __name__ == '__main__':
    run_candidate()
