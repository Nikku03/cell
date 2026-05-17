"""M7 self-improvement candidate program — ASI-Evolve target.

The ASI-Evolve Researcher mutates THIS file via diff blocks. Each
mutation produces a candidate that the Engineer runs; results are
scored on three axes (accuracy + speed + memory) by full_benchmark.py.

Mutable surface — anything in this file is fair game, but the
following high-leverage knobs and hooks are explicitly designed to be
mutated:

  HYPERPARAMETERS section
    FINE_TUNE_STEPS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    BIAS_STRENGTH, WEIGHT_COUNT, WEIGHT_FLUX, WEIGHT_CUM,
    TARGET_CATEGORY, T_SKIP_INITIAL

  INFERENCE-SPEED TOGGLES
    USE_TORCH_COMPILE, INFERENCE_DTYPE, INFERENCE_DISABLE_CHECKPOINT
    (set these to push speed_ratio down without hurting accuracy)

  HOOKS the Researcher may rewrite to inject new behaviour
    patch_model(model)          - monkey-patch model attrs/params
    custom_loss_addon(...)      - add a regularization or aux loss
    filter_weakness_by_category - choose which rows to focus training on

After running, the script MUST print metrics between the markers
'===EVAL_METRICS===' and '===END_EVAL_METRICS===' for evaluator.py
to parse.
"""
import json
import os
import sys

import torch

# ============================================================
# MUTABLE HYPERPARAMETERS
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
# INFERENCE-SPEED TOGGLES (Researcher may flip these for speed_ratio)
# ============================================================
# Wrap model with torch.compile before benchmark. Adds ~30s compile
# delay but typically gives 1.5-2x speedup on subsequent forward calls.
USE_TORCH_COMPILE         = False

# Cast model to bf16 for inference benchmark. Roughly halves memory
# traffic per forward pass; minor accuracy delta (~0.001 on val).
# Options: 'fp32' (default), 'bf16', 'fp16'
INFERENCE_DTYPE           = 'fp32'

# Disable gradient checkpointing during inference (it's overhead with
# no backward pass). May not work if checkpoint is hardcoded in model.
INFERENCE_DISABLE_CHECKPOINT = False


# ============================================================
# HOOK: patch_model
# Researcher may monkey-patch model attributes or replace submodules
# in this function. Called once after model load, before benchmarks.
# ============================================================
def patch_model(model):
    """Apply architectural tweaks. Default: no-op. Examples:
      model.rate_clip = 8.0                     # widen rate clip
      model.use_checkpoint = False              # save memory at infer
      model.pinn_head.rate_clip = 5.0           # tighter rate cap
    """
    return model


# ============================================================
# HOOK: custom_loss_addon
# Researcher may add a regularizer or auxiliary loss term here.
# Called inside the fine-tune loop; returned scalar is added to the
# primary loss before backward(). Default: returns 0.
# ============================================================
def custom_loss_addon(x_pred, x_actual, v_log, model):
    """Return a scalar Tensor (or 0) to add to the per-step loss.

    Examples:
      return 0.01 * v_log.abs().mean()                # rate magnitude reg
      return 0.05 * (x_pred[:, sbml_mask] - x_actual[:, sbml_mask]).pow(2).mean()
    """
    return torch.zeros((), device=x_pred.device, dtype=x_pred.dtype)


# ============================================================
# HOOK: filter_weakness_by_category
# Decide which subset of weak species the fine-tune should focus on.
# Researcher may replace the prefix_map or add new criteria.
# ============================================================
def filter_weakness_by_category(weakness, row_names, category):
    if category in (None, 'all'):
        return weakness
    prefix_map = {
        'count':     ('G_', 'R_', 'P_', 'M_'),
        'flux':      ('F_',),
        'cum':       ('C_',),
        'transport': ('M_',),       # extracellular metabolite proxy
        'ribo':      ('P_', 'R_'),  # ribosomal protein / mRNA
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


# ============================================================
# PIPELINE (Researcher may rewrite parts of this if needed)
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
DRIVE_DIR     = os.environ.get('M7_DRIVE_DIR',
                                '/content/drive/MyDrive/cell_count_dynamics')
INITIAL_CKPT  = os.environ.get('M7_INITIAL_CKPT', f'{DRIVE_DIR}/m7_7_best.pt')
SPECIES_GRAPH = os.environ.get('M7_SPECIES_GRAPH',
                                f'{DRIVE_DIR}/species_graph_v7.pt')
TRAIN_CACHE   = os.environ.get('M7_TRAIN_CACHE', f'{DRIVE_DIR}/preload_train.pt')
VAL_CACHE     = os.environ.get('M7_VAL_CACHE',   f'{DRIVE_DIR}/preload_val.pt')
SIGMA_PATH    = os.environ.get('M7_SIGMA',
                                f'{DRIVE_DIR}/sigma_train_reps_1to49.pt')
OUTPUT_CKPT   = os.environ.get('M7_OUTPUT_CKPT', '/tmp/m7_candidate.pt')


def apply_inference_optimizations(model):
    """Apply the speed/dtype toggles set at the top of this file."""
    if INFERENCE_DISABLE_CHECKPOINT and hasattr(model, 'use_checkpoint'):
        model.use_checkpoint = False
    if INFERENCE_DTYPE == 'bf16':
        model = model.to(torch.bfloat16)
    elif INFERENCE_DTYPE == 'fp16':
        model = model.to(torch.float16)
    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model, mode='reduce-overhead',
                                   dynamic=False)
        except Exception as e:
            print(f'[m7-candidate] torch.compile failed: {e}; continuing')
    return model


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
    model = patch_model(model)

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

    # Apply inference-time optimizations BEFORE benchmark so the speed
    # measurement reflects them.
    model = apply_inference_optimizations(model)

    metrics = compute_full_val_metrics(
        model, val_data, sigma, row_names,
    )

    torch.save({
        'state_dict': (model._orig_mod.state_dict()
                       if hasattr(model, '_orig_mod')
                       else model.state_dict()),
        'cfg': cfg,
        'architecture': 'M7_hybrid',
        'flux_indices': fidx.cpu().tolist(),
        'metrics': metrics,
        'mutated_params': {
            'FINE_TUNE_STEPS': FINE_TUNE_STEPS, 'BATCH_SIZE': BATCH_SIZE,
            'LEARNING_RATE': LEARNING_RATE, 'BIAS_STRENGTH': BIAS_STRENGTH,
            'WEIGHT_COUNT': WEIGHT_COUNT, 'WEIGHT_FLUX': WEIGHT_FLUX,
            'WEIGHT_CUM': WEIGHT_CUM, 'TARGET_CATEGORY': TARGET_CATEGORY,
            'USE_TORCH_COMPILE': USE_TORCH_COMPILE,
            'INFERENCE_DTYPE': INFERENCE_DTYPE,
        },
    }, OUTPUT_CKPT)

    print('===EVAL_METRICS===')
    print(json.dumps(metrics))
    print('===END_EVAL_METRICS===')


if __name__ == '__main__':
    run_candidate()
