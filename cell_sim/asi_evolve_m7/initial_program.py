"""M7 self-improvement candidate program — ASI-Evolve target.

================================================================
== STRUCTURE: this file has THREE sections.                    ==
==                                                              ==
==   1. HARNESS HEADER   - imports, paths. DO NOT MODIFY.       ==
==   2. MUTABLE SECTION  - the Researcher's only target.        ==
==   3. HARNESS FOOTER   - main() pipeline. DO NOT MODIFY.      ==
==                                                              ==
== Mutations must stay between the BEGIN/END MUTABLE markers.   ==
== Do NOT call iterate.main() or any argparse-based entry       ==
== point — that script is a CLI tool, not a library to invoke.  ==
== Only the specific functions imported in the header are safe. ==
================================================================
"""
# ================================================================
# === HARNESS HEADER — DO NOT MODIFY ===
# ================================================================
import json
import os
import sys

sys.path.insert(0, '/content/cell')

import torch

from cell_sim.lgnn.data.species_graph import load_species_graph
from cell_sim.lgnn.self_improve.iterate import (
    load_m7_with_state,
    analyze_model_weaknesses,
    fine_tune_on_weaknesses,
    compute_full_val_metrics,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DRIVE_DIR = '/content/drive/MyDrive/cell_count_dynamics'
INITIAL_CKPT = f'{DRIVE_DIR}/m7_7_best.pt'
OUTPUT_CKPT  = '/tmp/m7_candidate.pt'
# ================================================================
# === END HARNESS HEADER ===
# ================================================================


# ================================================================
# === BEGIN MUTABLE SECTION ===
# Researcher: this is the ONLY block you should change.
# ================================================================

# ---- Hyperparameters (safe to tune) ----
FINE_TUNE_STEPS  = 500     # default minimum. LLM may raise to up to 5000.
                            # 2000 was too slow on A100 40GB fp32 fine-tune
                            # (>15 min per candidate; eval.sh times out at 900s).
BATCH_SIZE       = 64      # 256 OOMs on A100 40GB (fp32 fine-tune + 12GB preload).
                            # Safe range: 32-128 on A100 40GB, up to 256 on 80GB+.
LEARNING_RATE    = 1e-4
BIAS_STRENGTH    = 0.70
WEIGHT_COUNT     = 0.05
WEIGHT_FLUX      = 1.0
WEIGHT_CUM       = 0.5
TARGET_CATEGORY  = 'all'    # 'all', 'count', 'flux', 'cum', 'transport', 'ribo'

# ---- Inference-speed toggles ----
USE_TORCH_COMPILE             = False
INFERENCE_DTYPE               = 'fp32'   # 'fp32' | 'bf16' | 'fp16'
INFERENCE_DISABLE_CHECKPOINT  = False


def patch_model(model):
    """Mutation hook: monkey-patch model attrs. Default no-op.
    Examples:
        model.rate_clip = 8.0
        model.use_checkpoint = False
    Must return model.
    """
    return model


def filter_weakness(weakness, row_names):
    """Mutation hook: restrict weakness-targeting to specific row prefixes.
    Default uses TARGET_CATEGORY above. Modify body to use richer criteria
    (e.g. specific gene-class subsets, uncertainty-based selection, etc.).
    """
    if TARGET_CATEGORY in (None, 'all'):
        return weakness
    prefix_map = {
        'count':     ('G_', 'R_', 'P_', 'M_'),
        'flux':      ('F_',),
        'cum':       ('C_',),
        'transport': ('M_',),
        'ribo':      ('P_', 'R_'),
    }
    prefixes = prefix_map.get(TARGET_CATEGORY, ())
    if not prefixes:
        return weakness
    new_idx = [i for i in weakness['worst_species_idx']
               if any(row_names[i].startswith(p) for p in prefixes)]
    if new_idx:
        weakness['worst_species_idx'] = new_idx[:50]
    return weakness


# ================================================================
# === END MUTABLE SECTION ===
# ================================================================


# ================================================================
# === HARNESS FOOTER — DO NOT MODIFY ===
# This main() pipeline calls only the specific functions imported
# in the header. It does NOT use argparse. It writes the candidate
# checkpoint to OUTPUT_CKPT and prints metrics between the markers.
# ================================================================
def _apply_inference_optimizations(model):
    if INFERENCE_DISABLE_CHECKPOINT and hasattr(model, 'use_checkpoint'):
        model.use_checkpoint = False
    if INFERENCE_DTYPE == 'bf16':
        model = model.to(torch.bfloat16)
    elif INFERENCE_DTYPE == 'fp16':
        model = model.to(torch.float16)
    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model, mode='reduce-overhead', dynamic=False)
        except Exception as e:
            print(f'[m7-candidate] torch.compile failed: {e}; continuing')
    return model


def main():
    print(f'[m7-candidate] device={DEVICE}  initial={INITIAL_CKPT}')

    sg = load_species_graph(f'{DRIVE_DIR}/species_graph_v7.pt')
    row_names = list(sg.row_names)
    train_data = torch.load(f'{DRIVE_DIR}/preload_train.pt',
                             map_location=DEVICE, weights_only=False)
    val_data = torch.load(f'{DRIVE_DIR}/preload_val.pt',
                           map_location=DEVICE, weights_only=False)
    sigma = torch.load(f'{DRIVE_DIR}/sigma_train_reps_1to49.pt',
                        map_location=DEVICE, weights_only=False)

    model, cfg, fidx = load_m7_with_state(INITIAL_CKPT, sg, row_names, DEVICE)
    model = patch_model(model)

    weakness = analyze_model_weaknesses(
        model, val_data, sigma, row_names, t_step_stride=100,
    )
    weakness = filter_weakness(weakness, row_names)

    model = fine_tune_on_weaknesses(
        model, train_data, sigma, val_data, row_names, fidx, weakness,
        n_steps=FINE_TUNE_STEPS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        bias_strength=BIAS_STRENGTH,
        weight_count=WEIGHT_COUNT,
        weight_flux=WEIGHT_FLUX,
        weight_cumulative=WEIGHT_CUM,
        log_every=50,
    )

    model = _apply_inference_optimizations(model)

    metrics = compute_full_val_metrics(model, val_data, sigma, row_names)

    torch.save({
        'state_dict': (model._orig_mod.state_dict()
                       if hasattr(model, '_orig_mod')
                       else model.state_dict()),
        'cfg': cfg,
        'architecture': 'M7_hybrid',
        'flux_indices': fidx.cpu().tolist(),
        'metrics': metrics,
    }, OUTPUT_CKPT)

    print('===EVAL_METRICS===')
    print(json.dumps(metrics))
    print('===END_EVAL_METRICS===')


if __name__ == '__main__':
    main()
# ================================================================
# === END HARNESS FOOTER ===
# ================================================================
