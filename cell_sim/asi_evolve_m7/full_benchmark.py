"""Multi-objective benchmark for M7-Evolve v2.

Every candidate is scored on THREE objectives:
  1. accuracy    — composite val MSE (lower better)
  2. inference   — wall-clock for a fixed knockout sweep workload
  3. memory      — peak VRAM during the same workload

The Researcher LLM can then propose mutations that target any one
or improve the Pareto frontier across all three. ASI-Evolve's score
field is the weighted sum (configurable in config.yaml); the
individual metrics are also returned so the database keeps them for
Pareto analysis.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, '/content/cell')

from cell_sim.lgnn.data.species_graph import load_species_graph
from cell_sim.lgnn.self_improve.iterate import (
    compute_full_val_metrics,
    load_m7_with_state,
)
from cell_sim.lgnn.analysis.knockout_sweep import (
    _full_locus_row_indices,
    run_knockout_sweep,
)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DRIVE_DIR = os.environ.get('M7_DRIVE_DIR',
                            '/content/drive/MyDrive/cell_count_dynamics')

# Multi-objective weights — tuning these shifts the Researcher's
# priorities. accuracy weight is the dominant signal by default;
# crank speed/memory weights up if you want the loop to optimise infra.
W_ACCURACY = float(os.environ.get('M7_W_ACCURACY',  '1.0'))
W_SPEED    = float(os.environ.get('M7_W_SPEED',     '0.3'))
W_MEMORY   = float(os.environ.get('M7_W_MEMORY',    '0.1'))

# Baseline values (M7.7 measured), used to compute relative scores
BASELINE = {
    'composite': 0.0837,
    'sweep_time_sec': 132.0,      # 490 KOs @ 200 steps each, batch=1
    'peak_memory_gb': 4.2,         # rough estimate
}


def measure_inference_speed(model, val_data, sigma, row_names, n_ko=50):
    """Run a fixed knockout-sweep workload, return wall time + memory."""
    gene_groups = _full_locus_row_indices(row_names)
    loci = sorted(gene_groups.keys())[:n_ko]
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    df = run_knockout_sweep(
        model, val_data, sigma, row_names,
        x0_step=100, rollout_steps=200,
        impact_metric='l2_drift',
        eval_dtype=torch.float32,
        knockout_mode='full_locus',
        knockout_loci=loci,
        log_every=10000,
    )
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    wall = time.time() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / 1024**3
               if DEVICE == 'cuda' else 0.0)
    return {
        'sweep_time_sec': float(wall),
        'sweep_ko_per_sec': float(n_ko / max(wall, 1e-6)),
        'peak_memory_gb': float(peak_gb),
        'n_ko_measured': n_ko,
    }


def composite_objective_score(metrics: dict) -> dict:
    """Combine accuracy + speed + memory into a single scalar.

    Lower is better. Each component is a relative ratio vs baseline,
    so a value of 1.0 is the baseline; <1 is improvement; >1 is
    regression.
    """
    acc_ratio = metrics['composite']        / max(BASELINE['composite'],      1e-9)
    spd_ratio = metrics['sweep_time_sec']   / max(BASELINE['sweep_time_sec'], 1e-9)
    mem_ratio = max(metrics['peak_memory_gb'], 0.1) / max(BASELINE['peak_memory_gb'], 0.1)

    composite_relative = (
        W_ACCURACY * acc_ratio
        + W_SPEED    * spd_ratio
        + W_MEMORY   * mem_ratio
    )
    return {
        'composite_relative': composite_relative,
        'accuracy_ratio': acc_ratio,
        'speed_ratio': spd_ratio,
        'memory_ratio': mem_ratio,
    }


def full_benchmark(checkpoint_path: str) -> dict:
    """Run accuracy + speed + memory benchmarks on a candidate."""
    sg = load_species_graph(f'{DRIVE_DIR}/species_graph_v7.pt')
    row_names = list(sg.row_names)
    val_data = torch.load(f'{DRIVE_DIR}/preload_val.pt',
                           map_location=DEVICE, weights_only=False)
    sigma = torch.load(f'{DRIVE_DIR}/sigma_train_reps_1to49.pt',
                        map_location=DEVICE, weights_only=False)
    print(f'[bench] loading {checkpoint_path}')
    model, cfg, _ = load_m7_with_state(checkpoint_path, sg, row_names, DEVICE)

    print('[bench] measuring accuracy...')
    acc = compute_full_val_metrics(model, val_data, sigma, row_names)

    print('[bench] measuring inference speed + memory...')
    spd = measure_inference_speed(model, val_data, sigma, row_names, n_ko=50)

    combined = {**acc, **spd}
    relative = composite_objective_score(combined)
    return {**combined, **relative}


def main():
    if len(sys.argv) < 2:
        print('usage: full_benchmark.py <checkpoint_path>', file=sys.stderr)
        sys.exit(2)
    metrics = full_benchmark(sys.argv[1])
    print('===EVAL_METRICS===')
    print(json.dumps(metrics))
    print('===END_EVAL_METRICS===')


if __name__ == '__main__':
    main()
