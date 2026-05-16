"""Evaluator for ASI-Evolve M7 experiment.

Reads the candidate program's stdout, parses the JSON metrics block,
returns a structured score dict on stdout (for ASI-Evolve to ingest).

The score signal is `-composite`: lower composite = higher score.
Composite = val_singlestep_mse + 0.3 * val_rollout_200_mse.

Usage (called by eval.sh):
    python evaluator.py <stdout_log_path>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


METRICS_RE = re.compile(
    r'===EVAL_METRICS===\s*\n(.*?)\n===END_EVAL_METRICS===',
    re.DOTALL,
)


def _crash_score(reason: str) -> dict:
    return {
        'score': -1.0e6,
        'error': reason,
        'composite': float('inf'),
        'val_singlestep_mse': float('inf'),
        'val_rollout_200_mse': float('inf'),
    }


def evaluate(stdout_path: str) -> dict:
    path = Path(stdout_path)
    if not path.exists():
        return _crash_score(f'stdout log missing: {stdout_path}')
    try:
        content = path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return _crash_score(f'failed to read log: {e}')

    m = METRICS_RE.search(content)
    if not m:
        return _crash_score('no ===EVAL_METRICS=== block in stdout')

    try:
        metrics = json.loads(m.group(1).strip())
    except json.JSONDecodeError as e:
        return _crash_score(f'metrics JSON parse error: {e}')

    # Multi-objective score: lower composite_relative = higher score.
    # composite_relative is a weighted sum of accuracy/speed/memory
    # ratios vs baseline, computed by full_benchmark.composite_objective_score.
    composite_rel = float(metrics.get('composite_relative',
                                       metrics.get('composite', float('inf'))))
    return {
        'score': -composite_rel,
        'composite_relative': composite_rel,
        # accuracy
        'composite': float(metrics.get('composite', float('inf'))),
        'val_singlestep_mse': float(metrics.get('val_singlestep_mse', float('inf'))),
        'val_rollout_200_mse': float(metrics.get('val_rollout_200_mse', float('inf'))),
        'val_mse_count': float(metrics.get('val_mse_count', float('inf'))),
        'val_mse_flux':  float(metrics.get('val_mse_flux',  float('inf'))),
        # speed / memory (set by full_benchmark.py)
        'sweep_time_sec':  float(metrics.get('sweep_time_sec',  float('inf'))),
        'sweep_ko_per_sec': float(metrics.get('sweep_ko_per_sec', 0.0)),
        'peak_memory_gb':  float(metrics.get('peak_memory_gb',  float('inf'))),
        # ratios for Pareto analysis
        'accuracy_ratio': float(metrics.get('accuracy_ratio', float('inf'))),
        'speed_ratio':    float(metrics.get('speed_ratio',    float('inf'))),
        'memory_ratio':   float(metrics.get('memory_ratio',   float('inf'))),
    }


def main():
    if len(sys.argv) < 2:
        sys.stderr.write('usage: evaluator.py <stdout_log_path>\n')
        sys.exit(2)
    result = evaluate(sys.argv[1])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
