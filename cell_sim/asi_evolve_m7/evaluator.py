"""Evaluator for ASI-Evolve M7 experiment.

ASI-Evolve interface (verified from engineer.py):
  - Reads results from results.json in cwd, NOT from stdout.
  - results.json MUST contain 'eval_score' field (lower composite -> higher score).
  - Optional but useful fields: 'success', 'error', and any custom metrics.

This script:
  - Reads the bench log (from full_benchmark.py stdout)
  - Parses the ===EVAL_METRICS=== JSON block within
  - Writes a complete results dict to the requested output path
  - eval_score = -composite_relative (ASI-Evolve maximises score)

Usage (called by eval.sh):
    python evaluator.py <bench_log_path> <results_json_output_path>
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


def _crash_result(reason: str) -> dict:
    return {
        'eval_score': -1.0e6,
        'success': False,
        'error': reason,
        'composite_relative': float('inf'),
        'val_singlestep_mse': float('inf'),
        'val_rollout_200_mse': float('inf'),
    }


def evaluate(bench_log_path: str) -> dict:
    path = Path(bench_log_path)
    if not path.exists():
        return _crash_result(f'bench log missing: {bench_log_path}')
    try:
        content = path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return _crash_result(f'failed to read bench log: {e}')

    m = METRICS_RE.search(content)
    if not m:
        tail = content[-500:].replace('"', "'")
        return _crash_result(f'no ===EVAL_METRICS=== block in bench log; tail: {tail}')

    try:
        metrics = json.loads(m.group(1).strip())
    except json.JSONDecodeError as e:
        return _crash_result(f'metrics JSON parse error: {e}')

    composite_rel = float(metrics.get(
        'composite_relative',
        metrics.get('composite', float('inf')),
    ))

    # ASI-Evolve maximises score; we want to minimise composite_relative,
    # so eval_score = -composite_relative.
    return {
        'eval_score': -composite_rel,
        'success': composite_rel < float('inf'),
        'composite_relative': composite_rel,
        # accuracy
        'composite': float(metrics.get('composite', float('inf'))),
        'val_singlestep_mse': float(metrics.get('val_singlestep_mse', float('inf'))),
        'val_rollout_200_mse': float(metrics.get('val_rollout_200_mse', float('inf'))),
        'val_mse_count': float(metrics.get('val_mse_count', float('inf'))),
        'val_mse_flux':  float(metrics.get('val_mse_flux',  float('inf'))),
        'val_mse_cum':   float(metrics.get('val_mse_cum',   float('inf'))),
        # speed / memory
        'sweep_time_sec':   float(metrics.get('sweep_time_sec',   float('inf'))),
        'sweep_ko_per_sec': float(metrics.get('sweep_ko_per_sec', 0.0)),
        'peak_memory_gb':   float(metrics.get('peak_memory_gb',   float('inf'))),
        # ratios for Pareto analysis
        'accuracy_ratio':   float(metrics.get('accuracy_ratio',   float('inf'))),
        'speed_ratio':      float(metrics.get('speed_ratio',      float('inf'))),
        'memory_ratio':     float(metrics.get('memory_ratio',     float('inf'))),
    }


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(
            'usage: evaluator.py <bench_log_path> [<results_json_output_path>]\n'
        )
        sys.exit(2)
    bench_log = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None

    result = evaluate(bench_log)

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
