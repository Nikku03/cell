"""Evaluator for ASI-Evolve artificial_atom v3 experiment.

Reads the bench log (candidate stdout), parses ===EVAL_METRICS=== block,
writes results.json with eval_score (= -composite; lower composite = higher score).
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


def _crash(reason: str) -> dict:
    return {
        'eval_score': -1.0e6,
        'success': False,
        'error': reason,
        'composite': float('inf'),
        'checks_passed': 0,
        'checks_total': 8,
        'pass_rate': 0.0,
    }


def evaluate(log_path: str) -> dict:
    path = Path(log_path)
    if not path.exists():
        return _crash(f'log missing: {log_path}')
    try:
        content = path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return _crash(f'failed to read log: {e}')

    m = METRICS_RE.search(content)
    if not m:
        tail = content[-500:].replace('"', "'")
        return _crash(f'no ===EVAL_METRICS=== block; tail: {tail}')

    try:
        metrics = json.loads(m.group(1).strip())
    except json.JSONDecodeError as e:
        return _crash(f'JSON parse error: {e}')

    composite = float(metrics.get('composite', float('inf')))
    out = {
        'eval_score': -composite,
        'success': metrics.get('checks_passed', 0) > 0,
        'composite': composite,
        'checks_passed': int(metrics.get('checks_passed', 0)),
        'checks_total': int(metrics.get('checks_total', 8)),
        'pass_rate': float(metrics.get('pass_rate', 0.0)),
        'wall_time_sec': float(metrics.get('wall_time_sec', float('inf'))),
        'n_params': int(metrics.get('n_params', 0)),
    }
    # Pass through all individual checks
    for k, v in metrics.items():
        if k.startswith('check_'):
            try:
                out[k] = v
            except Exception:
                pass
    return out


def main():
    if len(sys.argv) < 2:
        sys.stderr.write('usage: evaluator.py <log_path> [<results.json>]\n')
        sys.exit(2)
    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else None
    result = evaluate(log_path)
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
