"""Evaluator for the ASI-Evolve snn_scaling reversal experiment.

Reads candidate stdout, parses the ===EVAL_METRICS=== block, writes
results.json with eval_score = -composite (so higher score = lower
composite = better).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


METRICS_RE = re.compile(
    r"===EVAL_METRICS===\s*\n(.*?)\n===END_EVAL_METRICS===",
    re.DOTALL,
)


def _crash(reason: str) -> dict:
    return {
        "eval_score": -1.0e6,
        "success": False,
        "error": reason,
        "composite": float("inf"),
        "best_integrated_cum": float("-inf"),
        "gap_memrec_naive": float("-inf"),
    }


def evaluate(log_path: str) -> dict:
    path = Path(log_path)
    if not path.exists():
        return _crash(f"log missing: {log_path}")
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return _crash(f"failed to read log: {e}")

    m = METRICS_RE.search(content)
    if not m:
        tail = content[-500:].replace('"', "'")
        return _crash(f"no ===EVAL_METRICS=== block; tail: {tail}")

    try:
        metrics = json.loads(m.group(1).strip())
    except json.JSONDecodeError as e:
        return _crash(f"JSON parse error: {e}")

    composite = float(metrics.get("composite", float("inf")))
    out = {
        "eval_score": -composite,
        "success": composite < 1e5,
        "composite":              composite,
        "naive_cum_mean":         float(metrics.get("naive_cum_mean",  0.0)),
        "memrec_cum_mean":        float(metrics.get("memrec_cum_mean", 0.0)),
        "cls_cum_mean":           float(metrics.get("cls_cum_mean",    0.0)),
        "gap_memrec_naive":       float(metrics.get("gap_memrec_naive", 0.0)),
        "gap_cls_naive":          float(metrics.get("gap_cls_naive",   0.0)),
        "best_integrated_cum":    float(metrics.get("best_integrated_cum",     0.0)),
        "best_integrated_agent":  str(metrics.get("best_integrated_agent",   "n/a")),
        "wall_time_sec":          float(metrics.get("wall_time_sec",   float("inf"))),
        "sanity_penalty":         float(metrics.get("sanity_penalty",  0.0)),
        "n_reservoir":            int(metrics.get("n_reservoir", 0)),
        "t_timesteps":            int(metrics.get("t_timesteps", 0)),
    }
    return out


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: evaluator.py <log_path> [<results.json>]\n")
        sys.exit(2)
    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else None
    result = evaluate(log_path)
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
