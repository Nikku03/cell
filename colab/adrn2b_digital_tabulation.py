"""Is the Gate B answer already a coordinate of ADRN-2B's digital state block?

Gate B compares an arm carrying 288 exact ``digital_temporal_state_tensor``
coordinates against a polynomial control that carries none.  The claim under
test here is narrow and checkable without training anything:

    For each held-out temporal rule, is the rule's own output recoverable from
    the digital block by a SINGLE coordinate (up to sign), or by a threshold on
    the difference of TWO coordinates -- and is it recoverable at all from the
    polynomial basis?

If yes, then a Gate B pass says the online adapter can select a column, not that
a recurrent workspace computed anything.  This script measures that directly:
no learning, no adapter, no episodes -- just the representation against the
labels on fresh random sequences.

Four readouts per rule, all on held-out draws:
  best_single_digital      accuracy of the best sign(x_j) over the 288 digital coords
  best_pair_digital        accuracy of the best threshold on (x_j - x_k), digital only,
                           restricted to the first/last-occurrence-time coordinates
  linear_digital           balanced accuracy of a least-squares readout on all 288
  linear_polynomial        same on the 168-dim polynomial basis (the control's rank)

``best_single_digital`` near 1.0 is the tabulation result.  ``linear_polynomial``
tells you what the control could reach with unlimited labels, separating "the
control lacks the information" from "the control needs more data".

Usage
-----
    python adrn2b_digital_tabulation.py --package /path/to/adrn2b_package \\
        --output out/digital_tabulation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

TEMPORAL_OPERATIONS = ("temporal_order", "last_event", "running_parity", "toggle_reset")
DRAWS = 4_096


def _load(package: Path):
    package = package.resolve()
    sys.path.insert(0, str(package.parent))
    name = package.name
    tasks = __import__(f"{name}.adrn2b_tasks", fromlist=["*"])
    models = __import__(f"{name}.adrn2b_models", fromlist=["*"])
    experiment = __import__(f"{name}.adrn2b_experiment", fromlist=["*"])
    return tasks, models, experiment


def _balanced(pred: np.ndarray, truth: np.ndarray) -> float:
    scores = []
    for value in (0, 1):
        mask = truth == value
        if not mask.any():
            return float("nan")
        scores.append(float((pred[mask] == value).mean()))
    return float(np.mean(scores))


def _best_single(features: np.ndarray, truth: np.ndarray) -> tuple[float, int]:
    """Best balanced accuracy from thresholding one coordinate at zero, either sign."""

    positive = features > 0.0
    best_score, best_index = -1.0, -1
    for index in range(features.shape[1]):
        column = positive[:, index]
        for candidate in (column, ~column):
            score = _balanced(candidate.astype(np.int64), truth)
            if score == score and score > best_score:
                best_score, best_index = score, index
    return best_score, best_index


def _best_pair(features: np.ndarray, truth: np.ndarray, columns: range) -> float:
    """Best balanced accuracy from thresholding (x_j - x_k) at zero, either sign."""

    best = -1.0
    subset = features[:, columns.start : columns.stop]
    for j in range(subset.shape[1]):
        differences = subset[:, j : j + 1] - subset
        for k in range(subset.shape[1]):
            if j == k:
                continue
            column = differences[:, k] > 0.0
            for candidate in (column, ~column):
                score = _balanced(candidate.astype(np.int64), truth)
                if score == score and score > best:
                    best = score
    return best


def _linear_readout(train_x, train_y, test_x, test_y) -> float:
    """Balanced accuracy of a ridge least-squares readout -- a representational ceiling.

    This is deliberately generous: it is fit with far more labels than any online
    adapter sees, so a low score means the information is ABSENT from the basis,
    not merely hard to reach.
    """

    design = np.concatenate([train_x, np.ones((len(train_x), 1), np.float64)], axis=1)
    targets = 2.0 * train_y.astype(np.float64) - 1.0
    gram = design.T @ design + 1e-3 * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ targets)
    test_design = np.concatenate([test_x, np.ones((len(test_x), 1), np.float64)], axis=1)
    return _balanced((test_design @ weights > 0.0).astype(np.int64), test_y)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args(argv)

    tasks, models, experiment = _load(args.package)
    import torch

    settings = experiment.ADRN2BExperimentConfig.full()
    splits = tasks.make_meta_rule_splits(
        train_size=settings.meta_train_tasks,
        validation_size=settings.meta_validation_tasks,
        test_size=settings.meta_test_tasks,
        sequence_length=settings.sequence_length,
        seed=settings.seed,
    )
    held_out = [rule for rule in splits.test if rule.operation in TEMPORAL_OPERATIONS]
    print(f"held-out temporal rules: {len(held_out)}  "
          f"{dict(sorted(Counter(r.operation for r in held_out).items()))}\n")

    rng = np.random.default_rng(settings.seed + 5_150)
    sequences = rng.integers(
        0, 2, size=(args.draws, settings.sequence_length, 16), dtype=np.int8
    ).astype(np.float32)
    values = torch.as_tensor(sequences, dtype=torch.float32)
    with torch.no_grad():
        digital = models.digital_temporal_state_tensor(values).numpy().astype(np.float64)
        polynomial = models.signed_polynomial_tensor(values).numpy().astype(np.float64)
    print(f"digital block {digital.shape[1]} coords, polynomial basis "
          f"{polynomial.shape[1]} coords, {args.draws} held-out draws")

    # parity | first_times | last_times | toggle_reset  ==  16 | 16 | 16 | 240
    time_columns = range(16, 48)
    split = args.draws // 2
    rows = []
    for rule in held_out:
        truth = np.fromiter(
            (tasks.evaluate_rule(rule, sequence) for sequence in sequences),
            dtype=np.int64, count=args.draws,
        )
        if truth.min() == truth.max():
            continue
        single, index = _best_single(digital, truth)
        pair = _best_pair(digital, truth, time_columns)
        rows.append({
            "operation": rule.operation,
            "rule": repr(rule.exact_tuple),
            "positive_rate": float(truth.mean()),
            "best_single_digital": single,
            "best_single_digital_coord": index,
            "best_pair_digital_times": pair,
            "linear_digital": _linear_readout(
                digital[:split], truth[:split], digital[split:], truth[split:]),
            "linear_polynomial": _linear_readout(
                polynomial[:split], truth[:split], polynomial[split:], truth[split:]),
        })

    print(f"\n{'operation':16s} {'n':>3s} {'best 1 coord':>13s} {'best 2-coord':>13s} "
          f"{'linear(288)':>12s} {'linear(168)':>12s}")
    summary = {}
    for operation in TEMPORAL_OPERATIONS:
        subset = [row for row in rows if row["operation"] == operation]
        if not subset:
            continue
        def mean(key):
            return float(np.mean([row[key] for row in subset]))
        summary[operation] = {
            "n_rules": len(subset),
            "best_single_digital": mean("best_single_digital"),
            "best_pair_digital_times": mean("best_pair_digital_times"),
            "linear_digital": mean("linear_digital"),
            "linear_polynomial": mean("linear_polynomial"),
        }
        s = summary[operation]
        print(f"{operation:16s} {len(subset):3d} {s['best_single_digital']:13.4f} "
              f"{s['best_pair_digital_times']:13.4f} {s['linear_digital']:12.4f} "
              f"{s['linear_polynomial']:12.4f}")

    overall = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("best_single_digital", "best_pair_digital_times",
                    "linear_digital", "linear_polynomial")
    }
    print(f"\n{'ALL':16s} {len(rows):3d} {overall['best_single_digital']:13.4f} "
          f"{overall['best_pair_digital_times']:13.4f} {overall['linear_digital']:12.4f} "
          f"{overall['linear_polynomial']:12.4f}")
    print("\nAll figures are balanced accuracy on held-out draws; 0.5 is chance.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"draws": args.draws, "per_rule": rows, "by_operation": summary,
         "overall": overall}, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
