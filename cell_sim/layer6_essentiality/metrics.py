"""MCC and confusion-matrix utilities for binary essentiality evaluation.

Pure Python, no sklearn dependency. Kept small and readable so the math
is auditable against Breuer 2019's reported MCC of 0.59.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True, slots=True)
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int
    mcc: float
    precision: float
    recall: float
    specificity: float
    accuracy: float
    n: int

    def as_dict(self) -> dict:
        return {
            "n": self.n, "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "mcc": self.mcc, "precision": self.precision, "recall": self.recall,
            "specificity": self.specificity, "accuracy": self.accuracy,
        }


def _mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    num = (tp * tn) - (fp * fn)
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq == 0:
        return 0.0
    return num / sqrt(denom_sq)


def evaluate_binary(
    y_true: dict[str, int],
    y_pred: dict[str, int],
) -> Metrics:
    """Compute MCC over the intersection of keys. Raises if no overlap."""
    keys = set(y_true) & set(y_pred)
    if not keys:
        raise ValueError("no overlap between y_true and y_pred keys")
    tp = fp = tn = fn = 0
    for k in keys:
        t = int(y_true[k])
        p = int(y_pred[k])
        if t == 1 and p == 1: tp += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 0 and p == 0: tn += 1
        elif t == 1 and p == 0: fn += 1
    n = tp + fp + tn + fn
    mcc = _mcc(tp, fp, tn, fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    return Metrics(tp, fp, tn, fn, mcc, precision, recall, specificity, accuracy, n)
