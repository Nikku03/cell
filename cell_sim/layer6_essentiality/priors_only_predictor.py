"""Fast priors-only essentiality predictor (no simulator).

For many use cases — CI checks, rapid detector iteration, whole-genome
screening dashboards — running the full RealSimulator over every
gene takes 15-60 minutes of wall time and scales linearly in the
number of genes. The two prior-knowledge detectors (ComplexAssembly,
AnnotationClass) need no simulation at all; they're deterministic
dict lookups over the bundled repo data. This module exposes their
composition as a single-call CLI that scores all N Breuer-labeled
genes in under a second.

Exactly the same signals used by the ``composed`` detector in the
parallel sweep, minus the trajectory PerRule detector. On the
standard balanced n=40 panel the priors-only score is ~0.55 MCC vs
the full composed stack's 0.70 — so the simulator-dependent
PerRule is still worth running when you want the last drop of
metabolic-enzyme coverage. But for quick screens or development
loops, this is the right tool.

Zero network / API calls. Zero per-run cost.

Example::

    # Score all Breuer-labeled genes in ~0.1s.
    python -m cell_sim.layer6_essentiality.priors_only_predictor \\
        --breuer memory_bank/data/syn3a_essentiality_breuer2019.csv \\
        --balanced --n 40 --panel-seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Iterable, Optional

from cell_sim.layer6_essentiality.complex_assembly_detector import (
    ComplexAssemblyDetector,
)
from cell_sim.layer6_essentiality.annotation_class_detector import (
    AnnotationClassDetector,
)
from cell_sim.layer6_essentiality.harness import FailureMode


@dataclass(frozen=True)
class PriorsPrediction:
    locus_tag: str
    essential: bool
    confidence: float
    evidence: str
    source: str   # "complex", "annotation", "both", or "none"


@dataclass
class PriorsOnlyPredictor:
    """OR-composition of ComplexAssembly + AnnotationClass, no simulator."""
    complex: ComplexAssemblyDetector = field(default_factory=ComplexAssemblyDetector)
    annotation: AnnotationClassDetector = field(default_factory=AnnotationClassDetector)

    def predict(self, locus_tag: str) -> PriorsPrediction:
        c_mode, _, c_conf, c_ev = self.complex.detect_for_gene(locus_tag)
        a_mode, _, a_conf, a_ev = self.annotation.detect_for_gene(locus_tag)
        c_fires = c_mode != FailureMode.NONE
        a_fires = a_mode != FailureMode.NONE
        if c_fires and a_fires:
            return PriorsPrediction(locus_tag, True, max(c_conf, a_conf),
                                     f"{c_ev} & {a_ev}", "both")
        if c_fires:
            return PriorsPrediction(locus_tag, True, c_conf, c_ev, "complex")
        if a_fires:
            return PriorsPrediction(locus_tag, True, a_conf, a_ev, "annotation")
        return PriorsPrediction(locus_tag, False, 0.0,
                                 f"{c_ev}|{a_ev}", "none")

    def predict_many(self, locus_tags: Iterable[str]) -> list[PriorsPrediction]:
        return [self.predict(t) for t in locus_tags]


# --------- scoring helpers ---------


def score_against_breuer(
    predictor: PriorsOnlyPredictor,
    breuer_labels: dict[str, str],
    quasi_as_positive: bool = True,
) -> dict:
    tp = fp = tn = fn = 0
    predictions: list[dict] = []
    for locus, lbl in breuer_labels.items():
        is_positive_label = (
            lbl == "Essential"
            or (quasi_as_positive and lbl == "Quasiessential")
        )
        is_negative_label = lbl == "Nonessential" or (
            not quasi_as_positive and lbl == "Quasiessential"
        )
        p = predictor.predict(locus)
        if is_positive_label and p.essential:
            tp += 1
        elif is_positive_label and not p.essential:
            fn += 1
        elif is_negative_label and p.essential:
            fp += 1
        elif is_negative_label and not p.essential:
            tn += 1
        predictions.append({
            "locus_tag": locus,
            "true": lbl,
            "pred_essential": p.essential,
            "source": p.source,
            "confidence": p.confidence,
            "evidence": p.evidence,
        })
    num = tp * tn - fp * fn
    den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = num / den if den > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "mcc": mcc,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "predictions": predictions,
    }


def build_balanced_panel(
    breuer_labels: dict[str, str],
    n: int,
    seed: int,
) -> list[str]:
    """Sample ``n//2`` Essentials + ``n//2`` Nonessentials."""
    import random
    rng = random.Random(seed)
    ess = sorted(t for t, v in breuer_labels.items() if v == "Essential")
    non = sorted(t for t, v in breuer_labels.items() if v == "Nonessential")
    k = n // 2
    rng.shuffle(ess)
    rng.shuffle(non)
    return sorted(ess[:k] + non[:k])


def _load_breuer(path: Path) -> dict[str, str]:
    labels = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            labels[row["locus_tag"]] = row["essentiality"]
    return labels


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--breuer", type=Path,
                   default=(Path(__file__).resolve().parents[2]
                            / "memory_bank" / "data"
                            / "syn3a_essentiality_breuer2019.csv"))
    p.add_argument("--balanced", action="store_true",
                   help="Score on a balanced n-gene panel instead of "
                        "the full labeled set.")
    p.add_argument("--n", type=int, default=40,
                   help="Panel size (balanced mode only).")
    p.add_argument("--panel-seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None,
                   help="Optional JSON output path for per-gene "
                        "predictions + metrics.")
    p.add_argument("--quasi-positive", action="store_true", default=True,
                   help="Treat Quasiessential as positive (default).")
    p.add_argument("--quasi-negative", dest="quasi_positive",
                   action="store_false")
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    labels = _load_breuer(args.breuer)
    if args.balanced:
        panel = set(build_balanced_panel(labels, args.n, args.panel_seed))
        labels = {t: lbl for t, lbl in labels.items() if t in panel}

    predictor = PriorsOnlyPredictor()
    result = score_against_breuer(predictor, labels,
                                   quasi_as_positive=args.quasi_positive)
    elapsed = time.perf_counter() - t0

    print(f"Scored {len(labels)} genes in {elapsed*1000:.0f} ms "
          f"(no simulator).")
    print(f"  TP={result['tp']} FP={result['fp']} "
          f"TN={result['tn']} FN={result['fn']}")
    print(f"  MCC={result['mcc']:.3f}  "
          f"precision={result['precision']:.3f}  "
          f"recall={result['recall']:.3f}  "
          f"specificity={result['specificity']:.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "config": {
                "breuer": str(args.breuer),
                "balanced": args.balanced,
                "n": args.n if args.balanced else len(labels),
                "panel_seed": args.panel_seed,
                "quasi_as_positive": args.quasi_positive,
            },
            "metrics": {
                k: v for k, v in result.items() if k != "predictions"
            },
            "predictions": result["predictions"],
            "elapsed_s": elapsed,
        }
        args.out.write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
