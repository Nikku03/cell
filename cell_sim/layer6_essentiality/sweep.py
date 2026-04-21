"""Genome-wide essentiality sweep runner.

The sweep is embarrassingly parallel: one knockout simulation per gene,
completely independent. In this session we ship the orchestration layer
(genome enumeration, per-gene invocation, CSV output) against the
``MockSimulator``. A real sweep plugs in a simulator backed by
``cell_sim.layer2_field.FastEventSimulator`` - deferred.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cell_sim.layer0_genome.genome import Genome
from cell_sim.layer6_essentiality.harness import (
    KnockoutHarness, Prediction, Simulator,
)


@dataclass
class SweepConfig:
    t_end_s: float = 7200.0
    sample_dt_s: float = 60.0
    output_csv: Path = Path("predictions.csv")
    include_rna_genes: bool = False


def run_sweep(
    wt_simulator: Simulator,
    ko_simulator: Simulator,
    genome: Genome,
    config: SweepConfig | None = None,
    genes: Iterable[str] | None = None,
) -> list[Prediction]:
    """Predict essentiality for `genes` (default: all CDS). Writes a CSV
    and returns the list of Predictions."""
    cfg = config or SweepConfig()
    harness = KnockoutHarness(
        wt_simulator=wt_simulator,
        ko_simulator=ko_simulator,
        t_end_s=cfg.t_end_s,
        sample_dt_s=cfg.sample_dt_s,
    )
    if genes is None:
        source = genome.cds_genes() if not cfg.include_rna_genes else iter(genome)
        targets = [(g.locus_tag, g.gene_name) for g in source]
    else:
        targets = []
        for lt in genes:
            gene = genome[lt]
            targets.append((gene.locus_tag, gene.gene_name))

    preds: list[Prediction] = []
    for lt, gn in targets:
        preds.append(harness.predict(lt, gene_name=gn))

    _write_csv(preds, cfg.output_csv)
    return preds


def _write_csv(preds: list[Prediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "locus_tag", "gene_name", "essential",
                "time_to_failure_s", "failure_mode", "confidence",
            ],
        )
        w.writeheader()
        for p in preds:
            w.writerow(p.as_row())


def predictions_as_binary_dict(
    preds: list[Prediction],
) -> dict[str, int]:
    return {p.locus_tag: int(p.essential) for p in preds}
