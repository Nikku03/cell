"""ComposedDetector: OR the structural complex-assembly signal,
annotation-class signal, and any trajectory-based detector.

Design rationale:

* The existing trajectory-based detectors (PerRule, ShortWindow,
  RedundancyAware) bottomed out around MCC 0.12-0.16 on Breuer 2019
  because the simulator's 6 800-reaction network cannot expose
  non-catalytic essentiality (ribosomal, tRNA-synthetase, translation
  factors, transcription machinery) — those genes have no metabolic
  rule for a rule-silencing detector to watch and no pool that
  visibly drops on a sub-second simulation window.

* The structural ``ComplexAssemblyDetector`` is the complement for
  multi-protein essentials: ribosomal subunits, RNA polymerase, ATP
  synthase, SecYEGDF, primary ABC transporters. MCC 0.26 full
  Breuer, precision 0.97.

* The ``AnnotationClassDetector`` is the complement for single-gene
  essentials matching known bacterial essential-class keywords:
  aminoacyl-tRNA synthetases, translation factors, signal
  peptidases, flippases, primary RNA-processing nucleases,
  DNA replication core. MCC 0.22 full Breuer, precision 1.00.

The composition rule:

  essential(gene) := complex_says_essential(gene)
                     OR annotation_says_essential(gene)
                     OR trajectory_says_essential(gene)

Abstentions fall through to the next signal. Confidence is the max
of the firing detectors. Failure mode follows the highest-confidence
fire.

All inputs are bundled repo data (``complex_formation.xlsx``,
``syn3a_gene_table.csv``, SBML). Zero API / network / per-run cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

from cell_sim.layer6_essentiality.harness import FailureMode, Trajectory
from cell_sim.layer6_essentiality.complex_assembly_detector import (
    ComplexAssemblyDetector,
)
from cell_sim.layer6_essentiality.annotation_class_detector import (
    AnnotationClassDetector,
)


class TrajectoryDetector(Protocol):
    """Any existing detector whose detect_for_gene takes (locus_tag, ko)
    and returns (mode, t, conf, evidence)."""
    def detect_for_gene(
        self, locus_tag: str, ko: Trajectory,
    ) -> tuple[FailureMode, Optional[float], float, str]: ...


@dataclass
class ComposedDetector:
    """OR-composition of complex + annotation + trajectory detectors.

    ``annotation`` is optional for backwards compatibility; when None,
    behaves as a two-way compose (complex OR trajectory).
    """
    structural: ComplexAssemblyDetector
    trajectory: TrajectoryDetector
    annotation: Optional[AnnotationClassDetector] = None

    def detect_for_gene(
        self,
        locus_tag: str,
        ko: Trajectory,
    ) -> tuple[FailureMode, Optional[float], float, str]:
        s_mode, s_t, s_conf, s_ev = self.structural.detect_for_gene(locus_tag, ko)
        if self.annotation is not None:
            a_mode, a_t, a_conf, a_ev = self.annotation.detect_for_gene(
                locus_tag, ko
            )
        else:
            a_mode, a_t, a_conf, a_ev = FailureMode.NONE, None, 0.0, "ann_off"
        t_mode, t_t, t_conf, t_ev = self.trajectory.detect_for_gene(locus_tag, ko)

        fires = [
            ("cx", s_mode, s_t, s_conf, s_ev),
            ("ann", a_mode, a_t, a_conf, a_ev),
            ("traj", t_mode, t_t, t_conf, t_ev),
        ]
        firing = [f for f in fires if f[1] != FailureMode.NONE]
        if not firing:
            return (FailureMode.NONE, None, 0.0,
                    f"no_signal[cx:{s_ev}|ann:{a_ev}|traj:{t_ev}]")
        firing.sort(key=lambda f: f[3], reverse=True)
        best = firing[0]
        tag = best[0]
        mode = best[1]
        t_fail = best[2]
        conf = best[3]
        ev = best[4]
        if len(firing) == 1:
            suffix = f"{tag}_only[{ev}]"
        else:
            suffix = (f"{tag}+" + "+".join(f[0] for f in firing[1:])
                      + f"[{ev}]")
        return mode, t_fail, conf, suffix


__all__ = ["ComposedDetector", "TrajectoryDetector"]
