"""ComposedDetector: OR the structural complex-assembly signal with any
trajectory-based detector.

Design rationale:

* The existing trajectory-based detectors (PerRule, ShortWindow,
  RedundancyAware) bottomed out around MCC 0.12-0.16 on Breuer 2019
  because the simulator's 6 800-reaction network cannot expose
  non-catalytic essentiality (ribosomal, tRNA-synthetase, translation
  factors, transcription machinery) — those genes have no metabolic
  rule for a rule-silencing detector to watch and no pool that
  visibly drops on a sub-second simulation window.

* The structural ``ComplexAssemblyDetector`` is the complement: it
  catches exactly that non-catalytic cohort (ribosomal subunits,
  RNA polymerase, ATP synthase, SecYEGDF, primary ABC transporters,
  etc.) via known-complex subunit membership. Alone it scores
  MCC 0.26 on the full Breuer set at precision 0.97.

The composition rule:

  essential(gene) := structural_says_essential(gene)
                     OR trajectory_says_essential(gene)

High-confidence detectors fire; abstentions fall through. Failure
mode is reported from whichever detector fired, with structural
taking precedence when both fire (the structural signal is more
informative for the non-catalytic cohort).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

from cell_sim.layer6_essentiality.harness import FailureMode, Trajectory
from cell_sim.layer6_essentiality.complex_assembly_detector import (
    ComplexAssemblyDetector,
)


class TrajectoryDetector(Protocol):
    """Any existing detector whose detect_for_gene takes (locus_tag, ko)
    and returns (mode, t, conf, evidence)."""
    def detect_for_gene(
        self, locus_tag: str, ko: Trajectory,
    ) -> tuple[FailureMode, Optional[float], float, str]: ...


@dataclass
class ComposedDetector:
    """OR-composition of a structural detector + a trajectory detector."""
    structural: ComplexAssemblyDetector
    trajectory: TrajectoryDetector

    def detect_for_gene(
        self,
        locus_tag: str,
        ko: Trajectory,
    ) -> tuple[FailureMode, Optional[float], float, str]:
        s_mode, s_t, s_conf, s_ev = self.structural.detect_for_gene(locus_tag, ko)
        t_mode, t_t, t_conf, t_ev = self.trajectory.detect_for_gene(locus_tag, ko)

        s_fires = s_mode != FailureMode.NONE
        t_fires = t_mode != FailureMode.NONE

        if s_fires and t_fires:
            # Both agree; report the higher-confidence one.
            if s_conf >= t_conf:
                return (s_mode, s_t if s_t is not None else t_t,
                        s_conf, f"cx+traj[{s_ev} & {t_ev}]")
            return (t_mode, t_t if t_t is not None else s_t,
                    t_conf, f"traj+cx[{t_ev} & {s_ev}]")
        if s_fires:
            return s_mode, s_t, s_conf, f"cx_only[{s_ev}]"
        if t_fires:
            return t_mode, t_t, t_conf, f"traj_only[{t_ev}]"
        return FailureMode.NONE, None, 0.0, f"no_signal[cx:{s_ev}|traj:{t_ev}]"


__all__ = ["ComposedDetector", "TrajectoryDetector"]
