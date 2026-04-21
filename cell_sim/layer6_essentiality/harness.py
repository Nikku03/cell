"""KnockoutHarness + FailureDetector.

The harness wraps any simulator that produces a ``Trajectory`` (a time
series of named pool counts) and decides whether a knockout caused
failure. This module deliberately keeps the simulator integration
behind an interface so that:

1. Unit tests can exercise the detection logic against synthetic
   trajectories without booting the full Layer 3-4 metabolic stack.
2. The production path (``_run_real``) can plug in the existing
   ``cell_sim.layer2_field.FastEventSimulator`` + ``populate_real_syn3a``
   without changing the detection logic.

Failure thresholds are conservative first-pass defaults. They WILL be
tuned against Breuer 2019 labels once the full sweep runs end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Protocol


class FailureMode(str, Enum):
    NONE = "none"
    ATP_DEPLETION = "atp_depletion"
    ESSENTIAL_METABOLITE_DEPLETION = "essential_metabolite_depletion"
    TRANSLATION_STALL = "translation_stall"
    TRANSCRIPTION_STALL = "transcription_stall"
    MEMBRANE_INTEGRITY = "membrane_integrity"
    DNA_REPLICATION_BLOCKED = "dna_replication_blocked"
    # Session 7: per-rule event-count signal. Trips when all rules
    # catalysed by a knocked-out gene stop firing, regardless of whether
    # metabolite pools move.
    CATALYSIS_SILENCED = "catalysis_silenced"


# Metabolites worth watching under knockouts. Names match the
# Luthey-Schulten initial_concentrations.xlsx Intracellular Metabolites
# sheet abbreviations; the real simulator must expose pools under these
# keys.
ESSENTIAL_METABOLITES: tuple[str, ...] = (
    "G6P", "F6P", "PYR",
    "dATP", "dGTP", "dCTP", "dTTP",
    "CTP", "GTP", "UTP",
    "NADH", "NAD",
)

# Thresholds: `ratio = ko_pool / wt_pool`. A failure trips when
# `ratio < threshold` at TWO consecutive sample points.
THRESHOLDS: dict[str, float] = {
    "ATP": 0.5,
    "essential_metabolite": 0.2,
    "translation": 0.3,
    "transcription": 0.3,
}


@dataclass(frozen=True, slots=True)
class Sample:
    t_s: float
    pools: dict[str, float]    # metabolite / pool counts
    # Session 7: cumulative count of sim events keyed by rule name
    # (e.g. "catalysis:PGI" -> 7888). Populated by RealSimulator._snapshot
    # from state.events. None on synthetic Sample objects used by tests
    # that don't exercise the per-rule detector.
    event_counts_by_rule: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class Trajectory:
    samples: tuple[Sample, ...]

    def at(self, t: float) -> Sample:
        for s in self.samples:
            if s.t_s >= t:
                return s
        return self.samples[-1]

    def pool_series(self, name: str) -> list[tuple[float, float]]:
        return [(s.t_s, s.pools.get(name, 0.0)) for s in self.samples]


@dataclass(frozen=True, slots=True)
class Prediction:
    locus_tag: str
    gene_name: str
    essential: bool
    time_to_failure_s: float | None
    failure_mode: FailureMode
    confidence: float

    def as_row(self) -> dict:
        return {
            "locus_tag": self.locus_tag,
            "gene_name": self.gene_name,
            "essential": int(self.essential),
            "time_to_failure_s": (
                "" if self.time_to_failure_s is None else self.time_to_failure_s
            ),
            "failure_mode": self.failure_mode.value,
            "confidence": f"{self.confidence:.4f}",
        }


class Simulator(Protocol):
    """Anything that can produce a Trajectory given a knockout list."""
    def run(self, knockout: Iterable[str], *, t_end_s: float,
            sample_dt_s: float) -> Trajectory: ...


@dataclass
class FailureDetector:
    """Compare a knockout trajectory against a wild-type baseline."""
    wt: Trajectory
    thresholds: dict[str, float] = field(default_factory=lambda: dict(THRESHOLDS))

    def detect(self, ko: Trajectory) -> tuple[FailureMode, float | None, float]:
        """Return (mode, time_s, confidence).

        Walks paired samples; a signature trips the **second** time it
        crosses its threshold in a row. We report the time of the
        **first** of those two samples. Confidence is `1 - min_ratio`
        clamped to [0, 1]."""
        return _detect_impl(self.wt, ko, self.thresholds)


def _detect_impl(
    wt: Trajectory, ko: Trajectory, thresholds: dict[str, float]
) -> tuple[FailureMode, float | None, float]:
    # Align samples by index (simulators sample on the same grid)
    n = min(len(wt.samples), len(ko.samples))
    prev_trips: dict[str, tuple[float, float]] = {}
    # We iterate samples and for each checked signal record (t_first_trip, ratio).
    # If the signal trips at the next sample too, we confirm.
    checked = [
        ("ATP", "ATP", FailureMode.ATP_DEPLETION, thresholds["ATP"]),
        ("CHARGED_TRNA_FRACTION", "CHARGED_TRNA_FRACTION",
         FailureMode.TRANSLATION_STALL, thresholds["translation"]),
        ("NTP_TOTAL", "NTP_TOTAL",
         FailureMode.TRANSCRIPTION_STALL, thresholds["transcription"]),
    ]
    em_thr = thresholds["essential_metabolite"]
    for idx in range(n):
        wt_s = wt.samples[idx]
        ko_s = ko.samples[idx]
        t = ko_s.t_s

        def ratio(key: str) -> float | None:
            w = wt_s.pools.get(key)
            k = ko_s.pools.get(key)
            if w is None or k is None or w <= 0:
                return None
            return k / w

        for key, _, mode, thr in checked:
            r = ratio(key)
            if r is None:
                continue
            if r < thr:
                if key in prev_trips:
                    t_first, r_first = prev_trips[key]
                    conf = _conf(min(r_first, r))
                    return mode, t_first, conf
                prev_trips[key] = (t, r)
            else:
                prev_trips.pop(key, None)

        # Essential metabolites: we collapse all of them into one mode but
        # remember which one tripped first.
        for met in ESSENTIAL_METABOLITES:
            r = ratio(met)
            if r is None:
                continue
            key = f"em:{met}"
            if r < em_thr:
                if key in prev_trips:
                    t_first, r_first = prev_trips[key]
                    conf = _conf(min(r_first, r))
                    return (
                        FailureMode.ESSENTIAL_METABOLITE_DEPLETION,
                        t_first,
                        conf,
                    )
                prev_trips[key] = (t, r)
            else:
                prev_trips.pop(key, None)

    return FailureMode.NONE, None, 0.0


def _conf(r: float) -> float:
    return max(0.0, min(1.0, 1.0 - r))


# -------- KnockoutHarness --------


@dataclass
class KnockoutHarness:
    """Main entry point: given a WT simulator + a knockout-aware
    simulator, produce a Prediction for a single gene."""
    wt_simulator: Simulator
    ko_simulator: Simulator
    t_end_s: float = 7200.0
    sample_dt_s: float = 60.0
    wt_cache: Trajectory | None = None

    def ensure_wt(self) -> Trajectory:
        if self.wt_cache is None:
            self.wt_cache = self.wt_simulator.run(
                [], t_end_s=self.t_end_s, sample_dt_s=self.sample_dt_s
            )
        return self.wt_cache

    def predict(self, locus_tag: str, gene_name: str = "") -> Prediction:
        wt = self.ensure_wt()
        ko = self.ko_simulator.run(
            [locus_tag], t_end_s=self.t_end_s, sample_dt_s=self.sample_dt_s
        )
        mode, t_fail, conf = FailureDetector(wt=wt).detect(ko)
        essential = mode != FailureMode.NONE
        return Prediction(
            locus_tag=locus_tag,
            gene_name=gene_name,
            essential=essential,
            time_to_failure_s=t_fail,
            failure_mode=mode,
            confidence=conf,
        )


# -------- Synthetic simulator for tests --------


@dataclass
class MockSimulator:
    """In-memory simulator that replays pre-built trajectories. Used by
    tests so the full Layer 3-4 metabolic stack isn't needed."""
    responses: dict[str, Trajectory]
    default: Trajectory

    def run(self, knockout: Iterable[str], *, t_end_s: float,
            sample_dt_s: float) -> Trajectory:
        kos = tuple(sorted(knockout))
        if not kos:
            return self.default
        # Use the response for the first knocked-out gene if present,
        # else default.
        for k in kos:
            if k in self.responses:
                return self.responses[k]
        return self.default
