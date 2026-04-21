"""Detector tuned for short-window (sub-second bio-time) real simulations.

The default :class:`FailureDetector` looks for **depletion** at fixed
thresholds (``ratio < 0.5`` for ATP, ``< 0.2`` for essential metabolites).
That's the right semantics for a full doubling-time run where pools
visibly crash.

For sub-second runs at small population scale we observe two different
phenomena:

1. The upstream metabolite of a knocked-out enzyme **accumulates**
   (e.g. G6P rises ~3 % in pgi-KO over 0.5 s). The default detector
   misses this because it only checks the depletion direction.

2. The downstream metabolite **depletes**, but only mildly within
   0.5 s (e.g. F6P falls 11 % in pgi-KO). A 0.2 hard threshold misses
   this entirely.

This detector replaces hard thresholds with a per-pool **bidirectional
deviation** check ``|ko/wt - 1| > X`` and requires two consecutive
samples above threshold (rejecting single-sample stochastic spikes).
The deviation threshold is calibrated against the noise floor observed
on Breuer-non-essential genes at the same scale + seed.

The trade-off: this detector is more sensitive but also more dependent
on the noise floor staying flat. We characterise the floor empirically
and surface it in the predictions so the user can audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from cell_sim.layer6_essentiality.harness import (
    ESSENTIAL_METABOLITES, FailureMode, Trajectory,
)


# Pools watched for deviation. Excludes NADH/NAD because pool sizes are
# too small at scale<=0.1 to give a stable noise floor (we see ~10 %
# wobble even on non-essentials).
SHORT_WINDOW_POOLS: tuple[str, ...] = (
    # metabolite pools
    "ATP", "G6P", "F6P", "PYR", "CTP", "GTP", "UTP", "NTP_TOTAL",
    "dATP", "dGTP", "dCTP", "dTTP",
    # non-metabolic simulator-state signals (protein folding + complex
    # assembly); added in Session 6 to catch ribosomal / tRNA-synthetase
    # KOs that don't perturb central-carbon pools within 0.5 s.
    "TOTAL_COMPLEXES", "FOLDED_PROTEINS", "UNFOLDED_PROTEINS",
    "FOLDED_FRACTION", "BOUND_PROTEINS",
    # Cumulative event count. Sub-% noise at scale=0.05 (~140k events
    # in 0.5 s), so a KO that drops aggregate enzyme activity becomes
    # detectable even if no single metabolite pool moves much.
    "TOTAL_EVENTS",
)


@dataclass
class ShortWindowDetector:
    """Bidirectional deviation detector for short-window real runs.

    ``deviation_threshold`` may be either a single ``float`` (global
    threshold applied to every pool) or a ``dict[str, float]`` with
    per-pool thresholds. Pools missing from the dict fall back to
    ``fallback_threshold``.

    Default 0.10 catches the pgi-KO F6P depletion (~11 %) on the
    reference panel while staying above non-essential noise (~3 %)
    at scale 0.05 + seed 42 + t_end 0.5 s. Tune with
    :func:`calibrate_noise_floor` for higher scales / longer windows.
    """

    wt: Trajectory
    deviation_threshold: float | dict[str, float] = 0.10
    pools: tuple[str, ...] = field(default_factory=lambda: SHORT_WINDOW_POOLS)
    fallback_threshold: float = 0.10

    def _threshold_for(self, pool: str) -> float:
        thr = self.deviation_threshold
        if isinstance(thr, dict):
            return float(thr.get(pool, self.fallback_threshold))
        return float(thr)

    def detect(self, ko: Trajectory) -> tuple[FailureMode, float | None, float, str]:
        """Return ``(mode, time_s, confidence, evidence)``.

        ``evidence`` is a short string naming the pool and direction
        that tripped, e.g. ``"F6P -0.11"``. Empty if nothing tripped.
        """
        n = min(len(self.wt.samples), len(ko.samples))
        prev: dict[str, tuple[float, float]] = {}
        for idx in range(n):
            wt_s = self.wt.samples[idx]
            ko_s = ko.samples[idx]
            t = ko_s.t_s
            for pool in self.pools:
                w = wt_s.pools.get(pool)
                k = ko_s.pools.get(pool)
                if w is None or k is None or w <= 0:
                    continue
                dev = (k - w) / w
                thr = self._threshold_for(pool)
                if abs(dev) >= thr:
                    if pool in prev:
                        t_first, dev_first = prev[pool]
                        if (dev_first > 0) == (dev > 0):
                            mode = _mode_for_pool(pool, dev)
                            confidence = min(1.0, max(abs(dev_first), abs(dev)))
                            evidence = f"{pool} {dev_first:+.3f}->{dev:+.3f}"
                            return mode, t_first, confidence, evidence
                    prev[pool] = (t, dev)
                else:
                    prev.pop(pool, None)
        return FailureMode.NONE, None, 0.0, ""


def _mode_for_pool(pool: str, dev: float) -> FailureMode:
    if pool == "ATP":
        return FailureMode.ATP_DEPLETION
    if pool == "NTP_TOTAL":
        return FailureMode.TRANSCRIPTION_STALL
    if pool in ESSENTIAL_METABOLITES:
        return FailureMode.ESSENTIAL_METABOLITE_DEPLETION
    return FailureMode.ESSENTIAL_METABOLITE_DEPLETION


def calibrate_noise_floor(
    wt: Trajectory,
    nonessential_runs: list[Trajectory],
    pools: tuple[str, ...] = SHORT_WINDOW_POOLS,
) -> dict[str, float]:
    """Return the max |deviation| observed per pool across a panel of
    Breuer-non-essential KOs. Use this to set a sensible
    ``deviation_threshold`` for ``ShortWindowDetector``: anything above
    the max non-essential deviation is signal, below is noise."""
    max_dev: dict[str, float] = {p: 0.0 for p in pools}
    for ko in nonessential_runs:
        n = min(len(wt.samples), len(ko.samples))
        for idx in range(n):
            ws = wt.samples[idx]
            ks = ko.samples[idx]
            for pool in pools:
                w = ws.pools.get(pool)
                k = ks.pools.get(pool)
                if w is None or k is None or w <= 0:
                    continue
                dev = abs((k - w) / w)
                if dev > max_dev[pool]:
                    max_dev[pool] = dev
    return max_dev
