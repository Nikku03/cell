"""Unit tests for ShortWindowDetector and RealSimulator.

The RealSimulator tests are gated on the Luthey-Schulten input data
being staged on disk; they skip cleanly otherwise.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cell_sim.layer6_essentiality.harness import (
    FailureMode, Sample, Trajectory,
)
from cell_sim.layer6_essentiality.short_window_detector import (
    ShortWindowDetector, calibrate_noise_floor,
)


def _traj(items: list[tuple[float, dict[str, float]]]) -> Trajectory:
    return Trajectory(tuple(Sample(t_s=t, pools=p) for t, p in items))


# ---- ShortWindowDetector ----


def test_short_detector_requires_two_consecutive_same_direction():
    wt = _traj([(0, {"F6P": 100}), (0.05, {"F6P": 100}),
                (0.10, {"F6P": 100}), (0.15, {"F6P": 100})])
    # Single-sample dip should NOT trip.
    ko_blip = _traj([(0, {"F6P": 100}), (0.05, {"F6P": 80}),
                     (0.10, {"F6P": 100}), (0.15, {"F6P": 100})])
    mode, t, conf, ev = ShortWindowDetector(wt).detect(ko_blip)
    assert mode == FailureMode.NONE


def test_short_detector_trips_on_two_consecutive_depletion():
    wt = _traj([(0, {"F6P": 100}), (0.05, {"F6P": 100}),
                (0.10, {"F6P": 100}), (0.15, {"F6P": 100})])
    ko = _traj([(0, {"F6P": 100}), (0.05, {"F6P": 85}),
                (0.10, {"F6P": 80}), (0.15, {"F6P": 80})])
    mode, t, conf, ev = ShortWindowDetector(wt).detect(ko)
    assert mode == FailureMode.ESSENTIAL_METABOLITE_DEPLETION
    assert t == pytest.approx(0.05)
    assert "F6P" in ev
    assert conf >= 0.15  # min |dev| at trip


def test_short_detector_trips_on_substrate_buildup():
    """A substrate that ACCUMULATES upstream of a knocked-out enzyme
    must also flag failure (the depletion-only default detector misses
    this; that's the whole reason this class exists)."""
    wt = _traj([(0, {"G6P": 100}), (0.05, {"G6P": 100}),
                (0.10, {"G6P": 100}), (0.15, {"G6P": 100})])
    ko = _traj([(0, {"G6P": 100}), (0.05, {"G6P": 115}),
                (0.10, {"G6P": 120}), (0.15, {"G6P": 120})])
    mode, t, conf, ev = ShortWindowDetector(wt).detect(ko)
    assert mode == FailureMode.ESSENTIAL_METABOLITE_DEPLETION
    assert "G6P" in ev


def test_short_detector_atp_maps_to_atp_depletion_mode():
    wt = _traj([(0, {"ATP": 100}), (0.05, {"ATP": 100}),
                (0.10, {"ATP": 100})])
    ko = _traj([(0, {"ATP": 100}), (0.05, {"ATP": 80}),
                (0.10, {"ATP": 70})])
    mode, _, _, ev = ShortWindowDetector(wt).detect(ko)
    assert mode == FailureMode.ATP_DEPLETION
    assert "ATP" in ev


def test_short_detector_ntp_total_maps_to_transcription_stall():
    wt = _traj([(0, {"NTP_TOTAL": 100}), (0.05, {"NTP_TOTAL": 100}),
                (0.10, {"NTP_TOTAL": 100})])
    ko = _traj([(0, {"NTP_TOTAL": 100}), (0.05, {"NTP_TOTAL": 80}),
                (0.10, {"NTP_TOTAL": 60})])
    mode, _, _, ev = ShortWindowDetector(wt).detect(ko)
    assert mode == FailureMode.TRANSCRIPTION_STALL


def test_short_detector_ignores_direction_flip():
    """Spike up then crash down should not confirm — flips break the streak."""
    wt = _traj([(0, {"F6P": 100}), (0.05, {"F6P": 100}),
                (0.10, {"F6P": 100})])
    ko = _traj([(0, {"F6P": 100}), (0.05, {"F6P": 120}),
                (0.10, {"F6P": 60})])
    mode, _, _, _ = ShortWindowDetector(wt).detect(ko)
    assert mode == FailureMode.NONE


def test_calibrate_noise_floor_returns_max_per_pool():
    wt = _traj([(0, {"ATP": 100, "G6P": 100}),
                (0.05, {"ATP": 100, "G6P": 100})])
    nons = [
        _traj([(0, {"ATP": 100, "G6P": 100}),
               (0.05, {"ATP":  98, "G6P": 105})]),
        _traj([(0, {"ATP": 100, "G6P": 100}),
               (0.05, {"ATP": 102, "G6P":  97})]),
    ]
    floor = calibrate_noise_floor(wt, nons)
    assert floor["ATP"] == pytest.approx(0.02, rel=1e-3)
    assert floor["G6P"] == pytest.approx(0.05, rel=1e-3)


# ---- RealSimulator ----


GB_PATH = Path(__file__).resolve().parents[1] / "data" / "Minimal_Cell_ComplexFormation" / "input_data" / "syn3A.gb"


@pytest.mark.skipif(
    not GB_PATH.exists(),
    reason="Luthey-Schulten data not staged; see memory_bank/data/STAGING.md",
)
def test_real_simulator_smoke():
    """Tiny end-to-end: WT run produces samples with the expected pool
    keys. We deliberately use very small scale so the test is <30 s."""
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(scale_factor=0.02, seed=42))
    traj = sim.run([], t_end_s=0.05, sample_dt_s=0.05)
    assert len(traj.samples) >= 2
    assert "ATP" in traj.samples[0].pools
    assert "NTP_TOTAL" in traj.samples[0].pools
    assert traj.samples[0].pools["ATP"] > 0
