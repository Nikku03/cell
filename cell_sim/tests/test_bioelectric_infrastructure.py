"""Phase B1 bioelectric-infrastructure tests.

Verifies the architectural contract of ``cell_sim/layer5_bioelectric/``
without claiming a measurement. Specifically:

* The package and its submodules import cleanly.
* The Nernst and GHK computations produce numerically sensible values
  on a hand-constructed state (matches a textbook calculation).
* The ``enable_bioelectric`` flag defaults to False on
  ``RealSimulatorConfig``.
* Phase 2 gex-off bit-identity (separate test) still passes — i.e.
  flag-off behavior is unchanged. We don't run the full regression
  test here (it's slow and opt-in), but we verify the snapshot helper
  with ``enable_bioelectric=False`` does NOT add a VMEM_MV key, so
  the flag-off path stays unchanged.
* The feature extractor handles both populated and empty trajectories.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_CELL_SIM = Path(__file__).resolve().parents[1]
if str(_CELL_SIM) not in sys.path:
    sys.path.insert(0, str(_CELL_SIM))


# ---------------------------------------------------------------------
# 1. Package imports
# ---------------------------------------------------------------------


def test_bioelectric_package_imports_cleanly():
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from cell_sim.layer5_bioelectric import (  # noqa: F401
            BioelectricEstimatorConfig,
            BioelectricFeatures,
            DEFAULT_PERMEABILITY,
            DEFAULT_REFERENCE_EXTRACELLULAR_MM,
            estimate_vmem_mv,
            extract_bioelectric_features,
            nernst_potential_mv,
        )
    assert buf.getvalue() == "", (
        f"importing cell_sim.layer5_bioelectric produced unexpected "
        f"stdout: {buf.getvalue()!r}"
    )


# ---------------------------------------------------------------------
# 2. Nernst potential numerical sanity
# ---------------------------------------------------------------------


def test_nernst_potential_zero_when_concentrations_equal():
    from cell_sim.layer5_bioelectric import nernst_potential_mv

    e = nernst_potential_mv(10.0, 10.0, charge=+1)
    assert e == pytest.approx(0.0, abs=1e-9)


def test_nernst_potential_textbook_value():
    """At 37 C, RT/F ~= 26.73 mV. ln(10) ~= 2.302. So a 10x outside-
    inside ratio for a +1 cation gives ~+61.5 mV."""
    from cell_sim.layer5_bioelectric import nernst_potential_mv

    e = nernst_potential_mv(1.0, 10.0, charge=+1, temperature_K=310.15)
    assert 60.0 < e < 63.0, f"expected ~+61.5 mV, got {e}"


def test_nernst_potential_returns_zero_on_zero_concentration():
    from cell_sim.layer5_bioelectric import nernst_potential_mv

    assert nernst_potential_mv(0.0, 10.0) == 0.0
    assert nernst_potential_mv(10.0, 0.0) == 0.0


# ---------------------------------------------------------------------
# 3. GHK Vmem estimator on a hand-built state
# ---------------------------------------------------------------------


def _stub_state(k_in_mM=10.0, na_in_mM=10.0, cl_in_mM=0.0,
                vol_L=3.35e-17):
    """Build a minimal object with the attributes estimate_vmem_mv reads."""
    AVOGADRO = 6.022e23

    class _S:
        pass

    s = _S()
    s.metabolite_volume_L = vol_L

    def _count(c_mM):
        # inverse of count_to_mM: count = c_mM * 1e-3 * vol_L * AVOGADRO
        return int(round(c_mM * 1e-3 * vol_L * AVOGADRO))

    s.metabolite_counts = {
        "M_k_c": _count(k_in_mM),
        "M_na1_c": _count(na_in_mM),
        "M_cl_c": _count(cl_in_mM),
    }
    return s


def test_ghk_vmem_at_syn3a_resting_is_small_positive_mv():
    """With Luthey-Schulten Syn3A reference values
    ([K]_i=10, [Na]_i=10, [Cl]_i=0; [K]_o=12.67, [Na]_o=19.38,
    [Cl]_o=19.9) and bacterial-typical permeability ratios (P_K=1,
    P_Na=0.04, P_Cl=0.45), the GHK Vmem is small (single-digit
    positive mV). This is a feature of Syn3A's engineered minimal
    medium, not a bug — typical bacteria are -100 to -180 mV but
    Syn3A's medium has near-equal [K] inside and outside."""
    from cell_sim.layer5_bioelectric import estimate_vmem_mv

    s = _stub_state(k_in_mM=10.0, na_in_mM=10.0, cl_in_mM=0.0)
    v = estimate_vmem_mv(s)
    assert math.isfinite(v)
    # Small positive Vmem expected at the Syn3A defaults. Allow some
    # slack for the Cl- floor.
    assert -30.0 < v < 30.0, (
        f"expected near-zero Vmem at Syn3A defaults, got {v} mV. "
        f"This bound exists to flag any large change in the GHK "
        f"computation; if you intentionally changed permeability "
        f"defaults or extracellular references, update the bound."
    )


def test_ghk_vmem_more_negative_when_kin_larger():
    """If [K]_i goes from 10 to 200 mM (typical bacterial range) with
    [K]_o fixed at 12.67, V_m must drop substantially. This sanity-
    checks the GHK direction."""
    from cell_sim.layer5_bioelectric import estimate_vmem_mv

    s_low = _stub_state(k_in_mM=10.0)
    s_high = _stub_state(k_in_mM=200.0)
    v_low = estimate_vmem_mv(s_low)
    v_high = estimate_vmem_mv(s_high)
    assert v_high < v_low - 30.0, (
        f"raising [K]_i from 10 to 200 mM should drop Vmem by at "
        f"least 30 mV; got v_low={v_low}, v_high={v_high}"
    )


def test_ghk_vmem_returns_nan_on_missing_state():
    from cell_sim.layer5_bioelectric import estimate_vmem_mv

    class _Empty:
        pass

    s = _Empty()
    v = estimate_vmem_mv(s)
    assert math.isnan(v)

    class _NoVol:
        metabolite_counts = {}
        metabolite_volume_L = 0.0
    assert math.isnan(estimate_vmem_mv(_NoVol()))


# ---------------------------------------------------------------------
# 4. RealSimulatorConfig flag and snapshot wiring
# ---------------------------------------------------------------------


def test_real_simulator_config_default_is_flag_off():
    from cell_sim.layer6_essentiality.real_simulator import RealSimulatorConfig

    cfg = RealSimulatorConfig()
    assert cfg.enable_bioelectric is False, (
        "Phase B1 must default enable_bioelectric=False so flag-off "
        "behavior is bit-identical to v15 / v16."
    )


def test_snapshot_omits_vmem_when_flag_off():
    """Build a stub state with the metabolite_counts / volume the
    estimator needs, call _snapshot with enable_bioelectric=False,
    confirm no VMEM_MV key in the resulting pools dict."""
    from cell_sim.layer6_essentiality.real_simulator import _snapshot

    class _Stub:
        metabolite_counts = {"M_k_c": 1, "M_na1_c": 1, "M_cl_c": 0}
        metabolite_volume_L = 3.35e-17
        complexes = {}
        proteins = {}
        proteins_by_state = {}
        events = []

    def _gsc(state, sid):
        # Minimal stub: never matches anything in _POOL_KEY_TO_SPECIES.
        raise KeyError(sid)

    sample = _snapshot(_Stub(), _gsc, t=0.0, enable_bioelectric=False)
    assert "VMEM_MV" not in sample.pools


def test_snapshot_includes_vmem_when_flag_on():
    from cell_sim.layer6_essentiality.real_simulator import _snapshot

    AVOGADRO = 6.022e23
    vol_L = 3.35e-17

    def _count(c_mM):
        return int(round(c_mM * 1e-3 * vol_L * AVOGADRO))

    class _Stub:
        metabolite_counts = {
            "M_k_c": _count(10.0),
            "M_na1_c": _count(10.0),
            "M_cl_c": _count(0.0),
        }
        metabolite_volume_L = vol_L
        complexes = {}
        proteins = {}
        proteins_by_state = {}
        events = []

    def _gsc(state, sid):
        raise KeyError(sid)

    sample = _snapshot(_Stub(), _gsc, t=0.0, enable_bioelectric=True)
    assert "VMEM_MV" in sample.pools
    v = sample.pools["VMEM_MV"]
    assert math.isfinite(v)
    assert -30.0 < v < 30.0


# ---------------------------------------------------------------------
# 5. Feature extractor
# ---------------------------------------------------------------------


def _stub_trajectory(vmem_values: list):
    """Mirror cell_sim.layer6_essentiality.harness.Sample / Trajectory
    surface enough to drive the extractor."""

    class _Sample:
        def __init__(self, v):
            self.pools = {"VMEM_MV": v} if v is not None else {}

    class _Traj:
        def __init__(self, vs):
            self.samples = [_Sample(v) for v in vs]

    return _Traj(vmem_values)


def test_feature_extractor_handles_empty_trajectory():
    from cell_sim.layer5_bioelectric import extract_bioelectric_features

    feat = extract_bioelectric_features(_stub_trajectory([]))
    assert feat.has_data is False
    assert math.isnan(feat.vmem_mean_mv)
    assert math.isnan(feat.vmem_final_mv)
    assert math.isnan(feat.vmem_range_mv)
    assert feat.vmem_dev_from_wt_mv is None


def test_feature_extractor_handles_all_nan_trajectory():
    from cell_sim.layer5_bioelectric import extract_bioelectric_features

    feat = extract_bioelectric_features(
        _stub_trajectory([float("nan"), float("nan"), None])
    )
    assert feat.has_data is False


def test_feature_extractor_computes_summary():
    from cell_sim.layer5_bioelectric import extract_bioelectric_features

    feat = extract_bioelectric_features(_stub_trajectory([1.0, 2.0, 3.0]))
    assert feat.has_data is True
    assert feat.vmem_mean_mv == pytest.approx(2.0)
    assert feat.vmem_final_mv == pytest.approx(3.0)
    assert feat.vmem_range_mv == pytest.approx(2.0)
    assert feat.vmem_dev_from_wt_mv is None


def test_feature_extractor_dev_from_wt():
    from cell_sim.layer5_bioelectric import extract_bioelectric_features

    feat = extract_bioelectric_features(
        _stub_trajectory([2.0, 4.0]),
        wt_reference_mean_mv=2.5,
    )
    assert feat.vmem_dev_from_wt_mv == pytest.approx(0.5)


# ---------------------------------------------------------------------
# 6. Production sweep wiring stays inert at flag-off
# ---------------------------------------------------------------------


def test_real_simulator_does_not_invoke_bioelectric_at_flag_off():
    """The wiring touch in real_simulator.py is minimal: a flag, an
    arg pass, and a conditional pool addition. We guard the contract
    by reading the source file and confirming the bioelectric import
    is INSIDE the conditional block — not at module top-level. This
    protects the flag-off path from any future bioelectric module
    side effect."""
    real_sim = (
        Path(__file__).resolve().parents[1]
        / "layer6_essentiality" / "real_simulator.py"
    )
    text = real_sim.read_text()

    # The only top-level bioelectric reference allowed is the config
    # field declaration (a plain bool). Imports MUST happen inside a
    # conditional block so module import is unaffected.
    lines = text.splitlines()
    top_level_bio = [
        i for i, line in enumerate(lines)
        if line.startswith("from cell_sim.layer5_bioelectric")
        or line.startswith("import cell_sim.layer5_bioelectric")
    ]
    assert top_level_bio == [], (
        "cell_sim.layer5_bioelectric must be imported INSIDE a "
        "conditional, not at module top-level, to protect the "
        "flag-off code path from side effects."
    )
    assert "enable_bioelectric: bool = False" in text, (
        "RealSimulatorConfig.enable_bioelectric must default to False."
    )
    assert "VMEM_MV" in text, (
        "real_simulator.py must inject VMEM_MV into pools when "
        "enable_bioelectric is True."
    )
