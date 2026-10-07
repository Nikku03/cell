"""Tests for minimal-perturbation attribution.

These test the DISCIPLINE, not the lambda biology: that an unprovenanced knob can never win,
that an out-of-bounds knob is reported insufficient rather than silently clamped, and that the
plant-and-recover controls run in both directions.
"""
import math

import pytest

from rem import attribution as A


def test_uncostable_knob_can_never_be_selected():
    """The whole discipline: a knob with no provenance would otherwise reconcile at zero cost."""
    k = A.Knob("free", 1.0, A.UNCOSTABLE, (1e-12, 1e12), "NOT STATED")
    assert not k.costable
    assert k.cost(1.0) == float("inf")
    assert k.cost(1e6) == float("inf")
    res = A.attribute(A.burst_model, A.theta0(), A.lysogen_knobs(), A.S_OBS)
    assert all(r["knob"] not in ("f_assay", "k_extra") for r in res["costed"])
    assert {r["knob"] for r in res["uncostable"]} == {"f_assay", "k_extra"}


def test_out_of_bounds_knob_is_insufficient_not_clamped():
    res = A.attribute(A.burst_model, A.theta0(), A.lysogen_knobs(), A.S_OBS)
    assert "tau" in {r["knob"] for r in res["insufficient"]}


def test_vacuity_guard_fires_when_only_one_knob_can_reconcile():
    only = [A.Knob("k_on", A.K_ON_NOMINAL, A.K_ON_SIGMA_DEX, (0.01, 10.0), "m")]
    res = A.attribute(A.burst_model, A.theta0(), only, A.S_OBS)
    assert res["vacuous"]
    many = A.attribute(A.burst_model, A.theta0(), A.lysogen_knobs(), A.S_OBS)
    assert not many["vacuous"]


def test_sensitivity_equals_minus_n():
    """K4: the amplification is a derivative, and it equals the burst count exactly."""
    th = A.theta0()
    n = th["k_on"] * th["tau"] / math.log(2.0)
    assert A.sensitivity(A.burst_model, th, "k_on") == pytest.approx(-n, rel=1e-3)
    assert A.sensitivity(A.burst_model, th, "tau") == pytest.approx(-n, rel=1e-3)


def test_plant_and_recover_finds_a_costed_knob():
    res, _target, true_dex = A.plant_and_recover("k_on", 0.85)
    best = res["best"]
    assert best["knob"] == "k_on"
    assert best["dex"] == pytest.approx(true_dex, abs=0.05)
    assert best["cost"] <= 2.0


def test_plant_uncostable_is_not_cheaply_misattributed():
    res, _t = A.plant_uncostable(3.0e-9)
    assert res["best"] is None or res["best"]["cost"] > 2.0


def test_k1_fails_on_the_real_table_and_k5_names_the_missing_width():
    """The recorded result: the method locates k_on but rejects it at its own plausibility bar."""
    res = A.attribute(A.burst_model, A.theta0(), A.lysogen_knobs(), A.S_OBS)
    best = res["best"]
    assert best["knob"] == "k_on"
    assert not res["plausible"]
    need = A.required_width(best)
    assert need > A.K_ON_SIGMA_DEX
    assert need == pytest.approx(abs(best["dex"]) / 2.0)
