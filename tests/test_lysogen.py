"""Tests for the lambda lysogen L0 gate.

These test the GUARDS, not the biology. The biology is unretrieved and the module says so; what
must not break is the machinery that keeps an unretrieved number from acquiring a value.
"""
import math

import pytest

from rem import lysogen as L


def test_unretrieved_parameters_cannot_carry_a_value():
    with pytest.raises(ValueError):
        L.Param("x", 1.0, "u", L.FITTED, ("s", "d"), retrieved=False)
    with pytest.raises(ValueError):
        L.Param("x", None, "u", L.MEASURED, ("s", "d"), retrieved=True)


def test_origin_and_retrieval_are_independent_axes():
    """The defect this file was patched for: a fitted parameter that could not be read must still
    be counted as fitted, or the summary states the opposite of the truth."""
    c = L.provenance()
    assert c["origin"][L.FITTED] >= 2, "k_max and mu(T) are both fitted"
    assert c["fitted_but_unretrieved"] >= 2
    assert c["unretrieved"] > 0


def test_burst_survival_matches_the_closed_form():
    p, n = L.burst_survival(1.4, 30.0)
    assert n == pytest.approx(1.4 * 30.0 / math.log(2.0))
    assert p == pytest.approx(math.exp(-n))
    # monotone: a longer generation means more chances to fire, so a smaller switching rate
    assert L.burst_survival(1.4, 60.0)[0] < L.burst_survival(1.4, 20.0)[0]


def test_wild_type_check_is_unfalsifiable_over_the_whole_bracket():
    f = L.wild_type_falsifiability()
    assert not f["falsifiable"]
    assert f["orders_below_floor_min"] > 0
    assert f["P_at_tau_min"] < L.DETECTION_FLOOR


def test_l3_power_is_the_best_case_not_the_observed_case():
    pw = L.l3_power(n_at_floor=12, n_too_unstable=4, n_discriminating=2)
    assert pw["best_attainable_p"] == pytest.approx(0.25)
    assert not pw["powered"]
    assert pw["n_needed_for_alpha"] == 5
    # a censored mutant must not buy power
    more_censored = L.l3_power(n_at_floor=120, n_too_unstable=40, n_discriminating=2)
    assert more_censored["best_attainable_p"] == pw["best_attainable_p"]
    # and a powered set must be reachable, or the gate could never pass (ledger defect P)
    assert L.l3_power(0, 0, 5)["powered"]


def test_l0_returns_partial_not_pass_while_inputs_are_missing():
    verdict, _counts, missing = L.l0_verdict()
    assert verdict == "PARTIAL"
    assert any("mu(T)" in p.name for p in missing)
    assert any("doubling time" in p.name for p in missing)
