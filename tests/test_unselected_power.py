"""The forced-include cap is chosen against one chance function and the verdict computed
against another, in a different module. That is a drift hazard, and a comment is not a guard.

Ledger T was exactly this class of error: a design constant justified against a statistic the
test does not use. These tests hold the two implementations together and pin the cap's contract.
"""
import importlib

import pytest

U = importlib.import_module("benchmarks.db5_unselected")
A = importlib.import_module("benchmarks.db5_unselected_analysis")


def test_chance_hit_implementations_agree_exactly():
    """The duplicate must be a duplicate. Any divergence silently decouples cap from verdict."""
    for n in (50, 100, 510, 600, 1000):
        for k in (0, 1, 2, 5, 10, 37, 100, 300):
            if k > n:
                continue
            assert U.chance_hit(n, k) == pytest.approx(A.chance_hit(n, k), rel=1e-12, abs=1e-15)


def test_constants_match_the_analysis():
    assert U.TOPK == A.TOPK
    assert U.ALPHA == A.ALPHA


def test_cap_keeps_a_perfect_oracle_inside_alpha():
    """The cap's contract: an oracle must clear ALPHA/POWER_MARGIN at the returned cap."""
    for n_cx in (2, 3, 5, 10, 20):
        k = U.forced_cap_for_power(n_cx, 500)
        assert k > 0
        p = U.chance_hit(500 + k, k) ** n_cx
        assert p <= U.ALPHA / U.POWER_MARGIN


def test_cap_is_maximal():
    """One more forced pose must break the contract, or the cap is leaving power on the table."""
    for n_cx in (3, 5, 10):
        k = U.forced_cap_for_power(n_cx, 500)
        p_next = U.chance_hit(500 + k + 1, k + 1) ** n_cx
        assert p_next > U.ALPHA / U.POWER_MARGIN


def test_single_complex_cannot_be_powered_and_says_so():
    """The honest zero: with one complex there is no positive cap that keeps the gate powered."""
    assert U.forced_cap_for_power(1, 500) == 0


def test_the_rejected_cap_would_have_been_void():
    """Pin ledger T: the cap this run originally used cannot pass even with a perfect oracle."""
    p_oracle = U.chance_hit(600, 100) ** 5
    assert p_oracle > U.ALPHA
    assert p_oracle == pytest.approx(0.88, abs=0.02)
