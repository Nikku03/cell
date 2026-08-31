"""The closed form is the deliverable, so it is tested against exact solves."""
import numpy as np
from rem.crowding_errorbar import (implied_correction_error, rare_error_bound, sensitivity)


def _bd_mean(b, M=60, gamma=1.0):
    from rem.rare import Network, Reaction, stationary
    net = Network(["X"], [M], [
        Reaction("X+", lambda S: np.full(len(S), b), (1,)),
        Reaction("X-", lambda S: gamma * S[:, 0], (-1,))])
    p, _ = stationary(net)
    return float((np.arange(len(p)) * p).sum())


def test_closed_form_matches_exact_solve():
    """exp[(N - <X>) * eps] against the real ratio, in the small-error regime it is for."""
    from rem.rare import Network, Reaction, stationary

    def solve(b, M=60):
        net = Network(["X"], [M], [
            Reaction("X+", lambda S: np.full(len(S), b), (1,)),
            Reaction("X-", lambda S: 1.0 * S[:, 0], (-1,))])
        return stationary(net)[0]

    lam = 8.0
    base = solve(lam)
    mean0 = float((np.arange(len(base)) * base).sum())
    for eps in (0.02, 0.05):
        p = solve(lam * (1 + eps))
        for N in (20, 30, 40):
            meas = float(p[N:].sum() / base[N:].sum())
            pred = rare_error_bound(eps, 1.0, N, mean0)
            assert abs(pred - meas) / meas < 0.07, (eps, N, meas, pred)


def test_sensitivity_is_one_without_feedback():
    S = sensitivity(lambda f: _bd_mean(8.0 * f))
    assert abs(S - 1.0) < 1e-3, S


def test_feedback_damps_sensitivity_and_widens_the_bound():
    """Under homeostasis the same proteomics agreement must buy a WEAKER bound, not a
    stronger one. Getting this backwards is the blind spot the formula exists to encode."""
    b_no = rare_error_bound(0.03, 1.0, 30, 8.0)
    b_fb = rare_error_bound(0.03, 0.5, 30, 8.0)
    assert b_fb > b_no
    assert rare_error_bound(0.03, 0.0, 30, 8.0) == float("inf")


def test_bound_widens_with_tail_distance():
    """The check gets weaker the rarer the event -- backwards from what one would want, so it
    is asserted rather than left as a remark."""
    bounds = [rare_error_bound(0.03, 1.0, N, 8.0) for N in (10, 20, 30, 50, 100)]
    assert bounds == sorted(bounds)
    assert bounds[0] < 1.1 and bounds[-1] > 10.0


def test_implied_error_divides_by_sensitivity():
    assert abs(implied_correction_error(0.03, 1.0) - 0.03) < 1e-12
    assert abs(implied_correction_error(0.03, 0.5) - 0.06) < 1e-12
