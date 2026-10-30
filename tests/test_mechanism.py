"""V1-V4 for the committor/flux machinery. The verification IS the deliverable."""
import numpy as np
import pytest
from rem.mechanism import (committor, committor_bd_closed_form, harmonic_residual,
                           spans_band, reactive_flux, flux_balance, nonreversibility,
                           committor_mc, asym_toggle, corner_sets, stationary_of, verify)
from rem.switching import bd_generator, toggle_generator


def _chain(N=40, a=5, b=30):
    birth = 6.0 * np.ones(N + 1)
    death = 6.48 * np.ones(N + 1); death[0] = 0.0
    Q = bd_generator(birth, death, N)
    A = np.zeros(N + 1, bool); A[: a + 1] = True
    B = np.zeros(N + 1, bool); B[b:] = True
    return Q, A, B, birth, death, a, b


def test_v1_closed_form():
    Q, A, B, birth, death, a, b = _chain()
    q = committor(Q, A, B)
    qe = committor_bd_closed_form(birth, death, a, b)
    assert np.abs(q - qe).max() < 1e-12
    assert harmonic_residual(Q, q, A, B) < 1e-12
    assert np.abs(q[A]).max() == 0.0 and np.abs(q[B] - 1).max() == 0.0


def test_v1a_spanning_gate_rejects_a_vacuous_region():
    """The gate must FAIL a test region pinned near a boundary, which is the whole point."""
    assert not spans_band(np.array([0.990, 0.995, 0.999]))["ok"]
    assert not spans_band(np.array([0.001, 0.002, 0.004]))["ok"]
    assert spans_band(np.array([0.08, 0.20, 0.37, 0.63, 0.83]))["ok"]


def test_v1a_the_real_test_region_spans():
    Q, A, B, _bi, _d, a, b = _chain()
    q = committor(Q, A, B)
    assert spans_band(q[a + 1:b])["ok"]


def test_v2_monte_carlo_agrees():
    Q, A, B, _bi, _d, _a, _b = _chain()
    q = committor(Q, A, B)
    tested = []
    for st in (10, 20, 28):
        p, se = committor_mc(Q, A, B, st, n_traj=6000, seed=st)
        assert abs(q[st] - p) < 3.5 * se, (st, q[st], p, se)
        tested.append(q[st])
    assert spans_band(tested, need=2)["ok"]


def test_v3_symmetry_and_the_trap():
    M = 20
    Q, _n = toggle_generator(M, g=16.0, gamma=1.0, K=8.0, h=2.0)
    idx = lambda i, j: i * (M + 1) + j
    A, B = corner_sets(M, idx, lo=3, hi=12)
    q = committor(Q, A, B)
    sw = np.array([[q[idx(i, j)] + q[idx(j, i)] for j in range(M + 1)]
                   for i in range(M + 1)])
    assert np.abs(sw - 1.0).max() < 1e-10
    diag = np.array([q[idx(i, i)] for i in range(4, M - 3)])
    # THE TRAP, asserted: symmetry pins the diagonal at 0.5 whatever the mechanism, so a
    # symmetric system can never test whether x - y is a good reaction coordinate.
    assert np.abs(diag - 0.5).max() < 1e-9


def test_v4_flux_conservation_where_the_formula_is_exact():
    Q, A, B, _bi, _d, _a, _b = _chain()
    q = committor(Q, A, B)
    pi = stationary_of(Q)
    assert nonreversibility(Q, pi) < 1e-10          # birth-death chains are reversible
    bal = flux_balance(reactive_flux(Q, pi, q), A, B)
    assert bal["AB_imbalance"] < 1e-8
    assert bal["max_divergence_T"] < 1e-8


def test_v4_2d_toggle_is_not_reversible():
    """The simplified flux is the reversible form; assert the 2D system breaks detailed
    balance so a nonzero imbalance there is attributed rather than blamed on the solver."""
    M = 16
    Q, _n = toggle_generator(M, g=14.0, gamma=1.0, K=7.0, h=2.0)
    pi = stationary_of(Q)
    assert nonreversibility(Q, pi) > 1e-4


def test_verify_gates_pass():
    out = verify(verbose=False, n_traj=4000)
    assert out["V1"] and out["V1a"] and out["V3"] and out["V4"]
