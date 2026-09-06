"""Bucket elimination against brute force. Everything else in REM builds on this."""
import itertools
import numpy as np
import pytest

from rem.factorgraph import FactorGraph, random_graph, verify, verify_pathwidth_trap
from rem import circulant

TOL = 1e-10


@pytest.mark.parametrize("seed", range(12))
def test_min_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    g = random_graph(rng, n=int(rng.integers(3, 7)), card=int(rng.integers(2, 4)),
                     n_factors=int(rng.integers(2, 7)), arity=int(rng.integers(2, 4)))
    val, arg, info = g.eliminate("min")
    ref, _ = g.brute_force("min")
    assert abs(val - ref) < TOL
    # the assignment must actually achieve the reported energy
    tot = sum(f.table[tuple(arg[v] for v in f.vars)] for f in g.factors)
    assert abs(tot - val) < TOL


@pytest.mark.parametrize("seed", range(12))
def test_sum_matches_brute_force(seed):
    rng = np.random.default_rng(100 + seed)
    g = random_graph(rng, n=int(rng.integers(3, 7)), card=int(rng.integers(2, 4)),
                     n_factors=int(rng.integers(2, 7)), arity=2)
    val, _, _ = g.eliminate("sum")
    ref, _ = g.brute_force("sum")
    assert abs(val - ref) < TOL


@pytest.mark.parametrize("seed", range(8))
def test_marginals_match_brute_force(seed):
    rng = np.random.default_rng(200 + seed)
    g = random_graph(rng, n=5, card=3, n_factors=5, arity=2)
    m, bm = g.marginals(), g.brute_force_marginals()
    for v in m:
        assert np.max(np.abs(m[v] - bm[v])) < TOL


def test_orderings_agree_on_value():
    """The optimum cannot depend on the elimination order; only the cost can."""
    rng = np.random.default_rng(7)
    g = random_graph(rng, n=6, card=3, n_factors=6, arity=2)
    ref, _, _ = g.eliminate("min")
    names = list(g.cards)
    for _ in range(8):
        order = list(rng.permutation(names))
        val, _, _ = g.eliminate("min", order=order)
        assert abs(val - ref) < TOL


def test_pathwidth_trap_chain_stays_at_d_squared():
    r = verify_pathwidth_trap(n=40, card=5, verbose=False)
    assert r["treewidth"] == 1
    assert r["largest_table"] <= 25


def test_treewidth_known_graphs():
    def grid(rows, cols, d=2):
        g = FactorGraph()
        nm = lambda r, c: f"v{r}_{c}"
        for r in range(rows):
            for c in range(cols):
                g.add_var(nm(r, c), d)
        for r in range(rows):
            for c in range(cols):
                if r + 1 < rows: g.add_factor([nm(r, c), nm(r + 1, c)], np.zeros((d, d)))
                if c + 1 < cols: g.add_factor([nm(r, c), nm(r, c + 1)], np.zeros((d, d)))
        return g
    # a chain is treewidth 1; a k x m grid is treewidth min(k, m) and greedy must not beat it
    assert grid(1, 8).treewidth() == 1
    for k in (2, 3, 4):
        tw = grid(k, 6).treewidth()
        assert tw >= k, f"{k}x6 grid: greedy reported {tw}, below the true treewidth {k}"


def test_treewidth_wall_raises_not_hangs():
    g = FactorGraph()
    n, d = 24, 6
    for i in range(n):
        g.add_var(f"x{i}", d)
    for i in range(n):                      # dense graph -> treewidth ~ n
        for j in range(i + 1, n):
            g.add_factor([f"x{i}", f"x{j}"], np.zeros((d, d)))
    with pytest.raises(MemoryError, match="treewidth wall"):
        g.eliminate("min", max_table=1e6)


def test_verify_summary_is_exact():
    r = verify(seed=3, trials=15, verbose=False)
    assert r["max_err_min"] < TOL
    assert r["max_err_logZ"] < TOL
    assert r["max_err_marginal"] < TOL


def test_circulant_ring_and_fft():
    r = circulant.verify(verbose=False)
    assert r["max_err_ring"] < 1e-9
    assert r["max_err_circulant"] < 1e-9


def test_ring_closed_form_is_size_independent():
    """Z = tr(T^n) costs O(d^3); a million sites must not cost more than ten."""
    import time
    rng = np.random.default_rng(0)
    logT = rng.normal(size=(6, 6))
    ts = []
    for n in (10, 10 ** 6):
        t = time.perf_counter()
        for _ in range(20):
            circulant.ring_logZ_transfer(logT, n)
        ts.append(time.perf_counter() - t)
    assert ts[1] < 20 * ts[0] + 0.05
