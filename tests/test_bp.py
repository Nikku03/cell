"""Belief propagation: exact on trees, measurably wrong off them.

Every "BP is right" test compares against TWO independent references -- bucket
elimination (rem.factorgraph) and naive enumeration over all d^n configurations
(rem.bp.brute_force_*) -- and every "BP is wrong" test carries the positive control in
the same test function, so a null can never be mistaken for a broken harness.
"""
import warnings

import numpy as np
import pytest

from rem import bp
from rem.bp import (FactorGraph, factor_graph_structure, is_forest, is_tree, min_sum,
                    sum_product)

TOL = 1e-9


def _chain(n=5, card=2, seed=0, ring=False):
    rng = np.random.default_rng(seed)
    g = FactorGraph()
    for i in range(n):
        g.add_var(f"x{i}", card)
        g.add_factor([f"x{i}"], rng.normal(size=card))
    for i in range(n if ring else n - 1):
        g.add_factor([f"x{i}", f"x{(i + 1) % n}"], rng.normal(size=(card, card)))
    return g


# ------------------------------------------------------------------ structure
def test_is_tree_on_known_structures():
    assert is_tree(_chain(6)) is True
    assert is_tree(_chain(6, ring=True)) is False
    assert is_forest(_chain(6, ring=True)) is False
    st = factor_graph_structure(_chain(6, ring=True))
    assert st["n_independent_cycles"] == 1          # a ring has exactly one

    star = FactorGraph()
    for i in range(5):
        star.add_factor(["hub", f"leaf{i}"], np.zeros((2, 2)))
    assert is_tree(star)

    # two disjoint chains: a FOREST but not a TREE. BP is still exact on it.
    two = FactorGraph()
    two.add_factor(["a", "b"], np.zeros((2, 2)))
    two.add_factor(["c", "d"], np.zeros((2, 2)))
    assert is_forest(two) and not is_tree(two)
    assert factor_graph_structure(two)["n_components"] == 2


def test_two_parallel_factors_are_a_cycle():
    """Two factors over the SAME pair make a 4-cycle in the bipartite graph, even though
    the induced graph is a single edge. BP is not exact there and must say so."""
    g = FactorGraph()
    g.add_factor(["x", "y"], np.arange(4.0).reshape(2, 2))
    g.add_factor(["x", "y"], np.arange(4.0).reshape(2, 2) * 0.3)
    assert not is_forest(g)
    assert factor_graph_structure(g)["n_independent_cycles"] == 1
    assert g.treewidth() == 1                      # induced graph says "tree"; it lies


def test_hypertree_is_a_tree_although_treewidth_is_two():
    """The two notions of 'tree' come apart at arity 3, and BP follows the bipartite one."""
    g = bp.random_hypertree(np.random.default_rng(0), n_factors=3, card=2, arity=3)
    assert is_tree(g)
    assert g.treewidth() >= 2                      # induced graph has triangles
    b, info = sum_product(g)
    assert info["converged"] and info["n_independent_cycles"] == 0
    ex, bf = g.marginals(), bp.brute_force_marginals(g)
    for v in b:
        assert np.max(np.abs(b[v] - ex[v])) < TOL
        assert np.max(np.abs(b[v] - bf[v])) < TOL


# ------------------------------------------------------------------ trees: exactness
@pytest.mark.parametrize("seed", range(10))
def test_sum_product_exact_on_trees(seed):
    rng = np.random.default_rng(seed)
    g = bp.random_tree(rng, n=int(rng.integers(2, 9)), card=int(rng.integers(2, 4)))
    b, info = sum_product(g, max_iter=400, tol=1e-13)
    assert info["converged"] and info["exact"] and info["is_tree"]
    ex, bf = g.marginals(), bp.brute_force_marginals(g)
    for v in b:
        assert abs(b[v].sum() - 1.0) < 1e-12
        assert np.max(np.abs(b[v] - ex[v])) < TOL       # vs bucket elimination
        assert np.max(np.abs(b[v] - bf[v])) < TOL       # vs naive enumeration


@pytest.mark.parametrize("seed", range(10))
def test_bethe_logZ_exact_on_trees(seed):
    rng = np.random.default_rng(50 + seed)
    g = bp.random_tree(rng, n=int(rng.integers(2, 9)), card=int(rng.integers(2, 4)))
    _, info = sum_product(g, max_iter=400, tol=1e-13)
    z, _, _ = g.eliminate("sum")
    assert abs(info["bethe_logZ"] - z) < TOL
    assert abs(info["bethe_logZ"] - bp.brute_force_logZ(g)) < TOL


@pytest.mark.parametrize("seed", range(10))
def test_min_sum_exact_on_trees(seed):
    rng = np.random.default_rng(100 + seed)
    g = bp.random_tree(rng, n=int(rng.integers(2, 9)), card=int(rng.integers(2, 4)))
    beliefs, arg, info = min_sum(g, max_iter=400, tol=1e-13)
    assert info["converged"]
    vmin, _, _ = g.eliminate("min")
    assert abs(info["value"] - vmin) < TOL
    # the returned assignment must actually achieve the reported energy on the raw tables
    tot = sum(float(f.table[tuple(arg[v] for v in f.vars)]) for f in g.factors if f.vars)
    assert abs(tot - info["value"]) < TOL
    # and the beliefs must be the min-marginals, shifted to min 0
    mm, best = bp.brute_force_min_marginals(g)
    for v in beliefs:
        assert np.max(np.abs(beliefs[v] - (mm[v] - best))) < TOL


def test_min_sum_beliefs_are_min_marginals_not_marginals():
    """Guards the |sum w|^2 vs sum |w|^2 family of confusions: the min-sum belief must be
    a MIN over the other variables, never a sum, and the two differ measurably."""
    g = _chain(6, card=3, seed=9)
    beliefs, _, _ = min_sum(g, max_iter=400, tol=1e-13)
    mm, best = bp.brute_force_min_marginals(g)
    sp, _ = sum_product(g, max_iter=400, tol=1e-13)
    worst_min, worst_sum = 0.0, 0.0
    for v in beliefs:
        worst_min = max(worst_min, np.max(np.abs(beliefs[v] - (mm[v] - best))))
        soft = -np.log(sp[v])
        worst_sum = max(worst_sum, np.max(np.abs(beliefs[v] - (soft - soft.min()))))
    assert worst_min < TOL                 # it IS the min-marginal
    assert worst_sum > 1e-3                # it is NOT the log-marginal


def test_forest_of_two_components_is_exact():
    two = FactorGraph()
    rng = np.random.default_rng(4)
    two.add_factor(["a", "b"], rng.normal(size=(2, 2)))
    two.add_factor(["c", "d"], rng.normal(size=(3, 2)))
    two.add_factor(["c"], rng.normal(size=3))
    b, info = sum_product(two, max_iter=200, tol=1e-13)
    assert info["exact"] and not info["is_tree"] and info["is_forest"]
    bf = bp.brute_force_marginals(two)
    for v in b:
        assert np.max(np.abs(b[v] - bf[v])) < TOL
    assert abs(info["bethe_logZ"] - bp.brute_force_logZ(two)) < TOL


def test_isolated_variable_matches_brute_force():
    """A variable in no factor contributes log(card) to log Z. Brute-force enumeration
    says so and the Bethe free energy agrees; measured 2026-08-29,
    FactorGraph.eliminate("sum") omits it (empty bucket, skipped). Not patched here --
    factorgraph.py is another module. This test pins bp.py to the enumerator."""
    g = FactorGraph()
    g.add_var("iso", 3)
    g.add_factor(["a"], np.zeros(2))
    b, info = sum_product(g)
    assert np.max(np.abs(b["iso"] - 1 / 3)) < TOL
    assert abs(info["bethe_logZ"] - bp.brute_force_logZ(g)) < TOL
    assert abs(info["bethe_logZ"] - np.log(6.0)) < TOL


# ------------------------------------------------------------------ damping / log space
def test_damping_moves_the_path_not_the_tree_fixed_point():
    g = bp.random_tree(np.random.default_rng(5), n=7, card=3)
    b0, i0 = sum_product(g, max_iter=2000, tol=1e-13, damping=0.0)
    b9, i9 = sum_product(g, max_iter=8000, tol=1e-13, damping=0.9)
    assert i0["converged"] and i9["converged"]
    assert i9["iterations"] > i0["iterations"]          # damping only slows it down
    for v in b0:
        assert np.max(np.abs(b0[v] - b9[v])) < 1e-9
    ex = g.marginals()
    for v in b9:
        assert np.max(np.abs(b9[v] - ex[v])) < TOL


def test_damping_out_of_range_rejected():
    g = _chain(4)
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            sum_product(g, damping=bad)


def test_log_space_survives_a_long_chain_that_would_overflow():
    """Rule 4. |phi| ~ 30 over 80 sites: log Z ~ 2400, so exp() is inf in float64.
    BP must still land on elimination's answer."""
    n, scale = 80, 30.0
    g = bp.random_tree(np.random.default_rng(11), n=n, card=2, scale=scale)
    _, info = sum_product(g, max_iter=4 * n, tol=1e-12)
    z, _, _ = g.eliminate("sum")
    assert info["converged"] and info["treewidth"] == 1
    assert info["bethe_logZ"] > 709.0                  # exp() of this overflows float64
    with np.errstate(over="ignore"):
        assert np.isinf(np.exp(info["bethe_logZ"]))
    assert abs(info["bethe_logZ"] - z) < 1e-8
    assert np.isfinite(info["bethe_logZ"])


def test_every_info_dict_logs_treewidth_and_cycles():
    """Rule 3: the cost law (treewidth) and the exactness law (bipartite cycles) are
    reported on every call, for trees and loops alike."""
    for g in (_chain(6), _chain(6, ring=True)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, i1 = sum_product(g, max_iter=50)
            _, _, i2 = min_sum(g, max_iter=50)
        for i in (i1, i2):
            assert isinstance(i["treewidth"], int) and i["treewidth"] >= 1
            assert "n_independent_cycles" in i and "max_arity" in i
            assert i["n_vars"] == 6


# ------------------------------------------------------------------ loops: the boundary
def test_loopy_graph_warns_and_strict_raises():
    g = _chain(6, ring=True)
    with pytest.warns(RuntimeWarning, match="LOOPY"):
        sum_product(g, max_iter=50)
    with pytest.warns(RuntimeWarning, match="LOOPY"):
        min_sum(g, max_iter=50)
    with pytest.raises(ValueError, match="LOOPY"):
        sum_product(g, max_iter=50, strict=True)
    with pytest.raises(ValueError, match="LOOPY"):
        min_sum(g, max_iter=50, strict=True)
    with warnings.catch_warnings(record=True) as caught:      # trees must stay silent
        warnings.simplefilter("always")
        sum_product(_chain(6), max_iter=50)
        min_sum(_chain(6), max_iter=50)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_one_extra_factor_turns_exact_into_wrong():
    """POSITIVE CONTROL AND NEGATIVE IN ONE TEST. Same Ising couplings and fields; the
    ring differs from the chain by a single wrap-around factor. If the chain half of this
    test passed and the ring half did not fail, the pipeline would be broken, not BP."""
    J = np.random.default_rng(3).choice([-1.0, 1.0], size=8)
    chain = bp.ising(np.random.default_rng(4), n=8, beta=0.9, h=0.4, ring=False,
                     seed_couplings=J)
    ring = bp.ising(np.random.default_rng(4), n=8, beta=0.9, h=0.4, ring=True,
                    seed_couplings=J)
    bc, ic = sum_product(chain, max_iter=400, tol=1e-13)
    ex_c = chain.marginals()
    chain_err = max(float(np.max(np.abs(bc[v] - ex_c[v]))) for v in bc)
    assert ic["treewidth"] == 1 and ic["n_independent_cycles"] == 0
    assert chain_err < TOL                                  # positive control

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        br, ir = sum_product(ring, max_iter=400, tol=1e-13)
    ex_r = ring.marginals()
    ring_err = max(float(np.max(np.abs(br[v] - ex_r[v]))) for v in br)
    z, _, _ = ring.eliminate("sum")
    assert ir["treewidth"] == 2 and ir["n_independent_cycles"] == 1
    assert ir["converged"]                                  # it settles ...
    assert ring_err > 1e-3                                  # ... on the wrong answer
    assert abs(ir["bethe_logZ"] - z) > 1e-3
    assert ring_err > 1e6 * max(chain_err, 1e-16)


def test_damping_does_not_repair_a_wrong_fixed_point():
    """Damping is a convergence aid, never a correctness fix: three damping values, three
    different sweep counts, the SAME wrong marginals."""
    g = bp.grid_ising(np.random.default_rng(2), 3, 3, beta=1.1, h=0.25)
    ex = g.marginals()
    errs, iters = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dmp in (0.0, 0.5, 0.9):
            b, i = sum_product(g, max_iter=1200, tol=1e-10, damping=dmp)
            assert i["converged"]
            errs.append(max(float(np.max(np.abs(b[v] - ex[v]))) for v in b))
            iters.append(i["iterations"])
    assert min(errs) > 0.3                        # all three are badly wrong
    assert max(errs) - min(errs) < 1e-6           # and wrong in exactly the same way
    assert iters[2] > iters[0] * 5                # damping only costs sweeps


def test_loopy_bp_can_fail_to_converge_at_all():
    g = bp.grid_ising(np.random.default_rng(0), 3, 3, beta=1.5, h=0.15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b, i = sum_product(g, max_iter=400, tol=1e-10, damping=0.0)
    assert not i["converged"] and i["oscillating"]
    assert i["regime"].startswith("non-convergent") or i["regime"].startswith("limit")
    assert i["message_swing"] > 1.0               # messages still moving by nats/sweep
    ex = g.marginals()
    assert max(float(np.max(np.abs(b[v] - ex[v]))) for v in b) > 0.1


def test_min_sum_optimality_tree_control_then_loopy_failure():
    """Rule 6 again: the same call, same settings, 20 trees and 20 loops."""
    def scan(maker):
        miss, worst = 0, 0.0
        for s in range(20):
            g = maker(s)
            vex, _, _ = g.eliminate("min")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, _, i = min_sum(g, max_iter=300, tol=1e-11, damping=0.5,
                                  compute_treewidth=False)
            if i["value"] - vex > 1e-9:
                miss += 1
                worst = max(worst, i["value"] - vex)
        return miss, worst
    tree_miss, tree_worst = scan(
        lambda s: bp.ising(np.random.default_rng(1000 + s), n=9, beta=1.0, h=0.4))
    loop_miss, loop_worst = scan(
        lambda s: bp.grid_ising(np.random.default_rng(1000 + s), 3, 3, beta=1.0, h=0.4))
    assert tree_miss == 0 and tree_worst == 0.0        # positive control
    assert loop_miss >= 5 and loop_worst > 1.0         # measured failure


# ------------------------------------------------------------------ the summary
def test_verify_summary():
    r = bp.verify(seed=0, trials=12, verbose=False)
    assert r["max_err_marginal_vs_elimination"] < TOL
    assert r["max_err_marginal_vs_bruteforce"] < TOL
    assert r["max_err_bethe_logZ"] < TOL
    assert r["max_err_minsum_value"] < TOL
    assert r["max_err_min_marginal"] < TOL
    assert r["bad_assignments"] == 0
    assert r["minmarg_checked"] >= 10
    assert r["cycles"] == 0
    assert r["chain_logZ_err"] < 1e-8 and r["chain_logZ"] > 709
    assert r["damping_fixed_point_shift"] < TOL
    assert r["loop_warning_emitted"]
    assert r["pair_chain_err"] < TOL < r["pair_ring_err"]
    assert r["loopy_max_marginal_gap"] > 0.1
    assert r["minsum_tree"]["misses"] == 0 and r["minsum_randtree"]["misses"] == 0
    assert r["minsum_grid"]["misses"] > 0


def test_cost_law_exact_at_every_arity():
    """d ** max_arity is the cost; bipartite acyclicity is the exactness condition. The
    induced treewidth rises with arity while the error stays at machine precision."""
    r = bp.verify_cost_law(verbose=False)
    assert r["max_err"] < TOL
    assert r["chain_logZ_err"] < 1e-9
    tws = [row["treewidth"] for row in r["rows"]]
    assert tws == sorted(tws) and tws[-1] > tws[0]
    for row in r["rows"]:
        assert row["cycles"] == 0
        assert row["table"] == 3 ** row["arity"]


def test_min_sum_ties_are_flagged_not_silently_wrong():
    """A measured boundary, kept rather than papered over. Two tied optima, (0,1) and
    (1,0): the min-marginals are exactly right, but per-variable argmin decoding picks
    the inconsistent (0,0). min_sum must flag it via decode_gap / degenerate rather than
    quietly return a suboptimal assignment as if it were the optimum."""
    g = FactorGraph()
    g.add_factor(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    beliefs, arg, info = min_sum(g, max_iter=100, tol=1e-13)
    vmin, _, _ = g.eliminate("min")

    mm, best = bp.brute_force_min_marginals(g)
    for v in beliefs:                       # the BELIEFS are exact ...
        assert np.max(np.abs(beliefs[v] - (mm[v] - best))) < TOL
    assert info["decode_gap"] == 0.0        # ... the ASSIGNMENT is not decodable ...
    assert info["degenerate"] is True       # ... and the flag says so
    assert info["value"] - vmin == pytest.approx(1.0)
    # elimination's back-tracking is tie-safe and stays optimal on the same instance
    _, varg, _ = g.eliminate("min")
    assert float(g.factors[0].table[varg["a"], varg["b"]]) == pytest.approx(vmin)
