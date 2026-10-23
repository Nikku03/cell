"""Clustered topology: exactness vs two independent references, and the flat-treewidth
headline together with the contrast that makes it falsifiable.

    cost = d ** treewidth

Nothing here asserts that the treewidth is flat; every test MEASURES it.
"""
import itertools
import time

import numpy as np
import pytest

from rem.clusters import (build_clustered_graph, brute_force, chain_reference,
                          interface_scan, parse_name, plant_ground_state,
                          scaling_table, solve, var_name, verify)

TOL = 1e-9


# --------------------------------------------------------------------- construction
def test_variable_count_and_cardinality():
    g = build_clustered_graph(5, 4, 3, 2, seed=0)
    assert len(g.cards) == 20
    assert set(g.cards.values()) == {3}
    assert set(g.cards) == {var_name(k, i) for k in range(5) for i in range(4)}


def test_each_cluster_is_an_internal_clique():
    K, m = 4, 5
    g = build_clustered_graph(K, m, 2, 1, seed=1)
    adj = g.adjacency()
    for k in range(K):
        for i, j in itertools.combinations(range(m), 2):
            assert var_name(k, j) in adj[var_name(k, i)], f"cluster {k} not a clique"


def test_chain_interface_is_narrow_and_only_between_neighbours():
    K, m, s = 5, 6, 2
    g = build_clustered_graph(K, m, 2, s, seed=2, interface_coupling="matching")
    adj = g.adjacency()
    cross = set()
    for v, nbrs in adj.items():
        kv, _ = parse_name(v)
        for u in nbrs:
            ku, _ = parse_name(u)
            if ku != kv:
                cross.add(tuple(sorted((v, u))))
    # only consecutive clusters touch
    for a, b in cross:
        assert abs(parse_name(a)[0] - parse_name(b)[0]) == 1
    # exactly s edges cross each of the K-1 cuts
    assert len(cross) == s * (K - 1)


def test_complete_interface_has_s_squared_edges():
    K, s = 4, 3
    g = build_clustered_graph(K, 5, 2, s, seed=3, interface_coupling="complete")
    cross = sum(1 for f in g.factors
                if len(f.vars) == 2 and parse_name(f.vars[0])[0] != parse_name(f.vars[1])[0])
    assert cross == s * s * (K - 1)


def test_all_to_all_ports_form_a_clique():
    K = 6
    g = build_clustered_graph(K, 4, 2, 1, seed=4, topology="all-to-all")
    adj = g.adjacency()
    ports = [var_name(k, 0) for k in range(K)]
    for a, b in itertools.combinations(ports, 2):
        assert b in adj[a]


@pytest.mark.parametrize("kw", [dict(n_clusters=0), dict(cluster_size=0), dict(d=1),
                                dict(interface_size=0), dict(interface_size=99),
                                dict(topology="star"), dict(interface_coupling="mesh")])
def test_bad_arguments_raise(kw):
    args = dict(n_clusters=3, cluster_size=4, d=2, interface_size=2, seed=0)
    args.update(kw)
    topo = args.pop("topology", "chain")
    coup = args.pop("interface_coupling", "matching")
    with pytest.raises(ValueError):
        build_clustered_graph(**args, topology=topo, interface_coupling=coup)


# ---------------------------------------- (a) exactness vs INDEPENDENT brute force
SMALL = [(2, 3, 2, 1, "chain", "matching"),
         (3, 3, 2, 1, "chain", "matching"),
         (3, 3, 2, 2, "chain", "complete"),
         (2, 4, 2, 2, "chain", "matching"),
         (4, 2, 3, 1, "chain", "matching"),
         (4, 3, 2, 1, "all-to-all", "matching"),
         (5, 2, 2, 1, "all-to-all", "matching")]


@pytest.mark.parametrize("K,m,d,s,topo,coup", SMALL)
def test_min_matches_pure_python_brute_force(K, m, d, s, topo, coup):
    g = build_clustered_graph(K, m, d, s, seed=5, topology=topo,
                              interface_coupling=coup)
    r = solve(g, "min")
    ref, _ = brute_force(g, "min")
    assert abs(r["value"] - ref) < TOL
    # the reported assignment must actually achieve the reported energy
    tot = sum(float(f.table[tuple(r["assignment"][v] for v in f.vars)]) for f in g.factors)
    assert abs(tot - r["value"]) < TOL


@pytest.mark.parametrize("K,m,d,s,topo,coup", SMALL)
def test_logZ_matches_pure_python_brute_force(K, m, d, s, topo, coup):
    g = build_clustered_graph(K, m, d, s, seed=6, topology=topo,
                              interface_coupling=coup)
    r = solve(g, "sum")
    ref, _ = brute_force(g, "sum")
    assert abs(r["value"] - ref) < TOL


def test_brute_force_tracks_perturbations_independently():
    """Guard against the circular-check failure mode. brute_force must respond to changes
    in the model on its own terms: a uniform shift of one factor moves the optimum by
    exactly that shift, and driving one configuration far below the rest moves the argmin
    to it. Elimination must agree at every step."""
    g = build_clustered_graph(3, 3, 2, 1, seed=7)
    base_bf, _ = brute_force(g, "min")
    base_el = solve(g, "min")["value"]
    assert abs(base_bf - base_el) < TOL

    g.factors[0].table += 3.5
    assert abs(brute_force(g, "min")[0] - (base_bf + 3.5)) < TOL
    assert abs(solve(g, "min")["value"] - (base_el + 3.5)) < TOL

    target = {v: g.cards[v] - 1 for v in g.cards}
    for f in g.factors:
        f.table[tuple(target[v] for v in f.vars)] -= 50.0
    bf, arg = brute_force(g, "min")
    r = solve(g, "min")
    assert arg == target, "brute force did not find the planted configuration"
    assert r["assignment"] == target
    assert abs(bf - r["value"]) < TOL


# ------------------------------- (a2) exactness vs the CLUSTER TRANSFER-MATRIX DP
BIG = [(4, 4, 2, 1, "matching"), (8, 4, 2, 2, "matching"), (16, 5, 2, 2, "matching"),
       (32, 5, 2, 2, "matching"), (8, 4, 3, 1, "matching"), (12, 5, 2, 3, "complete")]


@pytest.mark.parametrize("K,m,d,s,coup", BIG)
def test_matches_transfer_matrix_reference(K, m, d, s, coup):
    """A second, genuinely different algorithm: collapse each cluster into one
    super-variable with d^m states and sweep the chain. No elimination order exists in it.
    This reaches instances far past brute force -- 32 clusters of 5 binaries is 2^160."""
    g = build_clustered_graph(K, m, d, s, seed=8, topology="chain",
                              interface_coupling=coup)
    assert abs(solve(g, "min")["value"] - chain_reference(g, "min")) < 1e-8
    assert abs(solve(g, "sum")["value"] - chain_reference(g, "sum")) < 1e-8


def test_transfer_matrix_reference_refuses_non_chain():
    g = build_clustered_graph(4, 3, 2, 1, seed=9, topology="all-to-all")
    with pytest.raises(ValueError):
        chain_reference(g, "min")


def test_log_space_where_probability_space_overflows():
    """1280 variables: logZ ~ 1.9e3, so exp(logZ) is inf in float64. The value must still
    be right, which is what LOG SPACE buys."""
    g = build_clustered_graph(256, 5, 2, 2, seed=0)
    r = solve(g, "sum")
    assert r["value"] > 709.78, "instance too small to exercise the overflow path"
    with np.errstate(over="ignore"):
        assert np.isinf(np.exp(r["value"]))
    assert abs(r["value"] - chain_reference(g, "sum")) < 1e-8


def test_value_is_independent_of_elimination_order():
    g = build_clustered_graph(4, 3, 2, 1, seed=10)
    ref = solve(g, "min")["value"]
    rng = np.random.default_rng(0)
    for _ in range(5):
        order = list(rng.permutation(list(g.cards)))
        assert abs(solve(g, "min", order=order)["value"] - ref) < TOL


# ------------------------------------------------------ (b) THE HEADLINE: flat cost
def test_treewidth_is_flat_as_n_clusters_grows():
    rows = scaling_table((2, 4, 8, 16, 32), cluster_size=5, d=3, interface_size=2,
                         seed=0, repeats=1, verbose=False)
    tws = [r["treewidth"] for r in rows]
    tables = [r["largest_table"] for r in rows]
    assert len(set(tws)) == 1, f"treewidth was not flat: {tws}"
    assert len(set(tables)) == 1, f"largest table was not flat: {tables}"
    assert rows[-1]["n_vars"] == 16 * rows[0]["n_vars"]      # 16x the system
    assert tables[0] == 3 ** (tws[0] + 1)                    # d^(tw+1), the governing law


@pytest.mark.parametrize("m", [3, 4, 5, 6])
def test_matching_interface_attains_the_clique_lower_bound(m):
    """Each cluster is a clique on m vertices, so treewidth >= m-1 for EVERY ordering --
    a lower bound that holds independently of any heuristic. With a matching interface of
    width s < m the greedy ordering ATTAINS m-1, so the widths reported everywhere above
    are the EXACT treewidth, not merely a heuristic upper bound."""
    for s in range(1, m):
        for K in (2, 4, 8, 16):
            g = build_clustered_graph(K, m, 2, s, seed=0, interface_coupling="matching")
            assert g.treewidth() == m - 1, f"m={m} s={s} K={K}"


@pytest.mark.parametrize("m,d,s", [(4, 2, 1), (5, 2, 2), (4, 3, 1), (6, 2, 3)])
def test_treewidth_flat_for_several_cluster_shapes(m, d, s):
    rows = scaling_table((2, 4, 8, 16), cluster_size=m, d=d, interface_size=s,
                         seed=1, repeats=1, verbose=False)
    tws = [r["treewidth"] for r in rows]
    assert len(set(tws[1:])) == 1, f"m={m} d={d} s={s}: treewidth {tws}"


def test_time_grows_linearly_not_exponentially():
    """Best-of-3 wall time. 16x the clusters must cost O(16x), not 3^(30) x."""
    rows = scaling_table((8, 16, 32, 64, 128), cluster_size=5, d=3, interface_size=2,
                         seed=0, repeats=3, verbose=False)
    spc = [r["sec_per_cluster"] for r in rows]
    assert max(spc) / min(spc) < 3.0, f"sec/cluster drifted: {spc}"
    x = np.array([r["n_clusters"] for r in rows], float)
    y = np.array([r["seconds"] for r in rows], float)
    A = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    assert r2 > 0.98, f"time vs n_clusters is not linear, R^2 = {r2}"


# ---------------------------------------- (c) THE CONTRAST: structure, not size
def test_all_to_all_treewidth_grows_with_n_clusters():
    """Same cluster internals, same d, same n_vars per cluster -- only the wiring changed.
    If this did NOT grow, the flat result above would be unfalsifiable."""
    rows = scaling_table((2, 4, 8, 12, 16), cluster_size=5, d=3, interface_size=1,
                         seed=0, topology="all-to-all", max_table=5e6, repeats=1,
                         verbose=False)
    tws = [r["treewidth"] for r in rows]
    assert tws == sorted(tws)
    assert tws[-1] > tws[0] + 5, f"contrast did not explode: {tws}"
    assert any(r["wall"] for r in rows), "no instance hit the treewidth wall"
    # the port set is a K-clique, so treewidth >= K-1
    for r in rows:
        assert r["treewidth"] >= r["n_clusters"] - 1


def test_all_to_all_and_chain_agree_where_both_are_cheap():
    """Positive control for the contrast harness itself: at K=4 the two topologies have
    the same treewidth, and both are exact against pure-python brute force."""
    for topo in ("chain", "all-to-all"):
        g = build_clustered_graph(4, 3, 2, 1, seed=12, topology=topo)
        r = solve(g, "min")
        ref, _ = brute_force(g, "min")
        assert abs(r["value"] - ref) < TOL


def test_widening_the_interface_grows_treewidth_at_constant_size():
    rows = interface_scan((1, 2, 3, 4, 5, 6), n_clusters=6, cluster_size=6, d=2,
                          seed=0, interface_coupling="complete", verbose=False)
    assert len({r["n_vars"] for r in rows}) == 1          # size is held constant
    tws = [r["treewidth"] for r in rows]
    assert tws == sorted(tws)
    assert tws[-1] > tws[0], f"treewidth did not grow with interface width: {tws}"


def test_treewidth_wall_raises_rather_than_hangs():
    g = build_clustered_graph(20, 5, 3, 1, seed=0, topology="all-to-all")
    assert g.treewidth() >= 19
    with pytest.raises(MemoryError, match="treewidth wall"):
        solve(g, "min", max_table=1e6)


# -------------------------------------------------- (d) POSITIVE CONTROL: planted
@pytest.mark.parametrize("K,m,d,s,topo", [(3, 3, 2, 1, "chain"), (8, 4, 3, 2, "chain"),
                                          (32, 5, 3, 2, "chain"), (16, 4, 2, 2, "chain"),
                                          (8, 4, 3, 1, "all-to-all")])
def test_planted_ground_state_is_recovered_exactly(K, m, d, s, topo):
    g = build_clustered_graph(K, m, d, s, seed=13, topology=topo)
    truth = plant_ground_state(g, seed=14)
    r = solve(g, "min")
    assert r["assignment"] == truth
    planted_E = sum(float(f.table[tuple(truth[v] for v in f.vars)]) for f in g.factors)
    assert abs(r["value"] - planted_E) < 1e-8


def test_planted_recovery_survives_many_seeds():
    ok = 0
    for sd in range(10):
        g = build_clustered_graph(6, 4, 3, 2, sd)
        truth = plant_ground_state(g, seed=100 + sd)
        ok += solve(g, "min")["assignment"] == truth
    assert ok == 10, f"planted signal recovered in only {ok}/10 seeds"


# --------------------------------------------------------------------------- verify
def test_verify_summary():
    r = verify(verbose=False)
    assert r["max_err_min_vs_bruteforce"] < 1e-10
    assert r["max_err_logZ_vs_bruteforce"] < 1e-10
    assert r["max_err_argmin_consistency"] < 1e-10
    assert r["max_err_min_vs_transfer_matrix"] < 1e-8
    assert r["max_err_logZ_vs_transfer_matrix"] < 1e-8
    assert r["headline_flat"] is True
    assert r["headline_treewidths"] == [4] * 7
    assert r["headline_fit_r2"] > 0.98
    assert r["contrast_grows"] is True
    assert r["contrast_hits_wall"] is True
    assert r["interface_scan_grows"] is True
    assert r["logspace_overflows_in_prob_space"] is True
    assert r["logspace_err"] < 1e-8
    assert r["positive_control_all_recovered"] is True
    assert r["positive_control_max_gap"] < 1e-8
