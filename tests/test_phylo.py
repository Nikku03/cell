"""Felsenstein pruning against brute-force enumeration over 4^(internal nodes).

Every check here has an independent reference: naive enumeration, an independent matrix
exponential, an analytic two-tip identity, normalisation over all data patterns, or REM's
own bucket elimination on the equivalent factor graph.
"""
import itertools
import math
import time

import numpy as np
import pytest

from rem import phylo
from rem.phylo import (JC69, K80, GTR, DNA, alignment_loglik, balanced_tree,
                       brute_force_loglik, caterpillar_tree, factorgraph_loglik,
                       felsenstein_loglik, felsenstein_site_logliks, fit_branch_scale,
                       gtr, gtr_Q, jukes_cantor, jukes_cantor_Q, kimura_2p, kimura_2p_Q,
                       node_labels, parse_newick, random_tree, simulate_alignment, taxa,
                       to_factorgraph, to_newick, tree_info, tip_vector, _expm_taylor)

TOL = 1e-10


def _models(rng):
    ex = list(rng.uniform(0.4, 2.5, size=6))
    p = rng.uniform(0.15, 0.35, size=4)
    p = p / p.sum()
    return [JC69, K80(2.7), K80(0.6), GTR(ex, p)]


# ------------------------------------------------------------------ substitution models
@pytest.mark.parametrize("t", [0.0, 0.001, 0.05, 0.3, 1.0, 3.0, 10.0])
def test_transition_matrices_match_independent_expm(t):
    """Closed forms vs scaling-and-squaring Taylor series -- a different algorithm."""
    rng = np.random.default_rng(int(t * 1000) + 1)
    ex = list(rng.uniform(0.3, 3.0, size=6))
    pi = rng.uniform(0.15, 0.35, size=4)
    pi = pi / pi.sum()
    Qg, pg = gtr_Q(ex, pi)
    for P, Q in ((jukes_cantor(t), jukes_cantor_Q()),
                 (kimura_2p(t, 2.3), kimura_2p_Q(2.3)),
                 (gtr(t, ex, pi), Qg)):
        assert np.max(np.abs(P - _expm_taylor(Q * t))) < 1e-11
        assert np.max(np.abs(P.sum(axis=1) - 1.0)) < 1e-12
        assert np.all(P >= -1e-15)


def test_transition_matrix_limits_and_stationarity():
    assert np.max(np.abs(jukes_cantor(0.0) - np.eye(4))) < 1e-15
    assert np.max(np.abs(kimura_2p(0.0, 3.0) - np.eye(4))) < 1e-15
    assert np.max(np.abs(jukes_cantor(500.0) - 0.25)) < 1e-12
    pi = np.array([0.2, 0.3, 0.35, 0.15])
    P = gtr(400.0, [1.0, 2.0, 0.7, 0.9, 2.4, 1.1], pi)
    assert np.max(np.abs(P - pi[None, :])) < 1e-9
    assert np.max(np.abs(pi @ gtr(0.7, [1.0, 2.0, 0.7, 0.9, 2.4, 1.1], pi) - pi)) < 1e-12


def test_k80_kappa_one_is_jukes_cantor():
    for t in (0.01, 0.4, 2.0):
        assert np.max(np.abs(kimura_2p(t, 1.0) - jukes_cantor(t))) < 1e-14


def test_chapman_kolmogorov():
    for s, t in ((0.1, 0.3), (0.7, 1.9)):
        assert np.max(np.abs(kimura_2p(s, 3.1) @ kimura_2p(t, 3.1)
                             - kimura_2p(s + t, 3.1))) < 1e-13
        pi = np.array([0.22, 0.28, 0.31, 0.19])
        ex = [1.0, 2.5, 0.8, 1.1, 2.9, 1.0]
        assert np.max(np.abs(gtr(s, ex, pi) @ gtr(t, ex, pi) - gtr(s + t, ex, pi))) < 1e-13


def test_normalised_rate_one_substitution_per_unit_time():
    """Branch lengths are expected substitutions per site: -sum_i pi_i Q_ii must be 1."""
    assert abs(-np.sum(0.25 * np.diag(jukes_cantor_Q())) - 1.0) < 1e-14
    for kappa in (0.5, 1.0, 4.0):
        assert abs(-np.sum(0.25 * np.diag(kimura_2p_Q(kappa))) - 1.0) < 1e-14
    Q, pi = gtr_Q([1.0, 2.0, 0.7, 0.9, 2.4, 1.1], [0.2, 0.3, 0.35, 0.15])
    assert abs(-np.sum(pi * np.diag(Q)) - 1.0) < 1e-14


# ---------------------------------------------------------- (a) pruning vs brute force
@pytest.mark.parametrize("seed", range(14))
def test_pruning_matches_brute_force_enumeration(seed):
    rng = np.random.default_rng(seed)
    n_tips = int(rng.integers(3, 8))              # <= 7 tips -> <= 6 internal -> 4^6
    tree = random_tree(n_tips, rng)
    names = taxa(tree)
    st = {nm: int(rng.integers(0, 4)) for nm in names}
    model = _models(rng)[int(rng.integers(0, 4))]
    info = tree_info(tree)
    assert info["n_internal"] == n_tips - 1 <= 6
    got = felsenstein_loglik(tree, st, model)
    ref = brute_force_loglik(tree, st, model)
    assert abs(got - ref) < TOL, f"{got} vs {ref}"


def test_brute_force_is_not_the_same_code_path():
    """The reference must be able to disagree. Perturb one transition matrix entry and
    check both routines move -- if the brute force silently tracked the pruning code it
    could not be used as ground truth."""
    rng = np.random.default_rng(5)
    tree = random_tree(4, rng)
    st = {nm: int(rng.integers(0, 4)) for nm in taxa(tree)}
    good = jukes_cantor(0.3)
    bad = good.copy()
    bad[0, 1] += 0.05
    bad[0, 0] -= 0.05
    model_bad = lambda t: bad                      # noqa: E731  constant, wrong matrix
    a = felsenstein_loglik(tree, st, model_bad)
    b = brute_force_loglik(tree, st, model_bad)
    assert abs(a - b) < TOL                        # still agree on a WRONG model
    c = felsenstein_loglik(tree, st, lambda t: good)
    assert abs(a - c) > 1e-6                       # and the wrong model gives a different answer


def test_multifurcation_matches_brute_force():
    tree = (("a", 0.11), ("b", 0.22), ("c", 0.33), ("d", 0.05))     # star, one internal
    st = {"a": 0, "b": 1, "c": 2, "d": 3}
    for model in (JC69, K80(3.0)):
        assert abs(felsenstein_loglik(tree, st, model)
                   - brute_force_loglik(tree, st, model)) < TOL
    assert tree_info(tree)["n_internal"] == 1


def test_ambiguity_and_missing_data_match_brute_force():
    rng = np.random.default_rng(11)
    tree = random_tree(5, rng)
    names = taxa(tree)
    st = {nm: "ACGT"[int(rng.integers(0, 4))] for nm in names}
    st[names[0]] = "N"
    st[names[1]] = "R"
    assert abs(felsenstein_loglik(tree, st, K80(2.0))
               - brute_force_loglik(tree, st, K80(2.0))) < TOL


def test_N_equals_sum_over_resolved_states():
    rng = np.random.default_rng(12)
    tree = random_tree(5, rng)
    names = taxa(tree)
    st = {nm: int(rng.integers(0, 4)) for nm in names}
    parts = []
    for k in range(4):
        s2 = dict(st, **{names[2]: k})
        parts.append(felsenstein_loglik(tree, s2, JC69))
    s2 = dict(st, **{names[2]: "N"})
    tot = math.log(sum(math.exp(p) for p in parts))
    assert abs(felsenstein_loglik(tree, s2, JC69) - tot) < TOL
    s2 = dict(st, **{names[2]: "R"})                     # R = A or G  -> states 0 and 2
    tot = math.log(math.exp(parts[0]) + math.exp(parts[2]))
    assert abs(felsenstein_loglik(tree, s2, JC69) - tot) < TOL


# ------------------------------------------------ (b) the same computation as a REM graph
@pytest.mark.parametrize("seed", range(10))
def test_factorgraph_elimination_agrees(seed):
    rng = np.random.default_rng(300 + seed)
    tree = random_tree(int(rng.integers(3, 9)), rng)
    st = {nm: int(rng.integers(0, 4)) for nm in taxa(tree)}
    model = _models(rng)[int(rng.integers(0, 4))]
    direct = felsenstein_loglik(tree, st, model)
    fg, info = factorgraph_loglik(tree, st, model)
    assert abs(direct - fg) < 1e-10
    assert info["treewidth"] == 1
    assert info["largest_table"] == 16          # d^(treewidth+1) = 4^2, the governing law


def test_factorgraph_treewidth_is_one_for_any_tree():
    rng = np.random.default_rng(4)
    for n in (4, 9, 17, 33):
        assert tree_info(random_tree(n, rng))["treewidth"] == 1
    for n in (4, 16, 64):
        assert tree_info(balanced_tree(n))["treewidth"] == 1
    assert tree_info(caterpillar_tree(40))["treewidth"] == 1


def test_factorgraph_variables_are_one_per_node():
    rng = np.random.default_rng(6)
    tree = random_tree(6, rng)
    st = {nm: int(rng.integers(0, 4)) for nm in taxa(tree)}
    g = to_factorgraph(tree, st, JC69)
    info = tree_info(tree)
    assert len(g.variables) == info["n_nodes"] == 11
    assert set(g.variables) == set(node_labels(tree))
    # one edge factor + one root unary + one unary per tip
    assert len(g.factors) == info["n_edges"] + 1 + info["n_tips"]


# --------------------------------------------------------------- structural identities
def test_likelihood_normalises_over_all_data_patterns():
    """Sum over all 4^tips observable patterns must be exactly 1. Needs no reference
    implementation at all -- it is a property the true likelihood cannot fail."""
    rng = np.random.default_rng(21)
    for n_tips in (3, 4, 5):
        tree = random_tree(n_tips, rng)
        names = taxa(tree)
        model = _models(rng)[int(rng.integers(0, 4))]
        tot = sum(math.exp(felsenstein_loglik(tree, dict(zip(names, pat)), model))
                  for pat in itertools.product(range(4), repeat=n_tips))
        assert abs(tot - 1.0) < 1e-12


def test_two_tip_analytic_closed_form():
    """A cherry collapses by reversibility to pi_a P(t1+t2)[a,b]: pure algebra."""
    rng = np.random.default_rng(31)
    for _ in range(20):
        t1, t2 = float(rng.uniform(0.01, 2.0)), float(rng.uniform(0.01, 2.0))
        a, b = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        kappa = float(rng.uniform(0.4, 6.0))
        got = felsenstein_loglik((("x", t1), ("y", t2)), {"x": a, "y": b}, K80(kappa))
        ref = math.log(0.25 * kimura_2p(t1 + t2, kappa)[a, b])
        assert abs(got - ref) < 1e-13


def test_root_position_does_not_matter_for_reversible_models():
    rng = np.random.default_rng(41)
    for _ in range(6):
        t1, t2, t3, t4, L = [float(rng.uniform(0.02, 0.9)) for _ in range(5)]
        u = (("a", t1), ("b", t2))
        v = (("c", t3), ("d", t4))
        st = {k: int(rng.integers(0, 4)) for k in "abcd"}
        model = _models(rng)[int(rng.integers(0, 4))]
        vals = [felsenstein_loglik(((u, f * L), (v, (1 - f) * L)), st, model)
                for f in (0.0, 0.3, 0.5, 1.0)]
        vals.append(felsenstein_loglik((("a", t1), ("b", t2), (v, L)), st, model))
        vals.append(felsenstein_loglik((("c", t3), ("d", t4), (u, L)), st, model))
        assert max(vals) - min(vals) < 1e-12


def test_zero_length_branch_is_an_identity():
    rng = np.random.default_rng(51)
    st = {"a": 1, "b": 3, "c": 0}
    inner = (("a", 0.2), ("b", 0.4))
    with_zero = ((inner, 0.0), ("c", 0.3))
    collapsed = (("a", 0.2), ("b", 0.4), ("c", 0.3))
    assert abs(felsenstein_loglik(with_zero, st, JC69)
               - felsenstein_loglik(collapsed, st, JC69)) < 1e-13


def test_infinite_branch_saturates_to_independence():
    """A very long branch decouples the tip: its likelihood factorises into pi_state."""
    st = {"a": 1, "b": 2}
    got = felsenstein_loglik((("a", 0.05), ("b", 60.0)), st, JC69)
    # branch b is saturated: P(60)[r, b] = 1/4 for every r, so the sum over the root
    # state r factorises into (1/4) * sum_r pi_r P(0.05)[r, a] = 1/4 * 1/4.
    ref = math.log(0.25 * sum(0.25 * jukes_cantor(0.05)[r, st["a"]] for r in range(4)))
    assert abs(got - ref) < 1e-9


# --------------------------------------------------------------------- alignments / API
def test_alignment_loglik_is_the_sum_of_site_logliks():
    rng = np.random.default_rng(61)
    tree = random_tree(6, rng)
    aln = simulate_alignment(tree, JC69, 40, rng)
    per_site = felsenstein_site_logliks(tree, aln, JC69)
    assert per_site.shape == (40,)
    assert abs(alignment_loglik(tree, aln, JC69) - float(per_site.sum())) < 1e-10
    for k in (0, 7, 39):
        single = {nm: aln[nm][k] for nm in aln}
        assert abs(felsenstein_loglik(tree, single, JC69) - per_site[k]) < 1e-12
        assert abs(brute_force_loglik(tree, single, JC69) - per_site[k]) < 1e-10


def test_branch_length_override_by_label_and_by_vector():
    tree = parse_newick("((a:0.1,b:0.2):0.3,c:0.4);")
    labels = node_labels(tree)
    base = felsenstein_loglik(tree, {"a": 0, "b": 1, "c": 2}, JC69)
    same = felsenstein_loglik(tree, {"a": 0, "b": 1, "c": 2}, JC69,
                              branch_lengths={"a": 0.1})
    assert abs(base - same) < 1e-14
    changed = felsenstein_loglik(tree, {"a": 0, "b": 1, "c": 2}, JC69,
                                 branch_lengths={"a": 0.9})
    assert abs(base - changed) > 1e-6
    equiv = parse_newick("((a:0.9,b:0.2):0.3,c:0.4);")
    assert abs(changed - felsenstein_loglik(equiv, {"a": 0, "b": 1, "c": 2}, JC69)) < 1e-14
    flat = phylo._flatten(tree)
    vec = [0.9 if labels[i] == "a" else float(flat.brlen[i])
           for i in range(1, len(labels))]
    assert abs(felsenstein_loglik(tree, {"a": 0, "b": 1, "c": 2}, JC69,
                                  branch_lengths=vec) - changed) < 1e-14


def test_scale_multiplies_every_branch():
    tree = parse_newick("((a:0.1,b:0.2):0.3,c:0.4);")
    scaled = parse_newick("((a:0.2,b:0.4):0.6,c:0.8);")
    st = {"a": 0, "b": 1, "c": 2}
    assert abs(felsenstein_loglik(tree, st, JC69, scale=2.0)
               - felsenstein_loglik(scaled, st, JC69)) < 1e-13


def test_newick_parse_and_roundtrip():
    tree = parse_newick("((a:0.1,b:0.2)AB:0.3,c:0.4);")
    assert tree == (((("a", 0.1), ("b", 0.2)), 0.3), ("c", 0.4))
    assert to_newick(tree) == "((a:0.1,b:0.2):0.3,c:0.4);"
    assert parse_newick(to_newick(tree)) == tree
    assert taxa(tree) == ["a", "b", "c"]
    assert node_labels(tree) == ["n0", "n1", "a", "b", "c"]
    # whitespace, quoted names, multifurcation
    t2 = parse_newick("( 'Homo sapiens':0.1 , mouse:0.2 , 'x,y':0.3 ) ;")
    assert taxa(t2) == ["Homo sapiens", "mouse", "x,y"]
    with pytest.raises(ValueError):
        parse_newick("((a:0.1,b:0.2;")


def test_duplicate_taxa_and_negative_lengths_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        felsenstein_loglik((("a", 0.1), ("a", 0.2)), {"a": 0}, JC69)
    with pytest.raises(ValueError):
        felsenstein_loglik((("a", -0.1), ("b", 0.2)), {"a": 0, "b": 1}, JC69)
    with pytest.raises(KeyError):
        felsenstein_loglik((("a", 0.1), ("b", 0.2)), {"a": 0}, JC69)


def test_internal_label_collision_is_avoided():
    tree = ((("n0", 0.1), ("n1", 0.2)), 0.3), ("n2", 0.4)
    labels = node_labels(tree)
    assert len(set(labels)) == len(labels)
    st = {"n0": 0, "n1": 1, "n2": 2}
    assert abs(felsenstein_loglik(tree, st, JC69)
               - brute_force_loglik(tree, st, JC69)) < TOL
    fg, info = factorgraph_loglik(tree, st, JC69)
    assert abs(fg - felsenstein_loglik(tree, st, JC69)) < 1e-10


def test_tip_vector_codes():
    assert np.allclose(tip_vector("A"), [1, 0, 0, 0])
    assert np.allclose(tip_vector("R"), [1, 0, 1, 0])
    assert np.allclose(tip_vector("N"), [1, 1, 1, 1])
    assert np.allclose(tip_vector("-"), [1, 1, 1, 1])
    assert np.allclose(tip_vector(2), [0, 0, 1, 0])
    with pytest.raises(ValueError):
        tip_vector("Z")


# ------------------------------------------------------------------- (c) linear scaling
def test_linear_scaling_in_number_of_tips():
    """Measured: doubling the tips must roughly double the time, while the summed-over
    search space squares. Loose bounds so the test is not flaky on a noisy box."""
    rng = np.random.default_rng(77)
    ns, ts = [], []
    for n_tips in (128, 256, 512, 1024):
        tree = balanced_tree(n_tips, rng=rng)
        st = {nm: int(rng.integers(0, 4)) for nm in taxa(tree)}
        felsenstein_loglik(tree, st, JC69)                 # warm the caches
        reps = max(1, 4000 // n_tips)
        t0 = time.perf_counter()
        for _ in range(reps):
            felsenstein_loglik(tree, st, JC69)
        ns.append(n_tips)
        ts.append((time.perf_counter() - t0) / reps)
    slope = float(np.polyfit(np.log(ns), np.log(ts), 1)[0])
    assert 0.7 < slope < 1.4, f"time scaled as n^{slope:.2f}, not linear"
    assert tree_info(balanced_tree(1024))["n_internal"] == 1023   # 4^1023 assignments


def test_big_tree_agrees_with_factorgraph_elimination():
    rng = np.random.default_rng(88)
    tree = balanced_tree(128, rng=rng)
    st = {nm: int(rng.integers(0, 4)) for nm in taxa(tree)}
    direct = felsenstein_loglik(tree, st, K80(2.0))
    fg, info = factorgraph_loglik(tree, st, K80(2.0))
    assert abs(direct - fg) < 1e-9
    assert info["treewidth"] == 1 and info["largest_table"] == 16


def test_log_space_survives_where_linear_space_underflows():
    """Rule 4, measured. At 512 tips the textbook linear-space pruning returns exactly
    0.0; the log-space version returns a finite log likelihood."""
    rng = np.random.default_rng(99)
    tree = balanced_tree(512, rng=rng)
    st = {nm: int(rng.integers(0, 4)) for nm in taxa(tree)}
    ll = felsenstein_loglik(tree, st, JC69)
    assert np.isfinite(ll) and ll < -700
    assert phylo._prune_linear_naive(tree, st, JC69) == 0.0


def test_deep_caterpillar_does_not_hit_the_recursion_limit():
    tree = caterpillar_tree(3000, 0.05)
    info = tree_info(tree)
    assert info["max_depth"] >= 2999 and info["treewidth"] == 1
    st = {nm: 0 for nm in taxa(tree)}
    assert np.isfinite(felsenstein_loglik(tree, st, JC69))


# ------------------------------------------------------------------- positive control
def test_recovers_planted_branch_scale_with_negative_control():
    r = phylo.verify_recovers_planted_scale(seed=3, n_tips=12, n_sites=600,
                                            true_scale=1.0, verbose=False)
    assert r["rel_error"] < 0.20, r
    assert r["loglik_at_fit"] >= r["loglik_at_truth"] - 1e-9
    # NEGATIVE control: no phylogenetic signal -> the fit runs to the bracket top
    assert r["fitted_scale_on_noise"] > 0.9 * r["noise_bracket_top"], r


@pytest.mark.parametrize("true_scale", [0.3, 1.0, 2.5])
def test_planted_scale_recovery_over_a_range(true_scale):
    r = phylo.verify_recovers_planted_scale(seed=17, n_tips=10, n_sites=800,
                                            true_scale=true_scale, verbose=False)
    assert r["rel_error"] < 0.25, r


def test_simulated_data_prefers_the_true_tree_over_a_shuffled_one():
    """A second positive control on topology, not just branch lengths."""
    rng = np.random.default_rng(123)
    tree = random_tree(8, rng)
    aln = simulate_alignment(tree, JC69, 1500, rng)
    true_ll = alignment_loglik(tree, aln, JC69)
    beaten = 0
    for _ in range(6):
        other = random_tree(8, rng, names=list(rng.permutation(taxa(tree))))
        if alignment_loglik(other, aln, JC69) < true_ll:
            beaten += 1
    assert beaten == 6, "the generating tree lost to a random topology"


# --------------------------------------------------------------------------- verify()
def test_module_verify_is_exact():
    r = phylo.verify(seed=0, verbose=False)
    assert r["max_err_vs_bruteforce"] < TOL
    assert r["max_err_factorgraph"] < TOL
    assert r["max_err_expm"] < 1e-11
    assert r["max_err_normalisation"] < 1e-11
    assert r["max_err_two_tip_analytic"] < TOL
    assert r["max_err_rerooting"] < TOL
    assert r["max_err_ambiguity"] < TOL
    assert r["treewidths"] == [1]
    assert r["largest_tables"] == [16]
    assert 0.7 < r["scaling_exponent"] < 1.4
    assert r["linear_space_underflows_at_tips"] is not None


# ------------------------------- a THIRD reference, written out by hand in this file ---
def test_hand_written_reference_three_and_four_tips():
    """Neither the pruning nor the module's brute force is trusted here: the sums are
    written out explicitly, so if BOTH module routines shared a bug this test would
    still catch it."""
    t1, t2, t3, t4, L = 0.13, 0.29, 0.07, 0.41, 0.22
    pi = np.array([0.22, 0.28, 0.31, 0.19])
    ex = [0.7, 2.1, 0.9, 1.3, 2.6, 1.0]
    model = GTR(ex, pi)
    P = {t: gtr(t, ex, pi) for t in (t1, t2, t3, t4, L)}

    # three tips on one internal node: sum over the single root state r
    st3 = {"a": 0, "b": 2, "c": 3}
    hand3 = 0.0
    for r in range(4):
        hand3 += pi[r] * P[t1][r, 0] * P[t2][r, 2] * P[t3][r, 3]
    tree3 = (("a", t1), ("b", t2), ("c", t3))
    assert abs(felsenstein_loglik(tree3, st3, model) - math.log(hand3)) < 1e-13
    assert abs(brute_force_loglik(tree3, st3, model) - math.log(hand3)) < 1e-13

    # four tips, two internal nodes: an explicit double loop over (root, inner)
    st4 = {"a": 1, "b": 2, "c": 0, "d": 3}
    hand4 = 0.0
    for r in range(4):                      # root state
        for u in range(4):                  # inner node state
            hand4 += (pi[r] * P[t1][r, 1] * P[t2][r, 2]
                      * P[L][r, u] * P[t3][u, 0] * P[t4][u, 3])
    tree4 = (("a", t1), ("b", t2), (((("c", t3), ("d", t4))), L))
    assert abs(felsenstein_loglik(tree4, st4, model) - math.log(hand4)) < 1e-13
    assert abs(brute_force_loglik(tree4, st4, model) - math.log(hand4)) < 1e-13
    fg, info = factorgraph_loglik(tree4, st4, model)
    assert abs(fg - math.log(hand4)) < 1e-11 and info["treewidth"] == 1
