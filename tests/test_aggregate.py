"""Exact aggregate / rare-event distributions against independent ground truth.

Three separate ground truths are used, none of which shares a code path with the
convolution being tested:
  * brute-force enumeration of every joint outcome (pure Python loops),
  * numpy's own np.convolve,
  * exact bigint rational arithmetic on the integer numerators (no floats at all),
and Monte Carlo as the positive control that all of them mean the same thing.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from rem import aggregate as tr
from rem.aggregate import (Aggregate, IntUnit, Unit, aggregate_distribution, as_unit,
                          brute_force_aggregate, brute_force_tail, convolve_direct,
                          convolve_fft, convolve_log, demo_portfolio,
                          exact_integer_aggregate, exact_rational_tail,
                          expected_shortfall, log_tail_curve, log_tail_probability,
                          mixture_aggregate, monte_carlo_tail, tail_probability,
                          underflow_report, value_at_risk)

TOL = 1e-13


def _units(rng, n, allow_holes=True):
    us = []
    for _ in range(n):
        m = int(rng.integers(2, 5))
        p = rng.random(m) ** 2 + 1e-3
        if allow_holes and m >= 3 and rng.random() < 0.4:
            p[int(rng.integers(1, m - 1))] = 0.0
        us.append(Unit(int(rng.integers(-2, 3)), p / p.sum()))
    return us


# ----------------------------------------------------------------------------------
# (1) the three convolutions against numpy's own convolution
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("la,lb", [(2, 2), (3, 5), (7, 1), (11, 4), (1, 1), (16, 16)])
def test_convolve_direct_matches_numpy(la, lb):
    rng = np.random.default_rng(la * 100 + lb)
    a, b = rng.random(la), rng.random(lb)
    assert np.max(np.abs(convolve_direct(a, b) - np.convolve(a, b))) < TOL


@pytest.mark.parametrize("la,lb", [(3, 5), (11, 4), (16, 16), (9, 9)])
def test_convolve_fft_matches_numpy_in_the_bulk(la, lb):
    rng = np.random.default_rng(la * 7 + lb)
    a, b = rng.random(la), rng.random(lb)
    ref = np.convolve(a, b)
    assert np.max(np.abs(convolve_fft(a, b) - ref)) < 1e-12 * ref.max()


@pytest.mark.parametrize("la,lb", [(3, 5), (11, 4), (6, 6)])
def test_convolve_log_matches_log_of_numpy(la, lb):
    rng = np.random.default_rng(la * 13 + lb)
    a, b = rng.random(la) + 1e-3, rng.random(lb) + 1e-3
    got = convolve_log(np.log(a), np.log(b))
    assert np.max(np.abs(got - np.log(np.convolve(a, b)))) < TOL


def test_convolve_log_handles_zero_entries():
    a = np.array([0.5, 0.0, 0.5])
    b = np.array([0.0, 1.0])
    got = np.exp(convolve_log(tr._safe_log(a), tr._safe_log(b)))
    assert np.max(np.abs(got - np.convolve(a, b))) < TOL


def test_convolve_is_commutative():
    rng = np.random.default_rng(3)
    a, b = rng.random(5), rng.random(9)
    assert np.max(np.abs(convolve_direct(a, b) - convolve_direct(b, a))) < TOL
    assert np.max(np.abs(convolve_log(np.log(a), np.log(b))
                         - convolve_log(np.log(b), np.log(a)))) < TOL


# ----------------------------------------------------------------------------------
# (2) the aggregate pmf against BRUTE-FORCE ENUMERATION of every joint outcome
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("method", ["log", "direct", "fft", "auto"])
def test_aggregate_matches_brute_force_enumeration(seed, method):
    rng = np.random.default_rng(seed)
    units = _units(rng, int(rng.integers(2, 6)))
    agg = aggregate_distribution(units, method=method)
    bf = brute_force_aggregate(units)
    assert agg.offset == bf.offset
    assert len(agg) == len(bf)
    assert np.max(np.abs(agg.pmf - bf.pmf)) < 1e-14


@pytest.mark.parametrize("seed", range(6))
def test_aggregate_pmf_sums_to_one(seed):
    rng = np.random.default_rng(100 + seed)
    agg = aggregate_distribution(_units(rng, 5))
    assert abs(float(agg.pmf.sum()) - 1.0) < 1e-14
    assert abs(agg.total_mass - 1.0) < 1e-14


@pytest.mark.parametrize("seed", range(5))
def test_aggregate_mean_and_variance_are_additive(seed):
    rng = np.random.default_rng(200 + seed)
    units = _units(rng, 6)
    agg = aggregate_distribution(units)
    assert abs(agg.mean - sum(u.mean for u in units)) < 1e-11
    assert abs(agg.var - sum(u.var for u in units)) < 1e-10


def test_units_are_heterogeneous_not_all_identical():
    """The whole point is heterogeneity; guard against a degenerate test fixture."""
    pf = demo_portfolio()
    sigs = {(u.offset, len(u), tuple(np.round(u.pmf, 12))) for u in pf.units}
    assert len(sigs) >= len(pf.units) - 2
    assert len({len(u) for u in pf.units}) > 1


# ----------------------------------------------------------------------------------
# (3) tail probabilities, VaR and expected shortfall against enumeration
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_tail_probability_matches_brute_force_at_every_threshold(seed):
    rng = np.random.default_rng(300 + seed)
    units = _units(rng, 4)
    agg = aggregate_distribution(units)
    for t in range(agg.offset - 2, agg.offset + len(agg) + 2):
        assert abs(tail_probability(agg, t) - brute_force_tail(units, t)) < 1e-14


@pytest.mark.parametrize("seed", range(4))
def test_upper_and_lower_tails_partition_the_mass(seed):
    rng = np.random.default_rng(400 + seed)
    agg = aggregate_distribution(_units(rng, 5))
    for t in agg.support:
        up = tail_probability(agg, t, side="upper")
        low = tail_probability(agg, t - 1, side="lower")
        assert abs(up + low - 1.0) < 1e-13


def test_tail_probability_out_of_range():
    agg = aggregate_distribution([{0: 0.5, 1: 0.5}, {0: 0.5, 1: 0.5}])
    assert tail_probability(agg, -5) == pytest.approx(1.0)
    assert tail_probability(agg, 99) == 0.0
    assert log_tail_probability(agg, 99) == -math.inf
    assert tail_probability(agg, -5, side="lower") == 0.0


def test_tail_probability_accepts_a_raw_array():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert tail_probability(p, 2, offset=0) == pytest.approx(0.7)
    assert tail_probability(p, 12, offset=10) == pytest.approx(0.7)


@pytest.mark.parametrize("seed", range(4))
def test_log_tail_curve_matches_pointwise(seed):
    rng = np.random.default_rng(500 + seed)
    agg = aggregate_distribution(_units(rng, 5))
    curve = log_tail_curve(agg)
    ref = np.array([log_tail_probability(agg, t) for t in agg.support])
    assert np.max(np.abs(curve - ref)) < 1e-12
    low = log_tail_curve(agg, side="lower")
    ref_low = np.array([log_tail_probability(agg, t, side="lower") for t in agg.support])
    assert np.max(np.abs(low - ref_low)) < 1e-12


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("alpha", [0.5, 0.9, 0.99])
def test_var_and_es_match_brute_force(seed, alpha):
    rng = np.random.default_rng(600 + seed)
    units = _units(rng, 4)
    agg = aggregate_distribution(units)
    bf = brute_force_aggregate(units)
    assert value_at_risk(agg, alpha) == value_at_risk(bf, alpha)
    assert abs(expected_shortfall(agg, alpha) - expected_shortfall(bf, alpha)) < 1e-12


# ----------------------------------------------------------------------------------
# (4) exact rational ground truth -- no floats in that code path at all
# ----------------------------------------------------------------------------------
def test_exact_integer_aggregate_total_mass_is_an_integer_identity():
    pf = demo_portfolio(n_units=12)
    num, off, den = exact_integer_aggregate(pf.int_units)
    assert sum(num) == den == 10000 ** 12          # exact, no rounding anywhere
    assert off == 0


def test_exact_rational_matches_brute_force_on_a_small_case():
    pf = demo_portfolio(n_units=4, seed=11)
    bf = brute_force_aggregate(pf.units)
    for t in range(bf.offset, bf.offset + len(bf) + 1):
        fr = exact_rational_tail(pf.int_units, t)
        assert abs(float(fr) - brute_force_tail(pf.units, t)) < 1e-14


def test_exact_rational_confirms_the_deep_tail():
    """The 1e-30 claim has ground truth: bigint rationals, not floats."""
    pf = demo_portfolio()
    agg = aggregate_distribution(pf.units, method="log")
    curve = log_tail_curve(agg)
    t = int(agg.support[int(np.argmin(np.abs(curve - math.log(1e-30))))])
    fr = exact_rational_tail(pf.int_units, t)
    assert isinstance(fr, Fraction)
    got = tail_probability(agg, t)
    assert 1e-34 < float(fr) < 1e-26
    assert abs(got - float(fr)) / float(fr) < 1e-12


def test_dyadic_inputs_are_bit_exact_so_error_is_pure_algorithm():
    pf = demo_portfolio(n_units=40, denominator=1024)
    for u, iu in zip(pf.units, pf.int_units):
        for v, n in zip(u.pmf, iu.num):
            assert Fraction(float(v)) == Fraction(n, 1024)
    agg = aggregate_distribution(pf.units, method="log")
    curve = log_tail_curve(agg)
    t = int(agg.support[int(np.argmin(np.abs(curve - math.log(1e-30))))])
    fr = exact_rational_tail(pf.int_units, t)
    rel = abs(tail_probability(agg, t) - float(fr)) / float(fr)
    assert rel < 1e-13


# ----------------------------------------------------------------------------------
# (5) THE HEADLINE: Monte Carlo agrees in the bulk (positive control) and returns
#     EXACTLY ZERO in the deep tail where the exact method returns ~1e-30
# ----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def headline():
    pf = demo_portfolio()
    agg = aggregate_distribution(pf.units, method="log")
    curve = log_tail_curve(agg)

    def closest(x):
        return int(agg.support[int(np.argmin(np.abs(curve - math.log(x))))])

    t_bulk, t_deep = closest(1e-2), closest(1e-30)
    mc = monte_carlo_tail(pf.units, [t_bulk, t_deep], n_samples=2_000_000, seed=1)
    return pf, agg, t_bulk, t_deep, mc


def test_monte_carlo_positive_control_in_the_bulk(headline):
    """Before any null claim: the SAME MC pipeline must recover the exact bulk value."""
    _pf, agg, t_bulk, _t_deep, mc = headline
    exact = tail_probability(agg, t_bulk)
    assert mc["hits"][0] > 1000
    z = abs(mc["estimate"][0] - exact) / mc["stderr"][0]
    assert z < 5.0, f"MC bulk {mc['estimate'][0]} vs exact {exact}, {z} sigma"


def test_monte_carlo_returns_exactly_zero_where_exact_returns_1e30(headline):
    _pf, agg, _t_bulk, t_deep, mc = headline
    exact = tail_probability(agg, t_deep)
    assert 1e-34 < exact < 1e-26
    assert mc["hits"][1] == 0
    assert mc["estimate"][1] == 0.0
    assert mc["max_sample_sum"] < t_deep
    # the only honest statement MC can make is 24-ish orders of magnitude too weak
    assert mc["rule_of_three_bound"] / exact > 1e20


def test_monte_carlo_recovers_a_planted_answer_exactly(headline):
    """Positive control for the sampler itself: a unit with a known mean."""
    u = Unit(0, np.array([0.25, 0.25, 0.5]))
    rng = np.random.default_rng(4)
    x = u.sample(400_000, rng)
    assert abs(x.mean() - u.mean) < 0.01
    assert set(np.unique(x)) <= {0, 1, 2}


def test_monte_carlo_scalar_threshold_shape():
    pf = demo_portfolio(n_units=6)
    mc = monte_carlo_tail(pf.units, 3, n_samples=20_000, seed=2)
    assert isinstance(mc["hits"], int)
    assert isinstance(mc["estimate"], float)
    agg = aggregate_distribution(pf.units)
    assert abs(mc["estimate"] - tail_probability(agg, 3)) < 0.02


# ----------------------------------------------------------------------------------
# (6) numerical care: FFT loses the tail, direct does not, log space has no floor
# ----------------------------------------------------------------------------------
def test_fft_loses_the_deep_tail_and_direct_does_not():
    pf = demo_portfolio()
    ex = aggregate_distribution(pf.units, method="log").pmf
    fft = aggregate_distribution(pf.units, method="fft").pmf
    dirc = aggregate_distribution(pf.units, method="direct").pmf
    deep = (ex > 0) & (ex < 1e-20)
    assert deep.sum() > 10
    rel_fft = np.max(np.abs(fft[deep] - ex[deep]) / ex[deep])
    rel_dir = np.max(np.abs(dirc[deep] - ex[deep]) / ex[deep])
    assert rel_fft > 1e3, "FFT was expected to destroy the tail"
    assert rel_dir < 1e-9, "direct convolution must keep relative accuracy"
    # and FFT produces entries that are impossible for a probability
    assert np.count_nonzero(fft < 0) > 0


def test_fft_is_fine_in_the_bulk():
    pf = demo_portfolio()
    ex = aggregate_distribution(pf.units, method="log").pmf
    fft = aggregate_distribution(pf.units, method="fft").pmf
    bulk = ex > 1e-6
    assert np.max(np.abs(fft[bulk] - ex[bulk]) / ex[bulk]) < 1e-10


def test_log_space_is_required_past_the_float64_floor():
    pf = demo_portfolio(n_units=600, seed=7)
    lg = aggregate_distribution(pf.units, method="log")
    lin = aggregate_distribution(pf.units, method="direct")
    t = int(lg.offset + len(lg) - 1)
    assert math.isfinite(log_tail_probability(lg, t))
    assert log_tail_probability(lg, t) < -1000.0           # p far below 1e-308
    assert tail_probability(lin, t) == 0.0                 # linear space says nothing
    # they still agree wherever linear space can represent the answer
    curve = log_tail_curve(lg)
    ok = curve > math.log(1e-250)
    lin_curve = log_tail_curve(lin)
    assert np.max(np.abs(curve[ok] - lin_curve[ok])) < 1e-9


def test_auto_method_picks_log_when_the_tail_would_underflow():
    small = aggregate_distribution(demo_portfolio(n_units=5).units, method="auto")
    big = aggregate_distribution(demo_portfolio(n_units=600, seed=7).units, method="auto")
    assert small.info["method"] == "direct"
    assert big.info["method"] == "log"
    assert small.info["linear_space_sufficient"] is True
    assert big.info["linear_space_sufficient"] is False


def test_underflow_report_numbers():
    fl = underflow_report()
    assert fl["float64_smallest_normal"] == pytest.approx(2.2250738585072014e-308)
    assert fl["float64_smallest_subnormal"] == pytest.approx(5e-324)
    assert fl["log_space_floor_as_log_p"] < -1e300


# ----------------------------------------------------------------------------------
# (7) treewidth / bond dimension logging -- THE GOVERNING LAW
# ----------------------------------------------------------------------------------
def test_bond_dimension_is_the_running_support_and_treewidth_is_one():
    pf = demo_portfolio(n_units=10)
    agg = aggregate_distribution(pf.units)
    info = agg.info
    assert info["treewidth"] == 1
    bonds = info["bond_dimensions"]
    assert len(bonds) == 10
    assert bonds[0] == len(pf.units[0])
    assert bonds[-1] == info["support_size"] == len(agg)
    assert bonds == sorted(bonds)                      # the running support only grows
    # cost = sum_k d_k^1 * m_k, and it is nowhere near the joint frontier prod_i m_i
    joint = int(np.prod([len(u) for u in pf.units]))
    assert info["cost_multiply_adds"] < joint


def test_cost_is_linear_in_the_number_of_units():
    a = aggregate_distribution(demo_portfolio(n_units=50, seed=3).units, method="direct")
    b = aggregate_distribution(demo_portfolio(n_units=100, seed=3).units, method="direct")
    # d_max doubles and N doubles => cost grows ~4x, not exponentially
    ratio = b.info["cost_multiply_adds"] / a.info["cost_multiply_adds"]
    assert 2.0 < ratio < 8.0


# ----------------------------------------------------------------------------------
# (8) mixture / common latent factor
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(4))
def test_mixture_matches_brute_force(seed):
    rng = np.random.default_rng(700 + seed)
    w = rng.random(3)
    w /= w.sum()
    sets = [_units(rng, 3) for _ in range(3)]
    mix = mixture_aggregate(w, sets)
    bfs = [brute_force_aggregate(s) for s in sets]
    lo = min(b.offset for b in bfs)
    hi = max(b.offset + len(b) for b in bfs)
    ref = np.zeros(hi - lo)
    for wi, b in zip(w, bfs):
        ref[b.offset - lo:b.offset - lo + len(b)] += wi * b.pmf
    assert mix.offset == lo
    assert np.max(np.abs(mix.pmf - ref)) < 1e-14
    assert abs(mix.pmf.sum() - 1.0) < 1e-13


# ----------------------------------------------------------------------------------
# (9) input coercion and validation
# ----------------------------------------------------------------------------------
def test_as_unit_forms_agree():
    a = as_unit({0: 0.25, 2: 0.75})
    b = as_unit([0.25, 0.0, 0.75])
    c = as_unit((0, np.array([0.25, 0.0, 0.75])))
    d = as_unit(Unit(0, np.array([0.25, 0.0, 0.75])))
    for x in (b, c, d):
        assert x.offset == a.offset
        assert np.max(np.abs(x.pmf - a.pmf)) < 1e-15


def test_as_unit_two_element_pmf_is_not_mistaken_for_offset_pair():
    u = as_unit((0.3, 0.7))
    assert u.offset == 0
    assert np.max(np.abs(u.pmf - np.array([0.3, 0.7]))) < 1e-15


@pytest.mark.parametrize("bad", [[0.5, 0.6], [-0.1, 1.1], [np.nan, 1.0], []])
def test_unit_rejects_bad_pmf(bad):
    with pytest.raises(ValueError):
        Unit(0, np.asarray(bad, dtype=float))


def test_intunit_rejects_wrong_denominator():
    with pytest.raises(ValueError):
        IntUnit(0, [1, 2], 4)


def test_aggregate_rejects_unknown_method_and_empty_input():
    with pytest.raises(ValueError):
        aggregate_distribution([{0: 1.0}], method="nope")
    with pytest.raises(ValueError):
        aggregate_distribution([])


def test_offsets_add():
    a = Unit(3, np.array([0.5, 0.5]))
    b = Unit(-7, np.array([0.25, 0.75]))
    agg = aggregate_distribution([a, b])
    assert agg.offset == -4
    assert np.max(np.abs(agg.pmf - np.array([0.125, 0.5, 0.375]))) < 1e-15


def test_single_unit_is_itself():
    u = Unit(2, np.array([0.1, 0.9]))
    agg = aggregate_distribution([u])
    assert agg.offset == 2
    assert np.max(np.abs(agg.pmf - u.pmf)) < 1e-15
    assert agg.info["bond_dimensions"] == [2]


# ----------------------------------------------------------------------------------
# (10) the module's own verify()
# ----------------------------------------------------------------------------------
def test_module_verify_passes():
    r = tr.verify(verbose=False, n_samples=2_000_000)
    assert r["ok"]
    assert r["a"]["max_err_log"] < 1e-14
    assert r["a"]["max_err_mass"] < 1e-14
    assert r["b"]["mc_hits"][2] == 0
    assert 1e-34 < r["b"]["p_deep"] < 1e-26
    assert r["b"]["rel_err_deep_vs_rational"] < 1e-12
    assert r["b"]["rational_mass_exact"]
    assert r["c"]["rel_fft_deep"] > 1e3
    assert r["c"]["rel_direct_deep"] < 1e-9
    assert r["c"]["big_linear_tail"] == 0.0
    assert r["d"]["inputs_bit_exact"]
