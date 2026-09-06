"""FFT translation search against brute force, and the sign convention pinned by test.

The sign/off-by-one/wraparound trap is settled here by MEASUREMENT: every planted shift
must come back with exact integer equality, not approximate agreement.
"""
import itertools
import math

import numpy as np
import pytest

from rem import fftcorr
from rem.fftcorr import (auto_origin, best_translation, correlate, correlate_direct,
                         correlate_direct_blocked, cost_model, ligand_grid,
                         receptor_grid, shift_to_world, top_translations,
                         voxelize, voxelize_bruteforce)

TOL = 1e-10


def _rand_grid(rng, shape, fill=0.5):
    g = np.zeros(shape)
    m = rng.random(shape) < fill
    g[m] = rng.normal(size=int(m.sum()))
    return g


# ----------------------------------------------------------------------------------
# (a) FFT correlation == explicit nested-loop direct correlation
# ----------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(3, 3, 3), (4, 4, 4), (5, 4, 3), (2, 5, 4)])
def test_fft_matches_direct_circular(shape):
    rng = np.random.default_rng(abs(hash(shape)) % 2**31)
    R, L = _rand_grid(rng, shape), _rand_grid(rng, shape)
    assert np.max(np.abs(correlate(R, L) - correlate_direct(R, L))) < TOL


@pytest.mark.parametrize("sign", ["plus", "minus"])
def test_fft_matches_direct_both_signs(sign):
    rng = np.random.default_rng(5)
    shape = (4, 5, 3)
    R, L = _rand_grid(rng, shape), _rand_grid(rng, shape)
    err = np.max(np.abs(correlate(R, L, sign=sign)
                        - correlate_direct(R, L, sign=sign)))
    assert err < TOL


@pytest.mark.parametrize("rshape,lshape", [((4, 3, 3), (3, 2, 3)),
                                           ((5, 4, 4), (2, 3, 2)),
                                           ((3, 3, 3), (4, 4, 4))])
def test_fft_matches_direct_linear(rshape, lshape):
    rng = np.random.default_rng(9)
    R, L = _rand_grid(rng, rshape), _rand_grid(rng, lshape)
    Sf = correlate(R, L, mode="linear")
    Sd = correlate_direct(R, L, mode="linear")
    assert Sf.shape == tuple(a + b - 1 for a, b in zip(rshape, lshape))
    assert np.max(np.abs(Sf - Sd)) < TOL


def test_blocked_direct_matches_pure_loop_direct():
    """The timing reference must compute the same thing as the correctness reference."""
    rng = np.random.default_rng(13)
    for shape in [(4, 4, 4), (5, 3, 4)]:
        R, L = _rand_grid(rng, shape), _rand_grid(rng, shape)
        assert np.max(np.abs(correlate_direct(R, L)
                             - correlate_direct_blocked(R, L))) < TOL


# ----------------------------------------------------------------------------------
# analytic identities -- independent of both implementations
# ----------------------------------------------------------------------------------

def test_total_score_equals_product_of_sums():
    """sum_t S[t] = (sum R)(sum L). Closed form, no correlation code involved."""
    rng = np.random.default_rng(17)
    shape = (6, 5, 4)
    R, L = _rand_grid(rng, shape), _rand_grid(rng, shape)
    S = correlate(R, L)
    assert abs(S.sum() - R.sum() * L.sum()) < 1e-9


def test_zero_shift_score_equals_elementwise_dot():
    """S[0] = sum_x R[x] L[x]. Closed form."""
    rng = np.random.default_rng(19)
    shape = (5, 5, 5)
    R, L = _rand_grid(rng, shape), _rand_grid(rng, shape)
    assert abs(correlate(R, L)[0, 0, 0] - float(np.sum(R * L))) < 1e-12


@pytest.mark.parametrize("a,b", [((0, 0, 0), (0, 0, 0)), ((3, 1, 2), (1, 4, 0)),
                                 ((1, 0, 5), (5, 3, 1))])
def test_two_deltas_peak_at_difference(a, b):
    """Delta at a correlated with delta at b must peak at exactly a - b."""
    shape = (7, 6, 6)
    R, L = np.zeros(shape), np.zeros(shape)
    R[a] = 1.0
    L[b] = 1.0
    S = correlate(R, L)
    want = np.array([((ai - bi + n // 2) % n) - n // 2
                     for ai, bi, n in zip(a, b, shape)], dtype=int)
    got, score = best_translation(S)
    assert np.array_equal(got, want)
    assert abs(score - 1.0) < 1e-12
    assert abs(S.sum() - 1.0) < 1e-9          # exactly one nonzero translation


# ----------------------------------------------------------------------------------
# (b) THE SIGN CONVENTION: planted shift must come back with EXACT integer equality
# ----------------------------------------------------------------------------------

SHIFTS = [(0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3), (2, -3, 1), (-4, 4, -2),
          (5, 0, 3), (-1, -1, -1), (5, -5, 3), (-5, 4, -3), (-6, 5, 4),
          (11, 9, 7), (-11, -9, -7), (6, -5, -4)]
SHAPE = (12, 10, 8)


def _folded(shift, shape=SHAPE):
    return np.array([((s + n // 2) % n) - n // 2 for s, n in zip(shift, shape)],
                    dtype=int)


def _planted_pair(shift, seed=7, shape=SHAPE):
    rng = np.random.default_rng(seed)
    L = np.zeros(shape)
    L[1:4, 1:3, 1:5] = rng.random((3, 2, 4)) + 0.5      # asymmetric, strictly positive
    R = np.roll(L, shift=shift, axis=(0, 1, 2))         # R[x] = L[x - shift]
    return R, L


@pytest.mark.parametrize("shift", SHIFTS)
def test_planted_translation_recovered_exactly(shift):
    """EXACT integer equality. Negative components and wrapping shifts included."""
    R, L = _planted_pair(shift)
    got, score = best_translation(correlate(R, L))
    want = _folded(shift)
    assert np.array_equal(got, want), f"planted {shift} folded {want} got {got}"
    assert got.dtype.kind == "i"
    assert score > 0


@pytest.mark.parametrize("shift", SHIFTS)
def test_planted_translation_recovered_by_direct_too(shift):
    """The independent O(N^6) search must land on the identical integer vector."""
    R, L = _planted_pair(shift)
    g_fft, _ = best_translation(correlate(R, L))
    g_dir, _ = best_translation(correlate_direct_blocked(R, L))
    assert np.array_equal(g_fft, g_dir)
    assert np.array_equal(g_dir, _folded(shift))


@pytest.mark.parametrize("shift", SHIFTS)
def test_sign_minus_is_the_trap(shift):
    """sign='minus' peaks at MINUS the planted shift. The trap, demonstrated."""
    R, L = _planted_pair(shift)
    got, _ = best_translation(correlate(R, L, sign="minus"))
    want = np.array([((-s + n // 2) % n) - n // 2
                     for s, n in zip(shift, SHAPE)], dtype=int)
    assert np.array_equal(got, want)


@pytest.mark.parametrize("shift", [(2, -3, 1), (-5, 4, -3), (-1, -1, -1)])
def test_raw_index_is_the_shift_mod_n(shift):
    """The unsigned peak index is the shift reduced mod the grid: the wraparound rule."""
    R, L = _planted_pair(shift)
    raw, _ = best_translation(correlate(R, L), signed=False)
    assert np.array_equal(raw, np.array([s % n for s, n in zip(shift, SHAPE)]))
    signed, _ = best_translation(correlate(R, L))
    assert np.array_equal(signed % np.array(SHAPE), raw)


def test_signed_fold_covers_the_fftfreq_range():
    for n in (7, 8, 12, 16):
        vals = sorted((((r + n // 2) % n) - n // 2) for r in range(n))
        assert vals == list(range(-(n // 2), n - n // 2))
        assert vals == sorted(int(v) for v in np.fft.fftfreq(n, 1.0 / n))


# ----------------------------------------------------------------------------------
# linear mode with mismatched shapes
# ----------------------------------------------------------------------------------

@pytest.mark.parametrize("rshape,lshape,p,q", [
    ((4, 4, 4), (6, 6, 6), (0, 0, 0), (3, 2, 1)),      # negative planted shift
    ((6, 5, 4), (4, 4, 3), (3, 2, 1), (0, 0, 0)),      # positive
    ((5, 5, 5), (5, 5, 5), (2, 0, 3), (0, 3, 1)),      # mixed sign
    ((7, 6, 5), (3, 3, 3), (4, 3, 2), (1, 1, 0)),
])
def test_linear_mode_exact_recovery(rshape, lshape, p, q):
    rng = np.random.default_rng(11)
    P = rng.random((2, 2, 2)) + 0.5
    R, L = np.zeros(rshape), np.zeros(lshape)
    R[p[0]:p[0] + 2, p[1]:p[1] + 2, p[2]:p[2] + 2] = P
    L[q[0]:q[0] + 2, q[1]:q[1] + 2, q[2]:q[2] + 2] = P
    got, _ = best_translation(correlate(R, L, mode="linear"), ligand_shape=lshape)
    assert np.array_equal(got, np.array(p, dtype=int) - np.array(q, dtype=int))


def test_linear_fold_is_a_bijection_onto_the_valid_window():
    rshape, lshape = (7, 6, 5), (3, 4, 2)
    out = tuple(a + b - 1 for a, b in zip(rshape, lshape))
    seen = set()
    for raw in np.ndindex(*out):
        f = fftcorr._fold(np.array(raw, dtype=int), out, lshape)
        assert np.all(f % np.array(out) == np.array(raw))
        assert np.all(f >= -(np.array(lshape) - 1))
        assert np.all(f <= np.array(rshape) - 1)
        seen.add(tuple(int(v) for v in f))
    assert len(seen) == int(np.prod(out))


def test_linear_mode_has_no_wraparound():
    """A ligand pushed past the receptor edge must score 0, not wrap into the far side."""
    R, L = np.zeros((5, 5, 5)), np.zeros((5, 5, 5))
    R[0, 0, 0] = 1.0
    L[4, 4, 4] = 1.0
    S = correlate(R, L, mode="linear")
    assert S.shape == (9, 9, 9)
    assert abs(S.sum() - 1.0) < 1e-12                 # exactly one overlapping offset
    got, _ = best_translation(S, ligand_shape=(5, 5, 5))
    assert np.array_equal(got, np.array([-4, -4, -4]))
    # circular mode on the same data wraps that offset to the opposite corner
    Sc = correlate(R, L, mode="circular")
    got_c, _ = best_translation(Sc)
    assert np.array_equal(got_c, np.array([1, 1, 1]))


# ----------------------------------------------------------------------------------
# reading the score volume
# ----------------------------------------------------------------------------------

def test_top_translations_sorted_and_consistent():
    rng = np.random.default_rng(23)
    S = _rand_grid(rng, (6, 5, 4), fill=1.0)
    tops = top_translations(S, 7)
    scores = [sc for _, sc in tops]
    assert scores == sorted(scores, reverse=True)
    assert abs(scores[0] - float(S.max())) < 1e-12
    best, bscore = best_translation(S)
    assert np.array_equal(best, tops[0][0]) and abs(bscore - scores[0]) < 1e-12
    for sh, sc in tops:
        raw = tuple(int(v) % n for v, n in zip(sh, S.shape))
        assert abs(S[raw] - sc) < 1e-12


def test_shift_to_world():
    assert np.allclose(shift_to_world([2, -3, 1], 0.75), [1.5, -2.25, 0.75])
    assert np.allclose(shift_to_world([1, 1, 1], [1.0, 2.0, 0.5]), [1.0, 2.0, 0.5])


def test_cost_model_reports_the_law():
    c = cost_model((64, 64, 64))
    assert c["n_translations"] == 64 ** 3
    assert c["direct_ops"] == 64 ** 6
    assert c["effective_treewidth"] == 0 and c["bond_dimension_fourier"] == 1
    assert c["predicted_speedup"] > 1000


# ----------------------------------------------------------------------------------
# voxelization
# ----------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_voxelize_matches_pure_python_bruteforce(seed):
    rng = np.random.default_rng(400 + seed)
    natoms = int(rng.integers(1, 6))
    coords = rng.normal(scale=2.5, size=(natoms, 3)) + math.pi / 7.0
    radii = rng.uniform(1.0, 2.2, size=natoms)
    shape = (9, 10, 8)
    g1 = voxelize(coords, radii, shape, spacing=0.9, surface_value=1.0,
                  core_value=-15.0, probe=1.4)
    g2 = voxelize_bruteforce(coords, radii, shape, spacing=0.9, surface_value=1.0,
                             core_value=-15.0, probe=1.4)
    assert np.array_equal(g1, g2)


def test_voxelize_anisotropic_spacing_and_explicit_origin():
    coords = np.array([[0.3, -0.2, 0.1], [2.1, 1.4, -1.7]])
    radii = np.array([1.6, 1.9])
    shape = (8, 7, 9)
    kw = dict(spacing=[0.7, 1.1, 0.5], origin=[-3.0, -3.5, -2.5],
              surface_value=2.0, core_value=-9.0, probe=1.0)
    assert np.array_equal(voxelize(coords, radii, shape, **kw),
                          voxelize_bruteforce(coords, radii, shape, **kw))


def test_voxelize_receptor_convention():
    """Surface shell strictly positive and strictly OUTSIDE the negative core."""
    coords = np.array([[0.0, 0.0, 0.0]])
    g = voxelize(coords, 2.0, (15, 15, 15), spacing=1.0, origin=[-7.0, -7.0, -7.0],
                 surface_value=1.0, core_value=-15.0, probe=1.5)
    core = g < 0
    shell = g > 0
    assert core.any() and shell.any()
    assert not (core & shell).any()
    ii = np.stack(np.meshgrid(*[np.arange(15) - 7.0] * 3, indexing="ij"))
    dist = np.sqrt((ii ** 2).sum(axis=0))
    assert dist[core].max() <= 2.0 + 1e-12
    assert dist[shell].min() > 2.0                 # shell is outside the core
    assert dist[shell].max() <= 3.5 + 1e-12        # and within r + probe
    assert set(np.unique(g)) == {-15.0, 0.0, 1.0}


def test_voxelize_scalar_radius_and_auto_origin_centres_molecule():
    coords = np.array([[10.0, 10.0, 10.0], [12.0, 10.0, 10.0]])
    shape = (16, 16, 16)
    org = auto_origin(coords, shape, 1.0)
    assert np.allclose(org, [11.0 - 7.5, 10.0 - 7.5, 10.0 - 7.5])
    g = voxelize(coords, 1.5, shape, spacing=1.0, probe=1.0)
    com_vox = np.array(np.nonzero(g != 0)).mean(axis=1)
    assert np.allclose(com_vox, (np.array(shape) - 1) / 2.0, atol=0.6)


def test_receptor_and_ligand_presets():
    coords = np.array([[0.0, 0.0, 0.0]])
    R = receptor_grid(coords, 2.0, (12, 12, 12), spacing=1.0,
                      origin=[-6.0, -6.0, -6.0])
    assert R.min() < 0 and R.max() > 0
    L = ligand_grid(coords, 2.0, (12, 12, 12), spacing=1.0, origin=[-6.0, -6.0, -6.0])
    assert set(np.unique(L)) == {0.0, 1.0}
    with pytest.raises(ValueError):
        receptor_grid(coords, 2.0, (8, 8, 8), core_value=1.0)


def test_voxelize_rejects_bad_input():
    with pytest.raises(ValueError):
        voxelize(np.zeros((2, 2)), 1.0, (4, 4, 4))
    with pytest.raises(ValueError):
        voxelize(np.zeros((2, 3)), [1.0, 2.0, 3.0], (4, 4, 4))
    with pytest.raises(ValueError):
        voxelize(np.zeros((1, 3)), 1.0, (4, 4))
    with pytest.raises(ValueError):
        voxelize(np.zeros((1, 3)), 1.0, (4, 4, 4), spacing=-1.0)
    with pytest.raises(ValueError):
        voxelize(np.zeros((1, 3)), -1.0, (4, 4, 4))


# ----------------------------------------------------------------------------------
# docking-shaped positive control, and the governing law
# ----------------------------------------------------------------------------------

def test_pocket_positive_control():
    """A ligand cube must be placed in the receptor's cavity, exactly."""
    res = fftcorr.verify_pocket(verbose=False)
    assert res["exact"], res
    assert res["score"] > res["runner_up"]


def test_voxelized_docking_recovers_a_planted_pose():
    """End to end: real atoms -> voxelize -> FFT -> exact planted translation."""
    rng = np.random.default_rng(31)
    shape = (24, 24, 24)
    coords = rng.normal(scale=2.0, size=(9, 3))
    Lg = ligand_grid(coords, 1.8, shape, spacing=1.0,
                     origin=coords.mean(axis=0) - np.array([4.0, 4.0, 4.0]))
    planted = (5, -7, 3)
    Rg = np.roll(Lg, shift=planted, axis=(0, 1, 2))
    got, _ = best_translation(correlate(Rg, Lg))
    assert np.array_equal(got, np.array(planted))
    assert np.allclose(shift_to_world(got, 1.0), [5.0, -7.0, 3.0])


def test_fourier_operator_is_diagonal():
    """Bond dimension 1 / treewidth 0 across a Fourier-mode cut, measured."""
    res = fftcorr.verify_fourier_diagonal(shape=(3, 3, 4), verbose=False)
    assert res["offdiag_ratio"] < 1e-12
    assert res["eigenvalue_err"] < 1e-10
    assert res["bond_dimension"] == 1 and res["effective_treewidth"] == 0


# ----------------------------------------------------------------------------------
# input validation on correlate
# ----------------------------------------------------------------------------------

def test_correlate_rejects_bad_input():
    R = np.zeros((4, 4, 4))
    with pytest.raises(ValueError):
        correlate(R, np.zeros((3, 4, 4)))                 # circular shape mismatch
    with pytest.raises(ValueError):
        correlate(R, R, mode="nonsense")
    with pytest.raises(ValueError):
        correlate(R, R, sign="nonsense")
    with pytest.raises(ValueError):
        correlate(np.zeros((4, 4)), np.zeros((4, 4)))     # not 3-D
    with pytest.raises(ValueError):
        best_translation(np.zeros((4, 4)))


# ----------------------------------------------------------------------------------
# the module-level verify()
# ----------------------------------------------------------------------------------

def test_verify_passes():
    res = fftcorr.verify(verbose=False, fast=True)
    assert res["ok"], res
    assert res["max_err_fft_vs_direct"] < 1e-10
    assert res["max_err_blocked_vs_direct"] < 1e-10
    assert res["planted"]["all_exact"]
    assert res["linear"]["all_exact"] and res["linear"]["fold_violations"] == 0
    assert res["pocket"]["exact"]
    assert res["voxelize_mismatched_voxels"] == 0
    assert res["fourier"]["offdiag_ratio"] < 1e-12
    assert res["effective_treewidth"] == 0


def test_fft_is_actually_faster_at_a_useful_size():
    """Not a scaling claim -- just that the speedup is real and large at n = 20."""
    import time
    rng = np.random.default_rng(41)
    R, L = _rand_grid(rng, (20, 20, 20)), _rand_grid(rng, (20, 20, 20))
    correlate(R, L)
    t0 = time.perf_counter(); Sf = correlate(R, L); t_fft = time.perf_counter() - t0
    t0 = time.perf_counter(); Sd = correlate_direct_blocked(R, L)
    t_dir = time.perf_counter() - t0
    assert np.max(np.abs(Sf - Sd)) < 1e-8
    assert t_dir > 50 * t_fft, f"fft {t_fft:.5f}s direct {t_dir:.5f}s"
