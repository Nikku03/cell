"""R1-R4 for the perturbation-reach map."""
import numpy as np
from rem.reach import (topology, distances, reach, linear_eps, by_distance,
                       correlation_length, decay_length, verify, LIN_TOL, UNIFORM_TOL)

N = 10


def test_r1_linearity_is_a_precondition_that_can_fail():
    """linear_eps must REJECT a perturbation that is not linear, not merely report on it."""
    regs = topology("chain", N)
    eps = linear_eps(N, regs)
    assert eps is not None and eps <= 0.10
    r1, _m, _d = reach(N, regs, eps=eps)
    r2, _m, _d = reach(N, regs, eps=eps / 2)
    big = np.abs(r1) > 1e-9
    assert np.abs(r2[big] / r1[big] - 0.5).max() <= LIN_TOL
    # and a deliberately huge perturbation must be rejected by the same function
    assert linear_eps(N, regs, candidates=(0.8,)) is None


def test_r3_local_wiring_decays_and_hub_plateaus():
    eps = linear_eps(N, topology("chain", N))
    ch = by_distance(*reach(N, topology("chain", N), eps=eps)[::2])
    ds = sorted(ch)
    assert all(ch[ds[i + 1]] < ch[ds[i]] for i in range(len(ds) - 2)), ch
    hub_rel, _m, _d = reach(N, topology("hub", N), eps=eps)
    far = np.abs(hub_rel[2:])
    assert far.std() / far.mean() <= UNIFORM_TOL          # the plateau is near-uniform


def test_r3_far_field_uniformity_is_what_licenses_one_aggregate():
    eps = linear_eps(N, topology("chain", N))
    rel, _m, _d = reach(N, topology("chain+global", N), eps=eps)
    far = np.abs(rel[2:])
    assert far.std() / far.mean() <= UNIFORM_TOL


def test_r4_direct_resolve_matches_finite_difference_in_the_linear_regime():
    regs = topology("chain", N)
    eps = linear_eps(N, regs)
    r, _m, _d = reach(N, regs, eps=eps)
    rs, _m, _d = reach(N, regs, eps=eps / 100)
    fd = rs / (eps / 100) * eps
    big = np.abs(r) > 1e-9
    assert np.abs(fd[big] - r[big]).max() / np.abs(r[big]).max() < 0.05


def test_r2_is_void_not_passed():
    """The correlation-length substitute is NOT equivalent out of equilibrium; the module must
    report it as void rather than quietly claiming two independent routes agree."""
    out = verify(n=N, verbose=False)
    assert out["R2"] == "VOID"
    assert np.isfinite(out["xi_pert"]) and np.isfinite(out["xi_corr"])
