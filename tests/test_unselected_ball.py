"""The reachable-set ceiling rests entirely on one closed form, so it is tested here.

If rmsd(t)^2 = rmsd_min^2 + ||t - t*||^2 is wrong, or if the enumeration misses a single
qualifying grid translation, the ceiling silently understates what the search could reach --
and a ceiling that is too low is exactly the error that would let "the search is the binding
constraint" be reported when it is not.
"""
import numpy as np
from benchmarks.db5_unselected import quad_ball, shifts_in_ball


def test_quadratic_identity():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        d0 = rng.normal(size=(int(rng.integers(5, 50)), 3)) * rng.uniform(0.5, 20)
        tstar, rmin, _rad = quad_ball(d0, 10.0)
        for _ in range(20):
            t = rng.normal(size=3) * rng.uniform(0, 30)
            lhs = np.sqrt(((d0 + t) ** 2).sum(1).mean())
            rhs = np.sqrt(rmin ** 2 + ((t - tstar) ** 2).sum())
            worst = max(worst, abs(lhs - rhs))
    assert worst < 1e-9, worst


def test_ball_is_sound_and_complete():
    """Sound: every enumerated shift meets the bar. Complete: no shift meeting the bar is
    left out. Completeness is the half that matters for a ceiling."""
    rng = np.random.default_rng(1)
    bad = miss = 0
    for _ in range(120):
        d0 = rng.normal(size=(int(rng.integers(5, 40)), 3)) * rng.uniform(0.5, 6)
        tstar, _rmin, rad = quad_ball(d0, 10.0)
        sh = shifts_in_ball(tstar, rad, 1.5, (64, 64, 64))
        inside = set(map(tuple, sh.tolist()))
        for s in sh[:50]:
            if np.sqrt(((d0 + np.asarray(s, float) * 1.5) ** 2).sum(1).mean()) > 10.0 + 1e-9:
                bad += 1
        for _ in range(200):
            s = tuple(int(x) for x in rng.integers(-32, 32, size=3))
            if (np.sqrt(((d0 + np.asarray(s, float) * 1.5) ** 2).sum(1).mean()) <= 10.0
                    and s not in inside):
                miss += 1
    assert bad == 0, f"{bad} enumerated shifts violate the bar"
    assert miss == 0, f"{miss} qualifying shifts were not enumerated"


def test_constant_displacement_is_fully_removable():
    """A rigid offset, however large, is undone by a translation -- so rmsd_min is 0 and the
    ball is non-empty. Getting this backwards would make the ceiling too LOW."""
    d0 = np.full((10, 3), 100.0)
    tstar, rmin, rad = quad_ball(d0, 10.0)
    assert rmin < 1e-9 and np.isfinite(rad)
    assert np.allclose(tstar, -100.0)


def test_empty_ball_when_unreachable():
    """Spread, not offset, is what a translation cannot fix. Here the spread exceeds the bar,
    so no translation qualifies and the ball must be empty."""
    rng = np.random.default_rng(3)
    d0 = rng.normal(size=(40, 3)) * 50.0
    tstar, rmin, rad = quad_ball(d0, 10.0)
    assert rmin > 10.0
    assert not np.isfinite(rad)
    assert len(shifts_in_ball(tstar, rad, 1.5, (64, 64, 64))) == 0


def test_shifts_stay_inside_the_representable_fold():
    """The enumeration must never invent a translation the FFT fold cannot express, or the
    ceiling would count poses the search could not produce."""
    rng = np.random.default_rng(4)
    n = 64
    for _ in range(50):
        d0 = rng.normal(size=(20, 3)) * rng.uniform(0.5, 4)
        tstar, _r, rad = quad_ball(d0, 10.0)
        sh = shifts_in_ball(tstar, rad, 1.5, (n, n, n))
        if len(sh):
            assert sh.min() >= -(n // 2) and sh.max() <= n - n // 2 - 1


def test_fastcapri_agrees_with_capri_metrics():
    """FastCapri hoists per-complex constants out of capri.f_nat so an exhaustive ceiling
    scan is affordable. It must be an optimisation and not a redefinition: if it disagreed,
    the ceiling would be measuring a different quantity than the benchmark reports."""
    import numpy as np
    from rem.docking import capri
    from rem.docking.data import load_case
    from rem.docking.rigid import apply_pose, randomize_pose, rotation_set
    from benchmarks.db5_unselected import FastCapri

    case = load_case("1A2K")
    rec, lig = case["r_u"], case["l_u"]
    native = lig.coords.copy()
    masks = capri.interface_mask(rec, lig, native)
    moved, _R, _t = randomize_pose(native, seed=7, max_shift=20.0)
    rots = rotation_set(8, seed=1)
    fast = FastCapri(rec, lig, native, masks)
    rng = np.random.default_rng(0)
    worst_f = worst_i = worst_l = 0.0
    for R in rots:
        for _ in range(3):
            t = rng.normal(size=3) * 12.0
            c = apply_pose(moved, R, t, centre=moved.mean(axis=0))
            a = capri.capri_metrics(rec, lig, native, c, masks)
            b = fast.metrics(c)
            assert a["quality"] == b["quality"], (a, b)
            worst_f = max(worst_f, abs(a["f_nat"] - b["f_nat"]))
            worst_i = max(worst_i, abs(a["I_rmsd"] - b["I_rmsd"]))
            worst_l = max(worst_l, abs(a["L_rmsd"] - b["L_rmsd"]))
    assert worst_f < 1e-12 and worst_i < 1e-9 and worst_l < 1e-9, (worst_f, worst_i, worst_l)
