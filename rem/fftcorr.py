"""3D voxelization and FFT translation search: ONE FFT pair scores ALL translations.

THE GOVERNING LAW,  cost = d ** treewidth,  in its cheapest possible case.

A rigid-body translation search asks, for every one of the N^3 integer offsets t,

    S(t) = sum_x  R(x) * L(x - t)

R the receptor grid, L the ligand grid. Done directly that is N^3 translations times
N^3 voxels each = O(N^6). But the receptor-ligand coupling depends only on the
DIFFERENCE x - t, so the N^3 x N^3 score operator M[t, x] = L[(x - t) mod N] is
block-circulant with circulant blocks. Circulant means DIAGONAL in the Fourier basis:
distinct Fourier modes do not couple, the bond dimension across any cut between modes
is 1, the effective treewidth of the translation search is 0, and the whole thing
collapses to a pointwise multiply. Cost O(N^3 log N) for the transforms plus O(N^3)
for the multiply.

That "treewidth 0 / bond dimension 1" is MEASURED, not asserted: verify_fourier_diagonal()
builds M explicitly on a small grid, conjugates it by the 3D DFT matrix, and prints
max|off-diagonal| / max|diagonal|.

WHAT THIS DOES NOT BUY YOU. Only TRANSLATION factorizes. Rotations do not commute with
translation, so a 6-D rigid docking search is still (number of rotations) independent
FFT searches. The speedup is exactly N^3 / log N per rotation, nothing more.

THE SIGN CONVENTION IS A TRAP, and it is resolved here BY MEASUREMENT (see
verify_sign_convention() and test_fftcorr.py), never by reasoning:

    correlate(R, L)                 ->  S[t] = sum_x R[x] * L[(x - t) mod N]
                                        =  ifftn( fftn(R) * conj(fftn(L)) )
        the peak sits at +t, the shift that was actually applied to the ligand.

    correlate(R, L, sign="minus")   ->  S[t] = sum_x R[x] * L[(x + t) mod N]
                                        =  ifftn( conj(fftn(R)) * fftn(L) )
        the peak sits at -t. This is the trap; it is implemented so the test suite can
        demonstrate the wrong answer rather than describe it.

INDEX CONVENTION. The score volume is periodic with the grid shape. best_translation()
returns the SIGNED representative, using numpy's own fftfreq ordering:

    signed = ((raw + n // 2) % n) - n // 2       range [-(n//2), n - n//2 - 1]

so for n = 16 the raw index 13 is the shift -3, and the representable range is [-8, 7].

VOXEL CONVENTION. Voxel (i, j, k) has its centre at  origin + (i, j, k) * spacing.
voxelize() paints the union of atomic balls (radius r) with `core_value` and the shell
of thickness `probe` just outside them with `surface_value`. For a receptor use
core_value < 0 (interior overlap is punished) and surface_value > 0 (the accessible
shell is where a ligand should sit); for a ligand use the plain occupancy grid,
surface_value = 0, core_value = 1. receptor_grid() and ligand_grid() are those two
presets.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "voxelize", "voxelize_bruteforce", "receptor_grid", "ligand_grid", "auto_origin",
    "correlate", "correlate_direct", "correlate_direct_blocked",
    "best_translation", "top_translations", "shift_to_world", "cost_model",
    "verify_linear_mode", "verify_pocket",
    "benchmark", "verify", "verify_sign_convention", "verify_fourier_diagonal",
]


# --------------------------------------------------------------------------------------
# grid geometry
# --------------------------------------------------------------------------------------

def _shape3(grid_shape) -> Tuple[int, int, int]:
    s = tuple(int(v) for v in np.asarray(grid_shape).ravel())
    if len(s) != 3 or any(v <= 0 for v in s):
        raise ValueError(f"grid_shape must be three positive ints, got {grid_shape!r}")
    return s  # type: ignore[return-value]


def _spacing3(spacing) -> np.ndarray:
    sp = np.asarray(spacing, dtype=float).ravel()
    if sp.size == 1:
        sp = np.repeat(sp, 3)
    if sp.size != 3 or np.any(sp <= 0):
        raise ValueError(f"spacing must be a positive scalar or 3-vector, got {spacing!r}")
    return sp


def auto_origin(coords, grid_shape, spacing=1.0) -> np.ndarray:
    """World coordinate of voxel (0,0,0) that centres `coords` in the grid."""
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    shape = np.asarray(_shape3(grid_shape), dtype=float)
    sp = _spacing3(spacing)
    return coords.mean(axis=0) - (shape - 1.0) / 2.0 * sp


# --------------------------------------------------------------------------------------
# voxelization
# --------------------------------------------------------------------------------------

def voxelize(coords, radii, grid_shape, spacing: float = 1.0, origin=None,
             surface_value: float = 1.0, core_value: float = -15.0,
             probe: float = 1.4) -> np.ndarray:
    """Paint atoms onto a 3-D grid: negative core, positive surface shell.

    coords         (n, 3) atom centres in world units
    radii          scalar or (n,) atomic radii, same units
    grid_shape     (nx, ny, nz)
    spacing        scalar or 3-vector, world units per voxel
    origin         world coordinate of voxel (0,0,0); None -> centre the molecule
    surface_value  value on the shell of thickness `probe` just OUTSIDE the atoms
    core_value     value inside the atoms (make it negative for a receptor)
    probe          shell thickness in world units (1.4 = water probe)

    Core wins over shell where they would overlap. Vectorized per atom over its own
    bounding box; voxelize_bruteforce() is the independent global-sweep reference.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (n, 3), got {coords.shape}")
    n = coords.shape[0]
    rad = np.asarray(radii, dtype=float).ravel()
    if rad.size == 1:
        rad = np.repeat(rad, n)
    if rad.size != n:
        raise ValueError(f"radii must be scalar or length {n}, got {rad.size}")
    if np.any(rad < 0):
        raise ValueError("radii must be non-negative")
    if probe < 0:
        raise ValueError("probe must be non-negative")

    shape = _shape3(grid_shape)
    sp = _spacing3(spacing)
    org = auto_origin(coords, shape, sp) if origin is None else \
        np.asarray(origin, dtype=float).ravel()
    if org.size != 3:
        raise ValueError(f"origin must be a 3-vector, got {origin!r}")

    core = np.zeros(shape, dtype=bool)
    shell = np.zeros(shape, dtype=bool)
    for c, r in zip(coords, rad):
        rout = r + probe
        lo = np.floor((c - rout - org) / sp).astype(int)
        hi = np.ceil((c + rout - org) / sp).astype(int) + 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(shape, dtype=int))
        if np.any(lo >= hi):
            continue
        ax = [org[d] + np.arange(lo[d], hi[d], dtype=float) * sp[d] - c[d]
              for d in range(3)]
        d2 = (ax[0][:, None, None] ** 2 + ax[1][None, :, None] ** 2
              + ax[2][None, None, :] ** 2)
        sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        core[sl] |= d2 <= r * r
        shell[sl] |= d2 <= rout * rout

    grid = np.zeros(shape, dtype=float)
    grid[shell & ~core] = float(surface_value)
    grid[core] = float(core_value)
    return grid


def voxelize_bruteforce(coords, radii, grid_shape, spacing: float = 1.0, origin=None,
                        surface_value: float = 1.0, core_value: float = -15.0,
                        probe: float = 1.4) -> np.ndarray:
    """INDEPENDENT reference for voxelize(): a pure-Python sweep over every voxel.

    Different algorithm on purpose. voxelize() visits, per atom, only the voxels inside
    that atom's bounding box, using numpy array arithmetic on squared distances. This
    visits EVERY voxel, loops over EVERY atom in pure Python, and uses math.sqrt on the
    true Euclidean distance. No array broadcasting, no bounding boxes, no shared code.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float)).tolist()
    rad = np.asarray(radii, dtype=float).ravel()
    if rad.size == 1:
        rad = np.repeat(rad, len(coords))
    rad = rad.tolist()
    nx, ny, nz = _shape3(grid_shape)
    sx, sy, sz = [float(v) for v in _spacing3(spacing)]
    if origin is None:
        ox, oy, oz = [float(v) for v in auto_origin(coords, (nx, ny, nz),
                                                    (sx, sy, sz))]
    else:
        ox, oy, oz = [float(v) for v in np.asarray(origin, dtype=float).ravel()]

    out = [[[0.0] * nz for _ in range(ny)] for _ in range(nx)]
    for i in range(nx):
        wx = ox + i * sx
        for j in range(ny):
            wy = oy + j * sy
            for k in range(nz):
                wz = oz + k * sz
                in_core = False
                in_shell = False
                for (cx, cy, cz), r in zip(coords, rad):
                    dist = math.sqrt((wx - cx) ** 2 + (wy - cy) ** 2 + (wz - cz) ** 2)
                    if dist <= r:
                        in_core = True
                        break
                    if dist <= r + probe:
                        in_shell = True
                if in_core:
                    out[i][j][k] = float(core_value)
                elif in_shell:
                    out[i][j][k] = float(surface_value)
    return np.array(out, dtype=float)


def receptor_grid(coords, radii, grid_shape, spacing: float = 1.0, origin=None,
                  surface_value: float = 1.0, core_value: float = -15.0,
                  probe: float = 1.4) -> np.ndarray:
    """Docking receptor preset: favourable surface shell, punishing negative core."""
    if core_value >= 0:
        raise ValueError("a receptor core_value must be negative to punish overlap; "
                         f"got {core_value}")
    return voxelize(coords, radii, grid_shape, spacing, origin,
                    surface_value, core_value, probe)


def ligand_grid(coords, radii, grid_shape, spacing: float = 1.0, origin=None,
                occupancy: float = 1.0, probe: float = 0.0) -> np.ndarray:
    """Docking ligand preset: plain occupancy, 1 inside the atoms and 0 outside.

    The asymmetry is the whole point of the Katchalski-Katzir score: the ligand's own
    VOLUME is correlated against the receptor's signed grid, so ligand volume landing in
    the receptor's shell scores +, and ligand volume landing in the receptor's core
    scores a large -.
    """
    return voxelize(coords, radii, grid_shape, spacing, origin,
                    surface_value=0.0, core_value=occupancy, probe=probe)


# --------------------------------------------------------------------------------------
# correlation
# --------------------------------------------------------------------------------------

def _pad_to(a: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    if a.shape == shape:
        return a
    out = np.zeros(shape, dtype=float)
    out[:a.shape[0], :a.shape[1], :a.shape[2]] = a
    return out


def _corr_shape(R: np.ndarray, L: np.ndarray, mode: str) -> Tuple[int, int, int]:
    if R.ndim != 3 or L.ndim != 3:
        raise ValueError(f"grids must be 3-D, got {R.ndim}-D and {L.ndim}-D")
    if mode == "circular":
        if R.shape != L.shape:
            raise ValueError("circular mode needs identical shapes, got "
                             f"{R.shape} and {L.shape}; use mode='linear' or pad first")
        return R.shape  # type: ignore[return-value]
    if mode == "linear":
        return tuple(a + b - 1 for a, b in zip(R.shape, L.shape))  # type: ignore
    raise ValueError(f"mode must be 'circular' or 'linear', got {mode!r}")


def correlate(receptor_grid: np.ndarray, ligand_grid: np.ndarray,
              mode: str = "circular", sign: str = "plus") -> np.ndarray:
    """Score EVERY translation of the ligand over the receptor with one FFT pair.

    Returns S with, for sign="plus" (the default and the correct convention),

        S[t] = sum_x  receptor[x] * ligand[(x - t) mod n]

    so S[t] is the score of the ligand translated by +t voxels and the peak index IS
    the shift. sign="minus" returns sum_x R[x] L[(x + t) mod n], whose peak is at -t;
    it exists only so the tests can exhibit the trap.

    mode="circular"  both grids must share a shape; the search is periodic (the classic
                     docking setup, where you pad the box yourself).
    mode="linear"    zero-pads both to R.shape + L.shape - 1 so nothing wraps; the
                     signed index range of the result is then exactly the range of
                     translations with any overlap at all.
    """
    R = np.asarray(receptor_grid, dtype=float)
    L = np.asarray(ligand_grid, dtype=float)
    shape = _corr_shape(R, L, mode)
    axes = (0, 1, 2)
    FR = np.fft.rfftn(R, s=shape, axes=axes)
    FL = np.fft.rfftn(L, s=shape, axes=axes)
    if sign == "plus":
        prod = FR * np.conj(FL)
    elif sign == "minus":
        prod = np.conj(FR) * FL
    else:
        raise ValueError(f"sign must be 'plus' or 'minus', got {sign!r}")
    return np.fft.irfftn(prod, s=shape, axes=axes)


def correlate_direct(receptor_grid: np.ndarray, ligand_grid: np.ndarray,
                     mode: str = "circular", sign: str = "plus") -> np.ndarray:
    """INDEPENDENT reference for correlate(): explicit nested loops, no FFT. O(N^6).

    Six nested Python loops over (translation, voxel) on plain Python lists. It shares
    no code path with correlate() -- no numpy FFT, no numpy arithmetic in the inner
    loop, not even numpy indexing. This is the ground truth in verify() case (a).
    """
    R = np.asarray(receptor_grid, dtype=float)
    L = np.asarray(ligand_grid, dtype=float)
    shape = _corr_shape(R, L, mode)
    n0, n1, n2 = shape
    Rl = _pad_to(R, shape).tolist()
    Ll = _pad_to(L, shape).tolist()
    if sign not in ("plus", "minus"):
        raise ValueError(f"sign must be 'plus' or 'minus', got {sign!r}")
    s_ = 1 if sign == "plus" else -1

    out = [[[0.0] * n2 for _ in range(n1)] for _ in range(n0)]
    for t0 in range(n0):
        for t1 in range(n1):
            for t2 in range(n2):
                acc = 0.0
                for x0 in range(n0):
                    y0 = (x0 - s_ * t0) % n0
                    for x1 in range(n1):
                        y1 = (x1 - s_ * t1) % n1
                        Rrow = Rl[x0][x1]
                        Lrow = Ll[y0][y1]
                        for x2 in range(n2):
                            acc += Rrow[x2] * Lrow[(x2 - s_ * t2) % n2]
                out[t0][t1][t2] = acc
    return np.array(out, dtype=float)


def correlate_direct_blocked(receptor_grid: np.ndarray, ligand_grid: np.ndarray,
                             mode: str = "circular") -> np.ndarray:
    """Same O(N^6) direct search, one numpy inner product per translation. No FFT.

    Identical arithmetic to correlate_direct() with a ~100x smaller constant, which is
    what makes the timing table in benchmark() reach useful grid sizes. Used for the
    SPEED comparison; correlate_direct() is used for the CORRECTNESS comparison.
    """
    R = np.asarray(receptor_grid, dtype=float)
    L = np.asarray(ligand_grid, dtype=float)
    shape = _corr_shape(R, L, mode)
    Rp = _pad_to(R, shape)
    Lp = _pad_to(L, shape)
    out = np.empty(shape, dtype=float)
    n0, n1, n2 = shape
    for t0 in range(n0):
        A = np.roll(Lp, t0, axis=0)
        for t1 in range(n1):
            B = np.roll(A, t1, axis=1)
            for t2 in range(n2):
                out[t0, t1, t2] = float(np.sum(Rp * np.roll(B, t2, axis=2)))
    return out


# --------------------------------------------------------------------------------------
# reading the score volume
# --------------------------------------------------------------------------------------

def _fold(raw: np.ndarray, shape, ligand_shape=None) -> np.ndarray:
    """Raw periodic index -> signed shift.

    ligand_shape=None: numpy's fftfreq ordering, offset n // 2, range
        [-(n//2), n - n//2 - 1]. This is the right fold for a circular search.
    ligand_shape=m:    offset m - 1, range [-(m-1), out - m], which for a LINEAR
        (zero-padded) search of a shape-n receptor with a shape-m ligand is exactly
        [-(m-1), n-1] -- every translation with any overlap, and no other. For equal
        shapes the two folds coincide, so the default is safe whenever the shapes match.
    """
    n = np.asarray(shape, dtype=int)
    off = n // 2 if ligand_shape is None else \
        np.asarray(_shape3(ligand_shape), dtype=int) - 1
    return ((raw + off) % n) - off


def best_translation(score_volume: np.ndarray, signed: bool = True,
                     ligand_shape=None) -> Tuple[np.ndarray, float]:
    """(shift_vector, score) of the highest-scoring translation.

    signed=True (default) folds the raw periodic index into numpy's fftfreq ordering,
    range [-(n//2), n - n//2 - 1] per axis: raw index 13 on a length-16 axis is -3.
    signed=False returns the raw non-negative index.

    ligand_shape must be supplied for a mode="linear" volume built from grids of
    DIFFERENT shapes -- there the valid translation window is [-(m-1), n-1], which is
    not centred, and the default fold would report the wrong sign at the extremes.
    When the two grids have the same shape both folds agree and it can be omitted.
    """
    S = np.asarray(score_volume)
    if S.ndim != 3:
        raise ValueError(f"score_volume must be 3-D, got {S.ndim}-D")
    raw = np.array(np.unravel_index(int(np.argmax(S)), S.shape), dtype=int)
    score = float(S[tuple(raw)])
    if not signed:
        return raw, score
    return _fold(raw, S.shape, ligand_shape), score


def top_translations(score_volume: np.ndarray, k: int = 5, signed: bool = True,
                     ligand_shape=None) -> List[Tuple[np.ndarray, float]]:
    """The k best translations, highest score first."""
    S = np.asarray(score_volume)
    k = int(min(k, S.size))
    flat = np.argpartition(S.ravel(), S.size - k)[S.size - k:]
    flat = flat[np.argsort(-S.ravel()[flat])]
    out = []
    for f in flat:
        raw = np.array(np.unravel_index(int(f), S.shape), dtype=int)
        shift = _fold(raw, S.shape, ligand_shape) if signed else raw
        out.append((shift, float(S[tuple(raw)])))
    return out


def shift_to_world(shift, spacing: float = 1.0) -> np.ndarray:
    """Voxel shift -> world-unit translation vector."""
    return np.asarray(shift, dtype=float) * _spacing3(spacing)


def cost_model(grid_shape) -> dict:
    """The governing law for a translation search, as counted operations.

    Translation is the one degree of freedom that factorizes exactly: the score
    operator is circulant, hence diagonal in Fourier, hence bond dimension 1 and
    effective treewidth 0 across any cut between Fourier modes.
    """
    n0, n1, n2 = _shape3(grid_shape)
    N = n0 * n1 * n2
    return {
        "grid_shape": (n0, n1, n2),
        "n_voxels": N,
        "n_translations": N,
        "direct_ops": N * N,
        "fft_ops": int(3 * N * math.log2(max(N, 2))),
        "predicted_speedup": N * N / max(3 * N * math.log2(max(N, 2)), 1.0),
        "effective_treewidth": 0,
        "bond_dimension_fourier": 1,
    }


# --------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------

def _random_grid(rng, shape, fill=0.35) -> np.ndarray:
    g = np.zeros(shape, dtype=float)
    mask = rng.random(shape) < fill
    g[mask] = rng.normal(size=int(mask.sum()))
    return g


def verify_fourier_diagonal(shape=(3, 3, 4), seed: int = 0, verbose: bool = True) -> dict:
    """MEASURE the claim that the translation operator is diagonal in Fourier.

    Builds the full N x N translation-score operator M[t, x] = L[(x - t) mod n], then
    conjugates it by the 3-D DFT matrix and reports max|off-diag| / max|diag|. If that
    is ~1e-16 the modes genuinely do not couple: bond dimension 1, treewidth 0.
    """
    rng = np.random.default_rng(seed)
    n0, n1, n2 = _shape3(shape)
    N = n0 * n1 * n2
    L = rng.normal(size=(n0, n1, n2))

    idx = np.array(list(np.ndindex(n0, n1, n2)), dtype=int)          # flat -> (x0,x1,x2)
    M = np.empty((N, N), dtype=float)
    for a, t in enumerate(idx):
        for b, x in enumerate(idx):
            y = (x - t) % np.array([n0, n1, n2])
            M[a, b] = L[y[0], y[1], y[2]]

    def dft(n):
        j = np.arange(n)
        return np.exp(-2j * np.pi * np.outer(j, j) / n)
    F = np.kron(np.kron(dft(n0), dft(n1)), dft(n2))
    D = F @ M @ (np.conj(F).T / N)

    diag = np.diag(D)
    off = D - np.diag(diag)
    ratio = float(np.max(np.abs(off)) / np.max(np.abs(diag)))
    # the diagonal must be the conjugate spectrum of L
    err_eig = float(np.max(np.abs(diag - np.conj(np.fft.fftn(L)).ravel())))
    if verbose:
        print(f"    (d) Fourier diagonality on {n0}x{n1}x{n2} (operator {N}x{N})")
        print(f"          max|off-diagonal| / max|diagonal|   {ratio:.3e}")
        print(f"          max|diagonal - conj(fftn(L))|       {err_eig:.3e}")
        print(f"          -> bond dimension across a mode cut 1, effective treewidth 0")
    return {"offdiag_ratio": ratio, "eigenvalue_err": err_eig, "N": N,
            "bond_dimension": 1, "effective_treewidth": 0}


def verify_sign_convention(verbose: bool = True) -> dict:
    """Plant a known translation, recover it, demand EXACT integer equality.

    The ligand is an asymmetric blob; the receptor is that blob rolled by the planted
    shift. The global peak of the autocorrelation is then unambiguously at the shift.
    Shifts include negative components and components that wrap the periodic boundary.
    Every recovery is checked against BOTH the FFT correlation and the independent
    O(N^6) blocked direct correlation.
    """
    rng = np.random.default_rng(7)
    shape = (12, 10, 8)
    L = np.zeros(shape)
    L[1:4, 1:3, 1:5] = rng.random((3, 2, 4)) + 0.5      # asymmetric, strictly positive
    shifts = [(0, 0, 0), (2, -3, 1), (-4, 4, -2), (5, 0, 3), (-1, -1, -1),
              (5, -5, 3), (-5, 4, -3), (11, 9, 7), (-6, 5, 4)]
    rows = []
    all_ok = True
    for s in shifts:
        R = np.roll(L, shift=s, axis=(0, 1, 2))          # R[x] = L[x - s]
        S = correlate(R, L)
        got, score = best_translation(S)
        raw, _ = best_translation(S, signed=False)
        want = np.array([((si + n // 2) % n) - n // 2
                         for si, n in zip(s, shape)], dtype=int)
        ok = bool(np.array_equal(got, want))
        # the same peak from the independent direct correlation
        Sd = correlate_direct_blocked(R, L)
        gotd, _ = best_translation(Sd)
        ok_d = bool(np.array_equal(gotd, want))
        all_ok = all_ok and ok and ok_d
        # and the trap: sign="minus" must peak at the NEGATED shift
        Sm = correlate(R, L, sign="minus")
        gotm, _ = best_translation(Sm)
        want_m = np.array([((-si + n // 2) % n) - n // 2
                           for si, n in zip(s, shape)], dtype=int)
        ok_m = bool(np.array_equal(gotm, want_m))
        all_ok = all_ok and ok_m
        rows.append({"planted": tuple(s), "folded": tuple(int(v) for v in want),
                     "raw_index": tuple(int(v) for v in raw),
                     "recovered_fft": tuple(int(v) for v in got),
                     "recovered_direct": tuple(int(v) for v in gotd),
                     "recovered_minus": tuple(int(v) for v in gotm),
                     "score": score, "exact": ok and ok_d and ok_m})
    if verbose:
        print(f"    (b) planted-translation recovery on grid {shape[0]}x{shape[1]}x{shape[2]}"
              f"  (signed range {[-(n // 2) for n in shape]} .. "
              f"{[n - n // 2 - 1 for n in shape]})")
        print(f"          {'planted':>14} {'folded':>14} {'raw idx':>14} "
              f"{'FFT':>14} {'direct':>14} {'sign=minus':>14}  exact")
        for r in rows:
            print(f"          {str(r['planted']):>14} {str(r['folded']):>14} "
                  f"{str(r['raw_index']):>14} {str(r['recovered_fft']):>14} "
                  f"{str(r['recovered_direct']):>14} {str(r['recovered_minus']):>14}"
                  f"  {'YES' if r['exact'] else 'NO'}")
        print(f"          all {len(rows)} shifts recovered EXACTLY (integer equality): "
              f"{'PASS' if all_ok else 'FAIL'}")
    return {"rows": rows, "all_exact": all_ok, "shape": shape}


def _pocket_case():
    """A receptor slab with a 3x3x3 cavity, and the ligand cube that fills it."""
    shape = (20, 20, 20)
    solid = np.zeros(shape, dtype=bool)
    solid[3:17, 3:17, 3:11] = True
    cavity = (slice(7, 10), slice(6, 9), slice(6, 9))
    solid[cavity] = False
    # shell = empty voxels 6-adjacent to solid
    nb = np.zeros(shape, dtype=bool)
    for ax in range(3):
        for d in (1, -1):
            nb |= np.roll(solid, d, axis=ax)
    shell = nb & ~solid
    R = np.zeros(shape, dtype=float)
    R[shell] = 1.0
    R[solid] = -15.0
    L = np.zeros(shape, dtype=float)
    L[0:3, 0:3, 0:3] = 1.0        # ligand at the grid origin corner
    planted = np.array([cavity[0].start, cavity[1].start, cavity[2].start], dtype=int)
    return R, L, planted


def verify_pocket(verbose: bool = True) -> dict:
    """Positive control on a docking-shaped problem, not just an autocorrelation.

    A solid receptor with a 3x3x3 cavity; a 3x3x3 ligand cube. The cavity placement
    touches 26 shell voxels with zero core overlap; every other placement is worse.
    The recovered translation must equal the cavity's corner exactly.
    """
    R, L, planted = _pocket_case()
    S = correlate(R, L)
    got, score = best_translation(S)
    ok = bool(np.array_equal(got, planted))
    runner = top_translations(S, 3)
    if verbose:
        print(f"    (e) pocket positive control: 20^3 receptor slab with a 3x3x3 cavity")
        print(f"          planted cavity corner {tuple(int(v) for v in planted)}   "
              f"recovered {tuple(int(v) for v in got)}   score {score:+.1f}   "
              f"{'PASS' if ok else 'FAIL'}")
        print(f"          runners-up: " + ", ".join(
            f"{tuple(int(v) for v in s)}={sc:+.1f}" for s, sc in runner[1:]))
    return {"planted": tuple(int(v) for v in planted),
            "recovered": tuple(int(v) for v in got), "score": score, "exact": ok,
            "runner_up": float(runner[1][1]) if len(runner) > 1 else float("nan")}


def _exponent(t0, n0, t1, n1) -> float:
    if t0 <= 0 or t1 <= 0 or n0 == n1:
        return float("nan")
    return math.log(t1 / t0) / math.log(n1 / n0)


def _loglog_slope(ns, ts) -> float:
    """Least-squares d log t / d log n. The single robust growth-exponent number."""
    x = np.log(np.asarray(ns, dtype=float))
    y = np.log(np.asarray(ts, dtype=float))
    if len(x) < 2:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def benchmark(pure_sizes: Sequence[int] = (4, 6, 8, 10, 12, 14),
              direct_sizes: Sequence[int] = (8, 12, 16, 20, 24, 32),
              fft_sizes: Sequence[int] = (32, 48, 64, 96, 128, 160),
              seed: int = 0, verbose: bool = True) -> dict:
    """MEASURE O(N^3 log N) against O(N^6). Every number below is wall clock.

    Three tables, because one table cannot show both things honestly:

      A  pure nested-loop direct vs FFT, small grids. The pure-Python loop has a STABLE
         per-operation constant, so its measured growth exponent is the clean test of
         O(N^6) -- it should converge on 6.
      B  numpy-blocked direct vs FFT, realistic grids. Same O(N^6) arithmetic with a
         ~100x smaller constant, so it reaches n = 32 and gives an honest speedup
         number. Its measured exponent UNDERSHOOTS 6 below n ~ 32 because per-call
         numpy overhead is still amortizing; that is stated, not hidden.
      C  FFT alone at grid sizes where the direct search is out of reach, with its own
         measured exponent, which should sit near 3.
    """
    rng = np.random.default_rng(seed)

    def timed(fn, n):
        shape = (n, n, n)
        R = _random_grid(rng, shape)
        L = _random_grid(rng, shape)
        t0 = time.perf_counter()
        A = fn(R, L)
        return time.perf_counter() - t0, A, R, L

    pure = []
    for n in pure_sizes:
        t_d, Sd, R, L = timed(correlate_direct, n)
        t0 = time.perf_counter()
        Sf = correlate(R, L)
        t_f = time.perf_counter() - t0
        pure.append({"n": n, "t_pure_direct": t_d, "t_fft": t_f,
                     "ops": n ** 6, "max_err": float(np.max(np.abs(Sf - Sd)))})
    for a, b in zip(pure, pure[1:]):
        b["exponent"] = _exponent(a["t_pure_direct"], a["n"], b["t_pure_direct"], b["n"])

    blocked = []
    for n in direct_sizes:
        t_d, Sd, R, L = timed(correlate_direct_blocked, n)
        correlate(R, L)                                   # warm
        t0 = time.perf_counter()
        Sf = correlate(R, L)
        t_f = time.perf_counter() - t0
        blocked.append({"n": n, "voxels": n ** 3, "ops": n ** 6,
                        "t_direct": t_d, "t_fft": t_f, "speedup": t_d / max(t_f, 1e-12),
                        "ops_per_s": n ** 6 / max(t_d, 1e-12),
                        "max_err": float(np.max(np.abs(Sf - Sd)))})
    for a, b in zip(blocked, blocked[1:]):
        b["exponent"] = _exponent(a["t_direct"], a["n"], b["t_direct"], b["n"])

    fft_only = []
    for n in fft_sizes:
        shape = (n, n, n)
        R = _random_grid(rng, shape)
        L = _random_grid(rng, shape)
        correlate(R, L)                                   # warm
        t0 = time.perf_counter()
        correlate(R, L)
        fft_only.append({"n": n, "voxels": n ** 3, "ops": n ** 6,
                         "t_fft": time.perf_counter() - t0})
    for a, b in zip(fft_only, fft_only[1:]):
        b["exponent"] = _exponent(a["t_fft"], a["n"], b["t_fft"], b["n"])

    if verbose:
        print("    (c) SPEED, measured. FFT O(N^3 log N) vs direct O(N^6)")
        print("        A  pure nested-loop direct (stable constant -> clean exponent)")
        print(f"          {'grid':>7} {'N^6 ops':>13} {'direct (s)':>11} {'FFT (s)':>9} "
              f"{'speedup':>9} {'exponent':>9} {'max err':>10}")
        for r in pure:
            ex = r.get("exponent", float("nan"))
            print(f"          {r['n']:>4}^3 {r['ops']:>13,} {r['t_pure_direct']:>11.4f} "
                  f"{r['t_fft']:>9.5f} {r['t_pure_direct']/max(r['t_fft'],1e-12):>8.0f}x "
                  f"{ex:>9.2f} {r['max_err']:>10.2e}")
        print("        B  numpy-blocked direct, same O(N^6) arithmetic, ~100x smaller "
              "constant")
        print(f"          {'grid':>7} {'N^6 ops':>15} {'direct (s)':>11} {'FFT (s)':>9} "
              f"{'speedup':>9} {'direct ops/s':>13} {'max err':>10}")
        for r in blocked:
            print(f"          {r['n']:>4}^3 {r['ops']:>15,} {r['t_direct']:>11.4f} "
                  f"{r['t_fft']:>9.5f} {r['speedup']:>8.0f}x {r['ops_per_s']:>13.2e} "
                  f"{r['max_err']:>10.2e}")
        print("          (its per-op constant is NOT stable -- ops/s climbs several-fold "
              "across the table as")
        print("           numpy per-call overhead amortizes, and non-power-of-two "
              "strides cost extra -- so its")
        print("           wall-clock exponent is not a valid scaling measurement. "
              "Table A is. Table B is the")
        print("           honest SPEEDUP at realistic grid sizes.)")
        print("        C  FFT alone, where the direct search is out of reach")
        print(f"          {'grid':>7} {'voxels':>11} {'N^6 ops':>20} {'FFT (s)':>9} "
              f"{'exponent':>9}")
        for r in fft_only:
            ex = r.get("exponent", float("nan"))
            print(f"          {r['n']:>4}^3 {r['voxels']:>11,} {r['ops']:>20,} "
                  f"{r['t_fft']:>9.4f} {ex:>9.2f}")
        sl_p = _loglog_slope([r["n"] for r in pure], [r["t_pure_direct"] for r in pure])
        sl_f = _loglog_slope([r["n"] for r in fft_only], [r["t_fft"] for r in fft_only])
        pe = [r["exponent"] for r in pure[1:]]
        print("        measured growth exponent  d log t / d log n:")
        print(f"          A pure direct  least-squares slope {sl_p:.2f}, "
              f"largest step {pe[-1]:.2f}   -> O(N^6) predicts 6.00")
        print(f"          C FFT          least-squares slope {sl_f:.2f}"
              f"                        -> O(N^3 log N) predicts just over 3.00")
        print(f"          B is excluded from this comparison for the reason printed "
              f"above it.")
    return {"pure": pure, "blocked": blocked, "fft_only": fft_only, "rows": blocked,
            "slope_pure_direct": _loglog_slope([r["n"] for r in pure],
                                               [r["t_pure_direct"] for r in pure]),
            "slope_fft": _loglog_slope([r["n"] for r in fft_only],
                                       [r["t_fft"] for r in fft_only])}


def verify_linear_mode(verbose: bool = True) -> dict:
    """mode='linear' with DIFFERENT receptor and ligand shapes, negative shift included.

    Zero-padding to R.shape + L.shape - 1 removes wraparound entirely, and the fold
    that knows the ligand shape must return the planted translation exactly, including
    when it is negative (ligand larger than receptor, pattern planted further along).
    """
    rng = np.random.default_rng(11)
    P = rng.random((2, 2, 2)) + 0.5
    rows = []
    ok_all = True
    for (rshape, lshape, p, q) in [((4, 4, 4), (6, 6, 6), (0, 0, 0), (3, 2, 1)),
                                   ((6, 5, 4), (4, 4, 3), (3, 2, 1), (0, 0, 0)),
                                   ((5, 5, 5), (5, 5, 5), (2, 0, 3), (0, 3, 1)),
                                   ((7, 6, 5), (3, 3, 3), (4, 3, 2), (1, 1, 0))]:
        R = np.zeros(rshape)
        L = np.zeros(lshape)
        R[p[0]:p[0] + 2, p[1]:p[1] + 2, p[2]:p[2] + 2] = P
        L[q[0]:q[0] + 2, q[1]:q[1] + 2, q[2]:q[2] + 2] = P
        want = np.array(p, dtype=int) - np.array(q, dtype=int)
        S = correlate(R, L, mode="linear")
        got, score = best_translation(S, ligand_shape=lshape)
        ok = bool(np.array_equal(got, want))
        ok_all = ok_all and ok
        rows.append({"receptor": rshape, "ligand": lshape, "out": S.shape,
                     "planted": tuple(int(v) for v in want),
                     "recovered": tuple(int(v) for v in got),
                     "score": score, "exact": ok})
    # exhaustive index-arithmetic check of the linear fold
    rshape, lshape = (7, 6, 5), (3, 4, 2)
    out = tuple(a + b - 1 for a, b in zip(rshape, lshape))
    bad = 0
    for raw in np.ndindex(*out):
        f = _fold(np.array(raw, dtype=int), out, lshape)
        if not np.all(f % np.array(out) == np.array(raw)):
            bad += 1
        if not np.all(f >= -(np.array(lshape) - 1)) or \
           not np.all(f <= np.array(rshape) - 1):
            bad += 1
    ok_all = ok_all and bad == 0
    if verbose:
        print("    (g) mode='linear' with mismatched shapes (no wraparound at all)")
        print(f"          {'receptor':>12} {'ligand':>12} {'output':>12} "
              f"{'planted':>13} {'recovered':>13}  exact")
        for r in rows:
            print(f"          {str(r['receptor']):>12} {str(r['ligand']):>12} "
                  f"{str(r['out']):>12} {str(r['planted']):>13} "
                  f"{str(r['recovered']):>13}  {'YES' if r['exact'] else 'NO'}")
        print(f"          exhaustive fold check over all {int(np.prod(out))} indices of "
              f"a {out} volume: {bad} violations")
    return {"rows": rows, "fold_violations": bad, "all_exact": ok_all}


def verify(verbose: bool = True, fast: bool = False) -> dict:
    """Check everything against genuinely independent references and print max errors.

    fast=True shrinks only the timing tables (they are the slow part); every
    correctness check runs identically.
    """
    if verbose:
        print("  rem.fftcorr.verify")

    rng = np.random.default_rng(3)

    # ---- (a) FFT correlation vs explicit nested-loop direct correlation -------------
    a_rows = []
    e_a = 0.0
    cases = [((4, 4, 4), "circular", "plus"), ((5, 4, 3), "circular", "plus"),
             ((5, 5, 5), "circular", "plus"), ((4, 5, 3), "circular", "minus"),
             ((4, 3, 3), "linear", "plus")]
    for shape, mode, sign in cases:
        R = _random_grid(rng, shape, fill=0.5)
        if mode == "linear":
            L = _random_grid(rng, (3, 2, 3), fill=0.6)
        else:
            L = _random_grid(rng, shape, fill=0.5)
        Sf = correlate(R, L, mode=mode, sign=sign)
        Sd = correlate_direct(R, L, mode=mode, sign=sign)
        err = float(np.max(np.abs(Sf - Sd)))
        e_a = max(e_a, err)
        a_rows.append({"shape": shape, "lig_shape": L.shape, "mode": mode,
                       "sign": sign, "out_shape": Sf.shape, "max_err": err})
    # blocked direct must agree with pure-loop direct too (guards the timing reference)
    e_blocked = 0.0
    for shape in [(4, 4, 4), (5, 4, 3)]:
        R = _random_grid(rng, shape, fill=0.5)
        L = _random_grid(rng, shape, fill=0.5)
        e_blocked = max(e_blocked, float(np.max(np.abs(
            correlate_direct(R, L) - correlate_direct_blocked(R, L)))))
    if verbose:
        print("    (a) FFT correlation vs DIRECT nested-loop correlation "
              "(pure Python, no FFT)")
        print(f"          {'receptor':>12} {'ligand':>10} {'mode':>9} {'sign':>7} "
              f"{'output':>12} {'max err':>10}")
        for r in a_rows:
            print(f"          {str(r['shape']):>12} {str(r['lig_shape']):>10} "
                  f"{r['mode']:>9} {r['sign']:>7} {str(r['out_shape']):>12} "
                  f"{r['max_err']:>10.2e}")
        print(f"          max over all cases                    {e_a:.3e}"
              f"   (tolerance 1e-10)")
        print(f"          blocked direct vs nested-loop direct  {e_blocked:.3e}")

    # ---- (b) planted-translation recovery -------------------------------------------
    b = verify_sign_convention(verbose=verbose)

    # ---- (c) speed ------------------------------------------------------------------
    if fast:
        c = benchmark(pure_sizes=(4, 6, 8), direct_sizes=(8, 12, 16),
                      fft_sizes=(32, 48, 64), verbose=verbose)
    else:
        c = benchmark(verbose=verbose)

    # ---- (d) Fourier diagonality ----------------------------------------------------
    d = verify_fourier_diagonal(verbose=verbose)

    # ---- (e) pocket positive control ------------------------------------------------
    e = verify_pocket(verbose=verbose)

    # ---- (f) voxelize vs pure-Python brute force ------------------------------------
    n_bad, n_vox = 0, 0
    e_vox = 0.0
    for t in range(5):
        rg = np.random.default_rng(100 + t)
        natoms = int(rg.integers(2, 7))
        coords = rg.normal(scale=2.5, size=(natoms, 3)) + np.pi / 7.0
        radii = rg.uniform(1.2, 2.0, size=natoms)
        shape = (11, 12, 10)
        g1 = voxelize(coords, radii, shape, spacing=0.9, surface_value=1.0,
                      core_value=-15.0, probe=1.4)
        g2 = voxelize_bruteforce(coords, radii, shape, spacing=0.9, surface_value=1.0,
                                 core_value=-15.0, probe=1.4)
        n_bad += int(np.count_nonzero(g1 != g2))
        n_vox += g1.size
        e_vox = max(e_vox, float(np.max(np.abs(g1 - g2))))
    if verbose:
        print("    (f) voxelize vs pure-Python global-sweep brute force")
        print(f"          mismatched voxels {n_bad} / {n_vox}      "
              f"max |difference| {e_vox:.3e}")

    # ---- (g) linear mode, mismatched shapes -----------------------------------------
    g = verify_linear_mode(verbose=verbose)

    ok = (e_a < 1e-10 and e_blocked < 1e-10 and b["all_exact"] and e["exact"]
          and n_bad == 0 and d["offdiag_ratio"] < 1e-12 and g["all_exact"]
          and max(r["max_err"] for r in c["blocked"]) < 1e-8
          and max(r["max_err"] for r in c["pure"]) < 1e-10)
    if verbose:
        print(f"    LAW: translation search is circulant -> diagonal in Fourier -> "
              f"bond dimension 1, treewidth 0")
        print(f"         cost O(N^3 log N) not d^tw with tw>0; only TRANSLATION "
              f"factorizes, rotations do not")
        print(f"    OVERALL {'PASS' if ok else 'FAIL'}")
    return {"max_err_fft_vs_direct": e_a,
            "max_err_blocked_vs_direct": e_blocked,
            "correlation_cases": a_rows,
            "planted": b, "timing": c, "fourier": d, "pocket": e, "linear": g,
            "voxelize_mismatched_voxels": n_bad, "voxelize_n_voxels": n_vox,
            "max_err_voxelize": e_vox,
            "effective_treewidth": 0, "bond_dimension_fourier": 1,
            "ok": bool(ok)}


if __name__ == "__main__":
    verify()
