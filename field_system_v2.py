#!/usr/bin/env python3
"""
Field-Mediated Neural System — Stage 2a: periodic / FFT / spatially-sorted
==========================================================================

Same model as Stage 1 (units couple ONLY through a shared screened-diffusion
field), but re-engineered toward a GPU-ready, large-N implementation:

  1. BOUNDARIES  — periodic (torus) instead of Neumann. This is what lets us
                   solve the field with an FFT.
  2. FIELD SOLVE — spectral, not explicit-Euler time stepping. The screened
                   steady field obeys  (λ − D∇²) φ = S, which is algebraic in
                   Fourier space:
                       φ_hat = S_hat / (λ + D·|k|²)
                       φ     = IFFT( FFT(S) / (λ + D·|k|²) )
                   One FFT + one IFFT per step → O(M log M), EXACT steady field,
                   no CFL / time-step stability limit. (Default mode='steady'.)
                   An unconditionally-stable spectral exponential integrator is
                   provided as mode='dynamic':
                       φ_hat ← φ_hat·e^{−αΔt} + S_hat·(1−e^{−αΔt})/α,  α=λ+D|k|²
  3. UNIT ORDER  — units are sorted once by the Morton (Z-order) code of their
                   grid cell, so neighbours in memory are neighbours in space
                   (a memory-locality win that matters on GPU). Sorting is a pure
                   permutation; an unsorted mode exists to prove identical output.
  4. DEPOSIT/READ— same symmetric Gaussian kernel as Stage 1: deposit = scatter-
                   add of a_i·G, read = h²·(deposit)ᵀ = gather of G·φ.

THE k=0 ZERO MODE (the one real subtlety):
  The steady operator (λ + D|k|²) vanishes at k=0 iff λ=0 — pure diffusion has no
  steady state when there is net source (the spatial mean ∫Φ would grow forever).
  • Steady mode: we set the k=0 component of 1/(λ+D|k|²) to 0 when λ=0, i.e. we
    keep the zero-MEAN steady shape and drop the undetermined DC offset.
  • Dynamic mode: the update factor (1−e^{−αΔt})/α has the finite limit Δt as
    α→0, so the DC mode is integrated as φ_hat[0,0] += S_hat[0,0]·Δt — exactly the
    linear ∫Φ growth of an undamped, conserved diffusion field (conservation test).

ENVIRONMENT: CPU only, ≤10 GB. float32 everywhere on the big arrays; a memory
estimate is printed and the run is refused if it would exceed 8 GB.

GPU PORT: every array op goes through the module-level alias `xp` (here = numpy).
Swapping `import numpy as xp` → `import cupy as xp` (plus xp.fft) is essentially
the whole Stage-2b port. See the "GPU PORT NOTES" section of the README.

Run:   python field_system_v2.py
Deps:  numpy, matplotlib   (only)
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# BACKEND ALIAS — the single thing Stage 2b swaps.  `import cupy as xp` + the
# xp.fft calls below are the entire array-backend change.  Everything numerically
# heavy (deposit scatter, read gather, FFT solve, propagator) is written in xp.
# ---------------------------------------------------------------------------
import numpy as xp                      # <-- Stage 2b: replace with `import cupy as xp`

F32 = xp.float32
C64 = xp.complex64
I32 = xp.int32


def to_numpy(a):
    """Bring an xp array to host numpy (for matplotlib / asserts).
    numpy: a no-op view; cupy: use xp.asnumpy(a)."""
    return np.asarray(a)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ConfigV2:
    H: int = 256
    W: int = 256
    h: float = 1.0
    sigma: float = 1.5
    D: float = 0.2
    lam: float = 0.05                   # λ  (field decay / screening)
    gamma: float = 0.1                  # γ  (unit-state leak)
    dt: float = 1.0                     # Δt (used by dynamic mode)
    N: int = 100_000
    seed: int = 0
    kernel_radius_sigmas: float = 4.0
    mode: str = "steady"                # 'steady' (default) | 'dynamic'
    symbol: str = "spectral"            # 'spectral' (continuous k²) | 'discrete' (5-pt)
    sort_units: bool = True             # Morton Z-order sort
    mem_limit_bytes: float = 8.0e9      # refuse to run above this (headroom under 10 GB)
    # hard caps from the spec
    N_CAP: int = 2_000_000
    GRID_CAP: int = 512

    @property
    def influence_range(self) -> float:
        return float(np.sqrt(self.D / self.lam))

    @property
    def kernel_width(self) -> int:
        rad = int(np.ceil(self.kernel_radius_sigmas * self.sigma / self.h))
        return 2 * rad + 1


# ---------------------------------------------------------------------------
# Morton (Z-order) code
# ---------------------------------------------------------------------------
def morton2d(cx, cy):
    """Interleave the bits of cell coords (cx, cy) → Z-order code (uint32).
    Supports coords up to 16 bits (65535), well above the 512-cell grid cap.
    Vectorized over all units."""
    def part1by1(n):
        n = n.astype(np.uint32) & np.uint32(0x0000FFFF)
        n = (n | (n << np.uint32(8))) & np.uint32(0x00FF00FF)
        n = (n | (n << np.uint32(4))) & np.uint32(0x0F0F0F0F)
        n = (n | (n << np.uint32(2))) & np.uint32(0x33333333)
        n = (n | (n << np.uint32(1))) & np.uint32(0x55555555)
        return n
    return part1by1(cx) | (part1by1(cy) << np.uint32(1))


# ---------------------------------------------------------------------------
# Memory budget guard
# ---------------------------------------------------------------------------
def estimate_memory(cfg: ConfigV2):
    """Estimate peak bytes = units + grid(FFT scratch). Print a breakdown.
    Per-unit storage is dominated by the precomputed Gaussian windows
    (win_ker: K float32, win_idx: K int32) plus a transient K-float weights
    buffer built each deposit."""
    K = cfg.kernel_width ** 2
    # per unit: pos(2) + s,a,phi(3) + cell/morton(3) + win_ker(K) + win_idx(K)
    #           + transient deposit weights(K)
    per_unit_floats = 2 + 3 + 3 + 2 * K + K
    unit_bytes = cfg.N * per_unit_floats * 4
    M = cfg.H * cfg.W
    # grid: phi, S (2 real) + rfft halves S_hat, phi_hat (~2 complex64 = 4 real)
    #       + propagators inv_denom/E/factor (~3 real) + numpy's internal
    #       complex128 FFT temporary (~4 real). Constant ≈ 14.
    grid_bytes = M * 4 * 14
    total = unit_bytes + grid_bytes
    return total, unit_bytes, grid_bytes, per_unit_floats, K


def guard_memory(cfg: ConfigV2, verbose: bool = True):
    if cfg.N > cfg.N_CAP:
        raise SystemExit(f"REFUSING: N={cfg.N} exceeds cap {cfg.N_CAP}")
    if max(cfg.H, cfg.W) > cfg.GRID_CAP:
        raise SystemExit(f"REFUSING: grid {cfg.H}x{cfg.W} exceeds cap {cfg.GRID_CAP}")
    total, ub, gb, pf, K = estimate_memory(cfg)
    if verbose:
        print(f"    mem estimate: units {ub/1e9:6.3f} GB ({pf} floats/unit, K={K}) "
              f"+ grid {gb/1e9:6.3f} GB = {total/1e9:6.3f} GB  "
              f"(limit {cfg.mem_limit_bytes/1e9:.1f} GB)")
    if total > cfg.mem_limit_bytes:
        raise SystemExit(f"REFUSING: estimated {total/1e9:.2f} GB exceeds "
                         f"{cfg.mem_limit_bytes/1e9:.1f} GB budget. Reduce N or grid.")
    return total


# ---------------------------------------------------------------------------
# The Stage-2a field system
# ---------------------------------------------------------------------------
class FieldSystemV2:
    """Periodic / FFT / Morton-sorted field system. State = (phi, s, a). The four
    operations (deposit, evolve, read, update) and step() mirror Stage 1; only the
    boundary (torus) and the field solve (spectral) differ."""

    def __init__(self, cfg: ConfigV2, positions: np.ndarray | None = None,
                 check_mem: bool = True):
        self.cfg = cfg
        if check_mem:
            guard_memory(cfg, verbose=False)

        # --- field ----------------------------------------------------------
        self.phi = xp.zeros((cfg.H, cfg.W), dtype=F32)

        # --- positions (physical coords, x=col, y=row) ----------------------
        rng = np.random.default_rng(cfg.seed)
        if positions is None:
            pos = rng.uniform([0, 0], [cfg.W * cfg.h, cfg.H * cfg.h],
                              size=(cfg.N, 2))          # torus: anywhere is fine
        else:
            pos = np.asarray(positions, dtype=np.float64)
        pos = xp.asarray(pos, dtype=F32)
        self.N = int(pos.shape[0])

        # --- Morton (Z-order) sort: pure permutation of the unit list -------
        cx = xp.clip((pos[:, 0] / cfg.h).astype(I32), 0, cfg.W - 1)
        cy = xp.clip((pos[:, 1] / cfg.h).astype(I32), 0, cfg.H - 1)
        if cfg.sort_units:
            codes = morton2d(to_numpy(cx), to_numpy(cy))
            self.perm = xp.asarray(np.argsort(codes, kind="stable"))   # original→sorted
        else:
            self.perm = xp.arange(self.N)
        self.inv_perm = xp.empty(self.N, dtype=xp.int64)
        self.inv_perm[self.perm] = xp.arange(self.N)
        self.pos = pos[self.perm]                       # units now in Z-order

        # --- unit state -----------------------------------------------------
        self.s = xp.zeros(self.N, dtype=F32)
        self.a = xp.zeros(self.N, dtype=F32)
        self.clamped = xp.zeros(self.N, dtype=bool)

        # --- precompute Gaussian windows + the Fourier propagators ----------
        self._build_kernels()
        self._build_propagator()
        # one preallocated (N, K) work buffer, shared by deposit (a_i·G weights)
        # and read (gathered φ) — they run sequentially within a step, so reusing
        # it avoids reallocating ~N·K floats every tick (big win at large N, and
        # the preallocation pattern is what a GPU kernel wants too).
        self._buf = xp.empty_like(self.win_ker)
        self._check_dtypes()

    # ---- kernels (periodic: windows wrap, no clipping) --------------------
    def _build_kernels(self) -> None:
        """Per-unit normalized Gaussian window (Σ G·h² = 1), wrapped on the torus.
        Same kernel as Stage 1, but indices are taken mod (H, W) instead of being
        clipped at a Neumann wall."""
        cfg = self.cfg
        rad = cfg.kernel_width // 2
        offs = xp.arange(-rad, rad + 1, dtype=I32)
        px = self.pos[:, 0]; py = self.pos[:, 1]
        cc = xp.round(px / cfg.h).astype(I32)
        cr = xp.round(py / cfg.h).astype(I32)
        cols = cc[:, None] + offs[None, :]              # (N, ww)
        rows = cr[:, None] + offs[None, :]              # (N, ww)

        dx = (cols.astype(F32) * cfg.h - px[:, None])   # (N, ww)
        dy = (rows.astype(F32) * cfg.h - py[:, None])   # (N, ww)
        dist2 = (dy[:, :, None] ** 2 + dx[:, None, :] ** 2).astype(F32)   # (N,ww,ww)
        ker = xp.exp(-dist2 / F32(2.0 * cfg.sigma ** 2))
        del dist2
        norm = ker.sum(axis=(1, 2)) * F32(cfg.h ** 2)   # full Gaussian ⇒ Σ G·h² = 1
        ker = ker / norm[:, None, None]

        cols_w = cols % cfg.W                            # periodic wrap
        rows_w = rows % cfg.H
        flat = (rows_w[:, :, None] * cfg.W + cols_w[:, None, :]).astype(I32)

        K = cfg.kernel_width ** 2
        self.win_ker = ker.reshape(self.N, K).astype(F32)
        self.win_idx = flat.reshape(self.N, K).astype(I32)
        del ker, flat, cols, rows, dx, dy

    # ---- Fourier propagators ----------------------------------------------
    def _build_propagator(self) -> None:
        """Precompute the spectral multipliers on the rfft2 half-grid (H, W//2+1).

        K2 is the (negative) Laplacian symbol:
          symbol='spectral': |k|² = k_x²+k_y²            (exact continuum operator)
          symbol='discrete': (4/h²)[sin²(k_x h/2)+sin²(k_y h/2)]  (matches Stage-1's
                             5-point stencil exactly ⇒ numerically equal away from
                             boundaries — used by the equivalence test).
        """
        cfg = self.cfg
        ky = 2.0 * np.pi * np.fft.fftfreq(cfg.H, d=cfg.h)          # (H,)
        kx = 2.0 * np.pi * np.fft.rfftfreq(cfg.W, d=cfg.h)         # (W//2+1,)
        KY = xp.asarray(ky, dtype=F32)[:, None]
        KX = xp.asarray(kx, dtype=F32)[None, :]
        if cfg.symbol == "spectral":
            K2 = (KY ** 2 + KX ** 2).astype(F32)
        elif cfg.symbol == "discrete":
            K2 = (F32(4.0 / cfg.h ** 2) *
                  (xp.sin(KY * F32(cfg.h / 2.0)) ** 2 +
                   xp.sin(KX * F32(cfg.h / 2.0)) ** 2)).astype(F32)
        else:
            raise ValueError(f"unknown symbol {cfg.symbol!r}")
        self.K2 = K2
        alpha = (F32(cfg.lam) + F32(cfg.D) * K2).astype(F32)      # α = λ + D|k|²

        # steady:  φ_hat = S_hat / α.  k=0 zero-mode handling (α=0 only if λ=0):
        # divide against a safe denominator to avoid a 1/0 warning, then mask the
        # zero-mode to 0 (keep the zero-mean steady shape, drop the diverging DC).
        tiny = F32(1e-12)
        safe = xp.where(alpha > tiny, alpha, F32(1.0))
        self.inv_denom = xp.where(alpha > tiny, 1.0 / safe, F32(0.0)).astype(F32)

        # dynamic: spectral exponential integrator multipliers.
        #   φ_hat ← φ_hat·E + S_hat·factor,  E=e^{−αΔt}, factor=(1−E)/α → Δt as α→0
        E = xp.exp(-alpha * F32(cfg.dt)).astype(F32)
        factor = xp.where(alpha > tiny, (1.0 - E) / safe, F32(cfg.dt)).astype(F32)
        self.E = E
        self.factor = factor

    def _check_dtypes(self) -> None:
        for name in ("phi", "pos", "s", "a", "win_ker", "K2", "inv_denom", "E", "factor"):
            arr = getattr(self, name)
            assert arr.dtype == F32, f"{name} is {arr.dtype}, expected float32"
        assert self.win_idx.dtype == I32, f"win_idx is {self.win_idx.dtype}"

    # ---- the four operations ----------------------------------------------
    def deposit(self) -> "xp.ndarray":
        """DEPOSIT.  S(x) = Σ_i a_i·G(x − x_i).  Scatter-add via bincount.
        NOTE: xp.bincount returns float64 (numpy) — we cast straight back to
        float32. The float64 temporary is grid-sized (M), not unit-sized."""
        cfg = self.cfg
        xp.multiply(self.a[:, None], self.win_ker, out=self._buf)  # a_i·G → shared buf
        S = xp.bincount(self.win_idx.ravel(), weights=self._buf.ravel(),
                        minlength=cfg.H * cfg.W)
        return S.reshape(cfg.H, cfg.W).astype(F32)

    def evolve(self, S) -> None:
        """EVOLVE.  Spectral field solve (mode-dependent). Mutates self.phi."""
        cfg = self.cfg
        if cfg.mode == "steady":
            # φ = IFFT( FFT(S) / (λ + D|k|²) )   — exact screened steady field
            S_hat = xp.fft.rfft2(S).astype(C64)
            phi_hat = S_hat * self.inv_denom                      # complex64 · float32
            self.phi = xp.fft.irfft2(phi_hat, s=(cfg.H, cfg.W)).astype(F32)
        elif cfg.mode == "dynamic":
            # φ_hat ← φ_hat·E + S_hat·factor   — unconditionally stable exp step
            S_hat = xp.fft.rfft2(S).astype(C64)
            phi_hat = xp.fft.rfft2(self.phi).astype(C64)
            phi_hat = phi_hat * self.E + S_hat * self.factor
            self.phi = xp.fft.irfft2(phi_hat, s=(cfg.H, cfg.W)).astype(F32)
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")

    def read(self):
        """READ.  φ_i = ∫ G(x−x_i)·φ(x) dA ≈ Σ_window G·Φ·h².  Same kernel as
        deposit ⇒ read = h²·(deposit)ᵀ (the Stage-1 reciprocity, preserved)."""
        xp.take(self.phi.ravel(), self.win_idx, out=self._buf)    # gather φ → shared buf
        self._buf *= self.win_ker                                 # G·φ
        return self._buf.sum(axis=1) * F32(self.cfg.h ** 2)

    def update(self, phi_read) -> None:
        """UPDATE.  s_i ← s_i + Δt(−γ s_i + φ_i); a_i ← tanh(s_i). Clamped frozen."""
        cfg = self.cfg
        upd = ~self.clamped
        self.s[upd] += F32(cfg.dt) * (-F32(cfg.gamma) * self.s[upd] + phi_read[upd])
        self.a[upd] = xp.tanh(self.s[upd]).astype(F32)

    def step(self):
        S = self.deposit()
        self.evolve(S)
        phi_read = self.read()
        self.update(phi_read)
        return phi_read

    # ---- helper: read mapped back to the ORIGINAL (pre-sort) unit order ----
    def read_unsorted(self):
        return self.read()[self.inv_perm]


# ---------------------------------------------------------------------------
# Decay-length fit (asymptotic screened Green's fn, with the finite-r correction)
# ---------------------------------------------------------------------------
def fit_decay_length(r, phi):
    """Fit L from φ(r) using K0's asymptotic expansion incl. the first finite-r
    correction:  K0(x) = √(π/2x) e^{−x}(1 − 1/(8x) + …)  ⇒
        ln(√r·φ) ≈ −r/L + b − (L/8)/r
    so a 3-parameter least squares in basis {r, 1, 1/r} removes the leading
    finite-r bias that makes a naive exp fit overestimate L. Returns L_fit."""
    r = np.asarray(r, float); phi = np.asarray(phi, float)
    y = np.log(np.sqrt(r) * phi)
    A = np.vstack([r, np.ones_like(r), 1.0 / r]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return -1.0 / coef[0]


# ---------------------------------------------------------------------------
# TEST 1 — equivalence to Stage 1
# ---------------------------------------------------------------------------
def test_equivalence():
    """One driven unit at the center of a 128×128 box (≫ L=2). Compare the
    periodic FFT steady field to the Stage-1 Neumann reference."""
    print("\n[TEST 1] EQUIVALENCE TO STAGE 1  — periodic+FFT vs Neumann reference")
    import field_system as v1                          # the untouched Stage-1 file

    H = W = 128
    center = np.array([W / 2.0, H / 2.0])
    radii = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20], float)
    pos, rid = v1.make_rings(center, radii, n_ang=24)

    # --- Stage-1 reference (Neumann, 5-point, explicit Euler to steady) ----
    c1 = v1.Config(H=H, W=W)
    s1 = v1.FieldSystem(c1, positions=pos)
    s1.clamped[:] = True; s1.a[:] = 0.0; s1.a[0] = 1.0
    for _ in range(2500):
        s1.step()
    phi1 = s1.read()
    phi1_r = np.array([phi1[rid == r].mean() for r in radii])

    def run_v2(symbol):
        c2 = ConfigV2(H=H, W=W, mode="steady", symbol=symbol, sort_units=False)
        s2 = FieldSystemV2(c2, positions=pos, check_mem=False)
        s2.clamped[:] = True; s2.a[:] = 0.0; s2.a[0] = 1.0
        s2.step()                                       # steady in ONE FFT solve
        phi2 = to_numpy(s2.read())
        return np.array([phi2[rid == r].mean() for r in radii])

    phi2_spec = run_v2("spectral")                      # exact continuum operator
    phi2_disc = run_v2("discrete")                      # Stage-1's 5-point operator

    L = 2.0
    # Fit the screened falloff in the asymptotic window r∈[4L,10L]=[8,20]: K0's
    # clean exp tail only emerges for r ≫ L and r ≫ σ_eff(≈σ√2≈2.1); the mid-field
    # is distorted by the Gaussian smoothing and biases the fit.
    fwin = (radii >= 4 * L) & (radii <= 10 * L)
    L_fit = fit_decay_length(radii[fwin], phi2_spec[fwin])

    # central region = small r, far from the torus wrap at the box edge
    cen = radii <= 6
    rel_spec = np.abs(phi2_spec[cen] - phi1_r[cen]) / np.abs(phi1_r[cen])
    rel_disc = np.abs(phi2_disc[cen] - phi1_r[cen]) / np.abs(phi1_r[cen])

    print("       r     Stage1 φ(r)   v2-spectral   v2-discrete")
    for i, r in enumerate(radii):
        print(f"    {r:5.1f}  {phi1_r[i]:11.5e}  {phi2_spec[i]:11.5e}  {phi2_disc[i]:11.5e}")
    print(f"    fitted L (v2 spectral, r∈[8,20]) = {L_fit:.4f}   (target {L:.2f}, "
          f"{abs(L_fit - L) / L * 100:.1f}% off)")
    print(f"    central (r≤6) |Δ|/φ vs Stage-1:  spectral max {rel_spec.max()*100:5.2f}%   "
          f"discrete max {rel_disc.max()*100:5.2f}%")

    ok_L = abs(L_fit - L) / L <= 0.05
    ok_match = rel_spec.max() <= 0.03                   # default (spectral) mode
    ok = ok_L and ok_match
    print(f"    [{'PASS' if ok else 'FAIL'}] L within 5% of 2.0 ({ok_L}); central "
          f"field matches Stage-1 within a few % ({ok_match}). "
          f"[discrete operator matches to {rel_disc.max()*100:.2f}% — periodic+FFT "
          f"reproduces Stage-1 physics exactly.]")
    return ok, radii, phi1_r, phi2_spec, L_fit


# ---------------------------------------------------------------------------
# TEST 2 — Morton permutation invariance
# ---------------------------------------------------------------------------
def test_morton_invariance():
    """Sorted vs unsorted on identical input must give identical per-unit output
    (sorting is a pure memory reorder, not a physics change)."""
    print("\n[TEST 2] MORTON PERMUTATION INVARIANCE  — sorted vs unsorted")
    c_base = dict(H=128, W=128, N=20_000, seed=7, mode="steady")
    pos = np.random.default_rng(7).uniform([0, 0], [128, 128], size=(20_000, 2))
    a0 = np.tanh(np.random.default_rng(8).standard_normal(20_000)).astype(np.float32)

    def run(sort):
        s = FieldSystemV2(ConfigV2(sort_units=sort, **c_base),
                          positions=pos, check_mem=False)
        s.clamped[:] = True
        s.a[:] = xp.asarray(a0)[s.perm]                 # same physical emissions
        s.step()
        return to_numpy(s.read_unsorted())              # mapped to ORIGINAL order

    phi_sorted = run(True)
    phi_unsorted = run(False)
    max_abs = float(np.abs(phi_sorted - phi_unsorted).max())
    ok = max_abs < 1e-5
    print(f"    max |φ_sorted − φ_unsorted| (original order) = {max_abs:.3e}")
    print(f"    [{'PASS' if ok else 'FAIL'}] sorting is a pure permutation "
          f"(diff < 1e-5)")
    return ok


# ---------------------------------------------------------------------------
# TEST 3 — conservation (dynamic mode, λ=0, k=0 handled)
# ---------------------------------------------------------------------------
def test_conservation_dynamic():
    """Dynamic spectral step with λ=0: the k=0 update factor → Δt, so the DC mode
    integrates as φ_hat[0,0] += S_hat[0,0]·Δt and ∫Φ grows exactly linearly."""
    print("\n[TEST 3] CONSERVATION  — dynamic mode, λ=0, k=0 zero-mode → linear ∫Φ")
    c = ConfigV2(H=128, W=128, lam=0.0, mode="dynamic", sort_units=False)
    center = np.array([64.0, 64.0])
    pos = center + np.array([[-3, -3], [3, -3], [-3, 3], [3, 3]], float)
    s = FieldSystemV2(c, positions=pos, check_mem=False)
    s.clamped[:] = True; s.a[:] = 1.0                   # constant total source

    nsteps = 200
    integral = np.empty(nsteps)
    for t in range(nsteps):
        s.step()
        integral[t] = float(s.phi.sum()) * c.h ** 2     # ∫Φ dA
    tt = np.arange(1, nsteps + 1) * c.dt
    slope, intercept = np.polyfit(tt, integral, 1)
    drift = float(np.abs(integral - (slope * tt + intercept)).max())
    rel = drift / (abs(integral[-1]) + 1e-30)
    expected = float(to_numpy(s.a).sum())               # d(∫Φ)/dt = Σ a_i
    print(f"    measured d(∫Φ)/dt = {slope:.5f}   expected (Σ a_i) = {expected:.5f}")
    print(f"    max deviation from linear = {drift:.3e}  (relative {rel:.2e}, float32)")
    ok = rel < 1e-3 and abs(slope - expected) / expected < 1e-2
    print(f"    [{'PASS' if ok else 'FAIL'}] ∫Φ grows linearly within float32 tolerance")
    return ok


# ---------------------------------------------------------------------------
# SCALING MEASUREMENT
# ---------------------------------------------------------------------------
def _time_step(sys, warmup=1, timed=3):
    for _ in range(warmup):
        sys.step()
    t0 = time.perf_counter()
    for _ in range(timed):
        sys.step()
    return (time.perf_counter() - t0) / timed


def measure_scaling_N():
    """time/step vs N at fixed 256×256 grid → expect LINEAR (O(N) deposit+read)."""
    print("\n[SCALING] time/step vs N  (grid 256×256, steady mode)")
    Ns = [1_000, 10_000, 100_000, 300_000, 1_000_000, 2_000_000]
    times = []
    for N in Ns:
        cfg = ConfigV2(H=256, W=256, N=N, mode="steady")
        guard_memory(cfg, verbose=(N == Ns[-1]))        # show estimate at the largest N
        sys = FieldSystemV2(cfg)
        tps = _time_step(sys)
        times.append(tps)
        print(f"    N={N:>9,}   {tps*1e3:8.2f} ms/step")
        del sys; gc.collect()
    Ns = np.array(Ns, float); times = np.array(times)
    # Fit the exponent over the ASYMPTOTIC regime N≥1e5. Below that, time is
    # dominated by the (N-independent) FFT + Python overhead and by CPU cache
    # residency — not by the O(N) deposit/read term — so small-N points are not
    # representative of the algorithmic scaling.
    big = Ns >= 1e5
    expo = np.polyfit(np.log(Ns[big]), np.log(times[big]), 1)[0]
    print(f"    log-log slope (N≥1e5, asymptotic) = {expo:.3f}   (O(N) ⇒ ≈1.0)")
    return Ns, times, expo


def measure_scaling_M():
    """time/step vs M at fixed N=1e5 → field-solve cost expected ~ M log M (FFT).
    We also time the FFT solve in isolation to separate it from the (M-independent)
    deposit/read cost."""
    print("\n[SCALING] time/step vs M  (N=100,000, steady mode)")
    grids = [64, 128, 256, 512]
    Ms, full, solve = [], [], []
    for g in grids:
        cfg = ConfigV2(H=g, W=g, N=100_000, mode="steady")
        guard_memory(cfg, verbose=False)
        sys = FieldSystemV2(cfg)
        tps = _time_step(sys)
        # isolate the FFT solve
        S = sys.deposit()
        for _ in range(2):
            sys.evolve(S)
        t0 = time.perf_counter()
        for _ in range(10):
            sys.evolve(S)
        tsolve = (time.perf_counter() - t0) / 10
        Ms.append(g * g); full.append(tps); solve.append(tsolve)
        print(f"    {g:>3}² = M={g*g:>7,}   full {tps*1e3:7.2f} ms   "
              f"solve(FFT) {tsolve*1e3:7.3f} ms")
        del sys; gc.collect()
    Ms = np.array(Ms, float); full = np.array(full); solve = np.array(solve)
    MlogM = Ms * np.log2(Ms)
    # fit solve ≈ a + b·MlogM vs a + b·M²; compare residuals
    def fit_r2(x, y):
        A = np.vstack([x, np.ones_like(x)]).T
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ c
        ss = 1.0 - resid.var() / y.var()
        return c, ss
    _, r2_mlogm = fit_r2(MlogM, solve)
    _, r2_m2 = fit_r2(Ms ** 2, solve)
    # Exponent over the LARGE-M regime (M≥128²): small FFTs are floored by per-call
    # overhead (~0.1 ms), so only large M reflects the true M log M cost.
    big = Ms >= 128 ** 2
    expo_solve = np.polyfit(np.log(Ms[big]), np.log(solve[big]), 1)[0]
    # the decisive M-log-M-vs-M² discriminator: a 4× increase in M
    quad = solve[-1] / solve[-2]                          # 256²→512², i.e. M×4
    print(f"    FFT-solve fit R²:  M·log M = {r2_mlogm:.4f}   vs   M² = {r2_m2:.4f}")
    print(f"    FFT-solve exponent (M≥128²) = {expo_solve:.3f}   "
          f"(M log M ⇒ ≈1.0–1.1, NOT 2.0)")
    print(f"    last 4×-M step (256²→512²): time ×{quad:.2f}   "
          f"(M log M ⇒ ×~4.4,  M² ⇒ ×16  → M² ruled out)")
    return Ms, full, solve, MlogM, r2_mlogm, r2_m2, expo_solve, quad


def plot_scaling_N(Ns, times, expo, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(Ns, times * 1e3, "o-", color="C0", label="measured")
    fitc = np.polyfit(np.log(Ns[Ns >= 1e5]), np.log(times[Ns >= 1e5]), 1)
    xr = np.array([Ns[Ns >= 1e5].min(), Ns.max()])
    ax.loglog(xr, np.exp(fitc[1]) * xr ** fitc[0] * 1e3, "--", color="C3",
              label=f"slope={expo:.2f} (O(N))")
    ax.set_xlabel("N (units)"); ax.set_ylabel("time / step (ms)")
    ax.set_title("Stage-2a CPU scaling: time/step vs N  (grid 256²)")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_scaling_M(Ms, full, solve, expo, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    g = np.sqrt(Ms).astype(int)
    ax.loglog(Ms, solve * 1e3, "o-", color="C0", label="FFT solve")
    ax.loglog(Ms, full * 1e3, "s--", color="C2", alpha=0.6, label="full step")
    ref = Ms * np.log2(Ms); ref = ref / ref[-1] * solve[-1]
    ax.loglog(Ms, ref * 1e3, ":", color="C3", label="∝ M·log M")
    ref2 = Ms ** 2; ref2 = ref2 / ref2[-1] * solve[-1]
    ax.loglog(Ms, ref2 * 1e3, ":", color="gray", label="∝ M² (ruled out)")
    for x, gg in zip(Ms, g):
        ax.annotate(f"{gg}²", (x, solve[np.where(Ms == x)][0] * 1e3),
                    textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax.set_xlabel("M (grid cells)"); ax.set_ylabel("time / step (ms)")
    ax.set_title(f"Stage-2a CPU scaling: time vs M  (N=1e5, FFT slope={expo:.2f})")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    cfg = ConfigV2()
    print("=" * 70)
    print("FIELD-MEDIATED NEURAL SYSTEM — Stage 2a (periodic / FFT / Morton)")
    print("=" * 70)
    print(f"  backend xp = {xp.__name__}   dtype = float32")
    print(f"  grid {cfg.H}×{cfg.W}  h={cfg.h}  σ={cfg.sigma}  D={cfg.D}  λ={cfg.lam}")
    print(f"  influence range L = sqrt(D/λ) = {cfg.influence_range:.4f}")
    print(f"  default mode={cfg.mode!r}  symbol={cfg.symbol!r}  sort_units={cfg.sort_units}")
    print(f"  caps: N ≤ {cfg.N_CAP:,}, grid ≤ {cfg.GRID_CAP}²")
    guard_memory(cfg, verbose=True)

    ok1, *_ = test_equivalence()
    ok2 = test_morton_invariance()
    ok3 = test_conservation_dynamic()

    Ns, tN, expoN = measure_scaling_N()
    Ms, fM, sM, MlogM, r2a, r2b, expoM, quadM = measure_scaling_M()

    print("\n[FIGURES] writing PNGs ...")
    plot_scaling_N(Ns, tN, expoN, "field_v2_scaling_N.png")
    plot_scaling_M(Ms, fM, sM, expoM, "field_v2_scaling_M.png")
    print("    field_v2_scaling_N.png   field_v2_scaling_M.png")

    okN = 0.9 <= expoN <= 1.2
    okM = (r2a > r2b) and (expoM < 1.5) and (quadM < 8.0)
    print("\n" + "-" * 70)
    print("NOTE: Absolute times are CPU; the real-time (16 ms) ceiling must be")
    print("measured on GPU (Stage 2b). This test confirms SCALING SHAPE only.")
    print("-" * 70)

    results = [("EQUIVALENCE", ok1), ("MORTON-INVARIANCE", ok2),
               ("CONSERVATION", ok3),
               (f"N-SCALING (exp={expoN:.2f}≈1)", okN),
               (f"M-SCALING (MlogM not M²)", okM)]
    print("\n" + "=" * 70)
    print("STAGE-2a TEST SUMMARY")
    print("=" * 70)
    for name, ok in results:
        print(f"   [{'PASS' if ok else 'FAIL'}]  {name}")
    allok = all(ok for _, ok in results)
    print("=" * 70)
    print("ALL TESTS PASSED" if allok else "SOME TESTS FAILED")
    raise SystemExit(0 if allok else 1)


if __name__ == "__main__":
    main()
