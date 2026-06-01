#!/usr/bin/env python3
"""
Field-Mediated Neural System — Stage 1 CPU Reference Implementation
===================================================================

"Neurons as atoms in a field." Discrete UNITS (fixed positions) never see one
another; they only DEPOSIT into a shared continuous scalar FIELD and READ back
from it. ALL coupling is mediated by the field, which evolves by
diffusion-with-decay driven by the unit sources:

        ∂φ/∂t = D·∇²φ − λ·φ + S            (S = sum of unit sources)

This is a correctness-first reference: NumPy + matplotlib only, no GPU, no
optimization tricks, no learning, no unit motion. The goal is to *verify*
(falsifiably) that a shared diffusion field can carry signal between units,
with the right stability, locality, causality and conservation behaviour.

One tick = four operations, in THIS exact order:

    1. DEPOSIT  S(x) = Σ_i a_i · G(x − x_i)          G: normalized 2D Gaussian
    2. EVOLVE   φ ← φ + Δt·(D·∇²φ − λ·φ + S)         explicit Euler, Neumann BC
    3. READ     φ_i = ∫ G(x − x_i)·φ(x) dA           SAME kernel  ⇒ symmetric
    4. UPDATE   s_i ← s_i + Δt·(−γ·s_i + φ_i);  a_i ← tanh(s_i)

Why read uses the same kernel G (the symmetry that makes this a real field
theory): with deposit  S = G·a  and read  φ = h²·Gᵀ·Φ, the read operator is
exactly h²·(deposit)ᵀ. The steady-state unit→unit coupling is therefore
h²·Gᵀ·M·G with M = (λI − D·L)⁻¹; because the Neumann 5-point Laplacian L is a
symmetric (graph-Laplacian) matrix, M is symmetric and so the coupling i↔j is
reciprocal. Action through the field obeys Newton's third law.

Run:   python field_system.py
Deps:  numpy, matplotlib   (only)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import matplotlib
matplotlib.use("Agg")              # headless backend: write PNGs, never open a window
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    """All parameters in one place. Defaults are the Stage-1 spec defaults."""
    H: int = 128                   # field grid height (rows, y)
    W: int = 128                   # field grid width  (cols, x)
    h: float = 1.0                 # grid spacing (physical length per cell)
    sigma: float = 1.5             # Gaussian deposit/read std  σ
    D: float = 0.2                 # diffusion coefficient  D
    lam: float = 0.05              # field decay rate  λ
    gamma: float = 0.1             # unit-state leak rate  γ
    dt: float = 1.0                # time step  Δt
    N: int = 10000                 # number of units
    seed: int = 0                  # RNG seed for unit placement
    kernel_radius_sigmas: float = 4.0   # Gaussian window half-width, in units of σ

    # --- derived quantities -------------------------------------------------
    @property
    def influence_range(self) -> float:
        """Screening / influence length of the field:  L = sqrt(D/λ).

        This is the length scale of the steady screened-diffusion Green's
        function K0(r/L): the field from a point source is large within r≈L
        and exponentially negligible beyond a few L."""
        return float(np.sqrt(self.D / self.lam))

    @property
    def dt_max(self) -> float:
        """Hard explicit-Euler stability bound for diffusion:  Δt ≤ h²/(4D)."""
        return self.h ** 2 / (4.0 * self.D)

    def describe(self) -> str:
        return (
            "  grid           H×W      = {H}×{W}  (h={h})\n"
            "  Gaussian       σ        = {sig}\n"
            "  diffusion      D        = {D}\n"
            "  field decay    λ        = {lam}\n"
            "  unit leak      γ        = {gam}\n"
            "  time step      Δt       = {dt}\n"
            "  units          N        = {N}   (seed={seed})\n"
            "  ----------------------------------------------------\n"
            "  influence range  L = sqrt(D/λ)        = {L:.4f}\n"
            "  stability bound  Δt_max = h²/(4D)     = {dtm:.4f}\n"
        ).format(H=self.H, W=self.W, h=self.h, sig=self.sigma, D=self.D,
                 lam=self.lam, gam=self.gamma, dt=self.dt, N=self.N,
                 seed=self.seed, L=self.influence_range, dtm=self.dt_max)


def assert_stable(cfg: Config) -> None:
    """HARD STABILITY CONSTRAINT. Refuse to run if Δt > h²/(4D)."""
    if cfg.dt > cfg.dt_max + 1e-12:
        print("\n*** REFUSING TO RUN — explicit Euler would be unstable ***")
        print(f"    Δt = {cfg.dt} violates  Δt ≤ h²/(4D) = {cfg.dt_max:.6f}")
        print(f"    Reduce Δt to at most {cfg.dt_max:.6f} and try again.")
        raise SystemExit(1)


# ----------------------------------------------------------------------------
# The field system
# ----------------------------------------------------------------------------
class FieldSystem:
    """Holds the field Φ, the units (fixed positions, state s, emission a) and
    the config. The four operations are deposit / evolve / read / update, and
    step() composes them in order. The ONLY persistent state is (phi, s, a):
    the transient source S and the read-back φ_i are passed explicitly between
    methods, never stashed on the object (so step() has no hidden mutation).
    """

    def __init__(self, cfg: Config, positions: np.ndarray | None = None):
        assert_stable(cfg)                     # safety: never build an unstable system
        self.cfg = cfg

        # --- the field: scalar Φ on an H×W grid, Φ[row, col] = Φ[y, x] -------
        self.phi = np.zeros((cfg.H, cfg.W), dtype=np.float64)

        # --- unit positions (physical coords, x=col, y=row) -----------------
        rng = np.random.default_rng(cfg.seed)
        if positions is None:
            # uniform random in the interior, kept ≥3σ from every edge
            m = 3.0 * cfg.sigma
            xs = rng.uniform(m * cfg.h, (cfg.W - 1) * cfg.h - m * cfg.h, cfg.N)
            ys = rng.uniform(m * cfg.h, (cfg.H - 1) * cfg.h - m * cfg.h, cfg.N)
            positions = np.stack([xs, ys], axis=1)
        self.pos = np.asarray(positions, dtype=np.float64)      # (N, 2): (x, y)
        self.N = self.pos.shape[0]

        # --- unit state -----------------------------------------------------
        self.s = np.zeros(self.N, dtype=np.float64)             # internal state s_i
        self.a = np.zeros(self.N, dtype=np.float64)             # emission a_i
        self.clamped = np.zeros(self.N, dtype=bool)             # clamped ⇒ skip update()

        # --- precompute the per-unit Gaussian windows (positions are fixed) --
        self._build_kernels()

    # ---- kernel precomputation --------------------------------------------
    def _build_kernels(self) -> None:
        """For each unit, precompute a local Gaussian window: the field indices
        it touches and the (normalized) kernel weights. Positions never move, so
        this is done once. The kernel is the normalized 2D Gaussian

            G(r) = (1 / 2πσ²) · exp(−|r|² / 2σ²),     ∫ G dA = 1,

        sampled on the grid and renormalized so the *discrete* integral
        Σ G·h² = 1 exactly. Renormalizing per-unit also makes truncation at the
        window edge and clipping at the grid boundary mass-conserving, and (since
        the diffusion propagator is symmetric) leaves the i↔j coupling reciprocal.
        """
        cfg = self.cfg
        rad = int(np.ceil(cfg.kernel_radius_sigmas * cfg.sigma / cfg.h))   # window half-width (cells)
        ww = 2 * rad + 1
        offs = np.arange(-rad, rad + 1)                         # (ww,) cell offsets

        px = self.pos[:, 0]                                     # (N,) x (col) physical
        py = self.pos[:, 1]                                     # (N,) y (row) physical
        cc = np.round(px / cfg.h).astype(np.int64)             # (N,) nearest center col
        cr = np.round(py / cfg.h).astype(np.int64)             # (N,) nearest center row

        cols = cc[:, None] + offs[None, :]                     # (N, ww) integer col indices
        rows = cr[:, None] + offs[None, :]                     # (N, ww) integer row indices

        # signed distances (physical) from each window cell center to the unit
        dx = cols * cfg.h - px[:, None]                        # (N, ww) along x (cols)
        dy = rows * cfg.h - py[:, None]                        # (N, ww) along y (rows)
        dist2 = dy[:, :, None] ** 2 + dx[:, None, :] ** 2      # (N, ww, ww): [unit, row, col]

        ker = np.exp(-dist2 / (2.0 * cfg.sigma ** 2))          # un-normalized Gaussian

        # mask out cells that fall outside the grid (Neumann box: nothing wraps)
        row_ok = (rows >= 0) & (rows < cfg.H)                  # (N, ww)
        col_ok = (cols >= 0) & (cols < cfg.W)                  # (N, ww)
        valid = row_ok[:, :, None] & col_ok[:, None, :]        # (N, ww, ww)
        ker = np.where(valid, ker, 0.0)

        # normalize so Σ G·h² = 1  (discrete integral = 1, like the continuous one)
        norm = ker.sum(axis=(1, 2)) * cfg.h ** 2               # (N,)
        norm = np.where(norm > 0, norm, 1.0)                   # guard (placement avoids 0)
        ker = ker / norm[:, None, None]

        # flat field indices for scatter/gather (clamp invalid cells; weight is 0 there)
        rclip = np.clip(rows, 0, cfg.H - 1)
        cclip = np.clip(cols, 0, cfg.W - 1)
        flat = rclip[:, :, None] * cfg.W + cclip[:, None, :]   # (N, ww, ww)

        self.win_ker = ker.reshape(self.N, ww * ww)            # (N, K) kernel weights
        self.win_idx = flat.reshape(self.N, ww * ww).astype(np.intp)   # (N, K) flat indices

    # ---- the four operations ----------------------------------------------
    def deposit(self) -> np.ndarray:
        """Operation 1 — DEPOSIT.   S(x) = Σ_i a_i · G(x − x_i).

        Zero the source array, then scatter each unit's emission a_i through its
        normalized Gaussian window into S. (bincount = vectorized scatter-add.)
        """
        cfg = self.cfg
        weights = (self.a[:, None] * self.win_ker).ravel()     # a_i · G(x − x_i)
        S = np.bincount(self.win_idx.ravel(), weights=weights,
                        minlength=cfg.H * cfg.W)
        return S.reshape(cfg.H, cfg.W)

    def evolve(self, S: np.ndarray) -> None:
        """Operation 2 — EVOLVE.   Φ ← Φ + Δt·(D·∇²Φ − λ·Φ + S).

        Explicit Euler in time; 5-point Laplacian in space with Neumann
        (zero-flux) boundaries. Mutates self.phi."""
        cfg = self.cfg
        lap = self._laplacian(self.phi)                        # ∇²Φ
        self.phi = self.phi + cfg.dt * (cfg.D * lap - cfg.lam * self.phi + S)

    def read(self) -> np.ndarray:
        """Operation 3 — READ.   φ_i = ∫ G(x − x_i)·φ(x) dA ≈ Σ_window G·Φ·h².

        Same kernel G as deposit ⇒ the read operator is h²·(deposit)ᵀ, which is
        what makes unit→unit coupling symmetric. With Σ G·h² = 1, this is exactly
        the Gaussian-weighted average of Φ around the unit."""
        cfg = self.cfg
        phi_at = self.phi.ravel()[self.win_idx]                # (N, K) gather field values
        return (self.win_ker * phi_at).sum(axis=1) * cfg.h ** 2

    def update(self, phi_read: np.ndarray) -> None:
        """Operation 4 — UPDATE.   s_i ← s_i + Δt·(−γ·s_i + φ_i); a_i ← tanh(s_i).

        Clamped units (e.g. a pinned driver, or passive probes) skip this so
        their (s, a) stay frozen. tanh caps |a| < 1, which (with decay λ) caps
        the field."""
        cfg = self.cfg
        upd = ~self.clamped
        self.s[upd] += cfg.dt * (-cfg.gamma * self.s[upd] + phi_read[upd])
        self.a[upd] = np.tanh(self.s[upd])

    def step(self) -> np.ndarray:
        """One tick: the four operations in order. Returns the read-back φ_i
        (handy for tests). The only state mutated is (phi, s, a)."""
        S = self.deposit()          # 1. build source from emissions
        self.evolve(S)              # 2. advance the field
        phi_read = self.read()      # 3. sample the field at each unit
        self.update(phi_read)       # 4. advance unit states
        return phi_read

    # ---- helpers -----------------------------------------------------------
    def _laplacian(self, f: np.ndarray) -> np.ndarray:
        """5-point Laplacian with Neumann (zero-flux) boundaries.

            ∇²Φ[p,q] = (Φ[p+1,q]+Φ[p−1,q]+Φ[p,q+1]+Φ[p,q−1] − 4Φ[p,q]) / h²

        'edge' padding sets each ghost cell equal to the boundary cell, so the
        flux across the boundary face is zero (reflecting box). A direct
        consequence: Σ ∇²Φ = 0 exactly, i.e. diffusion moves field around but
        never creates or destroys it (used by the conservation test)."""
        p = np.pad(f, 1, mode="edge")
        lap = (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2]
               - 4.0 * f) / self.cfg.h ** 2
        return lap


# ----------------------------------------------------------------------------
# Small geometry helper: a driven unit at the center plus rings of probes
# ----------------------------------------------------------------------------
def make_rings(center: np.ndarray, radii, n_ang: int):
    """Return (positions, ring_id). Index 0 is the central (driven) unit with
    ring_id 0; the rest are n_ang probes evenly spaced on each ring of the given
    radii. ring_id lets us angularly average φ over each ring."""
    pos = [center.astype(float).copy()]
    rid = [0.0]
    for r in radii:
        for k in range(n_ang):
            ang = 2.0 * np.pi * k / n_ang
            pos.append(center + r * np.array([np.cos(ang), np.sin(ang)]))
            rid.append(float(r))
    return np.asarray(pos), np.asarray(rid)


def fit_decay_length(r: np.ndarray, phi: np.ndarray):
    """Fit a screening length L to φ(r) using the large-r asymptote of the 2D
    screened-diffusion Green's function:

        K0(r/L)  ~  sqrt(π L / 2r) · exp(−r/L)      (r ≫ L)
        ⇒  ln( sqrt(r) · φ )  ≈  −(1/L)·r + const

    so a straight-line fit of ln(√r·φ) vs r has slope −1/L. Returns
    (L_fit, slope, intercept). Also returns a naive exp-only length for context.
    """
    y = np.log(np.sqrt(r) * phi)
    slope, intercept = np.polyfit(r, y, 1)
    L_fit = -1.0 / slope
    # naive fit without the √r prefactor, for comparison/printing
    slope2, _ = np.polyfit(r, np.log(phi), 1)
    L_naive = -1.0 / slope2
    return L_fit, slope, intercept, L_naive


# ----------------------------------------------------------------------------
# ACCEPTANCE TESTS  (each prints PASS/FAIL + diagnostics)
# ----------------------------------------------------------------------------
def test_stability(cfg: Config):
    """TEST 1 — STABILITY. Run 2000 steps with ALL units emitting; assert Φ
    never NaNs and stays bounded (tanh caps emission, decay caps the field)."""
    print("\n[TEST 1] STABILITY  — 2000 steps, all units emitting")
    sys = FieldSystem(cfg)
    # Seed every unit with a nonzero internal state so they are ALL emitting from
    # t=0 (a=tanh(s)≠0). Without this the system sits at the trivial a≡0, φ≡0
    # fixed point and the test would pass vacuously. Positive states make every
    # unit deposit the same sign — the maximal-field, worst-case stress test for
    # boundedness under the field's positive feedback.
    rng = np.random.default_rng(cfg.seed + 1)
    sys.s = rng.uniform(0.5, 1.5, sys.N)
    sys.a = np.tanh(sys.s)
    nsteps = 2000
    maxphi = np.empty(nsteps)
    t0 = time.time()
    for t in range(nsteps):
        sys.step()
        maxphi[t] = np.abs(sys.phi).max()
        if (t + 1) % 250 == 0:
            print(f"    step {t + 1:4d}   max|Φ| = {maxphi[t]:10.4f}")
    dt_wall = time.time() - t0

    finite = bool(np.isfinite(sys.phi).all() and np.isfinite(maxphi).all())
    bounded = bool(maxphi.max() < 1e6)
    tail = maxphi[-100:]
    plateau = float((tail.max() - tail.min()) / (tail.mean() + 1e-12))
    ok = finite and bounded                                   # spec criteria: finite & bounded
    print(f"    finite={finite}  bounded={bounded}  peak max|Φ|={maxphi.max():.4f}"
          f"  late-time spread={plateau:.2%}  ({dt_wall:.1f}s)")
    print(f"    [{'PASS' if ok else 'FAIL'}] Φ stays finite and bounded")
    return ok, maxphi


def test_locality(cfg: Config):
    """TEST 2 — LOCALITY (the critical one). Pin ONE unit's emission to +1; a
    ring of passive probes at increasing r reads the steady field. Verify φ(r)
    falls off like the screened Green's function K0(r/L), L = sqrt(D/λ)."""
    print("\n[TEST 2] LOCALITY  — one driven unit, probe rings, steady state")
    L = cfg.influence_range
    center = np.array([cfg.W * cfg.h / 2.0, cfg.H * cfg.h / 2.0])
    radii = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14], dtype=float)
    pos, rid = make_rings(center, radii, n_ang=8)

    sys = FieldSystem(cfg, positions=pos)
    sys.clamped[:] = True              # freeze everyone (skip update)
    sys.a[:] = 0.0                     # probes are passive sensors (no deposit)
    sys.a[0] = 1.0                     # driven unit pinned to +1

    nsteps = 3000                      # run to steady state
    for _ in range(nsteps):
        sys.step()
    phi_read = sys.read()              # steady φ_i at every unit

    # angularly average over each ring  → φ(r)
    phi_r = np.array([phi_read[rid == r].mean() for r in radii])

    # fit decay length over the screened regime (r ≥ 2σ, away from the smoothed core)
    fmask = (radii >= 2.0 * cfg.sigma) & (phi_r > 0)
    L_fit, slope, intercept, L_naive = fit_decay_length(radii[fmask], phi_r[fmask])

    print(f"    L = sqrt(D/λ) = {L:.4f}     (screening length to recover)")
    print("       r        φ(r)        φ(r)/φ(r0)")
    phi0 = phi_r[0]
    for r, p in zip(radii, phi_r):
        print(f"    {r:5.1f}   {p:11.5e}   {p / phi0:9.4f}")
    near = phi_r[np.argmin(np.abs(radii - L))]                # φ at r≈L
    far = phi_r[np.argmin(np.abs(radii - 5 * L))]             # φ at r≈5L
    print(f"    φ(r≈L)/φ(r≈5L) = {near / far:.1f}×   (large within L, tiny beyond a few L)")
    print(f"    fitted length: L_fit(K0 asymptote) = {L_fit:.4f}   "
          f"L_fit(naive exp) = {L_naive:.4f}")

    ok = bool(0.5 * L <= L_fit <= 2.0 * L)
    print(f"    [{'PASS' if ok else 'FAIL'}] fitted length within 2× of sqrt(D/λ)")
    return ok, sys, radii, phi_r, (L, L_fit, slope, intercept)


def test_causality(cfg: Config):
    """TEST 3 — CAUSALITY-ish. Drive one unit with a step input at t=0 (field
    starts at 0); confirm a distant probe responds only after a delay that grows
    with distance (diffusion spreads gradually, not instantly). Also captures
    field snapshots of the single-source spread for the figure."""
    print("\n[TEST 3] CAUSALITY  — step input at t=0, response delay vs distance")
    center = np.array([cfg.W * cfg.h / 2.0, cfg.H * cfg.h / 2.0])
    radii = np.array([3, 5, 7, 9, 12], dtype=float)
    pos, rid = make_rings(center, radii, n_ang=8)

    sys = FieldSystem(cfg, positions=pos)
    sys.clamped[:] = True
    sys.a[:] = 0.0
    sys.a[0] = 1.0                     # step on at t=0

    nsteps = 1500
    snap_times = [5, 20, 60, 200]      # transient → steady, for the "spreading" figure
    snaps = {}
    rec = np.zeros((nsteps, len(radii)))
    for t in range(nsteps):
        phir = sys.step()
        rec[t] = [phir[rid == r].mean() for r in radii]       # ring-averaged φ(r, t)
        if (t + 1) in snap_times:
            snaps[t + 1] = sys.phi.copy()

    steady = rec[-1]
    # response time = first time φ reaches 50% of its own steady value
    tresp = np.array([np.argmax(rec[:, k] >= 0.5 * steady[k]) * cfg.dt
                      for k in range(len(radii))])
    print("       r      steady φ      t(50% rise)")
    for r, sv, tr in zip(radii, steady, tresp):
        print(f"    {r:5.1f}   {sv:11.5e}   {tr:8.1f}")
    monotonic = bool(np.all(np.diff(tresp) > 0))
    print(f"    response time strictly increases with distance: {monotonic}")
    print(f"    [{'PASS' if monotonic else 'FAIL'}] no action at infinite speed "
          f"(delay grows with r)")
    return monotonic, snaps, snap_times


def test_conservation(cfg: Config):
    """TEST 4 — CONSERVATION SANITY. With λ=0, Neumann boundaries and a constant
    total source, the total field integral ∫Φ must grow exactly linearly (no
    decay, nothing leaks through the reflecting box)."""
    print("\n[TEST 4] CONSERVATION  — λ=0, constant source, ∫Φ should grow linearly")
    cfg0 = replace(cfg, lam=0.0)
    center = np.array([cfg0.W * cfg0.h / 2.0, cfg0.H * cfg0.h / 2.0])
    offsets = np.array([[-3, -3], [3, -3], [-3, 3], [3, 3]], dtype=float)
    pos = center + offsets

    sys = FieldSystem(cfg0, positions=pos)
    sys.clamped[:] = True
    sys.a[:] = 1.0                     # constant emission ⇒ constant total source

    nsteps = 200
    integral = np.empty(nsteps)
    for t in range(nsteps):
        sys.step()
        integral[t] = sys.phi.sum() * cfg0.h ** 2             # ∫Φ dA

    tt = np.arange(1, nsteps + 1) * cfg0.dt
    slope, intercept = np.polyfit(tt, integral, 1)
    drift = float(np.abs(integral - (slope * tt + intercept)).max())
    rel_drift = drift / (abs(integral[-1]) + 1e-30)
    expected_slope = sys.a.sum()       # d(∫Φ)/dt = Σ a_i  (each unit injects ∫G dA = 1)

    print(f"    measured d(∫Φ)/dt = {slope:.6f}   expected (Σ a_i) = {expected_slope:.6f}")
    print(f"    max deviation from linear fit (drift) = {drift:.3e}  "
          f"(relative {rel_drift:.2e})")
    ok = bool(rel_drift < 1e-6)
    print(f"    [{'PASS' if ok else 'FAIL'}] ∫Φ grows linearly (no leak, no decay)")
    return ok


# ----------------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------------
def fig_steady_heatmap(sys: FieldSystem, path: str) -> None:
    """(a) Steady-state field heatmap with the driven unit marked."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sys.phi, origin="lower", cmap="magma",
                   extent=[0, sys.cfg.W * sys.cfg.h, 0, sys.cfg.H * sys.cfg.h])
    drv = sys.pos[0]
    ax.plot(drv[0], drv[1], "c+", markersize=16, markeredgewidth=2.5,
            label="driven unit (a=+1)")
    ax.set_title("Steady-state field Φ  (single driven unit)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="Φ")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_locality_curve(radii, phi_r, fitinfo, path: str) -> None:
    """(b) φ(r) locality curve with the fitted decay length."""
    L, L_fit, slope, intercept = fitinfo
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogy(radii, phi_r, "o", color="C0", label="measured φ(r)")
    rr = np.linspace(radii.min(), radii.max(), 200)
    fit_curve = np.exp(intercept + slope * rr) / np.sqrt(rr)   # √r·φ = exp(slope·r+int)
    ax.semilogy(rr, fit_curve, "-", color="C3",
                label=f"K0 asymptote fit, L_fit={L_fit:.2f}")
    ax.axvline(L, color="gray", ls="--", label=f"L=sqrt(D/λ)={L:.2f}")
    ax.set_title("Locality: field falloff vs distance")
    ax.set_xlabel("distance r from driven unit"); ax.set_ylabel("φ(r)  (log scale)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_spread(snaps, snap_times, cfg: Config, path: str) -> None:
    """(c) Field snapshots at 4 timepoints showing a deposit spreading.

    A LOG color scale (shared across panels) is used on purpose: with L=sqrt(D/λ)
    ≈ 2 the field is strongly screened, so on a linear scale you'd only see a dot
    brighten. In log the exponential skirt is visible and you can watch the lit
    region grow outward over time — the signal arriving at larger r later (the
    same delay the causality test measures), i.e. no action at infinite speed."""
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    vmax = max(s.max() for s in snaps.values())
    norm = LogNorm(vmin=vmax * 1e-3, vmax=vmax)               # 3 decades of skirt
    crop = 28                                                 # zoom to the active region
    cx = cy = cfg.W // 2
    for ax, t in zip(axes, snap_times):
        im = ax.imshow(snaps[t], origin="lower", cmap="viridis", norm=norm,
                       extent=[0, cfg.W * cfg.h, 0, cfg.H * cfg.h])
        ax.plot(cfg.W * cfg.h / 2, cfg.H * cfg.h / 2, "r+", markersize=10, mew=2)
        ax.set_xlim(cx - crop, cx + crop); ax.set_ylim(cy - crop, cy + crop)
        ax.set_title(f"t = {t}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle("A single deposit spreading through the diffusion field "
                 "(log color scale; field front reaches larger r later)")
    fig.colorbar(im, ax=axes, label="Φ (log)", shrink=0.8)
    fig.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    cfg = Config()

    print("=" * 68)
    print("FIELD-MEDIATED NEURAL SYSTEM — Stage 1 CPU reference")
    print("=" * 68)
    print(cfg.describe())

    # HARD STABILITY CONSTRAINT — checked before anything runs.
    assert_stable(cfg)
    print("  stability check: Δt ≤ h²/(4D)  OK\n")

    # --- short demo: drive a few units, step 500× ---------------------------
    print("[DEMO] drive 5 units (a=+1), step 500× ...")
    demo = FieldSystem(cfg)
    demo.clamped[:5] = True
    demo.a[:5] = 1.0
    t0 = time.time()
    for _ in range(500):
        demo.step()
    print(f"    after 500 steps:  max|Φ|={np.abs(demo.phi).max():.4f}   "
          f"mean|a|(free units)={np.abs(demo.a[5:]).mean():.4f}   "
          f"({time.time() - t0:.1f}s)")

    # --- acceptance tests ---------------------------------------------------
    ok1, _maxphi = test_stability(cfg)
    ok2, loc_sys, radii, phi_r, fitinfo = test_locality(cfg)
    ok3, snaps, snap_times = test_causality(cfg)
    ok4 = test_conservation(cfg)

    # --- figures ------------------------------------------------------------
    print("\n[FIGURES] writing PNGs ...")
    fig_steady_heatmap(loc_sys, "field_steady_heatmap.png")
    fig_locality_curve(radii, phi_r, fitinfo, "field_locality_curve.png")
    fig_spread(snaps, snap_times, cfg, "field_spread_snapshots.png")
    print("    field_steady_heatmap.png   field_locality_curve.png   "
          "field_spread_snapshots.png")

    # --- summary ------------------------------------------------------------
    results = [("STABILITY", ok1), ("LOCALITY", ok2),
               ("CAUSALITY", ok3), ("CONSERVATION", ok4)]
    print("\n" + "=" * 68)
    print("ACCEPTANCE TEST SUMMARY")
    print("=" * 68)
    for name, ok in results:
        print(f"   [{'PASS' if ok else 'FAIL'}]  {name}")
    allok = all(ok for _, ok in results)
    print("=" * 68)
    print("ALL TESTS PASSED" if allok else "SOME TESTS FAILED")
    raise SystemExit(0 if allok else 1)


if __name__ == "__main__":
    main()
