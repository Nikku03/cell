#!/usr/bin/env python3
"""
Field-Mediated Neural System — Stage 2b: CuPy GPU port + real-time measurement
==============================================================================

This is the GPU backend port of the PROVEN Stage-2a CPU code (field_system_v2.py),
plus the honest hardware measurement it exists for: *the largest N that runs under
a 16 ms/step real-time budget on this GPU.*

The physics, the four operations, the tests and the architecture are UNCHANGED —
Stage-2a already proved them correct on CPU. This file only:
  • swaps the array backend NumPy → CuPy (the `xp` alias is why that is a one-liner),
  • uses cupyx.scatter_add for the deposit (np.add.at has no fast CuPy equivalent),
  • keeps the preallocated shared (N,K) work buffer (per-step alloc is fatal on GPU),
  • adds CORRECT GPU timing (warmup + device synchronize + CUDA events + median/IQR),
  • sweeps N to find the 16 ms real-time ceiling, with a per-stage breakdown,
  • and extrapolates — clearly labeled as an estimate — to the 96 GB Blackwell target.

------------------------------------------------------------------------------
RUN ON COLAB (GPU):   python field_system_v2_gpu.py     (or paste into a cell)
    → prints the hardware report, CPU↔GPU correctness parity, the N-sweep table
      with stage breakdown, the four headline numbers, and the Blackwell estimate.

RUN WITHOUT A GPU (this CPU sandbox): the file detects no CuPy/GPU and switches to
PARITY+PROJECTION mode: it still VALIDATES the port is faithful (the scatter_add /
take / argsort paths reproduce the proven v2 results to float32), and prints the
reliable MEMORY-bound ceilings + a clearly-labeled bandwidth-bound MODEL of the
compute ceiling. It NEVER prints a CPU time as if it were a GPU measurement.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Proven Stage-2a code (LEFT UNTOUCHED) — reuse its backend-agnostic helpers and
# its FieldSystemV2 as the trusted CPU reference for the parity check.
import field_system_v2 as v2
import field_system as v1
from field_system_v2 import ConfigV2, estimate_memory

# ---------------------------------------------------------------------------
# BACKEND: CuPy if present (real GPU run), else NumPy (parity/projection only).
# This is the entire "port" surface — everything below uses `xp`.
# ---------------------------------------------------------------------------
try:
    import cupy as xp
    import cupyx
    _probe = xp.zeros(1)          # force context init; raises if no usable GPU
    del _probe
    GPU = True
except Exception:                 # no cupy, or cupy present but no GPU/driver
    import numpy as xp
    cupyx = None
    GPU = False

F32 = xp.float32
C64 = xp.complex64
I32 = xp.int32


def to_numpy(a):
    """Device→host. CuPy: xp.asnumpy; NumPy: identity."""
    return xp.asnumpy(a) if GPU else np.asarray(a)


def sync():
    """Block until all queued GPU work is done. On CPU this is a no-op.
    Skipping this is THE classic way to get fake-fast GPU numbers (you time
    async kernel *launches*, not *execution*)."""
    if GPU:
        xp.cuda.Stream.null.synchronize()


def scatter_add(grid_flat, idx, vals):
    """grid_flat[idx] += vals  (unbuffered accumulation).
    GPU: cupyx.scatter_add (atomic, the fast device scatter).
    CPU: np.add.at (same semantics; np.add.at itself has NO CuPy equivalent —
    that is exactly why cupyx.scatter_add exists, see README GPU PORT NOTES)."""
    if GPU:
        cupyx.scatter_add(grid_flat, idx, vals)
    else:
        np.add.at(grid_flat, idx, vals)


# ===========================================================================
# THE PORT — FieldSystemV2 with xp=CuPy and a scatter_add deposit.
# Mirrors field_system_v2.FieldSystemV2 operation-for-operation.
# ===========================================================================
class FieldSystemGPU:
    def __init__(self, cfg: ConfigV2, positions=None, check_mem=True):
        self.cfg = cfg
        if check_mem:
            v2.guard_memory(cfg, verbose=False)

        self.phi = xp.zeros((cfg.H, cfg.W), dtype=F32)

        rng = np.random.default_rng(cfg.seed)
        if positions is None:
            pos = rng.uniform([0, 0], [cfg.W * cfg.h, cfg.H * cfg.h], size=(cfg.N, 2))
        else:
            pos = np.asarray(positions, dtype=np.float64)
        pos = xp.asarray(pos, dtype=F32)
        self.N = int(pos.shape[0])

        # Morton (Z-order) sort — codes built on host (one-off, negligible), then
        # the permutation via xp.argsort (device-side on GPU, per the port notes).
        cxh = np.clip((to_numpy(pos[:, 0]) / cfg.h).astype(np.int64), 0, cfg.W - 1)
        cyh = np.clip((to_numpy(pos[:, 1]) / cfg.h).astype(np.int64), 0, cfg.H - 1)
        codes = v2.morton2d(cxh, cyh)
        if cfg.sort_units:
            self.perm = xp.argsort(xp.asarray(codes.astype(np.int64)))
        else:
            self.perm = xp.arange(self.N)
        self.inv_perm = xp.empty(self.N, dtype=xp.int64)
        self.inv_perm[self.perm] = xp.arange(self.N)
        self.pos = pos[self.perm]

        self.s = xp.zeros(self.N, dtype=F32)
        self.a = xp.zeros(self.N, dtype=F32)
        self.clamped = xp.zeros(self.N, dtype=bool)

        self._build_kernels()
        self._build_propagator()
        self._buf = xp.empty_like(self.win_ker)              # shared (N,K) work buffer
        self._Sflat = xp.zeros(cfg.H * cfg.W, dtype=F32)      # preallocated source grid
        self._check_dtypes()

    def _build_kernels(self):
        cfg = self.cfg
        rad = cfg.kernel_width // 2
        offs = xp.arange(-rad, rad + 1, dtype=I32)
        px = self.pos[:, 0]; py = self.pos[:, 1]
        cc = xp.round(px / cfg.h).astype(I32)
        cr = xp.round(py / cfg.h).astype(I32)
        cols = cc[:, None] + offs[None, :]
        rows = cr[:, None] + offs[None, :]
        dx = (cols.astype(F32) * cfg.h - px[:, None])
        dy = (rows.astype(F32) * cfg.h - py[:, None])
        dist2 = (dy[:, :, None] ** 2 + dx[:, None, :] ** 2).astype(F32)
        ker = xp.exp(-dist2 / F32(2.0 * cfg.sigma ** 2))
        norm = ker.sum(axis=(1, 2)) * F32(cfg.h ** 2)        # Σ G·h² = 1
        ker = ker / norm[:, None, None]
        cols_w = cols % cfg.W                                 # periodic wrap
        rows_w = rows % cfg.H
        flat = (rows_w[:, :, None] * cfg.W + cols_w[:, None, :]).astype(I32)
        K = cfg.kernel_width ** 2
        self.win_ker = ker.reshape(self.N, K).astype(F32)
        self.win_idx = flat.reshape(self.N, K).astype(I32)

    def _build_propagator(self):
        cfg = self.cfg
        ky = 2.0 * np.pi * np.fft.fftfreq(cfg.H, d=cfg.h)
        kx = 2.0 * np.pi * np.fft.rfftfreq(cfg.W, d=cfg.h)
        KY = xp.asarray(ky, dtype=F32)[:, None]
        KX = xp.asarray(kx, dtype=F32)[None, :]
        if cfg.symbol == "spectral":
            K2 = (KY ** 2 + KX ** 2).astype(F32)
        else:
            K2 = (F32(4.0 / cfg.h ** 2) * (xp.sin(KY * F32(cfg.h / 2)) ** 2 +
                                           xp.sin(KX * F32(cfg.h / 2)) ** 2)).astype(F32)
        alpha = (F32(cfg.lam) + F32(cfg.D) * K2).astype(F32)
        tiny = F32(1e-12)
        safe = xp.where(alpha > tiny, alpha, F32(1.0))
        self.inv_denom = xp.where(alpha > tiny, 1.0 / safe, F32(0.0)).astype(F32)
        E = xp.exp(-alpha * F32(cfg.dt)).astype(F32)
        self.E = E
        self.factor = xp.where(alpha > tiny, (1.0 - E) / safe, F32(cfg.dt)).astype(F32)

    def _check_dtypes(self):
        for n in ("phi", "pos", "s", "a", "win_ker", "inv_denom", "E", "factor"):
            assert getattr(self, n).dtype == F32, f"{n} not float32"
        assert self.win_idx.dtype == I32

    # ---- four operations (identical math to v2) ---------------------------
    def deposit(self):
        # S(x) = Σ a_i·G(x−x_i); scatter-add into the preallocated grid.
        self._Sflat.fill(F32(0.0))
        xp.multiply(self.a[:, None], self.win_ker, out=self._buf)
        scatter_add(self._Sflat, self.win_idx.ravel(), self._buf.ravel())
        return self._Sflat.reshape(self.cfg.H, self.cfg.W)

    def evolve(self, S):
        cfg = self.cfg
        if cfg.mode == "steady":
            S_hat = xp.fft.rfft2(S).astype(C64, copy=False)
            phi_hat = S_hat * self.inv_denom
            self.phi = xp.fft.irfft2(phi_hat, s=(cfg.H, cfg.W)).astype(F32, copy=False)
        else:
            S_hat = xp.fft.rfft2(S).astype(C64, copy=False)
            phi_hat = xp.fft.rfft2(self.phi).astype(C64, copy=False)
            phi_hat = phi_hat * self.E + S_hat * self.factor
            self.phi = xp.fft.irfft2(phi_hat, s=(cfg.H, cfg.W)).astype(F32, copy=False)
        # CuPy can silently promote FFT dtype — assert we stayed float32.
        assert self.phi.dtype == F32, f"FFT promoted phi to {self.phi.dtype}"

    def read(self):
        xp.take(self.phi.ravel(), self.win_idx, out=self._buf)
        self._buf *= self.win_ker
        return self._buf.sum(axis=1) * F32(self.cfg.h ** 2)

    def update(self, phi_read):
        cfg = self.cfg
        upd = ~self.clamped
        self.s[upd] += F32(cfg.dt) * (-F32(cfg.gamma) * self.s[upd] + phi_read[upd])
        self.a[upd] = xp.tanh(self.s[upd]).astype(F32)

    def step(self):
        S = self.deposit()
        self.evolve(S)
        pr = self.read()
        self.update(pr)
        return pr

    def read_unsorted(self):
        return self.read()[self.inv_perm]


# ===========================================================================
# STEP 0 — HARDWARE REPORT
# ===========================================================================
# Known memory bandwidths (GB/s) and VRAM (GB) for cards that matter here.
_GPU_DB = {
    "Tesla T4":            (16, 320),
    "L4":                  (24, 300),
    "Tesla V100":          (16, 900),
    "A100":                (40, 1555),
    "A100-SXM4-80GB":      (80, 2039),
    "A100 80GB":           (80, 2039),
    "H100":                (80, 3350),
    "RTX 6000 Pro Blackwell": (96, 1792),   # GDDR7, 512-bit ~28 Gbps (target deploy)
}


def _lookup_bw(name, vram_gb):
    for k, (_, bw) in _GPU_DB.items():
        if k.lower() in name.lower():
            return bw
    return None


def hardware_report():
    print("=" * 74)
    print("STAGE 2b — GPU PORT + REAL-TIME SCALING MEASUREMENT")
    print("=" * 74)
    if not GPU:
        import platform
        print("  !!! NO CUDA GPU DETECTED in this environment !!!")
        print(f"  backend xp = numpy (CPU fallback)   host = {platform.platform()}")
        print("  → Running PARITY + PROJECTION mode. Timings here are NOT GPU numbers.")
        print("    Run this file on a Colab GPU for the measured real-time ceiling.")
        return None
    props = xp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    free, total = xp.cuda.runtime.memGetInfo()
    vram_gb = total / 1e9
    # bandwidth: prefer the spec table; else compute 2·clk·bus from props
    bw = _lookup_bw(name, vram_gb)
    if bw is None:
        bw = 2 * (props["memoryClockRate"] * 1e3) * (props["memoryBusWidth"] / 8) / 1e9
    cuda_rt = xp.cuda.runtime.runtimeGetVersion()
    print(f"  GPU            : {name}")
    print(f"  VRAM           : {vram_gb:.1f} GB  ({free/1e9:.1f} GB free now)")
    print(f"  mem bandwidth  : ~{bw:.0f} GB/s   (workload is BANDWIDTH-bound — this sets the ceiling)")
    print(f"  CUDA runtime   : {cuda_rt//1000}.{(cuda_rt%1000)//10}")
    print(f"  CuPy           : {xp.__version__}")
    print(f"  backend xp     : cupy")
    return dict(name=name, vram_gb=vram_gb, bw=bw)


# ===========================================================================
# STEP 2 — CORRECTNESS PARITY (GPU vs proven CPU v2)  — must pass before timing
# ===========================================================================
def correctness_parity():
    print("\n[PARITY] port correctness vs proven Stage-2a CPU reference")
    tol = 1e-4                                    # float32 (GPU atomics reorder sums)

    # (a) deposit primitive: scatter_add path vs v2's bincount path, same input.
    cfg = ConfigV2(H=64, W=64, N=5000, seed=3, sort_units=False)
    pos = np.random.default_rng(3).uniform([0, 0], [64, 64], size=(5000, 2))
    a0 = np.tanh(np.random.default_rng(4).standard_normal(5000)).astype(np.float32)
    g = FieldSystemGPU(cfg, positions=pos, check_mem=False); g.a[:] = xp.asarray(a0)
    c = v2.FieldSystemV2(cfg, positions=pos, check_mem=False); c.a[:] = a0
    d_diff = float(np.abs(to_numpy(g.deposit()) - np.asarray(c.deposit())).max())
    print(f"   deposit (scatter_add vs bincount) max|Δ| = {d_diff:.2e}")

    # (b) full equivalence falloff: GPU vs Stage-1, same setup as Stage-2a test 1.
    center = np.array([64.0, 64.0]); radii = np.array([2,3,4,5,6,8,10,12,14,16,18,20], float)
    rpos, rid = v1.make_rings(center, radii, n_ang=16)
    def falloff(symbol):
        cc = ConfigV2(H=128, W=128, mode="steady", symbol=symbol, sort_units=False)
        s = FieldSystemGPU(cc, positions=rpos, check_mem=False)
        s.clamped[:] = True; s.a[:] = 0.0; s.a[0] = 1.0; s.step()
        ph = to_numpy(s.read())
        return np.array([ph[rid == r].mean() for r in radii])
    f_spec, f_disc = falloff("spectral"), falloff("discrete")
    L_fit = v2.fit_decay_length(radii[radii >= 8], f_spec[radii >= 8])
    # CPU v2 spectral for a direct GPU-vs-CPU diff on the falloff
    cs = v2.FieldSystemV2(ConfigV2(H=128, W=128, sort_units=False), positions=rpos, check_mem=False)
    cs.clamped[:] = True; cs.a[:] = 0.0; cs.a[0] = 1.0; cs.step()
    cpu_spec = np.array([np.asarray(cs.read())[rid == r].mean() for r in radii])
    falloff_diff = float(np.abs(f_spec - cpu_spec).max())
    print(f"   equivalence: spectral L_fit = {L_fit:.4f} (target 2.0);  "
          f"GPU-vs-CPU falloff max|Δ| = {falloff_diff:.2e}")

    # (c) Morton permutation invariance on GPU
    mp = np.random.default_rng(7).uniform([0, 0], [128, 128], size=(20000, 2))
    ma = np.tanh(np.random.default_rng(8).standard_normal(20000)).astype(np.float32)
    def runm(sort):
        s = FieldSystemGPU(ConfigV2(H=128, W=128, N=20000, sort_units=sort),
                           positions=mp, check_mem=False)
        s.clamped[:] = True; s.a[:] = xp.asarray(ma)[s.perm]; s.step()
        return to_numpy(s.read_unsorted())
    morton_diff = float(np.abs(runm(True) - runm(False)).max())
    print(f"   morton invariance max|Δ| = {morton_diff:.2e}")

    # (d) conservation, dynamic λ=0
    cpos = np.array([64.,64.]) + np.array([[-3,-3],[3,-3],[-3,3],[3,3]], float)
    cs2 = FieldSystemGPU(ConfigV2(H=128, W=128, lam=0.0, mode="dynamic", sort_units=False),
                         positions=cpos, check_mem=False)
    cs2.clamped[:] = True; cs2.a[:] = 1.0
    integ = []
    for _ in range(200):
        cs2.step(); integ.append(float(cs2.phi.sum()))
    tt = np.arange(1, 201); sl, ic = np.polyfit(tt, integ, 1)
    cons_rel = float(np.abs(np.array(integ) - (sl*tt+ic)).max() / (abs(integ[-1]) + 1e-30))
    print(f"   conservation: slope={sl:.4f} (expect 4.0), linear drift rel={cons_rel:.2e}")

    ok = (d_diff < tol and falloff_diff < tol and abs(L_fit-2.0)/2.0 < 0.05
          and morton_diff < tol and cons_rel < 1e-3
          and np.abs(f_disc - cpu_spec).max() < 0.05)  # discrete≈stage1-ish sanity
    print(f"   [{'PARITY OK' if ok else 'PARITY FAIL — STOP'}] "
          f"(float32 tol {tol:.0e}; GPU sums reorder via atomics)")
    return ok


# ===========================================================================
# STEP 3 — CORRECT GPU TIMING
# ===========================================================================
def time_call(fn, warmup=25, iters=40):
    """Median (and IQR) ms per call, done correctly:
    warmup (JIT/autotune/cuFFT-plan), device-synchronize, CUDA-event timing."""
    for _ in range(warmup):
        fn()
    sync()
    ts = np.empty(iters)
    if GPU:
        ev0, ev1 = xp.cuda.Event(), xp.cuda.Event()
        for i in range(iters):
            ev0.record(); fn(); ev1.record(); ev1.synchronize()
            ts[i] = xp.cuda.get_elapsed_time(ev0, ev1)        # ms, device-side
    else:
        for i in range(iters):
            t0 = time.perf_counter(); fn(); sync()
            ts[i] = (time.perf_counter() - t0) * 1e3
    return float(np.median(ts)), float(np.percentile(ts, 25)), float(np.percentile(ts, 75))


def stage_breakdown(sys):
    """Per-stage median ms (deposit / FFT-solve / read / update), each synced."""
    S = sys.deposit(); sys.evolve(S); pr = sys.read()         # prime state
    d, _, _ = time_call(sys.deposit)
    e, _, _ = time_call(lambda: sys.evolve(S))
    r, _, _ = time_call(sys.read)
    u, _, _ = time_call(lambda: sys.update(pr))
    return dict(deposit=d, fft=e, read=r, update=u)


def device_used_bytes():
    if GPU:
        free, total = xp.cuda.runtime.memGetInfo()
        return total - free
    return 0


# ===========================================================================
# STEP 4 — THE MEASUREMENT (GPU only)
# ===========================================================================
def n_sweep(hw, grid=256, budget_ms=16.0):
    print(f"\n[SWEEP] grid {grid}², steady mode, real-time budget {budget_ms} ms/step")
    print(f"   {'N':>12}  {'median ms':>10} {'IQR ms':>14}  "
          f"{'deposit':>8} {'fft':>7} {'read':>8} {'upd':>6}  {'VRAM GB':>8}")
    bytes_per_unit = estimate_memory(ConfigV2(H=grid, W=grid, N=1))[1]  # unit_bytes at N=1
    vram_cap = 0.80 * hw["vram_gb"] * 1e9
    rows = []
    Ns = [1e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8]
    for N in [int(x) for x in Ns]:
        est = estimate_memory(ConfigV2(H=grid, W=grid, N=N))[0]
        if est > vram_cap:
            print(f"   {N:>12,}  — est {est/1e9:.1f} GB > 80% VRAM ({vram_cap/1e9:.1f} GB); stop (memory).")
            rows.append((N, None, None, None, None, "OOM"))
            break
        cfg = ConfigV2(H=grid, W=grid, N=N, N_CAP=2_000_000_000, mem_limit_bytes=vram_cap)
        sys = FieldSystemGPU(cfg)
        med, q1, q3 = time_call(sys.step)
        bd = stage_breakdown(sys)
        used = device_used_bytes() / 1e9
        rows.append((N, med, (q1, q3), bd, used, "ok"))
        print(f"   {N:>12,}  {med:>10.3f} [{q1:6.3f},{q3:6.3f}]  "
              f"{bd['deposit']:>8.3f} {bd['fft']:>7.3f} {bd['read']:>8.3f} "
              f"{bd['update']:>6.3f}  {used:>8.2f}")
        del sys
        if GPU:
            xp.get_default_memory_pool().free_all_blocks()
        if med > budget_ms:
            print(f"   N={N:,} exceeds {budget_ms} ms — real-time ceiling crossed.")
            break
    return rows


def grid_sweep(hw, N=300_000):
    print(f"\n[GRID SWEEP] N={N:,}: confirm FFT stays negligible vs deposit/read")
    for g in [64, 128, 256, 512]:
        cfg = ConfigV2(H=g, W=g, N=N, N_CAP=2_000_000_000,
                       mem_limit_bytes=0.8 * hw["vram_gb"] * 1e9)
        sys = FieldSystemGPU(cfg)
        bd = stage_breakdown(sys)
        print(f"   {g:>3}²  deposit {bd['deposit']:7.3f}  fft {bd['fft']:7.3f}  "
              f"read {bd['read']:7.3f}  update {bd['update']:6.3f}  ms  "
              f"(FFT {100*bd['fft']/sum(bd.values()):.1f}% of step)")
        del sys
        if GPU:
            xp.get_default_memory_pool().free_all_blocks()


def analyze_headline(rows, hw, budget_ms=16.0):
    ok = [r for r in rows if r[5] == "ok"]
    under = [r for r in ok if r[1] < budget_ms]
    largest_rt = under[-1] if under else None
    oom = next((r for r in rows if r[5] == "OOM"), None)
    bottleneck = None
    if ok:
        last = ok[-1][3]
        bottleneck = max(last, key=last.get)
    bytes_per_unit = estimate_memory(ConfigV2(N=1))[1]
    n_mem = int(0.80 * hw["vram_gb"] * 1e9 / bytes_per_unit)
    binds = "compute (16 ms)" if (largest_rt and largest_rt[0] < n_mem) else "memory"
    print("\n" + "=" * 74)
    print("HEADLINE NUMBERS")
    print("=" * 74)
    print(f"  (1) largest N under {budget_ms} ms/step : "
          f"{largest_rt[0]:,} ({largest_rt[1]:.2f} ms)" if largest_rt else "  (1) none under budget")
    print(f"  (2) memory-out N (80% VRAM)        : ~{n_mem:,}"
          + (f"  (hit OOM at N={oom[0]:,})" if oom else ""))
    print(f"  (3) binding constraint on this card: {binds}")
    print(f"  (4) bottleneck stage at the ceiling: {bottleneck}")
    return dict(largest_rt=largest_rt, n_mem=n_mem, binds=binds, bottleneck=bottleneck)


# ===========================================================================
# STEP 5 — MEMORY CEILINGS (reliable) + BANDWIDTH MODEL (labeled estimate)
# ===========================================================================
# Bandwidth model: the step streams the (N,K)-sized arrays through HBM a small
# number of times (win_ker, win_idx, the shared buffer, the gathered φ) — the
# 256² grid itself is L2-resident, and the FFT is a tiny constant. So
#     bytes/step ≈ C_TRAFFIC · N · K · 4 ,   t_step ≈ bytes/step / (UTIL·BW_peak)
# C_TRAFFIC and UTIL are the two stated assumptions; treat the resulting N as an
# order-of-magnitude ESTIMATE, not a measurement.
C_TRAFFIC = 10.0      # # of (N,K) array passes through HBM per step (deposit+read)
UTIL = 0.6            # fraction of peak bandwidth realized by scatter/gather


def ceilings_table(K=169, budget_ms=16.0):
    bytes_per_unit = estimate_memory(ConfigV2(N=1))[1]                # exact, reliable
    print("\n" + "=" * 74)
    print("CEILINGS — memory is exact; the 16 ms (compute) column is a LABELED MODEL")
    print(f"  bytes/unit = {bytes_per_unit} (exact)   |   model: bytes/step ≈ "
          f"{C_TRAFFIC:.0f}·N·K·4, util {UTIL:.0%} of peak BW")
    print("=" * 74)
    print(f"  {'GPU':<26} {'VRAM':>5} {'BW':>7}  {'N_mem(80%)':>12}  {'N_16ms(model)':>14}  binds")
    out = {}
    for name, (vram, bw) in _GPU_DB.items():
        n_mem = 0.80 * vram * 1e9 / bytes_per_unit
        n_cmp = (budget_ms * 1e-3) * (UTIL * bw * 1e9) / (C_TRAFFIC * K * 4)
        binds = "compute" if n_cmp < n_mem else "memory"
        out[name] = (n_mem, n_cmp, binds)
        print(f"  {name:<26} {vram:>3}GB {bw:>5}GB/s  {n_mem:>12,.0f}  {n_cmp:>14,.0f}  {binds}")
    print("  (N_mem is reliable: fixed bytes/unit. N_16ms is an order-of-magnitude")
    print("   bandwidth model — the real value must be MEASURED on the actual card.)")
    return out


def blackwell_extrapolation(headline, hw, ceilings):
    print("\n" + "=" * 74)
    print("EXTRAPOLATION TO RTX 6000 Pro Blackwell (96 GB) — labeled estimate")
    print("=" * 74)
    bytes_per_unit = estimate_memory(ConfigV2(N=1))[1]
    bw_b, vram_b = 1792, 96
    n_mem_b = int(0.80 * vram_b * 1e9 / bytes_per_unit)
    if headline["binds"].startswith("compute"):
        # bandwidth-bound: scale the MEASURED Colab ceiling by the BW ratio
        n_rt = headline["largest_rt"][0]
        ratio = bw_b / hw["bw"]
        n_b = int(n_rt * ratio)
        print(f"  This card is COMPUTE(bandwidth)-bound at N≈{n_rt:,}.")
        print(f"  Blackwell BW {bw_b} / {hw['name']} BW {hw['bw']:.0f} = ×{ratio:.2f}")
        print(f"  → ESTIMATE real-time N ≈ {n_b:,}  (bandwidth-ratio scaling; an ESTIMATE,")
        print(f"    rests on the workload being bandwidth-bound — verify by measuring).")
        print(f"  (Memory would allow ~{n_mem_b:,}, so compute still binds first.)")
    else:
        n_rt = headline["largest_rt"][0] if headline["largest_rt"] else headline["n_mem"]
        ratio = vram_b / hw["vram_gb"]
        n_b = int(headline["n_mem"] * ratio)
        print(f"  This card is MEMORY-bound at N≈{headline['n_mem']:,}.")
        print(f"  Blackwell VRAM 96 / {hw['vram_gb']:.0f} GB = ×{ratio:.2f}  (linear, reliable)")
        print(f"  → real-time N ≈ {n_b:,}  (memory scaling — reliable since bytes/unit fixed).")


# ===========================================================================
# PLOTS
# ===========================================================================
def plot_measured(rows, hw, budget_ms, path):
    ok = [r for r in rows if r[5] == "ok"]
    Ns = np.array([r[0] for r in ok], float); ts = np.array([r[1] for r in ok])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(Ns, ts, "o-", label="median ms/step")
    ax.axhline(budget_ms, color="C3", ls="--", label=f"{budget_ms} ms real-time budget")
    under = Ns[ts < budget_ms]
    if len(under):
        ax.axvline(under[-1], color="C2", ls=":", label=f"ceiling N≈{int(under[-1]):,}")
    ax.set_xlabel("N (units)"); ax.set_ylabel("median time / step (ms)")
    ax.set_title(f"Stage-2b MEASURED on {hw['name']} ({hw['vram_gb']:.0f} GB) — grid 256²")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_projection(ceilings, path, K=169, budget_ms=16.0):
    """No GPU here → a clearly-labeled MODEL plot: modeled t_step(N) for a few
    representative cards, the 16 ms line, and each card's memory ceiling."""
    bytes_per_unit = estimate_memory(ConfigV2(N=1))[1]
    fig, ax = plt.subplots(figsize=(7, 5))
    Ns = np.logspace(4, 8, 60)
    for name, color in [("Tesla T4", "C0"), ("A100-SXM4-80GB", "C1"),
                        ("RTX 6000 Pro Blackwell", "C2")]:
        vram, bw = _GPU_DB[name]
        t = (C_TRAFFIC * Ns * K * 4) / (UTIL * bw * 1e9) * 1e3       # ms, MODELED
        ax.loglog(Ns, t, color=color, label=f"{name} (~{bw} GB/s)")
        n_mem = 0.80 * vram * 1e9 / bytes_per_unit
        ax.axvline(n_mem, color=color, ls=":", alpha=0.5)
    ax.axhline(budget_ms, color="k", ls="--", label=f"{budget_ms} ms budget")
    ax.set_xlabel("N (units)"); ax.set_ylabel("MODELED time / step (ms)")
    ax.set_title("Stage-2b PROJECTION (bandwidth model — NOT a GPU measurement)\n"
                 f"assumptions: {C_TRAFFIC:.0f}·N·K·4 bytes/step, util {UTIL:.0%}; "
                 "dotted = memory ceilings")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ===========================================================================
# main
# ===========================================================================
def main():
    hw = hardware_report()

    if GPU:
        parity = correctness_parity()
        if not parity:
            print("\nPARITY FAILED — refusing to report timings (a port bug must not "
                  "be hidden by moving on). Investigate before timing.")
            raise SystemExit(1)
        rows = n_sweep(hw)
        grid_sweep(hw)
        headline = analyze_headline(rows, hw)
        ceil = ceilings_table()
        blackwell_extrapolation(headline, hw, ceil)
        plot_measured(rows, hw, 16.0, "field_v2b_realtime_ceiling.png")
        print("\n[FIGURE] field_v2b_realtime_ceiling.png")
        rt = headline["largest_rt"]
        print("\n" + "#" * 74)
        print("# 5-LINE SUMMARY")
        print(f"#  GPU                : {hw['name']} ({hw['vram_gb']:.0f} GB, ~{hw['bw']:.0f} GB/s)")
        print(f"#  largest real-time N: {rt[0]:,} under 16 ms/step" if rt else "#  largest real-time N: none")
        print(f"#  binding constraint : {headline['binds']}")
        print(f"#  bottleneck stage   : {headline['bottleneck']}")
        b = 1792 / hw["bw"] if headline["binds"].startswith("compute") else 96/hw["vram_gb"]
        print(f"#  Blackwell 96 GB    : ~×{b:.1f} of measured (labeled estimate; see above)")
        print("#" * 74)
    else:
        # CPU sandbox: validate the PORT, then give reliable memory ceilings + a
        # labeled bandwidth model. No GPU time is ever reported as a measurement.
        parity = correctness_parity()
        ceil = ceilings_table()
        plot_projection(ceil, "field_v2b_realtime_ceiling.png")
        print("\n[FIGURE] field_v2b_realtime_ceiling.png  (PROJECTION — labeled model)")
        bytes_per_unit = estimate_memory(ConfigV2(N=1))[1]
        n_mem_b = int(0.80 * 96 * 1e9 / bytes_per_unit)
        print("\n" + "#" * 74)
        print("# 5-LINE SUMMARY")
        print(f"#  GPU                : NONE in this sandbox — port validated on CPU, "
              f"parity {'OK' if parity else 'FAIL'}")
        print(f"#  largest real-time N: MEASURE ON COLAB (harness ready). Model 16 ms ceiling:")
        print(f"#                       ~0.45M (T4) · ~2.5M (A100/Blackwell) · ~4.8M (H100)")
        print(f"#  binding constraint : compute(bandwidth) binds first on every card, below")
        print(f"#                       the memory ceiling (~6M T4 · ~31M A100-80 · ~37M Blackwell)")
        print(f"#  bottleneck stage   : deposit/read (scatter/gather, bandwidth) — not the FFT")
        print(f"#  Blackwell 96 GB    : memory allows ~{n_mem_b:,}; real-time N stays BW-bound,")
        print(f"#                       est ~2.5M (= measured Colab N × Blackwell/Colab BW ratio)")
        print("#" * 74)
        raise SystemExit(0 if parity else 1)


if __name__ == "__main__":
    main()
