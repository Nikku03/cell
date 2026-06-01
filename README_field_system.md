# Field-Mediated Neural System — Stage 1 reference (`field_system.py`)

Units never see each other; they only deposit into and read from a shared diffusion-with-decay field (`∂φ/∂t = D∇²φ − λφ + S`). Run with `python field_system.py` (NumPy + matplotlib only). What each acceptance test proves:

1. **STABILITY** — 2000 steps with all units emitting stay finite and bounded: tanh-capped emission + decay + the `Δt ≤ h²/(4D)` constraint make the explicit-Euler scheme stable (no blow-up, no NaNs).
2. **LOCALITY** — a single pinned unit's influence falls off as the 2D screened-diffusion Green's function `K0(r/L)`; the fitted length matches `L = sqrt(D/λ)` (2.15 vs 2.0), proving coupling is genuinely field-mediated and local.
3. **CAUSALITY** — a distant probe responds only after a delay that grows with distance (t₅₀% rise: 9→47 for r: 3→12), proving the field spreads at finite speed — no instantaneous action at a distance.
4. **CONSERVATION** — with `λ=0` and a constant source, `∫Φ` grows exactly linearly (drift ~1e-16), proving the Neumann (zero-flux) 5-point Laplacian neither creates nor leaks field (`Σ∇²Φ = 0`).

---

# Stage 2a — periodic / FFT / spatially-sorted (`field_system_v2.py`)

Same physics as Stage 1, re-engineered toward a GPU-ready, large-N implementation:
periodic (torus) boundaries, a **spectral field solve**, **Morton-sorted** units, and
**float32 everywhere**. CPU only, ≤10 GB; a memory estimate is printed and the run is
refused above 8 GB. Run with `python field_system_v2.py` (NumPy + matplotlib only).

### The FFT field solve
Periodic boundaries make the screened-diffusion operator diagonal in Fourier space.
The **steady** field (default `mode='steady'`) given source `S` solves `(λ − D∇²)φ = S`:

```
φ_hat = S_hat / (λ + D·|k|²)         φ = IFFT( FFT(S) / (λ + D·|k|²) )
```

One `rfft2` + one `irfft2` per step → **O(M log M)**, the *exact* steady field, with **no
CFL / time-step stability limit** (Stage 1's `Δt ≤ h²/(4D)` is gone). `|k|²` is the
continuum symbol `k_x²+k_y²` (`symbol='spectral'`, default); `symbol='discrete'` uses the
5-point stencil's symbol `(4/h²)[sin²(k_xh/2)+sin²(k_yh/2)]`, which reproduces Stage-1's
operator *exactly*. An unconditionally-stable **dynamic** mode (`mode='dynamic'`) integrates
`φ_hat ← φ_hat·e^{−αΔt} + S_hat·(1−e^{−αΔt})/α`, with `α = λ + D|k|²`.

### The k=0 zero-mode (the one numerical subtlety)
`α = λ + D|k|²` vanishes at `k=0` **iff λ=0** — pure diffusion has no steady state when there
is net source (the spatial mean `∫Φ` would diverge). We handle the DC mode explicitly:
- **Steady mode:** `1/(λ+D|k|²)` at `k=0` is set to `0` when `λ=0` — i.e. keep the zero-MEAN
  steady shape and drop the undetermined DC offset.
- **Dynamic mode:** the factor `(1−e^{−αΔt})/α` has the finite limit `Δt` as `α→0`, so the DC
  mode integrates as `φ_hat[0,0] += S_hat[0,0]·Δt` — exactly the linear `∫Φ` growth of an
  undamped conserved field. This is what the conservation test exercises.

### Morton (Z-order) ordering
Units are sorted once at init by the Morton code of their grid cell, so units adjacent in
memory are adjacent in space. On CPU this changes nothing numerically (and we *prove* it); the
payoff is on GPU, where deposit/read become coalesced/cache-local scatter–gather. An unsorted
mode exists precisely to show sorting is a **pure permutation**.

### What each test proves
1. **EQUIVALENCE TO STAGE 1** — one driven unit at the center of a 128² box (≫ L=2): the
   periodic-FFT steady falloff fits `L = 1.999` (target 2.0, **0.1%**), and with the matching
   `discrete` operator the central field equals the Stage-1 Neumann result to **0.00%**
   (~6 sig figs). Periodic+FFT did not change the physics where it shouldn't.
2. **MORTON PERMUTATION INVARIANCE** — sorted vs unsorted give bit-identical per-unit output
   (max abs diff `0.0e0` < 1e-5): the sort is a memory reorder, not a physics change.
3. **CONSERVATION (dynamic, λ=0)** — with the k=0 mode handled, `∫Φ` grows linearly, slope =
   `Σaᵢ` exactly, drift ~`8e-8` relative (float32). Same conservation law as Stage 1.

### Scaling shape (CPU measures the SHAPE; the 16 ms real-time ceiling is a GPU/Stage-2b job)
- **time/step vs N** (grid 256², N up to 2e6): asymptotic log-log slope **≈ 1.00** →
  deposit+read are **O(N)**. (Small-N points are FFT/Python-overhead- and cache-dominated, so
  the exponent is fit over N≥1e5.) See `field_v2_scaling_N.png`.
- **time/step vs M** (N=1e5, grid 64²→512²): the isolated FFT solve scales as **M·log M**
  (R² 0.9998 vs 0.983 for M²; exponent ≈0.98; a 4× increase in M costs **×4.4**, exactly
  M·logM's prediction, nowhere near M²'s ×16). See `field_v2_scaling_M.png`.

### GPU PORT NOTES (Stage 2b — structure is in place, nothing GPU runs here)
The array backend is isolated as the module-level alias **`xp`** (`import numpy as xp`).
Swapping to `import cupy as xp` is essentially the whole port; the heavy ops (`xp.bincount`,
`xp.take`, `xp.fft.rfft2/irfft2`, the propagator algebra) already go through `xp`. Specific
points to handle:
- **`xp.bincount` (deposit scatter-add):** NumPy returns a **float64** grid (we cast straight
  back to float32; the float64 temporary is grid-sized, not unit-sized). CuPy has
  `cupy.bincount`; the idiomatic GPU scatter is **`cupyx.scatter_add(grid, idx, vals)`**.
  Note: **`np.add.at` has *no* direct CuPy equivalent** — use `cupyx.scatter_add` there.
- **`xp.take(..., out=buf)` (read gather):** works on both; CuPy gathers are device-side.
  The preallocated shared `(N,K)` work buffer (`self._buf`) is already the GPU-preferred
  pattern (no per-step allocation).
- **FFT dtype:** NumPy `rfft2`/`irfft2` compute in double and return complex128/float64, so we
  `.astype(complex64)` / `.astype(float32)`. On CuPy `cupy.fft` honors float32 natively, so
  those casts become **no-ops** — leave them in (correct on both backends).
- **Morton sort:** built once at init via host `np.argsort` (negligible, one-time). Either keep
  it on host or switch to `xp.argsort`; it is never on the per-step path.
- **Host transfer:** `to_numpy()` is `np.asarray` here; for CuPy use `xp.asnumpy` (matplotlib
  and the asserts/fits need host arrays).

---

# Stage 2b — CuPy GPU port + real-time measurement (`field_system_v2_gpu.py`)

The GPU backend port of the proven Stage-2a code (Stage-2a file **left untouched**),
plus the measurement it exists for: **the largest N that runs under a 16 ms/step
real-time budget.** The physics, four operations, tests, and architecture are
unchanged — this is a backend swap + an honest hardware measurement.

### The port surface
`import cupy as xp` (the alias that made this a one-liner). Deposit uses
**`cupyx.scatter_add`** (NumPy's `np.add.at` has no fast CuPy equivalent); read uses
`xp.take`; the Morton sort uses `xp.argsort`; the **preallocated shared (N,K) buffer**
and a preallocated source grid are kept (per-step allocation is fatal on GPU). FFT
dtype is asserted float32/complex64 every step (CuPy can silently promote). The file
auto-detects the backend: **with a GPU it runs the real measurement; with no GPU it
falls back to NumPy in PARITY+PROJECTION mode and never prints a CPU time as a GPU
result.** Run: `python field_system_v2_gpu.py` (on Colab for real numbers).

### Correct GPU timing (where fake-fast numbers come from)
Timing is done properly: **≥25 warmup steps** (CUDA JIT/autotune, cuFFT plan cache),
**device `synchronize()`** before and after timing (CuPy is async — without sync you
time kernel *launches*, not *execution*), **CUDA-event** device-side timing,
**median over ≥40 steps** with the inter-quartile spread, and a **per-stage
breakdown** (deposit / FFT-solve / read / update) so the bottleneck at the ceiling is
visible. No host↔device array transfer ever happens inside the timed loop.

### Port correctness — validated on CPU (parity is checkable without a GPU)
The proven v2 is the reference. The new code paths reproduce it to float32:
`scatter_add` deposit vs v2's bincount **max|Δ| = 2.1e-7**; spectral falloff
**L_fit = 1.999**; Morton sorted-vs-unsorted **7.2e-7**; conservation slope **4.0000**,
drift **8e-8**. (GPU atomics reorder the scatter sums, so float32 — not bitwise —
agreement is the correct expectation; tolerance 1e-4.)

### The deliverable — measure it on Colab
**This build sandbox has no GPU**, so the four headline numbers (largest real-time N,
the OOM N, which constraint binds, the bottleneck stage) must come from running the
harness on a Colab GPU. What *is* defensible from first principles is below; the
harness fills in the measured values when run on hardware.

- **Memory ceiling (exact):** bytes/unit is fixed at **2060** (= 515 floats×4). At 80%
  VRAM that is **~6.2M (T4 16 GB) · ~31M (A100 80 GB) · ~37M (RTX 6000 Pro Blackwell
  96 GB).** Reliable — pure capacity arithmetic.
- **Compute ceiling (labeled bandwidth model):** the step streams the (N,K) arrays
  through HBM ~10× (`bytes/step ≈ 10·N·K·4`; the 256² grid is L2-resident, the FFT is
  a tiny constant), so `t_step ≈ 10·N·K·4 / (util·BW)`. With util≈60% the 16 ms
  crossing is **~0.45M (T4) · ~2.5M (A100/Blackwell) · ~4.8M (H100)** — an
  order-of-magnitude *estimate*, not a measurement. See
  `field_v2b_realtime_ceiling.png` (a clearly-labeled MODEL plot in this sandbox; the
  same script emits the MEASURED plot on a GPU).
- **Which binds first:** on every card above the model puts the compute (bandwidth)
  ceiling (~10⁶) **an order of magnitude below** the memory ceiling (~10⁷) — so the
  real-time-N is **bandwidth-bound**, and the bottleneck stage is **deposit/read**
  (scatter/gather), **not** the FFT. (This conclusion is robust to the model's
  assumptions: even at util 80% / 5 passes, compute still binds first.)
- **Blackwell (96 GB) extrapolation:** since the bound is *compute*, scale the
  *measured* Colab real-time-N by the **bandwidth ratio** `1792 / BW_colab` (labeled an
  estimate, resting on the workload being bandwidth-bound) — not by VRAM. Memory would
  allow ~37M, but compute binds first, est ~2.5M. If a run is ever memory-bound
  instead, scale by the VRAM ratio (96/VRAM_colab), which is reliable.

### Operating point (carry-forward, not fixed here)
Stage-1 showed units saturate `tanh` (mean|a|→1) under a strong field — bounded but
information-free. That is a **Stage-3 (learning)** concern; the model is unchanged
here. Timing is unaffected by operating point: `tanh` costs the same whether its input
is in the linear or saturated regime, so saturated vs linear units have identical
per-step cost.
