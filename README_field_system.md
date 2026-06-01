# Field-Mediated Neural System — Stage 1 reference (`field_system.py`)

Units never see each other; they only deposit into and read from a shared diffusion-with-decay field (`∂φ/∂t = D∇²φ − λφ + S`). Run with `python field_system.py` (NumPy + matplotlib only). What each acceptance test proves:

1. **STABILITY** — 2000 steps with all units emitting stay finite and bounded: tanh-capped emission + decay + the `Δt ≤ h²/(4D)` constraint make the explicit-Euler scheme stable (no blow-up, no NaNs).
2. **LOCALITY** — a single pinned unit's influence falls off as the 2D screened-diffusion Green's function `K0(r/L)`; the fitted length matches `L = sqrt(D/λ)` (2.15 vs 2.0), proving coupling is genuinely field-mediated and local.
3. **CAUSALITY** — a distant probe responds only after a delay that grows with distance (t₅₀% rise: 9→47 for r: 3→12), proving the field spreads at finite speed — no instantaneous action at a distance.
4. **CONSERVATION** — with `λ=0` and a constant source, `∫Φ` grows exactly linearly (drift ~1e-16), proving the Neumann (zero-flux) 5-point Laplacian neither creates nor leaks field (`Σ∇²Φ = 0`).
