"""Phase 1: electric field + diffusing concentration field + particle sources.

Re-runs every Phase 0 check (regression guard) and adds four new gates
for the diffusion machinery and particle-source coupling.

  python -m cell_potential.demo_phase1
"""

from __future__ import annotations

import time

import torch

from .grid import Grid
from .verify_phase1 import format_results, run_all


def main() -> None:
    print("=" * 76)
    print("  Phase 1  field-particle hybrid: + diffusing concentration field")
    print("=" * 76)
    torch.manual_seed(0)

    grid = Grid(N=32, L=10.0)
    print(f"\nGrid: {grid.N}^3 = {grid.N**3:,} cells, L = {grid.L}, h = {grid.h:.4f}")
    print("Spectral Poisson + spectral diffusion (exact-in-mode) + CIC + interpolation.\n")

    t0 = time.time()
    results = run_all(grid)
    print(f"Phase 1 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    n_total = len(results)
    print(f"\n  -> {n_pass} / {n_total} checks passed.")
    if n_pass == n_total:
        print("\nPhase 1 contract MET. Concentration field + particle sources work.")
        print("Ready for Phase 2 (short-range + long-range hybrid).")
    else:
        print("\nPhase 1 contract NOT met. Resolve failures above before Phase 2.")


if __name__ == "__main__":
    main()
