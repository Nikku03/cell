"""Phase 0: one electric field + ~10 particles.

Run end-to-end with the full verification suite. No moving on to Phase 1
until every check passes.

  python -m cell_potential.demo_phase0
"""

from __future__ import annotations

import time

import torch

from .grid import Grid
from .verify_phase0 import format_results, run_all


def main() -> None:
    print("=" * 76)
    print("  Phase 0  field-particle hybrid: one electric field + N particles")
    print("=" * 76)
    torch.manual_seed(0)

    grid = Grid(N=32, L=10.0)
    print(f"\nGrid: {grid.N}^3 = {grid.N**3:,} cells, L = {grid.L}, h = {grid.h:.4f}")
    print(f"Spectral Poisson + spectral gradient + CIC deposition/interpolation.\n")

    t0 = time.time()
    results = run_all(grid)
    print(f"Phase 0 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    n_total = len(results)
    print(f"\n  -> {n_pass} / {n_total} checks passed.")
    if n_pass == n_total:
        print("\nPhase 0 contract MET. Field + particle coupling is correct.")
        print("Ready to add a diffusing concentration field (Phase 1).")
    else:
        print("\nPhase 0 contract NOT met. Resolve the failures above before Phase 1.")


if __name__ == "__main__":
    main()
