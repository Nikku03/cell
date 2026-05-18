"""Phase 2: field long-range + direct short-range, 100 particles.

Runs every Phase 0 + Phase 1 check, plus new gates for the
short-range Lennard-Jones, hybrid force assembly, radial scaling,
and many-body stability.

  python -m cell_potential.demo_phase2
"""

from __future__ import annotations

import time

import torch

from .grid import Grid
from .verify_phase2 import format_results, run_all


def main() -> None:
    print("=" * 76)
    print("  Phase 2  field-particle hybrid: + short-range LJ on 100 particles")
    print("=" * 76)
    torch.manual_seed(0)

    grid = Grid(N=32, L=10.0)
    print(f"\nGrid: {grid.N}^3 = {grid.N**3:,} cells, L = {grid.L}, h = {grid.h:.4f}")
    print("Long-range Coulomb via PIC + spectral Poisson.")
    print("Short-range Lennard-Jones via direct pair sum (placeholder for v3 kernel).\n")

    t0 = time.time()
    results = run_all(grid)
    print(f"Phase 2 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    n_total = len(results)
    print(f"\n  -> {n_pass} / {n_total} checks passed.")
    if n_pass == n_total:
        print("\nPhase 2 contract MET. Field + particle hybrid produces correct")
        print("dynamics from 1 particle to 100, with both long-range Coulomb and")
        print("short-range LJ composing cleanly. The architecture is validated.")
        print("Next step: swap the LJ short-range with a v3-class learned kernel.")
    else:
        print("\nPhase 2 contract NOT met. Resolve failures above.")


if __name__ == "__main__":
    main()
