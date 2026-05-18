"""Phase 3: cell_potential field substrate + v3-learned short-range kernel.

Trains a small v3 ArtificialAtomPotential on the LJ reference, plugs it in
as the short-range force, runs all Phase 0+1+2 checks plus new Phase 3
gates for v3's invariances and the hybrid's behaviour.

  python -m cell_potential.demo_phase3
"""

from __future__ import annotations

import time

import torch

from .grid import Grid
from .verify_phase3 import format_results, run_all


def main() -> None:
    print("=" * 76)
    print("  Phase 3  field-particle hybrid: + v3-learned short-range kernel")
    print("=" * 76)
    torch.manual_seed(0)

    grid = Grid(N=32, L=10.0)
    print(f"\nGrid: {grid.N}^3 = {grid.N**3:,} cells, L = {grid.L}, h = {grid.h:.4f}")
    print("Long-range Coulomb via PIC + spectral Poisson.")
    print("Short-range via v3 ArtificialAtomPotential trained on LJ reference.\n")
    print("Training v3 (one-time cost, ~30 s on CPU)...")

    t0 = time.time()
    results = run_all(grid)
    print(f"\nPhase 3 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    n_total = len(results)
    print(f"\n  -> {n_pass} / {n_total} checks passed.")
    if n_pass == n_total:
        print()
        print("Phase 3 contract MET. The full architecture is in place:")
        print("  - electric field on a grid via spectral Poisson")
        print("  - diffusion field with particle sources via spectral diffusion")
        print("  - long-range Coulomb forces via PIC (field path)")
        print("  - short-range learned chemistry via v3 (autograd-conservative)")
        print("  - Newton's 3rd law, energy conservation, and stability all hold")
        print("    end-to-end on 100-particle dynamics.")
    else:
        print("\nPhase 3 contract NOT met. Investigate failures above.")


if __name__ == "__main__":
    main()
