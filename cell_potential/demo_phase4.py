"""Phase 4: three extensions on top of the Phase 3 hybrid.

  1. PBC-aware v3 distances (minimum-image)
  2. v3.2 (MultiModePotentialV3) wired as an alternate short-range kernel
  3. FastPairKernel distilled from v3 for ~2x runtime speedup

Re-runs every Phase 0/1/2/3 gate (regression guard) and adds Phase 4
gates for each extension.

  python -m cell_potential.demo_phase4
"""

from __future__ import annotations

import time

import torch

from .grid import Grid
from .verify_phase4 import format_results, run_all


def main() -> None:
    print("=" * 76)
    print("  Phase 4  field-particle hybrid: + PBC, v3.2, FastPairKernel")
    print("=" * 76)
    torch.manual_seed(0)

    grid = Grid(N=32, L=10.0)
    print(f"\nGrid: {grid.N}^3 = {grid.N**3:,} cells, L = {grid.L}, h = {grid.h:.4f}")
    print("Training v3 (~30 s), v3.2 (~30 s), and distilling FastPairKernel (~15 s).\n")

    t0 = time.time()
    results = run_all(grid)
    print(f"\nPhase 4 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    n_total = len(results)
    print(f"\n  -> {n_pass} / {n_total} checks passed.")
    if n_pass == n_total:
        print()
        print("Phase 4 contract MET. The hybrid now supports:")
        print("  - PBC-aware distances in v3 (wrapping shifts preserve forces)")
        print("  - v3.2 (multi-channel, chemistry-aware) as an alternate kernel")
        print("  - FastPairKernel: tiny pair-MLP distilled from v3, ~2x faster")
        print()
        print("Architecture ready for: distillation at scale, multi-organism")
        print("training, and integration with cell_sim's reaction network.")
    else:
        print("\nPhase 4 contract NOT met. Investigate failures above.")


if __name__ == "__main__":
    main()
