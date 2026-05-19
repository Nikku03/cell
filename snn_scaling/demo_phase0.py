"""Phase 0 demo: vectorised LIF + sparse synapses.

End-to-end runner for the foundation. Verifies all 9 gates and prints
the final summary. After this passes, every subsequent phase builds
on this substrate.
"""

from __future__ import annotations

import time

import torch

from .verify_phase0 import format_results, run_all


def main() -> None:
    print("=" * 76)
    print("  snn_scaling Phase 0: tensor-vectorised LIF + sparse synapses")
    print("=" * 76)
    torch.manual_seed(0)
    t0 = time.time()
    results = run_all()
    print(f"\nPhase 0 verification ({time.time()-t0:.1f}s total):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    print(f"\n  -> {n_pass} / {len(results)} checks passed.")
    if n_pass == len(results):
        print()
        print("Phase 0 contract MET. The vectorised SNN substrate works:")
        print("  - matches bio_neural_net's dynamics at small scale")
        print("  - 12x+ faster than pure-Python on the same task")
        print("  - scales to 100k neurons via degree-constant connectivity")
        print()
        print("Ready for the seven scaling tricks.")


if __name__ == "__main__":
    main()
