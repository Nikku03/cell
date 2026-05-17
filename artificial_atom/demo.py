"""End-to-end demonstration of the artificial-atom potential.

  1. Generate a Lennard-Jones reference dataset (analytic energies & forces).
  2. Build the SchNet-style network.
  3. Train (joint energy + force loss).
  4. Run the full verification suite (accuracy + symmetries + relaxation).
  5. Show one relaxation trajectory: random start -> settled near the
     known LJ minimum.

Run: python -m artificial_atom.demo
"""

from __future__ import annotations

import time

import torch

from .dataset import dataset_stats, generate_dataset
from .network import ArtificialAtomPotential, PotentialConfig
from .physics import LennardJonesReference, pair_params
from .relax import RelaxConfig, relax
from .train import TrainConfig, evaluate, train
from .verify import format_results, run_all_checks


def main() -> None:
    print("=" * 72)
    print("  Artificial-atom MLIP: SchNet-style network, LJ reference, end-to-end")
    print("=" * 72)
    torch.manual_seed(0)

    # ---------- 1. Data ----------
    print("\n[1] Generating analytic LJ dataset (two element types: C and O) ...")
    t0 = time.time()
    train_set = generate_dataset(n_configs=600, seed=0)
    val_set = generate_dataset(n_configs=120, seed=1)
    test_set = generate_dataset(n_configs=120, seed=2)
    print(f"  generated train={len(train_set)} val={len(val_set)} test={len(test_set)} "
          f"in {time.time()-t0:.1f}s")
    s = dataset_stats(train_set)
    print(f"  energy: mean={s['energy_mean']:+.2f}  std={s['energy_std']:.2f}  "
          f"range=[{s['energy_min']:+.2f}, {s['energy_max']:+.2f}]  "
          f"force_rms={s['force_rms']:.2f}")

    # ---------- 2. Model ----------
    print("\n[2] Building SchNet-style potential ...")
    model = ArtificialAtomPotential(PotentialConfig(
        d_h=64, d_rbf=24, n_layers=3, r_cutoff=6.0,
    ))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  d_h=64, d_rbf=24, 3 message-passing layers, r_cut=6 A  "
          f"({n_params:,} parameters)")

    # ---------- 3. Train ----------
    print("\n[3] Training ...")
    train(
        model, train_set, val_set,
        TrainConfig(epochs=80, lr=3e-3, batch_size=8, force_weight=1.0, log_every=10),
    )

    # ---------- 4. Verify ----------
    print("\n[4] Verification suite (test set; this is the contract):")
    results = run_all_checks(model, test_set)
    print(format_results(results))
    n_passed = sum(r.passed for r in results)
    n_total = len(results)
    print(f"\n  -> {n_passed} / {n_total} checks passed.")

    # ---------- 5. Relaxation trajectory ----------
    print("\n[5] Sample relaxation trajectory (C-C dimer, learned potential):")
    sigma_cc, eps_cc = pair_params(6, 6)
    r_target = sigma_cc * 2 ** (1 / 6)
    print(f"  target equilibrium: r = sigma*2^(1/6) = {sigma_cc:.2f} * 2^(1/6) "
          f"= {r_target:.3f} A,  V = -eps = -{eps_cc:.2f}")
    pos = torch.tensor([[-1.7, 0.0, 0.0], [1.7, 0.0, 0.0]])   # r = 3.4 A
    Z = torch.tensor([6, 6], dtype=torch.long)
    pos_relaxed, e_hist, f_hist = relax(model, pos, Z, RelaxConfig(max_steps=400, step=0.02))
    r_final = torch.linalg.norm(pos_relaxed[1] - pos_relaxed[0]).item()
    print(f"  initial r=3.40 A,  final r={r_final:.3f} A,  "
          f"|final - target| = {abs(r_final - r_target):.3f} A")
    print(f"  energy trajectory (sampled): "
          f"{[round(e, 3) for e in e_hist[::max(1, len(e_hist)//8)]]}")
    print(f"  max-force trajectory (sampled): "
          f"{[round(f, 3) for f in f_hist[::max(1, len(f_hist)//8)]]}")

    if n_passed == n_total:
        print("\nAll checks pass. The learned potential reproduces the LJ reference")
        print("in energies, in forces, in symmetries, and in equilibrium geometry.")
    else:
        print("\nSome checks failed (see FAIL lines above).")


if __name__ == "__main__":
    main()
