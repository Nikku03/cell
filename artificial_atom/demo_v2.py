"""End-to-end demo of the multi-mode stateful potential + ensemble.

Run: python -m artificial_atom.demo_v2
"""

from __future__ import annotations

import time

import torch

from .dataset import generate_rich_dataset, rich_dataset_stats
from .multimode import MultiModeConfig, MultiModePotential
from .train_v2 import TrainV2Config, evaluate, train_one
from .uncertainty import EnsembleConfig, PotentialEnsemble
from .verify_v2 import format_results, run_all_checks


def main() -> None:
    print("=" * 76)
    print("  Multi-mode stateful artificial-atom potential: end-to-end")
    print("=" * 76)
    torch.manual_seed(0)

    print("\n[1] Generating bond-aware dataset (Morse for bonded pairs, LJ otherwise) ...")
    t0 = time.time()
    train_set = generate_rich_dataset(n_configs=300, seed=0)
    val_set = generate_rich_dataset(n_configs=80, seed=1)
    test_set = generate_rich_dataset(n_configs=80, seed=2)
    print(f"  train={len(train_set)} val={len(val_set)} test={len(test_set)} "
          f"in {time.time()-t0:.1f}s")
    s = rich_dataset_stats(train_set)
    print(f"  energy: mean={s['energy_mean']:+.2f}  std={s['energy_std']:.2f}  "
          f"force_rms={s['force_rms']:.2f}  "
          f"avg bonds/config={s['bonds_per_config_mean']:.2f}")

    # ---------- 2. Single model ----------
    print("\n[2] Building MultiModePotential (3 message channels + GRU + chem context)...")
    base_cfg = MultiModeConfig(d_h=48, d_rbf=20, n_layers=2, r_cutoff=6.0)
    model = MultiModePotential(base_cfg)
    print(f"  d_h=48, d_rbf=20, 2 layers x 3 modes; "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")

    print("\n[3] Training the primary model (40 epochs) ...")
    train_one(
        model, train_set, val_set,
        TrainV2Config(epochs=40, lr=3e-3, batch_size=8, log_every=5),
        seed=0,
    )

    # ---------- 4. Ensemble for uncertainty ----------
    print("\n[4] Training a 3-member ensemble (for uncertainty) ...")
    ens = PotentialEnsemble(EnsembleConfig(
        n_members=3, base_cfg=base_cfg, seeds=(0, 1, 2),
    ))
    # The first member is the model we already trained
    ens.members[0] = model
    for k, m in enumerate(ens.members[1:], start=1):
        print(f"  member {k+1} ...")
        train_one(
            m, train_set, val_set,
            TrainV2Config(epochs=25, lr=3e-3, batch_size=8, log_every=10),
            seed=k,
            quiet=False,
        )
    print(f"  {ens.cfg.n_members} members trained, "
          f"{ens.parameters_per_member():,} params each.")

    # ---------- 5. Verification suite ----------
    print("\n[5] Verification suite (the contract):")
    results = run_all_checks(model, test_set, ensemble=ens, train_set_for_ood=val_set)
    print(format_results(results))
    n_passed = sum(r.passed for r in results)
    print(f"\n  -> {n_passed} / {len(results)} checks passed.")

    # ---------- 6. Sample mode decomposition + uncertainty on one config ----------
    print("\n[6] Sample inference on one validation config (with ensemble):")
    c = val_set[0]
    sample = {
        "positions": c.positions, "Z": c.Z, "hybridisation": c.hybridisation,
        "formal_charge": c.formal_charge, "aromatic": c.aromatic,
        "in_ring": c.in_ring, "h_donor": c.h_donor, "h_acceptor": c.h_acceptor,
        "partial_charge": c.partial_charge,
        "bonds": c.bonds, "bond_type": c.bond_type,
    }
    out = ens.predict(sample)
    pos = c.positions.clone()
    kw = {k: v for k, v in sample.items() if k != "positions"}
    _, _, breakdown = model.forward(pos, **kw, return_breakdown=True)
    print(f"  ground truth E = {c.energy.item():+.4f}")
    print(f"  ensemble  mean E = {out['energy_mean']:+.4f}  +/- {out['energy_std']:.4f}")
    print(f"  mode decomposition (per-mode raw sums before global scale/shift):")
    print(f"    bonded   = {breakdown['bonded'].item():+.4f}")
    print(f"    nonbond  = {breakdown['nonbond'].item():+.4f}")
    print(f"    electro  = {breakdown['electro'].item():+.4f}")

    if n_passed == len(results):
        print("\nAll v2 checks pass. The unit now carries persistent state, chemistry")
        print("context, bond awareness, partial-charge channel, mode-decomposed energy,")
        print("and ensemble uncertainty - while preserving every v1 symmetry.")


if __name__ == "__main__":
    main()
