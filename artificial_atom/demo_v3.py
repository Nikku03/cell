"""End-to-end demo for the v3 (real-chemistry, pose-quality) potential.

Trains on MMFF94-labeled drug-like molecules from RDKit, with a pose-quality
classification head learned alongside energy and forces, then runs the v3
verification contract.

Run: python -m artificial_atom.demo_v3
"""

from __future__ import annotations

import time

import torch

from .mmff_dataset import chem_dataset_stats, generate_chem_dataset
from .multimode_v3 import MultiModePotentialV3, MultiModeV3Config
from .train_v3 import TrainV3Config, evaluate, train_v3
from .verify_v3 import format_results, run_all_checks


def main() -> None:
    print("=" * 76)
    print("  v3 artificial-atom potential: RDKit/MMFF94 chemistry + pose head")
    print("=" * 76)
    torch.manual_seed(0)

    print("\n[1] Building real-chemistry dataset from RDKit + MMFF94 ...")
    t0 = time.time()
    all_data = generate_chem_dataset(seed=0)
    print(f"  {len(all_data)} samples in {time.time()-t0:.1f}s")
    s = chem_dataset_stats(all_data)
    print(f"  E mean={s['energy_mean']:+.0f} std={s['energy_std']:.0f}  "
          f"force_rms={s['force_rms']:.0f}")
    print(f"  labels: near-native={s['n_near_native']}, "
          f"intermediate={s['n_intermediate']}, bad={s['n_bad']}")
    print(f"  unique SMILES used: {s['unique_smiles']}")

    # Split: stratify a bit by mixing pose labels evenly
    rng = torch.Generator().manual_seed(1)
    idx = torch.randperm(len(all_data), generator=rng).tolist()
    n_train = int(0.70 * len(all_data))
    n_val = int(0.15 * len(all_data))
    train_set = [all_data[i] for i in idx[:n_train]]
    val_set = [all_data[i] for i in idx[n_train:n_train + n_val]]
    test_set = [all_data[i] for i in idx[n_train + n_val:]]
    print(f"  split: train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    # ---------- 2. Build & train ----------
    print("\n[2] Building MultiModePotentialV3 (bonded/non-bond/electro channels"
          " + GRU + pose head) ...")
    cfg = MultiModeV3Config(d_h=40, d_rbf=20, n_layers=2, r_cutoff=6.0, r_cutoff_electro=8.0)
    model = MultiModePotentialV3(cfg)
    print(f"  d_h=40, d_rbf=20, 2 layers x 3 modes, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    print("\n[3] Training (energy + force + pose-quality BCE) ...")
    train_v3(
        model, train_set, val_set,
        TrainV3Config(epochs=18, lr=3e-4, batch_size=8, log_every=3,
                      force_weight=0.02, energy_weight=1.0, pose_weight=0.5),
        seed=0,
    )

    # ---------- 4. Mini ensemble for uncertainty ----------
    print("\n[4] Training a 2nd ensemble member (for uncertainty check) ...")
    model_b = MultiModePotentialV3(cfg)
    torch.manual_seed(13)
    train_v3(
        model_b, train_set, val_set,
        TrainV3Config(epochs=12, lr=3e-4, batch_size=8, log_every=5,
                      force_weight=0.02, energy_weight=1.0, pose_weight=0.5),
        seed=13,
    )

    # ---------- 5. Verification ----------
    print("\n[5] v3 verification suite (the contract):")
    results = run_all_checks(
        model, test_set,
        ensemble_members=[model, model_b],
        ood_pool=val_set,
    )
    print(format_results(results))
    passed = sum(r.passed for r in results)
    print(f"\n  -> {passed} / {len(results)} checks passed.")

    # ---------- 6. Sample inference ----------
    print("\n[6] Sample predictions on held-out test molecules:")
    print(f"  {'SMILES':<32}  {'E_true':>8}  {'E_pred':>8}  {'pose_lbl':>8}  {'pose_p':>8}")
    print("  " + "-" * 80)
    seen = set()
    shown = 0
    for c in test_set:
        if c.smiles in seen or c.pose_label == -1:
            continue
        seen.add(c.smiles)
        from .train_v3 import sample_kwargs
        kw = sample_kwargs(c)
        pos = kw.pop("positions").clone()
        E, _, logit = model.forward(pos, **kw)
        prob = torch.sigmoid(logit).item()
        lbl = "near" if c.pose_label == 1 else "bad"
        print(f"  {c.smiles:<32}  {c.energy.item():>8.1f}  {E.item():>8.1f}  "
              f"{lbl:>8}  {prob:>8.3f}")
        shown += 1
        if shown >= 10:
            break

    if passed == len(results):
        print("\nv3 contract met: real-chemistry features, learned MMFF surface, "
              "pose-quality classifier, and uncertainty all in one model.")


if __name__ == "__main__":
    main()
