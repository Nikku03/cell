"""
DMVC P8e: Attempted fix for P8c's training instability and error rate.

P8c training trajectory:
    epoch 0:  train=4.0e+06  val=4.2e+03     <- huge initial loss
    epoch 10: train=2.7e+04  val=1.8e+03     <- still huge
    epoch 20: train=2.7e-01  val=3.8e-01     <- finally stabilizing
    epoch 30: train=2.6e+01                   <- swings back up
    epoch 40: train=3.8e-01  val=2.2e-01     <- again
    epoch 50: train=5.7e+00  val=3.5e-01     <- spike
    epoch 60: train=1.6e-01                   <- finally stable
    epoch 70: train=1.5e-01  val=1.5e-01

The jagged trajectory with 6-7 order-of-magnitude swings suggests a few
pathological batches periodically overwhelm the gradient. Three targeted
interventions:

  Fix 1 — Gradient clipping (clip_grad_norm_ at max_norm=1.0):
    Prevents any single batch from overwriting accumulated learning.
    Cost: ~0 compute, trivial code.

  Fix 2 — Warmup learning rate:
    Linear ramp from lr=0 to lr=1e-3 over the first 5 epochs, then
    cosine decay as before. The initial loss of 4e+06 suggests the
    model is far from any sensible parameter regime at init; a full-lr
    Adam step at that starting point is destructive. Warmup lets the
    parameters migrate gently before aggressive optimization.

  Fix 3 — Clamp log-space targets:
    y_norm currently uses log1p(|r|/s) where s is the per-reaction median
    and r is the per-sample rate. If a sample has |r|/s = 1000, then
    y_norm = log(1001) = 6.9, which is fine. But if a rare sample has
    |r|/s = 1e6, then y_norm = 13.8 - huge, dominates MSE loss, and the
    gradient it produces is the source of the instability.
    Fix: clamp |y_norm| at 3.0 (corresponds to |r|/s ≈ 20, i.e. a sample
    20x larger than the reaction's median rate gets treated as 20x and no
    more). This is a controlled loss of information on extreme samples
    in exchange for training stability.

All three fixes are additive and cheap. Same architecture (hidden=512,
embed=16), same dataset (300 samples/rxn, 80 epochs).

Pre-registration:
  * If training is smoother (max/min loss ratio drops), interventions
    are doing something.
  * If median error drops meaningfully below 49.84%, we've actually
    improved rather than just smoothing.
  * If error doesn't improve, the 50% floor is architectural or data-
    driven, not optimization-driven.
"""

from __future__ import annotations
import sys, time
sys.path.insert(0, "/home/claude/dmvc")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prototype_p8b_full_syn3a import (
    MAX_PARTICIPANTS, EMB_DIM,
    encode_reaction_participants_with_idx,
    build_dataset_v2, y_norm_to_rate,
)
from prototype_p8c_scaled import (
    PermInvRateNetV2_Large,
)
from prototype_p3b_stamps import (
    load_sbml_model, extract_molecules_with_compartments, extract_reactions,
    find_h_and_water_per_compartment, rebalance_reaction, SBML_PATH,
)
from prototype_p4_kinetics import parse_all_kinetics


# y_norm clamp threshold (in log-space units).
# |y_norm| = 3.0 corresponds to |rate|/scale ≈ exp(3) - 1 = 19.1
# So a sample up to ~20x the reaction's median scale gets full gradient;
# anything beyond is clamped. This caps per-sample loss contribution.
YNORM_CLAMP = 3.0


def build_dataset_clamped(rxns_kin_list, species, init_concs,
                            n_samples_per_rxn, perturb_scale=3.0, seed=0,
                            per_rxn_scales=None):
    """Same as build_dataset_v2, but clamps y_norm to prevent extreme-sample
    loss dominance during training. Raw rates unchanged -- clamp is only
    applied to the training target, so we keep ground truth for evaluation."""
    d = build_dataset_v2(rxns_kin_list, species, init_concs,
                           n_samples_per_rxn=n_samples_per_rxn,
                           perturb_scale=perturb_scale, seed=seed,
                           per_rxn_scales=per_rxn_scales)
    # Track how many samples get clamped (for reporting)
    y_orig = d['y_norm']
    n_clamped = int((np.abs(y_orig) > YNORM_CLAMP).sum())
    d['y_norm'] = np.clip(y_orig, -YNORM_CLAMP, YNORM_CLAMP).astype(np.float32)
    d['n_clamped'] = n_clamped
    return d


def train_stable(model, train_data, val_data, n_epochs=80, batch_size=1024,
                   lr=1e-3, warmup_epochs=5, grad_clip=1.0, device=None):
    """P8c's training loop + 3 stability fixes: clipping, warmup, clamped y."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  (training on device: {device})")
    model = model.to(device)

    def to_tensors(d):
        return (torch.tensor(d['participants']).to(device),
                torch.tensor(d['indices']).to(device),
                torch.tensor(d['masks']).to(device),
                torch.tensor(d['globals']).to(device),
                torch.tensor(d['y_norm']).to(device))
    Ptr, Itr, Mtr, Gtr, Ytr = to_tensors(train_data)
    Pva, Iva, Mva, Gva, Yva = to_tensors(val_data)
    ds = TensorDataset(Ptr, Itr, Mtr, Gtr, Ytr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Warmup + cosine schedule (via lambda LR)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # cosine decay from 1.0 at warmup_epochs to ~0 at n_epochs
        progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    hist = {'train': [], 'val': [], 'lr': [], 'grad_norm': []}
    best_val = float('inf')

    for epoch in range(n_epochs):
        model.train()
        losses = []
        grad_norms = []
        for pb, ib, mb, gb, yb in loader:
            pred = model(pb, ib, mb, gb)
            loss = (pred - yb).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            # Gradient clipping
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norms.append(float(gn))
            opt.step()
            losses.append(loss.item())
        sched.step()
        tl = float(np.mean(losses))
        gn_avg = float(np.mean(grad_norms))
        current_lr = opt.param_groups[0]['lr']

        model.eval()
        with torch.no_grad():
            pred_va = model(Pva, Iva, Mva, Gva)
            vl = (pred_va - Yva).pow(2).mean().item()
        hist['train'].append(tl); hist['val'].append(vl)
        hist['lr'].append(current_lr); hist['grad_norm'].append(gn_avg)
        if vl < best_val: best_val = vl
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}  lr={current_lr:.2e}  "
                  f"train={tl:.4e}  val={vl:.4e}  "
                  f"grad_norm={gn_avg:.2e}  best={best_val:.4e}")
    return hist


def main():
    print("=" * 76)
    print("DMVC P8e: P8c + training stability fixes")
    print("  * gradient clipping (max_norm=1.0)")
    print("  * learning rate warmup (5 epochs linear)")
    print(f"  * y_norm clamp at {YNORM_CLAMP} (stops extreme samples from")
    print("     dominating the loss)")
    print("=" * 76)

    # ---- Setup (same as P8c) ----
    print("\n[1] Loading Syn3A...")
    sbml = load_sbml_model(SBML_PATH)
    mols = extract_molecules_with_compartments(sbml)
    rxns = extract_reactions(sbml)
    p_ids, w_ids = find_h_and_water_per_compartment(mols)
    rebalanced = []
    for r in rxns:
        if r.is_exchange or r.is_biomass: continue
        nr = rebalance_reaction(r, mols, p_ids, w_ids)
        if nr: rebalanced.append(nr)
    kinetics, init_concs = parse_all_kinetics()
    active = [(r.sbml_id, r, kinetics[r.sbml_id]) for r in rebalanced
               if r.sbml_id in kinetics and kinetics[r.sbml_id].is_usable()
               and not any(abs(c) > 3 for c in r.stoichiometry.values())]
    species = []
    seen = set()
    for rid, rxn, kin in active:
        for sid in rxn.stoichiometry:
            if sid not in seen:
                seen.add(sid); species.append(sid)
    print(f"    {len(active)} reactions, {len(species)} species")

    # ---- Build datasets with clamped y_norm ----
    print("\n[2] Building datasets with clamped y_norm...")
    t0 = time.time()
    train_d = build_dataset_clamped(active, species, init_concs,
                                       n_samples_per_rxn=300, seed=0)
    val_d = build_dataset_clamped(active, species, init_concs,
                                     n_samples_per_rxn=60, seed=1,
                                     per_rxn_scales=train_d['scales'])
    print(f"    datasets built in {time.time()-t0:.1f}s")
    print(f"    Train: {len(train_d['rates'])}  Val: {len(val_d['rates'])}")
    pct_clamped_train = 100 * train_d['n_clamped'] / len(train_d['rates'])
    pct_clamped_val = 100 * val_d['n_clamped'] / len(val_d['rates'])
    print(f"    Clamped fraction: train={pct_clamped_train:.2f}%  "
          f"val={pct_clamped_val:.2f}%")

    # ---- Train with fixes ----
    print("\n[3] Training with stability fixes...")
    model = PermInvRateNetV2_Large(n_species=len(species),
                                      hidden_dim=512, embed_dim=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params}")
    t0 = time.time()
    hist = train_stable(model, train_d, val_d, n_epochs=80,
                           batch_size=1024, lr=1e-3,
                           warmup_epochs=5, grad_clip=1.0)
    print(f"    Training time: {time.time()-t0:.1f}s")

    # Training stability metric (compare to P8c's 4e+06 / 1.5e-1 ≈ 2.7e+07)
    train_losses = hist['train']
    late_losses = train_losses[10:]  # skip warmup
    loss_range = max(late_losses) / (min(late_losses) + 1e-9)
    print(f"\n    Late-training loss oscillation: max/min = {loss_range:.2e}")
    print(f"    (P8c had oscillation ratio > 1e+07)")

    # ---- Evaluate ----
    print("\n[4] Evaluating per-reaction accuracy on val set...")
    model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        P = torch.tensor(val_d['participants']).to(dev)
        I = torch.tensor(val_d['indices']).to(dev)
        M = torch.tensor(val_d['masks']).to(dev)
        G = torch.tensor(val_d['globals']).to(dev)
        pred_norm = model(P, I, M, G).cpu().numpy()
    # Note: we evaluate against UNCLAMPED ground-truth rates.
    # Clamping was only for the training target.
    pred_rate = y_norm_to_rate(pred_norm, val_d['scales'], val_d['rxn_indices'])
    true_rate = val_d['rates']

    rxn_errs = []
    for ri in range(len(active)):
        m = val_d['rxn_indices'] == ri
        if not m.any(): continue
        y = true_rate[m]
        yhat = pred_rate[m]
        mm = float(np.abs(y).mean())
        me = float(np.abs(y - yhat).mean())
        if mm > 1e-10:
            rxn_errs.append(me / mm)
    rxn_errs = np.array(rxn_errs)
    sign_acc = float((np.sign(pred_rate) == np.sign(true_rate)).mean())
    under_20 = float((rxn_errs < 0.20).mean())
    under_50 = float((rxn_errs < 0.50).mean())
    under_100 = float((rxn_errs < 1.00).mean())

    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)
    print(f"\nPer-reaction rel err across {len(rxn_errs)} reactions:")
    print(f"  median: {np.median(rxn_errs):.2%}")
    print(f"  mean:   {np.mean(rxn_errs):.2%}")
    print(f"  25th:   {np.percentile(rxn_errs, 25):.2%}")
    print(f"  75th:   {np.percentile(rxn_errs, 75):.2%}")
    print(f"\nFraction under X% rel err:")
    print(f"  under  20%: {under_20:.1%}")
    print(f"  under  50%: {under_50:.1%}")
    print(f"  under 100%: {under_100:.1%}")
    print(f"\nSign agreement: {sign_acc:.2%}")

    # Tests
    t1 = float(np.median(rxn_errs)) < 0.50
    t2 = under_20 > 0.25
    t3 = sign_acc > 0.90
    print("\nTests:")
    print(f"  [T1] Median < 50%:           {np.median(rxn_errs):.2%}  "
          f"{'PASS' if t1 else 'FAIL'}")
    print(f"  [T2] > 25% under 20% err:    {under_20:.1%}  "
          f"{'PASS' if t2 else 'FAIL'}")
    print(f"  [T3] Sign agreement > 90%:   {sign_acc:.2%}  "
          f"{'PASS' if t3 else 'FAIL'}")

    # Comparison to P8c
    print(f"\nComparison (P8c -> P8e):")
    print(f"  median:        49.84% -> {np.median(rxn_errs):.2%}")
    print(f"  under 20%:     6.8%   -> {under_20:.1%}")
    print(f"  sign accuracy: 96.64% -> {sign_acc:.2%}")

    # Save
    import os
    out_dir = '/content/dmvc' if os.path.isdir('/content/dmvc') else '/home/claude/dmvc'
    np.savez(f'{out_dir}/p8e_results.npz',
              rxn_errs=rxn_errs, pred=pred_rate, true=true_rate,
              train_losses=np.array(train_losses),
              val_losses=np.array(hist['val']),
              grad_norms=np.array(hist['grad_norm']),
              lrs=np.array(hist['lr']))
    print(f"\nSaved results to {out_dir}/p8e_results.npz")

    # Interpret
    print("\n" + "=" * 76)
    print("INTERPRETATION")
    print("=" * 76)
    if t1 and t2:
        print("STRONG: stability fixes closed the gap. P8 architecture works.")
    elif t1 and under_20 > 0.15:
        print("PARTIAL+: meaningful improvement on precision metric,")
        print("         but still below T2 threshold. Fixes are working in the right direction.")
    elif np.median(rxn_errs) < 0.45:
        print("MILD: median improved vs P8c but precision did not.")
        print("      Suggests the 50% floor is deeper than optimization instability.")
    else:
        print("FAILED: stability fixes did not move median meaningfully.")
        print("        The error source is not optimization but something more structural.")


if __name__ == "__main__":
    main()
