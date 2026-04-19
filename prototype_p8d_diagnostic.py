"""
DMVC P8d: Diagnostic analysis of P8c's per-reaction error distribution.

P8c passed T1 (median < 50%) by 0.16 percentage points and failed T2
(fraction under 20% = 6.8%, target 25%). Training loss oscillated by
7 orders of magnitude between epochs. Before fixing anything, this
prototype runs the diagnostic analyses that should tell us what to fix.

Four specific diagnostic questions:

  Q1: Are pathological reactions dominating loss?
      Compute max single-sample loss per reaction; if a few reactions
      have max loss >100x the median, they're pulling the average.

  Q2: Does error correlate with reaction class?
      Group by number of participants, by whether it's transport,
      by whether it involves ATP/NADH cofactors. If some groups are
      systematically worse, chemistry heterogeneity is the bottleneck.

  Q3: Does error correlate with per-reaction rate range?
      For each reaction, compute max|rate| / median|rate| across training
      samples. If reactions with wider internal rate range have worse
      prediction error, normalization-within-reaction is insufficient.

  Q4: What do the 15 best and 15 worst reactions have in common?
      Look at reaction ids, compartments, kinetic parameters.

This prototype runs the P8c training loop exactly as P8c does, but adds
the diagnostic analyses at the end. No attempt to fix anything yet.
Output: comprehensive diagnostic report in stdout.
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
    PermInvRateNetV2_Large, train_v2_scaled,
)
from prototype_p3b_stamps import (
    load_sbml_model, extract_molecules_with_compartments, extract_reactions,
    find_h_and_water_per_compartment, rebalance_reaction, SBML_PATH,
    classify_by_compartment, RxnKind, Compartment,
)
from prototype_p4_kinetics import parse_all_kinetics


def main():
    print("=" * 76)
    print("DMVC P8d: Diagnostic analysis of P8c per-reaction errors")
    print("=" * 76)

    # ---- Step 1: Load Syn3A (same as P8c) ----
    print("\n[1] Loading Syn3A and building dataset (same as P8c)...")
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

    t0 = time.time()
    train_d = build_dataset_v2(active, species, init_concs,
                                  n_samples_per_rxn=300, seed=0)
    val_d = build_dataset_v2(active, species, init_concs,
                                n_samples_per_rxn=60, seed=1,
                                per_rxn_scales=train_d['scales'])
    print(f"    datasets built in {time.time()-t0:.1f}s")
    print(f"    Train: {len(train_d['rates'])}  Val: {len(val_d['rates'])}")

    # ---- Step 2: Train (same as P8c) ----
    print("\n[2] Training (hidden=512, embed=16, 80 epochs)...")
    model = PermInvRateNetV2_Large(n_species=len(species),
                                      hidden_dim=512, embed_dim=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params}")
    t0 = time.time()
    hist = train_v2_scaled(model, train_d, val_d, n_epochs=80,
                              batch_size=1024, lr=1e-3)
    print(f"    Training time: {time.time()-t0:.1f}s")

    # Capture training stability metric
    train_losses = [h[0] for h in hist]
    train_loss_ratio = max(train_losses[20:]) / (min(train_losses[20:]) + 1e-9)
    print(f"    Train loss oscillation ratio (epoch 20+): {train_loss_ratio:.2e}")

    # ---- Step 3: Evaluate (same as P8c) ----
    print("\n[3] Evaluating on val set...")
    model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        P = torch.tensor(val_d['participants']).to(dev)
        I = torch.tensor(val_d['indices']).to(dev)
        M = torch.tensor(val_d['masks']).to(dev)
        G = torch.tensor(val_d['globals']).to(dev)
        pred_norm = model(P, I, M, G).cpu().numpy()
    pred_rate = y_norm_to_rate(pred_norm, val_d['scales'], val_d['rxn_indices'])
    true_rate = val_d['rates']

    # Compute per-reaction metrics
    rxn_err = np.full(len(active), np.nan)
    rxn_max_sample_err = np.full(len(active), np.nan)
    rxn_rate_range = np.full(len(active), np.nan)   # max/median within reaction
    rxn_n_participants = np.zeros(len(active), dtype=int)
    rxn_kind = np.full(len(active), '', dtype=object)
    rxn_has_atp = np.zeros(len(active), dtype=bool)
    rxn_has_nad = np.zeros(len(active), dtype=bool)
    rxn_ids_all = [rid for rid, _, _ in active]

    for ri, (rid, rxn, kin) in enumerate(active):
        m = val_d['rxn_indices'] == ri
        if not m.any(): continue
        y = true_rate[m]
        yhat = pred_rate[m]
        mm = float(np.abs(y).mean())
        if mm > 1e-10:
            per_sample_abs_err = np.abs(y - yhat) / (np.abs(y) + 1e-10)
            rxn_err[ri] = float(np.mean(per_sample_abs_err))
            rxn_max_sample_err[ri] = float(np.max(per_sample_abs_err))

        # Rate range within the reaction (from training data)
        m_tr = train_d['rxn_indices'] == ri
        tr_rates = np.abs(train_d['rates'][m_tr])
        tr_rates = tr_rates[tr_rates > 1e-12]
        if len(tr_rates) > 5:
            rxn_rate_range[ri] = float(np.max(tr_rates) / np.median(tr_rates))

        # Reaction class features
        rxn_n_participants[ri] = len(rxn.stoichiometry)
        kind = classify_by_compartment(rxn, mols)
        rxn_kind[ri] = kind.name if kind else 'UNKNOWN'
        species_set = set(rxn.stoichiometry.keys())
        rxn_has_atp[ri] = any('atp' in s.lower() or 'adp' in s.lower() or 'amp' in s.lower()
                                for s in species_set)
        rxn_has_nad[ri] = any('nad' in s.lower() for s in species_set)

    # ---- Step 4: Diagnostic analyses ----
    print("\n" + "=" * 76)
    print("DIAGNOSTIC ANALYSIS")
    print("=" * 76)

    # Overall summary
    valid = ~np.isnan(rxn_err)
    errs = rxn_err[valid]
    print(f"\nOverall: {valid.sum()} reactions evaluated")
    print(f"  median error: {np.median(errs):.2%}")
    print(f"  mean error:   {np.mean(errs):.2%}")
    print(f"  25th/50th/75th: "
          f"{np.percentile(errs, 25):.2%} / "
          f"{np.percentile(errs, 50):.2%} / "
          f"{np.percentile(errs, 75):.2%}")
    print(f"  fraction under 20%: {(errs < 0.20).mean():.1%}")
    print(f"  fraction over 100%: {(errs > 1.00).mean():.1%}")

    # -----------------------------------------------------------------
    # Q1: Pathological reactions
    # -----------------------------------------------------------------
    print("\n" + "-" * 76)
    print("Q1: Are pathological reactions dominating loss?")
    print("-" * 76)

    max_err_valid = rxn_max_sample_err[valid]
    med_max = float(np.median(max_err_valid))
    print(f"\nMax single-sample relative error per reaction:")
    print(f"  median: {med_max:.2%}  mean: {np.mean(max_err_valid):.2%}")
    print(f"  95th percentile: {np.percentile(max_err_valid, 95):.2%}")
    print(f"  99th percentile: {np.percentile(max_err_valid, 99):.2%}")
    print(f"  max:             {np.max(max_err_valid):.2%}")

    # How many reactions have max error > 100x median?
    pathological = max_err_valid > med_max * 100
    print(f"\n  Reactions with max-sample-error > 100x median: "
          f"{pathological.sum()} / {valid.sum()}")
    if pathological.sum() > 0:
        valid_ids = [rxn_ids_all[i] for i in range(len(active)) if valid[i]]
        pathological_ids = [valid_ids[i] for i in range(len(valid_ids)) if pathological[i]]
        print(f"    Examples: {pathological_ids[:10]}")

    # -----------------------------------------------------------------
    # Q2: Reaction class
    # -----------------------------------------------------------------
    print("\n" + "-" * 76)
    print("Q2: Does error correlate with reaction class?")
    print("-" * 76)

    # By compartment kind
    print("\nBy compartment kind:")
    for kind_name in np.unique(rxn_kind[valid]):
        mask = (rxn_kind == kind_name) & valid
        if mask.sum() > 0:
            print(f"  {kind_name:20s}  n={mask.sum():3d}  "
                  f"median err={np.median(rxn_err[mask]):.2%}  "
                  f"mean err={np.mean(rxn_err[mask]):.2%}")

    # By cofactor involvement
    print("\nBy cofactor involvement:")
    for label, mask_extra in [
            ('has ATP/ADP/AMP', rxn_has_atp),
            ('has NAD(H)',      rxn_has_nad),
            ('neither',         ~(rxn_has_atp | rxn_has_nad)),
    ]:
        mask = mask_extra & valid
        if mask.sum() > 0:
            print(f"  {label:20s}  n={mask.sum():3d}  "
                  f"median err={np.median(rxn_err[mask]):.2%}")

    # By number of participants
    print("\nBy number of participants:")
    n_parts = rxn_n_participants[valid]
    for n_min, n_max in [(2, 3), (4, 5), (6, 7), (8, 20)]:
        mask = (rxn_n_participants >= n_min) & (rxn_n_participants <= n_max) & valid
        if mask.sum() > 0:
            print(f"  {n_min}-{n_max} participants  n={mask.sum():3d}  "
                  f"median err={np.median(rxn_err[mask]):.2%}")

    # -----------------------------------------------------------------
    # Q3: Rate range correlation
    # -----------------------------------------------------------------
    print("\n" + "-" * 76)
    print("Q3: Does error correlate with per-reaction rate range?")
    print("-" * 76)

    mask = valid & ~np.isnan(rxn_rate_range)
    r_range = rxn_rate_range[mask]
    r_err = rxn_err[mask]
    print(f"\nRate range within reactions (max/median), over {mask.sum()} reactions:")
    print(f"  25th: {np.percentile(r_range, 25):.1f}x")
    print(f"  50th: {np.percentile(r_range, 50):.1f}x")
    print(f"  75th: {np.percentile(r_range, 75):.1f}x")
    print(f"  95th: {np.percentile(r_range, 95):.1f}x")
    print(f"  max:  {r_range.max():.1e}x")

    # Binned analysis
    print("\nError vs rate-range bin:")
    log_range = np.log10(r_range + 1)
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 6)]:
        bin_mask = (log_range >= lo) & (log_range < hi)
        if bin_mask.sum() > 0:
            print(f"  rate range 10^{lo}-10^{hi}:  n={bin_mask.sum():3d}  "
                  f"median err={np.median(r_err[bin_mask]):.2%}")

    # Correlation
    if len(log_range) > 5:
        corr = float(np.corrcoef(log_range, r_err)[0, 1])
        print(f"\n  Pearson corr(log rate range, error): {corr:+.3f}")
        if corr > 0.3:
            print("  POSITIVE: reactions with wider rate range have higher error")
            print("  -> within-reaction normalization is insufficient")
        elif corr > 0.1:
            print("  MILD POSITIVE: some effect but not dominant")
        else:
            print("  WEAK: rate range isn't the main bottleneck")

    # -----------------------------------------------------------------
    # Q4: Best and worst
    # -----------------------------------------------------------------
    print("\n" + "-" * 76)
    print("Q4: What do best/worst reactions have in common?")
    print("-" * 76)

    valid_indices = np.where(valid)[0]
    valid_errs = rxn_err[valid_indices]
    sort_idx = np.argsort(valid_errs)

    print("\n15 BEST (lowest error) reactions:")
    for k in sort_idx[:15]:
        ri = valid_indices[k]
        rid = rxn_ids_all[ri]
        print(f"  {rid:20s}  err={rxn_err[ri]:.2%}  "
              f"kind={rxn_kind[ri]:15s}  "
              f"nparts={rxn_n_participants[ri]}  "
              f"range={rxn_rate_range[ri]:.1e}")

    print("\n15 WORST (highest error) reactions:")
    for k in sort_idx[-15:][::-1]:
        ri = valid_indices[k]
        rid = rxn_ids_all[ri]
        print(f"  {rid:20s}  err={rxn_err[ri]:.2%}  "
              f"kind={rxn_kind[ri]:15s}  "
              f"nparts={rxn_n_participants[ri]}  "
              f"range={rxn_rate_range[ri]:.1e}")

    # -----------------------------------------------------------------
    # Summary of findings -> which fix to try
    # -----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("DIAGNOSIS SUMMARY")
    print("=" * 76)

    print(f"\nTraining stability: loss oscillated {train_loss_ratio:.1e}x across epochs")
    print(f"Pathological sample-max rate: {(max_err_valid > 10).mean():.1%} of reactions")

    if len(log_range) > 5:
        corr = float(np.corrcoef(log_range, r_err)[0, 1])
        print(f"Correlation of rate range with error: {corr:+.3f}")

    # Save for downstream
    import os
    out_dir = '/content/dmvc' if os.path.isdir('/content/dmvc') else '/home/claude/dmvc'
    np.savez(f'{out_dir}/p8d_diagnostic.npz',
              rxn_err=rxn_err, rxn_max_sample_err=rxn_max_sample_err,
              rxn_rate_range=rxn_rate_range,
              rxn_n_participants=rxn_n_participants,
              rxn_kind=rxn_kind, rxn_has_atp=rxn_has_atp,
              rxn_has_nad=rxn_has_nad,
              rxn_ids=np.array(rxn_ids_all),
              train_losses=np.array(train_losses),
              train_loss_ratio=train_loss_ratio)
    print(f"\nSaved detailed diagnostics to {out_dir}/p8d_diagnostic.npz")

    return rxn_err, rxn_max_sample_err, rxn_rate_range


if __name__ == "__main__":
    main()
