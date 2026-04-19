"""
DMVC P8b: Permutation-invariant network at full Syn3A scale.

Re-attempt of P8 Scenario 2. P8's original run failed because the rate
distribution spans roughly 10^-8 to 10^-1 mM/s (seven orders of magnitude)
and a global-MSE loss was dominated by the largest-rate reactions. The
network's best response was to predict near-zero everywhere — loss looked
fine, predictions were useless.

Two targeted fixes, everything else unchanged:

  FIX 1 — Per-reaction log-scale normalization.
    For each reaction r_i, compute a per-reaction scale s_i from training
    samples (median absolute rate). Loss for a (sample, reaction) pair is
    ((pred - true) / s_i)^2. Now every reaction contributes comparably to
    the loss regardless of its absolute-rate magnitude.

  FIX 2 — Log-space prediction target.
    Network predicts y_norm = sign(r) * log1p(|r|/s_i) rather than raw r.
    This compresses the large-rate tail, stays smooth near zero, preserves
    sign, and makes a multiplicative-error criterion natural.

  FIX 3 — Species embeddings.
    P8 used only (concentration, stoichiometry, Km, is_substrate, is_product)
    per participant. At 221 reactions the network needs to distinguish
    chemically-distinct species. We add a learnable species embedding
    (dim 8) concatenated to each participant's feature vector.

Architecture: same Deep Sets shape as P8 (per-participant encoder + sum
aggregator + decoder with global features), but hidden dim scaled up
and input dim increased by the embedding.

Tests:
  T1 — All-reactions fit: median per-reaction relative error < 50%.
       The threshold at which the network genuinely learned kinetics
       rather than predicting means. For context, P8 Scenario 1 hit 47%
       on 3 held-out glycolysis reactions.
  T2 — Top-quartile coverage: at least 25% of reactions achieve
       relative error < 20% (strong fit on some, not a mean predictor).
  T3 — Sign-correctness: fraction of samples where sign(pred) == sign(true)
       is > 90% (even if magnitude is off, at least direction is right).
  T4 — Diagnostic: visible spread of predicted rates across reactions
       (not everything collapsed to zero). Checked by std of per-reaction
       mean predictions — should be > 0.001.

Honest pre-registration:
  If T1 passes, the "one network for Syn3A" architectural claim has
    substantive support.
  If T1 fails but T2/T3 pass, partial success — some reactions are
    learned, others aren't, and we'd need to understand which.
  If T1-T3 all fail, 221 reactions of distinct chemistry is genuinely
    beyond what this architecture at this capacity can represent,
    and the next step is either bigger model or per-family sub-networks.
"""

from __future__ import annotations
import sys, time
sys.path.insert(0, "/home/claude/dmvc")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prototype_p3b_stamps import (
    load_sbml_model, extract_molecules_with_compartments, extract_reactions,
    find_h_and_water_per_compartment, rebalance_reaction,
    SBML_PATH,
)
from prototype_p4_kinetics import parse_all_kinetics, ReactionKinetics
from prototype_p7_learned_rates import convenience_rate_passive


MAX_PARTICIPANTS = 12
EMB_DIM = 8


# =============================================================================
# Participant encoding: per-species features + species index for embedding
# =============================================================================
def encode_reaction_participants_with_idx(rxn, kin, species_to_idx, concs):
    """
    Return:
      feats  (MAX_PARTICIPANTS, 5) -- [conc, nu, Km, is_sub, is_prod]
      idx    (MAX_PARTICIPANTS,)   -- species index (for embedding lookup);
                                       0 for padded entries
      mask   (MAX_PARTICIPANTS,)   -- 1 for valid entries
      gfeats (4,)                  -- [kcat_f, kcat_r, enzyme_conc, n_part]
    """
    feats = np.zeros((MAX_PARTICIPANTS, 5), dtype=np.float32)
    idx = np.zeros(MAX_PARTICIPANTS, dtype=np.int64)
    mask = np.zeros(MAX_PARTICIPANTS, dtype=np.float32)

    ss_list = list(rxn.stoichiometry.items())[:MAX_PARTICIPANTS]
    for i, (sid, nu) in enumerate(ss_list):
        c = concs[species_to_idx[sid]] if sid in species_to_idx else 0.0
        km = kin.Km.get(sid, 1.0)
        is_sub = 1.0 if nu < 0 else 0.0
        is_prod = 1.0 if nu > 0 else 0.0
        feats[i] = [max(float(c), 0.0), float(nu), float(km), is_sub, is_prod]
        idx[i] = species_to_idx.get(sid, 0)
        mask[i] = 1.0

    gfeats = np.array([
        kin.kcat_forward, kin.kcat_reverse if kin.kcat_reverse else 0.0,
        kin.enzyme_conc if kin.enzyme_conc else 0.0,
        float(len(ss_list)),
    ], dtype=np.float32)
    return feats, idx, mask, gfeats


# =============================================================================
# Network with species embeddings
# =============================================================================
class PermInvRateNetV2(nn.Module):
    """
    Same Deep Sets shape as P8 but:
      - per-participant input includes a learnable species embedding
      - hidden dim scaled up
      - no Km-gating tricks; just straightforward Deep Sets
    """
    def __init__(self, n_species, participant_feat_dim=5, global_feat_dim=4,
                  hidden_dim=256, embed_dim=EMB_DIM):
        super().__init__()
        self.embed = nn.Embedding(n_species + 1, embed_dim)  # +1 for pad idx 0
        in_dim = participant_feat_dim + embed_dim

        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        dec_in = hidden_dim + global_feat_dim
        self.dec = nn.Sequential(
            nn.Linear(dec_in, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, p_feats, p_idx, mask, gfeats):
        # p_feats: (B, M, 5)  p_idx: (B, M)  mask: (B, M)  gfeats: (B, 4)
        emb = self.embed(p_idx)                              # (B, M, E)
        x = torch.cat([p_feats, emb], dim=-1)                 # (B, M, 5+E)
        h = self.enc(x)                                       # (B, M, H)
        h = h * mask.unsqueeze(-1)                            # zero padded
        g = h.sum(dim=1)                                      # (B, H)
        y = self.dec(torch.cat([g, gfeats], dim=-1))          # (B, 1)
        return y.squeeze(-1)


# =============================================================================
# Dataset with per-reaction scales and log-space targets
# =============================================================================
def build_dataset_v2(rxns_kin_list, species, init_concs,
                      n_samples_per_rxn, perturb_scale=3.0, seed=0,
                      per_rxn_scales=None):
    """
    Returns:
      a dict with participants, indices, masks, globals, rates (raw),
      rxn_indices, and y_norm (log-space normalized targets).

    per_rxn_scales: if None, compute from training data (per-reaction
      median absolute rate, floored at 1e-9). If provided, use those.
    """
    rng = np.random.default_rng(seed)
    species_to_idx = {s: i for i, s in enumerate(species)}
    n_sp = len(species)
    base = np.array([init_concs.get(s, 0.1) for s in species])

    total = len(rxns_kin_list) * n_samples_per_rxn

    P = np.zeros((total, MAX_PARTICIPANTS, 5), dtype=np.float32)
    I = np.zeros((total, MAX_PARTICIPANTS), dtype=np.int64)
    M = np.zeros((total, MAX_PARTICIPANTS), dtype=np.float32)
    G = np.zeros((total, 4), dtype=np.float32)
    rates = np.zeros(total, dtype=np.float32)
    rxn_idx = np.zeros(total, dtype=np.int64)

    k = 0
    for ri, (rid, rxn, kin) in enumerate(rxns_kin_list):
        for _ in range(n_samples_per_rxn):
            log_factor = rng.uniform(
                -np.log(perturb_scale), np.log(perturb_scale), size=n_sp)
            concs = base * np.exp(log_factor)
            if rng.random() < 0.05:
                concs[rng.integers(n_sp)] = 0.0

            feats, idx, mask, gfeats = encode_reaction_participants_with_idx(
                rxn, kin, species_to_idx, concs)
            P[k] = feats; I[k] = idx; M[k] = mask; G[k] = gfeats

            c_dict = {s: float(c) for s, c in zip(species, concs)}
            rates[k] = convenience_rate_passive(rxn, kin, c_dict)
            rxn_idx[k] = ri
            k += 1

    # Per-reaction scales
    if per_rxn_scales is None:
        scales = np.zeros(len(rxns_kin_list), dtype=np.float32)
        for ri in range(len(rxns_kin_list)):
            m = rxn_idx == ri
            if m.any():
                ra = np.abs(rates[m])
                med = float(np.median(ra))
                scales[ri] = max(med, 1e-9)
            else:
                scales[ri] = 1e-9
    else:
        scales = per_rxn_scales

    # Log-space normalized targets: y_norm = sign(r) * log1p(|r| / s_ri)
    s_per_sample = scales[rxn_idx]
    y_norm = np.sign(rates) * np.log1p(np.abs(rates) / s_per_sample)
    y_norm = y_norm.astype(np.float32)

    return {
        'participants': P, 'indices': I, 'masks': M, 'globals': G,
        'rates': rates, 'rxn_indices': rxn_idx, 'y_norm': y_norm,
        'scales': scales,
    }


def y_norm_to_rate(y_norm, scales, rxn_indices):
    """Invert: rate = sign(y) * s_ri * (exp(|y|) - 1)."""
    s = scales[rxn_indices]
    return np.sign(y_norm) * s * np.expm1(np.abs(y_norm))


# =============================================================================
# Training
# =============================================================================
def train_v2(model, train_data, val_data, n_epochs=40, batch_size=1024,
               lr=1e-3, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  (training on device: {device})")
    model = model.to(device)

    def to_tensors(d):
        return (
            torch.tensor(d['participants']).to(device),
            torch.tensor(d['indices']).to(device),
            torch.tensor(d['masks']).to(device),
            torch.tensor(d['globals']).to(device),
            torch.tensor(d['y_norm']).to(device),
        )
    Ptr, Itr, Mtr, Gtr, Ytr = to_tensors(train_data)
    Pva, Iva, Mva, Gva, Yva = to_tensors(val_data)

    ds = TensorDataset(Ptr, Itr, Mtr, Gtr, Ytr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    hist = {'train': [], 'val': []}
    for epoch in range(n_epochs):
        model.train()
        losses = []
        for pb, ib, mb, gb, yb in loader:
            pred = model(pb, ib, mb, gb)
            # MSE in log-normalized space
            loss = (pred - yb).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        tl = float(np.mean(losses))

        model.eval()
        with torch.no_grad():
            pred_va = model(Pva, Iva, Mva, Gva)
            vl = (pred_va - Yva).pow(2).mean().item()
        hist['train'].append(tl); hist['val'].append(vl)
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}  train={tl:.4e}  val={vl:.4e}")
    return hist


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 76)
    print("DMVC P8b: Permutation-invariant at full Syn3A scale")
    print("        with per-reaction log-scale loss + species embeddings")
    print("=" * 76)

    print("\n[1] Loading full Syn3A...")
    t0 = time.time()
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
    print(f"    {len(active)} reactions, {len(species)} species, "
          f"load time {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------
    print("\n[2] Building datasets with per-reaction log-scale targets...")
    t0 = time.time()
    train_d = build_dataset_v2(active, species, init_concs,
                                  n_samples_per_rxn=80, seed=0)
    val_d = build_dataset_v2(active, species, init_concs,
                                n_samples_per_rxn=20, seed=1,
                                per_rxn_scales=train_d['scales'])
    print(f"    Train samples: {len(train_d['rates'])}  "
          f"Val samples: {len(val_d['rates'])}  (build time {time.time()-t0:.1f}s)")

    # Report scale distribution
    scales = train_d['scales']
    scale_finite = scales[scales > 1e-9]
    if len(scale_finite):
        print(f"    Per-reaction scales: "
              f"min={scale_finite.min():.2e}  "
              f"median={np.median(scale_finite):.2e}  "
              f"max={scale_finite.max():.2e}  "
              f"range={scale_finite.max()/scale_finite.min():.1e}x")
    raw_rates = np.abs(train_d['rates'])
    raw_rates_nz = raw_rates[raw_rates > 1e-12]
    print(f"    Raw rate range: {raw_rates_nz.min():.2e} to "
          f"{raw_rates_nz.max():.2e} mM/s ({raw_rates_nz.max()/raw_rates_nz.min():.1e}x span)")

    # -----------------------------------------------------------------
    print("\n[3] Training permutation-invariant network (hidden=256)...")
    model = PermInvRateNetV2(n_species=len(species), hidden_dim=256)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params}")
    t0 = time.time()
    hist = train_v2(model, train_d, val_d, n_epochs=40, batch_size=1024, lr=1e-3)
    print(f"    Training time: {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------
    print("\n[4] Evaluating per-reaction accuracy on held-out val set...")
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

    rxn_errs = []
    rxn_pred_means = []
    n_nonzero = 0
    n_eval = 0
    for ri in range(len(active)):
        m = val_d['rxn_indices'] == ri
        if not m.any(): continue
        y = true_rate[m]
        yhat = pred_rate[m]
        mm = float(np.abs(y).mean())
        me = float(np.abs(y - yhat).mean())
        if mm > 1e-10:
            rxn_errs.append(me / mm)
            n_nonzero += 1
        n_eval += 1
        rxn_pred_means.append(float(np.abs(yhat).mean()))

    rxn_errs = np.array(rxn_errs)

    # Sign agreement
    sign_ok = (np.sign(pred_rate) == np.sign(true_rate))
    sign_acc = float(sign_ok.mean())

    # Pred mean spread
    pred_spread = float(np.std(rxn_pred_means))

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)

    print(f"\nEvaluated: {n_eval} reactions, {n_nonzero} with non-zero rates")
    print(f"Per-reaction relative error across {n_nonzero} reactions:")
    print(f"  median: {np.median(rxn_errs):.2%}")
    print(f"  mean:   {np.mean(rxn_errs):.2%}")
    print(f"  25th %: {np.percentile(rxn_errs, 25):.2%}")
    print(f"  75th %: {np.percentile(rxn_errs, 75):.2%}")

    # Breakdown
    under_20 = float((rxn_errs < 0.20).mean())
    under_50 = float((rxn_errs < 0.50).mean())
    under_100 = float((rxn_errs < 1.00).mean())
    print(f"\nFraction of reactions under X% relative error:")
    print(f"  under 20%:  {under_20:.1%}")
    print(f"  under 50%:  {under_50:.1%}")
    print(f"  under 100%: {under_100:.1%}")

    print(f"\nSign agreement (pred direction == true direction): {sign_acc:.2%}")
    print(f"Per-reaction pred-mean spread: {pred_spread:.4f}")

    # Tests
    t1 = float(np.median(rxn_errs)) < 0.50
    t2 = under_20 > 0.25
    t3 = sign_acc > 0.90
    t4 = pred_spread > 0.001

    print("\nTests:")
    print(f"  [T1] Median rel err < 50%: {np.median(rxn_errs):.2%}  "
          f"{'PASS' if t1 else 'FAIL'}")
    print(f"  [T2] > 25% of rxns under 20% err: {under_20:.1%}  "
          f"{'PASS' if t2 else 'FAIL'}")
    print(f"  [T3] Sign agreement > 90%: {sign_acc:.2%}  "
          f"{'PASS' if t3 else 'FAIL'}")
    print(f"  [T4] Pred spread > 0.001 (not all-zero): {pred_spread:.4f}  "
          f"{'PASS' if t4 else 'FAIL'}")

    print()
    if t1 and t2 and t3 and t4:
        print("STRONG: one network represents full Syn3A kinetics.")
    elif t3 and t4 and not t1:
        print("PARTIAL: network is learning meaningful signal (sign + diversity)")
        print("but absolute accuracy is low. Needs more capacity or curriculum.")
    elif not t4:
        print("FAIL: network collapsed to near-zero predictions; normalization")
        print("insufficient. This is the same failure mode as original P8 scenario 2.")
    else:
        print("PARTIAL: fix the specific failing test.")

    return rxn_errs, pred_rate, true_rate


if __name__ == "__main__":
    main()
