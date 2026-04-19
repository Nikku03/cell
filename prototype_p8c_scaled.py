"""
DMVC P8c: Scaled-up P8b.

P8b established that the original P8 Scenario 2 failure (collapse to zero)
was a loss-design problem, not an architectural one. With per-reaction
log-scale normalization and species embeddings, the network learned to
predict the right direction and rough magnitude across all 221 reactions,
but absolute accuracy was middling (median 59% error).

P8c: same architecture, more of everything that's cheap.

  hidden_dim        256 -> 512   (double per-layer capacity)
  embedding_dim       8 ->  16   (more room for species identity)
  samples / rxn      80 -> 300   (4x more training signal per reaction)
  epochs             40 ->  80   (val loss was still decreasing)

If this passes, the P8 architectural claim ("one permutation-invariant
network can represent full Syn3A kinetics") has clean support.
If it fails, we diagnose which reactions are hard.
"""

from __future__ import annotations
import sys, time
sys.path.insert(0, "/home/claude/dmvc")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Reuse infrastructure from P8b
from prototype_p8b_full_syn3a import (
    MAX_PARTICIPANTS, EMB_DIM,
    encode_reaction_participants_with_idx,
    build_dataset_v2, y_norm_to_rate,
    PermInvRateNetV2,  # default embed=EMB_DIM (8); we'll subclass for 16
)
from prototype_p3b_stamps import (
    load_sbml_model, extract_molecules_with_compartments, extract_reactions,
    find_h_and_water_per_compartment, rebalance_reaction, SBML_PATH,
)
from prototype_p4_kinetics import parse_all_kinetics


# Larger-embedding variant (the P8b class hardcodes embed=EMB_DIM via default)
class PermInvRateNetV2_Large(nn.Module):
    def __init__(self, n_species, participant_feat_dim=5, global_feat_dim=4,
                  hidden_dim=512, embed_dim=16):
        super().__init__()
        self.embed = nn.Embedding(n_species + 1, embed_dim)
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
        emb = self.embed(p_idx)
        x = torch.cat([p_feats, emb], dim=-1)
        h = self.enc(x)
        h = h * mask.unsqueeze(-1)
        g = h.sum(dim=1)
        y = self.dec(torch.cat([g, gfeats], dim=-1))
        return y.squeeze(-1)


def train_v2_scaled(model, train_data, val_data, n_epochs=80,
                      batch_size=1024, lr=1e-3, device=None):
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    hist = {'train': [], 'val': []}
    best_val = float('inf')
    for epoch in range(n_epochs):
        model.train()
        losses = []
        for pb, ib, mb, gb, yb in loader:
            pred = model(pb, ib, mb, gb)
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
        if vl < best_val: best_val = vl
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}  train={tl:.4e}  val={vl:.4e}  "
                  f"best={best_val:.4e}")
    return hist


def main():
    print("=" * 76)
    print("DMVC P8c: Scaled-up Syn3A learning")
    print("         hidden=512, embed=16, 300 samples/rxn, 80 epochs")
    print("=" * 76)

    print("\n[1] Loading full Syn3A...")
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

    print("\n[2] Building scaled datasets (300 train, 60 val per reaction)...")
    t0 = time.time()
    train_d = build_dataset_v2(active, species, init_concs,
                                  n_samples_per_rxn=300, seed=0)
    val_d = build_dataset_v2(active, species, init_concs,
                                n_samples_per_rxn=60, seed=1,
                                per_rxn_scales=train_d['scales'])
    print(f"    Train: {len(train_d['rates'])}  Val: {len(val_d['rates'])}  "
          f"(build time {time.time()-t0:.1f}s)")

    print("\n[3] Building scaled network...")
    model = PermInvRateNetV2_Large(n_species=len(species),
                                     hidden_dim=512, embed_dim=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params}  (vs P8b: 270,425)")

    print("\n[4] Training (80 epochs, batch=1024)...")
    t0 = time.time()
    hist = train_v2_scaled(model, train_d, val_d, n_epochs=80,
                              batch_size=1024, lr=1e-3)
    print(f"    Training time: {time.time()-t0:.1f}s")

    print("\n[5] Evaluating per-reaction accuracy on val set...")
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

    # Per-reaction error
    rxn_errs = []
    rxn_ids = []
    for ri in range(len(active)):
        m = val_d['rxn_indices'] == ri
        if not m.any(): continue
        y = true_rate[m]
        yhat = pred_rate[m]
        mm = float(np.abs(y).mean())
        me = float(np.abs(y - yhat).mean())
        if mm > 1e-10:
            rxn_errs.append(me / mm)
            rxn_ids.append(active[ri][0])
    rxn_errs = np.array(rxn_errs)
    sign_acc = float((np.sign(pred_rate) == np.sign(true_rate)).mean())

    # Results
    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)
    print(f"\nPer-reaction rel err across {len(rxn_errs)} reactions:")
    print(f"  median: {np.median(rxn_errs):.2%}")
    print(f"  mean:   {np.mean(rxn_errs):.2%}")
    print(f"  25th:   {np.percentile(rxn_errs, 25):.2%}")
    print(f"  75th:   {np.percentile(rxn_errs, 75):.2%}")
    print(f"\nFraction under X% rel err:")
    under_20 = float((rxn_errs < 0.20).mean())
    under_50 = float((rxn_errs < 0.50).mean())
    under_100 = float((rxn_errs < 1.00).mean())
    print(f"  under  20%: {under_20:.1%}")
    print(f"  under  50%: {under_50:.1%}")
    print(f"  under 100%: {under_100:.1%}")
    print(f"\nSign agreement: {sign_acc:.2%}")

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

    # Compare to P8b
    print(f"\nComparison to P8b:")
    print(f"  P8b median:  59.31%  -> P8c median:  {np.median(rxn_errs):.2%}")
    print(f"  P8b under 20%: 4.1%  -> P8c under 20%: {under_20:.1%}")
    print(f"  P8b sign:    91.79%  -> P8c sign:    {sign_acc:.2%}")

    # Save per-reaction errors for downstream diagnosis
    np.savez('/home/claude/dmvc/p8c_results.npz',
              rxn_errs=rxn_errs,
              rxn_ids=np.array(rxn_ids),
              pred=pred_rate, true=true_rate,
              rxn_indices=val_d['rxn_indices'],
              scales=val_d['scales'])
    print("\nSaved per-reaction errors to p8c_results.npz for diagnosis.")

    return rxn_errs, rxn_ids, pred_rate, true_rate


if __name__ == "__main__":
    main()
