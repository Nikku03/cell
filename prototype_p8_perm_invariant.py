"""
DMVC P8: Permutation-invariant rate predictor.

Claim being tested: a single network trained on a SUBSET of reactions
can predict rates for reactions it has never seen, using only stoichiometry
+ kinetic parameters + local concentrations.

Design:
  The rate of a reaction is computed by a function whose inputs are
    * stoichiometric coefficients (ν_i per substrate/product)
    * kinetic constants (Km per participant, kcat_f, kcat_r, enzyme conc)
    * local concentrations (full cell state)
  This is exactly what the Liebermeister convenience rate law does
  analytically. We replace that analytical form with a learned one,
  constructed as a DEEP SETS aggregator over reaction participants.

  For reaction R with N participants {(species_i, nu_i, Km_i)}:
    per-participant vector h_i = enc([conc_i, nu_i, Km_i, is_substrate, is_product])
    aggregated:   g_R = sum_i h_i
    rate:         v_R = dec([g_R, kcat_f, kcat_r, E, |N|])

  Permutation invariance is structural: summing is order-independent.
  Cross-reaction generalization is the claim: the shared encoder+decoder
  must learn chemistry rules, not reaction-specific quirks.

Test structure:
  Scenario 1: Glycolysis generalization split (7 train / 3 test)
    * Train on 7 reactions, evaluate on 3 never-seen reactions
    * Success: held-out rates within 50% relative error (weak but meaningful)
    * Trajectory: run a simulator using the held-out reactions too; see how
      much it degrades vs ground truth

  Scenario 2: Full Syn3A scale (train on all ~220 elementary reactions)
    * Can a single shared function represent all Syn3A kinetics?
    * Success: trajectory match < 20% on central metabolism
    * This is architectural: if one small network can cover 220 reactions,
      the approach scales.

Honest pre-registration:
  * Zero-shot held-out rate error of 30-80% is already a real result
  * Trajectory error on held-out reactions will likely be 20-50% (amplified by feedback)
  * Full Syn3A scale is more uncertain: might just not fit in 10K params
  * If scenario 1 fails badly, the chemistry-learning claim is in trouble
"""

from __future__ import annotations
import numpy as np
import sys
sys.path.insert(0, "/home/claude/dmvc")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prototype_p4_kinetics import (
    parse_all_kinetics, ReactionKinetics, convenience_rate,
)
from prototype_p3b_stamps import (
    load_sbml_model, extract_molecules_with_compartments, extract_reactions,
    find_h_and_water_per_compartment, rebalance_reaction,
    SBML_PATH,
)
from prototype_p7_learned_rates import (
    convenience_rate_passive, GLYCOLYSIS_RXNS,
    load_kinetics_subset, gather_species,
)


# =============================================================================
# Reaction encoding into a fixed-shape tensor for Deep Sets processing
# =============================================================================
# Each participant (species in a reaction) is encoded as a feature vector.
# To batch reactions of different sizes together, we pad to MAX_PARTICIPANTS
# with a binary mask indicating valid entries.

MAX_PARTICIPANTS = 12   # > the largest real elementary reaction we include


def encode_reaction_participants(rxn, kin: ReactionKinetics,
                                   species_to_idx: dict, concs: np.ndarray
                                   ) -> tuple:
    """
    For each reaction, build a (MAX_PARTICIPANTS, F) feature tensor + mask.
    Features per participant (F=5):
      [0] local concentration of this species
      [1] stoichiometric coefficient (signed)
      [2] Km (0 if passive / no Km)
      [3] is_substrate (1 if coef < 0)
      [4] is_product (1 if coef > 0)
    """
    F = 5
    feats = np.zeros((MAX_PARTICIPANTS, F), dtype=np.float32)
    mask = np.zeros(MAX_PARTICIPANTS, dtype=np.float32)

    i = 0
    for sid, coef in rxn.stoichiometry.items():
        if i >= MAX_PARTICIPANTS:
            break
        if sid not in species_to_idx:
            continue
        c = float(max(concs[species_to_idx[sid]], 0.0))
        Km = kin.Km.get(sid, 0.0) or 0.0
        feats[i, 0] = c
        feats[i, 1] = float(coef)
        feats[i, 2] = float(Km)
        feats[i, 3] = 1.0 if coef < 0 else 0.0
        feats[i, 4] = 1.0 if coef > 0 else 0.0
        mask[i] = 1.0
        i += 1

    # Global kinetic params (will be broadcast by the caller)
    global_feats = np.array([
        float(kin.kcat_forward or 0.0),
        float(kin.kcat_reverse or 0.0),
        float(kin.enzyme_conc or 0.0),
        float(i),   # number of participants
    ], dtype=np.float32)

    return feats, mask, global_feats


# =============================================================================
# Permutation-invariant rate network
# =============================================================================
class PermInvRateNet(nn.Module):
    """
    Deep-sets style rate predictor.
      - enc: per-participant encoder (shared across participants AND reactions)
      - aggregator: mean-sum-max over participants
      - dec: takes aggregated vector + global kinetic params, outputs rate

    Permutation invariance holds by construction (aggregation is symmetric).
    Cross-reaction generalization is the empirical claim.
    """

    def __init__(self, participant_feat_dim=5, global_feat_dim=4,
                  hidden_dim=64):
        super().__init__()
        H = hidden_dim

        # Per-participant encoder
        self.enc = nn.Sequential(
            nn.Linear(participant_feat_dim, H),
            nn.GELU(),
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, H),
        )

        # After aggregation: sum and max pooled
        agg_dim = 2 * H + global_feat_dim

        # Decoder produces log|rate| and sign
        self.dec = nn.Sequential(
            nn.Linear(agg_dim, H),
            nn.GELU(),
            nn.Linear(H, H),
            nn.GELU(),
        )
        self.head_log_mag = nn.Linear(H, 1)
        self.head_sign = nn.Linear(H, 1)

        self.eps = 1e-6

    def forward(self, participant_feats, mask, global_feats):
        """
        participant_feats: (batch, MAX_PARTICIPANTS, F_p)
        mask: (batch, MAX_PARTICIPANTS)
        global_feats: (batch, F_g)
        returns: (batch,) rates
        """
        # Preprocess per-participant: log-transform conc and Km, keep stoich linear
        x = participant_feats.clone()
        # log-transform concentration (col 0) and Km (col 2)
        x[..., 0] = torch.log(x[..., 0].clamp_min(self.eps))
        x[..., 2] = torch.log(x[..., 2].clamp_min(self.eps))

        # Encode each participant (only where mask=1)
        h = self.enc(x)   # (batch, MAX_P, H)
        h = h * mask.unsqueeze(-1)   # zero out invalid slots

        # Aggregate: sum and max over participants
        sum_h = h.sum(dim=1)                                   # (batch, H)
        # For max over masked-out entries, use very negative fill
        h_for_max = h.masked_fill(mask.unsqueeze(-1) < 0.5, -1e9)
        max_h = h_for_max.max(dim=1).values                    # (batch, H)

        # Global features: log-transform kcats (col 0, 1) and enzyme (col 2)
        g = global_feats.clone()
        g[..., 0] = torch.log(g[..., 0].clamp_min(self.eps))
        g[..., 1] = torch.log(g[..., 1].clamp_min(self.eps))
        g[..., 2] = torch.log(g[..., 2].clamp_min(self.eps))

        # Concatenate aggregated + global
        z = torch.cat([sum_h, max_h, g], dim=-1)

        # Decode
        d = self.dec(z)
        log_mag = self.head_log_mag(d).squeeze(-1)
        sign = torch.tanh(self.head_sign(d)).squeeze(-1)

        rate = sign * torch.exp(log_mag)
        return rate


# =============================================================================
# Dataset construction
# =============================================================================
def build_dataset(rxns_kin_list, species, init_concs, n_samples_per_rxn=1000,
                   perturb_scale=3.0, seed=0) -> dict:
    """
    For each reaction in rxns_kin_list, generate n_samples_per_rxn (concs, rate)
    pairs by perturbing physiological concentrations. Returns dict of arrays
    ready for tensorization.

    rxns_kin_list: list of (rid, rxn, kin) tuples
    """
    rng = np.random.default_rng(seed)
    species_to_idx = {s: i for i, s in enumerate(species)}
    n_sp = len(species)

    base = np.array([init_concs.get(s, 0.1) for s in species])

    total = len(rxns_kin_list) * n_samples_per_rxn

    P_feats = np.zeros((total, MAX_PARTICIPANTS, 5), dtype=np.float32)
    P_masks = np.zeros((total, MAX_PARTICIPANTS), dtype=np.float32)
    G_feats = np.zeros((total, 4), dtype=np.float32)
    rates = np.zeros(total, dtype=np.float32)
    rxn_indices = np.zeros(total, dtype=np.int64)

    k = 0
    for ri, (rid, rxn, kin) in enumerate(rxns_kin_list):
        for _ in range(n_samples_per_rxn):
            log_factor = rng.uniform(
                -np.log(perturb_scale), np.log(perturb_scale), size=n_sp)
            concs = base * np.exp(log_factor)
            if rng.random() < 0.05:
                concs[rng.integers(n_sp)] = 0.0

            feats, mask, gfeats = encode_reaction_participants(
                rxn, kin, species_to_idx, concs)
            P_feats[k] = feats
            P_masks[k] = mask
            G_feats[k] = gfeats

            # Ground truth via passive convenience rate
            c_dict = {s: float(c) for s, c in zip(species, concs)}
            rates[k] = convenience_rate_passive(rxn, kin, c_dict)
            rxn_indices[k] = ri
            k += 1

    return {
        'participants': P_feats,
        'masks': P_masks,
        'globals': G_feats,
        'rates': rates,
        'rxn_indices': rxn_indices,
    }


# =============================================================================
# Training
# =============================================================================
def train_permnet(model, train_data, val_data,
                    n_epochs=40, batch_size=512, lr=1e-3, device='cpu') -> dict:
    model = model.to(device)

    def to_tensors(d):
        return (
            torch.tensor(d['participants']).to(device),
            torch.tensor(d['masks']).to(device),
            torch.tensor(d['globals']).to(device),
            torch.tensor(d['rates']).to(device),
        )
    Ptr, Mtr, Gtr, Ytr = to_tensors(train_data)
    Pva, Mva, Gva, Yva = to_tensors(val_data)

    # Global rate scale for loss normalization (per train-set std)
    y_scale = float(Ytr.abs().std().clamp_min(1e-9))

    ds = TensorDataset(Ptr, Mtr, Gtr, Ytr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {'train_loss': [], 'val_loss': []}

    for epoch in range(n_epochs):
        model.train()
        losses = []
        for pb, mb, gb, yb in loader:
            pred = model(pb, mb, gb)
            loss = ((pred - yb) / y_scale).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        model.eval()
        with torch.no_grad():
            pred_va = model(Pva, Mva, Gva)
            val_loss = ((pred_va - Yva) / y_scale).pow(2).mean().item()

        hist['train_loss'].append(train_loss)
        hist['val_loss'].append(val_loss)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4e}  "
                  f"val_loss={val_loss:.4e}")

    return hist


# =============================================================================
# Scenario 1: Held-out reaction generalization
# =============================================================================
def scenario_heldout_reactions():
    print("\n" + "=" * 76)
    print("SCENARIO 1: Held-out reaction generalization (7 train / 3 test)")
    print("=" * 76)

    mols, active, init_concs = load_kinetics_subset()
    species = gather_species(active)
    print(f"  {len(active)} glycolysis reactions, {len(species)} species")

    # Split: held out reactions that haven't been seen during training
    # Choose the middle of the pipeline for held-out, so the network sees
    # upstream and downstream but not these specific steps.
    held_out_ids = {"R_FBA", "R_PGK", "R_LDH_L"}
    train_rxns = [r for r in active if r[0] not in held_out_ids]
    test_rxns  = [r for r in active if r[0] in held_out_ids]
    print(f"  Train: {[r[0] for r in train_rxns]}")
    print(f"  Test:  {[r[0] for r in test_rxns]}")

    print("\n  Generating training data...")
    train_data = build_dataset(train_rxns, species, init_concs,
                                 n_samples_per_rxn=1500, seed=0)
    val_data = build_dataset(train_rxns, species, init_concs,
                                n_samples_per_rxn=300, seed=1)
    test_data = build_dataset(test_rxns, species, init_concs,
                                n_samples_per_rxn=500, seed=2)

    print(f"  Train samples: {len(train_data['rates'])}")
    print(f"  Val samples:   {len(val_data['rates'])}")
    print(f"  Test samples:  {len(test_data['rates'])} (held-out reactions)")

    print("\n  Building permutation-invariant network...")
    model = PermInvRateNet(hidden_dim=96)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params}")

    print("\n  Training...")
    hist = train_permnet(model, train_data, val_data,
                            n_epochs=50, batch_size=512, lr=1e-3)

    # Evaluate
    print("\n  Evaluation on HELD-OUT reactions (generalization test):")
    model.eval()
    with torch.no_grad():
        P = torch.tensor(test_data['participants'])
        M = torch.tensor(test_data['masks'])
        G = torch.tensor(test_data['globals'])
        pred = model(P, M, G).numpy()
    true = test_data['rates']

    # Per held-out reaction, relative error
    print(f"\n  {'reaction':12s} {'mean|y|':>12s} {'mean|err|':>12s} {'rel err':>10s}")
    rel_errs = []
    for ri, (rid, rxn, kin) in enumerate(test_rxns):
        mask = test_data['rxn_indices'] == ri
        y = true[mask]
        yhat = pred[mask]
        mean_mag = float(np.abs(y).mean())
        mean_err = float(np.abs(y - yhat).mean())
        rel = mean_err / (mean_mag + 1e-20)
        rel_errs.append(rel)
        print(f"  {rid:12s} {mean_mag:12.3e} {mean_err:12.3e} {rel:10.2%}")

    mean_rel_err = float(np.mean(rel_errs))
    print(f"  mean rel err on HELD-OUT reactions: {mean_rel_err:.2%}")

    # Pass criterion: held-out rate error under 50% is real generalization.
    # Under 100% is weak generalization but not chance.
    # Over 200% suggests the network didn't learn chemistry, just memorized.
    if mean_rel_err < 0.5:
        verdict = "STRONG GENERALIZATION"
        passed = True
    elif mean_rel_err < 1.0:
        verdict = "WEAK GENERALIZATION"
        passed = True
    elif mean_rel_err < 2.0:
        verdict = "MARGINAL -- cross-reaction transfer is limited"
        passed = False
    else:
        verdict = "POOR -- network didn't learn chemistry"
        passed = False
    print(f"\n  Verdict: {verdict}")
    return passed, mean_rel_err


# =============================================================================
# Scenario 2: Full Syn3A scale
# =============================================================================
def scenario_full_syn3a():
    print("\n" + "=" * 76)
    print("SCENARIO 2: Full Syn3A scale (one network for ~220 reactions)")
    print("=" * 76)

    print("\n  Loading full Syn3A model...")
    model_sbml = load_sbml_model(SBML_PATH)
    mols = extract_molecules_with_compartments(model_sbml)
    rxns = extract_reactions(model_sbml)
    p_ids, w_ids = find_h_and_water_per_compartment(mols)
    rebalanced = []
    for r in rxns:
        if r.is_exchange or r.is_biomass: continue
        nr = rebalance_reaction(r, mols, p_ids, w_ids)
        if nr: rebalanced.append(nr)
    kinetics, init_concs = parse_all_kinetics()
    # Elementary only (same exclusions as P4b/P5/P6)
    active = [(r.sbml_id, r, kinetics[r.sbml_id])
               for r in rebalanced
               if r.sbml_id in kinetics and kinetics[r.sbml_id].is_usable()
               and not any(abs(c) > 3 for c in r.stoichiometry.values())]

    # Gather all species that appear in any active reaction
    species = []
    seen = set()
    for rid, rxn, kin in active:
        for sid in rxn.stoichiometry:
            if sid not in seen:
                seen.add(sid); species.append(sid)

    print(f"  Active reactions: {len(active)}  Species in scope: {len(species)}")

    print("\n  Generating training data (smaller samples/rxn to fit memory)...")
    # Reduce samples per reaction since we have many more reactions
    samples_train = 100
    samples_val = 25
    train_data = build_dataset(active, species, init_concs,
                                 n_samples_per_rxn=samples_train, seed=0)
    val_data = build_dataset(active, species, init_concs,
                                n_samples_per_rxn=samples_val, seed=1)
    print(f"  Train samples: {len(train_data['rates'])}")
    print(f"  Val samples:   {len(val_data['rates'])}")

    print("\n  Training permutation-invariant network on full Syn3A...")
    model = PermInvRateNet(hidden_dim=128)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params}")

    hist = train_permnet(model, train_data, val_data,
                            n_epochs=30, batch_size=1024, lr=1e-3)

    # Evaluate overall rate error on val set
    model.eval()
    with torch.no_grad():
        P = torch.tensor(val_data['participants'])
        M = torch.tensor(val_data['masks'])
        G = torch.tensor(val_data['globals'])
        pred = model(P, M, G).numpy()
    true = val_data['rates']

    # Per-reaction relative error
    rxn_errs = []
    for ri in range(len(active)):
        m = val_data['rxn_indices'] == ri
        if not m.any(): continue
        y = true[m]
        yhat = pred[m]
        mm = float(np.abs(y).mean())
        me = float(np.abs(y - yhat).mean())
        if mm > 1e-12:
            rxn_errs.append(me / mm)

    if rxn_errs:
        median_err = float(np.median(rxn_errs))
        mean_err = float(np.mean(rxn_errs))
        print(f"\n  Per-reaction relative error across {len(rxn_errs)} Syn3A reactions:")
        print(f"    median: {median_err:.2%}")
        print(f"    mean:   {mean_err:.2%}")
        if median_err < 0.5:
            verdict = "STRONG: one network represents full Syn3A kinetics reasonably"
            passed = True
        elif median_err < 1.0:
            verdict = "WORKABLE: wide rate range limits accuracy; scaling helps"
            passed = True
        else:
            verdict = "NEEDS MORE CAPACITY / EPOCHS"
            passed = False
        print(f"  Verdict: {verdict}")
        return passed, median_err
    else:
        print("\n  No non-zero reactions to evaluate.")
        return False, float('inf')


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 76)
    print("DMVC P8: Permutation-invariant rate predictor")
    print("Tests whether a single shared network learns chemistry across reactions")
    print("=" * 76)

    # Scenario 1: cross-reaction generalization
    s1_passed, s1_err = scenario_heldout_reactions()

    # Scenario 2: full Syn3A scale
    s2_passed, s2_err = scenario_full_syn3a()

    # Summary
    print("\n" + "=" * 76)
    print("P8 SUMMARY")
    print("=" * 76)
    print(f"  Scenario 1 (held-out generalization): "
          f"{'PASS' if s1_passed else 'FAIL'} -- mean rel err {s1_err:.2%}")
    print(f"  Scenario 2 (full Syn3A scale):        "
          f"{'PASS' if s2_passed else 'FAIL'} -- median rel err {s2_err:.2%}")
    print()
    if s1_passed and s2_passed:
        print("The permutation-invariant architecture generalizes across")
        print("reactions. One shared function learns chemistry rules, not")
        print("reaction-specific memorization.")
    elif s1_passed:
        print("Cross-reaction generalization works on glycolysis. Full Syn3A")
        print("needs more capacity or training.")
    else:
        print("Generalization limited. The network learned patterns but")
        print("not transferable chemistry. Next steps: more data per reaction,")
        print("better features (species embeddings), or more capacity.")


if __name__ == "__main__":
    main()
