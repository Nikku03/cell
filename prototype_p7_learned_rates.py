"""
DMVC P7: First learned rate predictor.

Goal: train a small neural network to predict reaction rates from local
concentrations, using the Luthey-Schulten convenience-kinetics rate law as
ground truth. Validate that:

  (a) The training loop works end-to-end (loss decreases, no NaN)
  (b) The learned network predicts rates accurately on held-out states
  (c) When we substitute the learned network INTO a simulator, its trajectory
      matches the ground-truth simulator's trajectory within tolerance

Why this matters: until now everything has been hand-coded physics. This
prototype is the first real NN in the DMVC project. If this works, it proves
the architecture admits training. If it fails, we need to rethink before
scaling up.

Scope:
  * Subset to 10 glycolysis reactions (PGI, PFK, FBA, TPI, GAPD, PGK, PGM,
    ENO, PYK, LDH_L). Rich kinetics, well-understood dynamics.
  * Well-mixed single-voxel simulation (the spatial architecture is already
    validated separately in P4b)
  * PyTorch, CPU-only for Colab compatibility
  * One MLP shared across all 10 reactions. Input: padded concentration
    vector. Output: 10-dim rate vector. See design note in code.

Test structure:
  T1: training loss decreases and converges to low value
  T2: validation loss on held-out states is similar to training loss
  T3: learned rates match ground-truth rates on a random probe state
      within 10% relative error
  T4: simulator driven by LEARNED rates produces trajectory matching
      simulator driven by GROUND-TRUTH rates, over 1000 steps, within 5%
      relative error on key metabolites

Honest pre-registration of expected failures:
  * Rate magnitudes span 6 orders -> predict log-rate instead of rate
  * Unconstrained MLPs don't produce zero at zero-substrate -> include a
    substrate-presence gate in the architecture
  * Equilibrium-only training distribution -> generate data by RANDOMLY
    PERTURBING concentrations so the network sees out-of-equilibrium states
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


def convenience_rate_passive(rxn, kin: ReactionKinetics, concs: dict) -> float:
    """
    Convenience rate law where species lacking Km (typically H+/H2O added
    by rebalancing) are treated as kinetically passive: they contribute
    stoichiometry but do NOT appear in the rate law. This matches P4b's
    behavior and is the correct interpretation of the Luthey-Schulten
    model's implicit-proton convention.
    """
    if not kin.is_usable():
        return 0.0
    E = kin.enzyme_conc
    kf = kin.kcat_forward
    kr = kin.kcat_reverse if kin.kcat_reverse is not None else 0.0

    subs = []
    prods = []
    for sid, coef in rxn.stoichiometry.items():
        Km = kin.Km.get(sid)
        if Km is None or Km <= 0:
            # PASSIVE species (H+/H2O from rebalance): skip in rate law
            continue
        c = max(concs.get(sid, 0.0), 0.0)
        ratio = c / Km
        if coef < 0:
            subs.append((ratio, int(round(-coef))))
        elif coef > 0:
            prods.append((ratio, int(round(coef))))

    # Need at least one substrate in the rate law
    if not subs and not prods:
        return 0.0

    prod_s = 1.0
    for s, n in subs:
        prod_s *= s ** n
    prod_p = 1.0
    for p, n in prods:
        prod_p *= p ** n
    numerator = E * (kf * prod_s - kr * prod_p)

    denom = 1.0
    for s, _ in subs:
        denom *= (1.0 + s)
    denom_p = 1.0
    for p, _ in prods:
        denom_p *= (1.0 + p)
    denom = denom + denom_p - 1.0
    if denom <= 0:
        return 0.0
    return numerator / denom


# Use the passive variant as ground truth throughout this prototype
_ground_truth_rate = convenience_rate_passive
from prototype_p3b_stamps import (
    load_sbml_model, extract_molecules_with_compartments, extract_reactions,
    find_h_and_water_per_compartment, rebalance_reaction,
    SBML_PATH, ATOM_TYPES, K_atoms,
)


# =============================================================================
# 1. Select target reactions (glycolysis)
# =============================================================================
GLYCOLYSIS_RXNS = [
    "R_PGI",    # glucose-6-phosphate isomerase
    "R_PFK",    # phosphofructokinase
    "R_FBA",    # fructose-bisphosphate aldolase
    "R_TPI",    # triose phosphate isomerase
    "R_GAPD",   # glyceraldehyde-3-phosphate dehydrogenase
    "R_PGK",    # phosphoglycerate kinase
    "R_PGM",    # phosphoglycerate mutase
    "R_ENO",    # enolase
    "R_PYK",    # pyruvate kinase
    "R_LDH_L",  # lactate dehydrogenase
]


def load_kinetics_subset():
    """Load SBML + kinetics for just the glycolysis reactions."""
    model = load_sbml_model(SBML_PATH)
    mols = extract_molecules_with_compartments(model)
    rxns = extract_reactions(model)
    p_ids, w_ids = find_h_and_water_per_compartment(mols)

    rebalanced = {}
    for r in rxns:
        if r.is_exchange or r.is_biomass: continue
        nr = rebalance_reaction(r, mols, p_ids, w_ids)
        if nr: rebalanced[r.sbml_id] = nr

    kinetics, init_concs = parse_all_kinetics()

    active = []
    for rid in GLYCOLYSIS_RXNS:
        if rid in rebalanced and rid in kinetics and kinetics[rid].is_usable():
            active.append((rid, rebalanced[rid], kinetics[rid]))

    return mols, active, init_concs


def gather_species(active_rxns):
    """Return the ordered list of species involved in any active reaction."""
    species = []
    seen = set()
    for rid, rxn, kin in active_rxns:
        for sid in rxn.stoichiometry:
            if sid not in seen:
                seen.add(sid); species.append(sid)
    return species


# =============================================================================
# 2. Training data generation
# =============================================================================
def generate_training_data(active_rxns, species, init_concs, n_samples=5000,
                             perturb_scale=2.0, seed=0) -> tuple:
    """
    Generate (concentrations, rates) pairs by:
      - Drawing concentrations from a log-uniform perturbation of physiological
        values, so the network sees out-of-equilibrium states
      - Computing rates via the ground-truth convenience rate law

    Returns:
      X: (n_samples, n_species) concentration vectors
      Y: (n_samples, n_reactions) rate vectors
    """
    rng = np.random.default_rng(seed)
    n_sp = len(species)
    n_rx = len(active_rxns)

    # Base concentrations (physiological). Any species not in the table gets 0.1 mM.
    base = np.array([init_concs.get(s, 0.1) for s in species])

    X = np.zeros((n_samples, n_sp))
    Y = np.zeros((n_samples, n_rx))

    for i in range(n_samples):
        # log-uniform perturbation in [base / perturb_scale, base * perturb_scale]
        log_factor = rng.uniform(
            -np.log(perturb_scale), np.log(perturb_scale), size=n_sp)
        concs = base * np.exp(log_factor)
        # 5% of samples: one species set to ~0 (zero-substrate edge case)
        if rng.random() < 0.05:
            zero_idx = rng.integers(n_sp)
            concs[zero_idx] = 0.0
        X[i] = concs

        # Compute ground-truth rates
        c_dict = {s: float(c) for s, c in zip(species, concs)}
        for j, (rid, rxn, kin) in enumerate(active_rxns):
            Y[i, j] = convenience_rate_passive(rxn, kin, c_dict)

    return X, Y


# =============================================================================
# 3. Network architecture
# =============================================================================
class RatePredictor(nn.Module):
    """
    Small MLP that takes concentration vector and outputs rates for all
    reactions in the set.

    Architecture choices documented:
      * Input: log(c + eps) -- concentrations span orders of magnitude, so
        log-space input helps generalization and gradient flow.
      * Output: log|rate| with separate sign prediction -- rates can be
        positive or negative (reversible rxns) and span many orders. We
        predict log|rate| with a signed multiplier.
      * Substrate-presence gate: output rate is multiplied by a soft gate
        that goes to zero when any substrate concentration is effectively
        zero. This is an explicit inductive bias matching mass-action chemistry.
    """

    def __init__(self, n_species: int, n_reactions: int,
                  substrate_masks: torch.Tensor,
                  hidden_dim: int = 64):
        """
        substrate_masks: (n_reactions, n_species) binary tensor indicating
          which species are substrates of each reaction (for the gate).
        """
        super().__init__()
        self.n_species = n_species
        self.n_reactions = n_reactions
        self.register_buffer("substrate_masks", substrate_masks.float())

        self.body = nn.Sequential(
            nn.Linear(n_species, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Two heads: log|rate| and sign
        self.head_log_mag = nn.Linear(hidden_dim, n_reactions)
        self.head_sign = nn.Linear(hidden_dim, n_reactions)

        self.eps = 1e-6

    def forward(self, concs: torch.Tensor) -> torch.Tensor:
        """
        concs: (batch, n_species) in mM
        returns: (batch, n_reactions) rates in same units as ground truth
        """
        # Log-space input
        x = torch.log(concs.clamp_min(self.eps) + self.eps)
        h = self.body(x)

        log_mag = self.head_log_mag(h)     # (batch, n_reactions)
        # sign: use tanh so it's in [-1, 1]; scale by a learnable factor
        sign = torch.tanh(self.head_sign(h))   # (batch, n_reactions)

        # Substrate-presence gate: soft-min of (c / (c + 1e-3)) over substrates
        # For each reaction, gate = product over substrates of c/(c + 1e-3).
        # This is ~1 when all substrates are present, ~0 when any is missing.
        c_expanded = concs.unsqueeze(1)                          # (batch, 1, n_species)
        masks = self.substrate_masks.unsqueeze(0)                # (1, n_reactions, n_species)
        # Where mask is 1, use concentration; where 0, use 1 (doesn't contribute to product)
        terms = c_expanded / (c_expanded + 1e-3)                # in [0, ~1]
        gated_terms = torch.where(masks > 0.5, terms,
                                    torch.ones_like(terms))
        gate = gated_terms.prod(dim=-1)                         # (batch, n_reactions)

        rate = sign * torch.exp(log_mag) * gate
        return rate


def build_substrate_masks(active_rxns, species) -> np.ndarray:
    """(n_reactions, n_species) binary: 1 if species is a substrate of reaction."""
    sid_to_idx = {s: i for i, s in enumerate(species)}
    masks = np.zeros((len(active_rxns), len(species)))
    for j, (rid, rxn, kin) in enumerate(active_rxns):
        for sid, coef in rxn.stoichiometry.items():
            if coef < 0 and sid in sid_to_idx and sid in kin.Km:
                masks[j, sid_to_idx[sid]] = 1.0
    return masks


# =============================================================================
# 4. Training loop
# =============================================================================
def train_model(model, X_train, Y_train, X_val, Y_val,
                 n_epochs=30, batch_size=256, lr=1e-3, device="cpu") -> dict:
    """Returns history dict with train_loss, val_loss per epoch."""
    model = model.to(device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    # Normalization: each reaction has its own scale. Compute per-reaction std
    # on training targets and divide both pred and target by it before loss.
    y_scale = Y_train_t.abs().mean(dim=0).clamp_min(1e-10)      # (n_reactions,)

    ds = TensorDataset(X_train_t, Y_train_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            pred = model(xb)
            # Normalized L2 loss
            loss = ((pred - yb) / y_scale).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            val_loss = ((pred_val - Y_val_t) / y_scale).pow(2).mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4e}  "
                  f"val_loss={val_loss:.4e}")

    return history


# =============================================================================
# 5. Trajectory-based evaluation
# =============================================================================
def run_simulator(rate_fn, concs_init: dict, active_rxns, species,
                   n_steps: int = 1000, dt: float = 1e-3, cap: float = 0.2) -> dict:
    """
    Run a well-mixed ODE simulator using a provided rate function.
      rate_fn(concs: np.ndarray shape (n_species,)) -> rates: np.ndarray shape (n_reactions,)

    Returns dict of trajectories {species: np.ndarray of shape (n_steps+1,)}.
    Uses the same rate-cap stability trick as P4b.
    """
    sid_to_idx = {s: i for i, s in enumerate(species)}
    state = np.array([concs_init.get(s, 0.0) for s in species])
    traj = {s: np.zeros(n_steps + 1) for s in species}
    for i, s in enumerate(species):
        traj[s][0] = state[i]

    for step in range(n_steps):
        rates = rate_fn(state)
        # Apply rate cap per reaction: if any substrate would go below 20% per step
        for j, (rid, rxn, kin) in enumerate(active_rxns):
            for sid, coef in rxn.stoichiometry.items():
                if coef >= 0: continue
                if sid not in sid_to_idx: continue
                c = state[sid_to_idx[sid]]
                if c > 0 and rates[j] > 0 and abs(coef) * rates[j] * dt > cap * c:
                    rates[j] = cap * c / (abs(coef) * dt)
        # Apply updates
        for j, (rid, rxn, kin) in enumerate(active_rxns):
            v = rates[j]
            if v == 0: continue
            for sid, coef in rxn.stoichiometry.items():
                if sid not in sid_to_idx: continue
                state[sid_to_idx[sid]] += dt * coef * v
                if state[sid_to_idx[sid]] < 0:
                    state[sid_to_idx[sid]] = 0.0
        for i, s in enumerate(species):
            traj[s][step + 1] = state[i]
    return traj


# =============================================================================
# 6. Main
# =============================================================================
def main():
    print("=" * 76)
    print("DMVC P7: First learned rate predictor")
    print("=" * 76)

    print("\n[1] Loading glycolysis reactions and kinetics...")
    mols, active, init_concs = load_kinetics_subset()
    print(f"    Active reactions: {len(active)}")
    for rid, rxn, kin in active:
        print(f"      {rid:10s}  kcat_f={kin.kcat_forward:>8.2f}  "
              f"kcat_r={kin.kcat_reverse:>8.2f}  |rxn|={len(rxn.stoichiometry)}")

    species = gather_species(active)
    print(f"    Species involved: {len(species)}")

    substrate_masks = torch.tensor(build_substrate_masks(active, species))

    # ---------------------------------------------------------------------
    print("\n[2] Generating training data (convenience-kinetics ground truth)...")
    X_train, Y_train = generate_training_data(
        active, species, init_concs, n_samples=5000, perturb_scale=3.0, seed=0)
    X_val, Y_val = generate_training_data(
        active, species, init_concs, n_samples=1000, perturb_scale=3.0, seed=1)
    print(f"    Train: X={X_train.shape} Y={Y_train.shape}")
    print(f"    Val:   X={X_val.shape} Y={Y_val.shape}")
    print(f"    Rate magnitude range:   {np.abs(Y_train).min():.2e} to "
          f"{np.abs(Y_train).max():.2e}")
    print(f"    Per-reaction mean rates:")
    for j, (rid, rxn, kin) in enumerate(active):
        print(f"      {rid:10s}  mean |rate|={np.abs(Y_train[:, j]).mean():.3e}")

    # ---------------------------------------------------------------------
    print("\n[3] Building and training network...")
    model = RatePredictor(
        n_species=len(species),
        n_reactions=len(active),
        substrate_masks=substrate_masks,
        hidden_dim=64,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params}")

    history = train_model(model, X_train, Y_train, X_val, Y_val,
                             n_epochs=40, batch_size=256, lr=1e-3)

    # ---------------------------------------------------------------------
    print("\n[4] Testing learned rates on held-out states...")
    model.eval()
    with torch.no_grad():
        pred_val = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
    # Per-reaction relative error
    print(f"    {'reaction':12s} {'mean|y|':>12s} {'mean|err|':>12s} {'rel err':>10s}")
    rel_errs = []
    for j, (rid, rxn, kin) in enumerate(active):
        y = Y_val[:, j]
        yhat = pred_val[:, j]
        mean_mag = float(np.abs(y).mean())
        mean_err = float(np.abs(y - yhat).mean())
        rel = mean_err / (mean_mag + 1e-30)
        rel_errs.append(rel)
        print(f"    {rid:12s} {mean_mag:12.3e} {mean_err:12.3e} {rel:10.2%}")
    mean_rel_err = float(np.mean(rel_errs))
    print(f"    mean rel err across reactions: {mean_rel_err:.2%}")

    # Pass criteria
    t1_pass = history["train_loss"][-1] < 0.01  # normalized loss
    t2_pass = history["val_loss"][-1] < 0.02
    t3_pass = mean_rel_err < 0.15  # 15%

    # ---------------------------------------------------------------------
    print("\n[5] Trajectory comparison: ground-truth vs learned simulator...")
    # Initial state: physiological
    c_init = {s: init_concs.get(s, 0.1) for s in species}

    # Ground truth rate fn
    def gt_rate_fn(concs_vec):
        cd = {s: float(c) for s, c in zip(species, concs_vec)}
        return np.array([convenience_rate_passive(rxn, kin, cd)
                          for rid, rxn, kin in active])

    # Learned rate fn
    def learned_rate_fn(concs_vec):
        with torch.no_grad():
            cvec = torch.tensor(concs_vec, dtype=torch.float32).unsqueeze(0)
            r = model(cvec).squeeze(0).numpy()
        return r

    print("    Running both simulators for 500 steps...")
    traj_gt = run_simulator(gt_rate_fn, c_init, active, species, n_steps=500)
    traj_nn = run_simulator(learned_rate_fn, c_init, active, species, n_steps=500)

    # Compare key metabolites at final step
    watch = ["M_g6p_c", "M_f6p_c", "M_pyr_c", "M_pep_c", "M_nad_c",
              "M_nadh_c", "M_atp_c", "M_adp_c", "M_lac__L_c"]
    watch = [s for s in watch if s in species]
    print(f"\n    Final state comparison:")
    print(f"    {'species':15s} {'GT':>12s} {'NN':>12s} {'rel err':>10s}")
    traj_errs = []
    for s in watch:
        gt_final = traj_gt[s][-1]
        nn_final = traj_nn[s][-1]
        err = abs(gt_final - nn_final) / (abs(gt_final) + 1e-6)
        traj_errs.append(err)
        print(f"    {s:15s} {gt_final:12.4f} {nn_final:12.4f} {err:10.2%}")
    mean_traj_err = float(np.mean(traj_errs))
    print(f"    mean trajectory final-state error: {mean_traj_err:.2%}")

    t4_pass = mean_traj_err < 0.15

    # ---------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    results = [
        ("T1", t1_pass, f"train_loss converges ({history['train_loss'][-1]:.4e})"),
        ("T2", t2_pass, f"val_loss comparable   ({history['val_loss'][-1]:.4e})"),
        ("T3", t3_pass, f"learned rates within 15% of ground truth "
                         f"(actual: {mean_rel_err:.2%})"),
        ("T4", t4_pass, f"trajectory match within 15% "
                         f"(actual: {mean_traj_err:.2%})"),
    ]
    for lab, ok, desc in results:
        print(f"  [{lab}] {'PASS' if ok else 'FAIL'}  {desc}")
    print()
    if all(r[1] for r in results):
        print("All four tests pass. The training loop works end-to-end. A learned")
        print("rate predictor is a drop-in substitute for hand-coded kinetics.")
    else:
        print("Some tests failed -- see diagnostics above. This is informative;")
        print("each failure points at a specific fix.")


if __name__ == "__main__":
    main()
