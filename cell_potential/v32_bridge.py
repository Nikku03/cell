"""Bridge from `cell_potential` ParticleState to the v3.2 MultiModePotentialV3.

v3.2 is the multi-mode, chemistry-aware extension of v3 (see
`artificial_atom.multimode_v3`). It accepts per-atom and per-bond
chemistry features: hybridisation, formal charge, aromatic/ring flags,
H-bond donor/acceptor, partial charge, vdW radius; and bond order, ring,
conjugated, rotatable flags.

cell_potential particles only carry (positions, charges, masses), so the
bridge supplies plausible defaults:
  Z = 6 (carbon), hybridisation = SP3, no rings, no aromatic, no H-bond
  partial_charge = 0 (the field handles long-range Coulomb separately)
  vdW radius = 1.7 (carbon default)
  no bonds (cell_potential particles are not chemistry-bonded)

With these defaults, v3.2 effectively runs its non-bonded short-range
channel and the bonded channel contributes zero. The result is a richer
short-range kernel than v3 that still composes correctly with the
field-mediated long-range Coulomb.

PBC: v3.2's `radius_edges_excluding` does not currently accept a box
size. For cell_potential geometries that don't wrap, this is fine. PBC
in v3.2 is a future extension (mirror the v3 patch).
"""

from __future__ import annotations

import torch

from artificial_atom.mmff_dataset import generate_chem_dataset
from artificial_atom.multimode_v3 import MultiModePotentialV3, MultiModeV3Config
from artificial_atom.train_v3 import TrainV3Config, evaluate, train_v3
from artificial_atom.unit import HYB_SP3


def build_and_train_v32(
    epochs: int = 15,
    n_train: int = 250,
    n_val: int = 60,
    d_h: int = 40,
    d_rbf: int = 20,
    n_layers: int = 2,
    seed: int = 0,
    verbose: bool = False,
) -> MultiModePotentialV3:
    """Build a small v3.2 model and train it on the rich (Morse + LJ) reference."""
    torch.manual_seed(seed)
    model = MultiModePotentialV3(MultiModeV3Config(
        d_h=d_h, d_rbf=d_rbf, n_layers=n_layers,
        r_cutoff=6.0, r_cutoff_electro=8.0,
    ))
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  v3.2 has {n_params:,} params")
    # NB: generate_chem_dataset returns at most n_configs but the SMILES
    # set has a hard ceiling, so we may get fewer. Slice if needed.
    train_set = generate_chem_dataset(seed=seed)[:n_train]
    val_set = generate_chem_dataset(seed=seed + 1)[:n_val]
    train_v3(
        model, train_set, val_set,
        TrainV3Config(
            epochs=epochs, lr=3e-4, batch_size=8,
            log_every=epochs, force_weight=0.02, energy_weight=1.0,
            pose_weight=0.0,        # no pose labels here; disable the BCE term
            bad_pose_weight=1.0,
        ),
        seed=seed,
        quiet=not verbose,
    )
    return model


def state_to_v32_inputs(state, default_Z: int = 6) -> dict:
    """Build a kwargs dict for v3.2.forward from a ParticleState.

    All chemistry features default to "generic neutral carbon"; the
    bonded channel sees an empty bond list. Per-atom partial charge is
    set to zero so the v3.2 electrostatic channel does NOT duplicate the
    field-mediated long-range Coulomb (the field path owns that).
    """
    N = state.n
    device = state.positions.device
    return {
        "positions": state.positions,
        "Z": torch.full((N,), int(default_Z), dtype=torch.long, device=device),
        "hybridisation": torch.full((N,), HYB_SP3, dtype=torch.long, device=device),
        "formal_charge": torch.zeros(N, dtype=torch.long, device=device),
        "aromatic": torch.zeros(N, dtype=torch.bool, device=device),
        "in_ring": torch.zeros(N, dtype=torch.bool, device=device),
        "h_donor": torch.zeros(N, dtype=torch.bool, device=device),
        "h_acceptor": torch.zeros(N, dtype=torch.bool, device=device),
        "partial_charge": torch.zeros(N, dtype=state.positions.dtype, device=device),
        "vdw_radius": torch.full((N,), 1.7, dtype=state.positions.dtype, device=device),
        "bonds": torch.empty(0, 2, dtype=torch.long, device=device),
        "bond_order": torch.empty(0, dtype=state.positions.dtype, device=device),
        "bond_in_ring": torch.empty(0, dtype=torch.bool, device=device),
        "bond_conjugated": torch.empty(0, dtype=torch.bool, device=device),
        "bond_rotatable": torch.empty(0, dtype=torch.bool, device=device),
    }


def v32_force_and_energy(
    state, model: MultiModePotentialV3,
    default_Z: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (force, energy) from v3.2 evaluated on a ParticleState.

    Forces come from autograd through v3.2's scalar energy.
    """
    kw = state_to_v32_inputs(state, default_Z=default_Z)
    pos = kw.pop("positions").detach().clone().requires_grad_(True)
    E, _, _ = model.forward(pos, **kw)
    (grad,) = torch.autograd.grad(E, pos, allow_unused=True)
    if grad is None:
        grad = torch.zeros_like(pos)
    return -grad.detach(), E.detach()
