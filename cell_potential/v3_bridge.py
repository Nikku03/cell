"""Bridge from `cell_potential` ParticleState to the v3 ArtificialAtomPotential.

The original v3 expects per-atom inputs (positions, Z) and returns total
energy / autograd-derived forces. cell_potential particles carry
(positions, charges, masses) but no atomic number. The simplest viable
mapping: treat every particle as the same element Z (default = 6,
carbon) and let v3 compute its learned short-range interactions.

The model is trained inside the bridge on the same Lennard-Jones-style
reference that v3 was originally trained on (the artificial_atom.dataset
module), so it learns to imitate LJ between equal-Z atoms. Phase 3's
verification then checks that
  * v3's symmetries hold on cell_potential geometries,
  * v3 produces forces in roughly the right magnitude vs the analytic
    LJ reference,
  * the hybrid (v3 short-range + field long-range) is stable many-body.
"""

from __future__ import annotations

import time

import torch

from artificial_atom.dataset import generate_dataset
from artificial_atom.network import ArtificialAtomPotential, PotentialConfig
from artificial_atom.train import TrainConfig, evaluate, train


def build_and_train_v3(
    epochs: int = 20,
    n_train: int = 250,
    n_val: int = 60,
    d_h: int = 32,
    d_rbf: int = 16,
    n_layers: int = 2,
    seed: int = 0,
    verbose: bool = False,
) -> ArtificialAtomPotential:
    """Build a small v3 model and train it on the LJ reference dataset."""
    torch.manual_seed(seed)
    model = ArtificialAtomPotential(
        PotentialConfig(d_h=d_h, d_rbf=d_rbf, n_layers=n_layers)
    )
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  v3 has {n_params:,} params")
    train_set = generate_dataset(n_configs=n_train, seed=seed)
    val_set = generate_dataset(n_configs=n_val, seed=seed + 1)
    train(model, train_set, val_set,
          TrainConfig(epochs=epochs, lr=3e-3, batch_size=8, log_every=epochs),
          quiet=not verbose)
    return model


def state_to_v3_inputs(state, default_Z: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a cell_potential ParticleState to (positions, Z) for v3.forward."""
    positions = state.positions.clone()
    Z = torch.full((state.n,), int(default_Z), dtype=torch.long,
                    device=state.positions.device)
    return positions, Z


def v3_force_and_energy(
    state, model: ArtificialAtomPotential,
    default_Z: int = 6,
    use_pbc: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (force, energy) from v3 evaluated on a ParticleState.

    Forces come from autograd through v3's scalar energy, so they are
    *conservative* by construction (same property as Phase 2's LJ, but
    learned).

    If `use_pbc` is True, distances are computed under the minimum-image
    convention using `state.box_size`. Otherwise absolute distances are
    used (v3's original behavior).
    """
    positions, Z = state_to_v3_inputs(state, default_Z=default_Z)
    box = state.box_size if use_pbc else None
    E, F = model.energy_and_forces(positions, Z=Z, box_size=box)
    return F, E
