"""Training loop for the SchNet-style potential.

We fit the network to reproduce both energies and forces of an analytic
reference (LJ here, but the trainer doesn't know that). Both targets are
needed: energy alone leaves a large family of curl-free vector fields that
produce the same scalar, and force alone is invariant up to an additive
constant - jointly they pin down the potential well.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from .dataset import Configuration
from .network import ArtificialAtomPotential


@dataclass
class TrainConfig:
    epochs: int = 80
    lr: float = 3e-3
    batch_size: int = 8
    force_weight: float = 1.0
    energy_weight: float = 1.0
    grad_clip: float = 5.0
    weight_decay: float = 0.0
    log_every: int = 5


def _fit_norm_to_data(model: ArtificialAtomPotential, configs: list[Configuration]) -> None:
    """Set the model's output normalisation so that an *untrained* network
    predicts in roughly the right energy range. The network learns a
    deviation around this rather than the full energy magnitude."""
    n_atoms = sum(c.n_atoms for c in configs)
    e_sum = sum(c.energy.item() for c in configs)
    atomwise_mean = e_sum / n_atoms
    # Std of per-atom-equivalent energies: spread of (E / n_atoms).
    spread = torch.tensor([c.energy.item() / c.n_atoms for c in configs])
    atomwise_std = float(spread.std().clamp_min(0.1))
    model.fit_normalization(atomwise_mean=atomwise_mean, atomwise_std=atomwise_std)


def _loss_for_config(
    model: ArtificialAtomPotential,
    c: Configuration,
    force_weight: float,
    energy_weight: float,
) -> tuple[torch.Tensor, float, float]:
    pos = c.positions.clone().requires_grad_(True)
    E_pred = model(pos, c.Z)
    # allow_unused: some isolated-atom configs have no edges, so E
    # depends only on element embeddings and grad-by-position is zero.
    (grad_pos,) = torch.autograd.grad(
        E_pred, pos, create_graph=True, allow_unused=True
    )
    if grad_pos is None:
        grad_pos = torch.zeros_like(pos)
    F_pred = -grad_pos

    # Energy loss is per-atom so configs with different N contribute fairly.
    n = float(c.n_atoms)
    e_err = (E_pred - c.energy) / n
    f_err = F_pred - c.forces
    loss = energy_weight * e_err.pow(2) + force_weight * f_err.pow(2).mean()
    return loss, e_err.detach().abs().item() * n, f_err.detach().abs().mean().item()


def train(
    model: ArtificialAtomPotential,
    train_set: list[Configuration],
    val_set: list[Configuration] | None = None,
    cfg: TrainConfig | None = None,
    quiet: bool = False,
) -> dict:
    cfg = cfg or TrainConfig()
    _fit_norm_to_data(model, train_set)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = {"train_loss": [], "val_e_mae": [], "val_f_mae": []}

    rng = torch.Generator().manual_seed(0)
    n = len(train_set)
    for epoch in range(cfg.epochs):
        t0 = time.time()
        perm = torch.randperm(n, generator=rng).tolist()
        running = 0.0
        n_batches = 0
        for b in range(0, n, cfg.batch_size):
            batch_ids = perm[b:b + cfg.batch_size]
            opt.zero_grad()
            batch_losses = []
            for k in batch_ids:
                loss, _, _ = _loss_for_config(
                    model, train_set[k], cfg.force_weight, cfg.energy_weight
                )
                batch_losses.append(loss)
            total = torch.stack(batch_losses).mean()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += total.item()
            n_batches += 1
        train_loss = running / max(1, n_batches)
        history["train_loss"].append(train_loss)

        if val_set is not None:
            e_mae, f_mae = evaluate(model, val_set)
            history["val_e_mae"].append(e_mae)
            history["val_f_mae"].append(f_mae)
            if not quiet and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
                dt = time.time() - t0
                print(f"  epoch {epoch:>3d}  train_loss={train_loss:.4f}  "
                      f"val_E_MAE={e_mae:.4f}  val_F_MAE={f_mae:.4f}  ({dt:.1f}s)")
        elif not quiet and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:>3d}  train_loss={train_loss:.4f}  ({time.time()-t0:.1f}s)")

    return history


@torch.no_grad()  # would block autograd; we override inside
def _no_op():
    pass


def evaluate(model: ArtificialAtomPotential, configs: list[Configuration]) -> tuple[float, float]:
    """Mean absolute energy and force errors on a list of configs."""
    model.eval()
    n_atoms = 0
    e_abs = 0.0
    f_abs = 0.0
    f_count = 0
    for c in configs:
        pos = c.positions.clone().requires_grad_(True)
        E = model(pos, c.Z)
        (g,) = torch.autograd.grad(E, pos, allow_unused=True)
        if g is None:
            g = torch.zeros_like(pos)
        F = -g
        e_abs += abs(E.item() - c.energy.item())
        f_abs += (F - c.forces).abs().sum().item()
        n_atoms += c.n_atoms
        f_count += c.forces.numel()
    return e_abs / len(configs), f_abs / f_count
