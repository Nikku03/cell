"""Trainer for the multi-mode (rich) potential."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .dataset import RichConfiguration
from .multimode import MultiModePotential


def sample_to_kwargs(c: RichConfiguration) -> dict:
    """Pack a RichConfiguration into a kwargs dict for MultiModePotential.forward."""
    return {
        "positions": c.positions,
        "Z": c.Z,
        "hybridisation": c.hybridisation,
        "formal_charge": c.formal_charge,
        "aromatic": c.aromatic,
        "in_ring": c.in_ring,
        "h_donor": c.h_donor,
        "h_acceptor": c.h_acceptor,
        "partial_charge": c.partial_charge,
        "bonds": c.bonds,
        "bond_type": c.bond_type,
        "energy": c.energy,
        "forces": c.forces,
    }


@dataclass
class TrainV2Config:
    epochs: int = 60
    lr: float = 3e-3
    batch_size: int = 8
    force_weight: float = 1.0
    energy_weight: float = 1.0
    grad_clip: float = 5.0
    log_every: int = 5


def _fit_norm(model: MultiModePotential, configs: list[RichConfiguration]) -> None:
    n_atoms = sum(c.n_atoms for c in configs)
    e_sum = sum(c.energy.item() for c in configs)
    mean = e_sum / n_atoms
    spread = torch.tensor([c.energy.item() / c.n_atoms for c in configs])
    std = float(spread.std().clamp_min(0.1))
    model.fit_normalization(mean, std)


def _loss(model: MultiModePotential, c: RichConfiguration,
          fw: float, ew: float) -> torch.Tensor:
    kw = sample_to_kwargs(c)
    pos = kw.pop("positions").clone().requires_grad_(True)
    E_true = kw.pop("energy")
    F_true = kw.pop("forces")
    E_pred, _ = model.forward(pos, **kw)
    (g,) = torch.autograd.grad(E_pred, pos, create_graph=True, allow_unused=True)
    if g is None:
        g = torch.zeros_like(pos)
    F_pred = -g
    n = float(c.n_atoms)
    e_err = (E_pred - E_true) / n
    f_err = F_pred - F_true
    return ew * e_err.pow(2) + fw * f_err.pow(2).mean()


def train_one(
    model: MultiModePotential,
    train_set: list[RichConfiguration],
    val_set: list[RichConfiguration] | None = None,
    cfg: TrainV2Config | None = None,
    seed: int = 0,
    quiet: bool = False,
) -> dict:
    cfg = cfg or TrainV2Config()
    _fit_norm(model, train_set)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = torch.Generator().manual_seed(seed)
    history = {"train_loss": [], "val_e_mae": [], "val_f_mae": []}
    n = len(train_set)
    for epoch in range(cfg.epochs):
        t0 = time.time()
        perm = torch.randperm(n, generator=rng).tolist()
        running = 0.0
        n_batches = 0
        model.train()
        for b in range(0, n, cfg.batch_size):
            opt.zero_grad()
            losses = []
            for k in perm[b:b + cfg.batch_size]:
                losses.append(_loss(model, train_set[k], cfg.force_weight, cfg.energy_weight))
            total = torch.stack(losses).mean()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += total.item()
            n_batches += 1
        history["train_loss"].append(running / max(1, n_batches))
        if val_set is not None:
            e_mae, f_mae = evaluate(model, val_set)
            history["val_e_mae"].append(e_mae)
            history["val_f_mae"].append(f_mae)
            if not quiet and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
                print(f"  epoch {epoch:>3d}  loss={history['train_loss'][-1]:.4f}  "
                      f"val_E_MAE={e_mae:.4f}  val_F_MAE={f_mae:.4f}  "
                      f"({time.time()-t0:.1f}s)")
        elif not quiet and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:>3d}  loss={history['train_loss'][-1]:.4f}  "
                  f"({time.time()-t0:.1f}s)")
    return history


def evaluate(
    model: MultiModePotential, configs: list[RichConfiguration]
) -> tuple[float, float]:
    model.eval()
    e_err = 0.0
    f_err = 0.0
    f_count = 0
    for c in configs:
        kw = sample_to_kwargs(c)
        pos = kw.pop("positions").clone().requires_grad_(True)
        E_true = kw.pop("energy")
        F_true = kw.pop("forces")
        E, _ = model.forward(pos, **kw)
        (g,) = torch.autograd.grad(E, pos, allow_unused=True)
        if g is None:
            g = torch.zeros_like(pos)
        F = -g
        e_err += abs(E.item() - E_true.item())
        f_err += (F - F_true).abs().sum().item()
        f_count += F_true.numel()
    return e_err / len(configs), f_err / max(1, f_count)
