"""Trainer for the v3 potential (real chemistry + pose-quality head)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .mmff_dataset import ChemConfiguration
from .multimode_v3 import MultiModePotentialV3


@dataclass
class TrainV3Config:
    epochs: int = 40
    lr: float = 1e-3
    batch_size: int = 8
    force_weight: float = 0.05      # forces are bigger numbers in MMFF; downweight
    energy_weight: float = 1.0
    pose_weight: float = 0.3
    grad_clip: float = 5.0
    log_every: int = 5


def sample_kwargs(c: ChemConfiguration) -> dict:
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
        "vdw_radius": c.vdw_radius,
        "bonds": c.bonds,
        "bond_order": c.bond_order,
        "bond_in_ring": c.bond_in_ring,
        "bond_conjugated": c.bond_conjugated,
        "bond_rotatable": c.bond_rotatable,
    }


def _fit_norm(model: MultiModePotentialV3, configs: list[ChemConfiguration]) -> None:
    n_atoms = sum(c.n_atoms for c in configs)
    e_sum = sum(c.energy.item() for c in configs)
    mean = e_sum / n_atoms
    spread = torch.tensor([c.energy.item() / c.n_atoms for c in configs])
    std = float(spread.std().clamp_min(1.0))
    model.fit_normalization(mean, std)


def _loss(
    model: MultiModePotentialV3,
    c: ChemConfiguration,
    ew: float, fw: float, pw: float,
) -> torch.Tensor:
    kw = sample_kwargs(c)
    pos = kw.pop("positions").clone().requires_grad_(True)
    E_pred, _, pose_logit = model.forward(pos, **kw)
    (g,) = torch.autograd.grad(E_pred, pos, create_graph=True, allow_unused=True)
    if g is None:
        g = torch.zeros_like(pos)
    F_pred = -g
    n = float(c.n_atoms)
    e_err = (E_pred - c.energy) / n
    f_err = F_pred - c.forces
    loss = ew * e_err.pow(2) + fw * f_err.pow(2).mean()
    if c.pose_label != -1:
        target = torch.tensor(float(c.pose_label))
        loss = loss + pw * F.binary_cross_entropy_with_logits(pose_logit, target)
    return loss


def train_v3(
    model: MultiModePotentialV3,
    train_set: list[ChemConfiguration],
    val_set: list[ChemConfiguration] | None = None,
    cfg: TrainV3Config | None = None,
    seed: int = 0,
    quiet: bool = False,
) -> dict:
    cfg = cfg or TrainV3Config()
    _fit_norm(model, train_set)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = torch.Generator().manual_seed(seed)
    history = {"train_loss": [], "val_e_mae": [], "val_f_mae": [], "val_pose_acc": []}
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
                losses.append(_loss(model, train_set[k],
                                    cfg.energy_weight, cfg.force_weight, cfg.pose_weight))
            total = torch.stack(losses).mean()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += total.item()
            n_batches += 1

        history["train_loss"].append(running / max(1, n_batches))
        if val_set is not None:
            e_mae, f_mae, pose_acc = evaluate(model, val_set)
            history["val_e_mae"].append(e_mae)
            history["val_f_mae"].append(f_mae)
            history["val_pose_acc"].append(pose_acc)
            if not quiet and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
                print(f"  epoch {epoch:>3d}  loss={history['train_loss'][-1]:.3f}  "
                      f"val_E_MAE={e_mae:.2f}  val_F_MAE={f_mae:.2f}  "
                      f"pose_acc={pose_acc:.3f}  ({time.time()-t0:.1f}s)")
        elif not quiet and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:>3d}  loss={history['train_loss'][-1]:.3f}  "
                  f"({time.time()-t0:.1f}s)")
    return history


def evaluate(
    model: MultiModePotentialV3, configs: list[ChemConfiguration]
) -> tuple[float, float, float]:
    """Returns (E_MAE, F_MAE, pose_classification_accuracy)."""
    model.eval()
    e_abs = 0.0
    f_abs = 0.0
    f_count = 0
    pose_correct = 0
    pose_total = 0
    for c in configs:
        kw = sample_kwargs(c)
        pos = kw.pop("positions").clone().requires_grad_(True)
        E, _, pose_logit = model.forward(pos, **kw)
        (g,) = torch.autograd.grad(E, pos, allow_unused=True)
        if g is None:
            g = torch.zeros_like(pos)
        Fp = -g
        e_abs += abs(E.item() - c.energy.item())
        f_abs += (Fp - c.forces).abs().sum().item()
        f_count += c.forces.numel()
        if c.pose_label != -1:
            pred = int(pose_logit.item() > 0)
            if pred == c.pose_label:
                pose_correct += 1
            pose_total += 1
    pose_acc = pose_correct / pose_total if pose_total > 0 else 0.0
    return e_abs / len(configs), f_abs / max(1, f_count), pose_acc
