"""FastPairKernel: a tiny pair potential distilled from v3.

The idea: v3 is a graph neural network with message passing -- expensive
because it does multiple forward passes through neural layers per atom.
For systems where the short-range physics is fundamentally pairwise
(LJ, screened Coulomb, simple chemistry), most of v3's accuracy can be
captured by a strictly pairwise potential:

    E_total = sum over pairs (i, j) within r_cutoff of  f_pair(r_ij, Z_i, Z_j)

where `f_pair` is a tiny MLP on an RBF-expanded distance and element
embeddings. No message passing.

The kernel is:
  - **Architecturally invariant**: only pairwise distances enter, so it
    is translation/rotation/permutation invariant by construction.
  - **Conservative**: forces come from autograd of a scalar energy.
  - **Symmetric**: f_pair(r, Z_i, Z_j) == f_pair(r, Z_j, Z_i) by
    construction (uses the symmetric features h_i + h_j and h_i * h_j).
  - **Tiny**: a few thousand parameters, ~10-100x faster than v3.
  - **PBC-aware**: optional `box_size` for minimum-image distances.

Use case: distill v3 into FastPairKernel once (offline), then run the
kernel at cell scale. Since v3 itself was trained on a pairwise LJ
reference, the kernel should match v3 to high accuracy with much less
compute.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from artificial_atom.network import (
    ArtificialAtomPotential,
    GaussianRBF,
    cosine_cutoff,
    radius_edges,
)
from artificial_atom.dataset import generate_dataset


# ----------------------- the kernel -----------------------

@dataclass
class FastPairConfig:
    d_embed: int = 16          # element embedding dim
    d_rbf: int = 12            # RBF basis size
    d_hidden: int = 32         # pair-MLP hidden width
    r_cutoff: float = 6.0
    n_elements: int = 119


class FastPairKernel(nn.Module):
    """Pair-only neural potential. Total energy = sum over pairs (i<j) of
    f_pair(r_ij, Z_i, Z_j). Forces via autograd.
    """

    def __init__(self, cfg: FastPairConfig | None = None):
        super().__init__()
        self.cfg = cfg or FastPairConfig()
        self.embed = nn.Embedding(self.cfg.n_elements, self.cfg.d_embed)
        self.rbf = GaussianRBF(n_rbf=self.cfg.d_rbf, r_min=0.0, r_max=self.cfg.r_cutoff)
        # Pair MLP input: rbf + symmetric element features (sum + product)
        in_dim = self.cfg.d_rbf + 2 * self.cfg.d_embed
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_dim, self.cfg.d_hidden), nn.SiLU(),
            nn.Linear(self.cfg.d_hidden, self.cfg.d_hidden), nn.SiLU(),
            nn.Linear(self.cfg.d_hidden, 1),
        )
        self.register_buffer("energy_scale", torch.tensor(1.0))
        self.register_buffer("energy_shift", torch.tensor(0.0))

    def fit_normalization(self, mean: float, std: float) -> None:
        self.energy_shift = torch.tensor(float(mean))
        self.energy_scale = torch.tensor(float(std) if std > 0 else 1.0)

    def total_energy(
        self, positions: torch.Tensor, Z: torch.Tensor,
        box_size: float | None = None,
    ) -> torch.Tensor:
        edge_dst, edge_src, edge_dist = radius_edges(
            positions, self.cfg.r_cutoff, box_size=box_size
        )
        if edge_dst.numel() == 0:
            return positions.sum() * 0.0
        # Build symmetric per-edge features
        h = self.embed(Z)                                    # (N, d_embed)
        h_i, h_j = h[edge_src], h[edge_dst]                   # (E, d_embed) each
        h_sum = h_i + h_j                                     # symmetric
        h_prod = h_i * h_j                                    # symmetric
        rbf = self.rbf(edge_dist)
        cut = cosine_cutoff(edge_dist, self.cfg.r_cutoff)
        x = torch.cat([rbf, h_sum, h_prod], dim=-1)
        e_per_edge = self.pair_mlp(x).squeeze(-1) * cut       # (E,)
        # Edges are directional (i, j) and (j, i) both appear; divide by 2.
        e_per_atom = torch.zeros(positions.shape[0], device=positions.device,
                                 dtype=positions.dtype)
        e_per_atom.index_add_(0, edge_dst, e_per_edge * 0.5)
        raw = e_per_atom.sum()
        return raw * self.energy_scale + self.energy_shift * positions.shape[0]

    def forward(self, positions: torch.Tensor, Z: torch.Tensor,
                  box_size: float | None = None) -> torch.Tensor:
        return self.total_energy(positions, Z, box_size=box_size)

    def energy_and_forces(
        self, positions: torch.Tensor, Z: torch.Tensor,
        box_size: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.detach().clone().requires_grad_(True)
        E = self.total_energy(positions, Z, box_size=box_size)
        (grad,) = torch.autograd.grad(E, positions, create_graph=False,
                                        allow_unused=True)
        if grad is None:
            grad = torch.zeros_like(positions)
        return E.detach(), -grad.detach()


# ----------------------- distillation -----------------------

@dataclass
class DistillConfig:
    epochs: int = 40
    lr: float = 3e-3
    batch_size: int = 16
    energy_weight: float = 1.0
    force_weight: float = 1.0
    n_train: int = 600
    n_val: int = 120
    seed: int = 0


def distill_from_v3(
    teacher: ArtificialAtomPotential,
    cfg: DistillConfig | None = None,
    student_cfg: FastPairConfig | None = None,
    verbose: bool = False,
) -> tuple[FastPairKernel, dict]:
    """Train a FastPairKernel to mimic v3 on the artificial_atom dataset.

    Strategy: generate the same kind of configurations v3 was trained on,
    have v3 produce per-configuration (energy, force) targets, then fit
    the student to those targets. The student's smaller architecture
    means it cannot capture true many-body terms, but since v3 was itself
    trained on a pair potential (LJ), the student should match v3 well.

    Returns (trained_student, history).
    """
    cfg = cfg or DistillConfig()
    student = FastPairKernel(student_cfg)
    if verbose:
        n_params = sum(p.numel() for p in student.parameters())
        n_v3 = sum(p.numel() for p in teacher.parameters())
        print(f"  teacher v3: {n_v3:,} params,  student: {n_params:,} params")

    # Generate config sets and run teacher to get (E, F) targets.
    train_configs = generate_dataset(n_configs=cfg.n_train, seed=cfg.seed)
    val_configs = generate_dataset(n_configs=cfg.n_val, seed=cfg.seed + 1)

    def teacher_targets(configs):
        Es, Fs = [], []
        for c in configs:
            pos = c.positions.clone().requires_grad_(True)
            E = teacher(pos, c.Z)
            (g,) = torch.autograd.grad(E, pos, allow_unused=True)
            if g is None:
                g = torch.zeros_like(pos)
            Es.append(E.detach())
            Fs.append(-g.detach())
        return Es, Fs

    if verbose:
        print("  evaluating teacher on training configurations ...")
    train_E, train_F = teacher_targets(train_configs)
    val_E, val_F = teacher_targets(val_configs)

    # Normalize student energies based on teacher targets
    e_mean = torch.stack(train_E).mean().item()
    e_per_atom_std = torch.tensor(
        [E.item() / c.n_atoms for E, c in zip(train_E, train_configs)]
    ).std().clamp_min(0.1).item()
    student.fit_normalization(mean=e_mean / max(1, sum(c.n_atoms for c in train_configs) / len(train_configs)),
                                std=e_per_atom_std)

    opt = torch.optim.Adam(student.parameters(), lr=cfg.lr)
    history = {"train_loss": [], "val_e_mae": [], "val_f_mae": []}
    rng = torch.Generator().manual_seed(cfg.seed)
    n_train = len(train_configs)

    for epoch in range(cfg.epochs):
        t0 = time.time()
        perm = torch.randperm(n_train, generator=rng).tolist()
        student.train()
        running = 0.0
        n_batches = 0
        for b in range(0, n_train, cfg.batch_size):
            opt.zero_grad()
            losses = []
            for k in perm[b:b + cfg.batch_size]:
                c = train_configs[k]
                E_target = train_E[k]
                F_target = train_F[k]
                pos = c.positions.clone().requires_grad_(True)
                E_pred = student(pos, c.Z)
                (gp,) = torch.autograd.grad(E_pred, pos, create_graph=True,
                                              allow_unused=True)
                if gp is None:
                    gp = torch.zeros_like(pos)
                F_pred = -gp
                n = float(c.n_atoms)
                e_err = (E_pred - E_target) / n
                f_err = F_pred - F_target
                losses.append(cfg.energy_weight * e_err.pow(2)
                                + cfg.force_weight * f_err.pow(2).mean())
            total = torch.stack(losses).mean()
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()
            running += total.item()
            n_batches += 1
        history["train_loss"].append(running / max(n_batches, 1))

        # validation
        student.eval()
        e_abs = 0.0
        f_abs = 0.0
        f_count = 0
        for c, E_t, F_t in zip(val_configs, val_E, val_F):
            pos = c.positions.clone().requires_grad_(True)
            E_p = student(pos, c.Z)
            (gp,) = torch.autograd.grad(E_p, pos, allow_unused=True)
            if gp is None:
                gp = torch.zeros_like(pos)
            F_p = -gp
            e_abs += abs(E_p.item() - E_t.item())
            f_abs += (F_p - F_t).abs().sum().item()
            f_count += F_t.numel()
        e_mae = e_abs / len(val_configs)
        f_mae = f_abs / max(1, f_count)
        history["val_e_mae"].append(e_mae)
        history["val_f_mae"].append(f_mae)
        if verbose and (epoch % max(1, cfg.epochs // 8) == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:>3d}  loss={history['train_loss'][-1]:.4f}  "
                  f"val_E_MAE={e_mae:.4f}  val_F_MAE={f_mae:.4f}  "
                  f"({time.time()-t0:.1f}s)")
    return student, history


def kernel_force_and_energy(
    state, kernel: FastPairKernel, default_Z: int = 6,
    use_pbc: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (force, energy) from the fast kernel on a ParticleState."""
    positions = state.positions.clone()
    Z = torch.full((state.n,), int(default_Z), dtype=torch.long,
                    device=positions.device)
    box = state.box_size if use_pbc else None
    E, F = kernel.energy_and_forces(positions, Z=Z, box_size=box)
    return F, E
