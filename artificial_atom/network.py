"""SchNet-style invariant atomic potential.

The model takes a configuration (positions, atomic numbers) and returns
a total potential energy as a scalar. Forces are obtained by
autograd-differentiating that energy w.r.t. the positions - see
`ArtificialAtomPotential.energy_and_forces`.

Architecture (SchNet, Schuett et al. 2017, simplified)::

    Z -> embedding -> h (N, d)
    for each layer:
        # for every (i, j) edge within cutoff:
        #     filter_ij  = MLP_filter( RBF(r_ij) )  *  cutoff(r_ij)
        #     msg_ij     = filter_ij  *  Linear(h_j)
        # aggregate by sum to atom i, then a residual update MLP
    energy_i = MLP_out(h_i)
    total_E  = sum_i energy_i

The network is invariant to rigid motion (only distances enter), exactly
permutation-equivariant (the only aggregation is unordered sum), and the
total energy is extensive in the number of atoms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------- pure tensor building blocks ----------

def cosine_cutoff(r: torch.Tensor, r_cut: float) -> torch.Tensor:
    """Smooth cutoff that goes to zero (and has zero derivative) at r=r_cut."""
    x = torch.clamp(r / r_cut, max=1.0)
    return 0.5 * (torch.cos(math.pi * x) + 1.0) * (r < r_cut).float()


class GaussianRBF(nn.Module):
    """Expand a distance scalar into a vector of Gaussian basis features.

    Centres are fixed and uniformly spaced; width is a learnable parameter
    so the model can sharpen / broaden its distance resolution.
    """

    def __init__(self, n_rbf: int = 24, r_min: float = 0.0, r_max: float = 6.0):
        super().__init__()
        centres = torch.linspace(r_min, r_max, n_rbf)
        self.register_buffer("centres", centres)
        # Width chosen so adjacent Gaussians overlap a bit.
        spacing = (r_max - r_min) / max(1, n_rbf - 1)
        self.log_width = nn.Parameter(torch.full((1,), math.log(spacing)))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # r: (..., 1) or (...). Returns (..., n_rbf).
        width = torch.exp(self.log_width)
        x = r.unsqueeze(-1) - self.centres
        return torch.exp(-(x ** 2) / (2.0 * width * width))


def radius_edges(
    positions: torch.Tensor, r_cut: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an edge list over atom pairs within r_cut, excluding self-pairs.

    Returns (edge_dst, edge_src, edge_dist) so that for each edge e:
        message flowing into edge_dst[e] comes from edge_src[e],
        with distance edge_dist[e].
    """
    N = positions.shape[0]
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)   # (N, N, 3)
    r = torch.linalg.norm(diff, dim=-1)                       # (N, N)
    diag = torch.eye(N, device=positions.device, dtype=torch.bool)
    mask = (r < r_cut) & (~diag)
    if not mask.any():
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return empty, empty, torch.empty(0, device=positions.device)
    pair_idx = mask.nonzero(as_tuple=False)                   # (E, 2)
    edge_dst = pair_idx[:, 0]
    edge_src = pair_idx[:, 1]
    edge_dist = r[edge_dst, edge_src]
    return edge_dst, edge_src, edge_dist


# ---------- the core continuous-filter convolution layer ----------

class CFConvLayer(nn.Module):
    """One SchNet continuous-filter convolution + residual update."""

    def __init__(self, d_h: int, d_rbf: int):
        super().__init__()
        self.in_proj = nn.Linear(d_h, d_h, bias=False)
        self.filter_net = nn.Sequential(
            nn.Linear(d_rbf, d_h),
            nn.SiLU(),
            nn.Linear(d_h, d_h),
        )
        self.update_net = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.SiLU(),
            nn.Linear(d_h, d_h),
        )

    def forward(
        self,
        h: torch.Tensor,           # (N, d_h)
        edge_dst: torch.Tensor,    # (E,)
        edge_src: torch.Tensor,    # (E,)
        edge_rbf: torch.Tensor,    # (E, d_rbf)
        edge_cutoff: torch.Tensor, # (E,)
    ) -> torch.Tensor:
        N = h.shape[0]
        if edge_dst.numel() == 0:
            return h
        h_src = self.in_proj(h[edge_src])               # (E, d_h)
        W = self.filter_net(edge_rbf) * edge_cutoff.unsqueeze(-1)
        msg = h_src * W                                  # (E, d_h)
        aggregated = torch.zeros_like(h).index_add_(0, edge_dst, msg)
        return h + self.update_net(aggregated)


# ---------- the full potential ----------

@dataclass
class PotentialConfig:
    d_h: int = 64
    d_rbf: int = 24
    n_layers: int = 3
    r_cutoff: float = 6.0
    n_elements: int = 119          # covers Z=0..118; row 0 unused
    energy_shift: float = 0.0      # subtracted from raw network output (set after training data is known)
    energy_scale: float = 1.0      # multiplies raw network output


class ArtificialAtomPotential(nn.Module):
    """Total potential energy as a sum over per-atom contributions."""

    def __init__(self, cfg: PotentialConfig | None = None):
        super().__init__()
        self.cfg = cfg or PotentialConfig()
        self.embed = nn.Embedding(self.cfg.n_elements, self.cfg.d_h)
        self.rbf = GaussianRBF(n_rbf=self.cfg.d_rbf, r_min=0.0, r_max=self.cfg.r_cutoff)
        self.layers = nn.ModuleList([
            CFConvLayer(self.cfg.d_h, self.cfg.d_rbf) for _ in range(self.cfg.n_layers)
        ])
        self.out_head = nn.Sequential(
            nn.Linear(self.cfg.d_h, self.cfg.d_h),
            nn.SiLU(),
            nn.Linear(self.cfg.d_h, 1),
        )
        # Normalisation constants - set by fit_normalization() from data.
        self.register_buffer("energy_shift", torch.tensor(self.cfg.energy_shift))
        self.register_buffer("energy_scale", torch.tensor(self.cfg.energy_scale))

    def fit_normalization(self, atomwise_mean: float, atomwise_std: float) -> None:
        """Set the output denormalisation so that, *initially*, the network's
        per-atom predictions roughly match the data's mean and std."""
        self.energy_shift = torch.tensor(float(atomwise_mean))
        self.energy_scale = torch.tensor(float(atomwise_std) if atomwise_std > 0 else 1.0)

    def per_atom_energy(self, positions: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        edge_dst, edge_src, edge_dist = radius_edges(positions, self.cfg.r_cutoff)
        h = self.embed(Z)
        rbf = self.rbf(edge_dist)
        cut = cosine_cutoff(edge_dist, self.cfg.r_cutoff)
        for layer in self.layers:
            h = layer(h, edge_dst, edge_src, rbf, cut)
        raw = self.out_head(h).squeeze(-1)   # (N,)
        return raw * self.energy_scale + self.energy_shift

    def forward(self, positions: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        return self.per_atom_energy(positions, Z).sum()

    def energy_and_forces(
        self, positions: torch.Tensor, Z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: returns (E, F) where F = -dE/dpositions."""
        positions = positions.detach().clone().requires_grad_(True)
        E = self.forward(positions, Z)
        (grad,) = torch.autograd.grad(
            E, positions, create_graph=False, allow_unused=True
        )
        if grad is None:
            grad = torch.zeros_like(positions)
        return E.detach(), (-grad).detach()
