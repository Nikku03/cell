"""v3 of the multi-mode artificial-atom potential.

Adds, relative to v2 (`multimode.py`):

  - Bond-feature embedding: bond_order, bond_in_ring, bond_conjugated,
    bond_rotatable, all consumed by the bonded message channel.
  - Per-atom vdW radius added to the chemistry-context projection.
  - Distance-shaped electrostatic gate:  q_i * q_j * smooth(1/r),
    instead of v2's plain q_i * q_j. Closer to the analytic Coulomb
    shape while staying invariant under rotations.
  - Pose-quality head: aggregates final per-atom features and outputs
    a logit for "near-native vs bad pose".

Symmetry guarantees from v2 are preserved by construction (everything
that enters a learnable function is still a pairwise scalar, a per-atom
scalar, or an unordered bond label).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .network import GaussianRBF, cosine_cutoff
from .unit import HYB_AROMATIC, N_BOND, N_HYB


# Smooth 1/r that goes to a finite value at r=0 (won't NaN) and to zero at r=r_cut.
def smooth_inv_r(r: torch.Tensor, r_cut: float, eps: float = 0.5) -> torch.Tensor:
    """`1 / (r + eps)` times the cosine cutoff, so the gate is bounded."""
    inv = 1.0 / (r + eps)
    return inv * cosine_cutoff(r, r_cut)


# ---------------- graph helpers (same as v2, kept local) ----------------

def radius_edges_excluding(
    positions: torch.Tensor, r_cut: float, bonds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = positions.shape[0]
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    r = torch.linalg.norm(diff, dim=-1)
    diag = torch.eye(N, dtype=torch.bool, device=positions.device)
    bond_mask = torch.zeros((N, N), dtype=torch.bool, device=positions.device)
    if bonds.numel() > 0:
        bond_mask[bonds[:, 0], bonds[:, 1]] = True
        bond_mask[bonds[:, 1], bonds[:, 0]] = True
    mask = (r < r_cut) & (~diag) & (~bond_mask)
    if not mask.any():
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return empty, empty, torch.empty(0, device=positions.device)
    pair = mask.nonzero(as_tuple=False)
    return pair[:, 0], pair[:, 1], r[pair[:, 0], pair[:, 1]]


def bond_edges_v3(
    positions: torch.Tensor, bonds: torch.Tensor,
    bond_order: torch.Tensor, bond_in_ring: torch.Tensor,
    bond_conjugated: torch.Tensor, bond_rotatable: torch.Tensor,
) -> dict:
    """Build directional edges + matching per-edge bond features."""
    if bonds.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return {
            "dst": empty, "src": empty,
            "dist": torch.empty(0, device=positions.device),
            "order": torch.empty(0, device=positions.device),
            "in_ring": torch.empty(0, dtype=torch.bool, device=positions.device),
            "conjugated": torch.empty(0, dtype=torch.bool, device=positions.device),
            "rotatable": torch.empty(0, dtype=torch.bool, device=positions.device),
        }
    diff = positions[bonds[:, 0]] - positions[bonds[:, 1]]
    d = torch.linalg.norm(diff, dim=-1)
    dst = torch.cat([bonds[:, 0], bonds[:, 1]], dim=0)
    src = torch.cat([bonds[:, 1], bonds[:, 0]], dim=0)
    dist = torch.cat([d, d], dim=0)
    order = torch.cat([bond_order, bond_order], dim=0)
    ir = torch.cat([bond_in_ring, bond_in_ring], dim=0)
    cj = torch.cat([bond_conjugated, bond_conjugated], dim=0)
    rt = torch.cat([bond_rotatable, bond_rotatable], dim=0)
    return {"dst": dst, "src": src, "dist": dist,
            "order": order, "in_ring": ir, "conjugated": cj, "rotatable": rt}


# ---------------- CF-conv variant that accepts an extra per-edge feature vector ----

class CFConvWithEdgeFeat(nn.Module):
    """SchNet CF-conv where the filter MLP also sees per-edge feature vectors
    (used for the bonded channel: bond order + flags concatenated)."""

    def __init__(self, d_h: int, d_rbf: int, d_edge_extra: int = 0, use_gate: bool = False):
        super().__init__()
        self.use_gate = use_gate
        self.in_proj = nn.Linear(d_h, d_h, bias=False)
        self.filter_net = nn.Sequential(
            nn.Linear(d_rbf + d_edge_extra, d_h), nn.SiLU(),
            nn.Linear(d_h, d_h),
        )
        self.update_net = nn.Sequential(
            nn.Linear(d_h, d_h), nn.SiLU(), nn.Linear(d_h, d_h),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_src: torch.Tensor,
        edge_filter_input: torch.Tensor,   # (E, d_rbf + d_edge_extra)
        edge_cutoff: torch.Tensor,
        edge_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_dst.numel() == 0:
            return h
        h_src = self.in_proj(h[edge_src])
        W = self.filter_net(edge_filter_input) * edge_cutoff.unsqueeze(-1)
        if self.use_gate and edge_gate is not None:
            W = W * edge_gate.unsqueeze(-1)
        msg = h_src * W
        aggregated = torch.zeros_like(h).index_add_(0, edge_dst, msg)
        return h + self.update_net(aggregated)


# ---------------- the v3 model ----------------

@dataclass
class MultiModeV3Config:
    d_h: int = 64
    d_rbf: int = 24
    d_bond_extra: int = 8       # bond order + 4 flags + slack -> small extra vector
    n_layers: int = 2
    r_cutoff: float = 6.0
    r_cutoff_electro: float = 8.0
    n_elements: int = 119
    energy_shift: float = 0.0
    energy_scale: float = 1.0


class MultiModePotentialV3(nn.Module):

    def __init__(self, cfg: MultiModeV3Config | None = None):
        super().__init__()
        self.cfg = cfg or MultiModeV3Config()
        d_h, d_rbf = self.cfg.d_h, self.cfg.d_rbf

        self.elem_embed = nn.Embedding(self.cfg.n_elements, d_h)
        self.hyb_embed = nn.Embedding(N_HYB, d_h)
        # 7 scalar / boolean per-atom features: aromatic, in_ring, h_donor,
        # h_acceptor, formal_charge, partial_charge, vdw_radius
        self.atom_proj = nn.Sequential(
            nn.Linear(7, d_h), nn.SiLU(), nn.Linear(d_h, d_h),
        )

        self.rbf = GaussianRBF(n_rbf=d_rbf, r_min=0.0, r_max=self.cfg.r_cutoff)
        # Project bond features: bond_order (1.0/1.5/2.0/3.0), 3 bool flags -> small vector
        self.bond_feat_proj = nn.Sequential(
            nn.Linear(4, self.cfg.d_bond_extra), nn.SiLU(),
            nn.Linear(self.cfg.d_bond_extra, self.cfg.d_bond_extra),
        )

        # 3 channels, each is a stack of CFConv layers.
        self.layers_bonded = nn.ModuleList([
            CFConvWithEdgeFeat(d_h, d_rbf, d_edge_extra=self.cfg.d_bond_extra)
            for _ in range(self.cfg.n_layers)
        ])
        self.layers_nonbond = nn.ModuleList([
            CFConvWithEdgeFeat(d_h, d_rbf, d_edge_extra=0)
            for _ in range(self.cfg.n_layers)
        ])
        self.layers_electro = nn.ModuleList([
            CFConvWithEdgeFeat(d_h, d_rbf, d_edge_extra=0, use_gate=True)
            for _ in range(self.cfg.n_layers)
        ])

        self.fuse = nn.Linear(3 * d_h, d_h)
        self.gru = nn.GRUCell(d_h, d_h)

        def head():
            return nn.Sequential(
                nn.Linear(d_h, d_h), nn.SiLU(), nn.Linear(d_h, 1),
            )

        self.head_bonded = head()
        self.head_nonbond = head()
        self.head_electro = head()

        # Pose-quality classifier reads the fused per-atom features (sum-pooled).
        self.pose_head = nn.Sequential(
            nn.Linear(d_h, d_h), nn.SiLU(),
            nn.Linear(d_h, d_h), nn.SiLU(),
            nn.Linear(d_h, 1),
        )

        self.register_buffer("energy_shift", torch.tensor(self.cfg.energy_shift))
        self.register_buffer("energy_scale", torch.tensor(self.cfg.energy_scale))

    def fit_normalization(self, mean: float, std: float) -> None:
        self.energy_shift = torch.tensor(float(mean))
        self.energy_scale = torch.tensor(float(std) if std > 0 else 1.0)

    def _initial_features(
        self, Z, hyb, formal_charge, aromatic, in_ring,
        h_donor, h_acceptor, partial_charge, vdw_radius,
    ):
        feats = torch.stack([
            aromatic.float(), in_ring.float(),
            h_donor.float(), h_acceptor.float(),
            formal_charge.float(), partial_charge, vdw_radius,
        ], dim=-1)
        return self.elem_embed(Z) + self.hyb_embed(hyb) + self.atom_proj(feats)

    def forward(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        hybridisation: torch.Tensor,
        formal_charge: torch.Tensor,
        aromatic: torch.Tensor,
        in_ring: torch.Tensor,
        h_donor: torch.Tensor,
        h_acceptor: torch.Tensor,
        partial_charge: torch.Tensor,
        vdw_radius: torch.Tensor,
        bonds: torch.Tensor,
        bond_order: torch.Tensor,
        bond_in_ring: torch.Tensor,
        bond_conjugated: torch.Tensor,
        bond_rotatable: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        return_breakdown: bool = False,
    ):
        N = positions.shape[0]
        h0 = self._initial_features(
            Z, hybridisation, formal_charge, aromatic, in_ring,
            h_donor, h_acceptor, partial_charge, vdw_radius,
        )
        h = h0 if initial_state is None else h0 + initial_state

        # Build edge sets
        nb_dst, nb_src, nb_dist = radius_edges_excluding(positions, self.cfg.r_cutoff, bonds)
        b_edges = bond_edges_v3(
            positions, bonds, bond_order, bond_in_ring,
            bond_conjugated, bond_rotatable,
        )

        rbf_nb = self.rbf(nb_dist)
        cut_nb = cosine_cutoff(nb_dist, self.cfg.r_cutoff)

        if b_edges["dist"].numel() > 0:
            bfeat = torch.stack([
                b_edges["order"],
                b_edges["in_ring"].float(),
                b_edges["conjugated"].float(),
                b_edges["rotatable"].float(),
            ], dim=-1)
            bond_extra = self.bond_feat_proj(bfeat)
            rbf_b_full = torch.cat([self.rbf(b_edges["dist"]), bond_extra], dim=-1)
            cut_b = cosine_cutoff(b_edges["dist"], self.cfg.r_cutoff)
        else:
            rbf_b_full = torch.empty(0, self.cfg.d_rbf + self.cfg.d_bond_extra, device=positions.device)
            cut_b = torch.empty(0, device=positions.device)

        # Electrostatic edges: full radius graph (with longer cutoff),
        # gated by q_i*q_j*smooth(1/r).
        el_dst, el_src, el_dist = radius_edges_excluding(
            positions, self.cfg.r_cutoff_electro,
            torch.empty(0, 2, dtype=torch.long, device=positions.device),
        )
        rbf_el = self.rbf(el_dist)
        cut_el = cosine_cutoff(el_dist, self.cfg.r_cutoff_electro)
        if el_dst.numel() > 0:
            inv_r = smooth_inv_r(el_dist, self.cfg.r_cutoff_electro)
            gate_el = partial_charge[el_src] * partial_charge[el_dst] * inv_r
        else:
            gate_el = None

        h_b, h_nb, h_el = h, h, h
        for L_b, L_nb, L_el in zip(self.layers_bonded, self.layers_nonbond, self.layers_electro):
            h_b = L_b(h_b, b_edges["dst"], b_edges["src"], rbf_b_full, cut_b)
            h_nb = L_nb(h_nb, nb_dst, nb_src, rbf_nb, cut_nb)
            h_el = L_el(h_el, el_dst, el_src, rbf_el, cut_el, edge_gate=gate_el)

        # Per-mode energies
        e_atom_b = self.head_bonded(h_b).squeeze(-1)
        e_atom_nb = self.head_nonbond(h_nb).squeeze(-1)
        e_atom_el = self.head_electro(h_el).squeeze(-1)
        e_atom = (e_atom_b + e_atom_nb + e_atom_el) * self.energy_scale + self.energy_shift
        total_E = e_atom.sum()

        # Fused features for state update + pose head
        h_fused = self.fuse(torch.cat([h_b, h_nb, h_el], dim=-1))
        new_state = self.gru(h_fused, h)
        pose_logit = self.pose_head(h_fused.mean(dim=0, keepdim=True)).squeeze()

        if return_breakdown:
            return total_E, new_state, pose_logit, {
                "bonded": e_atom_b.sum(),
                "nonbond": e_atom_nb.sum(),
                "electro": e_atom_el.sum(),
                "scale": self.energy_scale.detach().clone(),
                "shift": self.energy_shift.detach().clone() * N,
            }
        return total_E, new_state, pose_logit
