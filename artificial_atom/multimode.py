"""Multi-mode, stateful artificial-atom potential.

Differences from `network.py`:

  - Initial per-atom features mix element embedding with chemistry context
    embeddings (hybridisation, formal charge, aromatic, ring, donor/acceptor).
  - The radius graph is split into two edge sets - covalent bonds and the
    rest - that flow through *separate* filter MLPs at each layer.
  - A partial-charge-weighted scalar channel approximates the
    electrostatic mode (no equivariant vectors, so this is an
    invariant scalar field, not the full Coulomb vector field).
  - Three per-mode energy heads are summed for the total. The split is
    available to callers as a diagnostic.
  - A GRU updates a *persistent* per-atom hidden state across calls.
    Pass `initial_state` to read prior state in; capture `final_state`
    from the output to pass it back next call.

Symmetries (translation, rotation, permutation invariance + Newton's 3rd
law on forces) are preserved by construction: every input to a learnable
function is either a pairwise distance, a per-atom scalar, or a per-bond
type label - never a raw position vector or atom ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .network import GaussianRBF, cosine_cutoff
from .unit import HYB_AROMATIC, N_BOND, N_HYB, N_MODES


# ---------------- graph helpers ----------------

def radius_edges_excluding(
    positions: torch.Tensor, r_cut: float, bonds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Radius graph minus the bonds (which go on the bonded channel)."""
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


def bond_edges(
    positions: torch.Tensor, bonds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build directional edges for both directions of each undirected bond."""
    if bonds.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return empty, empty, torch.empty(0, device=positions.device)
    diff = positions[bonds[:, 0]] - positions[bonds[:, 1]]
    d = torch.linalg.norm(diff, dim=-1)
    # Build (i->j) and (j->i) so messages flow both ways.
    dst = torch.cat([bonds[:, 0], bonds[:, 1]], dim=0)
    src = torch.cat([bonds[:, 1], bonds[:, 0]], dim=0)
    dist = torch.cat([d, d], dim=0)
    return dst, src, dist


# ---------------- a CF-conv layer that also reads a scalar source feature ----

class GatedCFConv(nn.Module):
    """SchNet continuous-filter conv, plus an optional scalar per-edge gate.

    The gate is used by the electrostatic channel (q_source * q_dest as a
    multiplier on the distance filter). Setting `use_gate=False` recovers
    the plain SchNet block.
    """

    def __init__(self, d_h: int, d_rbf: int, use_gate: bool = False):
        super().__init__()
        self.use_gate = use_gate
        self.in_proj = nn.Linear(d_h, d_h, bias=False)
        self.filter_net = nn.Sequential(
            nn.Linear(d_rbf, d_h), nn.SiLU(), nn.Linear(d_h, d_h),
        )
        self.update_net = nn.Sequential(
            nn.Linear(d_h, d_h), nn.SiLU(), nn.Linear(d_h, d_h),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_src: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_cutoff: torch.Tensor,
        edge_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_dst.numel() == 0:
            return h
        h_src = self.in_proj(h[edge_src])
        W = self.filter_net(edge_rbf) * edge_cutoff.unsqueeze(-1)
        if self.use_gate and edge_gate is not None:
            W = W * edge_gate.unsqueeze(-1)
        msg = h_src * W
        aggregated = torch.zeros_like(h).index_add_(0, edge_dst, msg)
        return h + self.update_net(aggregated)


# ---------------- the main module ----------------

@dataclass
class MultiModeConfig:
    d_h: int = 64
    d_rbf: int = 24
    n_layers: int = 3
    r_cutoff: float = 6.0
    r_cutoff_electro: float = 6.0
    n_elements: int = 119
    energy_shift: float = 0.0
    energy_scale: float = 1.0


class MultiModePotential(nn.Module):
    """Stateful potential with bonded / non-bond / electrostatic channels."""

    def __init__(self, cfg: MultiModeConfig | None = None):
        super().__init__()
        self.cfg = cfg or MultiModeConfig()
        d_h = self.cfg.d_h
        d_rbf = self.cfg.d_rbf

        self.elem_embed = nn.Embedding(self.cfg.n_elements, d_h)
        self.hyb_embed = nn.Embedding(N_HYB, d_h)
        self.bond_type_embed = nn.Embedding(N_BOND, d_rbf)
        # 4 boolean chemistry flags + 1 formal charge + 1 partial charge -> projection.
        self.chem_proj = nn.Sequential(
            nn.Linear(6, d_h), nn.SiLU(), nn.Linear(d_h, d_h),
        )

        self.rbf = GaussianRBF(n_rbf=d_rbf, r_min=0.0, r_max=self.cfg.r_cutoff)

        # 3 channels of n_layers each
        self.layers_bonded = nn.ModuleList(
            [GatedCFConv(d_h, d_rbf) for _ in range(self.cfg.n_layers)]
        )
        self.layers_nonbond = nn.ModuleList(
            [GatedCFConv(d_h, d_rbf) for _ in range(self.cfg.n_layers)]
        )
        self.layers_electro = nn.ModuleList(
            [GatedCFConv(d_h, d_rbf, use_gate=True) for _ in range(self.cfg.n_layers)]
        )

        # GRU updates the persistent state from the fused channel features.
        self.fuse = nn.Linear(3 * d_h, d_h)
        self.gru = nn.GRUCell(d_h, d_h)

        # Per-mode energy heads.
        def head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_h, d_h), nn.SiLU(), nn.Linear(d_h, 1),
            )

        self.head_bonded = head()
        self.head_nonbond = head()
        self.head_electro = head()

        self.register_buffer("energy_shift", torch.tensor(self.cfg.energy_shift))
        self.register_buffer("energy_scale", torch.tensor(self.cfg.energy_scale))

    def fit_normalization(self, atomwise_mean: float, atomwise_std: float) -> None:
        self.energy_shift = torch.tensor(float(atomwise_mean))
        self.energy_scale = torch.tensor(float(atomwise_std) if atomwise_std > 0 else 1.0)

    def _initial_features(
        self,
        Z: torch.Tensor,
        hybridisation: torch.Tensor,
        formal_charge: torch.Tensor,
        aromatic: torch.Tensor,
        in_ring: torch.Tensor,
        h_donor: torch.Tensor,
        h_acceptor: torch.Tensor,
        partial_charge: torch.Tensor,
    ) -> torch.Tensor:
        N = Z.shape[0]
        # Bool flags + scalars stacked
        flags = torch.stack([
            aromatic.float(),
            in_ring.float(),
            h_donor.float(),
            h_acceptor.float(),
            formal_charge.float(),
            partial_charge,
        ], dim=-1)
        h = (
            self.elem_embed(Z)
            + self.hyb_embed(hybridisation)
            + self.chem_proj(flags)
        )
        return h

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
        bonds: torch.Tensor,
        bond_type: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        return_breakdown: bool = False,
    ):
        N = positions.shape[0]
        h0 = self._initial_features(
            Z, hybridisation, formal_charge, aromatic, in_ring,
            h_donor, h_acceptor, partial_charge,
        )
        h = h0 if initial_state is None else h0 + initial_state

        # ---- edge graphs ----
        e_nb_dst, e_nb_src, e_nb_dist = radius_edges_excluding(
            positions, self.cfg.r_cutoff, bonds
        )
        e_b_dst, e_b_src, e_b_dist = bond_edges(positions, bonds)

        # Bonded edges carry the bond-type embedding as a per-edge RBF offset.
        rbf_nb = self.rbf(e_nb_dist)
        cut_nb = cosine_cutoff(e_nb_dist, self.cfg.r_cutoff)
        rbf_b = (
            self.rbf(e_b_dist)
            + self.bond_type_embed(torch.cat([bond_type, bond_type], dim=0))
            if e_b_dist.numel() > 0
            else self.rbf(e_b_dist)
        )
        cut_b = cosine_cutoff(e_b_dist, self.cfg.r_cutoff)

        # Electrostatic channel uses the SAME radius graph as non-bonded
        # but adds a partial-charge gate q_src * q_dst.
        e_el_dst, e_el_src, e_el_dist = radius_edges_excluding(
            positions, self.cfg.r_cutoff_electro, torch.empty(0, 2, dtype=torch.long, device=positions.device)
        )
        rbf_el = self.rbf(e_el_dist)
        cut_el = cosine_cutoff(e_el_dist, self.cfg.r_cutoff_electro)
        if e_el_dst.numel() > 0:
            gate_el = partial_charge[e_el_src] * partial_charge[e_el_dst]
        else:
            gate_el = None

        # ---- run channels in parallel ----
        h_b, h_nb, h_el = h, h, h
        for L_b, L_nb, L_el in zip(self.layers_bonded, self.layers_nonbond, self.layers_electro):
            h_b = L_b(h_b, e_b_dst, e_b_src, rbf_b, cut_b)
            h_nb = L_nb(h_nb, e_nb_dst, e_nb_src, rbf_nb, cut_nb)
            h_el = L_el(h_el, e_el_dst, e_el_src, rbf_el, cut_el, edge_gate=gate_el)

        # ---- per-mode energies (per-atom contributions, summed) ----
        e_per_atom_bonded = self.head_bonded(h_b).squeeze(-1)
        e_per_atom_nonbond = self.head_nonbond(h_nb).squeeze(-1)
        e_per_atom_electro = self.head_electro(h_el).squeeze(-1)

        # Apply the same per-atom normalisation to the SUM (shifts cancel
        # under translation/permutation - they go on the totals).
        e_atom = e_per_atom_bonded + e_per_atom_nonbond + e_per_atom_electro
        e_atom = e_atom * self.energy_scale + self.energy_shift
        total_E = e_atom.sum()

        # ---- updated persistent state via GRU on the fused channel output ----
        h_fused = self.fuse(torch.cat([h_b, h_nb, h_el], dim=-1))
        new_state = self.gru(h_fused, h)

        if return_breakdown:
            return total_E, new_state, {
                "bonded": e_per_atom_bonded.sum(),
                "nonbond": e_per_atom_nonbond.sum(),
                "electro": e_per_atom_electro.sum(),
                "scale": self.energy_scale.detach().clone(),
                "shift": self.energy_shift.detach().clone() * N,
            }
        return total_E, new_state

    def energy_and_forces(self, positions: torch.Tensor, **kwargs):
        positions = positions.detach().clone().requires_grad_(True)
        E, new_state = self.forward(positions, **kwargs)
        (grad,) = torch.autograd.grad(E, positions, create_graph=False, allow_unused=True)
        if grad is None:
            grad = torch.zeros_like(positions)
        return E.detach(), (-grad).detach(), new_state.detach()
