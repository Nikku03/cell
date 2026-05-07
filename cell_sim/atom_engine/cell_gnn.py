"""CellGNN — heterogeneous-edge GNN for cell-scale particle dynamics.

Generalizes ``atom_engine.ml_model.AtomGNN`` from ~100-atom MD to cell-scale
(10⁴–10⁶ particles) by collapsing every kind of biological connection into
one mechanism: **typed edges with continuous thickness**.

Edge taxonomy
-------------
==========  ===================================================================
type        encodes
==========  ===================================================================
SPATIAL     close enough to interact; thickness from RBF(distance)
COVALENT    chain backbone (DNA, peptide); persistent, high thickness
COMPLEX     inside same macro-complex (ribosome subunits); very high thickness
REACTIVE    pair has a kinetic reaction rule; thickness = rate constant proxy
ENZYME      enzyme→substrate; directional, medium thickness
MEMBRANE    tethered to membrane patch
REGULATORY  TF→operator, sigma→RNAP; learned thickness
==========  ===================================================================

All edge information flows through a single ``EdgeFeaturizer`` that produces
one combined feature vector per edge:

    [ rbf(r) ; type_emb(t) ; thickness ; flags ]

The downstream message function consumes that vector identically for every
edge type — no architectural branching. This is the "edges are a universal
solvent" insight: hierarchical scales, polymer chains, membrane geometry,
reaction graphs all become *edge-type configurations*.

Heads
-----
* ``predict_forces_equivariant``  — per-particle 3D vector (E(3)-equivariant)
* ``predict_reaction``            — per-particle (n_reaction_classes,) logits
* ``predict_spawn_destroy``       — per-particle 2-class logits
                                    (positive=birth, negative=destroy)

Dynamic connections
-------------------
``build_dynamic_edges`` reconstructs the edge list fresh from current particle
state every step (the MeshGraphNets approach). Spatial edges come from a
radius cutoff; covalent/complex/reactive edges come from caller-supplied
membership tables. This means edges are *not* a fixed graph — they update
as particles move, react, or change complex membership.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
class EdgeType(IntEnum):
    SPATIAL = 0
    COVALENT = 1
    COMPLEX = 2
    REACTIVE = 3
    ENZYME = 4
    MEMBRANE = 5
    REGULATORY = 6


N_EDGE_TYPES = len(EdgeType)
N_RBF_BASIS = 16
EDGE_TYPE_DIM = 16
N_FLAG_DIM = 3


# ---------------------------------------------------------------------------
def rbf_expand(r: torch.Tensor, n_basis: int = N_RBF_BASIS,
                r_cut: float = 50.0) -> torch.Tensor:
    """Gaussian-RBF distance expansion. ``r`` shape (E,); returns (E, n_basis)."""
    centers = torch.linspace(0.0, r_cut, n_basis,
                              device=r.device, dtype=r.dtype)
    width = r_cut / max(n_basis - 1, 1)
    return torch.exp(-((r.unsqueeze(-1) - centers) / width) ** 2)


# ---------------------------------------------------------------------------
class EdgeFeaturizer(nn.Module):
    """Concatenates RBF(distance) + type embedding + thickness + flags.

    The output dimension is fixed once and the message MLPs consume it
    identically regardless of edge type — that's the whole point.
    """

    def __init__(self, n_basis: int = N_RBF_BASIS,
                 type_dim: int = EDGE_TYPE_DIM,
                 r_cut: float = 50.0):
        super().__init__()
        self.r_cut = r_cut
        self.n_basis = n_basis
        self.type_emb = nn.Embedding(N_EDGE_TYPES, type_dim)
        self.feat_dim = n_basis + type_dim + 1 + N_FLAG_DIM

    def forward(self, r: torch.Tensor, edge_type: torch.Tensor,
                thickness: Optional[torch.Tensor] = None,
                flags: Optional[torch.Tensor] = None) -> torch.Tensor:
        rbf = rbf_expand(r, self.n_basis, self.r_cut)
        te = self.type_emb(edge_type)
        if thickness is None:
            thickness = torch.ones_like(r)
        if flags is None:
            flags = torch.zeros((r.shape[0], N_FLAG_DIM),
                                 device=r.device, dtype=r.dtype)
        return torch.cat([rbf, te, thickness.unsqueeze(-1), flags], dim=-1)


# ---------------------------------------------------------------------------
class CellGNN(nn.Module):
    """E(3)-equivariant message-passing GNN for heterogeneous cell graphs.

    Parameters
    ----------
    n_species : int
        Vocabulary size for the species embedding table. Cells with ~hundreds
        of species are typical (Thornburg's syn3A 4DWCM has ~1700, our v15
        has 21).
    hidden : int
        Latent dimension. 64 is a reasonable default; scale up with data.
    n_rounds : int
        Number of message-passing rounds. 3 captures most local effects.
    n_reaction_classes : int
        Output dimension of the reaction-classifier head. For syn3A this
        could be ~7000; for our v15 model it's ~250.
    r_cut : float
        Spatial cutoff for the RBF expansion (nm).
    n_extra_node_features : int
        Per-particle scalar features beyond species id (e.g. mass, charge,
        is_alive, time_since_spawn).
    """

    def __init__(self, n_species: int, hidden: int = 64, n_rounds: int = 3,
                 n_reaction_classes: int = 64, r_cut: float = 50.0,
                 n_extra_node_features: int = 4):
        super().__init__()
        self.hidden = hidden
        self.n_rounds = n_rounds
        self.species_emb = nn.Embedding(n_species, hidden)
        self.featurizer = EdgeFeaturizer(r_cut=r_cut)
        ef_dim = self.featurizer.feat_dim

        self.encoder = nn.Sequential(
            nn.Linear(hidden + n_extra_node_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2 + ef_dim, hidden),
                          nn.SiLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_rounds)
        ])
        self.update = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_rounds)
        ])
        # Heads
        self.reaction_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_reaction_classes),
        )
        self.spawn_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),  # [logit_spawn, logit_destroy]
        )
        # Equivariant force scalar (per-edge) — same recipe as AtomGNN
        self.edge_scalar = nn.Sequential(
            nn.Linear(hidden * 2 + ef_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    # ---------- node embedding -----------------------------------------
    def _embed_nodes(self, species_id: torch.Tensor,
                     extra_feats: torch.Tensor) -> torch.Tensor:
        """species_id: (N,) long; extra_feats: (N, n_extra). returns (N, H)."""
        x = torch.cat([self.species_emb(species_id), extra_feats], dim=-1)
        return self.encoder(x)

    def _propagate(self, h: torch.Tensor, edges: torch.Tensor,
                   edge_features: torch.Tensor) -> torch.Tensor:
        for r in range(self.n_rounds):
            if edges.shape[0] == 0:
                continue
            src = edges[:, 0]
            dst = edges[:, 1]
            msg_in = torch.cat([h[src], h[dst], edge_features], dim=-1)
            m = self.msg[r](msg_in)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            h_upd = self.update[r](torch.cat([h, agg], dim=-1))
            h = h + h_upd
        return h

    # ---------- public API ---------------------------------------------
    def encode(self, species_id: torch.Tensor, extra_feats: torch.Tensor,
               edges: torch.Tensor, r: torch.Tensor,
               edge_type: torch.Tensor,
               thickness: Optional[torch.Tensor] = None,
               flags: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run encoder + n_rounds of message passing. Returns (N, H)."""
        h = self._embed_nodes(species_id, extra_feats)
        ef = self.featurizer(r, edge_type, thickness, flags)
        return self._propagate(h, edges, ef)

    def predict_reaction(self, *args, **kwargs) -> torch.Tensor:
        h = self.encode(*args, **kwargs)
        return self.reaction_head(h)

    def predict_spawn_destroy(self, *args, **kwargs) -> torch.Tensor:
        h = self.encode(*args, **kwargs)
        return self.spawn_head(h)

    def predict_forces_equivariant(self, species_id: torch.Tensor,
                                    extra_feats: torch.Tensor,
                                    edges: torch.Tensor,
                                    r: torch.Tensor,
                                    edge_type: torch.Tensor,
                                    pos: torch.Tensor,
                                    thickness: Optional[torch.Tensor] = None,
                                    flags: Optional[torch.Tensor] = None
                                    ) -> torch.Tensor:
        """Per-particle 3D force, E(3)-equivariant by construction.

        Force on i = Σ_j scalar_ij · unit_vector(r_j - r_i)
        scalar_ij is a function of node embeddings + edge features (all
        invariant); the unit vector rotates with the geometry.
        """
        n = species_id.shape[0]
        if edges.shape[0] == 0:
            return torch.zeros(n, 3, device=species_id.device, dtype=pos.dtype)
        h = self.encode(species_id, extra_feats, edges, r, edge_type,
                        thickness, flags)
        ef = self.featurizer(r, edge_type, thickness, flags)
        src = edges[:, 0]
        dst = edges[:, 1]
        msg_in = torch.cat([h[src], h[dst], ef], dim=-1)
        scalar = self.edge_scalar(msg_in).squeeze(-1)
        r_ij = pos[dst] - pos[src]
        r_mag = torch.clamp(r_ij.norm(dim=-1, keepdim=True), min=1e-6)
        u_ij = r_ij / r_mag
        edge_force = scalar.unsqueeze(-1) * u_ij
        forces = torch.zeros(n, 3, device=species_id.device, dtype=pos.dtype)
        forces.index_add_(0, src, edge_force)
        return forces

    def predict_all(self, species_id: torch.Tensor, extra_feats: torch.Tensor,
                    edges: torch.Tensor, r: torch.Tensor,
                    edge_type: torch.Tensor, pos: torch.Tensor,
                    thickness: Optional[torch.Tensor] = None,
                    flags: Optional[torch.Tensor] = None) -> dict:
        """Single forward pass returning all three head outputs.

        Convenient when training all heads jointly: avoids re-running message
        passing three times.
        """
        h = self.encode(species_id, extra_feats, edges, r, edge_type,
                        thickness, flags)
        ef = self.featurizer(r, edge_type, thickness, flags)
        # Forces (equivariant)
        n = species_id.shape[0]
        if edges.shape[0]:
            src = edges[:, 0]; dst = edges[:, 1]
            msg_in = torch.cat([h[src], h[dst], ef], dim=-1)
            scalar = self.edge_scalar(msg_in).squeeze(-1)
            r_ij = pos[dst] - pos[src]
            r_mag = torch.clamp(r_ij.norm(dim=-1, keepdim=True), min=1e-6)
            u_ij = r_ij / r_mag
            edge_force = scalar.unsqueeze(-1) * u_ij
            forces = torch.zeros(n, 3, device=species_id.device, dtype=pos.dtype)
            forces.index_add_(0, src, edge_force)
        else:
            forces = torch.zeros(n, 3, device=species_id.device, dtype=pos.dtype)
        return {
            'forces': forces,
            'reaction_logits': self.reaction_head(h),
            'spawn_destroy_logits': self.spawn_head(h),
            'embeddings': h,
        }


# ---------------------------------------------------------------------------
def _pairwise_distances(pos: torch.Tensor) -> torch.Tensor:
    """Full N×N distance matrix. Use only for small N (testing)."""
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    return diff.norm(dim=-1)


def build_dynamic_edges(
    pos: torch.Tensor,
    species_id: torch.Tensor,
    *,
    r_cut_spatial: float = 5.0,
    complex_membership: Optional[torch.Tensor] = None,
    backbone_chain: Optional[torch.Tensor] = None,
    reactive_pairs: Optional[torch.Tensor] = None,
    membrane_anchors: Optional[torch.Tensor] = None,
    regulatory_pairs: Optional[torch.Tensor] = None,
    enzyme_substrate: Optional[torch.Tensor] = None,
) -> dict:
    """Build a heterogeneous edge list from current particle state.

    Each call rebuilds edges fresh — that's "dynamic connections": as
    particles diffuse, react, or change complex membership, the graph
    rewires itself per step. This is the MeshGraphNets pattern.

    Inputs
    ------
    pos                  (N, 3) particle positions (nm)
    species_id           (N,)   long; species index per particle
    r_cut_spatial        spatial cutoff for SPATIAL edges (nm)
    complex_membership   (N,)   long or None; particles sharing the same
                                non-negative id are in the same complex
                                (e.g. ribosome subunits). -1 = no complex.
    backbone_chain       (M,)   long or None; ordered indices forming a
                                polymer backbone (DNA, peptide). Consecutive
                                pairs become COVALENT edges.
    reactive_pairs       (R, 3) long or None; columns [i, j, rate_idx] —
                                particles linked by a kinetic reaction.
                                Thickness = rate proxy in column 3.
    membrane_anchors     (K, 2) long or None; columns [particle_idx,
                                membrane_patch_idx]. Pairs become MEMBRANE
                                edges with high thickness.
    regulatory_pairs     (S, 2) long or None; columns [tf_idx, target_idx]
                                — TF binding operators / sigma factors etc.
    enzyme_substrate     (T, 2) long or None; columns [enzyme_idx, substrate_idx]

    Returns
    -------
    dict with keys:
      edges       (E, 2)   long; [src, dst]
      r           (E,)     float; pairwise distance
      edge_type   (E,)     long; EdgeType value
      thickness   (E,)     float; per-edge weight
    """
    device = pos.device
    dtype = pos.dtype
    n = pos.shape[0]

    edge_lists = []
    type_lists = []
    thick_lists = []

    # SPATIAL — pairs within r_cut. For small N use full pairwise; for big N
    # callers should batch / use a cell-list outside this function.
    if n > 0:
        d = _pairwise_distances(pos)
        mask = (d > 0) & (d <= r_cut_spatial)
        idx = mask.nonzero(as_tuple=False)        # (E, 2) — [src, dst]
        if idx.numel() > 0:
            r = d[idx[:, 0], idx[:, 1]]
            # Thickness for SPATIAL: 1/(1+r/r_cut) — thick when close, thin when far
            thick = 1.0 / (1.0 + r / r_cut_spatial)
            edge_lists.append(idx.long())
            type_lists.append(torch.full((idx.shape[0],), int(EdgeType.SPATIAL),
                                          dtype=torch.long, device=device))
            thick_lists.append(thick)

    # COVALENT — consecutive pairs in a polymer backbone
    if backbone_chain is not None and backbone_chain.numel() > 1:
        bc = backbone_chain.long()
        a, b = bc[:-1], bc[1:]
        e = torch.stack([torch.cat([a, b]), torch.cat([b, a])], dim=-1)
        edge_lists.append(e)
        type_lists.append(torch.full((e.shape[0],), int(EdgeType.COVALENT),
                                       dtype=torch.long, device=device))
        thick_lists.append(torch.ones(e.shape[0], dtype=dtype, device=device))

    # COMPLEX — full clique within each non-negative complex_membership group
    if complex_membership is not None:
        cm = complex_membership.long()
        for cid in cm.unique():
            if cid.item() < 0:
                continue
            members = (cm == cid).nonzero(as_tuple=False).squeeze(-1)
            if members.numel() < 2:
                continue
            # All pairs within complex (both directions)
            i, j = torch.meshgrid(members, members, indexing='ij')
            mask = i != j
            e = torch.stack([i[mask], j[mask]], dim=-1)
            edge_lists.append(e)
            type_lists.append(torch.full((e.shape[0],), int(EdgeType.COMPLEX),
                                           dtype=torch.long, device=device))
            thick_lists.append(torch.ones(e.shape[0], dtype=dtype, device=device))

    # REACTIVE — caller-supplied
    if reactive_pairs is not None and reactive_pairs.numel() > 0:
        rp = reactive_pairs.long()
        e = rp[:, :2]
        # If a third column gives a rate index, use it as thickness; else 1.0
        if rp.shape[1] > 2:
            thick = rp[:, 2].to(dtype)
            thick = thick / max(thick.max().item(), 1.0)   # normalize 0..1
        else:
            thick = torch.ones(e.shape[0], dtype=dtype, device=device)
        edge_lists.append(e)
        type_lists.append(torch.full((e.shape[0],), int(EdgeType.REACTIVE),
                                       dtype=torch.long, device=device))
        thick_lists.append(thick)

    # MEMBRANE — caller-supplied
    if membrane_anchors is not None and membrane_anchors.numel() > 0:
        e = membrane_anchors.long()[:, :2]
        edge_lists.append(e)
        type_lists.append(torch.full((e.shape[0],), int(EdgeType.MEMBRANE),
                                       dtype=torch.long, device=device))
        thick_lists.append(torch.ones(e.shape[0], dtype=dtype, device=device))

    # REGULATORY — caller-supplied
    if regulatory_pairs is not None and regulatory_pairs.numel() > 0:
        e = regulatory_pairs.long()[:, :2]
        edge_lists.append(e)
        type_lists.append(torch.full((e.shape[0],), int(EdgeType.REGULATORY),
                                       dtype=torch.long, device=device))
        # Regulatory edges default to medium thickness (0.5)
        thick_lists.append(torch.full((e.shape[0],), 0.5, dtype=dtype, device=device))

    # ENZYME — caller-supplied directional
    if enzyme_substrate is not None and enzyme_substrate.numel() > 0:
        e = enzyme_substrate.long()[:, :2]
        edge_lists.append(e)
        type_lists.append(torch.full((e.shape[0],), int(EdgeType.ENZYME),
                                       dtype=torch.long, device=device))
        thick_lists.append(torch.full((e.shape[0],), 0.7, dtype=dtype, device=device))

    if not edge_lists:
        edges = torch.empty((0, 2), dtype=torch.long, device=device)
        r = torch.empty((0,), dtype=dtype, device=device)
        edge_type = torch.empty((0,), dtype=torch.long, device=device)
        thickness = torch.empty((0,), dtype=dtype, device=device)
        return {'edges': edges, 'r': r, 'edge_type': edge_type,
                'thickness': thickness}

    edges = torch.cat(edge_lists, dim=0)
    edge_type = torch.cat(type_lists, dim=0)
    thickness = torch.cat(thick_lists, dim=0)
    r = (pos[edges[:, 1]] - pos[edges[:, 0]]).norm(dim=-1)
    return {'edges': edges, 'r': r, 'edge_type': edge_type,
            'thickness': thickness}
