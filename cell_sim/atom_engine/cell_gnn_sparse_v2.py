"""SparseCellGNN v2 — physics-aware refinement of cell_gnn_sparse.

Improvements over v1 that are *not* runtime monkey-patches but proper modules:

1. **Newton's third law (symmetric edges)**. The edge-scalar function is built
   so f(i, j) = -f(j, i). We iterate over unordered pairs and add ±F to each
   endpoint. Halves parameters used for the same job and bakes in conservation.

2. **Distance-decay envelope**. The edge scalar is multiplied by a smooth
   cutoff envelope that decays to zero at r_cut. No hard discontinuity.

3. **True per-edge-type message MLPs**. Each EdgeType gets its own message
   MLP. The previous attempt routed everything through one shared MLP — this
   actually realises the heterogeneous-edge premise.

4. **Iterative refinement**. After the first force prediction, append the
   normalised force as an extra node feature and run a second pass. Two-shot
   inference often beats a single deeper pass.

5. **Multi-task heads**. Forces (regression) + reactivity (binary BCE) +
   species reconstruction (classification from pattern code only — concept
   supervision). The auxiliary losses regularise the pattern code.

6. **Sinkhorn-style pattern usage normaliser**. Optional doubly-stochastic
   normalisation across the (batch × K) matrix of pattern logits before
   top-k. Strongly improves pattern coverage.

7. **Pattern dropout**. With probability p, drop one of the active patterns
   during training. Forces redundant pattern usage.

8. **Configurable losses**: MSE, Huber, magnitude-weighted MSE.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cell_gnn import EdgeType, N_EDGE_TYPES, EdgeFeaturizer
from .cell_gnn_sparse import top_k_gate


# ---------------------------------------------------------------------------
def smooth_cutoff(r: torch.Tensor, r_cut: float) -> torch.Tensor:
    """Polynomial cutoff envelope: 1 at r=0, 0 at r>=r_cut, smooth derivative.

    Equation 8 from Behler-Parrinello: 0.5*(cos(pi*r/r_cut)+1) for r<r_cut.
    """
    out = 0.5 * (torch.cos(torch.pi * r.clamp(max=r_cut) / r_cut) + 1.0)
    out = torch.where(r < r_cut, out, torch.zeros_like(out))
    return out


# ---------------------------------------------------------------------------
class SparsePatternHeadV2(nn.Module):
    """K-pattern bottleneck with optional Sinkhorn normalisation + dropout."""

    def __init__(self, hidden: int, k_patterns: int = 128, top_k: int = 8,
                 sinkhorn: bool = False, sinkhorn_iters: int = 3,
                 pattern_dropout: float = 0.0):
        super().__init__()
        self.k_patterns = k_patterns
        self.top_k = top_k
        self.sinkhorn = sinkhorn
        self.sinkhorn_iters = sinkhorn_iters
        self.pattern_dropout = pattern_dropout
        self.proj = nn.Linear(hidden, k_patterns)
        self.decoder = nn.Linear(k_patterns, hidden)

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        """Doubly-stochastic normalisation across (N, K). Encourages every
        pattern to be used by ~ N/K particles."""
        x = logits - logits.max()                # numerical stability
        x = torch.exp(x)
        for _ in range(self.sinkhorn_iters):
            x = x / x.sum(dim=0, keepdim=True).clamp(min=1e-9)   # column-norm
            x = x / x.sum(dim=1, keepdim=True).clamp(min=1e-9)   # row-norm
        return x

    def forward(self, h: torch.Tensor) -> dict:
        logits = self.proj(h)
        if self.sinkhorn:
            scores = self._sinkhorn(logits)
        else:
            scores = logits
        act = top_k_gate(scores, self.top_k)
        if self.training and self.pattern_dropout > 0:
            mask = (torch.rand_like(act) > self.pattern_dropout).float()
            act = act * mask
        recon = self.decoder(act)
        return {'pattern_logits': logits, 'pattern_act': act,
                'hidden_recon': recon}


# ---------------------------------------------------------------------------
class HeteroMessage(nn.Module):
    """One MLP per EdgeType. Routed by edge_type tensor."""

    def __init__(self, hidden: int, ef_dim: int):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2 + ef_dim, hidden), nn.SiLU(),
                           nn.Linear(hidden, hidden))
            for _ in range(N_EDGE_TYPES)
        ])

    def forward(self, msg_in: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(msg_in.shape[0], self.mlps[0][-1].out_features,
                           device=msg_in.device, dtype=msg_in.dtype)
        for t in range(N_EDGE_TYPES):
            mask = (edge_type == t)
            if mask.sum() == 0:
                continue
            out[mask] = self.mlps[t](msg_in[mask])
        return out


# ---------------------------------------------------------------------------
class SparseCellGNNV2(nn.Module):
    def __init__(self, n_species: int = 5, hidden: int = 64, n_rounds: int = 3,
                 n_reaction_classes: int = 8, r_cut: float = 12.0,
                 n_extra_node_features: int = 4,
                 k_patterns: int = 128, top_k: int = 8,
                 hetero_msg: bool = True,
                 sinkhorn_patterns: bool = False,
                 pattern_dropout: float = 0.0,
                 iterative_refinement: bool = False):
        super().__init__()
        self.hidden = hidden
        self.n_species = n_species
        self.n_rounds = n_rounds
        self.r_cut = r_cut
        self.iterative_refinement = iterative_refinement

        self.species_emb = nn.Embedding(n_species + 4, hidden)
        # +1 extra dim if iterative refinement (force magnitude as extra feature)
        extra_dim = n_extra_node_features + (4 if iterative_refinement else 0)
        self.encoder = nn.Sequential(
            nn.Linear(hidden + extra_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.ln_h = nn.LayerNorm(hidden)
        self.featurizer = EdgeFeaturizer(r_cut=r_cut)
        ef_dim = self.featurizer.feat_dim

        if hetero_msg:
            self.msg = nn.ModuleList(
                [HeteroMessage(hidden, ef_dim) for _ in range(n_rounds)])
        else:
            self.msg = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden * 2 + ef_dim, hidden), nn.SiLU(),
                               nn.Linear(hidden, hidden))
                for _ in range(n_rounds)])
        self.hetero_msg = hetero_msg

        self.update = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(),
                           nn.Linear(hidden, hidden))
            for _ in range(n_rounds)])

        self.pattern_head = SparsePatternHeadV2(
            hidden, k_patterns, top_k,
            sinkhorn=sinkhorn_patterns,
            pattern_dropout=pattern_dropout)

        self.reaction_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_reaction_classes))
        self.reactivity_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))            # binary: |f| > threshold ?
        self.species_from_pattern_head = nn.Linear(k_patterns, n_species + 4)

        # Edge scalar — input is symmetric in (i, j) by construction
        self.edge_scalar = nn.Sequential(
            nn.Linear(hidden * 2 + ef_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    # ------------------------------------------------------------------
    def _msg_round(self, h, edges, edge_features, edge_type, round_idx):
        if edges.shape[0] == 0:
            return h
        src, dst = edges[:, 0], edges[:, 1]
        msg_in = torch.cat([h[src], h[dst], edge_features], dim=-1)
        if self.hetero_msg:
            m = self.msg[round_idx](msg_in, edge_type)
        else:
            m = self.msg[round_idx](msg_in)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        return h + self.update[round_idx](torch.cat([h, agg], dim=-1))

    # ------------------------------------------------------------------
    def encode(self, species_id, extra_feats, edges, r, edge_type,
               thickness=None, flags=None):
        h0 = torch.cat([self.species_emb(species_id), extra_feats], dim=-1)
        h = self.ln_h(self.encoder(h0))
        ef = self.featurizer(r, edge_type, thickness, flags)
        for ri in range(self.n_rounds):
            h = self._msg_round(h, edges, ef, edge_type, ri)
        return h, ef

    def _force_pass(self, h, ef, edges, pos, r):
        """Symmetric edge force: f_i contribution is +s*u, f_j is -s*u."""
        n = h.shape[0]
        if edges.shape[0] == 0:
            return torch.zeros(n, 3, device=h.device, dtype=pos.dtype)
        src, dst = edges[:, 0], edges[:, 1]
        # Symmetrise: edge scalar is invariant to swapping src and dst
        msg_a = torch.cat([h[src], h[dst], ef], dim=-1)
        msg_b = torch.cat([h[dst], h[src], ef], dim=-1)
        scalar = 0.5 * (self.edge_scalar(msg_a) + self.edge_scalar(msg_b)
                         ).squeeze(-1)
        # Distance envelope
        env = smooth_cutoff(r, self.r_cut)
        scalar = scalar * env
        r_ij = pos[dst] - pos[src]
        r_mag = torch.clamp(r_ij.norm(dim=-1, keepdim=True), min=1e-6)
        u_ij = r_ij / r_mag
        edge_force = scalar.unsqueeze(-1) * u_ij
        forces = torch.zeros(n, 3, device=h.device, dtype=pos.dtype)
        forces.index_add_(0, src, edge_force)
        forces.index_add_(0, dst, -edge_force)
        return forces

    def forward(self, species_id, extra_feats, edges, r, edge_type, pos,
                thickness=None, flags=None) -> dict:
        if self.iterative_refinement:
            # Run once with zero-augmented features, take the prediction, then
            # append it to extra_feats and run again.
            zeros = torch.zeros(extra_feats.shape[0], 4,
                                  device=extra_feats.device,
                                  dtype=extra_feats.dtype)
            extra_aug = torch.cat([extra_feats, zeros], dim=-1)
            h, ef = self.encode(species_id, extra_aug, edges, r, edge_type,
                                 thickness, flags)
            forces1 = self._force_pass(h, ef, edges, pos, r)
            f_norm = forces1.norm(dim=-1, keepdim=True)
            extra_aug2 = torch.cat([extra_feats, forces1, f_norm], dim=-1)
            h, ef = self.encode(species_id, extra_aug2, edges, r, edge_type,
                                 thickness, flags)
            forces = self._force_pass(h, ef, edges, pos, r)
        else:
            h, ef = self.encode(species_id, extra_feats, edges, r, edge_type,
                                 thickness, flags)
            forces = self._force_pass(h, ef, edges, pos, r)
        pat = self.pattern_head(h)
        return {
            'forces': forces,
            'reaction_logits': self.reaction_head(h),
            'reactivity_logit': self.reactivity_head(h).squeeze(-1),
            'species_from_pattern': self.species_from_pattern_head(pat['pattern_act']),
            'pattern_act': pat['pattern_act'],
            'pattern_logits': pat['pattern_logits'],
            'hidden_recon': pat['hidden_recon'],
            'raw_hidden': h,
        }


# ---------------------------------------------------------------------------
def force_loss(pred, target, kind='mse', weight_by_magnitude=False):
    if kind == 'mse':
        per = (pred - target) ** 2
    elif kind == 'huber':
        per = F.huber_loss(pred, target, delta=1.0, reduction='none')
    else:
        raise ValueError(kind)
    if weight_by_magnitude:
        # Higher weight on particles with bigger |target|
        w = target.norm(dim=-1, keepdim=True).clamp(min=0.1)
        return (per * w).mean()
    return per.mean()


def compute_v2_losses(out: dict, target_forces: torch.Tensor,
                       target_species: torch.Tensor,
                       target_reactivity: Optional[torch.Tensor] = None,
                       w_force: float = 1.0,
                       w_recon: float = 0.1,
                       w_diversity: float = 0.1,
                       w_concept: float = 0.2,
                       w_reactivity: float = 0.1,
                       force_kind: str = 'mse',
                       weight_by_magnitude: bool = False) -> dict:
    losses = {}
    losses['force'] = force_loss(out['forces'], target_forces,
                                  kind=force_kind,
                                  weight_by_magnitude=weight_by_magnitude)
    losses['recon'] = F.mse_loss(out['hidden_recon'], out['raw_hidden'])
    pat_mean = out['pattern_act'].mean(dim=0)
    p = F.softmax(pat_mean, dim=-1)
    entropy = -(p * p.clamp(min=1e-9).log()).sum()
    target_entropy = torch.log(torch.tensor(float(out['pattern_act'].shape[1]),
                                              device=p.device))
    losses['diversity'] = (target_entropy - entropy)
    # Concept supervision: predict species from pattern_act alone
    losses['concept'] = F.cross_entropy(out['species_from_pattern'],
                                         target_species)
    if target_reactivity is not None:
        losses['reactivity'] = F.binary_cross_entropy_with_logits(
            out['reactivity_logit'], target_reactivity.float())
    else:
        losses['reactivity'] = torch.zeros((), device=target_forces.device)
    total = (w_force * losses['force']
             + w_recon * losses['recon']
             + w_diversity * losses['diversity']
             + w_concept * losses['concept']
             + w_reactivity * losses['reactivity'])
    losses['total'] = total
    return losses
