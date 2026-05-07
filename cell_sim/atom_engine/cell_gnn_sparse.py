"""SparseCellGNN — CellGNN augmented with a K-pattern sparse bottleneck head.

Bolts the SparseConceptLNN recipe (K=128 patterns, top-k selection, diversity +
reconstruction + alignment losses) onto the heterogeneous-edge CellGNN node
embeddings. The result is a dynamics surrogate that *also* exposes an
interpretable pattern code per particle.

Pattern head
------------
After message passing produces node embeddings h ∈ R^(N, hidden), we project to
K pattern logits, sparsify (top-k), and decode back via:

    pattern_logits = W_proj @ h               # (N, K)
    pattern_act    = top_k_gate(pattern_logits, k)   # (N, K), sparse
    h_recon        = W_dec @ pattern_act      # (N, hidden)

The pattern activations replace dense embeddings as the interpretable substrate
the cascade stacker / downstream heads can read.

Modality-aware patterns
-----------------------
Different edge types contribute different views of the particle. We pool node
hiddens per edge type to get K-pattern activations per modality (SPATIAL,
COVALENT, COMPLEX). A cosine-alignment loss then encourages the same particle
to fire the same patterns across modalities — that's the SparseLNN multimodal
recipe applied to edge types.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cell_gnn import CellGNN, EdgeType, N_EDGE_TYPES


# ---------------------------------------------------------------------------
def top_k_gate(x: torch.Tensor, k: int) -> torch.Tensor:
    """Keep top-k values per row, zero the rest. Differentiable through the
    selected positions.
    """
    if k >= x.shape[-1]:
        return F.relu(x)
    topk_vals, topk_idx = torch.topk(x, k, dim=-1)
    mask = torch.zeros_like(x)
    mask.scatter_(-1, topk_idx, 1.0)
    return F.relu(x) * mask


class SparsePatternHead(nn.Module):
    """K-pattern bottleneck. Projects node hidden -> K logits, top-k sparsify,
    and reconstructs hidden via a decoder. Returns activations + recon."""

    def __init__(self, hidden: int, k_patterns: int = 128, top_k: int = 5):
        super().__init__()
        self.k_patterns = k_patterns
        self.top_k = top_k
        self.proj = nn.Linear(hidden, k_patterns)
        self.decoder = nn.Linear(k_patterns, hidden)

    def forward(self, h: torch.Tensor) -> dict:
        logits = self.proj(h)                       # (N, K)
        act = top_k_gate(logits, self.top_k)        # (N, K) sparse
        recon = self.decoder(act)                   # (N, H)
        return {'pattern_logits': logits,
                'pattern_act': act,
                'hidden_recon': recon}


# ---------------------------------------------------------------------------
class SparseCellGNN(nn.Module):
    """CellGNN with sparse pattern head and modality-aware pattern views.

    Forward returns the same three task outputs as CellGNN.predict_all
    (forces, reaction_logits, spawn_destroy_logits), plus:
      pattern_act       (N, K)            single shared pattern code
      modality_acts     dict[mod_name -> (N, K)]   per-edge-type pattern view
      hidden_recon      (N, H)
      raw_hidden        (N, H)            pre-pattern hiddens, for recon target
    """

    def __init__(self, n_species: int, hidden: int = 64, n_rounds: int = 3,
                 n_reaction_classes: int = 64, r_cut: float = 50.0,
                 n_extra_node_features: int = 4,
                 k_patterns: int = 128, top_k: int = 5,
                 modalities: tuple = (EdgeType.SPATIAL,
                                       EdgeType.COVALENT,
                                       EdgeType.COMPLEX)):
        super().__init__()
        self.backbone = CellGNN(n_species=n_species, hidden=hidden,
                                 n_rounds=n_rounds,
                                 n_reaction_classes=n_reaction_classes,
                                 r_cut=r_cut,
                                 n_extra_node_features=n_extra_node_features)
        self.pattern_head = SparsePatternHead(hidden, k_patterns, top_k)
        self.modalities = tuple(int(m) for m in modalities)
        # Per-modality pattern projections (share decoder for parameter economy)
        self.modality_proj = nn.ModuleDict({
            str(int(m)): nn.Linear(hidden, k_patterns) for m in modalities
        })
        self.k_patterns = k_patterns
        self.top_k = top_k

    # ------------------------------------------------------------------
    def _pool_per_modality(self, h: torch.Tensor, edges: torch.Tensor,
                            edge_type: torch.Tensor) -> dict:
        """Mean-pool destination embeddings per edge type. Returns a dict
        mod_id_str -> (N, H) where particles touched by that modality have
        a pooled view, others get zeros.
        """
        n, hd = h.shape
        out = {}
        for m in self.modalities:
            key = str(m)
            mask = (edge_type == m)
            if mask.sum() == 0:
                out[key] = torch.zeros_like(h)
                continue
            sub = edges[mask]
            src, dst = sub[:, 0], sub[:, 1]
            agg = torch.zeros_like(h)
            count = torch.zeros(n, device=h.device, dtype=h.dtype)
            agg.index_add_(0, dst, h[src])
            count.index_add_(0, dst,
                             torch.ones(src.shape[0], device=h.device,
                                        dtype=h.dtype))
            count = count.clamp(min=1.0).unsqueeze(-1)
            out[key] = agg / count
        return out

    # ------------------------------------------------------------------
    def forward(self, species_id: torch.Tensor, extra_feats: torch.Tensor,
                edges: torch.Tensor, r: torch.Tensor,
                edge_type: torch.Tensor, pos: torch.Tensor,
                thickness: Optional[torch.Tensor] = None,
                flags: Optional[torch.Tensor] = None) -> dict:
        h = self.backbone.encode(species_id, extra_feats, edges, r, edge_type,
                                  thickness, flags)
        # Forces (equivariant) — replicate from backbone
        n = species_id.shape[0]
        ef = self.backbone.featurizer(r, edge_type, thickness, flags)
        if edges.shape[0]:
            src = edges[:, 0]; dst = edges[:, 1]
            msg_in = torch.cat([h[src], h[dst], ef], dim=-1)
            scalar = self.backbone.edge_scalar(msg_in).squeeze(-1)
            r_ij = pos[dst] - pos[src]
            r_mag = torch.clamp(r_ij.norm(dim=-1, keepdim=True), min=1e-6)
            u_ij = r_ij / r_mag
            edge_force = scalar.unsqueeze(-1) * u_ij
            forces = torch.zeros(n, 3, device=h.device, dtype=pos.dtype)
            forces.index_add_(0, src, edge_force)
        else:
            forces = torch.zeros(n, 3, device=h.device, dtype=pos.dtype)
        # Sparse pattern head (shared across modalities)
        pat = self.pattern_head(h)
        # Per-modality pooled views, projected to K-pattern space
        pooled = self._pool_per_modality(h, edges, edge_type)
        modality_acts = {}
        for key, h_mod in pooled.items():
            logits = self.modality_proj[key](h_mod)
            modality_acts[key] = top_k_gate(logits, self.top_k)
        return {
            'forces': forces,
            'reaction_logits': self.backbone.reaction_head(h),
            'spawn_destroy_logits': self.backbone.spawn_head(h),
            'pattern_act': pat['pattern_act'],
            'pattern_logits': pat['pattern_logits'],
            'hidden_recon': pat['hidden_recon'],
            'raw_hidden': h,
            'modality_acts': modality_acts,
        }


# ---------------------------------------------------------------------------
def compute_sparse_losses(out: dict, target_forces: torch.Tensor,
                           w_force: float = 1.0,
                           w_recon: float = 0.1,
                           w_diversity: float = 0.1,
                           w_align: float = 0.3) -> dict:
    """Return a dict {name: scalar_loss, total: scalar} for the sparse model."""
    losses = {}
    losses['force'] = F.mse_loss(out['forces'], target_forces)
    losses['recon'] = F.mse_loss(out['hidden_recon'], out['raw_hidden'])
    # Diversity: encourage broad coverage across the K patterns over the batch
    # Mean activation per pattern; we want the entropy of that distribution
    # to be high (uniform usage).
    pat_mean = out['pattern_act'].mean(dim=0)            # (K,)
    p = F.softmax(pat_mean, dim=-1)
    entropy = -(p * p.clamp(min=1e-9).log()).sum()
    target_entropy = torch.log(torch.tensor(float(out['pattern_act'].shape[1]),
                                              device=p.device))
    losses['diversity'] = (target_entropy - entropy)     # smaller is better
    # Cross-modality alignment: cosine similarity between modality views
    mods = list(out['modality_acts'].values())
    if len(mods) >= 2:
        align = 0.0
        n_pairs = 0
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                a = F.normalize(mods[i], dim=-1)
                b = F.normalize(mods[j], dim=-1)
                cos = (a * b).sum(dim=-1)
                align = align + (1.0 - cos).mean()
                n_pairs += 1
        losses['align'] = align / max(n_pairs, 1)
    else:
        losses['align'] = torch.zeros((), device=target_forces.device)
    total = (w_force * losses['force']
             + w_recon * losses['recon']
             + w_diversity * losses['diversity']
             + w_align * losses['align'])
    losses['total'] = total
    return losses
