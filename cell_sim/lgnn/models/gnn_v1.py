"""M1 — hetero-edge GNN delta-predictor.

Predicts Δsigned-log1p(x) given x and a fixed species-reaction graph
(see species_graph.py). Architecturally:

  x ∈ ℝ^N  ──────►  Linear(1, h)  ──►  h ∈ ℝ^{N×h}
                                          │
                                          ▼
                            ┌── GNNLayer × n_layers ──┐
                            │   per-EdgeKind message  │
                            │   MLP, sum-aggregate,   │
                            │   residual+LN           │
                            └─────────────────────────┘
                                          │
                                          ▼
                            LN  ──►  Linear(h, 1)  ──►  Δx ∈ ℝ^N

Each EdgeKind in the graph (SBML / SELF_LOOP / TRANSCRIPTION /
TRANSLATION / TRANSLOCATION / DEGRADATION / FLUX_COUPLING) gets its
own message MLP, so the model can learn distinct propagation rules
per reaction class. The graph itself is stored as buffers (sorted by
kind so per-kind slicing is contiguous) — moves with the model on
.to(device).

Gradient checkpointing is on by default. Each GNNLayer's forward is
wrapped in torch.utils.checkpoint so activations are recomputed in
backward instead of stashed; ~30% extra compute, halves activation
memory. Critical for k>1 rollout backprop on a T4.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from cell_sim.lgnn.data.species_graph import EdgeKind, SpeciesGraph


N_EDGE_KINDS = max(int(k) for k in EdgeKind) + 1


class _GNNLayer(nn.Module):
    """One round of message passing.

    For each EdgeKind k present in the graph, run an MLP over
    (h_src ‖ h_dst ‖ edge_attr) → ℝ^h, sum-aggregate at destination
    nodes, then h ← LayerNorm(h + agg).
    """

    def __init__(self, hidden: int, edge_attr_dim: int):
        super().__init__()
        in_dim = 2 * hidden + edge_attr_dim
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(N_EDGE_KINDS)
        ])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                kind_offsets: torch.Tensor) -> torch.Tensor:
        """h: (B, N, hidden), edge_index: (2, E), edge_attr: (E, A),
        kind_offsets: (N_KINDS+1,) cumulative start index per kind."""
        B = h.shape[0]
        agg = torch.zeros_like(h)
        for k in range(N_EDGE_KINDS):
            start = int(kind_offsets[k].item())
            end   = int(kind_offsets[k + 1].item())
            if start == end:
                continue
            src = edge_index[0, start:end]                       # (E_k,)
            dst = edge_index[1, start:end]
            attr = edge_attr[start:end]                          # (E_k, A)
            h_src = h.index_select(1, src)                       # (B, E_k, h)
            h_dst = h.index_select(1, dst)
            attr_b = attr.unsqueeze(0).expand(B, -1, -1)         # (B, E_k, A)
            inp = torch.cat([h_src, h_dst, attr_b], dim=-1)
            m = self.msg_mlps[k](inp)                            # (B, E_k, h)
            agg.index_add_(1, dst, m)
        return self.norm(h + agg)


class CellGNNv1(nn.Module):
    """Input: (B, N) signed-log1p of counts.  Output: (B, N) Δsigned-log1p."""

    def __init__(self, graph: SpeciesGraph,
                 hidden: int = 64,
                 n_layers: int = 3,
                 use_checkpoint: bool = True):
        super().__init__()
        self.n_nodes = graph.n_nodes
        self.hidden = hidden
        self.n_layers = n_layers
        self.use_checkpoint = use_checkpoint
        edge_attr_dim = int(graph.edge_attr.shape[1])

        # Sort edges by kind so per-kind slicing is contiguous. Avoids
        # masked indexing in the per-layer loop, saves ~all the
        # gather-overhead in forward.
        sort_idx = torch.argsort(graph.edge_kind, stable=True)
        ei  = graph.edge_index[:, sort_idx].contiguous()
        ea  = graph.edge_attr[sort_idx].contiguous()
        ek  = graph.edge_kind[sort_idx].contiguous()
        offsets = [0]
        for k in range(N_EDGE_KINDS):
            offsets.append(offsets[-1] + int((ek == k).sum().item()))
        kind_offsets = torch.tensor(offsets, dtype=torch.long)

        self.register_buffer('edge_index',   ei)
        self.register_buffer('edge_attr',    ea)
        self.register_buffer('kind_offsets', kind_offsets)

        self.input_proj = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([
            _GNNLayer(hidden=hidden, edge_attr_dim=edge_attr_dim)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N)
        h = self.input_proj(x.unsqueeze(-1))                 # (B, N, hidden)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                h = ckpt.checkpoint(
                    layer, h,
                    self.edge_index, self.edge_attr, self.kind_offsets,
                    use_reentrant=False,
                )
            else:
                h = layer(h, self.edge_index, self.edge_attr,
                          self.kind_offsets)
        h = self.out_norm(h)
        return self.out_head(h).squeeze(-1)                  # (B, N)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
