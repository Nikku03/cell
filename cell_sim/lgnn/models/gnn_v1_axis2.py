"""M1 + Axis 2 — hetero-edge GNN with attention + counterfactual support.

Adds two interventions on top of `gnn_v1.py`:

1. **Edge attention with sparsity penalty.** Per layer, a per-edge
   attention scalar is computed by an MLP over (h_src, h_dst, edge_feat),
   then segment-softmax-normalized over the destination node's incoming
   edges. Messages are weighted by attention before aggregation. Each
   layer also returns the per-(batch, dst-node) attention entropy so the
   training loss can add a sparsity term `λ_attn · mean(entropy)` —
   minimising entropy pushes each node's attention onto a few edges.

2. **Counterfactual edge dropout.** `forward(x, edge_dropout_p=p)` zeros
   out a Bernoulli(p) random subset of edges before the softmax (sets
   their logits to −∞). The training loop pairs this with the standard
   forward to compute `L_dropout`. Combined with `L_full` this teaches
   the model that connections are load-bearing — without dropout, the
   GNN learns correlations; with dropout, removing an edge has to
   actually hurt prediction on the affected node.

The dropout mask is sampled once per forward call and used for all
layers — same mask across the message-passing stack, since per-layer
different masks would prevent the model from learning a coherent
"which edges are load-bearing" signal.

Parameter cost: ~2× a `gnn_v1.py` layer (one extra small attention MLP
per EdgeKind). Forward cost: ~30% over `gnn_v1.py` per layer due to
the segment-softmax. Combined with the paired (full + dropout) forward
in training, M1+axis2 is ~2.6× the per-step cost of M1.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from cell_sim.lgnn.data.species_graph import EdgeKind, SpeciesGraph


N_EDGE_KINDS = max(int(k) for k in EdgeKind) + 1


def _segment_softmax(
    logit: torch.Tensor,
    dst: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """logit: (B, E), dst: (E,) long. Returns α: (B, E), softmax over
    edges sharing the same destination index. -inf logits → α=0."""
    B = logit.shape[0]
    device = logit.device
    dst_b = dst.unsqueeze(0).expand(B, -1)               # (B, E)

    max_per_node = torch.full((B, n_nodes), -float('inf'), device=device)
    max_per_node = max_per_node.scatter_reduce(
        1, dst_b, logit, reduce='amax', include_self=True,
    )
    max_per_node = torch.where(torch.isinf(max_per_node),
                                torch.zeros_like(max_per_node),
                                max_per_node)
    max_per_edge = max_per_node.gather(1, dst_b)         # (B, E)
    exp_logit = (logit - max_per_edge).exp()             # 0 where logit=-inf

    sum_per_node = torch.zeros(B, n_nodes, device=device)
    sum_per_node.index_add_(1, dst, exp_logit)
    sum_per_edge = sum_per_node.gather(1, dst_b)
    return exp_logit / (sum_per_edge + 1e-12)


class _AttentionGNNLayer(nn.Module):
    """One round of message passing with per-edge attention.

    For each EdgeKind k present in the graph:
      msg_e   = MsgMLP_k(concat(h_src, h_dst, edge_attr))
      logit_e = AttnMLP_k(concat(h_src, h_dst, edge_attr))      (scalar)
    Then logits across all incoming edges to a destination are
    softmax-normalized into α, msg ← α · msg, and aggregated.
    """

    def __init__(self, hidden: int, edge_attr_dim: int):
        super().__init__()
        in_dim = 2 * hidden + edge_attr_dim
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            ) for _ in range(N_EDGE_KINDS)
        ])
        self.attn_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, max(hidden // 2, 8)),
                nn.SiLU(),
                nn.Linear(max(hidden // 2, 8), 1),
            ) for _ in range(N_EDGE_KINDS)
        ])
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        kind_offsets: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """h: (B, N, hidden). edge_mask: optional (E,) float in {0,1}.
        Returns (h_new, entropy_per_node)."""
        B, N, H = h.shape
        device = h.device

        # Compute per-kind messages and attention logits, gather into
        # a single concatenated tensor over all kinds in offset order.
        msgs, logits, dsts = [], [], []
        for k in range(N_EDGE_KINDS):
            start = int(kind_offsets[k].item())
            end   = int(kind_offsets[k + 1].item())
            if start == end:
                continue
            src = edge_index[0, start:end]
            dst = edge_index[1, start:end]
            attr = edge_attr[start:end]
            h_src = h.index_select(1, src)
            h_dst = h.index_select(1, dst)
            attr_b = attr.unsqueeze(0).expand(B, -1, -1)
            inp = torch.cat([h_src, h_dst, attr_b], dim=-1)

            msgs.append(self.msg_mlps[k](inp))                       # (B, E_k, h)
            logits.append(self.attn_mlps[k](inp).squeeze(-1))        # (B, E_k)
            dsts.append(dst)

        if not msgs:
            return self.norm(h), torch.zeros(B, N, device=device)

        msg_cat = torch.cat(msgs, dim=1)                             # (B, E_active, h)
        logit_cat = torch.cat(logits, dim=1)                         # (B, E_active)
        dst_cat = torch.cat(dsts, dim=0)                             # (E_active,)

        # Edge mask (counterfactual dropout): masked-out edges get
        # logit -inf so segment-softmax weights them to 0.
        if edge_mask is not None:
            keep = edge_mask[:dst_cat.shape[0]].bool() \
                if edge_mask.shape[0] >= dst_cat.shape[0] \
                else edge_mask.bool()
            logit_cat = logit_cat.masked_fill(
                ~keep.unsqueeze(0), float('-inf'),
            )

        alpha = _segment_softmax(logit_cat, dst_cat, N)              # (B, E_active)
        weighted = msg_cat * alpha.unsqueeze(-1)                     # (B, E_active, h)

        agg = torch.zeros(B, N, H, device=device)
        agg.index_add_(1, dst_cat, weighted)
        h_new = self.norm(h + agg)

        # Entropy per (batch, dst): -Σ α log α  (sum over incoming edges)
        # Edges with α=0 contribute 0·log(0+eps) ≈ 0, safe.
        a_log_a = alpha * (alpha + 1e-12).log()
        ent_per_node = torch.zeros(B, N, device=device)
        ent_per_node.index_add_(1, dst_cat, -a_log_a)

        return h_new, ent_per_node


class CellGNNv1Axis2(nn.Module):
    """M1 + Axis 2 GNN. Same I/O contract as `CellGNNv1`.

    forward(x, edge_dropout_p=0.0, return_attention_entropy=False)
        x: (B, N) signed-log1p counts
        returns: (B, N) Δsigned-log1p
                  optionally + scalar mean attention entropy

    The dropout is sampled once inside forward() and shared across
    all layers (consistent ablation through the message-passing stack).
    """

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

        sort_idx = torch.argsort(graph.edge_kind, stable=True)
        ei = graph.edge_index[:, sort_idx].contiguous()
        ea = graph.edge_attr[sort_idx].contiguous()
        ek = graph.edge_kind[sort_idx].contiguous()
        offsets = [0]
        for k in range(N_EDGE_KINDS):
            offsets.append(offsets[-1] + int((ek == k).sum().item()))
        kind_offsets = torch.tensor(offsets, dtype=torch.long)

        self.register_buffer('edge_index', ei)
        self.register_buffer('edge_attr', ea)
        self.register_buffer('kind_offsets', kind_offsets)

        self.input_proj = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([
            _AttentionGNNLayer(hidden=hidden, edge_attr_dim=edge_attr_dim)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_head = nn.Linear(hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_dropout_p: float = 0.0,
        return_attention_entropy: bool = False,
    ):
        h = self.input_proj(x.unsqueeze(-1))                         # (B, N, hidden)

        # One mask per forward, shared across layers.
        if edge_dropout_p > 0.0 and self.training:
            E = int(self.edge_index.shape[1])
            edge_mask = (torch.rand(E, device=x.device) > edge_dropout_p).float()
        else:
            edge_mask = None

        total_entropy = torch.zeros(x.shape[0], self.n_nodes,
                                     device=x.device)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                h, ent = ckpt.checkpoint(
                    layer, h,
                    self.edge_index, self.edge_attr, self.kind_offsets,
                    edge_mask, use_reentrant=False,
                )
            else:
                h, ent = layer(h, self.edge_index, self.edge_attr,
                                self.kind_offsets, edge_mask)
            total_entropy = total_entropy + ent

        h = self.out_norm(h)
        pred = self.out_head(h).squeeze(-1)                          # (B, N)
        if return_attention_entropy:
            return pred, total_entropy.mean()
        return pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
