"""M6 - Physics-patched LGNN.

Implements roadmap Steps 2 + 3 (Step 1 is in train_m6.py):

  Step 2: Softplus additive aggregation (replaces softmax).
          Allows chemical pressures to ACCUMULATE instead of being averaged.
          softmax(logit) normalizes per-dst to sum=1 - violates mass conservation
          when many substrates flow into one node. softplus(logit) keeps positive
          weights but allows unbounded sum.

  Step 3: RATE_LAW / MASS_BALANCE edge-kind split (data side; this model uses
          N_EDGE_KINDS = 9, up from 7 in M3).

Architecture is otherwise identical to CellGNNv2 (CfC + axis-2 attention +
bug-1 self_mlp + self-loop masking).

Caveats:
  - Softplus output is unbounded for large logits. The CfC update bounds h_new
    via sigmoid(±gate) * A/B so blowup in `agg` is dampened in the layer. But
    if instability is observed, swap softplus for sigmoid-gated additive
    (`sigmoid(logit) * msg` then sum) which bounds each edge contribution to msg.
  - Self-loop edges still masked to 0 (Bug 1 fix carried over).

See memory_bank/architecture/lgnn_critiques_and_roadmap.md for the full
9-critique context and the 5-step roadmap M6 implements steps 1-3 of.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from cell_sim.lgnn.data.species_graph import EdgeKind, SpeciesGraph
from cell_sim.lgnn.models.gnn_v1_axis2 import N_EDGE_KINDS
from cell_sim.lgnn.models.gnn_v2 import _CfCAttentionGNNLayer as _M2Layer


class _M6Layer(nn.Module):
    """CfC + axis-2 attention with SOFTPLUS additive aggregation (not softmax).

    The only mathematical difference from _CfCAttentionGNNLayer (gnn_v2.py)
    is the per-edge weight:
        M2:  alpha = softmax(logit, dst, N)        # per-dst sum = 1
        M6:  alpha = softplus(logit_masked)        # per-edge in [0, +inf)
    where `logit_masked` has self-loops set to -inf so softplus(-inf) = 0
    keeps the Bug 1 fix (self-loops contribute zero through this channel;
    the self_mlp branch handles unconditional self-update).
    """

    def __init__(self, hidden: int, edge_attr_dim: int, n_nodes: int,
                 kind_embed_dim: int = 16, cfc_tau_min: float = 0.1):
        super().__init__()
        in_dim = 2 * hidden + edge_attr_dim + kind_embed_dim
        self.kind_embedding = nn.Embedding(N_EDGE_KINDS, kind_embed_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        attn_hidden = max(hidden // 2, 8)
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim, attn_hidden),
            nn.SiLU(),
            nn.Linear(attn_hidden, 1),
        )
        self.self_mlp = nn.Linear(hidden, hidden)

        self.cfc_W_proj = nn.Linear(hidden, hidden)
        self.cfc_b_proj = nn.Linear(hidden, hidden)
        self.cfc_A = nn.Parameter(torch.empty(n_nodes, hidden))
        self.cfc_B = nn.Parameter(torch.empty(n_nodes, hidden))
        nn.init.normal_(self.cfc_A, std=0.01)
        nn.init.normal_(self.cfc_B, std=0.01)
        self.cfc_tau_min = cfc_tau_min

    def _msg_logit_chunk(self, h, src_c, dst_c, attr_c, kind_emb_c):
        B = h.shape[0]
        h_src = h.index_select(1, src_c)
        h_dst = h.index_select(1, dst_c)
        attr_b = attr_c.unsqueeze(0).expand(B, -1, -1)
        kind_b = kind_emb_c.unsqueeze(0).expand(B, -1, -1)
        inp = torch.cat([h_src, h_dst, attr_b, kind_b], dim=-1)
        msg = self.msg_mlp(inp)
        logit = self.attn_mlp(inp).squeeze(-1)
        return msg, logit

    def forward(self, h, edge_index, edge_attr, edge_kind,
                edge_mask=None, chunk_size=None):
        B, N, H = h.shape
        device = h.device
        E = int(edge_index.shape[1])

        src = edge_index[0]
        dst = edge_index[1]
        kind_emb = self.kind_embedding(edge_kind)

        if chunk_size is None or chunk_size >= E:
            msg, logit = self._msg_logit_chunk(
                h, src, dst, edge_attr, kind_emb,
            )
        else:
            msg = torch.empty(B, E, H, device=device, dtype=h.dtype)
            logit = torch.empty(B, E, device=device, dtype=h.dtype)
            for start in range(0, E, chunk_size):
                end = min(start + chunk_size, E)
                msg_c, logit_c = ckpt.checkpoint(
                    self._msg_logit_chunk,
                    h, src[start:end], dst[start:end],
                    edge_attr[start:end], kind_emb[start:end],
                    use_reentrant=False,
                )
                msg[:, start:end] = msg_c
                logit[:, start:end] = logit_c

        # Mask self-loops + edge dropout: replace with very negative value so
        # softplus output is ~0. (Bug 1 fix: self_mlp handles self-update; this
        # channel must not redundantly add self-state.)
        is_self_loop = (edge_kind == int(EdgeKind.SELF_LOOP))
        to_mask = is_self_loop
        if edge_mask is not None:
            to_mask = to_mask | (~edge_mask.bool())
        logit = logit.masked_fill(to_mask.unsqueeze(0), -50.0)

        # === STEP 2: Softplus additive aggregation (replaces _segment_softmax) ===
        # softplus is positive, smooth, and bounded below by 0. For large
        # negative logits (masked self-loops), softplus(-50) ≈ 4e-22 ≈ 0.
        # No per-dst normalization → chemical pressures sum, not average.
        alpha = torch.nn.functional.softplus(logit)            # (B, E), all positive
        weighted = msg * alpha.unsqueeze(-1)                   # (B, E, H)

        agg = torch.zeros(B, N, H, device=device, dtype=msg.dtype)
        agg.index_add_(1, dst, weighted)
        h_self = self.self_mlp(h)

        # CfC update (unchanged from M2)
        mp_input = agg + h_self
        W = self.cfc_W_proj(mp_input)
        b = self.cfc_b_proj(mp_input)
        W_clip = 1.0 / max(self.cfc_tau_min, 1e-6)
        W = W.clamp(-W_clip, W_clip)
        gate = W * h + b
        A_b = self.cfc_A.to(h.dtype).unsqueeze(0)
        B_b = self.cfc_B.to(h.dtype).unsqueeze(0)
        h_new = torch.sigmoid(-gate) * A_b + torch.sigmoid(gate) * B_b

        # Entropy proxy: for softplus, use the L2 norm of alpha as a
        # signal of how concentrated the attention is. NOT the same as
        # softmax entropy but serves the same regularization purpose.
        # The training script's attn_entropy penalty is purely a regularizer.
        ent_per_node = torch.zeros(B, N, device=device, dtype=alpha.dtype)
        ent_per_node.index_add_(1, dst, alpha)
        # Normalize to per-node mean alpha so the scale matches softmax-entropy.
        ent_per_node = ent_per_node / max(1.0, E / N)

        return h_new, ent_per_node


class CellGNNv6(nn.Module):
    """M6 — same I/O contract as CellGNNv2 / CellGNNv3.

    Differences from v2:
      - softplus additive aggregation (Step 2)
      - N_EDGE_KINDS = 9 implicit via the embedding sizing (Step 3 data side)
      - constructor takes the same args as v2 for drop-in replacement
    """

    def __init__(self, graph: SpeciesGraph,
                 hidden: int = 64,
                 n_layers: int = 3,
                 use_checkpoint: bool = False,
                 edge_chunk_size: Optional[int] = None,
                 cfc_tau_min: float = 0.1):
        super().__init__()
        self.n_nodes = graph.n_nodes
        self.hidden = hidden
        self.n_layers = n_layers
        self.use_checkpoint = use_checkpoint
        self.edge_chunk_size = edge_chunk_size
        edge_attr_dim = int(graph.edge_attr.shape[1])

        sort_idx = torch.argsort(graph.edge_kind, stable=True)
        ei = graph.edge_index[:, sort_idx].contiguous()
        ea = graph.edge_attr[sort_idx].contiguous()
        ek = graph.edge_kind[sort_idx].contiguous()
        self.register_buffer('edge_index', ei)
        self.register_buffer('edge_attr', ea)
        self.register_buffer('edge_kind', ek)

        self.input_proj = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([
            _M6Layer(hidden=hidden, edge_attr_dim=edge_attr_dim,
                     n_nodes=graph.n_nodes, cfc_tau_min=cfc_tau_min)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_head = nn.Linear(hidden, 1)

    def forward(self, x, edge_dropout_p=0.0, return_attention_entropy=False):
        h = self.input_proj(x.unsqueeze(-1))
        if edge_dropout_p > 0.0 and self.training:
            E = int(self.edge_index.shape[1])
            edge_mask = (torch.rand(E, device=x.device) > edge_dropout_p).float()
        else:
            edge_mask = None
        total_entropy = torch.zeros(x.shape[0], self.n_nodes,
                                     device=x.device, dtype=h.dtype)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                h, ent = ckpt.checkpoint(
                    layer, h, self.edge_index, self.edge_attr, self.edge_kind,
                    edge_mask, self.edge_chunk_size, use_reentrant=False,
                )
            else:
                h, ent = layer(h, self.edge_index, self.edge_attr,
                                self.edge_kind, edge_mask,
                                chunk_size=self.edge_chunk_size)
            total_entropy = total_entropy + ent
        h = self.out_norm(h)
        pred = self.out_head(h).squeeze(-1)
        if return_attention_entropy:
            return pred, total_entropy.mean()
        return pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
