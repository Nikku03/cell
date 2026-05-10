"""M2 Phase 1 — Closed-form Continuous-time (CfC) node dynamics over
the M1+axis2 graph attention layer.

The M1+axis2 layer's node update was:
    h_new = LayerNorm(h + agg + h_self)

This replaces it with the CfC closed-form continuous-time variant per
the M2 spec:

    W, b = small MLPs of (agg + h_self)              -- gating from messages
    W ← clamp(W, -1/τ_min, +1/τ_min)                -- prevent fast-species blowup
    gate = W * h + b
    h_new = σ(-gate) ⊙ A + σ(gate) ⊙ B

where A, B are learned per-species bias vectors. The CfC is the
closed-form solution to a leaky-integrator ODE whose decay rate is
modulated by the message-passing input — slow species get long
effective τ, fast species get short τ. This directly addresses the
M1+axis2 ceiling: single-step prediction couldn't capture the
multi-timescale dynamics of count-level state, but CfC's per-species
implicit integration can.

Bug-1 fix carried over: self-loops still masked from attention softmax,
self_mlp still delivers unconditional self-update (folded into the
"agg + h_self" message-input to the CfC update).

Parameters added per layer over M1+axis2:
  cfc_W_proj : Linear(hidden, hidden)        -- ~4 K
  cfc_b_proj : Linear(hidden, hidden)        -- ~4 K
  cfc_A      : Parameter(n_nodes, hidden)    -- 549 K at N=8572, h=64
  cfc_B      : Parameter(n_nodes, hidden)    -- 549 K
Per-layer total: ~1.1 M. Model with 3 layers: ~3.4 M (vs M1+axis2's 69 K).
The per-species A/B is the biology-aligned design — fast metabolites
need different unconditional biases than slow proteins.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from cell_sim.lgnn.data.species_graph import EdgeKind, SpeciesGraph
from cell_sim.lgnn.models.gnn_v1_axis2 import (
    _segment_softmax, N_EDGE_KINDS,
)


class _CfCAttentionGNNLayer(nn.Module):
    """Message passing + per-edge attention (same as M1+axis2) followed
    by a CfC node update instead of LayerNorm-residual."""

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
        # Self-update channel (Bug 1 fix from axis-2)
        self.self_mlp = nn.Linear(hidden, hidden)

        # CfC dynamics
        self.cfc_W_proj = nn.Linear(hidden, hidden)
        self.cfc_b_proj = nn.Linear(hidden, hidden)
        # Per-species learned biases. Initialized small to keep CfC near
        # zero output at start (prevents gradient explosion through the
        # sigmoid gates on the first few steps).
        self.cfc_A = nn.Parameter(torch.empty(n_nodes, hidden))
        self.cfc_B = nn.Parameter(torch.empty(n_nodes, hidden))
        nn.init.normal_(self.cfc_A, std=0.01)
        nn.init.normal_(self.cfc_B, std=0.01)
        self.cfc_tau_min = cfc_tau_min                         # 0.1·Δt default

    def _msg_logit_chunk(
        self,
        h: torch.Tensor,
        src_c: torch.Tensor,
        dst_c: torch.Tensor,
        attr_c: torch.Tensor,
        kind_emb_c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same as M1+axis2 — compute message + attention logit for one
        edge-chunk. Pure function, safe under torch.utils.checkpoint."""
        B = h.shape[0]
        h_src = h.index_select(1, src_c)
        h_dst = h.index_select(1, dst_c)
        attr_b = attr_c.unsqueeze(0).expand(B, -1, -1)
        kind_b = kind_emb_c.unsqueeze(0).expand(B, -1, -1)
        inp = torch.cat([h_src, h_dst, attr_b, kind_b], dim=-1)
        msg = self.msg_mlp(inp)
        logit = self.attn_mlp(inp).squeeze(-1)
        return msg, logit

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_kind: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Mask self-loops + counterfactual dropout (Bug 1 fix)
        is_self_loop = (edge_kind == int(EdgeKind.SELF_LOOP))
        to_mask = is_self_loop
        if edge_mask is not None:
            to_mask = to_mask | (~edge_mask.bool())
        logit = logit.masked_fill(to_mask.unsqueeze(0), float('-inf'))

        alpha = _segment_softmax(logit, dst, N)
        weighted = msg * alpha.unsqueeze(-1)

        agg = torch.zeros(B, N, H, device=device, dtype=msg.dtype)
        agg.index_add_(1, dst, weighted)
        h_self = self.self_mlp(h)

        # === CfC update (replaces M1+axis2's LayerNorm-residual) ===
        # The "input current" to the dynamics is the message + self-update.
        mp_input = agg + h_self                              # (B, N, H)
        W = self.cfc_W_proj(mp_input)
        b = self.cfc_b_proj(mp_input)
        # Clamp W so |W| ≤ 1/τ_min — prevents fast-species blowup. At
        # Δt=1 (1-second steps) and τ_min=0.1, this is ±10 effective.
        W_clip = 1.0 / max(self.cfc_tau_min, 1e-6)
        W = W.clamp(-W_clip, W_clip)
        gate = W * h + b                                     # (B, N, H)
        # Per-species A, B broadcast across batch dim.
        A_b = self.cfc_A.to(h.dtype).unsqueeze(0)            # (1, N, H)
        B_b = self.cfc_B.to(h.dtype).unsqueeze(0)
        h_new = torch.sigmoid(-gate) * A_b + torch.sigmoid(gate) * B_b

        # Entropy same as M1+axis2
        a_log_a = (alpha * (alpha + 1e-12).log()).to(alpha.dtype)
        ent_per_node = torch.zeros(B, N, device=device, dtype=alpha.dtype)
        ent_per_node.index_add_(1, dst, -a_log_a)

        return h_new, ent_per_node


class CellGNNv2(nn.Module):
    """M2 Phase 1 — same I/O contract as CellGNNv1Axis2, but with CfC
    node dynamics. Drop-in replacement; the training loop, loss,
    schedules, and dataset can be reused unchanged."""

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
            _CfCAttentionGNNLayer(
                hidden=hidden, edge_attr_dim=edge_attr_dim,
                n_nodes=graph.n_nodes, cfc_tau_min=cfc_tau_min,
            )
            for _ in range(n_layers)
        ])
        # Output-side LayerNorm kept (only the per-layer LN was replaced
        # by CfC — output normalization still helps the prediction head).
        self.out_norm = nn.LayerNorm(hidden)
        self.out_head = nn.Linear(hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_dropout_p: float = 0.0,
        return_attention_entropy: bool = False,
    ):
        h = self.input_proj(x.unsqueeze(-1))                 # (B, N, hidden)

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
                    layer, h,
                    self.edge_index, self.edge_attr, self.edge_kind,
                    edge_mask, self.edge_chunk_size,
                    use_reentrant=False,
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
