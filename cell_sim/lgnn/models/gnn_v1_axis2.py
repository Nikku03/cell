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
    edges sharing the same destination index. -inf logits → α=0.
    All helper tensors take logit's dtype so this works under bf16
    autocast (scatter_reduce requires matching dtypes).
    """
    B = logit.shape[0]
    device = logit.device
    dtype = logit.dtype
    dst_b = dst.unsqueeze(0).expand(B, -1)               # (B, E)

    max_per_node = torch.full((B, n_nodes), -float('inf'),
                                device=device, dtype=dtype)
    max_per_node = max_per_node.scatter_reduce(
        1, dst_b, logit, reduce='amax', include_self=True,
    )
    max_per_node = torch.where(torch.isinf(max_per_node),
                                torch.zeros_like(max_per_node),
                                max_per_node)
    max_per_edge = max_per_node.gather(1, dst_b)         # (B, E)
    # `.exp()` is autocast-promoted to fp32 for numerical stability; after
    # max-subtraction the values are in [0, 1] so casting back to source
    # dtype is safe and lets index_add_ stay dtype-consistent.
    exp_logit = (logit - max_per_edge).exp().to(dtype)

    sum_per_node = torch.zeros(B, n_nodes, device=device, dtype=dtype)
    sum_per_node.index_add_(1, dst, exp_logit)
    sum_per_edge = sum_per_node.gather(1, dst_b)
    return exp_logit / (sum_per_edge + 1e-12)


class _AttentionGNNLayer(nn.Module):
    """One round of message passing with per-edge attention.

    Optimized: kind information is fed into a single shared MLP via a
    learned embedding (one row per EdgeKind), instead of running 7
    separate per-kind MLPs in a Python loop. Eliminates the loop's
    O(N_EDGE_KINDS) kernel-launch overhead — each forward is now a
    single big MLP call over all E edges, which uses the GPU at full
    throughput. Empirically ~5-8× faster on T4/L4/A100 vs the per-kind
    version for N_EDGE_KINDS=7. Same expressive capacity (the kind
    embedding lets the MLP specialize internally).

    For each edge e (kind k):
      input_e = concat(h_src, h_dst, edge_attr_e, kind_embed[k])
      msg_e   = MsgMLP(input_e)
      logit_e = AttnMLP(input_e)                                 (scalar)

    Then logits over each destination's incoming edges are
    softmax-normalized into α, msg ← α · msg, aggregated.

    `chunk_size` controls edge chunking. When set, edges are processed
    `chunk_size` at a time and each chunk's MLP forward is wrapped in
    torch.utils.checkpoint.checkpoint, so the (B, E_chunk, 2H+A+D) input
    tensor is recomputed during backward instead of retained. Cuts
    activation memory ~5× per layer at the cost of ~30% backward compute,
    which lets B return to 128+ without OOM at high rollout k.
    """

    def __init__(self, hidden: int, edge_attr_dim: int,
                 kind_embed_dim: int = 16):
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
        self.norm = nn.LayerNorm(hidden)

    def _msg_logit_chunk(
        self,
        h: torch.Tensor,
        src_c: torch.Tensor,
        dst_c: torch.Tensor,
        attr_c: torch.Tensor,
        kind_emb_c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (msg, logit) for one edge-chunk. Pure function of
        the args — safe to wrap in torch.utils.checkpoint."""
        B = h.shape[0]
        h_src = h.index_select(1, src_c)                # (B, E_c, h)
        h_dst = h.index_select(1, dst_c)
        attr_b = attr_c.unsqueeze(0).expand(B, -1, -1)
        kind_b = kind_emb_c.unsqueeze(0).expand(B, -1, -1)
        inp = torch.cat([h_src, h_dst, attr_b, kind_b], dim=-1)
        msg = self.msg_mlp(inp)                         # (B, E_c, h)
        logit = self.attn_mlp(inp).squeeze(-1)          # (B, E_c)
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
        """h: (B, N, hidden), edge_index: (2, E), edge_attr: (E, A),
        edge_kind: (E,) long. edge_mask: optional (E,) float in {0,1}.
        chunk_size: if set, process edges in chunks under checkpoint.
        Returns (h_new, entropy_per_node)."""
        B, N, H = h.shape
        device = h.device
        E = int(edge_index.shape[1])

        src = edge_index[0]
        dst = edge_index[1]
        kind_emb = self.kind_embedding(edge_kind)                    # (E, D)

        if chunk_size is None or chunk_size >= E:
            # Single-shot path. Fastest when memory permits.
            msg, logit = self._msg_logit_chunk(
                h, src, dst, edge_attr, kind_emb,
            )
        else:
            # Chunked path. Each chunk's MLP forward is checkpointed,
            # so the (B, E_chunk, 2H+A+D) input tensor isn't retained
            # for backward. msg/logit outputs ARE retained (small,
            # constant-size). Compute the full output buffers and fill
            # them per chunk.
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

        if edge_mask is not None:
            logit = logit.masked_fill(
                ~edge_mask.bool().unsqueeze(0), float('-inf'),
            )

        alpha = _segment_softmax(logit, dst, N)                      # (B, E)
        weighted = msg * alpha.unsqueeze(-1)                         # (B, E, h)

        agg = torch.zeros(B, N, H, device=device, dtype=msg.dtype)
        agg.index_add_(1, dst, weighted)
        h_new = self.norm(h + agg)

        # Entropy per (batch, dst): -Σ α log α  (.log() autocast-promotes
        # to fp32; cast back to alpha.dtype so index_add_ matches.)
        a_log_a = (alpha * (alpha + 1e-12).log()).to(alpha.dtype)
        ent_per_node = torch.zeros(B, N, device=device, dtype=alpha.dtype)
        ent_per_node.index_add_(1, dst, -a_log_a)

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
                 use_checkpoint: bool = True,
                 edge_chunk_size: Optional[int] = None):
        super().__init__()
        self.n_nodes = graph.n_nodes
        self.hidden = hidden
        self.n_layers = n_layers
        self.use_checkpoint = use_checkpoint
        self.edge_chunk_size = edge_chunk_size
        edge_attr_dim = int(graph.edge_attr.shape[1])

        # Sort edges by kind for cache-friendliness. With the fused
        # layer the sort isn't strictly required (no per-kind slicing)
        # but it costs nothing and helps the GPU's L2 cache when
        # many edges of the same kind are processed together.
        sort_idx = torch.argsort(graph.edge_kind, stable=True)
        ei = graph.edge_index[:, sort_idx].contiguous()
        ea = graph.edge_attr[sort_idx].contiguous()
        ek = graph.edge_kind[sort_idx].contiguous()

        self.register_buffer('edge_index', ei)
        self.register_buffer('edge_attr', ea)
        self.register_buffer('edge_kind', ek)

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
        pred = self.out_head(h).squeeze(-1)                          # (B, N)
        if return_attention_entropy:
            return pred, total_entropy.mean()
        return pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
