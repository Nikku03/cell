"""M3 — Source-1-complete CfC-attention dynamics surrogate.

Architecture identical to M2 (CfC + axis-2 attention + bug-1 self_mlp)
except for the input projection: M3 accepts a TWO-channel per-species
input — the current signed-log1p count AND the per-species cross-
replicate standard deviation at the same timestep.

The variance channel exposes the simulator's stochasticity to the model
directly. Counts in the cell that are tightly regulated (essential
machinery: aaRS, polymerases, ribosomes) have systematically lower
cross-replicate variance than dispensable accessory species — a signal
M2 had to infer indirectly from temporal patterns. M3 sees it as a
feature.

The training loss is multi-task: count rows / flux rows / cumulative-
counter rows are weighted separately so the under-represented F_* (175
rows out of 8572) and rate-counters PM/RPM/DM (1400) contribute their
own gradient signal instead of being averaged into the count majority.
The model architecture doesn't have to know about row categories —
only the loss does. See train_m3.py.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.models.gnn_v2 import _CfCAttentionGNNLayer


class CellGNNv3(nn.Module):
    """M3 — 2-channel input (count + cross-replicate std). Otherwise
    drop-in compatible with M2's training loop: forward returns count
    deltas of the same shape (B, N).

    Constructor signature matches M2 + n_input_channels for forward-
    compat. Default is 2 (count, variance); set to 1 to disable the
    variance channel and exactly recover M2.
    """

    def __init__(self, graph: SpeciesGraph,
                 hidden: int = 64,
                 n_layers: int = 3,
                 n_input_channels: int = 2,
                 use_checkpoint: bool = False,
                 edge_chunk_size: Optional[int] = None,
                 cfc_tau_min: float = 0.1):
        super().__init__()
        self.n_nodes = graph.n_nodes
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_input_channels = n_input_channels
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

        self.input_proj = nn.Linear(n_input_channels, hidden)
        self.layers = nn.ModuleList([
            _CfCAttentionGNNLayer(
                hidden=hidden, edge_attr_dim=edge_attr_dim,
                n_nodes=graph.n_nodes, cfc_tau_min=cfc_tau_min,
            )
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_head = nn.Linear(hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_var: Optional[torch.Tensor] = None,
        edge_dropout_p: float = 0.0,
        return_attention_entropy: bool = False,
    ):
        """x: (B, N) signed-log1p counts.
        x_var: (B, N) per-species cross-replicate std (signed-log1p),
            required if n_input_channels >= 2. If None and channels=2,
            falls back to a zero tensor (degrades cleanly during inference
            when variance isn't available).
        """
        B, N = x.shape
        if self.n_input_channels == 1:
            channels = x.unsqueeze(-1)                       # (B, N, 1)
        else:
            if x_var is None:
                x_var = torch.zeros_like(x)
            # Stack along the channel axis. Order: [count, variance, ...].
            channels = torch.stack([x, x_var], dim=-1)        # (B, N, 2)
        h = self.input_proj(channels)                         # (B, N, hidden)

        if edge_dropout_p > 0.0 and self.training:
            E = int(self.edge_index.shape[1])
            edge_mask = (torch.rand(E, device=x.device) > edge_dropout_p).float()
        else:
            edge_mask = None

        total_entropy = torch.zeros(B, N, device=x.device, dtype=h.dtype)
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
