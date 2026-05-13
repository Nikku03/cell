"""M4 - CfC-attention dynamics surrogate with gene-class node features.

Extends M3 (Source-1-complete) by adding a STATIC per-node feature vector
built from syn3A.gb gene annotations. Each LGNN node gets a per-locus
feature broadcast: all species belonging to locus L (G_L_C1/C2, R_L/R_L_d,
P_L, PM_L, D_L/DM_L, RP_L_*, RPM_L) share the same gene-class feature
row. Cofactor species (ribosomeP, RNAP, Degradosome, chromosome) and
flux/metabolite species (F_*, M_*) get zero vectors.

The gene-class features were the highest-leverage unused Source 3 signal:
they're exactly the features v15's symbolic detector uses to hit MCC=0.53.
Adding them as LGNN inputs means the model now knows WHAT each gene does,
not just how its counts move.

Input to forward: (B, N) count + (B, N) variance + (N, F_static) per-node
class features, concatenated to (B, N, 2 + F_static) before input_proj.

Static features included:
  14 keyword categories  is_ribosomal, is_synthetase, is_trna, is_polymerase,
                          is_dna_machinery, is_translation, is_transcription,
                          is_membrane, is_atp_binding, is_kinase,
                          is_phosphatase, is_synthase, is_uncharacterized,
                          is_metabolism
  3 sequence summaries   protein_length_aa (log-normalized), tm_helix_count
                          (raw), has_signal_pep (0/1)
  2 gene metadata        has_gene_name (0/1), length_bp (log-normalized)

Total: 19 per-locus features, broadcast to ~17 species per locus on average.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.models.gnn_v2 import _CfCAttentionGNNLayer


class CellGNNv4(nn.Module):
    """M4 — 2 dynamic input channels + static per-node features.

    static_features: (N, F_static) tensor, registered as a buffer so it
    moves to device with .to() but isn't a trainable parameter. Provided
    at construction time and broadcast across the batch each forward pass.
    """

    def __init__(self, graph: SpeciesGraph,
                 static_features: torch.Tensor,
                 hidden: int = 64,
                 n_layers: int = 3,
                 n_dynamic_channels: int = 2,
                 use_checkpoint: bool = False,
                 edge_chunk_size: Optional[int] = None,
                 cfc_tau_min: float = 0.1):
        super().__init__()
        if static_features.shape[0] != graph.n_nodes:
            raise ValueError(
                f'static_features rows {static_features.shape[0]} != '
                f'graph.n_nodes {graph.n_nodes}'
            )
        self.n_nodes           = graph.n_nodes
        self.hidden            = hidden
        self.n_layers          = n_layers
        self.n_dynamic_channels = n_dynamic_channels
        self.n_static_features  = int(static_features.shape[1])
        self.use_checkpoint    = use_checkpoint
        self.edge_chunk_size   = edge_chunk_size

        edge_attr_dim = int(graph.edge_attr.shape[1])

        sort_idx = torch.argsort(graph.edge_kind, stable=True)
        ei = graph.edge_index[:, sort_idx].contiguous()
        ea = graph.edge_attr[sort_idx].contiguous()
        ek = graph.edge_kind[sort_idx].contiguous()
        self.register_buffer('edge_index', ei)
        self.register_buffer('edge_attr', ea)
        self.register_buffer('edge_kind', ek)
        self.register_buffer('static_features', static_features.float())

        total_in = n_dynamic_channels + self.n_static_features
        self.input_proj = nn.Linear(total_in, hidden)
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
        B, N = x.shape
        dyn_channels = [x]
        if self.n_dynamic_channels >= 2:
            if x_var is None:
                x_var = torch.zeros_like(x)
            dyn_channels.append(x_var)
        dyn = torch.stack(dyn_channels, dim=-1)                       # (B, N, 2)
        # Broadcast static features across batch
        stat = self.static_features.to(dyn.dtype).unsqueeze(0)        # (1, N, F)
        stat = stat.expand(B, -1, -1)                                  # (B, N, F)
        channels = torch.cat([dyn, stat], dim=-1)                     # (B, N, 2+F)
        h = self.input_proj(channels)                                  # (B, N, hidden)

        if edge_dropout_p > 0.0 and self.training:
            E = int(self.edge_index.shape[1])
            edge_mask = (torch.rand(E, device=x.device) > edge_dropout_p).float()
        else:
            edge_mask = None

        total_entropy = torch.zeros(B, N, device=x.device, dtype=h.dtype)
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
