"""M-Bio - CellGNNv4 + per-gene essentiality classification head.

Built to learn two complementary objectives jointly:

  1. Dynamics: per-species Δx prediction (the v4/M3 head).
  2. Essentiality: per-gene P(essential) prediction from the encoder's
     hidden state at the corresponding G_<locus> node.

Why this architecture:
  Earlier post-mortem (M-PINN failure analysis) showed that the LGNN's
  encoder DOES capture essentiality-relevant information in its hidden
  representations, but the knockout_sweep readout extracts it
  inefficiently. A direct supervised head on h[G_<g>] makes that signal
  explicit. Then the encoder is regularised by both objectives, and at
  evaluation time we have TWO essentiality predictions to compare:
  the head's direct logit vs. the autoregressive knockout_sweep
  divergence.

The two heads share the encoder. Only the per-gene essentiality MLP
and a 0.1-dropout layer are essentiality-specific.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.models.gnn_v4 import CellGNNv4


class CellGNN_Bio(CellGNNv4):
    """v4 encoder + dynamics head + essentiality classification head.

    Args
    ----
    gene_indices : (n_genes,) long tensor — row indices of the G_<locus>
        species in the LGNN's row order. The essentiality head reads
        h at exactly these indices.
    essentiality_dropout : dropout in the head MLP (regularisation).

    Forward
    -------
    Standard v4 forward returns the per-species delta. To get
    essentiality logits in the same pass, set return_essentiality=True;
    the call returns (delta, ess_logits) or (delta, attn_entropy,
    ess_logits) when both flags are set.
    """

    def __init__(
        self,
        graph: SpeciesGraph,
        static_features: torch.Tensor,
        gene_indices: torch.Tensor,
        hidden: int = 64,
        n_layers: int = 3,
        n_dynamic_channels: int = 2,
        use_checkpoint: bool = False,
        edge_chunk_size: Optional[int] = None,
        cfc_tau_min: float = 0.1,
        essentiality_dropout: float = 0.1,
    ):
        super().__init__(
            graph=graph,
            static_features=static_features,
            hidden=hidden,
            n_layers=n_layers,
            n_dynamic_channels=n_dynamic_channels,
            use_checkpoint=use_checkpoint,
            edge_chunk_size=edge_chunk_size,
            cfc_tau_min=cfc_tau_min,
        )
        if gene_indices.numel() == 0:
            raise ValueError('gene_indices is empty — no G_<locus> rows in graph')
        self.register_buffer('gene_indices', gene_indices.long())
        self.n_genes = int(gene_indices.numel())
        self.essentiality_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(essentiality_dropout),
            nn.Linear(hidden, 1),
        )

    # --- staged forward so trainer can reuse the encoder ---

    def encode(
        self,
        x: torch.Tensor,
        x_var: Optional[torch.Tensor] = None,
        edge_dropout_p: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N = x.shape
        dyn_channels = [x]
        if self.n_dynamic_channels >= 2:
            if x_var is None:
                x_var = torch.zeros_like(x)
            dyn_channels.append(x_var)
        dyn = torch.stack(dyn_channels, dim=-1)
        stat = self.static_features.to(dyn.dtype).unsqueeze(0).expand(B, -1, -1)
        channels = torch.cat([dyn, stat], dim=-1)
        h = self.input_proj(channels)

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
        return h, total_entropy.mean()

    def predict_dynamics(self, h: torch.Tensor) -> torch.Tensor:
        return self.out_head(h).squeeze(-1)

    def predict_essentiality(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, hidden) -> per-batch per-gene logits (B, n_genes)."""
        h_genes = h.index_select(1, self.gene_indices)
        return self.essentiality_head(h_genes).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        x_var: Optional[torch.Tensor] = None,
        edge_dropout_p: float = 0.0,
        return_attention_entropy: bool = False,
        return_essentiality: bool = False,
    ):
        h, ent = self.encode(x, x_var, edge_dropout_p)
        delta = self.predict_dynamics(h)
        if not return_attention_entropy and not return_essentiality:
            return delta
        out = [delta]
        if return_attention_entropy:
            out.append(ent)
        if return_essentiality:
            out.append(self.predict_essentiality(h))
        return tuple(out)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
