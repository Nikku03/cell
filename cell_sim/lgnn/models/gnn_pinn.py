"""M-PINN: physics-informed LGNN with hardwired stoichiometric mass balance.

Architecture: same encoder as CellGNNv2 (CfC + axis-2 attention) producing
hidden state h per node, but the output head is PINNHead instead of a
free Linear. PINNHead pulls hidden state at the F_* species (one per
PINN-tracked reaction), decodes log-space rates v_log, exponentiates,
applies Δx = S · v via the hardwired stoichiometric matrix, and returns
to log space.

Mass / atom conservation is GUARANTEED by construction for reactions and
species that map into S. For LGNN-only species (RP_*, RPM_*, PM_*, DM_*,
M_* without an SBML counterpart, etc.), the PINN's Δx is zero; their
dynamics must come from a residual encoder branch.

To keep the first version simple, this minimum-viable M-PINN does NOT
include a residual branch. Non-spatial LGNN species are treated as
having Δx = 0 from the PINN. The training script can supervise them
via a separate count-regularization loss term (Critique 11) to prevent
them from drifting wildly during long rollouts.

See cell_sim/lgnn/models/pinn_head.py for the log-space bridge and the
mass-conservation unit tests.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.models.gnn_v2 import _CfCAttentionGNNLayer
from cell_sim.lgnn.models.pinn_head import PINNHead


class CellGNN_PINN(nn.Module):
    """Encoder identical to CellGNNv2 but the output is a PINNHead.

    Args
    ----
    graph        : SpeciesGraph (use v6 graph with RATE_LAW / MASS_BALANCE
                    split for best results)
    stoich_matrix: (n_species, n_reactions) float tensor — Δx = S · v.
                    Load via
                    cell_sim.lgnn.data.stoichiometric_matrix
                    .load_stoichiometric_matrix_for_pinn(...)
    flux_indices : (n_reactions,) long tensor — index of each reaction's
                    F_* species in graph.row_names. Same loader returns this.
    hidden       : hidden dim of encoder
    n_layers     : number of message-passing layers
    rate_clip    : clamp on log-space rate to prevent expm1 blowup
                    (default 12.0 -> max linear rate ~163K which is huge but
                     finite; tighter clamps like 8.0 reduce blowup risk but
                     may clip slow-but-still-large fluxes)

    Forward
    -------
    x_log : (B, n_species) — current state in signed-log1p

    Returns
    -------
    x_next_log : (B, n_species) — predicted next state
                  Where mapped_mask is False, x_next == x (no PINN update)
    v_log      : (B, n_reactions) — predicted log-space rate vector
                  (for direct flux supervision against F_* observations)
    """

    def __init__(self, graph: SpeciesGraph,
                 stoich_matrix: torch.Tensor,
                 flux_indices: torch.Tensor,
                 hidden: int = 64,
                 n_layers: int = 3,
                 use_checkpoint: bool = False,
                 edge_chunk_size: Optional[int] = None,
                 cfc_tau_min: float = 0.1,
                 rate_clip: float = 12.0):
        super().__init__()
        self.n_nodes = graph.n_nodes
        self.hidden = hidden
        self.n_layers = n_layers
        self.use_checkpoint = use_checkpoint
        self.edge_chunk_size = edge_chunk_size
        edge_attr_dim = int(graph.edge_attr.shape[1])

        # ---- Encoder buffers (identical to CellGNNv2) ----
        sort_idx = torch.argsort(graph.edge_kind, stable=True)
        ei = graph.edge_index[:, sort_idx].contiguous()
        ea = graph.edge_attr[sort_idx].contiguous()
        ek = graph.edge_kind[sort_idx].contiguous()
        self.register_buffer('edge_index', ei)
        self.register_buffer('edge_attr', ea)
        self.register_buffer('edge_kind', ek)

        # ---- Encoder layers ----
        self.input_proj = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([
            _CfCAttentionGNNLayer(
                hidden=hidden, edge_attr_dim=edge_attr_dim,
                n_nodes=graph.n_nodes, cfc_tau_min=cfc_tau_min,
            )
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden)

        # ---- Physics-informed output head ----
        # Replaces the free Linear(hidden, 1) of CellGNNv2 with the
        # stoichiometric-matrix-enforced update.
        self.pinn_head = PINNHead(
            hidden=hidden,
            stoich_matrix=stoich_matrix,
            flux_indices=flux_indices,
            rate_clip=rate_clip,
        )

    def forward(self, x: torch.Tensor,
                edge_dropout_p: float = 0.0,
                return_attention_entropy: bool = False,
                return_hidden: bool = False):
        B, N = x.shape
        h = self.input_proj(x.unsqueeze(-1))                  # (B, N, hidden)

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

        # Physics-informed output
        x_next_log, v_log = self.pinn_head(x, h)

        result = (x_next_log, v_log)
        if return_attention_entropy:
            result = result + (total_entropy.mean(),)
        if return_hidden:
            result = result + (h,)
        return result


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
