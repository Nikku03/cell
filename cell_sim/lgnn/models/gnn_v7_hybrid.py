"""M7 - hybrid PINN + residual surrogate, M3 encoder + PINNHead.

Architecture:
  - Encoder: identical to CellGNNv3 (M3). 2-channel input (count +
    cross-replicate sigma, both signed-log1p), input_proj -> 3x
    _CfCAttentionGNNLayer -> out_norm. RATE_LAW / MASS_BALANCE edge
    split (N_EDGE_KINDS=9) consumed automatically through the graph.
  - Head: PINNHead with use_residual=True.
      For SBML-covered species (rows where the stoichiometric matrix
      has at least one non-zero entry) the next-step state is computed
      by the hardwired log-space exponential bridge
          dx_linear   = S @ expm1(v_log)
          x_next_log  = signed_log1p( x_linear + dx_linear )
      Mass conservation is exact by construction.
      For non-SBML species (RP_*, RPM_*, PM_*, DM_*, M_*-without-SBML,
      etc.) a learned residual MLP predicts a small additive delta in
      signed-log1p space. This is what unfreezes RP/RPM/PM/DM dynamics
      relative to a pure-PINN baseline that masks those species out.

The M7 contribution vs M-PINN: wire M3's 2-channel encoder + RATE_LAW/
MASS_BALANCE edge split into PINNHead. M-PINN shipped with a 1-channel
M2-style encoder; M7 keeps the M3 input (count + sigma) so the same
variance signal that lifted single-step R^2 in M3 is available to the
PINN's rate predictions.

Forward returns (x_next_log, v_log), optionally with attention entropy.
The trainer supervises x_next_log against the next-step state across
the count / flux / cumulative loss categories AND supervises v_log
against the observed F_* fluxes directly (flux-primary loss, Critique 11
of lgnn_critiques_and_roadmap.md).
"""
from __future__ import annotations
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from cell_sim.lgnn.data.species_graph import SpeciesGraph
from cell_sim.lgnn.models.gnn_v2 import _CfCAttentionGNNLayer
from cell_sim.lgnn.models.pinn_head import PINNHead


class CellGNNv7Hybrid(nn.Module):
    """M7 - CellGNNv3-style encoder followed by a hybrid PINNHead.

    Constructor matches CellGNN_PINN's argument set plus n_input_channels
    (for the variance channel) and use_residual (defaults True; expose it
    so the trainer can ablate the non-SBML branch if desired).

    Args
    ----
    graph        : SpeciesGraph. Build with split_flux_coupling=True so
                    the encoder sees RATE_LAW / MASS_BALANCE edge kinds.
    stoich_matrix: (n_species, n_reactions) float tensor. Build via
                    cell_sim.lgnn.data.stoichiometric_matrix
                    .load_stoichiometric_matrix_for_pinn(sbml_xml, row_names).
    flux_indices : (n_reactions,) long tensor returned by the same loader.
    hidden, n_layers, cfc_tau_min : as M3.
    n_input_channels: 2 (count + variance) by default. Set 1 to disable
                       the variance channel and exactly recover the
                       M-PINN-style 1-channel input.
    rate_clip    : log-space clamp on predicted v_log. 6.0 matches
                    PINNHead's default; |v_linear| < ~400, safe in bf16.
    use_residual : enable the residual head for non-SBML species.
                    Default True (this is the entire point of M7); set
                    False only for ablations.
    use_checkpoint, edge_chunk_size : memory knobs forwarded to the
                    CfC-attention layers.
    """

    def __init__(
        self,
        graph: SpeciesGraph,
        stoich_matrix: torch.Tensor,
        flux_indices: torch.Tensor,
        hidden: int = 64,
        n_layers: int = 3,
        n_input_channels: int = 2,
        static_features: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False,
        edge_chunk_size: Optional[int] = None,
        cfc_tau_min: float = 0.1,
        rate_clip: float = 6.0,
        use_residual: bool = True,
    ):
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

        # Static per-species categorical features (e.g. row-type indicators).
        # Constant across batch and time, broadcast at forward. If None,
        # M7 acts identically to the pre-M7.1 version (n_static=0). The
        # M5 failure mode (shortcut learning) is mitigated by keeping
        # n_static small and categorical - 8 role flags can only encode
        # 8 species types, not 8572 individual rows, so the model can't
        # use them as a row-identity lookup.
        if static_features is not None:
            self.register_buffer('static_features', static_features.float())
            self.n_static = int(static_features.shape[1])
        else:
            self.static_features = None
            self.n_static = 0

        total_in = n_input_channels + self.n_static
        self.input_proj = nn.Linear(total_in, hidden)
        self.layers = nn.ModuleList([
            _CfCAttentionGNNLayer(
                hidden=hidden, edge_attr_dim=edge_attr_dim,
                n_nodes=graph.n_nodes, cfc_tau_min=cfc_tau_min,
            )
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.pinn_head = PINNHead(
            hidden=hidden,
            stoich_matrix=stoich_matrix,
            flux_indices=flux_indices,
            rate_clip=rate_clip,
            use_residual=use_residual,
        )

    # ------------------------------------------------------------------
    @property
    def s_covered_mask(self) -> torch.Tensor:
        """Bool mask (n_species,): True where the species has at least one
        non-zero entry in the stoichiometric matrix (i.e., updated by the
        PINN branch). The non-covered complement is what the residual head
        is responsible for.
        """
        return self.pinn_head.s_covered_mask > 0

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        x_var: Optional[torch.Tensor] = None,
        edge_dropout_p: float = 0.0,
        return_attention_entropy: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """x: (B, N) signed-log1p counts.
        x_var: (B, N) per-species cross-replicate sigma (signed-log1p);
            required when n_input_channels >= 2. If None, falls back to
            zeros (degrades cleanly at inference when sigma isn't available).
        """
        B, N = x.shape
        if self.n_input_channels == 1:
            dyn = x.unsqueeze(-1)
        else:
            if x_var is None:
                x_var = torch.zeros_like(x)
            dyn = torch.stack([x, x_var], dim=-1)

        if self.n_static > 0:
            # static_features: (N, F_static) -> broadcast to (B, N, F_static)
            static_b = self.static_features.unsqueeze(0).expand(B, -1, -1)
            static_b = static_b.to(dyn.dtype)
            channels = torch.cat([dyn, static_b], dim=-1)
        else:
            channels = dyn
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
        x_next_log, v_log = self.pinn_head(x, h)

        if return_attention_entropy:
            return x_next_log, v_log, total_entropy.mean()
        return x_next_log, v_log


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
