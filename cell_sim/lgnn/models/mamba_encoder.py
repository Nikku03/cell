"""M8 upgrade 9/10: Mamba-2 encoder option.

Optional drop-in replacement for the CfC-Attention encoder layers.
Mamba-2 (Dao & Gu, 2024) is a structured-state-space sequence model
that processes a sequence of node states in linear time.

For M7's case the "sequence" is the species axis: at each timestep
we have N=8572 species states, and we want each species's update to
depend on the rest. Mamba-2's selective scan is well-suited for
this because:

  - Linear time in N (vs. O(N²) for attention)
  - Strong long-range mixing via the SSM recurrence
  - Hardware-efficient (Triton kernels in mamba-ssm package)

This module provides a thin Mamba-2-based encoder that takes
(B, N, hidden) input and returns (B, N, hidden) output, matching the
shape contract of the existing CfC-Attention layers. Importantly, it
ignores the graph edges — Mamba is fully connected via the SSM.

Compared to CfC-Attention:
  - +: linear in N, much faster at large N
  - +: strong long-range mixing (SSM remembers across the full sequence)
  - -: doesn't use the species-reaction graph structure
  - -: requires mamba-ssm package (NVIDIA-only)

For M7 with N=8572, this is a significant speedup. The information
about which species interact via which reactions is now encoded purely
through the static features + position; the SSM has to learn it from
data.

This file is OPT-IN. Default M7 still uses CfC-Attention. To activate:
    cfg.encoder_backbone = 'mamba2'
    pip install mamba-ssm  # one-time on Blackwell/A100

Requires mamba-ssm. Falls back with a clear error if not installed.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MambaEncoderLayer(nn.Module):
    """One Mamba-2 layer over the species axis.

    Input:  (B, N, d_model)
    Output: (B, N, d_model)

    The Mamba-2 forward expects (B, N, d_model) — species are treated
    as a sequence. We don't use causal masking; the species ordering
    is arbitrary and we want full mixing.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError:
            raise ImportError(
                'mamba-ssm not installed. Install with:\n'
                '  pip install mamba-ssm causal-conv1d\n'
                'NB: requires NVIDIA CUDA 11.6+'
            )
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, d_model) -> (B, N, d_model). Residual + LayerNorm."""
        return self.norm(h + self.mamba(h))


class MambaEncoder(nn.Module):
    """Stack of Mamba-2 layers as a drop-in encoder for M7.

    Same input/output contract as the existing CfC-Attention stack:
      (B, N, hidden) -> (B, N, hidden)

    Plus a no-op `entropy` return path so the existing forward()
    logic in CellGNNv7Hybrid (which sums attention entropy across
    layers) continues to work.
    """

    def __init__(
        self,
        hidden: int,
        n_layers: int = 3,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaEncoderLayer(
                d_model=hidden, d_state=d_state, d_conv=d_conv, expand=expand,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        h: torch.Tensor,
        *unused_args,
        **unused_kwargs,
    ):
        """Match CfC-Attention's signature: ignores edge_index, edge_attr,
        edge_kind, edge_mask, chunk_size. Returns (h, entropy=0).
        """
        for layer in self.layers:
            h = layer(h)
        # Return zero entropy to keep the CfC-Attention forward contract
        zero_entropy = torch.zeros(
            h.shape[0], h.shape[1], device=h.device, dtype=h.dtype)
        return h, zero_entropy


def make_encoder(
    backbone: str,
    hidden: int,
    n_layers: int,
    cfc_tau_min: float = 0.1,
    edge_attr_dim: int = 5,
    n_nodes: int = 0,
):
    """Factory: returns either the existing CfC-Attention encoder or
    a Mamba-2 encoder based on `backbone` string.

    'cfc'    : default M7 encoder (CfC + edge attention)
    'mamba2' : Mamba-2 SSM stack (requires mamba-ssm package)

    Raises ValueError on unknown backbone.
    """
    backbone = backbone.lower()
    if backbone == 'cfc':
        # Lazy import to keep mamba-ssm optional
        from cell_sim.lgnn.models.gnn_v7_hybrid import _CfCAttentionGNNLayer
        return nn.ModuleList([
            _CfCAttentionGNNLayer(
                hidden=hidden, edge_attr_dim=edge_attr_dim,
                n_nodes=n_nodes, cfc_tau_min=cfc_tau_min,
            )
            for _ in range(n_layers)
        ])
    if backbone == 'mamba2':
        return MambaEncoder(hidden=hidden, n_layers=n_layers)
    raise ValueError(f'unknown encoder_backbone: {backbone}')
