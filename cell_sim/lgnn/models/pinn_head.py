"""PINN output head: hardwired mass-conservation via Δx = S · v.

Replaces the M3-style "per-species delta" output head with a physics-
informed update. The GNN's hidden representation is decoded into a
reaction rate vector v; the stoichiometric matrix S (loaded from .lm
via stoichiometric_matrix.py) then maps rates onto species deltas
through a NON-LEARNED matmul.

Mass / atom conservation is guaranteed by construction: no matter
how badly v is predicted, S · v respects the stoichiometric ratios
the simulator was built from.

The log-space exponential bridge

The whole LGNN operates in signed-log1p space (counts/fluxes span
6 orders of magnitude; linear gradients explode at high values).
But the stoichiometric equation Δx = S · v is strictly LINEAR.
Applying it directly to log-space tensors is mathematically wrong:

    naive (wrong): x_next_log = x_log + S @ v_log
                   ↑ this is log(x · exp(S @ v_log)), not log(x + S·v)

The bridge:
    1. v_linear     = signed_expm1(v_log)
    2. x_linear     = signed_expm1(x_log)
    3. dx_linear    = S @ v_linear                  (hardwired, exact)
    4. x_next_linear= x_linear + dx_linear
    5. x_next_log   = signed_log1p(x_next_linear)

Gradient analysis:
    ∂x_next_log/∂v_log
        = ∂x_next_log/∂x_next_linear
          · ∂x_next_linear/∂v_linear
          · ∂v_linear/∂v_log
        ≈ (1 / x_next_linear) · S · v_linear
        ≈ S · (v_linear / x_next_linear)
    Since x_next_linear ~ v_linear in magnitude for non-tiny species,
    the gradient is O(S) — well-conditioned. The exp/log derivatives
    cancel by design.

Two pieces this module provides:

1. PINNHead — the module that does the bridge.

2. signed_log1p / signed_expm1 — the elementwise transforms used.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn


def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * log1p(|x|). Handles negatives without artefacts at zero."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def signed_expm1(y: torch.Tensor) -> torch.Tensor:
    """Inverse of signed_log1p: sign(y) * (exp(|y|) - 1)."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


class PINNHead(nn.Module):
    """Maps the GNN's per-reaction hidden state to a count update via S · v.

    Args
    ----
    hidden       : hidden dim of GNN
    stoich_matrix: (n_species, n_reactions) float tensor. Use the loader at
                    cell_sim.lgnn.data.stoichiometric_matrix.load_stoichiometric_matrix
                    to get this from a MinCell_*.lm file.
    flux_indices : indices into the GNN's node array that correspond to F_*
                    species (one per reaction). The model's hidden state at
                    these positions encodes the reaction rate; the rate head
                    decodes them into log-space v.
    rate_clip    : optional clamp on log-space rate predictions to prevent
                    expm1 blowup. Default None (no clamp).

    Forward
    -------
    x_log : (B, n_species_lgnn) — current state in signed-log1p
    h     : (B, n_species_lgnn, hidden) — GNN's per-node hidden state

    Returns
    -------
    x_next_log : (B, n_species_lgnn) — predicted next-step state in signed-log1p
    v_log      : (B, n_reactions) — predicted log-space rate vector (auxiliary
                  output for direct flux supervision against F_* observations)
    """

    def __init__(self, hidden: int, stoich_matrix: torch.Tensor,
                 flux_indices: torch.Tensor,
                 rate_clip: Optional[float] = 12.0):
        super().__init__()
        n_species, n_reactions = stoich_matrix.shape
        if flux_indices.numel() != n_reactions:
            raise ValueError(
                f'flux_indices has {flux_indices.numel()} entries but '
                f'stoich_matrix has {n_reactions} reactions'
            )
        self.n_species = n_species
        self.n_reactions = n_reactions
        self.register_buffer('S', stoich_matrix.float())
        self.register_buffer('flux_indices', flux_indices.long())
        self.rate_head = nn.Linear(hidden, 1)
        self.rate_clip = rate_clip

    def forward(self, x_log: torch.Tensor, h: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, H = h.shape
        if N != self.n_species:
            raise ValueError(
                f'hidden state has {N} nodes but stoich matrix expects '
                f'{self.n_species}'
            )

        # 1. Pull hidden state at the F_* nodes (one per reaction)
        h_rxn = h.index_select(1, self.flux_indices)            # (B, R, H)

        # 2. Decode rate in log space
        v_log = self.rate_head(h_rxn).squeeze(-1)               # (B, R)
        if self.rate_clip is not None:
            v_log = v_log.clamp(-self.rate_clip, self.rate_clip)

        # 3. ─── Log-space exponential bridge ───
        v_linear = signed_expm1(v_log)                          # (B, R)
        x_linear = signed_expm1(x_log)                          # (B, S)

        # 4. Hardwired stoichiometric update: Δx = S · v
        #    S: (S, R), v_linear: (B, R) -> dx_linear: (B, S)
        dx_linear = v_linear @ self.S.T                         # (B, S)
        x_next_linear = x_linear + dx_linear

        # 5. Bridge back to log space
        x_next_log = signed_log1p(x_next_linear)                # (B, S)

        return x_next_log, v_log


def build_flux_indices(row_names) -> torch.Tensor:
    """Return indices of F_* species in row_names, in the order they appear.

    The stoichiometric matrix is indexed by reaction; this returns the
    LGNN node index for each reaction's F_<rxn> species. Used to look up
    the GNN's hidden state at the rate-encoding nodes.

    NOTE: this requires that the order of F_* species in row_names matches
    the reaction order in the stoichiometric matrix. If they don't match,
    you'll need to rebuild via reaction_id-keyed lookup.
    """
    return torch.tensor(
        [i for i, n in enumerate(row_names) if n.startswith('F_')],
        dtype=torch.long,
    )
