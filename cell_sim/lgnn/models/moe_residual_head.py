"""M8 upgrade 7/10: Mixture-of-Experts residual head.

The PINN head handles the ~140 SBML-mapped rows via Δx = S·v. The
remaining ~8400 species rows (mRNA, protein, complexes, gene copies,
etc.) need a residual head that learns their dynamics directly.

Currently M7.* uses a single Linear(hidden, 1) for all 8400 rows. This
forces one set of weights to model very different dynamics:
  - gene copies (G_*)      : binary, slow timescale
  - mRNA (R_*)             : count, fast turnover
  - protein (P_*)          : count, slow degradation
  - protein-metab (PM_*)   : complex, allosteric
  - protein-DNA (D_*)      : binding events
  - cumulative (C_*)       : monotonic accumulators

MoE solves this by routing each row to a specialised expert.

Architecture
------------
  - K experts: each is a small Linear(hidden, 1) — 65 params each
  - Router: STATIC per-row routing based on the row name prefix.
    Each row is assigned to ONE expert at construction time.
    No router parameters; no router learning; no routing overhead at
    inference. Trivially differentiable: each row only sees its
    expert's gradient.

  - Routing: by row-prefix lookup, e.g.
        G_  -> expert 0  (gene copies)
        R_  -> expert 1  (mRNA)
        P_  -> expert 2  (protein)
        PM_ -> expert 3  (protein-metab complex)
        M_  -> expert 4  (metabolite)
        C_  -> expert 5  (cumulative)
        F_  -> expert 6  (flux - covered by PINN but residual gives bias)
        *   -> expert 7  (fallback)

Compared to "soft" MoE (top-k routing with learnable gate): static
routing is much simpler, has zero router-noise during early training,
and matches the natural row-type structure of M7's data. Loses the
ability to discover non-prefix-aligned clusters but those don't exist
in this dataset.

Parameter count: 8 experts × (hidden+1) ≈ 520 vs. single Linear's ~65,
so MoE adds ~450 params total. Negligible vs. the 3.4M-param backbone.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


# Map row prefix -> expert index. Add new entries as the SBML evolves.
DEFAULT_PREFIX_ROUTING = {
    'G_':   0,    # gene copies
    'R_':   1,    # mRNA
    'P_':   2,    # protein
    'PM_':  3,    # protein-metabolite complex
    'RP_':  4,    # ribosome-protein complex
    'M_':   5,    # metabolite
    'C_':   6,    # cumulative accumulator
    'F_':   7,    # flux (covered by PINN; residual gives bias)
    'D_':   8,    # DNA-bound complex
    'DM_':  9,    # double protein complex
    'ATP':  10,   # ATP/UTP/CTP cost trackers
    'UTP':  10,
    'CTP':  10,
    'GTP':  10,
}


def build_row_to_expert(
    row_names: Sequence[str],
    routing: Optional[dict] = None,
    n_experts: int = 11,
) -> torch.LongTensor:
    """Return a (len(row_names),) LongTensor mapping each row index to
    an expert index in [0, n_experts).

    Uses longest-prefix match against `routing`. Rows that don't match
    any prefix get assigned to the LAST expert (n_experts - 1).
    """
    if routing is None:
        routing = DEFAULT_PREFIX_ROUTING
    # Sort prefixes longest first so PM_ matches before P_, RP_ before R_, etc.
    sorted_prefixes = sorted(routing.keys(), key=lambda s: -len(s))
    fallback = n_experts - 1

    out = torch.zeros(len(row_names), dtype=torch.long)
    for i, n in enumerate(row_names):
        assigned = fallback
        for p in sorted_prefixes:
            if n.startswith(p):
                assigned = routing[p]
                break
        out[i] = assigned
    return out


class MoEResidualHead(nn.Module):
    """K-expert residual head with static per-row routing.

    Forward: given encoder hidden h of shape (B, N, hidden), produces
    a per-node residual delta of shape (B, N). Each row uses ONLY its
    pre-assigned expert (no soft mixing, no router).

    Args
    ----
    hidden         : encoder hidden dim
    n_experts      : number of experts
    row_to_expert  : LongTensor of shape (N,), maps row idx to expert idx
    delta_clip     : tanh-clip on the delta to prevent extreme updates
                      (matches PINNHead's rate_clip philosophy)
    """

    def __init__(
        self,
        hidden: int,
        n_experts: int,
        row_to_expert: torch.LongTensor,
        delta_clip: float = 0.5,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_experts = n_experts
        self.delta_clip = delta_clip
        self.experts = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in range(n_experts)
        ])
        # Small init for stable early training
        with torch.no_grad():
            for exp in self.experts:
                exp.weight.normal_(std=0.01)
                exp.bias.fill_(0.0)
        # Buffer so the routing moves with the model device
        self.register_buffer('row_to_expert', row_to_expert.long())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, hidden)  ->  delta: (B, N)

        Implementation: gather rows-per-expert, run each expert on its
        rows, scatter back. O(N) total work, no router-softmax overhead.
        """
        B, N, H = h.shape
        device = h.device
        delta = torch.zeros(B, N, device=device, dtype=h.dtype)
        # Group rows by expert (this gathering is cheap; ~K iterations)
        for k, expert in enumerate(self.experts):
            mask = (self.row_to_expert == k)
            if not mask.any():
                continue
            # h_k: (B, n_rows_k, H) -> delta_k: (B, n_rows_k)
            idx = mask.nonzero(as_tuple=True)[0]
            h_k = h[:, idx, :]                            # (B, n_k, H)
            d_k = expert(h_k).squeeze(-1)                 # (B, n_k)
            delta[:, idx] = d_k
        # Clip via tanh × delta_clip for stability
        return self.delta_clip * torch.tanh(delta)


def expert_assignment_summary(
    row_names: Sequence[str],
    row_to_expert: torch.LongTensor,
    n_experts: int,
) -> str:
    """Pretty-print expert -> row count mapping for the cfg log."""
    lines = []
    for k in range(n_experts):
        idx = (row_to_expert == k).nonzero(as_tuple=True)[0].tolist()
        if not idx:
            lines.append(f'  expert {k}: 0 rows')
            continue
        sample_names = [row_names[i] for i in idx[:3]]
        lines.append(f'  expert {k}: {len(idx)} rows '
                     f'(e.g. {", ".join(sample_names)})')
    return '\n'.join(lines)
