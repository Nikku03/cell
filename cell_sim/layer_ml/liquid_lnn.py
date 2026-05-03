"""Core Liquid Neural Net primitives — adaptive time constants + edge-aware
message passing + ODE-style integration step.

Adapted from Nikku03/enzyme_Software/src/enzyme_software/liquid_nn_v2/model/
liquid_branch.py (chemistry LNN) for gene essentiality:

  - tau bounds tuned for gene-network timescales (instead of atomic timescales)
  - edge features represent operon-membership (gap_bp, strand_match, distance)
  - message passing aggregates over a within-organism subgraph (one organism
    = one connected component) so global pooling means within-organism mean

Self-test: run `python -m cell_sim.layer_ml.liquid_lnn` to verify the three
classes forward-pass cleanly on dummy tensors.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextAwareTau(nn.Module):
    """Predict per-gene time constant tau in [tau_min, tau_max] from the
    gene's local hidden state, the per-organism global hidden mean, and
    optional retrieved memory state. Higher tau = slower integration =
    state changes more gradually (suitable for slow-responding pathways
    like cell division). Lower tau = fast integration (metabolism)."""

    def __init__(self, hidden: int, tau_min: float = 0.1, tau_max: float = 2.0,
                 dropout: float = 0.05):
        super().__init__()
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.net = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_local: torch.Tensor, h_global: torch.Tensor,
                h_memory: torch.Tensor | None = None) -> torch.Tensor:
        """h_local, h_global, h_memory: [N, H] tensors. Returns tau: [N].
        h_memory may be None (use h_local in its place)."""
        if h_memory is None:
            h_memory = h_local
        x = torch.cat([h_local, h_global, h_memory], dim=-1)
        tau_norm = self.net(x).squeeze(-1)
        return self.tau_min + (self.tau_max - self.tau_min) * tau_norm


class EdgeAwareMessagePassing(nn.Module):
    """Message passing along operon-graph edges with edge features
    (gap_bp_log, strand_match). Each gene's message is the mean over
    incoming neighbor messages, conditioned on the edge attribute.

    Implementation note: we use scatter-add then divide by in-degree
    rather than scatter-mean directly so we can handle isolated nodes
    (in-degree = 0) by treating their message as zero — this matches
    the LNN's inductive bias that genes without operon neighbors only
    receive memory + self-feedback, not graph signal."""

    def __init__(self, hidden: int, edge_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.message_net = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """h: [N, H], edge_index: [2, E], edge_attr: [E, edge_dim].
        Returns aggregated messages [N, H]."""
        if edge_index.numel() == 0 or h.size(0) == 0:
            return torch.zeros_like(h)
        src, dst = edge_index[0], edge_index[1]
        e = self.edge_proj(edge_attr)
        m = self.message_net(torch.cat([h[src], h[dst], e], dim=-1))
        out = torch.zeros_like(h)
        out.index_add_(0, dst, m)
        # In-degree per node, clamped to 1 to avoid div-by-zero on isolated nodes
        deg = torch.bincount(dst, minlength=h.size(0)).clamp(min=1).to(h.dtype)
        return out / deg.unsqueeze(-1)


class LiquidStep(nn.Module):
    """One ODE-style integration step:
        tau   = ContextAwareTau(h_local, h_global, h_memory)
        msg   = EdgeAwareMessagePassing(h, edges, edge_attr)
        dh/dt = (msg + h0 + h_memory - h) / tau
        h    += dt * dh
    Wraps the two primitives + the integration logic. Pulled out as its
    own class so the EssentialityLNN can stack n_steps of them and so
    we can unit-test the integration logic in isolation."""

    def __init__(self, hidden: int, edge_dim: int = 2, dt: float = 0.2,
                 tau_min: float = 0.1, tau_max: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        self.tau_pred = ContextAwareTau(hidden, tau_min=tau_min, tau_max=tau_max)
        self.passing = EdgeAwareMessagePassing(hidden, edge_dim=edge_dim,
                                                dropout=dropout)
        self.dt = float(dt)

    def forward(self, h: torch.Tensor, h0: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: torch.Tensor,
                h_memory: torch.Tensor | None = None) -> torch.Tensor:
        """One liquid step. Returns updated h: [N, H]."""
        h_global = h.mean(dim=0, keepdim=True).expand_as(h)
        tau = self.tau_pred(h, h_global, h_memory)
        msg = self.passing(h, edge_index, edge_attr)
        if h_memory is None:
            h_memory = torch.zeros_like(h)
        dh = (msg + h_memory + h0 - h) / tau.unsqueeze(-1)
        return h + self.dt * dh


# ─────────────── self-test ───────────────


def _self_test():
    """Verify forward pass on dummy data."""
    print('liquid_lnn self-test:')
    torch.manual_seed(42)
    N, H, E = 20, 32, 30  # 20 genes, 32 hidden dim, 30 edges
    h = torch.randn(N, H)
    h0 = torch.randn(N, H)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, 2)

    tau_pred = ContextAwareTau(H, tau_min=0.1, tau_max=2.0)
    passing = EdgeAwareMessagePassing(H, edge_dim=2)
    step = LiquidStep(H, edge_dim=2)

    h_global = h.mean(dim=0, keepdim=True).expand_as(h)
    tau = tau_pred(h, h_global, None)
    print(f'  ContextAwareTau output: shape={tuple(tau.shape)}, '
          f'min={tau.min().item():.3f}, max={tau.max().item():.3f}')
    assert tau.shape == (N,)
    assert tau.min().item() >= 0.1 and tau.max().item() <= 2.0

    msg = passing(h, edge_index, edge_attr)
    print(f'  EdgeAwareMessagePassing output: shape={tuple(msg.shape)}, '
          f'mean_norm={msg.norm(dim=-1).mean().item():.3f}')
    assert msg.shape == (N, H)

    h_new = step(h, h0, edge_index, edge_attr, h_memory=None)
    print(f'  LiquidStep output: shape={tuple(h_new.shape)}, '
          f'delta_norm={(h_new - h).norm(dim=-1).mean().item():.3f}')
    assert h_new.shape == (N, H)

    # Edge-case: empty edges
    msg_empty = passing(h, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 2))
    assert msg_empty.shape == (N, H) and msg_empty.abs().sum().item() == 0
    print('  empty-edges edge case: passes (zero message)')

    # Edge-case: with memory
    h_mem = torch.randn(N, H)
    h_with_mem = step(h, h0, edge_index, edge_attr, h_memory=h_mem)
    assert h_with_mem.shape == (N, H)
    assert not torch.allclose(h_with_mem, h_new)  # memory changes output
    print('  with-memory step: changes output as expected')

    print('  ALL PASS')


if __name__ == '__main__':
    _self_test()
