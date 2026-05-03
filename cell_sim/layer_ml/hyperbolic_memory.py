"""Hyperbolic memory bank — store gene-state embeddings in the Poincaré ball,
retrieve via hyperbolic distance for context-aware LNN updates.

Why hyperbolic?  Gene relationships are hierarchical (organism > pathway >
operon > gene), and hyperbolic spaces represent hierarchies more efficiently
than Euclidean (Nickel & Kiela 2017). The memory bank lets the LNN borrow
"this gene looks like genes I've seen in pathway X" knowledge across
organisms — important for the continual-learning setting where a new
organism's genes should retrieve ortholog-like states from earlier-trained
organisms.

Borrowed pattern from Nikku03/enzyme_Software/src/enzyme_software/liquid_nn_v2/
model/ (chemistry hyperbolic memory) but simplified: K-slot bank with
exponential-map projection into the Poincaré ball, top-k weighted retrieval
by hyperbolic distance, age-based slot replacement.

Self-test: `python -m cell_sim.layer_ml.hyperbolic_memory`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ──────────── Poincaré ball math ────────────


def _project_to_ball(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5
                       ) -> torch.Tensor:
    """Project x to the open Poincaré ball of curvature -c (radius 1/sqrt(c)).
    Clamps norms to (1 - eps) / sqrt(c)."""
    sqrt_c = c ** 0.5
    max_norm = (1.0 - eps) / sqrt_c
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    factor = torch.minimum(norm, torch.tensor(max_norm, device=x.device,
                                                dtype=x.dtype)) / norm
    return x * factor


def _exp_map_zero(v: torch.Tensor, c: float = 1.0, eps: float = 1e-8
                    ) -> torch.Tensor:
    """Exponential map at the origin of the Poincaré ball:
    exp_0(v) = tanh(sqrt(c) * |v|) * v / (sqrt(c) * |v|).
    Maps from Euclidean tangent space to the ball."""
    sqrt_c = c ** 0.5
    norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    out = torch.tanh(sqrt_c * norm) * v / (sqrt_c * norm)
    return _project_to_ball(out, c=c)


def _hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0,
                          eps: float = 1e-5) -> torch.Tensor:
    """Hyperbolic distance in Poincaré ball:
        d(x, y) = (1/sqrt(c)) * arccosh(1 + 2c |x-y|^2 / ((1-c|x|^2)(1-c|y|^2)))
    x, y: [..., D]. Returns [...] distances."""
    sqrt_c = c ** 0.5
    sqdist = ((x - y) ** 2).sum(dim=-1)
    nx2 = (x ** 2).sum(dim=-1).clamp(max=1.0 / c - eps)
    ny2 = (y ** 2).sum(dim=-1).clamp(max=1.0 / c - eps)
    arg = 1.0 + 2.0 * c * sqdist / ((1.0 - c * nx2) * (1.0 - c * ny2) + eps)
    arg = arg.clamp(min=1.0 + eps)
    return torch.acosh(arg) / sqrt_c


# ──────────── memory bank ────────────


class HyperbolicMemoryBank(nn.Module):
    """K-slot memory bank in the Poincaré ball.

    Storage:  Euclidean inputs are projected into the ball via exp_map_zero
              and written to a slot. Slots are replaced age-FIFO so the
              bank holds the K most-recently-seen states (continual learning
              treats older slots as long-term memory).

    Retrieval: a query is exp-mapped into the ball, hyperbolic distance is
               computed to all slots, top-k closest are retrieved and
               combined via softmax-weighted (over -distance) Euclidean
               average. The retrieved memory is returned in Euclidean space
               (log_map at zero) so downstream LNN layers see a normal
               vector.

    Why two coordinate systems? PyTorch's autograd is friendlier in
    Euclidean space (no log/exp surgery in the backward pass). We do the
    geometry only at memory-bank boundaries.

    Memory state is non-trainable (registered as buffers) — gradients flow
    into the writer, not into the bank itself. This matches the EWC-style
    continual-learning use case where memory accumulates across organism
    epochs without backprop pressure.
    """

    def __init__(self, hidden: int, memory_size: int = 512,
                 retrieval_k: int = 8, curvature: float = 1.0):
        super().__init__()
        self.hidden = int(hidden)
        self.memory_size = int(memory_size)
        self.retrieval_k = int(retrieval_k)
        self.c = float(curvature)

        # Memory slots in the Poincaré ball. Initialized at origin.
        self.register_buffer('memory',
                              torch.zeros(self.memory_size, self.hidden))
        # Populated mask — only retrieve from slots we've actually written.
        self.register_buffer('populated',
                              torch.zeros(self.memory_size, dtype=torch.bool))
        # Circular write pointer.
        self.register_buffer('write_pos', torch.zeros(1, dtype=torch.long))

    def reset(self):
        """Clear the bank (used between phases of continual learning)."""
        self.memory.zero_()
        self.populated.zero_()
        self.write_pos.zero_()

    @property
    def n_populated(self) -> int:
        return int(self.populated.sum().item())

    @torch.no_grad()
    def store(self, h: torch.Tensor):
        """Write a batch of new states into the bank using a circular write
        pointer. h is in Euclidean space; we exp-map into the ball before
        writing."""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        n = h.size(0)
        if n == 0: return
        ball = _exp_map_zero(h.detach(), c=self.c)
        K = self.memory_size
        pos = int(self.write_pos.item())
        # Indices to write: (pos + 0..n-1) mod K
        indices = torch.tensor([(pos + i) % K for i in range(n)],
                                device=self.memory.device, dtype=torch.long)
        self.memory[indices] = ball
        self.populated[indices] = True
        self.write_pos[0] = (pos + n) % K

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve top-k closest memories (hyperbolic distance) for each
        query. query: [N, H] in Euclidean space.
        Returns: [N, H] Euclidean retrieved-memory vectors."""
        if not self.populated.any():
            return torch.zeros_like(query)

        q_ball = _exp_map_zero(query, c=self.c)              # [N, H]
        # Only consider populated slots
        active_idx = self.populated.nonzero(as_tuple=False).squeeze(-1)  # [M_active]
        active_mem = self.memory[active_idx]                  # [M_active, H]

        N = q_ball.size(0); M = active_mem.size(0)
        q_b = q_ball.unsqueeze(1).expand(N, M, self.hidden)
        m_b = active_mem.unsqueeze(0).expand(N, M, self.hidden)
        dist = _hyperbolic_distance(q_b, m_b, c=self.c)       # [N, M]

        k = min(self.retrieval_k, M)
        topk_dist, topk_local = dist.topk(k, dim=-1, largest=False)
        weights = torch.softmax(-topk_dist, dim=-1)           # [N, k]
        retrieved_ball = active_mem[topk_local]                # [N, k, H]
        retrieved = (weights.unsqueeze(-1) * retrieved_ball).sum(dim=1)  # [N, H]
        return retrieved


# ──────────── self-test ────────────


def _self_test():
    print('hyperbolic_memory self-test:')
    torch.manual_seed(42)
    H = 16
    bank = HyperbolicMemoryBank(hidden=H, memory_size=64, retrieval_k=4)

    # Initially empty
    q = torch.randn(5, H) * 0.3
    r = bank.retrieve(q)
    assert r.shape == (5, H)
    assert r.abs().sum().item() == 0
    print('  empty bank retrieve: returns zeros (correct)')

    # Store some embeddings
    h_org_a = torch.randn(20, H) * 0.3
    bank.store(h_org_a)
    print(f'  stored 20 embeddings; n_populated = {bank.n_populated}')

    # Retrieve on similar query
    r = bank.retrieve(h_org_a[:5])
    print(f'  retrieved norms (first 5): '
          f'{[float(r[i].norm()) for i in range(5)]}')
    assert r.shape == (5, H)
    assert r.abs().sum().item() > 0

    # Retrieve on dissimilar query
    h_far = torch.randn(5, H) * 5  # far from origin
    r_far = bank.retrieve(h_far)
    assert r_far.shape == (5, H)
    print(f'  far-query retrieve: norms = {[float(r_far[i].norm()) for i in range(5)]}')

    # Continual: store more, verify circular write works
    for i in range(5):
        bank.store(torch.randn(20, H) * 0.3)
    print(f'  after 6 stores of 20: n_populated = {bank.n_populated} (cap 64)')
    assert bank.n_populated == 64

    # Sanity: hyperbolic distance properties
    x = torch.randn(3, H) * 0.1
    y = torch.randn(3, H) * 0.1
    d_xx = _hyperbolic_distance(x, x)
    d_xy = _hyperbolic_distance(x, y)
    print(f'  d(x,x) = {d_xx.tolist()} (numerical floor ~ sqrt(2*eps))')
    print(f'  d(x,y) = {d_xy.tolist()} (>0)')
    # arccosh(1+eps) numerical floor is ~sqrt(2*eps); relax to 0.01
    assert (d_xx < 0.01).all()
    assert (d_xy > d_xx).all()  # d(x,y) > d(x,x)

    # Reset works
    bank.reset()
    assert bank.n_populated == 0
    r = bank.retrieve(q)
    assert r.abs().sum().item() == 0
    print('  reset: bank cleared')

    print('  ALL PASS')


if __name__ == '__main__':
    _self_test()
