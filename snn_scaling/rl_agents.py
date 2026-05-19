"""RL agents for the snn_scaling online-learning benchmarks.

Three agents:
  - NaiveQAgent           : linear Q + TD updates, eps-greedy (baseline)
  - TimeWeightedMemoryAgent : single hippocampus, recency-weighted cosine
                               k-NN retrieval of (state -> signed-reward-per-
                               action) entries
  - CLSAgent               : dual-system (hippocampus + neocortex). Hippo
                               is fast + recency-weighted; cortex is a
                               clustered Q-table updated by EMA. Decision
                               is a convex blend of the two.

All three agents are designed to be drop-in interchangeable: each has
`.act(x) -> int` and `.update(x, a, r) -> None`. The benchmark loop
treats them identically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import EpisodicMemory, MemoryConfig


# ============================================================
# Naive linear Q-learning baseline
# ============================================================

class NaiveQAgent:
    """Linear Q(s, a) = phi(s) @ W_a, updated by TD: W_a += lr * (r - Q) * phi.
    Eps-greedy action selection."""

    def __init__(self, feat_dim: int, n_actions: int,
                 lr: float = 0.05, eps: float = 0.1,
                 device: str = "cpu", seed: int = 0):
        self.W = torch.zeros(feat_dim, n_actions, device=device)
        self.lr = lr
        self.eps = eps
        self.n_actions = n_actions
        self.device = device
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def act(self, x: torch.Tensor) -> int:
        if torch.rand((), generator=self.rng).item() < self.eps:
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())
        return int((x @ self.W).argmax().item())

    def update(self, x: torch.Tensor, a: int, r: float) -> None:
        td = r - (self.W[:, a] * x).sum()
        self.W[:, a] = self.W[:, a] + self.lr * td * x


# ============================================================
# Single-bank recency-weighted memory agent
# ============================================================

class TimeWeightedMemoryAgent:
    """Episodic memory of (feature -> signed-reward-per-action).

    Every trial is stored with v[a] = r (and 0 for other slots). Retrieval is
    cosine k-NN over stored keys, with each entry's similarity multiplied by
    exp(-(t_now - age) / tau) before the top-k selection. Older entries fade
    exponentially - this is what lets the agent track non-stationarity.
    """

    def __init__(self, feat_dim: int, n_actions: int,
                 capacity: int = 1000, top_k: int = 10,
                 eps: float = 0.1, tau: float = 50.0,
                 device: str = "cpu", seed: int = 0):
        self.memory = EpisodicMemory(
            MemoryConfig(capacity=capacity, key_dim=feat_dim,
                          value_dim=n_actions, top_k=top_k,
                          similarity="cosine"),
            device=device,
        )
        self.n_actions = n_actions
        self.top_k = top_k
        self.eps = eps
        self.tau = tau
        self.device = device
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def act(self, x: torch.Tensor) -> int:
        if (self.memory.n_stored == 0
                or torch.rand((), generator=self.rng).item() < self.eps):
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())
        out = self.memory.read(x, top_k=min(self.top_k, self.memory.n_stored),
                                  recency_tau=self.tau)
        return int(out["mixed_value"].argmax().item())

    def update(self, x: torch.Tensor, a: int, r: float) -> None:
        v = torch.zeros(self.n_actions, device=self.device)
        v[a] = float(r)
        self.memory.write(x, v)
        self.memory.step_time(1.0)


# ============================================================
# Neocortex: clustered Q-table updated by EMA
# ============================================================

class CortexMemory(nn.Module):
    """Slow consolidating memory: cluster centroids discovered on the fly by
    cosine threshold; each cluster carries a Q-table (n_actions,) updated
    by EMA after each trial assigned to it.

    Contrast to hippocampus:
      - hippocampus stores RAW trials, recency-weighted, transient
      - cortex stores CONSOLIDATED Q-tables per cluster, slow EMA, durable
    """

    def __init__(self, key_dim: int, n_actions: int, max_clusters: int = 50,
                 similarity_threshold: float = 0.6, lr: float = 0.05,
                 centroid_lr: float = 0.05, device: str = "cpu"):
        super().__init__()
        device = torch.device(device)
        self.max_clusters = max_clusters
        self.threshold = similarity_threshold
        self.lr = lr
        self.centroid_lr = centroid_lr
        self.n_actions = n_actions
        self.device = device
        self.register_buffer("keys", torch.zeros(max_clusters, key_dim, device=device))
        self.register_buffer("q_tables", torch.zeros(max_clusters, n_actions, device=device))
        self.register_buffer("counts", torch.zeros(max_clusters, device=device))
        self.n_clusters = 0

    def _nearest(self, x: torch.Tensor):
        if self.n_clusters == 0:
            return -1, -1.0
        x_n = F.normalize(x.unsqueeze(0), dim=-1)
        k_n = F.normalize(self.keys[:self.n_clusters], dim=-1)
        sims = (k_n @ x_n.t()).squeeze(-1)
        return int(sims.argmax().item()), float(sims.max().item())

    @torch.no_grad()
    def update(self, x: torch.Tensor, a: int, r: float) -> int:
        idx, sim = self._nearest(x)
        if idx == -1 or (sim < self.threshold and self.n_clusters < self.max_clusters):
            slot = self.n_clusters
            self.keys[slot] = x
            self.q_tables[slot] = 0.0
            self.q_tables[slot, a] = float(r)
            self.counts[slot] = 1
            self.n_clusters += 1
            return slot
        # Slow centroid drift + Q-table EMA
        self.keys[idx] = (1 - self.centroid_lr) * self.keys[idx] + self.centroid_lr * x
        self.q_tables[idx, a] = (1 - self.lr) * self.q_tables[idx, a] + self.lr * float(r)
        self.counts[idx] += 1
        return idx

    @torch.no_grad()
    def read(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_clusters == 0:
            return torch.zeros(self.n_actions, device=self.device)
        idx, _ = self._nearest(x)
        return self.q_tables[idx].clone()


# ============================================================
# CLS agent: hippocampus + neocortex
# ============================================================

class CLSAgent:
    """Complementary Learning Systems: fast hippocampus + slow neocortex.

    Decision:
      q_blend = alpha * q_hippo + (1 - alpha) * q_cortex
      action  = argmax(q_blend)

    Update: every trial goes to hippo (raw write) and cortex (EMA on
    nearest cluster, or new cluster if no match)."""

    def __init__(self, feat_dim: int, n_actions: int,
                 hippo_capacity: int = 200, hippo_tau: float = 20.0,
                 hippo_top_k: int = 10,
                 cortex_clusters: int = 50, cortex_threshold: float = 0.6,
                 cortex_lr: float = 0.05, alpha: float = 0.7,
                 eps: float = 0.1, device: str = "cpu", seed: int = 0):
        self.hippo = EpisodicMemory(
            MemoryConfig(capacity=hippo_capacity, key_dim=feat_dim,
                          value_dim=n_actions, top_k=hippo_top_k,
                          similarity="cosine"),
            device=device,
        )
        self.cortex = CortexMemory(
            key_dim=feat_dim, n_actions=n_actions,
            max_clusters=cortex_clusters,
            similarity_threshold=cortex_threshold,
            lr=cortex_lr, centroid_lr=0.05, device=device,
        )
        self.n_actions = n_actions
        self.hippo_top_k = hippo_top_k
        self.hippo_tau = hippo_tau
        self.alpha = alpha
        self.eps = eps
        self.device = device
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def act(self, x: torch.Tensor) -> int:
        if torch.rand((), generator=self.rng).item() < self.eps:
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())
        if self.hippo.n_stored == 0 and self.cortex.n_clusters == 0:
            return int(torch.randint(0, self.n_actions, (1,), generator=self.rng).item())
        if self.hippo.n_stored > 0:
            q_hippo = self.hippo.read(
                x, top_k=min(self.hippo_top_k, self.hippo.n_stored),
                recency_tau=self.hippo_tau,
            )["mixed_value"]
        else:
            q_hippo = torch.zeros(self.n_actions, device=self.device)
        q_cortex = self.cortex.read(x)
        q_blend = self.alpha * q_hippo + (1 - self.alpha) * q_cortex
        return int(q_blend.argmax().item())

    def update(self, x: torch.Tensor, a: int, r: float) -> None:
        v = torch.zeros(self.n_actions, device=self.device)
        v[a] = float(r)
        self.hippo.write(x, v)
        self.hippo.step_time(1.0)
        self.cortex.update(x, a, r)
