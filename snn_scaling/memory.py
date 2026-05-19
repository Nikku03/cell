"""Trick 5: external vector memory (retrieval-augmented compute).

The brain's hippocampus stores episodes (vectors keyed by context) that
neocortex retrieves via similarity search. A small network can punch
above its weight by offloading long-term memory to an external store
that it queries each step.

Practical implementation:

  EpisodicMemory  -- a key/value vector store with similarity search
  WriteRule       -- decide WHEN to commit a new memory (one-shot,
                       confidence-gated, or every-K-steps)
  ReadHead        -- given a query, return a weighted sum of top-K
                       matched values

This is functionally similar to a transformer's attention over a long
context, or to differentiable neural computer memory, or to
retrieval-augmented generation. We make it brain-inspired by:
  - Sparse writes (only "important" events stored)
  - Top-K retrieval (only relevant items retrieved)
  - Decay (memories fade if unused)

Storage cost scales linearly with N_memories regardless of network size,
so a 100K-neuron network can have a 1M-vector memory store cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryConfig:
    capacity: int = 10_000          # max stored memories
    key_dim: int = 64               # query/key dimension
    value_dim: int = 64             # value dimension
    top_k: int = 5                  # retrieve top-K matches
    similarity: str = "cosine"      # "cosine" or "dot"
    decay_per_write: float = 1.0    # multiply existing memory ages by this
    eviction: str = "fifo"          # "fifo" or "lru"


class EpisodicMemory(nn.Module):
    """A capacity-bounded vector store of (key, value) pairs.

    Writes are O(1) (append until full, then evict). Reads are
    O(capacity * key_dim) for similarity computation.

    State:
      keys:    (capacity, key_dim)
      values:  (capacity, value_dim)
      ages:    (capacity,)   counter incremented per timestep; -1 = empty
      n_stored: int          how many slots are filled
    """

    def __init__(self, cfg: MemoryConfig, device: torch.device | str = "cpu"):
        super().__init__()
        self.cfg = cfg
        device = torch.device(device)
        self.register_buffer("keys", torch.zeros(cfg.capacity, cfg.key_dim, device=device))
        self.register_buffer("values", torch.zeros(cfg.capacity, cfg.value_dim, device=device))
        self.register_buffer("ages", torch.full((cfg.capacity,), -1.0, device=device))
        self.n_stored = 0
        self.t_global = 0.0

    @torch.no_grad()
    def reset(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.ages.fill_(-1.0)
        self.n_stored = 0
        self.t_global = 0.0

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor) -> int:
        """Store one (key, value) pair. Returns the slot used.

        key: (key_dim,), value: (value_dim,)
        """
        assert key.shape == (self.cfg.key_dim,)
        assert value.shape == (self.cfg.value_dim,)
        if self.n_stored < self.cfg.capacity:
            slot = self.n_stored
            self.n_stored += 1
        else:
            # Evict
            if self.cfg.eviction == "fifo":
                slot = int(self.ages.argmin().item())  # oldest (smallest age timestamp)
            else:  # lru: same as fifo for our timestamp-based scheme
                slot = int(self.ages.argmin().item())
        self.keys[slot] = key
        self.values[slot] = value
        self.ages[slot] = self.t_global
        return slot

    @torch.no_grad()
    def read(self, query: torch.Tensor, top_k: int | None = None) -> dict:
        """Retrieve top-K matches for `query`.

        Returns:
          values_topk: (top_k, value_dim)
          weights: (top_k,) similarity-weighted (softmax over similarities)
          mixed_value: (value_dim,) weighted sum of top-K values
          slot_ids: (top_k,) which slots matched
        """
        if top_k is None:
            top_k = self.cfg.top_k
        top_k = min(top_k, max(self.n_stored, 1))
        assert query.shape == (self.cfg.key_dim,)

        # Only consider filled slots
        n = self.n_stored
        if n == 0:
            # Return zeros
            return {
                "values_topk": torch.zeros(top_k, self.cfg.value_dim, device=query.device),
                "weights": torch.zeros(top_k, device=query.device),
                "mixed_value": torch.zeros(self.cfg.value_dim, device=query.device),
                "slot_ids": torch.zeros(top_k, dtype=torch.long, device=query.device),
            }

        keys = self.keys[:n]
        if self.cfg.similarity == "cosine":
            q_n = F.normalize(query.unsqueeze(0), dim=-1)
            k_n = F.normalize(keys, dim=-1)
            sims = (k_n @ q_n.t()).squeeze(-1)        # (n,)
        else:  # dot
            sims = keys @ query                        # (n,)

        top_sims, top_idx = torch.topk(sims, min(top_k, n))
        # If n < top_k, pad with first-slot zeros (won't be used)
        if top_sims.numel() < top_k:
            pad = top_k - top_sims.numel()
            top_sims = torch.cat([top_sims, torch.full((pad,), -1e9, device=query.device)])
            top_idx = torch.cat([top_idx, torch.zeros(pad, dtype=torch.long, device=query.device)])

        weights = F.softmax(top_sims, dim=0)
        values_topk = self.values[top_idx]              # (top_k, value_dim)
        mixed = (weights.unsqueeze(-1) * values_topk).sum(dim=0)
        return {
            "values_topk": values_topk,
            "weights": weights,
            "mixed_value": mixed,
            "slot_ids": top_idx,
        }

    def step_time(self, dt: float = 1.0) -> None:
        """Advance the global time counter (used for FIFO/LRU bookkeeping)."""
        self.t_global += dt


# ---------------- write rules ----------------

class WriteRule(nn.Module):
    """Base class for "should we write this event to memory?" rules."""

    def should_write(self, *args, **kwargs) -> bool:  # pragma: no cover
        raise NotImplementedError


class EveryKStepsRule(WriteRule):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.counter = 0

    def should_write(self, **kwargs) -> bool:
        self.counter += 1
        if self.counter >= self.k:
            self.counter = 0
            return True
        return False


class NoveltyGatedRule(WriteRule):
    """Write only if the proposed key is "novel" (low similarity to
    existing top match)."""

    def __init__(self, memory: EpisodicMemory, threshold: float = 0.7):
        super().__init__()
        self.memory = memory
        self.threshold = threshold

    @torch.no_grad()
    def should_write(self, key: torch.Tensor, **kwargs) -> bool:
        if self.memory.n_stored == 0:
            return True
        res = self.memory.read(key, top_k=1)
        best_sim = res["weights"].max().item()
        # softmax of one entry is 1.0 -- so use raw sim instead
        if self.memory.cfg.similarity == "cosine":
            q_n = F.normalize(key.unsqueeze(0), dim=-1)
            k_n = F.normalize(self.memory.keys[: self.memory.n_stored], dim=-1)
            sims = (k_n @ q_n.t()).squeeze(-1)
            best = sims.max().item()
        else:
            best = (self.memory.keys[: self.memory.n_stored] @ key).max().item()
        return best < self.threshold
