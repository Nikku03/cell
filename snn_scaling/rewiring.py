"""Trick 10 / SparseSpike-GEMM steal #6: DEEP-R structural plasticity.

Biological synapses are not fixed: they physically form, strengthen,
weaken and disappear over a lifetime. DEEP-R (Bellec et al. 2018) is the
machine-learning form of this - "deep rewiring" - and the spec calls out
its GPU-tractable trick:

  maintain an OVER-PROVISIONED index buffer so you never reallocate the
  sparse data structure mid-run - you just flip which entries are 'live'.

`RewiringSynapses` implements exactly that. `capacity` edge slots are
pre-allocated once; at most `n_live <= capacity` are live at any moment.
Rewiring prunes the weakest live synapses and grows fresh ones into dead
slots - O(swaps) flag/index updates, no tensor reallocation, no memory
fragmentation. The connectivity stays sparse while the SET of
connections evolves, so the network can re-route around damage.

Dead slots are held inert by zeroing their conductance (g_max = 0,
g = 0), so the standard scatter-add forward pass needs no masking - dead
edges simply contribute nothing.

The class mirrors SparseSynapses' interface (post_currents /
deliver_spikes / decay / reset_state) so it is drop-in compatible with
SNNCluster, and adds prune / grow / rewire / lesion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .population import SYN_EXC, _E_SYN


class RewiringSynapses(nn.Module):
    """Sparse excitatory synapses with a DEEP-R over-provisioned edge buffer.

    All edge tensors have a fixed length `capacity`. A `live` boolean mask
    marks which slots are real synapses; dead slots carry g_max = 0 so
    they are inert in the forward pass. Rewiring never changes any tensor
    size - it only toggles `live` and overwrites dead slots.
    """

    def __init__(self, n_pre: int, n_post: int, capacity: int,
                 n_initial: int, g_max_range: tuple[float, float] = (0.02, 0.10),
                 tau_syn: float = 5.0, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32, seed: int = 0):
        super().__init__()
        device = torch.device(device)
        if not (0 <= n_initial <= capacity):
            raise ValueError(f"n_initial must be in 0..capacity ({capacity})")
        self.n_pre = n_pre
        self.n_post = n_post
        self.capacity = capacity
        self.device = device
        self.dtype = dtype
        self.g_max_range = (float(g_max_range[0]), float(g_max_range[1]))
        self.tau_value = float(tau_syn)

        g = torch.Generator(device="cpu").manual_seed(seed)
        edge_pre = torch.zeros(capacity, dtype=torch.long)
        edge_post = torch.zeros(capacity, dtype=torch.long)
        g_max = torch.zeros(capacity, dtype=dtype)
        live = torch.zeros(capacity, dtype=torch.bool)
        # Seed the first n_initial slots with random live excitatory edges.
        if n_initial > 0:
            edge_pre[:n_initial] = torch.randint(0, n_pre, (n_initial,), generator=g)
            edge_post[:n_initial] = torch.randint(0, n_post, (n_initial,), generator=g)
            lo, hi = self.g_max_range
            g_max[:n_initial] = lo + (hi - lo) * torch.rand(n_initial, generator=g)
            live[:n_initial] = True

        self.register_buffer("edge_pre", edge_pre.to(device=device))
        self.register_buffer("edge_post", edge_post.to(device=device))
        self.register_buffer("g_max", g_max.to(device=device))
        self.register_buffer("live", live.to(device=device))
        self.register_buffer("g", torch.zeros(capacity, device=device, dtype=dtype))
        self.register_buffer(
            "tau_syn", torch.full((capacity,), self.tau_value, device=device, dtype=dtype))
        self.register_buffer(
            "E_syn", torch.full((capacity,), _E_SYN[SYN_EXC], device=device, dtype=dtype))

    # ---- structural-plasticity operations ----

    @property
    def n_live(self) -> int:
        return int(self.live.sum().item())

    @torch.no_grad()
    def _kill(self, slots: torch.Tensor) -> None:
        """Make the given slots dead + inert (zero conductance)."""
        self.live[slots] = False
        self.g_max[slots] = 0.0
        self.g[slots] = 0.0

    @torch.no_grad()
    def prune(self, n: int) -> int:
        """Prune the n WEAKEST live synapses (smallest g_max). Returns the
        number actually pruned."""
        live_slots = self.live.nonzero(as_tuple=False).squeeze(-1)
        if live_slots.numel() == 0 or n <= 0:
            return 0
        n = min(n, int(live_slots.numel()))
        order = self.g_max[live_slots].argsort()        # ascending strength
        self._kill(live_slots[order[:n]])
        return n

    @torch.no_grad()
    def lesion(self, n: int, generator: torch.Generator) -> int:
        """Prune n RANDOM live synapses (models damage, not weakest-first).
        Returns the number actually pruned."""
        live_slots = self.live.nonzero(as_tuple=False).squeeze(-1)
        if live_slots.numel() == 0 or n <= 0:
            return 0
        n = min(n, int(live_slots.numel()))
        pick = torch.randperm(int(live_slots.numel()), generator=generator)[:n]
        self._kill(live_slots[pick])
        return n

    @torch.no_grad()
    def grow(self, n: int, generator: torch.Generator) -> int:
        """Grow n fresh random synapses into dead slots. Returns the number
        actually grown (capped by the number of dead slots)."""
        dead_slots = (~self.live).nonzero(as_tuple=False).squeeze(-1)
        if dead_slots.numel() == 0 or n <= 0:
            return 0
        n = min(n, int(dead_slots.numel()))
        pick = dead_slots[torch.randperm(int(dead_slots.numel()), generator=generator)[:n]]
        self.edge_pre[pick] = torch.randint(0, self.n_pre, (n,), generator=generator).to(self.device)
        self.edge_post[pick] = torch.randint(0, self.n_post, (n,), generator=generator).to(self.device)
        lo, hi = self.g_max_range
        self.g_max[pick] = (lo + (hi - lo) * torch.rand(n, generator=generator)).to(
            device=self.device, dtype=self.dtype)
        self.g[pick] = 0.0
        self.live[pick] = True
        return n

    @torch.no_grad()
    def rewire(self, n: int, generator: torch.Generator) -> int:
        """One rewiring round: prune the n weakest live synapses, then grow
        n fresh ones. The live-edge count is preserved (turnover). Returns
        the number of synapses swapped."""
        pruned = self.prune(n)
        self.grow(pruned, generator)
        return pruned

    # ---- forward-pass interface (mirrors SparseSynapses) ----

    @torch.no_grad()
    def reset_state(self) -> None:
        """Reset electrical state (conductance). Structure is preserved -
        rewiring is structural, reset is electrical."""
        self.g.zero_()

    @torch.no_grad()
    def decay(self, dt: float) -> None:
        self.g.mul_(torch.exp(-dt / self.tau_syn))

    @torch.no_grad()
    def deliver_spikes(self, pre_spiked: torch.Tensor) -> None:
        """Add g_max to each edge whose pre-neuron fired. Dead slots have
        g_max = 0 so they stay inert automatically."""
        fire_e = pre_spiked[self.edge_pre]
        self.g.add_(self.g_max * fire_e.to(self.dtype))

    @torch.no_grad()
    def post_currents(self, V_post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate (I_syn, g_syn) per post-neuron via scatter-add. Dead
        slots have g = 0 so they contribute nothing."""
        V_at_post = V_post.index_select(0, self.edge_post)
        I_per_edge = self.g * (self.E_syn - V_at_post)
        I_per_post = torch.zeros(self.n_post, device=self.device, dtype=self.dtype)
        g_per_post = torch.zeros(self.n_post, device=self.device, dtype=self.dtype)
        I_per_post.index_add_(0, self.edge_post, I_per_edge)
        g_per_post.index_add_(0, self.edge_post, self.g)
        return I_per_post, g_per_post
