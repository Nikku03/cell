"""Trick 4: modular architecture with task routing.

Different parts of a network handle different input types. The brain's
cortical areas are specialised (V1 for visual, A1 for auditory, M1 for
motor, etc.) and communicate via narrow bottlenecks rather than dense
all-to-all.

A modular N-neuron network has N/k neurons per module times k modules.
Each module is active only when its modality is being processed. Total
effective compute scales with the number of modules visited per task,
without the cross-module quadratic synapse cost.

This module provides:
  - Module: one self-contained SNNCluster + IO ports
  - ModularNetwork: a collection of modules + routing edges
  - TaskRouter: picks which modules are active for an input
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .population import (
    LIFPopulation,
    NeuronParams,
    SparseSynapses,
    SNNCluster,
    SYN_EXC,
    SYN_INH,
    build_ei_microcircuit,
)


@dataclass
class ModuleSpec:
    name: str
    n_e: int
    n_i: int
    role: str = "general"          # "sensory", "motor", "integration", etc.
    seed: int = 0


class Module(nn.Module):
    """One self-contained SNN microcircuit acting as a brain area."""

    def __init__(self, spec: ModuleSpec, device: torch.device | str = "cpu"):
        super().__init__()
        self.spec = spec
        self.device = torch.device(device)
        # Build a standard E/I microcircuit
        self.cluster = build_ei_microcircuit(
            n_e=spec.n_e, n_i=spec.n_i, seed=spec.seed,
            device=device, degree_mode=False,
        )
        # Tracks whether this module is "active" this step (for sparse modular gating)
        self.active = True

    @property
    def n(self) -> int:
        return self.cluster.population.n

    def reset(self) -> None:
        self.cluster.reset_state()
        self.active = True


# ---------------- modular network ----------------

class ModularNetwork(nn.Module):
    """A set of Modules connected by inter-module synapses.

    Each module is a separate microcircuit; inter-module connections
    are SPARSE (a fraction of E neurons in source project to a small
    fraction of E in target).

    Active modules: per step, the router decides which modules are
    "on". Inactive modules' membrane dynamics still tick (so they
    decay back to V_rest) but they receive no external input and
    their spikes don't propagate.
    """

    def __init__(self, modules: list[Module], device: torch.device | str = "cpu"):
        super().__init__()
        self.modules_list = modules
        self.n_modules = len(modules)
        self.device = torch.device(device)
        # Inter-module synapse banks: dict ("src_name", "dst_name") -> SparseSynapses
        self.inter_syns: dict[tuple[str, str], SparseSynapses] = nn.ModuleDict()  # type: ignore

    @property
    def total_n(self) -> int:
        return sum(m.n for m in self.modules_list)

    def add_intermodule_projection(
        self,
        src_module: Module,
        dst_module: Module,
        n_src: int = 30,
        n_dst: int = 30,
        n_edges: int = 100,
        g_max: float = 0.2,
        kind: int = SYN_EXC,
        tau_syn: float = 5.0,
        seed: int = 0,
    ) -> None:
        """Random sparse projection from n_src of src's E neurons to n_dst of dst's.

        Builds a small SparseSynapses with n_pre = src_module.n,
        n_post = dst_module.n. Stored under the (src.name, dst.name) key.
        """
        g = torch.Generator(device="cpu").manual_seed(seed)
        src_n_e = src_module.spec.n_e
        dst_n_e = dst_module.spec.n_e
        src_chosen = torch.randperm(src_n_e, generator=g)[:n_src]
        dst_chosen = torch.randperm(dst_n_e, generator=g)[:n_dst]
        # Random pairings
        pre = src_chosen[torch.randint(0, n_src, (n_edges,), generator=g)]
        post = dst_chosen[torch.randint(0, n_dst, (n_edges,), generator=g)]
        kinds = torch.full((n_edges,), kind, dtype=torch.long)
        gmax = torch.full((n_edges,), float(g_max))
        tau = torch.full((n_edges,), float(tau_syn))
        syn = SparseSynapses(
            n_pre=src_module.n, n_post=dst_module.n,
            edge_pre=pre, edge_post=post,
            kind=kinds, g_max=gmax, tau_syn=tau,
            device=self.device,
        )
        key = f"{src_module.spec.name}__{dst_module.spec.name}"
        self.inter_syns[key] = syn

    def reset(self) -> None:
        for m in self.modules_list:
            m.reset()
        for s in self.inter_syns.values():
            s.reset_state()

    @torch.no_grad()
    def step(
        self,
        dt: float,
        t: float,
        external_currents: dict[str, torch.Tensor] | None = None,
        active_modules: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """One step across all modules.

        external_currents: {module_name: (n_module,) drive tensor}
        active_modules: list of names; if None, all modules active

        Returns: {module_name: (n_module,) spike mask}
        """
        if active_modules is None:
            active_modules = [m.spec.name for m in self.modules_list]
        active_set = set(active_modules)

        # Step 1: aggregate intra-module + inter-module currents for each module
        all_modules = {m.spec.name: m for m in self.modules_list}
        for m in self.modules_list:
            m.active = m.spec.name in active_set

        # Compute spikes for each module based on its current state, using
        # the post-currents from its OWN synapses + inter-module banks.
        spike_outputs: dict[str, torch.Tensor] = {}

        for m in self.modules_list:
            if not m.active:
                # Inactive: no external drive, no spike propagation, but dynamics still tick
                I_total = torch.zeros(m.n, device=self.device)
                g_total = torch.zeros(m.n, device=self.device)
                # Intra-module synapses still produce decaying currents
                V = m.cluster.population.V
                for s in m.cluster.synapses:
                    I, g = s.post_currents(V)
                    I_total = I_total + I
                    g_total = g_total + g
                m.cluster.population.step(I_total, g_total, dt=dt, t=t)
                for s in m.cluster.synapses:
                    s.decay(dt)
                spike_outputs[m.spec.name] = torch.zeros(m.n, dtype=torch.bool, device=self.device)
                continue

            # Active module: external + intra + inter
            I_total = torch.zeros(m.n, device=self.device)
            g_total = torch.zeros(m.n, device=self.device)
            if external_currents and m.spec.name in external_currents:
                I_total = I_total + external_currents[m.spec.name].to(self.device)

            V = m.cluster.population.V
            for s in m.cluster.synapses:
                I, g = s.post_currents(V)
                I_total = I_total + I
                g_total = g_total + g

            # Inter-module incoming
            for (src_dst_key, syn) in self.inter_syns.items():
                src_name, dst_name = src_dst_key.split("__")
                if dst_name != m.spec.name:
                    continue
                I_inter, g_inter = syn.post_currents(V)
                I_total = I_total + I_inter
                g_total = g_total + g_inter

            spike = m.cluster.population.step(I_total, g_total, dt=dt, t=t)
            spike_outputs[m.spec.name] = spike

            # Decay intra-module synapses
            for s in m.cluster.synapses:
                s.decay(dt)

        # Step 2: deliver spikes to intra and inter-module synapses
        for m in self.modules_list:
            sp = spike_outputs[m.spec.name]
            for s in m.cluster.synapses:
                s.deliver_spikes(sp)
            # Inter-module: deliver this module's spikes to outgoing inter-syns
            for (key, syn) in self.inter_syns.items():
                src_name, dst_name = key.split("__")
                if src_name != m.spec.name:
                    continue
                syn.deliver_spikes(sp)
            # Inactive modules: their spike output is already zero, no delivery needed

        # Decay inter-module synapses
        for s in self.inter_syns.values():
            s.decay(dt)

        return spike_outputs


# ---------------- task router ----------------

class TaskRouter:
    """Decide which modules are active given an input descriptor.

    Routes can be:
      - Static (hard-coded mapping from task_type to module set)
      - Learned (a small classifier that predicts the right module set)

    This implementation is static: caller provides the active modules.
    """

    def __init__(self, mapping: dict[str, list[str]]):
        """mapping: task_type -> list of module names to activate."""
        self.mapping = mapping

    def route(self, task_type: str) -> list[str]:
        if task_type not in self.mapping:
            raise KeyError(f"unknown task_type: {task_type}")
        return self.mapping[task_type]


# ---------------- learned MoE-style router ----------------

class LearnedRouter(nn.Module):
    """Mixture-of-Experts top-k gate over modules (SparseSpike-GEMM steal).

    The static TaskRouter maps a hard-coded task_type STRING to a module
    set - it cannot route an input it has not been given an exact key
    for. The LearnedRouter instead scores every module from an input
    FEATURE VECTOR via a trained linear gate and activates the top-k
    highest scorers, so it generalises to unseen inputs by learned
    similarity. This is the spec's "MoE routing as fan-out": of N expert
    modules, only top-k are active per input, which both bounds compute
    and plays the role of sparse cortico-cortical projections.

    route()/route_one() return module-name lists directly usable as the
    `active_modules` argument of ModularNetwork.step().
    """

    def __init__(self, input_dim: int, module_names: list[str],
                 top_k: int, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.module_names = list(module_names)
        self.n_modules = len(self.module_names)
        if not (0 < top_k <= self.n_modules):
            raise ValueError(f"top_k must be in 1..{self.n_modules}, got {top_k}")
        self.top_k = top_k
        self.input_dim = input_dim
        self.gate = nn.Linear(input_dim, self.n_modules)

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """x: (input_dim,) or (B, input_dim) -> gate scores (B, n_modules)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.gate(x)

    @torch.no_grad()
    def route(self, x: torch.Tensor) -> list[list[str]]:
        """Return, per input row, the list of top-k module names."""
        idx = self.scores(x).topk(self.top_k, dim=-1).indices
        return [[self.module_names[i] for i in row.tolist()] for row in idx]

    @torch.no_grad()
    def route_one(self, x: torch.Tensor) -> list[str]:
        """Single input feature vector -> top-k module names."""
        return self.route(x)[0]


def train_learned_router(
    router: LearnedRouter,
    X: torch.Tensor,
    Y: torch.Tensor,
    steps: int = 300,
    lr: float = 5e-3,
    batch_size: int = 64,
    seed: int = 0,
) -> list[float]:
    """Train a LearnedRouter's gate so its top-k matches target module sets.

    X: (n, input_dim) input feature vectors.
    Y: (n, n_modules) multi-hot target module sets (1.0 = module should be
       in the active set for that input).
    Loss: per-module binary cross-entropy on the gate logits. Returns the
    per-step loss history.
    """
    opt = torch.optim.Adam(router.gate.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    history: list[float] = []
    for _ in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        scores = router.scores(X[idx])
        loss = F.binary_cross_entropy_with_logits(scores, Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return history
