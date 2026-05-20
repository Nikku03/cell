"""Phase 10 verification: DEEP-R structural plasticity (synapse rewiring).

RewiringSynapses keeps a fixed, over-provisioned edge buffer: `capacity`
slots pre-allocated once, at most `n_live` live at any moment. Rewiring
prunes the weakest live synapses and grows fresh ones into dead slots -
O(swaps) flag/index updates, never a tensor reallocation. Dead slots are
held inert (g_max = 0), so the standard forward pass needs no masking.

Gates:

  1. The over-provisioned edge buffer is constant-size across many
     rewiring rounds (no reallocation, no fragmentation)
  2. Dead buffer slots are inert (carry zero conductance); live slots
     carry conductance
  3. Rewiring preserves the live-edge count - sparsity is invariant
     while the connection SET turns over
  4. After a lesion (random synapse loss) the network regrows
     connectivity and recovers its synaptic throughput
"""

from __future__ import annotations

import time

import torch

from .rewiring import RewiringSynapses
from .verify_phase0 import CheckResult, format_results


def _throughput(syn: RewiringSynapses) -> float:
    """Total synaptic conductance delivered when every pre-neuron fires
    once - a structural measure of how much signal the synapse bank can
    carry. Equals the sum of g_max over live edges."""
    syn.reset_state()
    syn.deliver_spikes(torch.ones(syn.n_pre, dtype=torch.bool))
    return float(syn.g.sum())


# ---------------- gates ----------------

def check_rewiring_buffer_constant() -> CheckResult:
    """The over-provisioned edge buffer never reallocates: every edge
    tensor keeps length == capacity through 100 rewiring rounds."""
    syn = RewiringSynapses(n_pre=200, n_post=200, capacity=2000,
                            n_initial=600, seed=0)
    before = (syn.edge_pre.numel(), syn.edge_post.numel(), syn.g.numel(),
              syn.g_max.numel(), syn.live.numel())
    g = torch.Generator().manual_seed(1)
    for _ in range(100):
        syn.rewire(25, g)
    after = (syn.edge_pre.numel(), syn.edge_post.numel(), syn.g.numel(),
             syn.g_max.numel(), syn.live.numel())
    all_capacity = all(x == 2000 for x in after)
    passed = (before == after) and all_capacity
    return CheckResult(
        "over-provisioned edge buffer is constant-size across rewiring",
        passed, float(after[0]), 2000.0,
        note=f"(buffer sizes before={before}, after 100 rewires={after})",
    )


def check_dead_slots_inert() -> CheckResult:
    """Dead buffer slots carry zero conductance even when their (stale)
    pre-neuron fires; live slots carry conductance. This is what lets the
    plain scatter-add forward pass ignore dead edges without any mask."""
    syn = RewiringSynapses(n_pre=150, n_post=150, capacity=1000,
                            n_initial=400, seed=2)
    syn.reset_state()
    syn.deliver_spikes(torch.ones(150, dtype=torch.bool))     # every pre fires
    dead_silent = bool((syn.g[~syn.live] == 0).all())
    live_active = bool((syn.g[syn.live] > 0).all())
    passed = dead_silent and live_active
    return CheckResult(
        "dead buffer slots are inert; live slots carry conductance",
        passed, float(dead_silent and live_active), 0.5,
        note=f"(dead-slot g all zero={dead_silent}, "
             f"live-slot g all >0={live_active}, n_live={syn.n_live})",
    )


def check_rewiring_preserves_live_count() -> CheckResult:
    """Rewiring (prune n weakest + grow n fresh) preserves the live-edge
    count exactly - the network stays equally sparse while the connection
    SET turns over."""
    syn = RewiringSynapses(n_pre=200, n_post=200, capacity=2000,
                            n_initial=700, seed=3)
    g = torch.Generator().manual_seed(4)
    counts = [syn.n_live]
    for _ in range(50):
        syn.rewire(35, g)
        counts.append(syn.n_live)
    passed = all(c == 700 for c in counts)
    return CheckResult(
        "rewiring preserves the live-edge count (sparsity invariant)",
        passed, float(max(counts) - min(counts)), 0.0,
        note=f"(n_live across 50 rewires: min={min(counts)}, max={max(counts)})",
    )


def check_rewiring_recovers_from_lesion() -> CheckResult:
    """Structural plasticity recovers from damage: lesioning half the
    synapses roughly halves throughput; growing fresh synapses back
    restores both the live-edge count and most of the throughput."""
    syn = RewiringSynapses(n_pre=200, n_post=200, capacity=2000,
                            n_initial=800, seed=0)
    g = torch.Generator().manual_seed(7)
    baseline = _throughput(syn)
    syn.lesion(400, g)                       # damage: 400 random synapses lost
    lesioned = _throughput(syn)
    syn.grow(400, g)                         # structural regrowth
    recovered = _throughput(syn)
    passed = (lesioned < 0.65 * baseline
              and recovered > 0.90 * baseline
              and syn.n_live == 800)
    return CheckResult(
        "rewiring regrows connectivity + throughput after a lesion",
        passed, recovered / baseline, 0.90,
        note=f"(throughput baseline={baseline:.1f} -> lesioned={lesioned:.1f} "
             f"-> recovered={recovered:.1f}; n_live restored to {syn.n_live})",
    )


# ---------------- runner ----------------

def run_all() -> list[CheckResult]:
    return [
        check_rewiring_buffer_constant(),
        check_dead_slots_inert(),
        check_rewiring_preserves_live_count(),
        check_rewiring_recovers_from_lesion(),
    ]


if __name__ == "__main__":
    t0 = time.time()
    results = run_all()
    print(f"Phase 10 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    print(f"\n  -> {n_pass} / {len(results)} checks passed.")
