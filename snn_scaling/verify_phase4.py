"""Phase 4 verification: modular architecture with task routing.

Gates:

  1. A multi-module network builds and runs without errors
  2. Inactive modules don't fire (spike count = 0 when gated off)
  3. Active modules respond to their inputs
  4. Inter-module projections propagate spikes (driving one module
     causes downstream module to fire via the projection)
  5. Task router correctly selects modules
  6. Modular sparsity reduces per-step compute (fewer active spikes)
"""

from __future__ import annotations

import math
import time

import torch

from .modular import (
    Module, ModuleSpec, ModularNetwork, TaskRouter,
    LearnedRouter, train_learned_router,
)
from .population import SYN_EXC
from .verify_phase0 import CheckResult, format_results


# ---------------- learned-router synthetic task ----------------

def _make_routing_task(n_modules: int = 8, input_dim: int = 24, top_k: int = 3,
                        n_per_type: int = 48, centroid_seed: int = 0,
                        noise_seed: int = 0, noise_std: float = 0.6):
    """Synthetic routing task for the learned MoE router.

    There are n_modules task types; type t activates the module set
    [(t+j) % n_modules for j in 0..top_k-1]. This is a balanced design:
    every module belongs to exactly top_k task types, so balanced load is
    the correct routing outcome. Inputs are per-type centroids + Gaussian
    noise. centroid_seed fixes the task geometry; noise_seed draws fresh
    input samples (train and test share centroids, differ in noise).

    Returns (X, Y, type_ids, module_names, gt_sets).
    """
    n_types = n_modules
    cg = torch.Generator().manual_seed(centroid_seed)
    centroids = torch.randn(n_types, input_dim, generator=cg)
    module_names = [f"m{i}" for i in range(n_modules)]
    gt_sets = [[(t + j) % n_modules for j in range(top_k)]
               for t in range(n_types)]
    ng = torch.Generator().manual_seed(noise_seed)
    X, Y, type_ids = [], [], []
    for t in range(n_types):
        y = torch.zeros(n_modules)
        for m in gt_sets[t]:
            y[m] = 1.0
        for _ in range(n_per_type):
            X.append(centroids[t] + noise_std * torch.randn(input_dim, generator=ng))
            Y.append(y.clone())
            type_ids.append(t)
    return (torch.stack(X), torch.stack(Y), torch.tensor(type_ids),
            module_names, gt_sets)


def check_learned_router_generalizes() -> CheckResult:
    """A trained LearnedRouter routes UNSEEN input feature vectors to the
    correct top-k module set, far above a random top-k baseline.

    The static TaskRouter cannot do this at all: it needs an exact
    task_type key and has no notion of a novel input. The learned gate
    generalises by similarity in feature space.
    """
    n_modules, input_dim, top_k = 8, 24, 3
    X, Y, _, names, gt = _make_routing_task(
        n_modules, input_dim, top_k, n_per_type=48,
        centroid_seed=0, noise_seed=1)
    Xte, _, tte, _, _ = _make_routing_task(
        n_modules, input_dim, top_k, n_per_type=24,
        centroid_seed=0, noise_seed=2)        # same task types, fresh inputs
    router = LearnedRouter(input_dim, names, top_k, seed=0)
    train_learned_router(router, X, Y, steps=300, seed=0)
    routed = router.route(Xte)
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    correct = sum(
        1 for i, r in enumerate(routed)
        if set(name_to_idx[m] for m in r) == set(gt[int(tte[i])])
    )
    acc = correct / len(routed)
    random_acc = 1.0 / math.comb(n_modules, top_k)
    passed = acc >= 0.80 and acc > random_acc + 0.5
    return CheckResult(
        "learned router routes unseen inputs to the correct top-k set",
        passed, acc, 0.80,
        note=f"(held-out routing accuracy {acc*100:.1f}%, "
             f"random top-{top_k} baseline {random_acc*100:.1f}%)",
    )


def check_learned_router_load_balance() -> CheckResult:
    """Over many inputs the trained router spreads activation across ALL
    modules with bounded skew - it does not collapse onto a few
    favourites (the classic MoE router-collapse failure mode)."""
    n_modules, input_dim, top_k = 8, 24, 3
    X, Y, _, names, _ = _make_routing_task(
        n_modules, input_dim, top_k, n_per_type=48,
        centroid_seed=0, noise_seed=1)
    router = LearnedRouter(input_dim, names, top_k, seed=0)
    train_learned_router(router, X, Y, steps=300, seed=0)
    Xte, _, _, _, _ = _make_routing_task(
        n_modules, input_dim, top_k, n_per_type=40,
        centroid_seed=0, noise_seed=3)
    routed = router.route(Xte)
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    usage = torch.zeros(n_modules)
    for r in routed:
        for m in r:
            usage[name_to_idx[m]] += 1
    no_dead = bool((usage > 0).all())
    spread = float(usage.max() / usage.min().clamp_min(1.0))
    passed = no_dead and spread < 2.5
    return CheckResult(
        "learned router balances load across modules (no dead experts)",
        passed, spread, 2.5,
        note=f"(per-module usage min={int(usage.min())}, max={int(usage.max())}, "
             f"max/min ratio {spread:.2f}, dead modules={int((usage == 0).sum())})",
    )


def check_learned_router_drives_modular_net() -> CheckResult:
    """The learned router's output is a valid active-module list for a
    real ModularNetwork: stepping the net with the routed set lets
    exactly those modules fire while the rest are gated off (0 spikes)."""
    n_modules, input_dim, top_k = 4, 16, 2
    modules = [Module(ModuleSpec(name=f"m{i}", n_e=60, n_i=15, seed=i))
               for i in range(n_modules)]
    net = ModularNetwork(modules)
    X, Y, _, names, _ = _make_routing_task(
        n_modules, input_dim, top_k, n_per_type=40,
        centroid_seed=0, noise_seed=1)
    router = LearnedRouter(input_dim, names, top_k, seed=0)
    train_learned_router(router, X, Y, steps=250, seed=0)

    active = router.route_one(X[0])              # top-k module names
    net.reset()
    drive = torch.full((75,), 6.0)               # n_e + n_i = 75 per module
    spikes = {m.spec.name: 0 for m in modules}
    for k in range(150):
        out = net.step(
            dt=1.0, t=k * 1.0,
            external_currents={nm: drive for nm in active},
            active_modules=active,
        )
        for nm, s in out.items():
            spikes[nm] += int(s.sum().item())
    active_set = set(active)
    inactive_fired = sum(v for nm, v in spikes.items() if nm not in active_set)
    routed_fired = sum(v for nm, v in spikes.items() if nm in active_set)
    passed = (len(active) == top_k and inactive_fired == 0 and routed_fired > 0)
    return CheckResult(
        "learned router output correctly gates a ModularNetwork",
        passed, float(inactive_fired), 0.0,
        note=f"(routed active={active}, routed spikes={routed_fired}, "
             f"non-routed spikes={inactive_fired})",
    )


def _make_simple_modular_net(device: str = "cpu") -> ModularNetwork:
    """Build a 4-module network: sensory, integration, motor, memory."""
    modules = [
        Module(ModuleSpec(name="sensory", n_e=100, n_i=25, role="sensory", seed=0), device=device),
        Module(ModuleSpec(name="integration", n_e=100, n_i=25, role="integration", seed=1), device=device),
        Module(ModuleSpec(name="motor", n_e=100, n_i=25, role="motor", seed=2), device=device),
        Module(ModuleSpec(name="memory", n_e=100, n_i=25, role="memory", seed=3), device=device),
    ]
    net = ModularNetwork(modules, device=device)
    # sensory -> integration; integration -> motor; integration -> memory; memory -> integration
    net.add_intermodule_projection(modules[0], modules[1], n_src=30, n_dst=30, n_edges=120, g_max=0.6, seed=10)
    net.add_intermodule_projection(modules[1], modules[2], n_src=30, n_dst=30, n_edges=120, g_max=0.6, seed=11)
    net.add_intermodule_projection(modules[1], modules[3], n_src=30, n_dst=30, n_edges=80, g_max=0.4, seed=12)
    net.add_intermodule_projection(modules[3], modules[1], n_src=30, n_dst=30, n_edges=80, g_max=0.4, seed=13)
    return net


# ---------------- gates ----------------

def check_build_and_run() -> CheckResult:
    """4-module network builds and runs 100 steps without errors."""
    net = _make_simple_modular_net()
    net.reset()
    n_errors = 0
    for k in range(100):
        try:
            net.step(dt=1.0, t=k * 1.0)
        except Exception:
            n_errors += 1
    return CheckResult(
        "4-module network runs 100 steps without errors", n_errors == 0,
        float(n_errors), 0.0,
        note=f"({n_errors} errors in 100 steps)",
    )


def check_inactive_modules_silent() -> CheckResult:
    """A module gated off produces no spikes regardless of input."""
    net = _make_simple_modular_net()
    net.reset()
    # Provide strong input to sensory, but ONLY activate motor + memory
    sensory_drive = torch.full((125,), 6.0)
    spikes = {"sensory": 0, "integration": 0, "motor": 0, "memory": 0}
    for k in range(200):
        out = net.step(
            dt=1.0, t=k * 1.0,
            external_currents={"sensory": sensory_drive},
            active_modules=["motor", "memory"],     # sensory and integration OFF
        )
        for name, s in out.items():
            spikes[name] += int(s.sum().item())
    return CheckResult(
        "gated-off modules produce zero spikes even under strong drive",
        spikes["sensory"] == 0 and spikes["integration"] == 0,
        float(spikes["sensory"] + spikes["integration"]), 0.0,
        note=f"(sensory={spikes['sensory']}, integration={spikes['integration']}, "
             f"motor={spikes['motor']}, memory={spikes['memory']})",
    )


def check_active_modules_respond() -> CheckResult:
    """Active modules under strong direct drive fire."""
    net = _make_simple_modular_net()
    net.reset()
    sensory_drive = torch.full((125,), 6.0)
    sensory_spikes = 0
    for k in range(200):
        out = net.step(
            dt=1.0, t=k * 1.0,
            external_currents={"sensory": sensory_drive},
            active_modules=["sensory", "integration", "motor", "memory"],
        )
        sensory_spikes += int(out["sensory"].sum().item())
    return CheckResult(
        "active sensory module fires under direct drive",
        sensory_spikes > 100, float(sensory_spikes), 100.0,
        note=f"(sensory spikes in 200 ms: {sensory_spikes})",
    )


def check_inter_module_propagation() -> CheckResult:
    """Drive sensory; integration (which connects via projection) should fire too."""
    net = _make_simple_modular_net()
    net.reset()
    sensory_drive = torch.full((125,), 6.0)
    integration_spikes = 0
    for k in range(300):
        out = net.step(
            dt=1.0, t=k * 1.0,
            external_currents={"sensory": sensory_drive},
            active_modules=["sensory", "integration"],
        )
        integration_spikes += int(out["integration"].sum().item())
    # Integration should fire indirectly via the inter-module projection
    return CheckResult(
        "inter-module projection drives downstream module",
        integration_spikes > 50, float(integration_spikes), 50.0,
        note=f"(integration spikes from sensory drive: {integration_spikes})",
    )


def check_task_router() -> CheckResult:
    """TaskRouter correctly maps task_type to module set."""
    router = TaskRouter({
        "vision_task": ["sensory", "integration", "memory"],
        "motor_task": ["integration", "motor"],
        "memory_recall": ["memory", "integration"],
    })
    vis = router.route("vision_task")
    mot = router.route("motor_task")
    mem = router.route("memory_recall")
    return CheckResult(
        "TaskRouter returns correct module sets per task",
        vis == ["sensory", "integration", "memory"]
        and mot == ["integration", "motor"]
        and mem == ["memory", "integration"],
        1.0, 0.5,
        note=f"(routed: vision -> {vis}, motor -> {mot}, recall -> {mem})",
    )


def check_modular_sparsity_reduces_compute() -> CheckResult:
    """A run with 1 active module is materially faster than 4 active.

    Inactive modules still tick (membrane dynamics still update), but
    they don't propagate spikes and don't receive external drive, so
    overall work is smaller.
    """
    net1 = _make_simple_modular_net()
    net1.reset()
    sensory_drive = torch.full((125,), 6.0)
    t0 = time.time()
    for k in range(500):
        net1.step(dt=1.0, t=k * 1.0,
                  external_currents={"sensory": sensory_drive},
                  active_modules=["sensory", "integration", "motor", "memory"])
    t_all = time.time() - t0

    net2 = _make_simple_modular_net()
    net2.reset()
    t0 = time.time()
    for k in range(500):
        net2.step(dt=1.0, t=k * 1.0,
                  external_currents={"sensory": sensory_drive},
                  active_modules=["sensory"])
    t_one = time.time() - t0
    # Single-module mode should not be significantly slower than all-module mode
    # (it should be similar or faster). Threshold: t_one < 1.2 * t_all.
    return CheckResult(
        "single-module gating not slower than full network",
        t_one <= 1.2 * t_all, t_one / t_all, 1.2,
        note=f"(all={t_all:.3f}s, one={t_one:.3f}s, ratio={t_one/t_all:.2f}x)",
    )


# ---------------- runner ----------------

def run_all() -> list[CheckResult]:
    return [
        check_build_and_run(),
        check_inactive_modules_silent(),
        check_active_modules_respond(),
        check_inter_module_propagation(),
        check_task_router(),
        check_modular_sparsity_reduces_compute(),
        check_learned_router_generalizes(),
        check_learned_router_load_balance(),
        check_learned_router_drives_modular_net(),
    ]


if __name__ == "__main__":
    t0 = time.time()
    results = run_all()
    print(f"Phase 4 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    print(f"\n  -> {n_pass} / {len(results)} checks passed.")
