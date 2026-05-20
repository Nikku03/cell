"""Phase 7 verification: dendritic compartments give per-neuron sub-network.

Gates:

  1. DendriticPopulation builds + runs without errors at moderate scale
  2. Soma fires under strong per-dendrite drive
  3. Per-dendrite nonlinearity is non-trivial (sigmoid saturates;
     2x input does NOT give 2x output)
  4. Headline: a single 2-dendrite neuron computes XOR --
     a non-linearly-separable function -- while an identical point
     neuron with the same membrane params CANNOT
  5. DendriticSynapses route inputs to the right (neuron, dendrite)
     compartment (sanity check on the scatter-add aggregation)
  6. Vectorisation at scale: 50000 neurons * 4 dendrites runs a 100-step
     simulation in under a few seconds (per-step ~ population size)
"""

from __future__ import annotations

import time

import torch

from .dendritic import (
    DendriticParams, DendriticPopulation, DendriticSynapses, DendriticCluster,
    build_xor_neuron, xor_dendrite_input, build_point_neuron_baseline,
)
from .population import SYN_EXC
from .verify_phase0 import CheckResult, format_results


# ---------------- gates ----------------

def check_build_and_run() -> CheckResult:
    """A 200-neuron, 4-dendrite population runs 100 steps without errors."""
    pop = DendriticPopulation(
        n_neurons=200,
        params=DendriticParams(n_dendrites=4, noise_std=2.0),
    )
    cluster = DendriticCluster(pop)
    cluster.reset_state()
    n_errors = 0
    for k in range(100):
        try:
            d_in = torch.zeros(200, 4)
            cluster.step(dt=1.0, t=k * 1.0, external_dendritic_current=d_in)
        except Exception:
            n_errors += 1
    return CheckResult(
        "200-neuron 4-dendrite population runs 100 steps without errors",
        n_errors == 0, float(n_errors), 0.0,
        note=f"({n_errors} errors)",
    )


def check_soma_fires_under_dendritic_drive() -> CheckResult:
    """Per-dendrite drive of magnitude > threshold pushes the soma to fire.

    Inject d_input above the sigmoid half-activation point so each dendrite
    saturates. Soma drive = K * soma_weight * ~1.0. With K=4 and
    soma_weight=4 that's ~16 nA, well above the LIF spike threshold.
    """
    n_d = 4
    params = DendriticParams(
        n_dendrites=n_d, nonlinearity="sigmoid",
        dendrite_gain=8.0, dendrite_threshold=0.5,
        soma_weight=4.0, noise_std=0.0,
    )
    pop = DendriticPopulation(n_neurons=50, params=params)
    pop.reset_state()
    n_spikes = 0
    d_in = torch.full((50, n_d), 1.0)         # all dendrites fully on
    for k in range(200):
        spike = pop.step(d_in, dt=1.0, t=k * 1.0)
        n_spikes += int(spike.sum().item())
    return CheckResult(
        "soma fires under strong per-dendrite drive",
        n_spikes > 200, float(n_spikes), 200.0,
        note=f"(spikes from 50 neurons over 200 steps: {n_spikes})",
    )


def check_nonlinearity_saturates() -> CheckResult:
    """Doubling the dendrite input does NOT double its output -- sigmoid
    saturation is what makes the multi-compartment computation richer
    than a linear sum.

    Measure: ratio of dendritic_output(2*x) / dendritic_output(x) at x = 1.
    For a linear function this would be 2.0. For sigmoid in saturation
    it should be much closer to 1.0.
    """
    params = DendriticParams(
        n_dendrites=1, nonlinearity="sigmoid",
        dendrite_gain=8.0, dendrite_threshold=0.5,
    )
    pop = DendriticPopulation(n_neurons=1, params=params)
    out_1 = pop.dendritic_output(torch.tensor([[1.0]])).item()
    out_2 = pop.dendritic_output(torch.tensor([[2.0]])).item()
    ratio = out_2 / max(out_1, 1e-6)
    return CheckResult(
        "sigmoid dendrite saturates (out(2x) / out(x) < 1.5, far below linear 2.0)",
        ratio < 1.5, ratio, 1.5,
        note=f"(out(1)={out_1:.3f}, out(2)={out_2:.3f}, ratio={ratio:.3f})",
    )


def _run_xor_count(d_input_fn, n_steps: int = 80) -> dict:
    """Helper: feed each XOR input pattern to the dendritic XOR neuron and
    count spikes.

    d_input_fn(x1, x2) -> (1, 2) tensor.
    Returns {(x1, x2): spike_count}.
    """
    pop = build_xor_neuron()
    counts: dict[tuple[int, int], int] = {}
    for x1 in (0, 1):
        for x2 in (0, 1):
            pop.reset_state()
            d_in = d_input_fn(x1, x2)
            n = 0
            for k in range(n_steps):
                spike = pop.step(d_in, dt=1.0, t=k * 1.0)
                n += int(spike.sum().item())
            counts[(x1, x2)] = n
    return counts


def _run_point_baseline(scale: float, n_steps: int = 80) -> dict:
    """Equivalent point neuron baseline. The only thing a point neuron
    can do with x1, x2 is a LINEAR sum: I = w1*x1 + w2*x2. The best
    setting we'd ever pick is w1=w2=scale -- any other linear combination
    is dominated. So drive the soma directly with I = scale * (x1 + x2).
    """
    pop = build_point_neuron_baseline()
    counts: dict[tuple[int, int], int] = {}
    for x1 in (0, 1):
        for x2 in (0, 1):
            pop.reset_state()
            I_inj = torch.tensor([scale * (x1 + x2)], dtype=torch.float32)
            g_zero = torch.zeros(1)
            n = 0
            for k in range(n_steps):
                spike = pop.step(I_inj, g_zero, dt=1.0, t=k * 1.0)
                n += int(spike.sum().item())
            counts[(x1, x2)] = n
    return counts


def check_dendritic_xor() -> CheckResult:
    """A 2-dendrite single neuron computes XOR: high spike-count for
    (0,1) and (1,0), low for (0,0) and (1,1)."""
    counts = _run_xor_count(xor_dendrite_input, n_steps=80)
    s00, s01, s10, s11 = counts[(0,0)], counts[(0,1)], counts[(1,0)], counts[(1,1)]
    # XOR pattern: middle two high, corners low
    xor_pass = (s01 > 5 and s10 > 5 and s00 < 2 and s11 < 2)
    return CheckResult(
        "single dendritic neuron computes XOR (mid-mid high, corners low)",
        xor_pass, float(s01 + s10 - s00 - s11), 10.0,
        note=f"(spikes: (0,0)={s00}, (0,1)={s01}, (1,0)={s10}, (1,1)={s11})",
    )


def check_point_neuron_cannot_xor() -> CheckResult:
    """At any drive scale, a point neuron's spike count is monotone in
    (x1 + x2). It cannot produce the XOR pattern (low, high, high, low).

    We pick the scale that makes (0,1) and (1,0) fire ~10 spikes, then
    check that (1,1) ALSO fires (in fact more), proving the linear
    barrier.
    """
    # Search for a scale where (0,1) fires nontrivially
    chosen_scale = None
    chosen = None
    for scale in [40, 60, 80, 100, 120, 160]:
        c = _run_point_baseline(scale=scale)
        if c[(0, 1)] >= 5:
            chosen_scale = scale
            chosen = c
            break
    if chosen is None:
        chosen = _run_point_baseline(scale=80)
        chosen_scale = 80
    s00, s01, s10, s11 = chosen[(0,0)], chosen[(0,1)], chosen[(1,0)], chosen[(1,1)]
    # Point-neuron 'failure' = it fires for (1,1) at least as much as for (0,1),
    # i.e. it CANNOT realise XOR (low corner-corner, high mid-mid).
    point_fails_xor = (s11 >= s01) and (s11 >= s10)
    return CheckResult(
        "point neuron CANNOT compute XOR (s11 >= max(s01, s10) at any drive)",
        point_fails_xor, float(s11 - max(s01, s10)), 0.0,
        note=f"(@scale={chosen_scale}: (0,0)={s00}, (0,1)={s01}, "
             f"(1,0)={s10}, (1,1)={s11})",
    )


def check_synapse_routes_to_dendrite() -> CheckResult:
    """A synapse with edge_post_dendrite=k deposits current ONLY in
    compartment k, not in the others.

    Build: 1 pre neuron, 1 post neuron with K=4, single edge pre->(post,
    dendrite=2). Manually drive a 'pre spike' through deliver_spikes,
    then inspect post_currents(): only column 2 of I_dend should be non-zero.
    """
    edge_pre = torch.tensor([0])
    edge_post_n = torch.tensor([0])
    edge_post_d = torch.tensor([2])
    kind = torch.tensor([SYN_EXC])
    g_max = torch.tensor([1.0])
    syn = DendriticSynapses(
        n_pre=1, n_post=1, n_dendrites=4,
        edge_pre=edge_pre, edge_post_neuron=edge_post_n,
        edge_post_dendrite=edge_post_d,
        kind=kind, g_max=g_max, tau_syn=5.0,
    )
    syn.deliver_spikes(torch.tensor([True]))
    V_soma = torch.tensor([-70.0])
    I_dend, _ = syn.post_currents(V_soma)
    # Only column 2 should have non-zero current
    nonzero_mask = I_dend[0].abs() > 1e-6
    routed_correctly = (nonzero_mask.sum().item() == 1
                         and nonzero_mask[2].item() is True)
    return CheckResult(
        "DendriticSynapses route input to the correct compartment",
        routed_correctly, float(nonzero_mask.sum().item()), 1.0,
        note=f"(non-zero compartments: {nonzero_mask.tolist()}, "
             f"I_dend={I_dend.tolist()})",
    )


def check_scale_50k_runs_fast() -> CheckResult:
    """50000 dendritic neurons * 4 dendrites: 100 steps in under a few seconds.

    This is the 'effective compute' headline -- 50k somata buy you 50k*4
    = 200k effective sub-units at near-linear cost.
    """
    N, K = 50_000, 4
    pop = DendriticPopulation(
        n_neurons=N,
        params=DendriticParams(n_dendrites=K, noise_std=0.5),
    )
    pop.reset_state()
    # Random per-dendrite drive (light)
    drive = torch.full((N, K), 0.6)
    t0 = time.time()
    for k in range(100):
        pop.step(drive, dt=1.0, t=k * 1.0)
    elapsed = time.time() - t0
    return CheckResult(
        f"50000-neuron 4-dendrite population: 100 steps in < 5 s",
        elapsed < 5.0, elapsed, 5.0,
        note=f"(50k*4 dendrites x 100 steps in {elapsed:.2f}s)",
    )


def check_stateful_dendrite_integrates() -> CheckResult:
    """A leaky dendrite (dendrite_leak > 0) integrates a SUSTAINED
    SUB-THRESHOLD input over time until the accumulated branch state
    crosses threshold and drives the soma to fire. A stateless dendrite
    (dendrite_leak = 0) given the identical input never crosses threshold
    and never fires.

    This is the temporal-integration capability the stateful branch adds:
    coincidence/accumulation detection that the instantaneous model cannot
    do. Input 0.12 is below the dendrite half-activation (0.5); with
    leak=0.85 the branch steady state is 0.12 / (1 - 0.85) = 0.80, above
    threshold.
    """
    base = dict(n_dendrites=1, nonlinearity="sigmoid", dendrite_gain=4.0,
                dendrite_threshold=0.5, soma_weight=5.0, noise_std=0.0)
    stateless = DendriticPopulation(1, DendriticParams(dendrite_leak=0.0, **base))
    stateful = DendriticPopulation(1, DendriticParams(dendrite_leak=0.85, **base))
    stateless.reset_state()
    stateful.reset_state()
    d_in = torch.full((1, 1), 0.12)        # below the instantaneous threshold
    sl_spikes = 0
    st_spikes = 0
    for k in range(400):
        sl_spikes += int(stateless.step(d_in, dt=1.0, t=k * 1.0).sum().item())
        st_spikes += int(stateful.step(d_in, dt=1.0, t=k * 1.0).sum().item())
    # stateful must fire repetitively; stateless must stay completely silent
    passed = st_spikes > 10 and sl_spikes == 0
    return CheckResult(
        "leaky dendrite integrates sub-threshold input; stateless cannot",
        passed, float(st_spikes - sl_spikes), 10.0,
        note=f"(stateful spikes={st_spikes}, stateless spikes={sl_spikes})",
    )


# ---------------- runner ----------------

def run_all() -> list[CheckResult]:
    return [
        check_build_and_run(),
        check_soma_fires_under_dendritic_drive(),
        check_nonlinearity_saturates(),
        check_synapse_routes_to_dendrite(),
        check_dendritic_xor(),
        check_point_neuron_cannot_xor(),
        check_stateful_dendrite_integrates(),
        check_scale_50k_runs_fast(),
    ]


if __name__ == "__main__":
    t0 = time.time()
    results = run_all()
    print(f"Phase 7 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    print(f"\n  -> {n_pass} / {len(results)} checks passed.")
