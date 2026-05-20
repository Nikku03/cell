"""Phase 11 verification: effective-synaptic-event accounting.

The SparseSpike-GEMM design measures energy per "effective synaptic
event" - one spike crossing one synapse (a signed accumulate on real
hardware). SparseSynapses now counts these events; the count is the
exact, hardware-agnostic efficiency metric for the substrate, and
synaptic_event_energy_pj() turns it into a picojoule estimate under an
explicit per-event energy assumption.

Gates:

  1. The synaptic-event counter is EXACT - it equals the number of
     edges whose pre-neuron fired, on a hand-checkable example
  2. The counter accumulates over a run and is cleared by reset_state()
  3. Lower neural activity produces proportionally fewer synaptic
     events and a lower energy estimate - the sparsity-equals-efficiency
     relationship the design relies on
"""

from __future__ import annotations

import time

import torch

from .population import (
    SparseSynapses,
    SYN_EXC,
    build_ei_microcircuit,
    synaptic_event_energy_pj,
)
from .verify_phase0 import CheckResult, format_results


# ---------------- gates ----------------

def check_event_counter_exact() -> CheckResult:
    """On an explicit 5-pre / 3-post / 5-edge synapse bank, the event
    counter equals the number of edges whose pre-neuron fired - checked
    against hand-computed values for three spike patterns."""
    edge_pre = torch.tensor([0, 0, 1, 2, 4])      # edges 0,1 leave pre-0
    edge_post = torch.tensor([0, 1, 2, 0, 1])
    kind = torch.full((5,), SYN_EXC, dtype=torch.long)
    g_max = torch.full((5,), 0.1)
    syn = SparseSynapses(n_pre=5, n_post=3, edge_pre=edge_pre, edge_post=edge_post,
                          kind=kind, g_max=g_max, tau_syn=5.0)
    syn.reset_state()
    # fire only pre-0 -> its 2 out-edges -> +2 events
    syn.deliver_spikes(torch.tensor([True, False, False, False, False]))
    e1 = syn.n_synaptic_events
    # fire pre-0 and pre-4 -> 2 (pre-0) + 1 (pre-4) -> +3, total 5
    syn.deliver_spikes(torch.tensor([True, False, False, False, True]))
    e2 = syn.n_synaptic_events
    # fire all 5 pre -> all 5 edges -> +5, total 10
    syn.deliver_spikes(torch.tensor([True, True, True, True, True]))
    e3 = syn.n_synaptic_events
    passed = (e1 == 2 and e2 == 5 and e3 == 10)
    return CheckResult(
        "synaptic-event counter is exact (matches hand-computed counts)",
        passed, float(e3), 10.0,
        note=f"(events after patterns: {e1} (expect 2), {e2} (expect 5), "
             f"{e3} (expect 10))",
    )


def check_event_counter_accumulates_and_resets() -> CheckResult:
    """The counter starts at zero, accumulates as a microcircuit runs,
    and is cleared back to zero by reset_state()."""
    cluster = build_ei_microcircuit(n_e=200, n_i=50, seed=0)
    syn = cluster.synapses[0]
    cluster.reset_state()
    at_start = syn.n_synaptic_events
    for k in range(100):
        cluster.step(dt=1.0, t=k * 1.0)
    after_run = syn.n_synaptic_events
    cluster.reset_state()
    after_reset = syn.n_synaptic_events
    passed = (at_start == 0 and after_run > 0 and after_reset == 0)
    return CheckResult(
        "event counter accumulates over a run and resets cleanly",
        passed, float(after_run), 1.0,
        note=f"(events: at start={at_start}, after 100 steps={after_run}, "
             f"after reset={after_reset})",
    )


def check_sparsity_reduces_events() -> CheckResult:
    """Two identical microcircuits (same seed -> same synapses) run at
    different activity levels: the lower-activity one produces fewer
    synaptic events and a lower estimated energy. This is the
    sparsity-equals-efficiency relationship the design depends on."""
    hi = build_ei_microcircuit(n_e=200, n_i=50, noise_std_e=7.0, seed=0)
    lo = build_ei_microcircuit(n_e=200, n_i=50, noise_std_e=3.0, seed=0)
    hi.reset_state()
    lo.reset_state()
    for k in range(200):
        hi.step(dt=1.0, t=k * 1.0)
        lo.step(dt=1.0, t=k * 1.0)
    ev_hi = hi.synapses[0].n_synaptic_events
    ev_lo = lo.synapses[0].n_synaptic_events
    e_hi = synaptic_event_energy_pj(ev_hi)
    e_lo = synaptic_event_energy_pj(ev_lo)
    passed = (ev_lo > 0 and ev_lo < ev_hi and e_lo < e_hi)
    ratio = ev_lo / max(ev_hi, 1)
    return CheckResult(
        "lower activity yields fewer synaptic events + lower energy",
        passed, ratio, 1.0,
        note=f"(events high-activity={ev_hi}, low-activity={ev_lo}, "
             f"ratio={ratio:.2f}; energy {e_hi/1000:.1f} nJ vs {e_lo/1000:.1f} nJ "
             f"@ 0.1 pJ/event)",
    )


# ---------------- runner ----------------

def run_all() -> list[CheckResult]:
    return [
        check_event_counter_exact(),
        check_event_counter_accumulates_and_resets(),
        check_sparsity_reduces_events(),
    ]


if __name__ == "__main__":
    t0 = time.time()
    results = run_all()
    print(f"Phase 11 verification ({time.time()-t0:.1f}s):")
    print(format_results(results))
    n_pass = sum(r.passed for r in results)
    print(f"\n  -> {n_pass} / {len(results)} checks passed.")
