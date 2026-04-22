"""Metabolite sink rules for short-window runs with knocked-out
upstream consumers.

Problem
-------
At ``t_end >> 0.5 s``, a transporter-KO can cause its upstream
metabolite to accumulate unboundedly in the simulator (observed in v7:
transporter 0034's pool blew up to 13.3x WT at t_end=5.0 s). Real
cells have alternate consumers and passive diffusion equilibria that
cap this; the simulator models neither. The blow-up pushes max_pool_dev
through any threshold, turning a Breuer-Nonessential gene into a
persistent false positive in any ensemble detector that uses
pool-confirmation.

Fix
---
Add a first-order sink per metabolite:

    rate(sink:M) = k_sink * max(0, count(M) - tolerance * initial(M))
    apply       : count(M) -= 1

Sinks fire only above a tolerance multiple of the initial count, so
wild-type dynamics (where pools stay within ~2x initial) are unaffected
but KO-induced blow-ups are drained back toward equilibrium. Equivalent
to a diffusion-to-medium term with a hard threshold.

Honest caveats
--------------
1. Sinks only matter at ``t_end >= 1 s`` where blowups actually grow.
   At the default t_end=0.5 s they fire rarely and contribute little.
2. The k_sink and tolerance parameters are biophysical approximations.
   Real diffusion-across-membrane timescales for small metabolites are
   ~1 s for small solutes; we use k_sink=100 /s per excess molecule as
   a conservative value that drains a 10x blowup in ~100 ms.
3. Sinks affect the full rule-vector propensity: FastEventSimulator
   will correctly schedule them as Gillespie events. This does
   slightly slow the simulation (more rules to evaluate) but the
   effect is minor.
"""
from __future__ import annotations

from dataclasses import dataclass

from layer2_field.dynamics import TransitionRule
from layer3_reactions.coupled import get_species_count, update_species_count


@dataclass(frozen=True)
class SinkConfig:
    k_sink_per_s: float = 100.0
    tolerance: float = 3.0
    min_initial_count: int = 10
    max_tokens: int = 1000


def make_metabolite_sink_rules(
    state,
    cfg: SinkConfig | None = None,
) -> list[TransitionRule]:
    """Build one TransitionRule per tracked intracellular metabolite.

    Each rule fires at a rate proportional to the excess above
    ``tolerance * initial_count`` and decrements the metabolite count
    by one per event. Only emits rules for metabolites with initial
    counts above ``min_initial_count`` so we don't add ~300 trivial
    rules.
    """
    cfg = cfg or SinkConfig()
    rules: list[TransitionRule] = []
    initial_counts: dict[str, int] = dict(state.metabolite_counts)
    infinite = getattr(state, "metabolite_infinite", set())

    for sid, init_c in initial_counts.items():
        if sid in infinite:
            continue
        if init_c < cfg.min_initial_count:
            continue
        threshold = cfg.tolerance * init_c
        k_sink = cfg.k_sink_per_s
        max_tokens = cfg.max_tokens

        def _make_can_fire(species_id, thr, max_t):
            def can_fire(state_):
                current = get_species_count(state_, species_id)
                excess = current - thr
                if excess <= 0:
                    return []
                n_tokens = min(int(excess), max_t)
                return [(species_id, 1)] * n_tokens
            return can_fire

        def _make_apply(species_id):
            def apply(state_, cands, rng):
                if not cands:
                    return
                current = get_species_count(state_, species_id)
                if current <= 0:
                    return
                update_species_count(state_, species_id, -1)
                state_.log_event(
                    f"sink:{species_id}",
                    [],
                    f"sink drain of {species_id} (current={current})",
                )
            return apply

        rule = TransitionRule(
            name=f"sink:{sid}",
            participants=[sid],
            rate=float(k_sink),
            rate_source="metabolite_sink_model",
            can_fire=_make_can_fire(sid, threshold, max_tokens),
            apply=_make_apply(sid),
        )
        rule.compiled_spec = {
            "kind": "sink",
            "species": sid,
            "threshold": float(threshold),
            "initial_count": int(init_c),
            "k_sink": float(k_sink),
        }
        rules.append(rule)

    return rules
