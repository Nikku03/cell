"""Phase B2 bioelectric dynamics — voltage-gated K+ leak rule.

This module is gated entirely by ``RealSimulatorConfig.enable_bioelectric``.
At flag-off, ``build_bioelectric_rules`` is not called, so the simulator's
rule list is unchanged and bit-identity to v15 / v16 baselines is
preserved (verified by ``test_phase2_gex_off_bit_identity``).

What the rule does
------------------

A passive, voltage-gated K+ leak. Each event moves one K+ ion across
the membrane in the direction that reduces |Vmem - E_K|. Token count
is proportional to the driving force in mV, capped at 100 to keep
propensity bounded. The rule's scalar rate (events per second per
unit token) is a placeholder; see caveats.

Biology
-------

Real K+ leak channels (e.g. Trk in B. subtilis, KdpFABC in E. coli)
are passive: K+ moves down its electrochemical gradient at a rate
proportional to (Vmem - E_K), where E_K is the K+ Nernst potential.
For Syn3A's near-equal [K+]_in / [K+]_out, E_K ~ +6.3 mV at 37 C, so
when the GHK Vmem is far from +6 mV, K+ leaks until Vmem ~ E_K. This
is the simplest possible homeostatic feedback and the foundation of
real bacterial Vmem regulation.

What this rule does NOT do
--------------------------

* No proton motive force pumping (the F1F0 H+ ATPase is not modeled
  in the Luthey-Schulten Syn3A simulator at all; its M_h_c is held
  at sentinel infinity for propensity-calc purposes).
* No Na+ or Cl- channels. Phase B2 is K+-only by design; multi-ion
  channels would inflate the rule list disproportionately for a
  Phase 2-style observable test.
* No voltage-dependent gating beyond linear-in-driving-force. Real
  channels have sigmoidal Boltzmann gating (Hodgkin-Huxley style);
  Phase B2 uses a linear approximation that is biology-consistent at
  small driving forces and breaks down at large ones. The token cap
  at 100 makes the breakdown bounded.
* No coupling to the GHK estimator's permeability ratios.
  ``estimate_vmem_mv`` uses fixed P_K=1, P_Na=0.04, P_Cl=0.45;
  this rule changes K+ counts but does NOT also adjust permeabilities
  to reflect channel state.

Caveats (read before using results downstream)
----------------------------------------------

1. **Rate constant is a placeholder.** ``K_LEAK_RATE_PER_S_PER_MV`` is
   set to give roughly-percent-level K+ count drift over a 0.5 s
   simulation window — enough that flag-on differs measurably from
   flag-off, not so much that metabolic dynamics are perturbed
   beyond noise. Real bacterial K+ leak rates are not measured for
   Syn3A; the value here is engineering, not biology.

2. **Bit-identity at flag-off is the contract.** This rule is added
   to the rule list by ``build_bioelectric_rules(state)`` only when
   ``RealSimulatorConfig.enable_bioelectric`` is True. The
   ``test_phase2_gex_off_bit_identity`` regression covers flag-off
   trajectories; if it ever fails after a change here, the change
   broke the contract and must be reverted.

3. **The rule fires often when Vmem deviates from E_K.** With the
   placeholder rate (100/s per token) and a typical driving force
   of ~7 mV at the simulator's resting state, propensity is
   ~700 / s, so ~350 K+ leak events in a 0.5 s window. That's
   ~0.18 percent of the intracellular K+ count. The trajectory
   sees this as a small but measurable VMEM_MV drift, which is
   the whole point.

4. **No rule-knockout coupling.** Knocking out a gene does not
   change the K+ leak rule (no specific gene encodes "passive K+
   leak" in the simulator). The bioelectric rule is global and
   per-cell, not per-gene. A future Phase B2.5 could add specific
   ion-channel-protein knockouts with rule modulation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Layer 2 / 3 modules use unprefixed imports + sys.path tricks. Mirror.
_CELL_SIM = Path(__file__).resolve().parents[1]
if str(_CELL_SIM) not in sys.path:
    sys.path.insert(0, str(_CELL_SIM))

from layer2_field.dynamics import CellState, TransitionRule  # noqa: E402
from layer3_reactions.coupled import (  # noqa: E402
    update_species_count,
)

from cell_sim.layer5_bioelectric.bioelectric import (
    BioelectricEstimatorConfig,
    estimate_vmem_mv,
    nernst_potential_mv,
    _count_to_mM,
    DEFAULT_REFERENCE_EXTRACELLULAR_MM,
    DEFAULT_TEMPERATURE_K,
)


# ============================================================================
# Tunable parameters
# ============================================================================
K_LEAK_RATE_PER_S_PER_MV = 100.0
K_LEAK_TOKEN_CAP = 100
K_LEAK_DEAD_BAND_MV = 0.5  # below this driving force, no leak fires


# ============================================================================
# Rule factory
# ============================================================================
def make_voltage_gated_K_leak_rule(
    estimator_config: BioelectricEstimatorConfig | None = None,
    rate_per_s_per_mv: float = K_LEAK_RATE_PER_S_PER_MV,
    token_cap: int = K_LEAK_TOKEN_CAP,
    dead_band_mv: float = K_LEAK_DEAD_BAND_MV,
) -> TransitionRule:
    """Return the voltage-gated K+ leak ``TransitionRule``.

    The rule is a single global rule (not per-gene). Each fire moves
    one K+ ion across the membrane in the direction that reduces
    |Vmem - E_K|. Token count per ``can_fire`` call is roughly the
    driving force in mV, capped.
    """
    cfg = estimator_config or BioelectricEstimatorConfig()
    name = "bio_k_leak"

    # Pre-compute E_K at the reference K+ concentrations so we don't
    # have to rerun the Nernst computation every can_fire.
    e_k_ref_mv = nernst_potential_mv(
        # Use the simulator-default intracellular concentration as the
        # reference baseline. The actual driving force is (Vmem - E_K_now),
        # where E_K_now uses the current state's [K+]_i, but for
        # propensity bookkeeping we recompute E_K from current state.
        conc_inside_mM=10.0,
        conc_outside_mM=float(cfg.reference_extracellular_mM.get("M_k_e", 12.67)),
        charge=+1,
        temperature_K=cfg.temperature_K,
    )

    def _current_e_k_mv(state) -> float:
        vol = float(getattr(state, "metabolite_volume_L", 0.0) or 0.0)
        if vol <= 0:
            return e_k_ref_mv
        mc = getattr(state, "metabolite_counts", None) or {}
        k_in_mM = _count_to_mM(float(mc.get("M_k_c", 0)), vol)
        if k_in_mM <= 0:
            return e_k_ref_mv
        return nernst_potential_mv(
            conc_inside_mM=k_in_mM,
            conc_outside_mM=float(cfg.reference_extracellular_mM.get("M_k_e", 12.67)),
            charge=+1,
            temperature_K=cfg.temperature_K,
        )

    def can_fire(state):
        # Only fire when the simulator has the metabolite-tracking state
        # the GHK estimator needs.
        mc = getattr(state, "metabolite_counts", None)
        if mc is None or "M_k_c" not in mc:
            return []
        v_mv = estimate_vmem_mv(state, cfg)
        # Defensive: NaN / inf -> no fires.
        if not (v_mv == v_mv) or v_mv == float("inf") or v_mv == float("-inf"):
            return []
        e_k_mv = _current_e_k_mv(state)
        driving_mv = v_mv - e_k_mv
        if abs(driving_mv) < dead_band_mv:
            return []
        # Direction: +1 means K+ flows IN (cell gets more K+),
        # -1 means K+ flows OUT (cell loses K+).
        # If Vmem > E_K: outside is electrically attractive for K+ to
        # leave (K+ is positive, exits down electrochemical gradient).
        # That's K+ leaving, direction = -1.
        direction = -1 if driving_mv > 0 else +1
        n_tokens = min(token_cap, max(1, int(abs(driving_mv))))
        return [(direction,)] * n_tokens

    def apply(state, cands, rng):
        if not cands:
            return
        # All tokens for one fire are the same direction (set in
        # can_fire); pick the first.
        direction = int(cands[0][0])
        update_species_count(state, "M_k_c", direction)
        state.log_event(
            name, [],
            f"bioelectric K+ leak (direction={'in' if direction > 0 else 'out'})",
        )

    rule = TransitionRule(
        name=name,
        participants=[],
        rate=float(rate_per_s_per_mv),
        rate_source="placeholder_bacterial_default",
        can_fire=can_fire,
        apply=apply,
    )
    return rule


def build_bioelectric_rules(state) -> List[TransitionRule]:
    """Phase B2 entrypoint. Returns the list of bioelectric rules to
    add to the simulator's rule list when
    ``RealSimulatorConfig.enable_bioelectric`` is True.

    Phase B2 ships exactly one rule (voltage-gated K+ leak). Phase B2.5
    or B3 may add Na+/Cl- counterparts; the entrypoint is the place
    those would slot in.
    """
    return [make_voltage_gated_K_leak_rule()]


__all__ = [
    "K_LEAK_DEAD_BAND_MV",
    "K_LEAK_RATE_PER_S_PER_MV",
    "K_LEAK_TOKEN_CAP",
    "build_bioelectric_rules",
    "make_voltage_gated_K_leak_rule",
]
