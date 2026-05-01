"""Layer 5: bioelectric — Phase B1 OBSERVABLE-ONLY.

This package adds a derived membrane-voltage (`Vmem`) observable to the
production simulator, defaulting OFF. Phase B1 is intentionally minimal:

* No new dynamical state is added to the simulator.
* No new transition rules are added to the rule list.
* When ``RealSimulatorConfig.enable_bioelectric`` is True, the
  per-snapshot `_snapshot()` augments its `pools` dict with a single
  derived value `VMEM_MV` computed from existing ion counts via the
  Goldman-Hodgkin-Katz equation.
* When the flag is False, the snapshot is bit-identical to the v15 /
  v16 baseline. This is enforced by the existing Phase 2 regression
  test.

The intent is the smallest possible step toward a Levin-style
bioelectric layer: get a real membrane-voltage observable into the
trajectory snapshots so a future Phase B3 detector can use it as a
feature. Phase B2 (future) adds voltage-gated ion-flux dynamics with
real feedback. Phase B3 (future) measures whether v17 detector
features derived from Vmem improve MCC against the v16 baseline.

The bioelectric framework here is classical, not quantum. The
"quantum biology speculation synthesis" note at
``experiments/quantum_biology_speculation/SYNTHESIS.md`` documents
why classical bioelectric (Levin lab) is the published-evidence-
supported framework for cellular computation; quantum frameworks
are not. This layer is the scoped engineering follow-up.

Wiring is intentionally light. The only `RealSimulator` change is:

  1. New config flag ``enable_bioelectric: bool = False``.
  2. ``_snapshot()`` calls
     :func:`cell_sim.layer5_bioelectric.bioelectric.estimate_vmem_mv`
     when the flag is on and adds the result as ``pools["VMEM_MV"]``.

That's it. The simulator's `_step` / `_apply` paths are untouched.

References
----------

* Goldman-Hodgkin-Katz equation. See e.g. Hille, "Ion Channels of
  Excitable Membranes" (3rd ed., 2001), chapter 14.
* Luthey-Schulten Syn3A simulation medium — ion concentrations from
  ``cell_sim/data/Minimal_Cell_ComplexFormation/input_data/initial_concentrations.xlsx``
  (sheets "Simulation Medium" + "Intracellular Metabolites").
* Levin lab on bioelectric cellular cognition. See
  ``memory_bank/facts/structural/regulation_layer_phase_r1.json``
  caveats for the broader context; primary refs in the synthesis
  note at ``experiments/quantum_biology_speculation/SYNTHESIS.md``.
"""
from __future__ import annotations

from cell_sim.layer5_bioelectric.bioelectric import (
    DEFAULT_PERMEABILITY,
    DEFAULT_REFERENCE_EXTRACELLULAR_MM,
    BioelectricEstimatorConfig,
    estimate_vmem_mv,
    nernst_potential_mv,
)
from cell_sim.layer5_bioelectric.detector_features import (
    BioelectricFeatures,
    extract_bioelectric_features,
)
from cell_sim.layer5_bioelectric.dynamics import (
    K_LEAK_DEAD_BAND_MV,
    K_LEAK_RATE_PER_S_PER_MV,
    K_LEAK_TOKEN_CAP,
    build_bioelectric_rules,
    make_voltage_gated_K_leak_rule,
)

__all__ = [
    "BioelectricEstimatorConfig",
    "BioelectricFeatures",
    "DEFAULT_PERMEABILITY",
    "DEFAULT_REFERENCE_EXTRACELLULAR_MM",
    "K_LEAK_DEAD_BAND_MV",
    "K_LEAK_RATE_PER_S_PER_MV",
    "K_LEAK_TOKEN_CAP",
    "build_bioelectric_rules",
    "estimate_vmem_mv",
    "extract_bioelectric_features",
    "make_voltage_gated_K_leak_rule",
    "nernst_potential_mv",
]
