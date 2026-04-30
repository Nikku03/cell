"""Layer 4: regulation — INFRASTRUCTURE ONLY (Phase R1).

This package contains the rule factories, pseudo-species namespace, and
YAML loader for transcription factor binding, sigma factor competition,
riboswitch folding, and two-component signaling. **No regulatory entries
are curated yet.** The YAML config at
``cell_sim/data/regulation_network_syn3a.yaml`` is intentionally empty
(schema-only). The production essentiality sweep does NOT invoke this
layer; the v15 (gex-off) and v16 (gex-on) MCC values are unaffected.

The four rule factories are syntactically complete and produce valid
``TransitionRule`` objects, but their kinetic parameters are unspecified
at this phase. Calling a factory without a verified ``source_citation``
emits a ``BiologyNotCurated`` warning. Tests suppress the warning with
``warnings.catch_warnings`` so the architectural contract can be checked
without claiming biology.

Wiring this layer into ``layer6_essentiality.real_simulator.RealSimulator``
is Phase R2 work and requires:

  1. Curating ``regulation_network_syn3a.yaml`` from real Syn3A regulatory
     data (sigma factors via sequence homology, TFs via Pfam HMM, etc.).
  2. Adding an ``enable_regulation: bool = False`` flag to
     ``RealSimulatorConfig``.
  3. Re-measuring the bit-identity regression at flag-off (must equal
     the v16 baseline) and the v17 MCC at flag-on.

See ``FUTURE_WORK.md`` for the curation checklist.
"""
from __future__ import annotations

from cell_sim.layer4_regulation.regulation import (
    BiologyNotCurated,
    make_riboswitch_rule,
    make_sigma_competition_rule,
    make_tf_binding_rule,
    make_two_component_signaling_rule,
)
from cell_sim.layer4_regulation.network_loader import (
    DEFAULT_NETWORK_YAML_PATH,
    RegulationNetworkConfig,
    load_regulation_network,
)
from cell_sim.layer4_regulation.pseudo_species import (
    PHOS_STATE_PREFIX,
    PROMOTER_STATE_PREFIX,
    RIBOSWITCH_STATE_PREFIX,
    SIGMA_CLASS_PREFIX,
    TF_BOUND_PREFIX,
    initialize_regulation_state,
)

__all__ = [
    "BiologyNotCurated",
    "DEFAULT_NETWORK_YAML_PATH",
    "PHOS_STATE_PREFIX",
    "PROMOTER_STATE_PREFIX",
    "RIBOSWITCH_STATE_PREFIX",
    "RegulationNetworkConfig",
    "SIGMA_CLASS_PREFIX",
    "TF_BOUND_PREFIX",
    "initialize_regulation_state",
    "load_regulation_network",
    "make_riboswitch_rule",
    "make_sigma_competition_rule",
    "make_tf_binding_rule",
    "make_two_component_signaling_rule",
]
