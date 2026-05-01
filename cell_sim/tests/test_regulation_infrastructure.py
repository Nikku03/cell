"""Phase R1 regulation-layer infrastructure tests.

These tests verify the architectural contract of
``cell_sim/layer4_regulation/`` without claiming any biology. The
pattern follows the project convention of treating Phase R1 as
infrastructure-only:

* Rule factories must import cleanly and produce
  architecturally-valid ``TransitionRule`` objects when called with
  explicit dummy parameters.
* The ``BiologyNotCurated`` warning must fire when factories are
  called without ``source_citation`` and must be suppressible via
  ``warnings.catch_warnings``.
* The state initializer must run on an empty config and emit no
  pseudo-species — i.e. regulation-disabled is the default Phase R1
  behavior.
* The empty ``regulation_network_syn3a.yaml`` shipped with the repo
  must parse cleanly and produce zero entries.
* Importing ``cell_sim.layer4_regulation`` must NOT change the rules
  the production sweep builds. The v15 baseline is unaffected.
* The Phase 2 gex-off bit-identity regression
  (``test_phase2_gex_off_bit_identity``) must still pass after this
  layer lands. That test is a separate file; this one covers
  Phase R1 specifically.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

# Layer 2 / 3 modules use unprefixed imports + sys.path tricks. Mirror
# that pattern so importing the regulation layer (which depends on
# layer2_field.dynamics for the TransitionRule type) works without a
# top-level pyproject.
_CELL_SIM = Path(__file__).resolve().parents[1]
if str(_CELL_SIM) not in sys.path:
    sys.path.insert(0, str(_CELL_SIM))


# ---------------------------------------------------------------------
# 1. Imports clean
# ---------------------------------------------------------------------


def test_regulation_module_imports_cleanly():
    """All four rule factories + BiologyNotCurated + state init are
    importable through the ``cell_sim.layer4_regulation`` package
    without ImportError or side effects on stdout."""
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from cell_sim.layer4_regulation import (  # noqa: F401
            BiologyNotCurated,
            initialize_regulation_state,
            load_regulation_network,
            make_riboswitch_rule,
            make_sigma_competition_rule,
            make_tf_binding_rule,
            make_two_component_signaling_rule,
        )
    # Importing the package must not print anything. The [INFO]
    # regulation-disabled message belongs to initialize_regulation_state,
    # not to import time.
    assert buf.getvalue() == "", (
        f"importing cell_sim.layer4_regulation produced unexpected "
        f"stdout: {buf.getvalue()!r}"
    )


# ---------------------------------------------------------------------
# 2. Factories produce architecturally-valid rules with warning suppressed
# ---------------------------------------------------------------------


def _is_valid_transition_rule(obj) -> bool:
    """Duck-type the TransitionRule contract."""
    from layer2_field.dynamics import TransitionRule
    if not isinstance(obj, TransitionRule):
        return False
    if not obj.name:
        return False
    if not isinstance(obj.participants, list):
        return False
    if obj.rate < 0:
        return False
    if obj.can_fire is None or obj.apply is None:
        return False
    return True


def test_tf_binding_rule_is_architecturally_valid():
    from cell_sim.layer4_regulation import (
        BiologyNotCurated, make_tf_binding_rule,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiologyNotCurated)
        rule = make_tf_binding_rule(
            "JCVISYN3A_TEST_TF", "JCVISYN3A_TEST_TARGET",
            k_bind=1.0, k_unbind=0.5,
        )
    assert _is_valid_transition_rule(rule)
    assert rule.name == "tf_binding:JCVISYN3A_TEST_TF->JCVISYN3A_TEST_TARGET"
    assert rule.rate_source == "placeholder_phase_r1"


def test_sigma_competition_rule_is_architecturally_valid():
    from cell_sim.layer4_regulation import (
        BiologyNotCurated, make_sigma_competition_rule,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiologyNotCurated)
        rule = make_sigma_competition_rule(
            "JCVISYN3A_TEST_SIG", "housekeeping", competition_factor=0.7,
        )
    assert _is_valid_transition_rule(rule)
    assert "sigma_competition" in rule.name


def test_riboswitch_rule_is_architecturally_valid():
    from cell_sim.layer4_regulation import (
        BiologyNotCurated, make_riboswitch_rule,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiologyNotCurated)
        rule = make_riboswitch_rule(
            "JCVISYN3A_TEST_GENE", "M_test_c",
            k_fold=0.1, k_unfold=0.05,
        )
    assert _is_valid_transition_rule(rule)
    assert "M_test_c" in rule.name


def test_two_component_signaling_rule_is_architecturally_valid():
    from cell_sim.layer4_regulation import (
        BiologyNotCurated, make_two_component_signaling_rule,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiologyNotCurated)
        rule = make_two_component_signaling_rule(
            "JCVISYN3A_TEST_SK", "JCVISYN3A_TEST_RR",
            k_phos=1.0, k_dephos=0.2,
        )
    assert _is_valid_transition_rule(rule)
    assert "two_component" in rule.name


# ---------------------------------------------------------------------
# 3. BiologyNotCurated fires when source_citation is missing
# ---------------------------------------------------------------------


def test_biology_not_curated_fires_without_source():
    from cell_sim.layer4_regulation import (
        BiologyNotCurated, make_tf_binding_rule,
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        make_tf_binding_rule(
            "JCVISYN3A_x", "JCVISYN3A_y",
            k_bind=1.0, k_unbind=1.0,
        )
    assert any(
        issubclass(w.category, BiologyNotCurated) for w in recorded
    ), "BiologyNotCurated should fire when source_citation is missing"


def test_biology_not_curated_silent_with_source():
    from cell_sim.layer4_regulation import (
        BiologyNotCurated, make_tf_binding_rule,
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        make_tf_binding_rule(
            "JCVISYN3A_x", "JCVISYN3A_y",
            k_bind=1.0, k_unbind=1.0,
            source_citation="phase_r1_test_dummy",
        )
    assert not any(
        issubclass(w.category, BiologyNotCurated) for w in recorded
    ), "BiologyNotCurated must NOT fire when source_citation is set"


# ---------------------------------------------------------------------
# 4. initialize_regulation_state is a no-op on empty config
# ---------------------------------------------------------------------


def _minimal_state():
    """A CellState that does not require the full Syn3A spec to
    construct. We only need the regulation_pseudo_species attribute,
    so a bare object with an events list works."""
    class _Stub:
        def __init__(self):
            self.events = []
            self.time = 0.0
            self.regulation_pseudo_species = {}

        def log_event(self, *args, **kwargs):
            pass

    return _Stub()


def test_initialize_regulation_state_with_none_config_is_noop(capsys):
    from cell_sim.layer4_regulation import initialize_regulation_state

    state = _minimal_state()
    summary = initialize_regulation_state(state, network_config=None)

    assert summary["active"] is False
    assert summary["n_tf_bound"] == 0
    assert summary["n_promoter"] == 0
    assert summary["n_sigma"] == 0
    assert summary["n_riboswitch"] == 0
    assert summary["n_phos"] == 0
    assert state.regulation_pseudo_species == {}

    captured = capsys.readouterr()
    assert "[INFO]" in captured.out
    assert "Regulation network configuration is empty" in captured.out


def test_initialize_regulation_state_with_empty_config_is_noop(capsys):
    from cell_sim.layer4_regulation import (
        RegulationNetworkConfig, initialize_regulation_state,
    )

    state = _minimal_state()
    cfg = RegulationNetworkConfig()  # all four lists default empty
    summary = initialize_regulation_state(state, network_config=cfg)

    assert summary["active"] is False
    assert state.regulation_pseudo_species == {}

    captured = capsys.readouterr()
    assert "[INFO]" in captured.out


# ---------------------------------------------------------------------
# 5. Empty YAML config loads without errors
# ---------------------------------------------------------------------


def test_default_yaml_exists_and_loads():
    """The production YAML must exist, parse cleanly through the
    Phase R1 loader, and reflect the current curation state.

    Phase R1 shipped this file empty (all four lists ``[]``).
    Phase R2b promoted 2 entries: 1 sigma factor (rpoD /
    JCVISYN3A_0407) and 1 transcription factor (mraZ /
    JCVISYN3A_0525); both at confidence: inferred. Riboswitches
    and two-component systems remain empty pending future
    curation. See memory_bank/facts/structural/
    regulation_curation_phase_r2b.json for the audit trail.
    """
    from cell_sim.layer4_regulation import (
        DEFAULT_NETWORK_YAML_PATH, load_regulation_network,
    )

    assert DEFAULT_NETWORK_YAML_PATH.exists(), (
        f"Production regulation YAML missing at {DEFAULT_NETWORK_YAML_PATH}."
    )
    cfg = load_regulation_network()

    # Sigma: exactly the rpoD entry from R2b, confidence inferred.
    assert len(cfg.sigma_factors) == 1, (
        f"expected 1 sigma factor (rpoD), got {len(cfg.sigma_factors)}"
    )
    sigma = cfg.sigma_factors[0]
    assert sigma["sigma_locus"] == "JCVISYN3A_0407"
    assert sigma["gene_class"] == "housekeeping"
    assert sigma["confidence"] == "inferred"
    assert sigma["curated_on"] == "2026-04-30" or sigma["curated_on"]

    # TF: exactly the mraZ entry from R2b, confidence inferred,
    # target_genes intentionally empty (target inference deferred).
    assert len(cfg.transcription_factors) == 1, (
        f"expected 1 TF (mraZ), got {len(cfg.transcription_factors)}"
    )
    tf = cfg.transcription_factors[0]
    assert tf["tf_locus"] == "JCVISYN3A_0525"
    assert tf["target_genes"] == []
    assert tf["confidence"] == "inferred"

    # Riboswitches and two-component systems remain empty.
    assert cfg.riboswitches == []
    assert cfg.two_component_systems == []


def test_loader_rejects_entry_missing_provenance(tmp_path):
    """A populated entry without source / confidence / curated_on
    must raise RegulationConfigError. This is the safety check that
    Phase R2 cannot wire in TF binding without a paper trail."""
    from cell_sim.layer4_regulation.network_loader import (
        RegulationConfigError, load_regulation_network,
    )

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "transcription_factors:\n"
        "  - tf_locus: JCVISYN3A_x\n"
        "    target_genes: [JCVISYN3A_y]\n"
        "    k_bind: 1.0\n"
        "    k_unbind: 1.0\n"
        # source / confidence / curated_on intentionally absent
        "sigma_factors: []\n"
        "riboswitches: []\n"
        "two_component_systems: []\n"
    )
    with pytest.raises(RegulationConfigError):
        load_regulation_network(bad_yaml)


# ---------------------------------------------------------------------
# 6. Production sweep does NOT invoke regulation
# ---------------------------------------------------------------------


def test_real_simulator_does_not_reference_regulation_layer():
    """real_simulator.py must contain zero references to layer4
    regulation. Phase R1 is infrastructure-only; integration is
    Phase R2 work."""
    real_sim = (
        Path(__file__).resolve().parents[1]
        / "layer6_essentiality" / "real_simulator.py"
    )
    text = real_sim.read_text()
    assert "layer4_regulation" not in text, (
        "real_simulator.py references layer4_regulation. Phase R1 "
        "should NOT integrate regulation into the production sweep — "
        "see the Phase R2 entry in FUTURE_WORK.md."
    )
    assert "regulation_pseudo_species" not in text
    assert "BiologyNotCurated" not in text


def test_importing_regulation_does_not_register_global_rules():
    """Importing the regulation package must not add any side-effect
    rules to the gene-expression rule builder or the real_simulator
    rule list. We verify by snapshotting the public symbols of the
    closest neighbors before vs after import."""
    import cell_sim.layer3_reactions.gene_expression as gex

    before = set(dir(gex))
    import cell_sim.layer4_regulation  # noqa: F401
    after = set(dir(gex))
    assert before == after, (
        f"importing cell_sim.layer4_regulation mutated "
        f"cell_sim.layer3_reactions.gene_expression: "
        f"added={after - before} removed={before - after}"
    )
