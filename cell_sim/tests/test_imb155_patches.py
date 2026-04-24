"""Tests for the iMB155 pathway patches (Session 15 item 1).

The patch clears three over-assigned loci from catalysis rules so the
PerRuleDetector returns ``no_catalytic_rules`` for their KOs. Tests
exercise:

1. ``apply_imb155_patches`` leaves non-catalysis rules and non-patched
   catalysis rules untouched.
2. Rules gated SOLELY by a patched locus lose their ``compiled_spec``
   entirely (become python-closure rules) so ``build_gene_to_rules``
   excludes them.
3. Rules gated by a patched locus PLUS an alternate enzyme keep their
   compiled_spec but drop the patched locus from ``enzyme_loci``.
4. ``count_patched_rules`` returns the expected multiplicities against
   the real Syn3A rule set.
5. End-to-end: running the real simulator with
   ``enable_imb155_patches=True`` empties
   ``gene_to_rules`` for the three patched loci.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cell_sim.layer3_reactions.imb155_patches import (
    PATCHED_LOCI,
    apply_imb155_patches,
    count_patched_rules,
)


def _make_mm_rule(name: str, loci: list[str], *,
                  kcat: float = 50.0) -> object:
    """Minimal fake rule carrying a compiled_spec['kind']='mm' dict —
    the only bit the patch inspects."""
    from cell_sim.layer2_field.dynamics import TransitionRule
    r = TransitionRule(
        name=name, participants=["M_atp_c"], rate=kcat,
        rate_source="test", can_fire=None, apply=None,
    )
    r.compiled_spec = {
        "kind": "mm",
        "rxn_name": name,
        "direction": "fwd",
        "substrates": [("M_atp_c", 1.0)],
        "products": [("M_adp_c", 1.0)],
        "enzyme_loci": list(loci),
        "kcat": float(kcat),
        "Km": {"M_atp_c": 0.1},
        "include_saturation": True,
        "substrate_ids_for_sat": ["M_atp_c"],
    }
    return r


def _make_non_mm_rule(name: str) -> object:
    """Python-closure rule without compiled_spec — the kind the patch
    passes through untouched (folding, complex formation, etc.)."""
    from cell_sim.layer2_field.dynamics import TransitionRule
    return TransitionRule(
        name=name, participants=[], rate=1.0,
        rate_source="test", can_fire=None, apply=None,
    )


def test_patched_loci_are_the_expected_three():
    assert PATCHED_LOCI == frozenset({
        "JCVISYN3A_0034",
        "JCVISYN3A_0228",
        "JCVISYN3A_0732",
    })


def test_apply_imb155_patches_passes_through_non_mm_rules():
    folding = _make_non_mm_rule("folding:generic")
    complex_form = _make_non_mm_rule("complex:ribosome")
    patched = apply_imb155_patches([folding, complex_form])
    assert patched[0] is folding
    assert patched[1] is complex_form


def test_apply_imb155_patches_passes_through_unrelated_catalysis():
    """A catalysis rule whose enzyme_loci contains NONE of the
    patched loci should survive unchanged (same object reference)."""
    rule = _make_mm_rule("catalysis:PGI", loci=["JCVISYN3A_0445"])
    out = apply_imb155_patches([rule])
    assert out == [rule]
    assert out[0].compiled_spec is rule.compiled_spec


def test_apply_imb155_patches_drops_patched_locus_with_alternate():
    """A rule gated by both a patched locus AND a real enzyme should
    keep its compiled_spec but drop the patched locus from
    enzyme_loci."""
    rule = _make_mm_rule(
        "catalysis:PDH_E3",
        loci=["JCVISYN3A_0228", "JCVISYN3A_9999"],   # hypothetical alt
    )
    out = apply_imb155_patches([rule])
    assert len(out) == 1
    new = out[0]
    # compiled_spec preserved (so the rule stays compiled-MM)
    assert new.compiled_spec is not None
    assert new.compiled_spec.get("kind") == "mm"
    assert new.compiled_spec["enzyme_loci"] == ["JCVISYN3A_9999"]
    # The patched locus is gone from this gene's rule set
    assert "JCVISYN3A_0228" not in new.compiled_spec["enzyme_loci"]


def test_apply_imb155_patches_replaces_sole_patched_rule():
    """A rule gated SOLELY by a patched locus becomes a python-
    closure rule (no compiled_spec), so build_gene_to_rules skips it
    entirely — closing the PerRuleDetector FP."""
    rule = _make_mm_rule("catalysis:CHOLt", loci=["JCVISYN3A_0034"])
    out = apply_imb155_patches([rule])
    assert len(out) == 1
    new = out[0]
    # Name + rate carried over
    assert new.name == "catalysis:CHOLt"
    assert new.rate == pytest.approx(50.0)
    assert new.rate_source == "imb155_patch"
    # compiled_spec gone -> build_gene_to_rules will skip this rule
    assert getattr(new, "compiled_spec", None) is None
    # Custom can_fire / apply are now python closures
    assert callable(new.can_fire)
    assert callable(new.apply)


def test_count_patched_rules_on_synthetic_set():
    """count_patched_rules should tally each patched locus's gating
    count before the patch is applied."""
    rules = [
        _make_mm_rule("catalysis:CHOLt", loci=["JCVISYN3A_0034"]),
        _make_mm_rule("catalysis:CHOLt:rev", loci=["JCVISYN3A_0034"]),
        _make_mm_rule("catalysis:DRPA", loci=["JCVISYN3A_0732"]),
        _make_mm_rule("catalysis:DRPA:rev", loci=["JCVISYN3A_0732"]),
        _make_mm_rule("catalysis:PDH_E3", loci=["JCVISYN3A_0228"]),
        _make_mm_rule("catalysis:PGI", loci=["JCVISYN3A_0445"]),  # not patched
        _make_non_mm_rule("folding:generic"),                      # not MM
    ]
    counts = count_patched_rules(rules)
    assert counts["JCVISYN3A_0034"] == 2
    assert counts["JCVISYN3A_0732"] == 2
    assert counts["JCVISYN3A_0228"] == 1


def test_apply_imb155_patches_is_pure():
    """The input list and its rules' compiled_spec dicts must not be
    mutated. The patch returns a new list with new rule objects for
    the patched entries; unrelated rules keep their original
    identity."""
    original = _make_mm_rule(
        "catalysis:CHOLt", loci=["JCVISYN3A_0034"],
    )
    original_enzyme_loci_before = list(
        original.compiled_spec["enzyme_loci"]
    )
    _ = apply_imb155_patches([original])
    assert (
        original.compiled_spec["enzyme_loci"]
        == original_enzyme_loci_before
    )


# ---------------------------------------------------------------------
# Integration test against the REAL rule set (imports the live SBML,
# kinetics, and nutrient_uptake module). Runs in ~2-3s because it
# only assembles rules; no simulation is executed.
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_rev_extra_rules():
    import contextlib
    import io
    import sys as _sys
    cell_sim = Path(__file__).resolve().parents[1]
    if str(cell_sim) not in _sys.path:
        _sys.path.insert(0, str(cell_sim))
    from cell_sim.layer3_reactions.sbml_parser import parse_sbml
    from cell_sim.layer3_reactions.kinetics import load_all_kinetics
    from cell_sim.layer3_reactions.reversible import (
        build_reversible_catalysis_rules,
    )
    from cell_sim.layer3_reactions.nutrient_uptake import (
        build_missing_transport_rules,
    )
    sbml_path = (
        cell_sim / "data" / "Minimal_Cell_ComplexFormation"
        / "input_data" / "Syn3A_updated.xml"
    )
    with contextlib.redirect_stdout(io.StringIO()):
        sbml = parse_sbml(sbml_path)
        kinetics = load_all_kinetics()
        rev_rules, _ = build_reversible_catalysis_rules(sbml, kinetics)
        extra_rules = build_missing_transport_rules(sbml, kinetics)
    return rev_rules + extra_rules


def test_real_rules_have_expected_patched_locus_counts(real_rev_extra_rules):
    """Sanity: on the real rule set, JCVISYN3A_0034 should gate
    exactly the 12 placeholder-locus reactions (8 SBML transport
    reactions + 4 synthetic pseudo-reactions + their reverses for
    reversible ones), 0228 should gate PDH_E3 (fwd + rev = 2), and
    0732 should gate DRPA (fwd + rev = 2)."""
    c = count_patched_rules(real_rev_extra_rules)
    # 0034 over-assignment (placeholder): 4 reversible transporters
    # (GLYCt, O2t, CHOLt, TAGt) + 4 irreversible pseudo-reactions
    # (ADEt/GUAt/URAt/CYTDt) = 4*2 + 4 = 12 rules.
    assert c["JCVISYN3A_0034"] == 12, (
        f"unexpected placeholder count: {c['JCVISYN3A_0034']}"
    )
    assert c["JCVISYN3A_0228"] == 2   # PDH_E3 fwd + rev
    assert c["JCVISYN3A_0732"] == 2   # DRPA fwd + rev


def test_real_rules_after_patch_empty_patched_loci(real_rev_extra_rules):
    """After patch, the three loci gate ZERO compiled-MM rules."""
    patched = apply_imb155_patches(real_rev_extra_rules)
    c = count_patched_rules(patched)
    assert c["JCVISYN3A_0034"] == 0
    assert c["JCVISYN3A_0228"] == 0
    assert c["JCVISYN3A_0732"] == 0


def test_real_rules_after_patch_gene_to_rules_drops_patched_loci(
    real_rev_extra_rules,
):
    """The downstream consumer (PerRuleDetector) reads from
    build_gene_to_rules. After the patch, the three FPs should have
    no entry there."""
    from cell_sim.layer6_essentiality.gene_rule_map import (
        build_gene_to_rules,
    )
    patched = apply_imb155_patches(real_rev_extra_rules)
    mapping = build_gene_to_rules(patched)
    assert "JCVISYN3A_0034" not in mapping
    assert "JCVISYN3A_0228" not in mapping
    assert "JCVISYN3A_0732" not in mapping


def test_real_simulator_config_flag_threads_through():
    """RealSimulatorConfig exposes enable_imb155_patches, default
    False. When set to True, the patched rule list backs the sim."""
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    cfg_off = RealSimulatorConfig()
    assert cfg_off.enable_imb155_patches is False

    cfg_on = RealSimulatorConfig(enable_imb155_patches=True)
    sim = RealSimulator(cfg_on)
    sim._ensure_setup()
    mapping = sim.build_gene_to_rules_map()
    assert "JCVISYN3A_0034" not in mapping
    assert "JCVISYN3A_0228" not in mapping
    assert "JCVISYN3A_0732" not in mapping
    # Sanity: a known MM gene (pgi) should still be there.
    assert "JCVISYN3A_0445" in mapping
