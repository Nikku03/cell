"""Tests for gene_rule_map + PerRuleDetector.

The tests use synthetic rule-like objects and synthetic Sample
trajectories so they don't need the real simulator. A RealSimulator
smoke test at the end is gated on the Luthey-Schulten data being
staged and just confirms the gene->rules map has the expected shape
for pgi/ptsG/rpsR.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from cell_sim.layer6_essentiality.gene_rule_map import (
    build_gene_to_rules, summarise,
)
from cell_sim.layer6_essentiality.harness import (
    FailureMode, Sample, Trajectory,
)
from cell_sim.layer6_essentiality.per_rule_detector import PerRuleDetector


# ---- gene_rule_map ----


@dataclass
class _FakeRule:
    name: str
    compiled_spec: dict | None = None


def test_build_gene_to_rules_basic():
    rules = [
        _FakeRule("catalysis:PGI", {"kind": "mm",
                                    "enzyme_loci": ["JCVISYN3A_0445"]}),
        _FakeRule("catalysis:PGI:rev", {"kind": "mm",
                                        "enzyme_loci": ["JCVISYN3A_0445"]}),
        _FakeRule("catalysis:GLCpts", {"kind": "mm",
                                       "enzyme_loci": ["JCVISYN3A_0779"]}),
    ]
    m = build_gene_to_rules(rules)
    assert m["JCVISYN3A_0445"] == {"catalysis:PGI", "catalysis:PGI:rev"}
    assert m["JCVISYN3A_0779"] == {"catalysis:GLCpts"}


def test_build_gene_to_rules_skips_non_mm():
    rules = [
        _FakeRule("catalysis:PGI", {"kind": "mm",
                                    "enzyme_loci": ["JCVISYN3A_0445"]}),
        _FakeRule("folding:generic", {"kind": "folding",
                                      "enzyme_loci": ["JCVISYN3A_0445"]}),
        _FakeRule("complex_formation:ribosome",
                  {"kind": "complex_formation", "enzyme_loci": []}),
        _FakeRule("unspecced", None),
    ]
    m = build_gene_to_rules(rules)
    assert m == {"JCVISYN3A_0445": {"catalysis:PGI"}}


def test_build_gene_to_rules_multi_enzyme_reaction():
    """A reaction catalysed by two genes should appear under both."""
    rules = [
        _FakeRule("catalysis:RXN_X", {
            "kind": "mm", "enzyme_loci": ["JCVISYN3A_0001",
                                          "JCVISYN3A_0002"],
        }),
    ]
    m = build_gene_to_rules(rules)
    assert m["JCVISYN3A_0001"] == {"catalysis:RXN_X"}
    assert m["JCVISYN3A_0002"] == {"catalysis:RXN_X"}


def test_summarise_empty_and_nonempty():
    assert summarise({})["genes_with_rules"] == 0
    m = {"a": {"r1", "r2"}, "b": {"r3"}}
    s = summarise(m)
    assert s["genes_with_rules"] == 2
    assert s["min_rules"] == 1
    assert s["max_rules"] == 2
    assert s["avg_rules"] == 1.5
    assert s["total_rule_pairs"] == 3


# ---- PerRuleDetector ----


def _traj(last_event_counts: dict[str, int], t: float = 0.5) -> Trajectory:
    s = Sample(t_s=t, pools={}, event_counts_by_rule=last_event_counts)
    return Trajectory(samples=(s,))


def test_detector_unknown_gene_returns_none():
    wt = _traj({"catalysis:PGI": 7888})
    ko = _traj({"catalysis:PGI": 7900})
    d = PerRuleDetector(wt=wt, gene_to_rules={"JCVISYN3A_0445": {"catalysis:PGI"}})
    mode, t, conf, ev = d.detect_for_gene("JCVISYN3A_0999", ko)
    assert mode == FailureMode.NONE
    assert ev == "no_catalytic_rules"


def test_detector_trips_on_full_silence():
    wt = _traj({"catalysis:PGI": 7888, "catalysis:PGI:rev": 450})
    ko = _traj({"catalysis:O2t": 12345})  # pgi rules absent entirely
    d = PerRuleDetector(
        wt=wt,
        gene_to_rules={"JCVISYN3A_0445":
                       {"catalysis:PGI", "catalysis:PGI:rev"}},
        min_wt_events=20,
    )
    mode, t, conf, ev = d.detect_for_gene("JCVISYN3A_0445", ko)
    assert mode == FailureMode.CATALYSIS_SILENCED
    assert conf == 1.0
    assert "silenced" in ev
    assert "catalysis:PGI" in ev


def test_detector_refuses_when_wt_under_threshold():
    """If WT itself didn't exercise the rule enough, abstain rather
    than call it essential off a zero-vs-zero."""
    wt = _traj({"catalysis:RARE": 5})
    ko = _traj({})
    d = PerRuleDetector(
        wt=wt,
        gene_to_rules={"JCVISYN3A_0999": {"catalysis:RARE"}},
        min_wt_events=20,
    )
    mode, _, _, ev = d.detect_for_gene("JCVISYN3A_0999", ko)
    assert mode == FailureMode.NONE
    assert "wt_under_threshold" in ev


def test_detector_refuses_on_partial_silence():
    """Multi-enzyme reactions where some rules still fire in KO must
    not trip - that means the KO isn't the sole catalyst."""
    wt = _traj({"catalysis:A": 500, "catalysis:B": 500})
    ko = _traj({"catalysis:A": 0, "catalysis:B": 480})
    d = PerRuleDetector(
        wt=wt,
        gene_to_rules={"JCVISYN3A_0001": {"catalysis:A", "catalysis:B"}},
    )
    mode, _, _, ev = d.detect_for_gene("JCVISYN3A_0001", ko)
    assert mode == FailureMode.NONE
    assert "partial_silence" in ev


# ---- real-simulator smoke (gated) ----


GB_PATH = (
    Path(__file__).resolve().parents[1]
    / "data" / "Minimal_Cell_ComplexFormation"
    / "input_data" / "syn3A.gb"
)


@pytest.mark.skipif(
    not GB_PATH.exists(),
    reason="Luthey-Schulten data not staged; see memory_bank/data/STAGING.md",
)
def test_gene_to_rules_from_real_simulator_has_pgi_and_ptsG():
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(scale_factor=0.02, seed=42))
    m = sim.build_gene_to_rules_map()
    assert "JCVISYN3A_0445" in m          # pgi
    assert "JCVISYN3A_0779" in m          # ptsG
    assert "JCVISYN3A_0025" not in m      # rpsR (ribosomal, not catalytic)
    assert "JCVISYN3A_0199" not in m      # rpmI (ribosomal)
    assert "catalysis:PGI" in m["JCVISYN3A_0445"]
    assert "catalysis:GLCpts" in m["JCVISYN3A_0779"]
