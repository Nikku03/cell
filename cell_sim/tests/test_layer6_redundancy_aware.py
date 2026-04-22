"""Tests for RedundancyAwareDetector + gene_rule_map extensions +
metabolite sink.

Pure-python tests with synthetic rule / trajectory objects so they
don't require the full simulator. A RealSimulator smoke test at the
bottom is gated on the Luthey-Schulten data being staged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from cell_sim.layer6_essentiality.gene_rule_map import (
    build_gene_to_rules,
    build_metabolite_producers,
    build_rule_products,
)
from cell_sim.layer6_essentiality.harness import (
    FailureMode, Sample, Trajectory,
)
from cell_sim.layer6_essentiality.redundancy_aware_detector import (
    RedundancyAwareDetector,
)


@dataclass
class _FakeRule:
    name: str
    compiled_spec: dict | None = None


def _sample(event_counts):
    return Sample(t_s=0.5, pools={}, event_counts_by_rule=event_counts)


def _traj(event_counts):
    return Trajectory(samples=(_sample(event_counts),))


# ---- gene_rule_map extensions ----


def test_build_metabolite_producers_and_rule_products():
    rules = [
        _FakeRule("catalysis:PGI", {
            "kind": "mm",
            "enzyme_loci": ["G1"],
            "products": [("F6P", 1.0)],
            "substrates": [("G6P", 1.0)],
        }),
        _FakeRule("catalysis:PGI:rev", {
            "kind": "mm",
            "enzyme_loci": ["G1"],
            "products": [("G6P", 1.0)],
            "substrates": [("F6P", 1.0)],
        }),
        _FakeRule("catalysis:PFK:rev", {
            "kind": "mm",
            "enzyme_loci": ["G2"],
            "products": [("F6P", 1.0)],
            "substrates": [("F16BP", 1.0)],
        }),
        _FakeRule("folding:generic", {"kind": "folding"}),
    ]
    producers = build_metabolite_producers(rules)
    assert producers["F6P"] == [
        ("catalysis:PGI", 1.0),
        ("catalysis:PFK:rev", 1.0),
    ]
    assert producers["G6P"] == [("catalysis:PGI:rev", 1.0)]
    # folding rule was correctly skipped
    assert all("folding" not in p[0] for lst in producers.values() for p in lst)

    rule_products = build_rule_products(rules)
    assert rule_products["catalysis:PGI"] == [("F6P", 1.0)]
    assert "folding:generic" not in rule_products


# ---- RedundancyAwareDetector behaviour ----


def _setup(gene_rules, rule_products_spec, wt_events, ko_events):
    """Helper: spin up rule objects matching the spec and a detector."""
    rules = []
    for rn, products in rule_products_spec.items():
        enzyme_loci = [g for g, rs in gene_rules.items() if rn in rs]
        rules.append(_FakeRule(rn, {
            "kind": "mm",
            "enzyme_loci": enzyme_loci,
            "products": products,
            "substrates": [],
        }))
    gene_to_rules = build_gene_to_rules(rules)
    producers = build_metabolite_producers(rules)
    rps = build_rule_products(rules)
    return RedundancyAwareDetector(
        wt=_traj(wt_events),
        gene_to_rules=gene_to_rules,
        metabolite_producers=producers,
        rule_products=rps,
        min_wt_production=20,
        drop_threshold=0.30,
    )


def test_unknown_gene_returns_none():
    d = _setup({"G1": {"catalysis:PGI"}},
               {"catalysis:PGI": [("F6P", 1.0)]},
               {"catalysis:PGI": 100},
               {"catalysis:PGI": 100})
    mode, _, _, ev = d.detect_for_gene("NO_SUCH_GENE", _traj({}))
    assert mode == FailureMode.NONE
    assert ev == "no_catalytic_rules"


def test_unique_producer_drops_production_trips():
    # G1 is the only producer of F6P in WT (PGI), ko sees 0 events.
    # Detector should trip because production collapsed.
    d = _setup(
        {"G1": {"catalysis:PGI"}},
        {"catalysis:PGI": [("F6P", 1.0)]},
        wt_events={"catalysis:PGI": 1000},
        ko_events={"catalysis:PGI": 0},
    )
    mode, _, conf, ev = d.detect_for_gene("G1", _traj({"catalysis:PGI": 0}))
    assert mode == FailureMode.CATALYSIS_SILENCED
    assert conf > 0.9
    assert "production_collapse" in ev
    assert "F6P" in ev


def test_redundant_producer_abstains():
    # G1 catalyses PGI -> F6P. G2 catalyses PFK:rev -> F6P. In WT both
    # fire. In KO of G1, PFK:rev maintains F6P production at same rate.
    # Detector should abstain.
    d = _setup(
        {"G1": {"catalysis:PGI"}, "G2": {"catalysis:PFK:rev"}},
        {
            "catalysis:PGI":     [("F6P", 1.0)],
            "catalysis:PFK:rev": [("F6P", 1.0)],
        },
        wt_events={"catalysis:PGI": 500, "catalysis:PFK:rev": 500},
        ko_events={"catalysis:PGI": 0,   "catalysis:PFK:rev": 1000},
    )
    mode, _, _, ev = d.detect_for_gene(
        "G1", _traj({"catalysis:PGI": 0, "catalysis:PFK:rev": 1000}),
    )
    assert mode == FailureMode.NONE
    assert "alternates_compensate" in ev


def test_redundant_producer_partial_dropout_trips():
    # Alternate producer exists but doesn't pick up the slack.
    # Production dropped to 20 %% of WT -> trip.
    d = _setup(
        {"G1": {"catalysis:PGI"}, "G2": {"catalysis:PFK:rev"}},
        {
            "catalysis:PGI":     [("F6P", 1.0)],
            "catalysis:PFK:rev": [("F6P", 1.0)],
        },
        wt_events={"catalysis:PGI": 500, "catalysis:PFK:rev": 500},
        ko_events={"catalysis:PGI": 0,   "catalysis:PFK:rev": 200},
    )
    mode, _, _, _ = d.detect_for_gene(
        "G1", _traj({"catalysis:PGI": 0, "catalysis:PFK:rev": 200}),
    )
    assert mode == FailureMode.CATALYSIS_SILENCED


def test_wt_under_threshold_abstains():
    # WT production is below min_wt_production -> test is uninformative
    d = _setup(
        {"G1": {"catalysis:RARE"}},
        {"catalysis:RARE": [("X", 1.0)]},
        wt_events={"catalysis:RARE": 5},
        ko_events={"catalysis:RARE": 0},
    )
    mode, _, _, ev = d.detect_for_gene(
        "G1", _traj({"catalysis:RARE": 0}),
    )
    assert mode == FailureMode.NONE
    assert "wt_under_threshold" in ev


def test_drop_threshold_boundary():
    # ratio = 0.30 exactly should NOT trip (strict <)
    d = _setup(
        {"G1": {"catalysis:R"}, "G2": {"catalysis:R_alt"}},
        {
            "catalysis:R":     [("X", 1.0)],
            "catalysis:R_alt": [("X", 1.0)],
        },
        wt_events={"catalysis:R": 100, "catalysis:R_alt": 0},
        ko_events={"catalysis:R": 0,   "catalysis:R_alt": 30},
    )
    mode, _, _, _ = d.detect_for_gene(
        "G1", _traj({"catalysis:R": 0, "catalysis:R_alt": 30}),
    )
    assert mode == FailureMode.NONE, "ratio exactly at 0.30 should abstain"


# ---- sink + real-simulator smoke (gated) ----


GB_PATH = (
    Path(__file__).resolve().parents[1]
    / "data" / "Minimal_Cell_ComplexFormation"
    / "input_data" / "syn3A.gb"
)


@pytest.mark.skipif(
    not GB_PATH.exists(),
    reason="Luthey-Schulten data not staged; see memory_bank/data/STAGING.md",
)
def test_redundancy_aware_real_simulator_smoke():
    """End-to-end: small scale, 2 genes from the reference panel.
    Verifies the detector integrates with the real pipeline without
    asserting on specific MCC numbers - that's what the measurement
    sweep does."""
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(scale_factor=0.02, seed=42))
    sim._ensure_setup()
    wt = sim.run([], t_end_s=0.05, sample_dt_s=0.05)
    assert wt.samples[-1].event_counts_by_rule is not None
    assert len(wt.samples[-1].event_counts_by_rule) > 10

    gene_to_rules = sim.build_gene_to_rules_map()
    assert "JCVISYN3A_0445" in gene_to_rules     # pgi
    all_rules = list(sim._rev_rules or []) + list(sim._extra_rules or [])
    producers = build_metabolite_producers(all_rules)
    rule_products = build_rule_products(all_rules)
    assert len(producers) > 50
    assert any(k.startswith("catalysis:PGI") for v in producers.values()
               for (k, _) in v)


@pytest.mark.skipif(
    not GB_PATH.exists(),
    reason="Luthey-Schulten data not staged",
)
def test_metabolite_sinks_build_from_real_state():
    """SinkConfig helper returns a non-empty rule list for real
    CellState and does not crash."""
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    from cell_sim.layer6_essentiality.metabolite_sink import (
        SinkConfig, make_metabolite_sink_rules,
    )
    sim = RealSimulator(RealSimulatorConfig(
        scale_factor=0.02, seed=42, enable_metabolite_sinks=False,
    ))
    sim._ensure_setup()
    state, _ = sim._build_state_and_rules(())
    sinks = make_metabolite_sink_rules(state, SinkConfig(tolerance=3.0))
    assert len(sinks) > 0
    assert all(r.name.startswith("sink:") for r in sinks)
    # Every sink rule must have a compiled_spec of kind == "sink"
    assert all(r.compiled_spec["kind"] == "sink" for r in sinks)
