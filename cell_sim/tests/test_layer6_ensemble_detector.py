"""Tests for EnsembleDetector + rule-necessity helpers.

All tests use hand-built Sample/Trajectory objects + synthetic
gene_to_rules maps; no full-simulator dependency.
"""
from __future__ import annotations

import pytest

from cell_sim.layer6_essentiality.ensemble_detector import (
    EnsembleDetector, EnsemblePolicy,
)
from cell_sim.layer6_essentiality.gene_rule_map import (
    invert_to_rule_catalysers, unique_rules_per_gene,
)
from cell_sim.layer6_essentiality.harness import (
    FailureMode, Sample, Trajectory,
)
from cell_sim.layer6_essentiality.per_rule_detector import PerRuleDetector
from cell_sim.layer6_essentiality.short_window_detector import (
    ShortWindowDetector,
)


def _traj(pools_per_t, event_counts_per_t) -> Trajectory:
    samples = []
    for i, (t, pools) in enumerate(pools_per_t):
        ec = event_counts_per_t[i] if event_counts_per_t else None
        samples.append(Sample(t_s=t, pools=pools, event_counts_by_rule=ec))
    return Trajectory(samples=tuple(samples))


# ---- rule-necessity helpers ----


def test_invert_to_rule_catalysers_basic():
    m = {
        "G1": {"catalysis:A", "catalysis:B"},
        "G2": {"catalysis:A"},
        "G3": {"catalysis:C"},
    }
    inv = invert_to_rule_catalysers(m)
    assert inv["catalysis:A"] == {"G1", "G2"}
    assert inv["catalysis:B"] == {"G1"}
    assert inv["catalysis:C"] == {"G3"}


def test_unique_rules_per_gene_filters_redundant():
    m = {
        "G1": {"catalysis:A", "catalysis:B"},   # A redundant, B unique
        "G2": {"catalysis:A"},                  # A redundant
        "G3": {"catalysis:C"},                  # C unique
    }
    u = unique_rules_per_gene(m)
    assert u["G1"] == {"catalysis:B"}
    assert "G2" not in u  # every rule G2 had is redundant
    assert u["G3"] == {"catalysis:C"}


# ---- ensemble policies ----


def _make_detectors(wt_traj, gene_to_rules, pools_list=None):
    pr = PerRuleDetector(wt=wt_traj, gene_to_rules=gene_to_rules,
                         min_wt_events=20)
    sw = ShortWindowDetector(wt=wt_traj, deviation_threshold=0.10)
    return pr, sw


_WT_SAMPLES = [
    (0.0, {"ATP": 100, "F6P": 100}),
    (0.1, {"ATP": 100, "F6P": 100}),
    (0.2, {"ATP": 100, "F6P": 100}),
    (0.3, {"ATP": 100, "F6P": 100}),
    (0.4, {"ATP": 100, "F6P": 100}),
    (0.5, {"ATP": 100, "F6P": 100}),
]
_WT_EVENTS = [
    {"catalysis:PGI": 1500},
    {"catalysis:PGI": 3000},
    {"catalysis:PGI": 4500},
    {"catalysis:PGI": 6000},
    {"catalysis:PGI": 7500},
    {"catalysis:PGI": 8000},
]

GENE_TO_RULES = {"JCVISYN3A_0445": {"catalysis:PGI"}}


def test_ensemble_per_rule_with_pool_confirm_fires_when_both_agree():
    wt = _traj(_WT_SAMPLES, _WT_EVENTS)
    # KO silences PGI and drops F6P to 80 - above 0.02 min_pool_dev
    ko_samples = [
        (0.0, {"ATP": 100, "F6P": 100}),
        (0.1, {"ATP": 100, "F6P":  95}),
        (0.2, {"ATP": 100, "F6P":  90}),
        (0.3, {"ATP": 100, "F6P":  85}),
        (0.4, {"ATP": 100, "F6P":  80}),
        (0.5, {"ATP": 100, "F6P":  80}),
    ]
    ko_events = [{"catalysis:PGI": 0}] * 6
    ko = _traj(ko_samples, ko_events)

    pr, sw = _make_detectors(wt, GENE_TO_RULES)
    ens = EnsembleDetector(per_rule=pr, short_window=sw,
                           policy=EnsemblePolicy.PER_RULE_WITH_POOL_CONFIRM,
                           min_pool_dev=0.02)
    mode, _, conf, ev = ens.detect_for_gene("JCVISYN3A_0445", ko)
    assert mode == FailureMode.CATALYSIS_SILENCED
    assert conf > 0
    assert "pool_confirm" in ev


def test_ensemble_per_rule_with_pool_confirm_abstains_when_pools_flat():
    """The v5 FP case: catalysis rule goes silent but pools don't move,
    because the simulator lacks the biological redundancy that would
    otherwise mask the KO."""
    wt = _traj(_WT_SAMPLES, _WT_EVENTS)
    # Pools identical to WT; events silenced.
    ko_samples = list(_WT_SAMPLES)  # same pool trajectory
    ko_events = [{"catalysis:PGI": 0}] * 6
    ko = _traj(ko_samples, ko_events)

    pr, sw = _make_detectors(wt, GENE_TO_RULES)
    ens = EnsembleDetector(per_rule=pr, short_window=sw,
                           policy=EnsemblePolicy.PER_RULE_WITH_POOL_CONFIRM,
                           min_pool_dev=0.02)
    mode, _, _, ev = ens.detect_for_gene("JCVISYN3A_0445", ko)
    assert mode == FailureMode.NONE
    assert "no_pool_confirm" in ev


def test_ensemble_per_rule_with_pool_confirm_abstains_when_pr_abstains():
    wt = _traj(_WT_SAMPLES, _WT_EVENTS)
    # Gene has no catalytic rules -> PerRule returns no_catalytic_rules
    ko = _traj(_WT_SAMPLES, _WT_EVENTS)
    pr, sw = _make_detectors(wt, {})
    ens = EnsembleDetector(per_rule=pr, short_window=sw,
                           policy=EnsemblePolicy.PER_RULE_WITH_POOL_CONFIRM)
    mode, _, _, ev = ens.detect_for_gene("JCVISYN3A_0445", ko)
    assert mode == FailureMode.NONE
    assert "pr_abstain" in ev


def test_ensemble_AND_requires_both():
    wt = _traj(_WT_SAMPLES, _WT_EVENTS)
    ko_samples = [
        (0.0, {"ATP": 100, "F6P": 100}),
        (0.1, {"ATP":  40, "F6P":  40}),
        (0.2, {"ATP":  30, "F6P":  30}),
        (0.3, {"ATP":  30, "F6P":  30}),
        (0.4, {"ATP":  30, "F6P":  30}),
        (0.5, {"ATP":  30, "F6P":  30}),
    ]
    ko_events = [{"catalysis:PGI": 0}] * 6
    ko = _traj(ko_samples, ko_events)

    pr, sw = _make_detectors(wt, GENE_TO_RULES)
    ens = EnsembleDetector(per_rule=pr, short_window=sw,
                           policy=EnsemblePolicy.AND)
    mode, _, _, ev = ens.detect_for_gene("JCVISYN3A_0445", ko)
    assert mode == FailureMode.CATALYSIS_SILENCED
    assert ev.startswith("AND[")


def test_ensemble_AND_refuses_when_only_one_fires():
    wt = _traj(_WT_SAMPLES, _WT_EVENTS)
    # Events silenced but pools unchanged - PerRule fires, ShortWindow doesn't
    ko = _traj(_WT_SAMPLES, [{"catalysis:PGI": 0}] * 6)
    pr, sw = _make_detectors(wt, GENE_TO_RULES)
    ens = EnsembleDetector(per_rule=pr, short_window=sw,
                           policy=EnsemblePolicy.AND)
    mode, _, _, ev = ens.detect_for_gene("JCVISYN3A_0445", ko)
    assert mode == FailureMode.NONE
    assert "no_agreement" in ev
