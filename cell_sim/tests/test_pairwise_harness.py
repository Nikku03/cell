"""Session 26: tests for KnockoutHarness.predict_pair (pairwise knockouts).

Covers the three viability invariants in the synthetic-lethality
research spec, plus two property-style smoke tests:

  Invariant 1 — predict_pair(X, X) is identical to predict(X) on a
  representative panel of genes (essential and non-essential).
  Invariant 2 — predict_pair(A, B) and predict_pair(B, A) yield the
  same essential/non-essential prediction (and same failure mode).
  Smoke test — predict_pair(X, Y) where both X and Y individually
  trigger essentiality must itself be essential (a sanity check
  that joint knockouts can't accidentally clear a single-knockout
  failure).

The tests use ``MockSimulator`` and synthetic ``Trajectory``
fixtures so the heavy Layer-3 metabolic stack does NOT need to
boot. The ``MockSimulator``'s knockout-routing logic
(``responses[first-knocked-out-gene]``) is sufficient to exercise
both single and pairwise paths deterministically.

The third Session-26 invariant (v15 single-knockout reproducibility
on the bench_rust_vs_python 20-gene sample) is verified by the
run script ``scripts/run_synthlet_pilot.py`` rather than here,
because it requires the production simulator stack.
"""
from __future__ import annotations

import pytest

from cell_sim.layer6_essentiality.harness import (
    FailureMode,
    KnockoutHarness,
    MockSimulator,
    PairPrediction,
    Prediction,
    Sample,
    Trajectory,
)


# ---------------------------------------------------------------------
# Trajectory fixtures.
# ---------------------------------------------------------------------


def _flat_trajectory(value: float = 1000.0) -> Trajectory:
    """Constant-pool trajectory; will not trip any failure mode."""
    pools = {
        "ATP": value,
        "CHARGED_TRNA_FRACTION": value,
        "NTP_TOTAL": value,
        # essential metabolites carried at the same level
        "G6P": value, "F6P": value, "PYR": value,
        "dATP": value, "dGTP": value, "dCTP": value, "dTTP": value,
        "CTP": value, "GTP": value, "UTP": value,
        "NADH": value, "NAD": value,
    }
    samples = tuple(
        Sample(t_s=t, pools=dict(pools))
        for t in (0.0, 1.0, 2.0, 3.0, 4.0)
    )
    return Trajectory(samples=samples)


def _atp_collapse_trajectory() -> Trajectory:
    """ATP drops to 10 % of the WT value at t=1 s and stays there."""
    base_pools = {
        "ATP": 1000.0,
        "CHARGED_TRNA_FRACTION": 1000.0,
        "NTP_TOTAL": 1000.0,
        "G6P": 1000.0, "F6P": 1000.0, "PYR": 1000.0,
        "dATP": 1000.0, "dGTP": 1000.0, "dCTP": 1000.0, "dTTP": 1000.0,
        "CTP": 1000.0, "GTP": 1000.0, "UTP": 1000.0,
        "NADH": 1000.0, "NAD": 1000.0,
    }
    collapsed = dict(base_pools)
    collapsed["ATP"] = 50.0   # ratio 0.05 vs WT 1000 — well below 0.5
    samples = (
        Sample(t_s=0.0, pools=dict(base_pools)),
        Sample(t_s=1.0, pools=dict(collapsed)),
        Sample(t_s=2.0, pools=dict(collapsed)),
        Sample(t_s=3.0, pools=dict(collapsed)),
        Sample(t_s=4.0, pools=dict(collapsed)),
    )
    return Trajectory(samples=samples)


def _trna_collapse_trajectory() -> Trajectory:
    """Charged-tRNA fraction drops to 10 % of WT — different failure mode
    so the symmetry test can exercise non-trivial detector output."""
    base_pools = {
        "ATP": 1000.0,
        "CHARGED_TRNA_FRACTION": 1000.0,
        "NTP_TOTAL": 1000.0,
        "G6P": 1000.0, "F6P": 1000.0, "PYR": 1000.0,
        "dATP": 1000.0, "dGTP": 1000.0, "dCTP": 1000.0, "dTTP": 1000.0,
        "CTP": 1000.0, "GTP": 1000.0, "UTP": 1000.0,
        "NADH": 1000.0, "NAD": 1000.0,
    }
    collapsed = dict(base_pools)
    collapsed["CHARGED_TRNA_FRACTION"] = 100.0   # ratio 0.10 vs 0.30 thresh
    samples = (
        Sample(t_s=0.0, pools=dict(base_pools)),
        Sample(t_s=1.0, pools=dict(collapsed)),
        Sample(t_s=2.0, pools=dict(collapsed)),
        Sample(t_s=3.0, pools=dict(collapsed)),
        Sample(t_s=4.0, pools=dict(collapsed)),
    )
    return Trajectory(samples=samples)


@pytest.fixture
def harness_with_mock_responses():
    """Build a harness whose KO simulator returns:
        gene_E1   -> ATP collapse (essential, ATP_DEPLETION)
        gene_E2   -> tRNA collapse (essential, TRANSLATION_STALL)
        gene_N1   -> flat trajectory (non-essential)
        gene_N2   -> flat trajectory (non-essential)
        gene_N3   -> flat trajectory (non-essential)
    """
    flat = _flat_trajectory()
    responses = {
        "gene_E1": _atp_collapse_trajectory(),
        "gene_E2": _trna_collapse_trajectory(),
        "gene_N1": flat,
        "gene_N2": flat,
        "gene_N3": flat,
    }
    sim = MockSimulator(responses=responses, default=flat)
    return KnockoutHarness(
        wt_simulator=sim, ko_simulator=sim,
        t_end_s=4.0, sample_dt_s=1.0,
    )


# ---------------------------------------------------------------------
# Invariant 1 — predict_pair(X, X) == predict(X) on an essential and a
# non-essential gene.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("gene", ["gene_E1", "gene_E2", "gene_N1", "gene_N2", "gene_N3"])
def test_pairwise_with_same_gene_matches_single(
    harness_with_mock_responses, gene: str,
) -> None:
    """Calling predict_pair(X, X) must return the same essentiality call,
    same failure mode, and same time-to-failure as predict(X).

    Why this matters: synthetic-lethality work is additive — it must not
    perturb the existing single-knockout pipeline. The simulator's
    knockout-tuple collapse semantics guarantee this at the trajectory
    level; this test confirms it survives the harness wrapping."""
    h = harness_with_mock_responses
    single = h.predict(gene)
    pair = h.predict_pair(gene, gene)
    assert single.essential == pair.essential
    assert single.failure_mode == pair.failure_mode
    assert single.time_to_failure_s == pair.time_to_failure_s
    assert single.confidence == pytest.approx(pair.confidence, abs=1e-9)


# ---------------------------------------------------------------------
# Invariant 2 — predict_pair(A, B) is symmetric in (A, B).
# ---------------------------------------------------------------------


SYMMETRIC_PAIRS = [
    ("gene_N1", "gene_N2"),     # both non-essential
    ("gene_E1", "gene_N1"),     # one essential, one non-essential
    ("gene_E1", "gene_E2"),     # both essential, different modes
    ("gene_E2", "gene_N3"),
    ("gene_N1", "gene_N3"),
]


@pytest.mark.parametrize("a,b", SYMMETRIC_PAIRS)
def test_pairwise_symmetry(
    harness_with_mock_responses, a: str, b: str,
) -> None:
    """predict_pair(A, B) must equal predict_pair(B, A).

    This isn't quite a Gillespie symmetry guarantee (sample-by-sample
    pool counts can differ across runs because of RNG differences in
    tuple ordering), but the essential/non-essential VERDICT and the
    failure mode must be identical because the detector is gene-
    agnostic and the simulator sorts the knockout tuple before use."""
    h = harness_with_mock_responses
    p_ab = h.predict_pair(a, b)
    p_ba = h.predict_pair(b, a)
    assert p_ab.essential == p_ba.essential
    assert p_ab.failure_mode == p_ba.failure_mode


# ---------------------------------------------------------------------
# Smoke test — pair of two essentials must be essential.
# ---------------------------------------------------------------------


def test_pair_with_two_essentials_is_essential(
    harness_with_mock_responses,
) -> None:
    """If both X and Y are individually essential, the joint pair
    cannot suddenly be non-essential. Sanity check that the OR-style
    failure-detection path never silently drops a fire."""
    h = harness_with_mock_responses
    p = h.predict_pair("gene_E1", "gene_E2")
    assert p.essential is True
    assert p.failure_mode != FailureMode.NONE


def test_pair_with_two_nonessentials_is_nonessential(
    harness_with_mock_responses,
) -> None:
    """In the mock harness all three non-essential genes return a flat
    trajectory; their pair must also not trip the detector. This is the
    NEGATIVE control: most random non-essential pairs should not flag
    as synthetic lethal in the real screen, and the detector must be
    capable of producing a non-essential verdict on a joint knockout."""
    h = harness_with_mock_responses
    p = h.predict_pair("gene_N1", "gene_N2")
    assert p.essential is False
    assert p.failure_mode == FailureMode.NONE


# ---------------------------------------------------------------------
# Existing single-knockout API must not regress on the same harness.
# ---------------------------------------------------------------------


def test_existing_single_knockout_unchanged(
    harness_with_mock_responses,
) -> None:
    """A spot-check that ``predict()`` still returns the same shape +
    behaviour after the predict_pair extension. If this regresses,
    Session 26 has broken something it was hard-required not to."""
    h = harness_with_mock_responses
    p = h.predict("gene_E1")
    assert isinstance(p, Prediction)
    assert p.essential is True
    assert p.failure_mode == FailureMode.ATP_DEPLETION
    p2 = h.predict("gene_N1")
    assert p2.essential is False
    assert p2.failure_mode == FailureMode.NONE


# ---------------------------------------------------------------------
# PairPrediction shape contract.
# ---------------------------------------------------------------------


def test_pair_prediction_as_row_schema(
    harness_with_mock_responses,
) -> None:
    """The CSV serialisation contract used by the pilot run script."""
    h = harness_with_mock_responses
    pp = h.predict_pair(
        "gene_N1", "gene_N2",
        gene_name_a="alpha", gene_name_b="beta",
    )
    assert isinstance(pp, PairPrediction)
    row = pp.as_row()
    expected = {
        "locus_tag_a", "locus_tag_b",
        "gene_name_a", "gene_name_b",
        "essential", "time_to_failure_s",
        "failure_mode", "confidence",
    }
    assert set(row.keys()) == expected
