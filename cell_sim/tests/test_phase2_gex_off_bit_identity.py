"""Phase 2 regression test: gex-off bit-identity invariant.

The Session 27 phase-1 wiring landed ``enable_gene_expression`` defaulting
to False; v0..v15 measurements are claimed to be bit-identical at flag-off.
Phase 2 (vectorized-propensity gex compilation) refactors
``cell_sim/layer2_field/fast_dynamics.py`` and the rule factories in
``cell_sim/layer3_reactions/gene_expression.py``. The refactor MUST NOT
perturb the gex-off code path. This test enforces that.

Comparison surface — 5 runs (1 WT + 4 KOs from the test_knockouts
reference panel) at scale=0.05, t_end=0.5, sample_dt_s=0.05, seed=42,
use_rust_backend=False, enable_gene_expression=False:

  - per-sample ``pools`` dict (metabolite + non-metabolic counts)
  - per-sample ``event_counts_by_rule`` dict (cumulative rule firing)

Both must be exactly equal to the committed baseline at
``cell_sim/tests/data/phase2_gex_off_baseline.json`` for every sample
of every run. The simulator is fully deterministic at fixed seed, so
this is bit-identity not approximate match.

Run modes:

  default (compare):   pytest -k phase2_gex_off_bit_identity
                       (requires RUN_PHASE2_REGRESSION=1 to opt in;
                       gated because each run takes ~14 s wall and the
                       full panel is ~70 s, too slow for the default
                       sandbox suite)

  regenerate baseline: PHASE2_REGENERATE_BASELINE=1
                       RUN_PHASE2_REGRESSION=1 pytest -k phase2_gex_off
                       (writes the fixture from a fresh run; only
                       legitimate when an upstream change to the
                       metabolic core is intentional and expected to
                       shift the trajectory)

If this test fails on a phase-2 commit, REVERT the commit. The gex-off
path is the production v15 measurement; perturbing it without explicit
approval breaks the v15 MCC 0.5372 reference fact.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Schema version bumped only when the snapshot format itself changes
# (e.g. new keys, restructured layout). Mismatching versions force a
# regeneration step rather than a silent comparison.
SCHEMA_VERSION = 1

# 4-gene reference panel + WT baseline. Same panel as test_knockouts.py
# so the trajectory is interpretable against the existing knockout
# notebook.
REFERENCE_PANEL: tuple[tuple[str, ...], ...] = (
    (),                           # WT baseline
    ("JCVISYN3A_0445",),          # pgi (essential)
    ("JCVISYN3A_0779",),          # ptsG (essential)
    ("JCVISYN3A_0522",),          # ftsZ (nonessential at 0.5 s)
    ("JCVISYN3A_0305",),          # uncharacterized metallopeptidase
)

# Run config — must match the values referenced in
# memory_bank/facts/measured/mcc_v15_replicates.json so the regression
# guards the same code path as v15.
CFG = {
    "scale_factor": 0.05,
    "t_end_s": 0.5,
    "sample_dt_s": 0.05,
    "seed": 42,
    "use_rust_backend": False,
    "enable_gene_expression": False,
}

_FIXTURE_DIR = Path(__file__).resolve().parent / "data"
_FIXTURE_PATH = _FIXTURE_DIR / "phase2_gex_off_baseline.json"

_GB_PATH = (
    Path(__file__).resolve().parents[1]
    / "data" / "Minimal_Cell_ComplexFormation" / "input_data" / "syn3A.gb"
)
_NEEDS_DATA = pytest.mark.skipif(
    not _GB_PATH.exists(),
    reason="Luthey-Schulten data not staged; see memory_bank/data/STAGING.md",
)
_NEEDS_OPT_IN = pytest.mark.skipif(
    os.environ.get("RUN_PHASE2_REGRESSION", "0") != "1",
    reason="set RUN_PHASE2_REGRESSION=1 to run (each panel takes ~70 s)",
)


# ---------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------


def _run_panel() -> list[dict]:
    """Run the reference panel under the gex-off config and return a
    list of run snapshots ready for JSON serialization."""
    # Repo path: scripts/ + cell_sim/ both expect cell_sim on sys.path.
    cell_sim = Path(__file__).resolve().parents[1]
    if str(cell_sim) not in sys.path:
        sys.path.insert(0, str(cell_sim))

    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(
        scale_factor=CFG["scale_factor"],
        seed=CFG["seed"],
        use_rust_backend=CFG["use_rust_backend"],
        enable_gene_expression=CFG["enable_gene_expression"],
    ))
    runs: list[dict] = []
    for ko in REFERENCE_PANEL:
        traj = sim.run(
            list(ko),
            t_end_s=CFG["t_end_s"],
            sample_dt_s=CFG["sample_dt_s"],
        )
        samples_payload = []
        for s in traj.samples:
            samples_payload.append({
                "t_s": s.t_s,
                "pools": {k: float(v) for k, v in s.pools.items()},
                "event_counts_by_rule": (
                    dict(s.event_counts_by_rule)
                    if s.event_counts_by_rule is not None
                    else None
                ),
            })
        runs.append({"knockout": list(ko), "samples": samples_payload})
    return runs


def _serialize(runs: list[dict]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": dict(CFG),
        "panel": [list(ko) for ko in REFERENCE_PANEL],
        "runs": runs,
    }


def _compare(actual: list[dict], expected: list[dict]) -> None:
    """Bit-identity comparison; raises AssertionError with a precise
    diagnostic on the first divergence."""
    assert len(actual) == len(expected), (
        f"run-count mismatch: actual={len(actual)} expected={len(expected)}"
    )
    for run_idx, (a_run, e_run) in enumerate(zip(actual, expected)):
        ko_label = "WT" if not a_run["knockout"] else a_run["knockout"][0]
        assert a_run["knockout"] == e_run["knockout"], (
            f"run {run_idx}: knockout mismatch "
            f"actual={a_run['knockout']!r} expected={e_run['knockout']!r}"
        )
        a_samples = a_run["samples"]
        e_samples = e_run["samples"]
        assert len(a_samples) == len(e_samples), (
            f"run {run_idx} ({ko_label}): sample-count mismatch "
            f"actual={len(a_samples)} expected={len(e_samples)}"
        )
        for sidx, (a_s, e_s) in enumerate(zip(a_samples, e_samples)):
            assert a_s["t_s"] == e_s["t_s"], (
                f"run {run_idx} ({ko_label}) sample {sidx}: t_s mismatch "
                f"actual={a_s['t_s']} expected={e_s['t_s']}"
            )
            # Pools — exact float equality. The simulator is fully
            # deterministic at fixed seed; any drift means a code path
            # changed, not numerical noise.
            a_pools = a_s["pools"]
            e_pools = e_s["pools"]
            a_keys = set(a_pools.keys())
            e_keys = set(e_pools.keys())
            assert a_keys == e_keys, (
                f"run {run_idx} ({ko_label}) sample {sidx} (t={a_s['t_s']}): "
                f"pool keys differ "
                f"only-in-actual={sorted(a_keys - e_keys)} "
                f"only-in-expected={sorted(e_keys - a_keys)}"
            )
            for k in sorted(a_keys):
                assert a_pools[k] == e_pools[k], (
                    f"run {run_idx} ({ko_label}) sample {sidx} "
                    f"(t={a_s['t_s']}) pool {k!r}: "
                    f"actual={a_pools[k]} expected={e_pools[k]} "
                    f"(diff={a_pools[k] - e_pools[k]})"
                )
            # Event counts — exact integer equality.
            a_ev = a_s["event_counts_by_rule"]
            e_ev = e_s["event_counts_by_rule"]
            if a_ev is None or e_ev is None:
                assert a_ev == e_ev, (
                    f"run {run_idx} ({ko_label}) sample {sidx}: "
                    f"event_counts_by_rule None mismatch "
                    f"actual={a_ev!r} expected={e_ev!r}"
                )
                continue
            a_ev_keys = set(a_ev.keys())
            e_ev_keys = set(e_ev.keys())
            assert a_ev_keys == e_ev_keys, (
                f"run {run_idx} ({ko_label}) sample {sidx}: "
                f"event_counts_by_rule keys differ "
                f"only-in-actual={sorted(a_ev_keys - e_ev_keys)[:10]} "
                f"only-in-expected={sorted(e_ev_keys - a_ev_keys)[:10]}"
            )
            for k in sorted(a_ev_keys):
                assert a_ev[k] == e_ev[k], (
                    f"run {run_idx} ({ko_label}) sample {sidx} "
                    f"event {k!r}: actual={a_ev[k]} expected={e_ev[k]}"
                )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


@_NEEDS_DATA
@_NEEDS_OPT_IN
def test_gex_off_bit_identity_against_committed_baseline():
    """The gex-off code path produces a per-sample pool + event-count
    snapshot that is bit-identical to the committed baseline."""
    runs = _run_panel()

    if os.environ.get("PHASE2_REGENERATE_BASELINE", "0") == "1":
        _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        payload = _serialize(runs)
        _FIXTURE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
        pytest.skip(
            f"baseline regenerated at {_FIXTURE_PATH} "
            f"(PHASE2_REGENERATE_BASELINE=1 was set); "
            f"comparison skipped this run"
        )

    assert _FIXTURE_PATH.exists(), (
        f"baseline fixture not found at {_FIXTURE_PATH}. "
        "Run PHASE2_REGENERATE_BASELINE=1 RUN_PHASE2_REGRESSION=1 "
        "pytest -k phase2_gex_off to generate it."
    )
    expected_payload = json.loads(_FIXTURE_PATH.read_text())
    assert expected_payload["schema_version"] == SCHEMA_VERSION, (
        f"baseline schema_version={expected_payload['schema_version']} "
        f"doesn't match test SCHEMA_VERSION={SCHEMA_VERSION}; "
        "regenerate the baseline."
    )
    assert expected_payload["config"] == CFG, (
        f"baseline config doesn't match test CFG: "
        f"actual={CFG} baseline={expected_payload['config']}"
    )
    assert expected_payload["panel"] == [list(ko) for ko in REFERENCE_PANEL], (
        "baseline panel doesn't match REFERENCE_PANEL; regenerate the baseline."
    )
    _compare(runs, expected_payload["runs"])


@_NEEDS_DATA
def test_baseline_fixture_is_committed_and_well_formed():
    """Sanity check that the committed fixture exists and parses; runs
    without RUN_PHASE2_REGRESSION so it executes in the default sandbox
    suite. Catches the case where someone deletes the fixture and the
    main regression test is silently skipped."""
    assert _FIXTURE_PATH.exists(), (
        f"baseline fixture missing: {_FIXTURE_PATH}. "
        "Phase 2 cannot start without it. Regenerate via "
        "PHASE2_REGENERATE_BASELINE=1 RUN_PHASE2_REGRESSION=1 "
        "pytest -k phase2_gex_off."
    )
    payload = json.loads(_FIXTURE_PATH.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["config"] == CFG
    assert payload["panel"] == [list(ko) for ko in REFERENCE_PANEL]
    assert len(payload["runs"]) == len(REFERENCE_PANEL)
    for run, ko in zip(payload["runs"], REFERENCE_PANEL):
        assert run["knockout"] == list(ko)
        # 11 samples = t in {0.0, 0.05, 0.10, ..., 0.50}.
        assert len(run["samples"]) >= 11, (
            f"baseline run {ko}: expected >=11 samples, got "
            f"{len(run['samples'])}"
        )
        # First sample must carry ATP and have non-zero count, so we
        # know the snapshot was populated, not a stub.
        first = run["samples"][0]
        assert "ATP" in first["pools"]
        assert first["pools"]["ATP"] > 0
