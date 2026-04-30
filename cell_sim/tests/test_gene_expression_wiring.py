"""Tests for the Session 27 phase 1 gene-expression wiring.

The wiring threads ``cell_sim.layer3_reactions.gene_expression`` into the
production simulator behind a ``RealSimulatorConfig.enable_gene_expression``
flag. Default off so v0..v15 measurements remain bit-identical.

Tests cover:

1. Default config has the flag off and ``_gex_rules`` is None.
2. Flag-on setup caches the gex rule list (4 rules per CDS).
3. Flag-on: knockout removes the gex rules for the knocked-out gene
   AND zeroes its mRNA + promoter strength.
4. Flag-on smoke run actually fires transcribe / translate / degrade_mrna
   events (i.e. the rules are reachable from the FastEventSimulator).
5. Flag-off smoke run does NOT fire any gex events even though the
   gex module is importable and the spec contains every CDS.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_GB_PATH = (
    Path(__file__).resolve().parents[1]
    / "data" / "Minimal_Cell_ComplexFormation" / "input_data" / "syn3A.gb"
)
_SBML_PATH = (
    Path(__file__).resolve().parents[1]
    / "data" / "Minimal_Cell_ComplexFormation" / "input_data"
    / "Syn3A_updated.xml"
)
_NEEDS_DATA = pytest.mark.skipif(
    not (_GB_PATH.exists() and _SBML_PATH.exists()),
    reason="Luthey-Schulten data not staged; see memory_bank/data/STAGING.md",
)


def test_default_flag_is_off_and_no_gex_rules_cached():
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    cfg = RealSimulatorConfig()
    assert cfg.enable_gene_expression is False
    assert cfg.gene_expression_scale_factor == 0.1
    sim = RealSimulator(cfg)
    assert sim._gex_rules is None


@_NEEDS_DATA
def test_flag_off_setup_does_not_build_gex_rules():
    """The expensive gex-rule construction should not run when the flag
    is off — protects existing v0..v15 measurements from any incidental
    cost or side effect."""
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(enable_gene_expression=False))
    sim._ensure_setup()
    assert sim._gex_rules is None


@_NEEDS_DATA
def test_flag_on_setup_caches_four_rules_per_cds():
    """build_gene_expression_rules emits 4 rules (transcribe / translate /
    degrade_mrna / degrade_protein) per CDS by default. The cached list
    on the simulator instance should match."""
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(enable_gene_expression=True))
    sim._ensure_setup()
    assert sim._gex_rules is not None
    n_cds = len(sim._spec.proteins)
    assert len(sim._gex_rules) == 4 * n_cds, (
        f"expected 4 rules per CDS ({4 * n_cds}), got {len(sim._gex_rules)}"
    )
    # Every rule has its locus tag as the first participant — this is
    # the contract _build_state_and_rules uses to filter knockouts.
    for r in sim._gex_rules:
        assert r.participants, f"rule {r.name} has empty participants"
        assert r.participants[0].startswith("JCVISYN3A"), (
            f"rule {r.name} participant {r.participants[0]!r} is not a "
            "Syn3A locus tag"
        )


@_NEEDS_DATA
def test_flag_on_rules_cover_expected_factory_prefixes():
    """build_gene_expression_rules emits rules with names of the form
    ``<prefix>:<locus>``. Each prefix should appear once per CDS."""
    from collections import Counter
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(enable_gene_expression=True))
    sim._ensure_setup()
    prefixes = Counter(r.name.split(":", 1)[0] for r in sim._gex_rules)
    n_cds = len(sim._spec.proteins)
    assert prefixes["transcribe"] == n_cds
    assert prefixes["translate"] == n_cds
    assert prefixes["degrade_mrna"] == n_cds
    assert prefixes["degrade_protein"] == n_cds


@_NEEDS_DATA
def test_knockout_drops_gex_rules_and_zeroes_state_for_ko_gene():
    """When a gene is knocked out, the per-run rule list must not contain
    its tx / translate / degrade_mrna / degrade_protein rules, and the
    state.mrna_counts / state.promoter_strength entries for that gene
    must be zeroed so transcription cannot fire at the floor strength."""
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(
        enable_gene_expression=True, scale_factor=0.02, seed=42,
    ))
    sim._ensure_setup()
    ko = ("JCVISYN3A_0445",)  # pgi
    state, rules = sim._build_state_and_rules(ko)
    # No gex rule for the knocked-out gene.
    ko_gex = [r for r in rules
              if r.name.split(":", 1)[0] in (
                  "transcribe", "translate", "degrade_mrna", "degrade_protein",
              )
              and r.participants and r.participants[0] == "JCVISYN3A_0445"]
    assert ko_gex == [], (
        f"expected zero gex rules for knocked-out gene; got {len(ko_gex)}"
    )
    # State zeroed for the knocked-out gene.
    assert state.mrna_counts.get("JCVISYN3A_0445", 0) == 0
    assert state.promoter_strength.get("JCVISYN3A_0445", 0.0) == 0.0
    # Sanity: at least one OTHER gex rule survives in the rule list (the
    # filter is gene-specific, not blanket-disable).
    other_gex = [r for r in rules
                 if r.name.split(":", 1)[0] in (
                     "transcribe", "translate", "degrade_mrna",
                     "degrade_protein",
                 )]
    assert len(other_gex) > 0


@_NEEDS_DATA
def test_flag_on_smoke_run_fires_gene_expression_events():
    """End-to-end: with the flag on, a tiny run produces visible
    transcribe / translate / degrade events. This proves the rules are
    actually reachable from the FastEventSimulator (i.e. the wiring
    survives the spec → state → rules pipeline)."""
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(
        enable_gene_expression=True, scale_factor=0.02, seed=42,
    ))
    sim._ensure_setup()
    state, rules = sim._build_state_and_rules(())
    # Use the same FastEventSimulator wiring as run(), but inline so we
    # can read state.events directly without the Trajectory aggregation
    # path. A 0.05 s window is enough to see at least one tx event with
    # ~50 free ribosomes and ~14 free RNAPs at scale 0.02.
    from layer2_field.fast_dynamics import FastEventSimulator
    fes = FastEventSimulator(state, rules, mode="gillespie", seed=42)
    fes.run_until(t_end=0.05, max_events=200_000)

    gex_event_prefixes = {
        e.rule_name.split(":", 1)[0] for e in state.events
    }
    # At least one of the four gex rule families should fire in 0.05 s
    # of biological time at this scale. translate fires fastest because
    # ribosome × mRNA × all-genes propensity is large.
    fired_gex = gex_event_prefixes & {
        "transcribe", "translate", "degrade_mrna", "degrade_protein",
    }
    assert fired_gex, (
        f"no gex events fired in 0.05 s window; got prefixes {gex_event_prefixes}"
    )


@_NEEDS_DATA
def test_flag_off_smoke_run_fires_zero_gene_expression_events():
    """Mirror of the previous test, with the flag off. Confirms the
    invariant that v0..v15 measurements (gex-off) cannot be perturbed
    by gex events leaking into the trajectory."""
    from cell_sim.layer6_essentiality.real_simulator import (
        RealSimulator, RealSimulatorConfig,
    )
    sim = RealSimulator(RealSimulatorConfig(
        enable_gene_expression=False, scale_factor=0.02, seed=42,
    ))
    sim._ensure_setup()
    state, rules = sim._build_state_and_rules(())
    from layer2_field.fast_dynamics import FastEventSimulator
    fes = FastEventSimulator(state, rules, mode="gillespie", seed=42)
    fes.run_until(t_end=0.05, max_events=200_000)
    gex_event_prefixes = {
        e.rule_name.split(":", 1)[0] for e in state.events
    } & {"transcribe", "translate", "degrade_mrna", "degrade_protein"}
    assert gex_event_prefixes == set(), (
        f"flag-off run leaked gex events: {gex_event_prefixes}"
    )
