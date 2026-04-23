"""Bridge the atom engine to the existing Layer-6 essentiality detector
framework (``cell_sim.layer6_essentiality``).

"Genes" in this mapping = bondable element-pairs (e.g. ``pair_CH``,
``pair_CO``, ``pair_NH``). A "knockout" disables that pair's bond
formation in the integrator for an entire run. The atom engine then
emits a ``Trajectory`` of ``Sample`` records with:

  - ``pools``: per-element free-atom counts + per-formula molecule
    counts. These drive the metabolite-pool detectors
    (``ShortWindowDetector``, the default ``FailureDetector``).

  - ``event_counts_by_rule``: cumulative bond-formation counts keyed
    by ``bond:X-Y`` rule name. This drives ``PerRuleDetector`` and
    ``RedundancyAwareDetector``.

Because the atom engine produces the EXACT interface the existing
detectors consume, no detector code is modified — the existing
``ShortWindowDetector``, ``PerRuleDetector``, ``EnsembleDetector``,
and ``RedundancyAwareDetector`` all run against atom-engine output
out of the box.

Ground-truth "essential" label for a pair P is defined physically:
P is essential iff its disruption substantially suppresses the
downstream molecule populations the wild-type run produces. For the
default chemistry soup (H/C/N/O at reactive T), the essential pairs
are those that carry nonzero WT bond-formation events above a
small noise threshold.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .atom_soup import SoupSpec, build_soup
from .atom_unit import AtomUnit, Bond, BondType
from .element import Element, pair_is_bondable
from .force_field import ForceFieldConfig
from .integrator import IntegratorConfig, SimState, _rule_name_for_pair, step
from .molecule_builder import classify_molecules

from cell_sim.layer6_essentiality.harness import (
    FailureMode,
    Prediction,
    Sample,
    Trajectory,
)


# Reactive elements the atom engine knows. Any bondable pair over this
# set is a candidate "gene".
_REACTIVE_ELEMENTS = (
    Element.H, Element.C, Element.N, Element.O, Element.P, Element.S,
)


def enumerate_pair_genes() -> list[tuple[str, frozenset]]:
    """All legal (Element, Element) pairs over the reactive set."""
    out: list[tuple[str, frozenset]] = []
    seen: set[frozenset] = set()
    for i, a in enumerate(_REACTIVE_ELEMENTS):
        for b in _REACTIVE_ELEMENTS[i:]:
            if not pair_is_bondable(a, b):
                continue
            fs = frozenset([a, b])
            if fs in seen:
                continue
            names = sorted([a.name, b.name])
            gene_id = f"pair_{names[0]}{names[1]}"
            out.append((gene_id, fs))
            seen.add(fs)
    return out


@dataclass
class AtomEngineSimConfig:
    composition: dict = field(default_factory=lambda: {
        Element.H: 40, Element.C: 10, Element.N: 4, Element.O: 6,
    })
    radius_nm: float = 1.2
    temperature_K: float = 2500.0
    dt_ps: float = 0.001
    steps: int = 4000
    snapshot_every_steps: int = 200
    reactive_sigma_scale: float = 0.3
    seed: int = 42


@dataclass
class AtomEngineSimulator:
    """Implements the Layer-6 ``Simulator`` protocol: ``run(knockout,
    t_end_s, sample_dt_s) -> Trajectory``.

    ``knockout`` is an iterable of gene_ids like ``["pair_CO"]`` — the
    simulator disables the corresponding element pairs for the run.
    Unknown gene_ids are silently ignored (same as the real simulator).

    ``t_end_s`` and ``sample_dt_s`` are accepted for interface
    compatibility but the atom engine runs in picoseconds. They are
    re-interpreted as scaled picoseconds via the cfg.
    """
    cfg: AtomEngineSimConfig = field(default_factory=AtomEngineSimConfig)

    def run(
        self,
        knockout: Iterable[str],
        *,
        t_end_s: float = 0.0,         # ignored, kept for interface
        sample_dt_s: float = 0.0,     # ignored, kept for interface
    ) -> Trajectory:
        gene_map = dict(enumerate_pair_genes())
        ko_set = frozenset(gene_map[g] for g in knockout if g in gene_map)

        atoms = build_soup(SoupSpec(
            composition=self.cfg.composition,
            radius_nm=self.cfg.radius_nm,
            temperature_K=self.cfg.temperature_K,
            seed=self.cfg.seed,
        ))
        state = SimState(atoms=atoms, bonds=[])

        ff = ForceFieldConfig(
            lj_cutoff_nm=1.0, use_confinement=True,
            confinement_radius_nm=self.cfg.radius_nm,
            use_neighbor_list=True,
            reactive_sigma_scale=self.cfg.reactive_sigma_scale,
        )
        int_cfg = IntegratorConfig(
            dt_ps=self.cfg.dt_ps,
            target_temperature_K=self.cfg.temperature_K,
            dynamic_bonding=True,
            bond_form_distance_nm=0.2,
            neighbor_rebuild_every=10,
            disabled_pairs=ko_set if ko_set else None,
        )

        samples: list[Sample] = []
        forces = None
        for k in range(self.cfg.steps):
            forces = step(state, ff, int_cfg, forces)
            if (k + 1) % self.cfg.snapshot_every_steps == 0 or k == 0:
                samples.append(_sample_from_state(state))
        # Always record the final state.
        if samples and samples[-1].t_s < state.t_ps:
            samples.append(_sample_from_state(state))
        return Trajectory(samples=tuple(samples))


def _sample_from_state(state: SimState) -> Sample:
    """Convert atom-engine state to a detector-compatible Sample.

    Pools:
      - per-element free-atom counts, keyed by element name ("H" -> 23)
      - per-formula molecule counts from classify_molecules ("H2O" -> 4)
    """
    pools: dict[str, float] = {}
    # Free-atom counts.
    for a in state.atoms:
        if len(a.bonds) == 0:
            name = a.element.name
            pools[name] = pools.get(name, 0.0) + 1.0
    # Molecule counts (non-singleton components only).
    formulas = classify_molecules(state.atoms)
    for f, c in formulas.items():
        if f not in pools:
            pools[f] = 0.0
        pools[f] = float(c)
    event_counts_by_rule = dict(state.events_formed_by_rule)
    return Sample(
        t_s=float(state.t_ps),
        pools=pools,
        event_counts_by_rule=event_counts_by_rule,
    )


# ---------- ground-truth essentiality label -------------------------


def wt_essentiality_labels(
    sim: AtomEngineSimulator,
    min_events: int = 3,
) -> dict[str, bool]:
    """Run the wild-type simulation and derive ground-truth labels.

    A pair gene is labelled essential iff the WT run records at least
    ``min_events`` bond-formed events involving that pair in its
    lifetime. Pairs that never fire in WT are non-essential by
    definition (knocking out a gene that does nothing cannot break
    the system).
    """
    wt_traj = sim.run([])
    last = wt_traj.samples[-1]
    counts = last.event_counts_by_rule or {}
    out: dict[str, bool] = {}
    for gene_id, pair in enumerate_pair_genes():
        elems = list(pair)
        a = elems[0]
        b = elems[1] if len(elems) > 1 else elems[0]
        rule = _rule_name_for_pair(a, b)
        n = counts.get(rule, 0)
        out[gene_id] = bool(n >= min_events)
    return out
