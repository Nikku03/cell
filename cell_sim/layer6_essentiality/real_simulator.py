"""Real-simulator backend for Layer 6 essentiality.

Wraps the existing ``cell_sim.layer2_field.FastEventSimulator`` +
``populate_real_syn3a`` machinery behind the ``Simulator`` Protocol used
by ``KnockoutHarness``.

Pragmatic choices baked into the defaults (see :class:`RealSimulatorConfig`):

* ``t_end_s = 0.5`` and ``sample_dt_s = 0.05`` (10 samples per run).
  The brief's ideal window is one doubling time (7200 s) but that's
  >30 hours/gene at any usable scale. The existing knockout test
  panel (``cell_sim/tests/test_knockouts.py``) demonstrates that 0.5
  s of bio-time is enough to see the metabolic essentiality signal
  (e.g. ATP / G6P crash) for the four reference genes (pgi, ptsG,
  ftsZ, JCVISYN3A_0305). We follow that precedent for the first
  full-sweep MCC measurement and treat the longer window as future
  work once the simulator is faster.

* ``scale_factor = 0.05`` (5 % of biological copy numbers). Small
  enough to keep per-gene wall time well under one second on a
  single CPU core. The existing knockout test uses 0.5; at 0.05 the
  signal-to-noise on per-gene predictions is reduced but the
  qualitative essentiality call is preserved for genes whose KO
  causes a sharp metabolic crash. Scale up for a higher-fidelity
  rerun.

* Heavy setup (SBML parse, kinetics load, medium load, base
  CellSpec build) is cached on the simulator instance and reused
  across knockouts. Per-knockout work is just CellState +
  populate_real_syn3a + run.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Existing cell_sim modules use unprefixed imports; ensure cell_sim/ is on
# sys.path before they import.
_CELL_SIM = Path(__file__).resolve().parents[1]  # /home/user/cell/cell_sim
if str(_CELL_SIM) not in sys.path:
    sys.path.insert(0, str(_CELL_SIM))

import numpy as np  # noqa: E402

from cell_sim.layer6_essentiality.harness import (  # noqa: E402
    Sample, Simulator, Trajectory,
)


# Mapping from FailureDetector pool keys -> Syn3A SBML species ids.
# All Syn3A intracellular metabolites use the M_<bigg>_c convention.
_POOL_KEY_TO_SPECIES: dict[str, str] = {
    "ATP": "M_atp_c",
    "ADP": "M_adp_c",
    "G6P": "M_g6p_c",
    "F6P": "M_f6p_c",
    "PYR": "M_pyr_c",
    "dATP": "M_datp_c",
    "dGTP": "M_dgtp_c",
    "dCTP": "M_dctp_c",
    "dTTP": "M_dttp_c",
    "CTP":  "M_ctp_c",
    "GTP":  "M_gtp_c",
    "UTP":  "M_utp_c",
    "NADH": "M_nadh_c",
    "NAD":  "M_nad_c",
}
# Lumped pools the detector also watches.
_NTP_COMPONENTS = ("M_atp_c", "M_ctp_c", "M_gtp_c", "M_utp_c")


_DEFAULT_SBML_PATH = (
    _CELL_SIM
    / "data" / "Minimal_Cell_ComplexFormation"
    / "input_data" / "Syn3A_updated.xml"
)


@dataclass
class RealSimulatorConfig:
    scale_factor: float = 0.05
    seed: int = 42
    sbml_path: Path = _DEFAULT_SBML_PATH
    cell_volume_um3: float = (4.0 / 3.0) * 3.141592653589793 * (0.2 ** 3)
    folding_rate_per_s: float = 20.0
    complex_assembly_rate_per_uM_per_s: float = 0.05
    max_events_per_chunk: int = 1_000_000
    chunk_dt_s: float = 0.05  # sim chunk size for run_until


class RealSimulator(Simulator):
    """Production backend that drives FastEventSimulator under a knockout.

    One instance is reused across many knockouts to amortise the heavy
    one-time setup. Each ``run()`` call rebuilds the per-knockout state.
    """

    def __init__(self, cfg: RealSimulatorConfig | None = None) -> None:
        self.cfg = cfg or RealSimulatorConfig()
        self._setup_done = False
        # Heavy artefacts cached after the first call.
        self._spec = None
        self._counts_template: dict | None = None
        self._complexes = None
        self._sbml = None
        self._kinetics = None
        self._medium = None
        self._rev_rules = None
        self._extra_rules = None

    # ----- one-time heavy setup -----
    def _ensure_setup(self) -> None:
        if self._setup_done:
            return
        # Imports here so module import doesn't pay this cost.
        from layer0_genome.syn3a_real import build_real_syn3a_cellspec
        from layer3_reactions.sbml_parser import parse_sbml
        from layer3_reactions.kinetics import load_all_kinetics, load_medium
        from layer3_reactions.reversible import build_reversible_catalysis_rules
        from layer3_reactions.nutrient_uptake import build_missing_transport_rules

        with contextlib.redirect_stdout(io.StringIO()):
            spec, counts, complexes, _ = build_real_syn3a_cellspec()
            sbml = parse_sbml(self.cfg.sbml_path)
            kinetics = load_all_kinetics()
            medium = load_medium()
            rev_rules, _ = build_reversible_catalysis_rules(sbml, kinetics)
            extra_rules = build_missing_transport_rules(sbml, kinetics)
        self._spec = spec
        self._counts_template = counts
        self._complexes = complexes
        self._sbml = sbml
        self._kinetics = kinetics
        self._medium = medium
        self._rev_rules = rev_rules
        self._extra_rules = extra_rules
        self._setup_done = True

    # ----- per-run state build -----
    def _build_state_and_rules(self, knockout: tuple[str, ...]):
        from layer2_field.dynamics import CellState
        from layer2_field.real_syn3a_rules import (
            populate_real_syn3a, make_folding_rule, make_complex_formation_rules,
        )
        from layer3_reactions.coupled import initialize_metabolites
        from layer3_reactions.reversible import initialize_medium

        counts = dict(self._counts_template)  # shallow copy is enough; values are scalars
        for tag in knockout:
            counts.pop(tag, None)

        state = CellState(self._spec)
        max_per = _max_per_for_scale(self.cfg.scale_factor)
        with contextlib.redirect_stdout(io.StringIO()):
            populate_real_syn3a(
                state, counts,
                scale_factor=self.cfg.scale_factor,
                max_per_gene=max_per,
            )
        # Belt-and-suspenders: ensure no instances of knocked-out genes survive.
        if knockout:
            ko_set = set(knockout)
            ids_to_remove = [
                pid for pid, p in state.proteins.items() if p.gene_id in ko_set
            ]
            for pid in ids_to_remove:
                state.proteins.pop(pid, None)
                for _, bucket in list(state.proteins_by_state.items()):
                    if isinstance(bucket, set):
                        bucket.discard(pid)
                    elif isinstance(bucket, list) and pid in bucket:
                        bucket.remove(pid)

        initialize_metabolites(
            state, self._sbml,
            cell_volume_um3=self.cfg.cell_volume_um3,
        )
        initialize_medium(state, self._medium)

        rules = (
            [make_folding_rule(self.cfg.folding_rate_per_s)]
            + self._rev_rules
            + self._extra_rules
            + make_complex_formation_rules(
                self._complexes,
                self.cfg.complex_assembly_rate_per_uM_per_s,
            )
        )
        return state, rules

    # ----- Simulator protocol -----
    def run(
        self,
        knockout: Iterable[str],
        *,
        t_end_s: float,
        sample_dt_s: float,
    ) -> Trajectory:
        from layer2_field.fast_dynamics import FastEventSimulator
        from layer3_reactions.coupled import get_species_count

        self._ensure_setup()
        ko = tuple(sorted(knockout))
        state, rules = self._build_state_and_rules(ko)
        sim = FastEventSimulator(state, rules, mode="gillespie", seed=self.cfg.seed)

        samples: list[Sample] = [_snapshot(state, get_species_count, t=0.0)]
        next_sample = sample_dt_s
        chunk = self.cfg.chunk_dt_s
        while state.time < t_end_s:
            target = min(state.time + chunk, t_end_s)
            sim.run_until(t_end=target, max_events=self.cfg.max_events_per_chunk)
            while next_sample <= state.time and next_sample <= t_end_s + 1e-9:
                samples.append(_snapshot(state, get_species_count, t=next_sample))
                next_sample += sample_dt_s
        # Always include a final-sample at t_end if not already.
        if not samples or samples[-1].t_s < t_end_s - 1e-9:
            samples.append(_snapshot(state, get_species_count, t=t_end_s))
        return Trajectory(tuple(samples))


# ---- helpers -------------------------------------------------------------


def _snapshot(state, get_species_count, *, t: float) -> Sample:
    pools: dict[str, float] = {}
    for key, sid in _POOL_KEY_TO_SPECIES.items():
        try:
            pools[key] = float(get_species_count(state, sid))
        except Exception:
            # Species not present in this model - skip silently. The
            # detector tolerates missing keys.
            pass
    ntp_total = 0.0
    saw_any = False
    for sid in _NTP_COMPONENTS:
        try:
            ntp_total += float(get_species_count(state, sid))
            saw_any = True
        except Exception:
            pass
    if saw_any:
        pools["NTP_TOTAL"] = ntp_total
    return Sample(t_s=t, pools=pools)


def _max_per_for_scale(scale: float) -> int:
    # Mirrors the heuristic in cell_sim/tests/test_knockouts.py.
    if scale <= 0.02: return 10
    if scale <= 0.10: return 50
    if scale <= 0.25: return 125
    if scale <= 0.50: return 250
    return 500
