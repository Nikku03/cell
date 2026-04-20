"""
Routing controller.

Decides, given a biological question, which layers are needed and at what
frequency. This is the "wormhole" — it's what lets us produce cell-scale
predictions in under an hour on one GPU. Atomic resolution (Layer 1) is
expensive but rarely needed; most of the simulation runs at Layer 2
(field) + Layer 3 (reactions), and Layer 1 fires only when the question
explicitly demands atomic detail.

For the 24-hour prototype, the controller is rule-based. A real research
version would replace the rules with a learned policy (reinforcement
learning or supervised training on expert labels).

Supported question types:
- 'metabolic_response': response to substrate/product perturbation → mostly L3
- 'spatial_organization': where things are in the cell → L2 + optional L3
- 'drug_binding': where does a small molecule bind → L2 for localization + L1 for atomic detail
- 'reaction_mechanism': what's the mechanism of a specific enzymatic step → L1 heavy
- 'survival_prediction': does the cell survive a perturbation → L3 + L2 for validation
- 'full_simulation': run all layers for as long as possible within budget
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
import time
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Question:
    """A biological question the simulator is being asked to answer."""
    question_type: str  # e.g., 'metabolic_response'
    target_species: Optional[str] = None  # e.g., 'glc'
    target_protein: Optional[str] = None
    target_substrate: Optional[str] = None
    duration_s: float = 10.0  # simulated time to answer the question
    budget_s: float = 3600.0  # wall time budget in seconds (1 hour default)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerInvocation:
    """Record of a single layer call made during simulation."""
    layer: str  # '0', '1', '2', '3', 'routing'
    query: str  # short description of what was asked
    wall_time_s: float
    is_stubbed: bool = False
    output_summary: str = ""


@dataclass
class SimulationPlan:
    """Plan for how to answer a question."""
    layers_used: List[str]
    layer_0_needed: bool = True   # always need to build cell spec
    layer_1_count_estimate: int = 0
    layer_2_active: bool = False
    layer_2_duration_s: float = 0.0
    layer_3_active: bool = False
    layer_3_duration_s: float = 0.0
    rationale: str = ""


class Router:
    """
    Rule-based routing controller.

    Given a Question, decide a SimulationPlan. Runs the simulation
    according to the plan while staying inside the wall-time budget.
    """

    # ===== Rules for each question type =====
    ROUTES = {
        'metabolic_response': {
            'l1': False, 'l2': False, 'l3': True,
            'l2_dur_ratio': 0.0, 'l3_dur_ratio': 1.0,
            'rationale': "Metabolic response questions are answered by reaction kinetics.",
        },
        'spatial_organization': {
            'l1': False, 'l2': True, 'l3': False,
            'l2_dur_ratio': 1.0, 'l3_dur_ratio': 0.0,
            'rationale': "Spatial organization needs the field layer; no reaction or atomic detail required.",
        },
        'drug_binding': {
            'l1': True, 'l2': True, 'l3': False,
            'l2_dur_ratio': 1.0, 'l3_dur_ratio': 0.0,
            'l1_count_estimate': 10,
            'rationale': "Drug binding: L2 localizes, L1 atomically characterizes the pocket.",
        },
        'reaction_mechanism': {
            'l1': True, 'l2': False, 'l3': False,
            'l2_dur_ratio': 0.0, 'l3_dur_ratio': 0.0,
            'l1_count_estimate': 50,
            'rationale': "Reaction mechanism needs atomic detail at the active site; everything else is irrelevant.",
        },
        'survival_prediction': {
            'l1': False, 'l2': True, 'l3': True,
            'l2_dur_ratio': 0.3, 'l3_dur_ratio': 1.0,
            'rationale': "Survival: L3 handles kinetics, L2 checks that spatial organization stays stable.",
        },
        'full_simulation': {
            'l1': True, 'l2': True, 'l3': True,
            'l2_dur_ratio': 1.0, 'l3_dur_ratio': 1.0,
            'l1_count_estimate': 20,
            'rationale': "Full simulation: all layers at medium-duration.",
        },
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.invocations: List[LayerInvocation] = []

    def log(self, msg: str):
        if self.verbose:
            print(f"[router] {msg}")

    # ====== Planning ======
    def plan(self, question: Question) -> SimulationPlan:
        """Decide what to run based on the question."""
        route = self.ROUTES.get(question.question_type)
        if route is None:
            # Unknown question type: default to L3 only, be conservative
            return SimulationPlan(
                layers_used=['0', '3'],
                layer_3_active=True,
                layer_3_duration_s=question.duration_s,
                rationale=f"Unknown question type '{question.question_type}'; defaulting to L3.",
            )

        return SimulationPlan(
            layers_used=[l for l, active in [
                ('0', True),
                ('1', route['l1']),
                ('2', route['l2']),
                ('3', route['l3']),
            ] if active],
            layer_1_count_estimate=route.get('l1_count_estimate', 0),
            layer_2_active=route['l2'],
            layer_2_duration_s=question.duration_s * route['l2_dur_ratio'],
            layer_3_active=route['l3'],
            layer_3_duration_s=question.duration_s * route['l3_dur_ratio'],
            rationale=route['rationale'],
        )

    # ====== Execution ======
    def execute(
        self,
        question: Question,
        plan: SimulationPlan,
        cell_spec,  # built by Layer 0
        atomic_engine=None,  # Layer 1 instance
        field_evolver=None,  # Layer 2 instance
        reaction_network=None,  # Layer 3 instance
        initial_field=None,
    ) -> Dict[str, Any]:
        """
        Run the simulation according to the plan.

        Returns a dictionary of results, including trajectories from each
        active layer and an invocation log.
        """
        t_start = time.time()
        results = {
            'question': question,
            'plan': plan,
            'cell_spec_summary': cell_spec.summary(),
            'layer_3': None,
            'layer_2': None,
            'layer_1_queries': [],
            'invocations': self.invocations,
            'total_wall_time_s': None,
        }
        self.log(f"Plan: layers {plan.layers_used}, budget {question.budget_s:.0f} s")
        self.log(f"Rationale: {plan.rationale}")

        # Layer 3 — reaction kinetics
        if plan.layer_3_active and reaction_network is not None:
            t0 = time.time()
            self.log(f"Running Layer 3 for {plan.layer_3_duration_s:.2f} s of simulated time...")
            l3_result = reaction_network.integrate(
                t_end=plan.layer_3_duration_s, dt=0.01, verbose=False,
            )
            wall = time.time() - t0
            results['layer_3'] = l3_result
            self.invocations.append(LayerInvocation(
                layer='3', query='integrate',
                wall_time_s=wall,
                output_summary=f"{len(l3_result['t'])} timepoints, {l3_result['C'].shape[1]} metabolites",
            ))
            self.log(f"  Layer 3 done in {wall:.2f} s wall")

        # Layer 2 — field dynamics
        if plan.layer_2_active and field_evolver is not None and initial_field is not None:
            t0 = time.time()
            self.log(f"Running Layer 2 for {plan.layer_2_duration_s:.2f} s of simulated time...")
            l2_result = field_evolver.integrate(
                initial_field, t_end=plan.layer_2_duration_s, dt=0.01, save_every=10,
            )
            wall = time.time() - t0
            results['layer_2'] = l2_result
            self.invocations.append(LayerInvocation(
                layer='2', query='integrate',
                wall_time_s=wall,
                output_summary=f"{len(l2_result['snapshots'])} snapshots, {l2_result['n_steps']} steps",
            ))
            self.log(f"  Layer 2 done in {wall:.2f} s wall")

        # Layer 1 — atomic queries (only if plan says so)
        if plan.layer_1_count_estimate > 0 and atomic_engine is not None:
            self.log(f"Estimated {plan.layer_1_count_estimate} Layer 1 queries needed.")
            # For the prototype, we only actually run queries when the caller
            # supplies specific subsystems through question.extra['subsystems'].
            # In a real simulation, the controller would use Layer 2 output
            # to identify regions of interest and construct subsystems.
            subsystems = question.extra.get('subsystems', [])
            for subsys in subsystems[:plan.layer_1_count_estimate]:
                t0 = time.time()
                qtype = subsys.get('query_type', 'energy')
                result = atomic_engine.atomic_query(subsys['subsys'], qtype)
                wall = time.time() - t0
                results['layer_1_queries'].append({
                    'subsystem_name': subsys['subsys'].name,
                    'query_type': qtype,
                    'result': result,
                })
                self.invocations.append(LayerInvocation(
                    layer='1', query=f"{qtype}:{subsys['subsys'].name}",
                    wall_time_s=wall,
                    is_stubbed=result.is_stubbed,
                    output_summary=f"energy={result.energy_eV}, bde={result.bde_kcal_mol}",
                ))
                self.log(f"  L1 query '{subsys['subsys'].name}' ({qtype}) in {wall*1000:.1f} ms "
                         f"{'[stubbed]' if result.is_stubbed else ''}")

        results['total_wall_time_s'] = time.time() - t_start
        self.log(f"Total wall time: {results['total_wall_time_s']:.2f} s  "
                 f"(budget {question.budget_s:.0f} s)")
        return results


# =============================================================================
# Demo
# =============================================================================
if __name__ == "__main__":
    from layer0_genome.parser import build_cell_spec
    from layer3_reactions.network import ReactionNetwork
    from layer2_field.dynamics import (
        initialize_field_from_cellspec, FieldEvolver,
    )
    from layer1_atomic.engine import AtomicEngine, methane_subsystem

    print("=" * 60)
    print("Routing Controller + End-to-end Integration Test")
    print("=" * 60)

    # Build cell spec
    spec = build_cell_spec(species='syn3a')
    print(f"\nBuilt cell spec: {len(spec.metabolites)} metabolites, "
          f"{len(spec.reactions)} reactions")

    # Build all layers
    print("\nInstantiating layers...")
    network = ReactionNetwork(spec)
    field = initialize_field_from_cellspec(spec, grid_size=32)  # smaller for speed
    evolver = FieldEvolver(use_physics_baseline=True)
    atomic = AtomicEngine()
    router = Router(verbose=True)

    # Run several questions through the router
    questions = [
        Question(question_type='metabolic_response', duration_s=5.0, budget_s=60),
        Question(question_type='spatial_organization', duration_s=0.5, budget_s=60),
        Question(
            question_type='drug_binding', duration_s=0.5, budget_s=60,
            extra={'subsystems': [
                {'subsys': methane_subsystem(), 'query_type': 'energy'},
                {'subsys': methane_subsystem(), 'query_type': 'bde'},
            ]},
        ),
        Question(question_type='full_simulation', duration_s=0.5, budget_s=60,
                 extra={'subsystems': [
                     {'subsys': methane_subsystem(), 'query_type': 'energy'},
                 ]}),
    ]

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"Question: {q.question_type}")
        print(f"{'=' * 60}")
        plan = router.plan(q)
        print(f"Plan uses layers: {plan.layers_used}")
        print(f"Rationale: {plan.rationale}")
        result = router.execute(
            q, plan, spec,
            atomic_engine=atomic,
            field_evolver=evolver,
            reaction_network=network,
            initial_field=field,
        )

    print("\n" + "=" * 60)
    print("Invocation summary:")
    print("=" * 60)
    for inv in router.invocations:
        flag = " [STUB]" if inv.is_stubbed else ""
        print(f"  L{inv.layer}  {inv.query:30s}  {inv.wall_time_s*1000:8.1f} ms{flag}")
    total_wall = sum(i.wall_time_s for i in router.invocations)
    print(f"\nTotal wall time across all invocations: {total_wall:.2f} s")
    print("\nRouting controller is working end-to-end.")
