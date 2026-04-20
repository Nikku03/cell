"""
Layer 2 (rebuilt): Event-driven molecular state tracker.

Replaces the FNO-based field dynamics with the right primitive for the
actual goal: tracking molecular identity and events at microsecond
resolution.

Core ideas:
- Every protein molecule has an identity (gene_of_origin + instance_id)
- Every protein molecule has a state (conformation, modifications, partners)
- Transitions are discrete events with rates
- Timesteps are natural to the slowest event we care about
- An event log is a first-class output — you can query "what happened
  between t=a and t=b" and get the molecular events
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable, Any
from collections import defaultdict
import time
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from layer0_genome.parser import CellSpec


# =============================================================================
# Entity types and state
# =============================================================================
@dataclass
class ProteinInstance:
    """One specific molecule of a protein, with full state."""
    instance_id: int
    gene_id: str
    conformation: str
    modifications: Dict[str, Any] = field(default_factory=dict)
    bound_partners: Set[int] = field(default_factory=set)
    compartment: str = 'cytoplasm'
    birth_time: float = 0.0
    history: List[Tuple[float, str]] = field(default_factory=list)

    def record(self, t: float, what: str):
        self.history.append((t, what))


@dataclass
class MetabolitePool:
    met_id: str
    compartment: str
    count: int
    volume_L: float

    @property
    def concentration_mM(self) -> float:
        N_A = 6.022e23
        return (self.count / N_A) / self.volume_L * 1000.0


@dataclass
class Complex:
    complex_id: int
    member_instance_ids: List[int]
    quaternary_arrangement: str
    formation_time: float
    compartment: str = 'cytoplasm'


# =============================================================================
# Transitions
# =============================================================================
@dataclass
class TransitionRule:
    name: str
    participants: List[str]
    rate: float
    rate_source: str = 'default'
    can_fire: Optional[Callable] = None
    apply: Optional[Callable] = None


@dataclass
class Event:
    time: float
    rule_name: str
    participants: List[int]
    description: str


# =============================================================================
# Cell state
# =============================================================================
class CellState:
    def __init__(self, spec: CellSpec):
        self.spec = spec
        self.time = 0.0
        self.next_instance_id = 1
        self.next_complex_id = 1

        self.proteins: Dict[int, ProteinInstance] = {}
        self.complexes: Dict[int, Complex] = {}
        self.metabolite_pools: Dict[Tuple[str, str], MetabolitePool] = {}

        self.proteins_by_gene: Dict[str, Set[int]] = defaultdict(set)
        self.proteins_by_state: Dict[str, Set[int]] = defaultdict(set)

        self.events: List[Event] = []

    def new_protein(self, gene_id: str, conformation: str = 'native',
                    compartment: str = 'cytoplasm') -> ProteinInstance:
        inst_id = self.next_instance_id
        self.next_instance_id += 1
        p = ProteinInstance(
            instance_id=inst_id, gene_id=gene_id,
            conformation=conformation, compartment=compartment,
            birth_time=self.time,
        )
        self.proteins[inst_id] = p
        self.proteins_by_gene[gene_id].add(inst_id)
        self.proteins_by_state[f"{gene_id}:{conformation}"].add(inst_id)
        return p

    def change_protein_state(self, inst_id: int,
                             new_conformation: Optional[str] = None,
                             add_modification: Optional[Tuple[str, Any]] = None,
                             remove_modification: Optional[str] = None):
        p = self.proteins[inst_id]
        if new_conformation is not None:
            old_key = f"{p.gene_id}:{p.conformation}"
            new_key = f"{p.gene_id}:{new_conformation}"
            self.proteins_by_state[old_key].discard(inst_id)
            self.proteins_by_state[new_key].add(inst_id)
            p.conformation = new_conformation
        if add_modification is not None:
            mod_name, mod_value = add_modification
            p.modifications[mod_name] = mod_value
        if remove_modification is not None:
            p.modifications.pop(remove_modification, None)

    def log_event(self, rule_name: str, participants: List[int],
                  description: str):
        ev = Event(time=self.time, rule_name=rule_name,
                   participants=participants, description=description)
        self.events.append(ev)
        for pid in participants:
            if pid in self.proteins:
                self.proteins[pid].record(self.time, f"{rule_name}: {description}")

    def count_by_state(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.proteins_by_state.items() if v}

    def events_in_window(self, t_start: float, t_end: float) -> List[Event]:
        return [e for e in self.events if t_start <= e.time < t_end]

    def molecule_history(self, inst_id: int) -> List[Tuple[float, str]]:
        return self.proteins[inst_id].history if inst_id in self.proteins else []


# =============================================================================
# Event-driven simulator (Gillespie + tau-leaping)
# =============================================================================
class EventSimulator:
    def __init__(self, state: CellState, rules: List[TransitionRule],
                 mode: str = 'gillespie', tau: float = 1e-3, seed: int = 42):
        self.state = state
        self.rules = rules
        self.mode = mode
        self.tau = tau
        self.rng = np.random.default_rng(seed)

    def _all_propensities(self) -> Tuple[List[float], List[Any]]:
        props, cands = [], []
        for rule in self.rules:
            if rule.can_fire is None:
                props.append(0.0); cands.append(None); continue
            c = rule.can_fire(self.state)
            if not c:
                props.append(0.0); cands.append(None); continue
            props.append(rule.rate * len(c))
            cands.append(c)
        return props, cands

    def step_gillespie(self) -> bool:
        props, cands = self._all_propensities()
        total = sum(props)
        if total <= 0.0:
            return False
        dt = self.rng.exponential(1.0 / total)
        self.state.time += dt
        r = self.rng.random() * total
        cum = 0.0
        for i, p in enumerate(props):
            cum += p
            if r < cum:
                self.rules[i].apply(self.state, cands[i], self.rng)
                return True
        return True

    def step_tau_leap(self, tau: Optional[float] = None) -> int:
        if tau is None:
            tau = self.tau
        props, cands = self._all_propensities()
        n = 0
        for rule, prop, cand in zip(self.rules, props, cands):
            if prop <= 0.0:
                continue
            k = self.rng.poisson(prop * tau)
            for _ in range(int(k)):
                rule.apply(self.state, cand, self.rng)
                n += 1
        self.state.time += tau
        return n

    def run_until(self, t_end: float, max_events: int = 10_000_000,
                  verbose: bool = False) -> Dict:
        t0 = time.time()
        n0 = len(self.state.events)
        if self.mode == 'gillespie':
            while self.state.time < t_end and len(self.state.events) - n0 < max_events:
                if not self.step_gillespie():
                    if verbose:
                        print(f"  No more events possible at t={self.state.time:.6f}s")
                    break
        elif self.mode == 'tau_leap':
            n_steps = max(1, int((t_end - self.state.time) / self.tau))
            for i in range(n_steps):
                self.step_tau_leap()
                if verbose and i % max(1, n_steps // 10) == 0:
                    print(f"  t={self.state.time:.4f}s  events={len(self.state.events)}")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        wall = time.time() - t0
        return {
            't_final': self.state.time,
            'n_events': len(self.state.events) - n0,
            'wall_time_s': wall,
            'events_per_wall_sec': (len(self.state.events) - n0) / max(wall, 1e-9),
        }


# =============================================================================
# Example transition rules
# =============================================================================
def make_example_rules() -> List[TransitionRule]:
    rules = []

    # Folding
    def folding_can_fire(state):
        r = []
        for gene_id in state.proteins_by_gene:
            r.extend(list(state.proteins_by_state.get(f"{gene_id}:unfolded", set())))
        return r

    def folding_apply(state, cands, rng):
        if not cands:
            return
        inst_id = cands[rng.integers(0, len(cands))]
        p = state.proteins[inst_id]
        state.change_protein_state(inst_id, new_conformation='native')
        state.log_event('folding', [inst_id],
                        f"Protein {p.gene_id}#{inst_id} folded to native")

    rules.append(TransitionRule(
        name='folding', participants=['unfolded protein'], rate=100.0,
        can_fire=folding_can_fire, apply=folding_apply,
    ))

    # Phosphorylation
    def phospho_can_fire(state):
        kinases, substrates = [], []
        for gene_id, inst_ids in state.proteins_by_gene.items():
            natives = inst_ids & state.proteins_by_state.get(f"{gene_id}:native", set())
            if 'kinase' in gene_id.lower() or 'kin' in gene_id.lower():
                kinases.extend(natives)
            else:
                substrates.extend(natives)
        return [(k, s) for k in kinases for s in substrates
                if 'phospho_S' not in state.proteins[s].modifications]

    def phospho_apply(state, cands, rng):
        if not cands:
            return
        k, s = cands[rng.integers(0, len(cands))]
        state.change_protein_state(s, add_modification=('phospho_S', True))
        state.log_event(
            'phosphorylation', [k, s],
            f"Kinase {state.proteins[k].gene_id}#{k} phosphorylated "
            f"{state.proteins[s].gene_id}#{s}",
        )

    rules.append(TransitionRule(
        name='phosphorylation', participants=['kinase', 'substrate'],
        rate=0.5, can_fire=phospho_can_fire, apply=phospho_apply,
    ))

    # Dimerization
    def dimer_can_fire(state):
        r = []
        for gene_id in state.proteins_by_gene:
            natives = list(state.proteins_by_state.get(f"{gene_id}:native", set()))
            unbound = [i for i in natives if not state.proteins[i].bound_partners]
            if len(unbound) >= 2:
                for i in range(len(unbound)):
                    for j in range(i+1, len(unbound)):
                        r.append((unbound[i], unbound[j]))
        return r

    def dimer_apply(state, cands, rng):
        if not cands:
            return
        a, b = cands[rng.integers(0, len(cands))]
        pa, pb = state.proteins[a], state.proteins[b]
        if a in pb.bound_partners or b in pa.bound_partners:
            return
        pa.bound_partners.add(b)
        pb.bound_partners.add(a)
        cid = state.next_complex_id
        state.next_complex_id += 1
        state.complexes[cid] = Complex(
            complex_id=cid, member_instance_ids=[a, b],
            quaternary_arrangement='dimer', formation_time=state.time,
            compartment=pa.compartment,
        )
        state.log_event('dimerization', [a, b],
                        f"Dimer #{cid} formed from {pa.gene_id}#{a} + {pb.gene_id}#{b}")

    rules.append(TransitionRule(
        name='dimerization', participants=['monomer', 'monomer'],
        rate=0.01, can_fire=dimer_can_fire, apply=dimer_apply,
    ))

    # Tetramerization
    def tetramer_can_fire(state):
        dimers = [c for c in state.complexes.values()
                  if c.quaternary_arrangement == 'dimer']
        by_gene = defaultdict(list)
        for d in dimers:
            g = state.proteins[d.member_instance_ids[0]].gene_id
            if state.proteins[d.member_instance_ids[1]].gene_id == g:
                by_gene[g].append(d)
        r = []
        for g, ds in by_gene.items():
            for i in range(len(ds)):
                for j in range(i+1, len(ds)):
                    r.append((ds[i], ds[j]))
        return r

    def tetramer_apply(state, cands, rng):
        if not cands:
            return
        d1, d2 = cands[rng.integers(0, len(cands))]
        members = d1.member_instance_ids + d2.member_instance_ids
        for pid in d1.member_instance_ids:
            state.proteins[pid].bound_partners.update(d2.member_instance_ids)
        for pid in d2.member_instance_ids:
            state.proteins[pid].bound_partners.update(d1.member_instance_ids)
        state.complexes.pop(d1.complex_id, None)
        state.complexes.pop(d2.complex_id, None)
        cid = state.next_complex_id
        state.next_complex_id += 1
        state.complexes[cid] = Complex(
            complex_id=cid, member_instance_ids=members,
            quaternary_arrangement='tetramer', formation_time=state.time,
            compartment=state.proteins[members[0]].compartment,
        )
        g = state.proteins[members[0]].gene_id
        state.log_event(
            'tetramerization', members,
            f"Tetramer #{cid} formed from 2 dimers of {g} (instances {members})",
        )

    rules.append(TransitionRule(
        name='tetramerization', participants=['dimer', 'dimer'],
        rate=0.001, can_fire=tetramer_can_fire, apply=tetramer_apply,
    ))

    return rules


# =============================================================================
# Demo
# =============================================================================
if __name__ == "__main__":
    from layer0_genome.parser import build_cell_spec, Protein

    print("=" * 60)
    print("Layer 2 (rebuilt): Event-driven molecular state tracker")
    print("=" * 60)

    spec = build_cell_spec(species='syn3a')
    spec.proteins['kinase_A'] = Protein(
        gene_id='kinase_A', sequence='M'*200, length=200, function_class='enzyme')
    spec.proteins['substrate_B'] = Protein(
        gene_id='substrate_B', sequence='M'*150, length=150, function_class='enzyme')
    spec.proteins['monomer_C'] = Protein(
        gene_id='monomer_C', sequence='M'*100, length=100, function_class='structural')

    state = CellState(spec)
    print("\nPopulating cell with molecules:")
    for _ in range(10):
        state.new_protein('kinase_A', conformation='unfolded')
    for _ in range(50):
        state.new_protein('substrate_B', conformation='unfolded')
    for _ in range(100):
        state.new_protein('monomer_C', conformation='unfolded')

    print(f"  Initial state counts:")
    for k, v in sorted(state.count_by_state().items()):
        print(f"    {k}: {v}")

    rules = make_example_rules()
    sim = EventSimulator(state, rules, mode='gillespie')
    print(f"\nRunning Gillespie simulation for 1 second of cell time...")
    stats = sim.run_until(t_end=1.0, verbose=False)
    print(f"  Wall time: {stats['wall_time_s']:.3f} s")
    print(f"  Events fired: {stats['n_events']}")
    print(f"  Event rate: {stats['events_per_wall_sec']:.0f} events/wall-sec")

    print(f"\nFinal state counts:")
    for k, v in sorted(state.count_by_state().items()):
        print(f"  {k}: {v}")

    print(f"\nEvent log (first 15):")
    for ev in state.events[:15]:
        print(f"  t={ev.time*1e6:8.1f} μs  [{ev.rule_name:16s}]  {ev.description}")
    if len(state.events) > 15:
        print(f"  ... {len(state.events) - 15} more events ...")

    print(f"\nComplexes at end of simulation:")
    for cid, cplx in list(state.complexes.items())[:5]:
        print(f"  Complex #{cid} ({cplx.quaternary_arrangement}): "
              f"members {cplx.member_instance_ids}, "
              f"formed at t={cplx.formation_time*1e6:.1f} μs")
    print(f"  Total complexes: {len(state.complexes)}")

    # Show history of one molecule
    print(f"\nHistory of one specific molecule (instance #1):")
    hist = state.molecule_history(1)
    for t, desc in hist[:10]:
        print(f"  t={t*1e6:8.1f} μs  {desc}")

    print("\nLayer 2 (rebuilt) is working.")
