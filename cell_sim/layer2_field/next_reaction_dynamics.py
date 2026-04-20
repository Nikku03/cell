"""
Next-reaction method simulator (phase 2).

The Gibson-Bruck next-reaction method maintains a priority queue of
per-rule putative firing times and a dependency graph mapping species
to rules whose propensities depend on them. Per step:

  1. Pop the soonest rule from the heap
  2. Fire it (update metabolite counts, protein states)
  3. For each rule affected by species that changed:
       - recompute propensity
       - rescale its putative time (Gibson-Bruck) or resample
       - reinsert in heap

Complexity per step: O(log R + k) where R = total rules and k = avg
number of dependent rules per species. For Syn3A, k is typically 2-10.

Compared to the Direct Method (Phase 1), which recomputes all 250+
propensities per step, Phase 2 recomputes only what changed. The
algorithmic win stacks multiplicatively with Phase 1's numpy win.

NOT bit-identical to the Direct Method simulators. The two methods use
different RNG consumption patterns — Direct draws once per step for dt,
NextReaction draws once per rule per activation. Both are correct
implementations of the Gillespie SSA; they produce statistically
equivalent trajectories but different event sequences for a given seed.

Correctness verified by comparing metabolite counts and event-type
distributions to Phase 1 — see tests/test_next_reaction.py.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from layer2_field.dynamics import CellState, TransitionRule
from layer3_reactions.coupled import COUNTABLE_THRESHOLD, AVOGADRO


class NextReactionSimulator:
    """
    Gibson-Bruck next-reaction method simulator.

    Constructor signature matches EventSimulator / FastEventSimulator.
    """

    INFINITY_TIME = math.inf

    def __init__(self, state: CellState, rules: List[TransitionRule],
                 mode: str = 'gillespie', tau: float = 1e-3, seed: int = 42):
        if mode != 'gillespie':
            raise NotImplementedError('phase 2 supports gillespie only')
        self.state = state
        self.rules = rules
        self.mode = mode
        self.tau = tau
        self.rng = np.random.default_rng(seed)
        self._n_rules = len(rules)

        # -----------------------------------------------------------
        # Species index
        # -----------------------------------------------------------
        self._species_order: List[str] = []
        self._species_to_idx: Dict[str, int] = {}
        def _register(sid: str):
            if sid not in self._species_to_idx:
                self._species_to_idx[sid] = len(self._species_order)
                self._species_order.append(sid)
        for sid in state.metabolite_counts:
            _register(sid)
        for rule in rules:
            if rule.compiled_spec is None:
                continue
            for s, _ in rule.compiled_spec.get('substrates', ()):
                _register(s)
            for s, _ in rule.compiled_spec.get('products', ()):
                _register(s)
        self._n_species = len(self._species_order)

        self._counts = np.zeros(self._n_species, dtype=np.int64)
        for sid, c in state.metabolite_counts.items():
            self._counts[self._species_to_idx[sid]] = c
        self._is_infinite = np.zeros(self._n_species, dtype=bool)
        for sid in state.metabolite_infinite:
            idx = self._species_to_idx.get(sid)
            if idx is not None:
                self._is_infinite[idx] = True
        self._inf_value = COUNTABLE_THRESHOLD * 10
        self._vol_L = state.metabolite_volume_L

        # -----------------------------------------------------------
        # Partition rules and build compiled-MM data (as scalar per-rule
        # lists for per-rule propensity computation — we don't need
        # vectorisation here because we evaluate one rule at a time).
        # -----------------------------------------------------------
        self._compiled_rule_indices: List[int] = []
        self._python_rule_indices: List[int] = []
        self._compiled_data: Dict[int, dict] = {}   # rule_idx -> {...}

        for i, rule in enumerate(rules):
            spec = rule.compiled_spec
            if spec is not None and spec.get('kind') == 'mm':
                self._compiled_rule_indices.append(i)
                sub = [(self._species_to_idx[s], float(st))
                        for s, st in spec['substrates']]
                prd = [(self._species_to_idx[s], float(st))
                        for s, st in spec['products']]
                km = []
                for sid in spec['substrate_ids_for_sat']:
                    v = spec['Km'].get(sid) or 0.0
                    if v > 0 and sid in self._species_to_idx:
                        km.append((self._species_to_idx[sid], float(v)))
                self._compiled_data[i] = {
                    'sub': sub,
                    'sub_arr_idx': np.array([s for s, _ in sub], dtype=np.int64),
                    'sub_arr_stoich': np.array([st for _, st in sub], dtype=np.float64),
                    'prd': prd,
                    'prd_arr_idx': np.array([s for s, _ in prd], dtype=np.int64),
                    'prd_arr_stoich': np.array([st for _, st in prd], dtype=np.float64),
                    'km': km,
                    'kcat': float(spec['kcat']),
                    'include_sat': bool(spec['include_saturation']),
                    'enzyme_keys': [f'{loc}:native' for loc in spec['enzyme_loci']],
                }
            else:
                self._python_rule_indices.append(i)
        self._python_rule_indices_set = set(self._python_rule_indices)
        self._compiled_rule_indices_set = set(self._compiled_rule_indices)

        # -----------------------------------------------------------
        # Dependency graph: for each species, the list of rules whose
        # propensity depends on it (as substrate or Km species; products
        # don't affect propensity).
        # -----------------------------------------------------------
        self._species_dependents: List[List[int]] = [[] for _ in range(self._n_species)]
        for rule_idx in self._compiled_rule_indices:
            cd = self._compiled_data[rule_idx]
            touched = set()
            for s, _ in cd['sub']:
                touched.add(s)
            for s, _ in cd['km']:
                touched.add(s)
            for s in touched:
                self._species_dependents[s].append(rule_idx)

        # -----------------------------------------------------------
        # Per-rule state: propensity, putative_time, generation counter
        # -----------------------------------------------------------
        self._propensities = np.zeros(self._n_rules, dtype=np.float64)
        self._putative_times = np.full(self._n_rules, self.INFINITY_TIME,
                                         dtype=np.float64)
        self._generation = np.zeros(self._n_rules, dtype=np.int64)
        # Heap of (putative_time, generation_counter, rule_idx). Entries
        # whose generation doesn't match _generation[rule_idx] are stale.
        self._heap: List[Tuple[float, int, int]] = []

        # Cached python-rule candidates (for apply) — refreshed each time
        # any python rule fires.
        self._py_cands_cache: Dict[int, Any] = {}
        self._py_cache_valid = False

        # Enzyme count cache: per compiled rule
        self._enzyme_counts: Dict[int, int] = {}
        self._enzyme_counts_dirty = True

        # Debug counters
        self._n_heap_pops = 0
        self._n_stale_pops = 0
        self._n_rule_updates = 0

        # -----------------------------------------------------------
        # Initialise all propensities and putative times
        # -----------------------------------------------------------
        self._initialize_times()

    # =======================================================================
    # Public API
    # =======================================================================
    def run_until(self, t_end: float, max_events: int = 10_000_000,
                   verbose: bool = False) -> Dict:
        t0 = time.time()
        n0 = len(self.state.events)
        while (self.state.time < t_end
                and len(self.state.events) - n0 < max_events):
            if not self._step():
                if verbose:
                    print(f'  No more events at t={self.state.time:.6f}s')
                break
        wall = time.time() - t0
        return {
            't_final': self.state.time,
            'n_events': len(self.state.events) - n0,
            'wall_time_s': wall,
            'events_per_wall_sec': (len(self.state.events) - n0) / max(wall, 1e-9),
            'heap_pops': self._n_heap_pops,
            'stale_pops': self._n_stale_pops,
            'rule_updates': self._n_rule_updates,
        }

    # =======================================================================
    # Propensity computation (per-rule, scalar)
    # =======================================================================
    def _compute_mm_propensity(self, rule_idx: int) -> float:
        """Compute propensity for one compiled MM rule using current state."""
        cd = self._compiled_data[rule_idx]
        n_enz = self._enzyme_counts.get(rule_idx, 0)
        if n_enz == 0:
            return 0.0

        # Substrate availability check (min capacity)
        min_avail = math.inf
        for sp_idx, stoich in cd['sub']:
            c = (self._inf_value if self._is_infinite[sp_idx]
                 else int(self._counts[sp_idx]))
            cap = c / stoich
            if cap < min_avail:
                min_avail = cap
        if min_avail < 1:
            return 0.0

        # MM saturation. Replicate Python reversible.py exactly:
        # product of c_mM/(c_mM + Km) across Km species, in the order
        # they were declared. Use same count_to_mM FP semantics.
        if cd['include_sat']:
            product = 1.0
            for sp_idx, km in cd['km']:
                c = (self._inf_value if self._is_infinite[sp_idx]
                     else int(self._counts[sp_idx]))
                # Match count_to_mM: (count / AVOGADRO) / vol_L * 1000
                c_mM = 1000.0 if self._is_infinite[sp_idx] \
                    else (c / AVOGADRO) / self._vol_L * 1000.0
                product *= c_mM / (c_mM + km)
            sat = product
        else:
            sat = 1.0

        n_eff = max(1, min(100, int(round(n_enz * sat))))
        return cd['kcat'] * n_eff

    def _compute_python_propensity(self, rule_idx: int) -> Tuple[float, Any]:
        """Call the Python closure. Returns (propensity, cands)."""
        rule = self.rules[rule_idx]
        if rule.can_fire is None:
            return 0.0, None
        cands = rule.can_fire(self.state)
        if not cands:
            return 0.0, None
        return rule.rate * len(cands), cands

    # =======================================================================
    # Heap management
    # =======================================================================
    def _initialize_times(self) -> None:
        # Refresh enzyme counts
        self._refresh_enzyme_counts()

        # Compiled rules
        for rule_idx in self._compiled_rule_indices:
            a = self._compute_mm_propensity(rule_idx)
            self._propensities[rule_idx] = a
            if a > 0.0:
                tau = self.rng.exponential(1.0 / a)
                t_k = self.state.time + tau
            else:
                t_k = self.INFINITY_TIME
            self._putative_times[rule_idx] = t_k
            if t_k < self.INFINITY_TIME:
                heapq.heappush(self._heap,
                                (t_k, int(self._generation[rule_idx]), rule_idx))

        # Python rules
        for rule_idx in self._python_rule_indices:
            a, cands = self._compute_python_propensity(rule_idx)
            self._propensities[rule_idx] = a
            self._py_cands_cache[rule_idx] = cands
            if a > 0.0:
                tau = self.rng.exponential(1.0 / a)
                t_k = self.state.time + tau
            else:
                t_k = self.INFINITY_TIME
            self._putative_times[rule_idx] = t_k
            if t_k < self.INFINITY_TIME:
                heapq.heappush(self._heap,
                                (t_k, int(self._generation[rule_idx]), rule_idx))
        self._py_cache_valid = True

    def _refresh_enzyme_counts(self) -> None:
        pbs = self.state.proteins_by_state
        for rule_idx in self._compiled_rule_indices:
            cd = self._compiled_data[rule_idx]
            n = 0
            for key in cd['enzyme_keys']:
                n += len(pbs.get(key, ()))
            self._enzyme_counts[rule_idx] = n
        self._enzyme_counts_dirty = False

    def _update_rule_time(self, rule_idx: int, now: float,
                           is_firing_rule: bool) -> None:
        """
        Recompute propensity for rule_idx and update its putative time.

        is_firing_rule=True means this is the rule that just fired
        — its putative time must be regenerated from scratch (not rescaled).
        For other affected rules, use Gibson-Bruck scaling to preserve
        correctness without redrawing.
        """
        old_a = self._propensities[rule_idx]
        old_t = self._putative_times[rule_idx]

        # Compute new propensity
        if rule_idx in self._compiled_rule_indices_set:
            new_a = self._compute_mm_propensity(rule_idx)
        else:
            new_a, cands = self._compute_python_propensity(rule_idx)
            self._py_cands_cache[rule_idx] = cands

        self._propensities[rule_idx] = new_a

        if is_firing_rule:
            # Regenerate from scratch
            if new_a > 0.0:
                tau = self.rng.exponential(1.0 / new_a)
                new_t = now + tau
            else:
                new_t = self.INFINITY_TIME
        else:
            # Gibson-Bruck scaling
            if old_a > 0.0 and new_a > 0.0 and old_t < self.INFINITY_TIME:
                new_t = now + (old_t - now) * (old_a / new_a)
            elif new_a > 0.0:
                # Rule reactivated from zero
                tau = self.rng.exponential(1.0 / new_a)
                new_t = now + tau
            else:
                new_t = self.INFINITY_TIME

        self._putative_times[rule_idx] = new_t
        self._generation[rule_idx] += 1            # invalidate old heap entry
        self._n_rule_updates += 1

        if new_t < self.INFINITY_TIME:
            heapq.heappush(self._heap,
                            (new_t, int(self._generation[rule_idx]), rule_idx))

    # =======================================================================
    # Step
    # =======================================================================
    def _step(self) -> bool:
        # If enzyme cache dirty, refresh and rebuild all compiled rule times
        # (the affected set is all compiled rules since enzyme counts may
        # have changed for any of them after a python event).
        # This is already handled at the end of _step when a python rule fires.

        # Pop the next valid heap entry
        while self._heap:
            t_next, gen, rule_idx = heapq.heappop(self._heap)
            self._n_heap_pops += 1
            if gen != int(self._generation[rule_idx]):
                self._n_stale_pops += 1
                continue    # stale entry
            break
        else:
            return False    # heap empty

        # Fire this rule
        self.state.time = t_next
        fired_ok = self._fire(rule_idx)
        if not fired_ok:
            # Rule fired-but-noop (e.g. apply re-verified and bailed). Just
            # regenerate its time and keep going.
            self._update_rule_time(rule_idx, now=self.state.time,
                                     is_firing_rule=True)
            return True

        # Determine which species' counts changed
        # - For compiled MM: substrates and products
        # - For python rule: conservative — could be anything, but most
        #   python rules (folding/assembly) don't touch counts at all
        touched_species: List[int] = []
        if rule_idx in self._compiled_rule_indices_set:
            cd = self._compiled_data[rule_idx]
            touched_species.extend(cd['sub_arr_idx'].tolist())
            touched_species.extend(cd['prd_arr_idx'].tolist())
            # After compiled rule fires, refresh our count mirror for
            # the touched species from the authoritative dict.
            touched_species_unique = list(set(touched_species))
            for idx in touched_species_unique:
                if self._is_infinite[idx]:
                    continue
                sid = self._species_order[idx]
                self._counts[idx] = self.state.metabolite_counts.get(sid, 0)
            touched_species = touched_species_unique

        # Affected rules: union of species_dependents over touched species,
        # plus the rule that just fired (its propensity must be recomputed)
        affected: set = {rule_idx}
        for s in touched_species:
            for r in self._species_dependents[s]:
                affected.add(r)

        # Python rule firing → invalidate python cache + enzyme counts,
        # and mark all python rules as affected (they depend on protein
        # states which changed). Also mark all compiled rules' enzyme
        # counts as stale.
        if rule_idx in self._python_rule_indices_set:
            self._enzyme_counts_dirty = True
            self._refresh_enzyme_counts()
            # Any compiled rule whose enzyme count changed is "affected"
            # — all of them, since we don't track per-rule enzyme deltas.
            affected.update(self._compiled_rule_indices)
            # All python rules are affected (protein state changed)
            affected.update(self._python_rule_indices)
            # We must also resync metabolite counts (for novel-substrate
            # rules that update them)
            for sid in self.state.metabolite_counts:
                idx = self._species_to_idx.get(sid)
                if idx is not None:
                    self._counts[idx] = self.state.metabolite_counts[sid]

        # Now update affected rules. Firing rule uses fresh sample;
        # others use Gibson-Bruck scaling.
        for r in affected:
            self._update_rule_time(r, now=self.state.time,
                                     is_firing_rule=(r == rule_idx))

        return True

    # =======================================================================
    # Fire (apply the rule)
    # =======================================================================
    def _fire(self, rule_idx: int) -> bool:
        """
        Apply the rule. Returns True if it ran, False if it was a noop
        (e.g. apply re-verified substrate and bailed).
        """
        rule = self.rules[rule_idx]
        spec = rule.compiled_spec

        if spec is not None and spec.get('kind') == 'mm':
            cd = self._compiled_data[rule_idx]
            pbs = self.state.proteins_by_state
            enzyme_instances: List[int] = []
            for key in cd['enzyme_keys']:
                enzyme_instances.extend(pbs.get(key, ()))
            if not enzyme_instances:
                return False

            # Build cands the way reversible.py's apply expects.
            # Use current counts (not eff) for min_avail; it's a bookkeeping
            # value, not used in firing math.
            min_avail = math.inf
            for sp_idx, stoich in cd['sub']:
                c = int(self._counts[sp_idx])
                if self._is_infinite[sp_idx]:
                    c = self._inf_value
                cap = c / stoich
                if cap < min_avail:
                    min_avail = cap

            # Saturation (same math as _compute_mm_propensity)
            if cd['include_sat']:
                product = 1.0
                for sp_idx, km in cd['km']:
                    c = (self._inf_value if self._is_infinite[sp_idx]
                         else int(self._counts[sp_idx]))
                    c_mM = (1000.0 if self._is_infinite[sp_idx]
                             else (c / AVOGADRO) / self._vol_L * 1000.0)
                    product *= c_mM / (c_mM + km)
                sat = product
            else:
                sat = 1.0
            n_eff = max(1, min(100, int(round(len(enzyme_instances) * sat))))
            cands = [(enzyme_instances, min_avail)] * n_eff
            rule.apply(self.state, cands, self.rng)
            return True
        else:
            cands = self._py_cands_cache.get(rule_idx)
            if cands is None or rule.apply is None:
                return False
            rule.apply(self.state, cands, self.rng)
            return True
