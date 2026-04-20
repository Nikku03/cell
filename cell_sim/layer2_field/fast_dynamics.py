"""
Vectorized Gillespie simulator (phase 1).

Bit-identical to the reference `EventSimulator` but evaluates all
compiled Michaelis-Menten propensities in a handful of numpy operations
per step instead of 200+ Python closure calls.

The key design choice: all per-rule data (substrate indices, stoichiometry,
Km values, enzyme sourcing) is pre-flattened at init time into padded
2D numpy arrays of shape (n_compiled_rules, max_per_rule). One step of
the hot path is ~10 numpy vectorised ops total, regardless of rule count.

Rules with `compiled_spec = None` (folding, complex formation, gene
expression, novel substrates) still run through their Python
can_fire/apply closures. Those are a tiny fraction of steps in
Priority 1.5 — the architecture tolerates this hybrid cleanly.

Correctness is covered by tests/test_fast_equivalence.py.
Next phase of optimisation: the next-reaction / Gibson-Bruck method.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Any

import numpy as np

from layer2_field.dynamics import (
    CellState, TransitionRule,
)
from layer3_reactions.coupled import COUNTABLE_THRESHOLD, AVOGADRO


# ============================================================================
# FastEventSimulator
# ============================================================================
class FastEventSimulator:

    def __init__(self, state: CellState, rules: List[TransitionRule],
                 mode: str = 'gillespie', tau: float = 1e-3, seed: int = 42):
        if mode != 'gillespie':
            raise NotImplementedError(
                'FastEventSimulator phase 1 supports gillespie only')
        self.state = state
        self.rules = rules
        self.mode = mode
        self.tau = tau
        self.rng = np.random.default_rng(seed)
        self._n_rules = len(rules)

        # ------------------------------------------------------------------
        # Build the species index
        # ------------------------------------------------------------------
        self._species_order: List[str] = []
        self._species_to_idx: Dict[str, int] = {}
        def _register(sid: str) -> int:
            if sid not in self._species_to_idx:
                self._species_to_idx[sid] = len(self._species_order)
                self._species_order.append(sid)
            return self._species_to_idx[sid]
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

        # ------------------------------------------------------------------
        # Partition rules: compiled-MM vs python-closure
        # ------------------------------------------------------------------
        self._compiled_rule_indices: List[int] = []
        self._python_rule_indices: List[int] = []
        specs = []
        for i, rule in enumerate(rules):
            spec = rule.compiled_spec
            if spec is not None and spec.get('kind') == 'mm':
                self._compiled_rule_indices.append(i)
                specs.append(spec)
            else:
                self._python_rule_indices.append(i)
        self._python_rule_indices_set = set(self._python_rule_indices)
        self._n_compiled = len(self._compiled_rule_indices)

        # ------------------------------------------------------------------
        # Flatten every compiled rule into padded 2D arrays
        # ------------------------------------------------------------------
        if self._n_compiled > 0:
            max_sub = max(len(s['substrates']) for s in specs)
            max_prd = max(len(s['products']) for s in specs)
            # Km list can be shorter than substrates (Km may be missing)
            max_km = max(
                sum(1 for sid in s['substrate_ids_for_sat']
                     if (s['Km'].get(sid) or 0) > 0
                     and sid in self._species_to_idx)
                for s in specs
            )
            max_km = max(max_km, 1)   # don't allow zero-width
        else:
            max_sub = max_prd = max_km = 1

        # Substrate tables (padded). Padded entries use stoich=inf so that
        # capacity = count / stoich = 0 and does NOT affect min().
        # Note: we use count=inf + stoich=1 would give capacity=inf (bad).
        # Instead use a mask: compute raw counts/stoich and override padded
        # slots with +inf so min() ignores them.
        self.C_sub_idx    = np.zeros((self._n_compiled, max_sub), dtype=np.int64)
        self.C_sub_stoich = np.ones ((self._n_compiled, max_sub), dtype=np.float64)
        self.C_sub_mask   = np.zeros((self._n_compiled, max_sub), dtype=bool)
        self.C_prd_idx    = np.zeros((self._n_compiled, max_prd), dtype=np.int64)
        self.C_prd_stoich = np.zeros((self._n_compiled, max_prd), dtype=np.float64)
        self.C_prd_mask   = np.zeros((self._n_compiled, max_prd), dtype=bool)
        self.C_km_idx     = np.zeros((self._n_compiled, max_km), dtype=np.int64)
        self.C_km_val     = np.zeros((self._n_compiled, max_km), dtype=np.float64)
        self.C_km_mask    = np.zeros((self._n_compiled, max_km), dtype=bool)
        self.C_kcat       = np.zeros(self._n_compiled, dtype=np.float64)
        self.C_include_sat = np.zeros(self._n_compiled, dtype=bool)
        self.C_rule_idx   = np.array(self._compiled_rule_indices, dtype=np.int64)

        # Enzyme sourcing: per-rule list of proteins_by_state keys
        self._enzyme_state_keys: List[List[str]] = []

        for k, spec in enumerate(specs):
            for j, (sid, st) in enumerate(spec['substrates']):
                self.C_sub_idx[k, j]    = self._species_to_idx[sid]
                self.C_sub_stoich[k, j] = float(st)
                self.C_sub_mask[k, j]   = True
            for j, (sid, st) in enumerate(spec['products']):
                self.C_prd_idx[k, j]    = self._species_to_idx[sid]
                self.C_prd_stoich[k, j] = float(st)
                self.C_prd_mask[k, j]   = True
            km_pairs = []
            for sid in spec['substrate_ids_for_sat']:
                km = spec['Km'].get(sid) or 0.0
                if km > 0 and sid in self._species_to_idx:
                    km_pairs.append((self._species_to_idx[sid], km))
            for j, (idx, v) in enumerate(km_pairs):
                self.C_km_idx[k, j]  = idx
                self.C_km_val[k, j]  = v
                self.C_km_mask[k, j] = True
            self.C_kcat[k] = float(spec['kcat'])
            self.C_include_sat[k] = bool(spec['include_saturation'])
            self._enzyme_state_keys.append(
                [f'{locus}:native' for locus in spec['enzyme_loci']]
            )

        # Enzyme counts per compiled rule — cached, refreshed after events
        # that could change protein states (python-closure fallback rules).
        self.C_enzyme_counts = np.zeros(self._n_compiled, dtype=np.int64)
        self._enzyme_counts_dirty = True

        # Python-closure rule cache. Folding and complex-formation propensities
        # depend only on protein states (not metabolite counts), so between
        # python-rule events they never change. MM events never touch protein
        # states → python-rule cache stays valid through 99%+ of Gillespie steps.
        self._py_props_cache = np.zeros(self._n_rules, dtype=np.float64)
        self._py_cands_cache: List[Any] = [None] * self._n_rules
        self._py_cache_valid = False

        # Counts sync: self._counts mirrors state.metabolite_counts, but we
        # update it in-place when a compiled MM rule fires. A python-closure
        # rule might have touched counts (novel substrates, etc.), so we
        # mark dirty and do a full dict→array sync on the next step.
        self._counts_dirty = False

        # O(1) lookup: global rule_idx -> row in compiled tables
        self._rule_to_k: Dict[int, int] = {
            int(self.C_rule_idx[k]): k for k in range(self._n_compiled)
        }

        # Cache the per-rule saturation we just computed in _step, so
        # _apply can rebuild the cands list without redoing the math.
        self._last_saturation = np.ones(self._n_compiled, dtype=np.float64)
        self._props = np.zeros(self._n_rules, dtype=np.float64)
        self._cands_scratch: List[Any] = [None] * self._n_rules
        # For apply after firing a compiled rule — need original species
        # counts for the cands tuple the rule.apply expects
        self._last_compiled_enzyme_list: Dict[int, List[int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_until(self, t_end: float, max_events: int = 10_000_000,
                   verbose: bool = False) -> Dict:
        t0 = time.time()
        n0 = len(self.state.events)
        while (self.state.time < t_end
                and len(self.state.events) - n0 < max_events):
            if not self._step():
                if verbose:
                    print(f'  No more events possible at t={self.state.time:.6f}s')
                break
        wall = time.time() - t0
        return {
            't_final': self.state.time,
            'n_events': len(self.state.events) - n0,
            'wall_time_s': wall,
            'events_per_wall_sec': ((len(self.state.events) - n0)
                                     / max(wall, 1e-9)),
        }

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------
    def _step(self) -> bool:
        # --- refresh enzyme counts if needed ---
        if self._enzyme_counts_dirty:
            pbs = self.state.proteins_by_state
            for k in range(self._n_compiled):
                n = 0
                for key in self._enzyme_state_keys[k]:
                    n += len(pbs.get(key, ()))
                self.C_enzyme_counts[k] = n
            self._enzyme_counts_dirty = False

        # --- pull dict counts into the array ---
        # Only sync if a python-closure rule last fired (which might have
        # modified metabolite counts unknown to us, e.g. novel-substrate
        # rules). Compiled MM fires update self._counts in-place in _apply,
        # so no sync is needed between MM-only event runs.
        if self._counts_dirty:
            for sid, c in self.state.metabolite_counts.items():
                idx = self._species_to_idx.get(sid)
                if idx is not None:
                    self._counts[idx] = c
            self._counts_dirty = False

        # --- vectorised compiled-rule propensities ---
        self._props.fill(0.0)
        if self._n_compiled > 0:
            # Effective counts (infinity sentinel for infinite species)
            eff = np.where(self._is_infinite, self._inf_value, self._counts)
            # mM concentrations (for MM saturation). Match count_to_mM's
            # floating-point-exact order of operations: (count / AVOGADRO)
            # / vol_L * 1000.0. Doing the two divisions separately matters
            # for bit-identical reproducibility with EventSimulator.
            counts_mM = (eff / AVOGADRO) / self._vol_L * 1000.0
            counts_mM = np.where(self._is_infinite, 1000.0, counts_mM)

            # Gather substrate counts and compute min-capacity per rule.
            # Padded slots are masked to +inf so they don't become the min.
            sub_counts = eff[self.C_sub_idx]                            # (R, max_sub)
            caps = sub_counts / self.C_sub_stoich                       # (R, max_sub)
            caps = np.where(self.C_sub_mask, caps, np.inf)
            min_avail = caps.min(axis=1)                                # (R,)

            # Saturation factor per rule.
            # For masked-out Km slots, ratio should be 1 (multiplicative identity).
            km_c = counts_mM[self.C_km_idx]                             # (R, max_km)
            ratios = km_c / (km_c + self.C_km_val)                      # (R, max_km)
            ratios = np.where(self.C_km_mask, ratios, 1.0)
            saturation = ratios.prod(axis=1)                            # (R,)
            # If a rule opted out of saturation entirely, force sat=1
            saturation = np.where(self.C_include_sat, saturation, 1.0)
            # Cache for _apply reuse
            self._last_saturation = saturation

            # n_effective = max(1, min(100, round(E * sat))) — matches
            # reversible.py:177 exactly.
            n_eff = np.clip(np.round(self.C_enzyme_counts * saturation),
                             1, 100).astype(np.int64)

            # Valid rule = have enzyme AND at least one unit of substrate
            valid = (self.C_enzyme_counts > 0) & (min_avail >= 1.0)
            compiled_props = np.where(valid, self.C_kcat * n_eff, 0.0)

            # Scatter into full rule-indexed propensity vector
            self._props[self.C_rule_idx] = compiled_props

        # --- python-closure fallback (cached) ---
        # These rules' propensities only change after another python-rule
        # event fires. Between such events (99.3% of steps for Priority 1.5)
        # we can re-use the cached values.
        if not self._py_cache_valid:
            for i in self._python_rule_indices:
                rule = self.rules[i]
                if rule.can_fire is None:
                    self._py_cands_cache[i] = None
                    self._py_props_cache[i] = 0.0
                    continue
                c = rule.can_fire(self.state)
                if not c:
                    self._py_cands_cache[i] = None
                    self._py_props_cache[i] = 0.0
                    continue
                self._py_cands_cache[i] = c
                self._py_props_cache[i] = rule.rate * len(c)
            self._py_cache_valid = True

        # Scatter cached python propensities into props.
        # cands_scratch gets a pointer to the cache — no copy.
        for i in self._python_rule_indices:
            self._props[i] = self._py_props_cache[i]
            self._cands_scratch[i] = self._py_cands_cache[i]

        # --- Gillespie draws ---
        # Match the reference EventSimulator's floating-point behaviour
        # exactly: use Python's builtin sum (sequential over the list) and
        # convert to Python float before sampling, so that tiny FP drift
        # between numpy tree-reduce and sequential sum can't flip which
        # rule wins the categorical draw at a propensity boundary.
        total = float(sum(self._props.tolist()))
        if total <= 0.0:
            return False
        dt = self.rng.exponential(1.0 / total)
        self.state.time += dt
        r = self.rng.random() * total
        cum = 0.0
        chosen = -1
        props_list = self._props.tolist()
        for i in range(self._n_rules):
            cum += props_list[i]
            if r < cum:
                chosen = i
                break
        if chosen < 0:
            chosen = self._n_rules - 1  # numerical safety

        self._apply(chosen)
        return True

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------
    def _apply(self, rule_idx: int) -> None:
        rule = self.rules[rule_idx]
        spec = rule.compiled_spec

        if spec is not None and spec.get('kind') == 'mm':
            # O(1) lookup; avoid np.where
            k = self._rule_to_k.get(rule_idx)
            if k is None:
                return

            enzyme_instances: List[int] = []
            pbs = self.state.proteins_by_state
            for key in self._enzyme_state_keys[k]:
                enzyme_instances.extend(pbs.get(key, ()))
            if not enzyme_instances:
                return

            # Rebuild the cands list can_fire would have returned, using
            # the saturation we already computed in _step (not recomputed).
            # min_avail: the smallest capacity across substrates, matching
            # can_fire's return value. Compute locally but cheaply.
            min_avail = float('inf')
            for j in range(self.C_sub_idx.shape[1]):
                if not self.C_sub_mask[k, j]:
                    continue
                idx = int(self.C_sub_idx[k, j])
                count = (self._inf_value if self._is_infinite[idx]
                         else int(self._counts[idx]))
                cap = count / float(self.C_sub_stoich[k, j])
                if cap < min_avail:
                    min_avail = cap

            sat = float(self._last_saturation[k])
            n_eff = max(1, min(100, int(round(len(enzyme_instances) * sat))))
            cands = [(enzyme_instances, min_avail)] * n_eff
            rule.apply(self.state, cands, self.rng)

            # Update self._counts in place for the species this rule touched.
            # The rule.apply already updated state.metabolite_counts; we mirror
            # by re-reading only those species from the dict. This is O(few)
            # instead of O(n_species) and preserves exact dict semantics
            # (including the clamp-to-zero in update_species_count).
            touched_idx: List[int] = []
            for j in range(self.C_sub_idx.shape[1]):
                if self.C_sub_mask[k, j]:
                    touched_idx.append(int(self.C_sub_idx[k, j]))
            for j in range(self.C_prd_idx.shape[1]):
                if self.C_prd_mask[k, j]:
                    touched_idx.append(int(self.C_prd_idx[k, j]))
            for idx in set(touched_idx):
                if self._is_infinite[idx]:
                    continue
                sid = self._species_order[idx]
                self._counts[idx] = self.state.metabolite_counts.get(sid, 0)
        else:
            cands = self._cands_scratch[rule_idx]
            if cands is None or rule.apply is None:
                return
            rule.apply(self.state, cands, self.rng)

        # Python rules can change protein states (folding / assembly) or,
        # for novel-substrate rules, metabolite counts. Invalidate both
        # caches defensively.
        if rule_idx in self._python_rule_indices_set:
            self._enzyme_counts_dirty = True
            self._py_cache_valid = False
            self._counts_dirty = True
