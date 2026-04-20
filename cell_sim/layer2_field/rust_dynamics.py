"""
Rust-backed FastEventSimulator.

Identical Python-visible behaviour to `FastEventSimulator` — same seeded
event sequence, metabolite trajectories, and event log — but the per-step
propensity computation runs in Rust via the `cell_sim_rust` extension.

All RNG draws and the categorical-selection cumsum stay in Python so
that bit-identity with the reference simulator is preserved exactly.

Build and install the Rust extension first:

    cd cell_sim_rust
    maturin build --release
    pip install --force-reinstall target/wheels/cell_sim_rust-*.whl

Then use as a drop-in replacement:

    from layer2_field.rust_dynamics import RustBackedFastEventSimulator
    sim = RustBackedFastEventSimulator(state, rules, mode='gillespie', seed=42)
    sim.run_until(t_end=1.0)

Correctness is verified by `tests/test_rust_equivalence.py` — runs the
Rust-backed and pure-Python simulators side by side, asserts event-by-event
match over the full trajectory.
"""

from __future__ import annotations

import numpy as np

try:
    import cell_sim_rust
except ImportError as e:
    raise ImportError(
        'cell_sim_rust extension not installed. Build it with:\n'
        '    cd cell_sim_rust && maturin build --release\n'
        '    pip install --force-reinstall target/wheels/cell_sim_rust-*.whl'
    ) from e

from layer2_field.fast_dynamics import FastEventSimulator


class RustBackedFastEventSimulator(FastEventSimulator):
    """
    FastEventSimulator with the compiled-MM propensity computation in Rust.

    Inherits all setup, apply, and event-logging logic from the Python
    FastEventSimulator. Overrides only `_step`, which replaces the
    vectorised-numpy propensity block with a single `cell_sim_rust`
    call that produces the same float64 values.

    Invariants (necessary for bit-identity):
      1. RNG draws happen in Python in the order `exponential → random`.
      2. Cumsum selection runs in Python on `props.tolist()` — matches
         FastEventSimulator's FP semantics.
      3. Python-closure rule propensities are computed the same way as
         in FastEventSimulator; Rust receives them as input and never
         overwrites non-compiled slots.
      4. The saturation array returned by Rust is assigned to
         `self._last_saturation` so that `_apply` (inherited) can
         reconstruct cands without redoing the math.
    """

    def _step(self) -> bool:
        # ---- refresh enzyme counts (same as Python Fast) ----
        if self._enzyme_counts_dirty:
            pbs = self.state.proteins_by_state
            for k in range(self._n_compiled):
                n = 0
                for key in self._enzyme_state_keys[k]:
                    n += len(pbs.get(key, ()))
                self.C_enzyme_counts[k] = n
            self._enzyme_counts_dirty = False

        # ---- sync dict counts into array (same as Python Fast) ----
        if self._counts_dirty:
            mc = getattr(self.state, 'metabolite_counts', None)
            if mc is not None:
                for sid, c in mc.items():
                    idx = self._species_to_idx.get(sid)
                    if idx is not None:
                        self._counts[idx] = c
            self._counts_dirty = False

        # ---- refresh python-closure rule cache (same as Python Fast) ----
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

        # ---- Rust hot path: compute MM propensities, combine with py cache ----
        total, props, saturation = cell_sim_rust.compute_propensities(
            self._counts,
            self._is_infinite,
            self._inf_value,
            self._vol_L,
            self.C_sub_idx, self.C_sub_stoich, self.C_sub_mask,
            self.C_km_idx, self.C_km_val, self.C_km_mask,
            self.C_kcat, self.C_include_sat,
            self.C_enzyme_counts, self.C_rule_idx,
            self._py_props_cache,
        )

        if total <= 0.0:
            return False

        # Keep the returned propensity vector and saturation visible in
        # `self._props` / `self._last_saturation` so `_apply` (inherited)
        # and external observers see the same surface as FastEventSimulator.
        self._props = props
        self._last_saturation = saturation

        # Rebuild cands_scratch for python-closure rules (needed by _apply
        # when a python-rule fires). Compiled rules ignore cands_scratch.
        for i in self._python_rule_indices:
            self._cands_scratch[i] = self._py_cands_cache[i]

        # ---- Gillespie draws in Python (preserves RNG bit-identity) ----
        # Recompute total with Python sum over Python list, to match
        # FastEventSimulator's FP semantics exactly. Rust's sum is
        # bit-identical in practice, but using Python sum here makes
        # equivalence robust to future Rust refactors.
        total = float(sum(props.tolist()))
        if total <= 0.0:
            return False

        dt = self.rng.exponential(1.0 / total)
        self.state.time += dt
        r = self.rng.random() * total

        # Cumsum selection — same float-list iteration as FastEventSimulator.
        cum = 0.0
        chosen = -1
        props_list = props.tolist()
        for i in range(self._n_rules):
            cum += props_list[i]
            if r < cum:
                chosen = i
                break
        if chosen < 0:
            chosen = self._n_rules - 1  # numerical safety

        # Apply (inherited from FastEventSimulator, handles both compiled
        # and python-closure rule kinds, and updates self._counts in place
        # for compiled fires).
        self._apply(chosen)
        return True
