# Phase 1 — vectorized Gillespie simulator

## Headline

**10.6x speedup on Priority 1.5, 9.6x on Priority 2, bit-identical results.**

## What changed

New `layer2_field/fast_dynamics.py` containing `FastEventSimulator`. Drop-in
replacement for `EventSimulator`:

    from layer2_field.fast_dynamics import FastEventSimulator
    sim = FastEventSimulator(state, rules, mode='gillespie', seed=42)
    sim.run_until(t_end=1.0)

The reference Python `EventSimulator` is unchanged and remains the ground
truth. Switching is opt-in.

## Architecture

Three changes stacked:

1. **Compiled rule specs.** Each Michaelis-Menten rule, at build time,
   attaches a `compiled_spec` dict to itself — explicit substrate/product
   index lists, Km values, enzyme-locus keys, kcat. The fast simulator
   uses this data; it never calls the rule's `can_fire` closure.

2. **Padded 2D numpy arrays.** All compiled rules' substrates/products/
   Km's are pre-flattened into (n_compiled_rules, max_per_rule) arrays.
   One step's propensity computation is ~10 numpy ops total, regardless
   of rule count. This replaces ~250 Python closure calls per step.

3. **Python-closure cache.** Folding and complex-formation rules have
   no `compiled_spec`; they still run via closure. But their propensities
   only depend on protein states, which compiled-MM events never touch.
   The cache stays valid through 99%+ of Gillespie steps. Only invalidates
   after folding/assembly events (rare).

## Correctness

`FastEventSimulator` produces **identical event sequences** to
`EventSimulator` given the same seed and rules — verified event-by-event
across all 27,463 events of a 300 ms Priority 1.5 run, and all 18,815
events of a 200 ms Priority 2 run. See `tests/test_fast_equivalence.py`.

To achieve this, we match the reference implementation's exact
floating-point semantics:

- `AVOGADRO = 6.022e23` (same value as `coupled.py`; not the more precise
  6.02214076e23)
- `count_to_mM` uses `(count / AVOGADRO) / vol_L * 1000.0` (two divides,
  not one combined)
- Final propensity summation uses Python's sequential `sum()`, not numpy's
  tree-reduce
- Cumulative-sum-based rule selection iterates the propensity list in
  Python

Without these, simulators drift by 1 event per ~10,000 because occasionally
two rules have nearly tied propensities and a 1-ULP difference in total
flips which one wins the categorical draw.

## Performance

Priority 1.5, scale_factor=0.02, Syn3A, single Colab CPU core:

| t_end (sim s) | events | Python wall | Fast wall | Speedup |
|---|---|---|---|---|
| 0.05 | 4,699  |  5.6 s | 0.70 s | **8.03x** |
| 0.10 | 9,329  | 10.7 s | 1.24 s | **8.71x** |
| 0.20 | 18,678 | 22.3 s | 2.27 s | **9.80x** |
| 0.30 | 27,463 | 33.5 s | 3.13 s | **10.62x** |

Priority 2 (with transcription/translation/degradation on top):

| t_end | events | Python wall | Fast wall | Speedup |
|---|---|---|---|---|
| 0.10 | 9,423  | 12.1 s | 1.42 s | **8.47x** |
| 0.20 | 18,815 | 25.1 s | 2.63 s | **9.57x** |

Speedup grows with simulation length because initialization overhead
amortizes.

## Where the remaining time goes

After Phase 1, profiling the 300 ms Priority 1.5 run (3.13 s total):

- `_step` inner ops (numpy calls, sorting, etc.): ~1.5 s
- Python apply (reversible.py `apply`): ~0.4 s
- Enzyme-count caching lookups: ~0.3 s
- `can_fire` calls on invalidation (~600 folding events): ~0.4 s
- Everything else: ~0.5 s

The next big win (Phase 2) is the **next-reaction method / Gibson-Bruck**:
don't recompute propensities for rules whose dependencies didn't change.
99% of rules don't need updates after any given event, so this is
algorithmically 50-100x improvement on top of Phase 1.

## What's not in Phase 1 (future work)

- **Phase 2:** Next-reaction method with dependency graph. Expected 30-50x on
  top of Phase 1. Estimated 4-6 hours work.
- **Phase 3:** Rust core via pyo3. Expected 5-10x on top of phases 1+2.
  Estimated 1-2 weeks.
- **τ-leaping** for high-count reactions. Orthogonal; another 10-30x for
  metabolism-dominant runs.
- **Compiled-spec for non-MM rules**: complex formation, novel substrates,
  gene expression. Currently the python-closure path. Adding compiled
  specs for them would remove the cache-invalidation cost.

## Apply

From the top of your cell/ checkout:

    tar -xzf phase1_patch.tar.gz
    # Verify:
    cd cell_sim && python tests/test_fast_equivalence.py
    # Expected output: MATCH, speedup ~8-10x

    # Commit and push:
    cd ..
    git add cell_sim/layer2_field/dynamics.py \
            cell_sim/layer2_field/fast_dynamics.py \
            cell_sim/layer3_reactions/reversible.py \
            cell_sim/tests/test_fast_equivalence.py
    git commit -m "Phase 1: FastEventSimulator, 10x speedup, bit-identical"
    git push

## Extrapolation to full Syn3A on RTX 6000

Current progression:

    Python baseline, Colab CPU:    1 s bio-time  ≈ 50 s wall (Priority 1.5)
    Phase 1, Colab CPU:            1 s bio-time  ≈  5 s wall
    Phase 1, RTX 6000 (est 2x):    1 s bio-time  ≈  2.5 s wall

Target: 2 hours bio-time (cell cycle) in <1 hour wall on RTX 6000.
Current Phase 1 projects to 2.5 s/bio-s × 7200 s = ~5 hours wall.

Need another ~5-10x from Phase 2+3 to hit the goal. The math works.
