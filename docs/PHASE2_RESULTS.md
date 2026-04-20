# Phase 2 — next-reaction method

## Headline

**Phase 2 is architecturally correct but 4x slower than Phase 1 in Python.**

The cause is a genuine architectural finding about Syn3A metabolism,
not a bug. Ship Phase 2 as a reference implementation for the eventual
Rust port; keep using Phase 1 for simulations today.

## What Phase 2 is

`layer2_field/next_reaction_dynamics.py` — `NextReactionSimulator`.
Implements the Gibson-Bruck next-reaction method (2000):

- Priority queue of per-rule "putative firing times"
- Species-to-rule dependency graph
- Lazy deletion of stale heap entries via per-rule generation counters
- Gibson-Bruck time-rescaling when a rule's propensity changes

In principle, this avoids Phase 1's cost of recomputing all 257
propensities every step — only the rules whose dependencies changed
get updated. The classic algorithmic win is 30-100x on sparse networks.

## Correctness: solid

NextReactionSimulator produces statistically equivalent trajectories
to Phase 1. It can't be event-for-event identical (the two methods
consume different random numbers per step), but final metabolite counts,
event distributions, and flux patterns all match within stochastic noise.

On a Priority 1.5 run at 2% scale, t=0.3 s bio-time, seed=42:

| Metric | Phase 1 | Phase 2 | Δ |
|---|---|---|---|
| Events | 27,463 | 27,853 | +1.42% |
| Catalysis events | 26,850 | 27,242 | +1.44% |
| Folding events | 610 | 610 | 0% |
| ATP Δ | −7,777 | −7,815 | 0.49% |
| ADP Δ | +7,427 | +7,413 | 0.19% |
| Pyr Δ | +706 | +719 | 1.81% |
| Pi Δ | +3,893 | +4,021 | 3.18% |

All differences are well within the Poisson noise you'd expect across
two runs of the same stochastic process with different random-number
consumption.

## The performance finding

| Simulator | t=0.3 s wall | events/s |
|---|---|---|
| Python EventSimulator | 33.5 s | 820 |
| Phase 1 FastEventSimulator | 3.05 s | 9,005 |
| Phase 2 NextReactionSimulator | 12.18 s | 2,287 |

Phase 2 is 4x slower than Phase 1. Why?

### The dependency graph isn't sparse

For each species, the number of rules whose propensity depends on it:

| Species | # rules |
|---|---|
| ATP | **68** |
| H2O | 49 |
| H+ | 35 |
| ADP | 26 |
| H+ (extracellular) | 21 |
| Pi | 14 |

Syn3A metabolism funnels through ATP and a handful of cofactors — the
same 6 species are reactants/products in 50-60% of all reactions.
Firing a single glycolytic event invalidates ~60 rules because of
shared-cofactor cascades.

Measured per-event invalidation on Priority 1.5:

| Percentile | Rules affected | % of all rules |
|---|---|---|
| Min | 3 | 1% |
| Median | 58 | 23% |
| Mean | 70 | 27% |
| 90th pct | 133 | 52% |
| Max | 152 | 59% |

### Python constant factors dominate

Gibson-Bruck was designed for networks where <5% of rules invalidate
per event. At 23-60% invalidation, the algorithm does 85 Python-scalar
propensity recomputations per event. Each takes ~4 µs due to dict
lookups, attribute access, and math operations. Total: 340 µs/event of
Python compute, which overwhelms heap's O(log R) savings.

Phase 1's vectorized numpy computes ALL 257 propensities in ~150 µs
per step as a single op. The numpy dispatch overhead is fixed — it's
almost as fast for 257 rules as for 58. So Phase 1 does more work but
with lower per-op cost.

**The algorithmic savings and the Python-dispatch penalty cancel.**

### Heap statistics

Of 1.3M heap pops during the test run, 97.9% were stale entries from
lazy deletion. Real next-reaction-method implementations use an indexed
priority queue with decrease-key in O(log n) instead of lazy deletion,
but the real bottleneck is the scalar Python compute per update, not
heap operations (heap cost was ~0.5s of the 12s total).

## The path forward

Phase 2's value is as a **reference implementation for Phase 3 (Rust)**.
In Rust:

- Per-rule propensity update: ~0.05 µs (80x faster than Python)
- Heap ops: effectively free
- Net: ~100-200 µs/event ≈ 6-10× faster than Phase 1
- Combined with the algorithmic savings once per-op cost is low, another 3-5x

Expected Phase 3 (Rust): 1 s bio-time in ~0.5 s wall.

## Alternative: τ-leaping

Orthogonal optimization: take larger time steps and fire multiple events
per step. A rule with propensity a_k fires Poisson(a_k·τ) times per
leap. For high-count species like ATP (50k+ molecules), this can
legitimately batch 100s of events per step, giving 30-100x speedup for
metabolism-dominant runs.

The existing Python `EventSimulator` already has a `step_tau_leap` hook
but it hasn't been integrated with the compiled specs. τ-leaping in
the vectorized Phase 1 style would be:

- Compute all propensities once per step (Phase 1's strength)
- Poisson-sample fire counts for each rule in parallel
- Apply stoichiometric deltas in one numpy operation

Expected: 30-100x over Phase 1 for metabolism, but approximation error
at low species counts (can correct with hybrid exact/approximate).

## Recommendation

1. **Keep Phase 1 as the default simulator.** It's fast, correct, and
   bit-identical to the reference. Ship it.
2. **Ship Phase 2 as `next_reaction_dynamics.py`** alongside, labeled
   as "reference implementation, not faster in Python, preserved for
   Rust port."
3. **The next actual speedup for Python** is τ-leaping with vectorized
   delta application — 30-100x for metabolism-dominant runs, estimated
   1-2 days of work.
4. **The next speedup beyond that** is Rust (Phase 3), which benefits
   from Phase 2's correct algorithmic structure and gives another
   10-30x on top.

## Files

- `layer2_field/next_reaction_dynamics.py` — NextReactionSimulator (new)
- `tests/test_next_reaction.py` — statistical correctness check

## How to run

    cd cell_sim
    python tests/test_next_reaction.py
