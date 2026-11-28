# interaction_profiler

**What is my optimisation problem actually made of?**

One file, `numpy` only. Drop `interaction_profiler.py` into any project.

```python
from interaction_profiler import profile_objective

report = profile_objective(my_cost_fn, variables=range(20), state_counts=6)
print(report.summary())
```

`my_cost_fn` is called with a complete assignment `{variable: state}` and returns
a float. Nothing else is assumed — no gradients, no structure, no source.

---

## What it tells you

```
variables            : 12
irreducible orders   : {2: 4, 3: 1}
max order            : 3
interaction edges    : 4
treewidth (upper)    : 2
exact inference cost : 2**7.0 table entries
full state space     : 2**27.9 configurations
noise floor          : deterministic   (tau = 1e-09)
objective calls      : 3,507 (exhaustive would be 87,630, 96.0% avoided)

STRATEGY: EXACT -- variable elimination (carrying order-3 factors)
  irreducible 3-body structure is present, so a pairwise model is not
  faithful. Treewidth 2 still makes exact inference affordable.
```

Choosing the wrong solver costs far more than choosing a slightly worse one for
the right problem. This tells you which of these you are in **before** you commit:

| verdict | what to do |
|---|---|
| `SEPARABLE` | optimise each variable independently, linear time |
| `EXACT` | variable elimination — take the guarantee |
| `APPROXIMATE` | belief propagation or local search, no error bound |
| `HARD` | high-order and wide; compare against a heuristic first |
| `INCONCLUSIVE` | your objective is too noisy to profile — **not** separable |

---

## Verify it before trusting it

```bash
python interaction_profiler.py --selftest
```

Runs 13 checks against objectives whose structure is known in advance —
separable, pairwise, 3-body, sparse at n=30, and four noise levels.

---

## API

```python
report = profile_objective(
    objective,                 # (config: dict) -> float
    variables,                 # ids
    state_counts,              # {var: n_states}, or one int for all
    tau="auto",                # threshold; "auto" measures the noise floor
    max_order=3,               # highest arity probed
    n_references=3,            # reference configs per group
    adaptive=True,             # escalate only where lower orders interact
)

report.strategy              # the recommendation
report.rationale             # why, in a sentence
report.strengths             # {group: irreducible interaction strength}
report.interacting_pairs()   # [(i, j), ...]
report.high_order_groups()   # [(i, j, k), ...]
report.treewidth_upper       # constructive upper bound
report.interaction_graph     # {var: set(neighbours)}
report.to_networkx()         # optional, needs networkx
report.separable             # bool
report.inconclusive          # bool -- check this before believing `separable`
report.probe_saving          # fraction of exhaustive probing avoided
```

---

## How it works

For a group `S` and a reference configuration, the inclusion–exclusion residual
is the iterated finite difference of the objective over `S`:

```
Δ_ij  = E_ij − E_i − E_j + E_0
Δ_ijk = E_ijk − E_ij − E_ik − E_jk + E_i + E_j + E_k − E_0
```

It is **exactly zero** whenever the objective decomposes into terms of order
below `|S|`, so its magnitude *is* the irreducible interaction at that order.

Two things make it affordable:

- **Adaptive escalation.** Probe all pairs, then only those triples whose
  variables already interact pairwise. An interaction cannot appear from
  nothing, so this is exact rather than a heuristic prune. Measured saving:

  | n | calls | exhaustive | avoided |
  |---|---|---|---|
  | 10 | 1,389 | 25,320 | 94.5% |
  | 40 | 21,504 | 1,934,880 | 98.9% |
  | 80 | 86,124 | 15,927,360 | 99.5% (3.5 s) |

- **Multiple references.** A single reference can hide an interaction by
  coincidence; strength is the max over several.

---

## Two things that matter more than they sound

**Noise.** Real objectives are often Monte-Carlo. Against a *fixed* threshold,
per-call noise of sd 0.1 reports **all 45 pairs** of a 10-variable problem as
interacting — the profile is worthless. `tau="auto"` measures the per-call
spread first and sets the threshold above it, accounting for how noise
accumulates through an order-`k` finite difference and for taking a maximum over
`d^k` cells. Detection stays correct from deterministic up to sd 0.1.

**`SEPARABLE` vs `INCONCLUSIVE`.** Finding nothing has two opposite causes — "no
interaction exists" and "the noise is louder than any interaction would be" —
and they lead to opposite decisions. The report distinguishes them rather than
defaulting to the flattering one. **Always check `report.inconclusive`.**

---

## Limits, honestly

- Discrete variables only. A continuous objective needs discretising first, and
  the discretisation can create or destroy apparent interactions.
- Cost is `d^k` per probed group, so `max_order=4` is much more expensive than 3
  and rarely changes the recommendation.
- The adaptive prune assumes an order-`k` interaction implies its `(k−1)`
  sub-interactions are nonzero. True for the reference-point residual; set
  `adaptive=False` if you want the exhaustive probe.
- Treewidth is an upper bound from elimination heuristics, not exact (that's
  NP-hard). It is constructive — the returned order achieves it.
- A recommendation is a starting point. Benchmark against a good local-search
  baseline before concluding inference is the right tool.

---

## Prior work

Detecting *pairwise* variable interaction in black-box optimisation is mature —
**differential grouping** (Omidvar et al.) and its descendants (DG2, recursive
DG, overlapping-enhanced DG) do exactly that via finite differences for
large-scale global optimisation. That's the right citation for the underlying
idea.

What's different here: the output is **graded and of arbitrary order** rather
than a binary interaction graph, it works on discrete state spaces with an exact
Möbius decomposition rather than continuous finite differences, and it carries a
noise model.
