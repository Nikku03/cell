# Essential vs non-essential gene knockout test

## What this tests

**Thesis**: a dynamical whole-cell simulator should produce measurably
different trajectories when an essential gene is knocked out vs a
non-essential gene. If this signal exists cleanly, we have the
foundation for a full 452-gene essentiality screen — and a genuinely
novel scientific contribution (FBA cannot do mechanistic essentiality).

**Result so far (2% scale × 100 ms smoke test):**
- KO_ptsG: GLCpts events drop to **exactly 0** ✓
- KO_pgi: PGI events drop to **exactly 0** ✓
- KO_ptsG: G6P falls **below** wild-type (no glucose import)
- KO_pgi: G6P rises **above** wild-type (G6P can't isomerize away)
- KO_ftsZ, KO_0305: all metabolic pools track wild-type

The mechanistic signal is real even at tiny scale. The next test
(50% × 0.5 s) is to see if it becomes quantitatively decisive.

## The five conditions

| Condition | Gene | Role | Class (Breuer 2019) |
|---|---|---|---|
| wildtype | — | control | — |
| KO_pgi | `JCVISYN3A_0445` | glucose-6-P isomerase (core glycolysis) | essential |
| KO_ptsG | `JCVISYN3A_0779` | glucose PTS transporter | essential |
| KO_ftsZ | `JCVISYN3A_0522` | cell division | non-essential |
| KO_0305 | `JCVISYN3A_0305` | uncharacterized peptidase | non-essential |

All five are explicitly classified in the Breuer et al. 2019 eLife
paper Table 1 / Figure 2. Those two non-essential genes are the
highest-abundance non-essential genes in Syn3A (specifically called
out by Breuer).

## How to run

### Option A: on Colab (recommended)

Add this cell to the notebook after Section 11:

```python
# Section 12: Gene knockout test
import os
os.environ['KO_SCALE']    = '0.5'      # 50% scale — enough signal, fits in ~25 min
os.environ['KO_T_END']    = '0.5'      # 500 ms biological time
os.environ['KO_USE_RUST'] = '1' if rust_ok else '0'
os.environ['KO_SEED']     = '42'

!python tests/test_knockouts.py
```

Then display the result:

```python
from IPython.display import Image
Image('data/knockout_test/knockout_comparison.png')
```

And read the summary:

```python
print(open('data/knockout_test/knockout_summary.txt').read())
```

### Option B: locally

```bash
cd cell_sim
KO_SCALE=0.5 KO_T_END=0.5 python tests/test_knockouts.py
```

### Runtime

- 2% scale × 0.1 s: ~10 s total (smoke test)
- 10% scale × 0.5 s: ~2 min total
- **50% scale × 0.5 s: ~25 min total** (the real test)
- 100% scale × 1.0 s: ~2 hours total (overkill for a 5-gene proof-of-concept)

## What to look for in the output

### In `knockout_summary.txt`

Section "Event-count changes" is the primary signal. The knocked-out
reaction should show **zero events** in its KO condition and unchanged
events in all other conditions:

```
  rxn             wildtype        KO_pgi       KO_ptsG       KO_ftsZ       KO_0305
  GLCpts             2,400          2,450             0          2,410         2,380   ← ptsG KO blocks GLCpts
  PGI               50,000              0        52,000         50,500        50,200   ← pgi KO blocks PGI
  PFK                 ...              ...
```

### In `knockout_comparison.png` (four panels: ATP, G6P, pyruvate, lactate)

At 50% × 0.5 s, the essential-KO lines should visibly peel away from
the wild-type line, while non-essential-KO lines should track wild-type
closely. Specifically:

- **ATP panel**: All decay some (since wild-type itself decays). Essentials should decay FASTER.
- **G6P panel**: KO_pgi G6P should **accumulate** (can't be consumed). KO_ptsG G6P should **deplete faster** (not being replenished).
- **Pyruvate panel**: Both essential KOs should show reduced pyruvate production (glycolysis blocked).
- **Lactate panel**: Both essential KOs should show reduced lactate (no pyruvate to ferment).

### Pass criteria

- Essential KOs deviate from WT by >20% on at least one metabolite by t=500 ms
- Non-essential KOs deviate from WT by <5% on all metabolites
- Zero events for directly-knocked-out reactions

## If it passes

Commit to the full 452-gene screen:
- Automate the knockout loop for all protein-coding genes
- Run each KO for 500 ms bio time at 50% scale
- Total runtime: ~452 × 5 min = ~38 hours, or ~15 hours on RTX 6000 with Rust
- Cross-reference with Breuer 2019 essentiality classifications
- Compute Matthews Correlation Coefficient (Breuer's FBA got 0.59)
- Analyze mechanism for each KO: which metabolite crashes first? time-to-failure?

This is the path to a Cell Systems / Mol Sys Bio paper if the
mechanistic classification works.

## If it doesn't pass

Two possibilities:

**1. Signal is there but too weak.** Fix: run longer (1-2 s bio),
higher scale (100%), and the accumulated divergence should grow.

**2. Signal is buried in decay noise.** Fix: first stabilize
wild-type by filling missing k_cats (the 196 unmeasured reactions)
and raising the `MAX_N_EFFECTIVE` clamp. Then revisit.

Either way the test tells us what to do next.
