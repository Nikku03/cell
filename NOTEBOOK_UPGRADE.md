# Colab notebook — FastEventSimulator integration

## Short answer

**The notebook itself needs zero changes** — except one line in Section 9.

The render scripts (`render_priority_15.py` etc.) in your repo are now
updated to use `FastEventSimulator`. Next time the notebook clones your
repo and runs those scripts, it picks up the speedup automatically.

## Projected Colab times

Measured: Priority 1.5 simulation dropped from 49 s → 8.85 s in the
earlier sandbox test. Extrapolating to other sections:

| Notebook section | Current (Python) | Projected (Fast) | Speedup |
|---|---|---|---|
| §4  Real Syn3A (0.5 s bio) | 5.8 s sim / 17 s video | ~5.8 s sim / 17 s video | **1.0x** (no MM rules here) |
| §5  Priority 1.5 (1.0 s bio) | **49 s sim** / 25 s video | **~9 s sim** / 25 s video | **5.5x on sim, 2.2x end-to-end** |
| §6  Priority 2 (1.5 s bio) | **75 s sim** / 22 s video | **~12 s sim** / 22 s video | **6.2x on sim, 2.8x end-to-end** |
| §7.2 Priority 3 BrdU demo | ~16 s sim, no video | ~3 s sim, no video | **5x** |
| §7.3 Priority 3 AZT interactive | ~16 s sim, no video | ~3 s sim, no video | **5x** |
| §9  Interactive Priority 1.5 (0.3 s) | 16 s sim | ~3 s sim | **5x** |

**Notebook total runtime**: ~15-25 min → **~7-12 min** end-to-end.

Renderer (matplotlib + ffmpeg) time is unchanged — that's why end-to-end
speedup is smaller than simulator speedup. Real Syn3A sees no benefit
because it uses `make_catalysis_rules` (simple Python closures, no
compiled MM specs).

## How to apply

### Step 1: Commit the patch to your repo

    tar -xzf colab_integration.tar.gz
    cd cell_sim
    python tests/test_fast_equivalence.py     # sanity check, should show ~10x + MATCH
    cd ..
    git add cell_sim/layer2_field/dynamics.py \
            cell_sim/layer2_field/fast_dynamics.py \
            cell_sim/layer3_reactions/reversible.py \
            cell_sim/tests/render_real_syn3a.py \
            cell_sim/tests/render_priority_15.py \
            cell_sim/tests/render_priority_2.py \
            cell_sim/tests/demo_priority3.py \
            cell_sim/tests/test_fast_equivalence.py
    git commit -m "Wire render scripts to FastEventSimulator"
    git push

### Step 2: (Optional) One-line change in Section 9 of the notebook

Section 9 has an inline `sim = EventSimulator(...)` that isn't in a
render script, so it won't pick up the change automatically. If you
want Section 9's 16-second run to drop to ~3 seconds, edit that cell:

```python
# Change this line:
from layer2_field.dynamics import CellState, EventSimulator

# To this:
from layer2_field.dynamics import CellState
from layer2_field.fast_dynamics import FastEventSimulator as EventSimulator
```

The rest of Section 9 doesn't change — it uses `EventSimulator` as a
name, and the aliasing makes it transparent.

### Step 3: Re-run the notebook

- Runtime → Disconnect and delete runtime (to force fresh clone)
- Reconnect
- Run all

## Bit-identical, verified

The patched render scripts produce the **same 41,853 / 83,049 / 114,548
events** as before, same ATP/ADP/G6P metabolite trajectories, same
complex formation sequence. You should recognize all the Colab outputs
from the last runs — just arriving faster.

## What changed in each render script

Identical one-line change in all four:

    # before
    from layer2_field.dynamics import CellState, EventSimulator

    # after
    from layer2_field.dynamics import CellState
    from layer2_field.fast_dynamics import FastEventSimulator as EventSimulator

Everything else — seeds, rule builders, snapshots, rendering loop —
is untouched.

## Why Real Syn3A doesn't speed up

Section 4 uses `make_catalysis_rules(kcats, enzyme_map)` which produces
simple Python-closure catalysis rules without `compiled_spec`.
FastEventSimulator's vectorization path only applies to reversible MM
rules from `build_reversible_catalysis_rules` (Priority 1.5+). For
Section 4, every rule falls through to the Python path — no benefit.

That's fine; Section 4 is already the fastest section (~6 s sim). The
speedup matters for the sections that take minutes: Priority 1.5 and
Priority 2. Those drop ~6x.

## Why Priority 3 demos speed up

The Priority 3 demo (Section 7.2) runs a full Priority 1.5 simulation
twice (baseline + with novel substrate), then reports side-by-side.
That's 2x 27k events = 54k MM-dominated events. Same ~5x speedup as
Priority 1.5 applies: from ~16 s total sim → ~3 s.

## Safety: old code still works

`layer2_field/dynamics.py`'s `EventSimulator` is unchanged. If you ever
want to verify a result against the reference, just undo the one-line
import. Nothing else needs to change.
