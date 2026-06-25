# Can structural "truck-filtering" push non-essential coverage up?

Your idea: filter out the trucks the cell doesn't need (redundant / disconnected
/ orphans-nothing) → confident non-essential calls → raise non-essential
coverage toward 99-100%. Tested on iML1515 vs Keio truth, pure graph, no FBA.

## The structural flags work — but cap at ~0.86 precision

Test universe = 1,130 metabolic genes (194 essential, 936 non-essential):

| flag (non-essential call) | calls | precision | coverage(recall) |
|---|---|---|---|
| redundant (isozyme backup) | 566 | 0.866 | 0.524 |
| disconnected (dead-end rxn) | 48 | 0.896 | 0.046 |
| orphans-nothing | 592 | 0.865 | 0.547 |
| strict (redundant ∨ disconnected) | 591 | 0.870 | 0.549 |
| loose (+ orphans-nothing) | 814 | 0.865 | 0.752 |

Every structural signal lands at **~0.86-0.87 precision** — exactly the
prediction, and **nowhere near 99%.**

## Why it can't raise coverage: two walls

**Wall 1 — in its own domain, W1 already wins.** On these same metabolic genes,
the transformer (W1) alone already achieves:
- **neCov = 0.936 at precision ≥ 0.95**
- **neCov = 0.985 at precision ≥ 0.90**

The structural filter is *less* precise (0.86) than W1 (0.95) on the very genes
it can assess. Adding a 0.86-precision batch to a 0.95 operating point drags
combined precision below target, so it can't be added without losing the line
(the −93pp / −98pp collapses are exactly that: the union can't hold P≥0.95).

**Wall 2 — the genes W1 misses are not metabolic.** The reason W1's *global*
non-essential coverage is ~0.83 (not 0.94) is the **non-metabolic** genes —
regulators, transporters, membrane, hypotheticals. Those aren't in the
metabolic model at all, so the structural filter has **no opinion** on them. It
can only operate where W1 is already excellent, and is silent exactly where the
gap is.

## Honest verdict

The truck-filter is a sound idea and the flags are real (~0.86 precision), but
it **cannot push non-essential coverage to 99%**:
- where it can see (metabolic genes), W1 is already at 0.94-0.98 and more
  precise, so the filter is redundant;
- the non-essentials W1 misses are non-metabolic, and a metabolic-network
  filter is structurally blind to them.

This is the same wall, now localized precisely: **the non-essential-coverage gap
lives in the non-metabolic genome.** Closing it needs a signal that sees
transporters/regulators/membrane — condition-specific fitness (Wheel 4), not
metabolic structure.

## The decisive number

Of all 2,980 Keio non-essential genes, only **936 (31%) are metabolic** (in
iML1515). The other **2,044 (69%) are non-metabolic** and structurally
invisible. So even a perfect metabolic truck-filter could touch at most 31% of
the non-essentials — and on that 31%, W1 is already at 0.94. The 99% target is
unreachable from metabolic structure by construction.
