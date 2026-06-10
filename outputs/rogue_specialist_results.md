# Two-tier rogue-essential experiment — leak-free, decisive

**Question:** does a dedicated specialist (family_frac removed, organism-intrinsic
features only) recover more *rogue essentials* (essential genes whose family is
NOT usually essential, family_frac<0.4) than the simple fused model?

**Testbed:** 32,397 labeled genes, 10 genome-joined organisms, leave-one-
organism-out. family_frac recomputed per-fold excluding the held-out organism
(leak-free). All intrinsic features computed from each gene's own genome,
no labels. No neighbour-label features. Audited at runtime.

## The rogue zone (24,934 genes, 3,276 rogue essentials, 13%)

### Ranking quality (threshold-free)
| metric | conservation | fused | TIER-2 specialist |
|---|---|---|---|
| ROC-AUC | 0.528 | 0.595 | **0.643** |
| PR-AUC | — | **0.235** | 0.225 |

### Recall at MATCHED precision (the decision-relevant, fair comparison)
| operating point | conservation | fused | TIER-2 |
|---|---|---|---|
| precision ≥ 0.30 | 0.228 | **0.268** | 0.149 |
| precision ≥ 0.40 | 0.055 | **0.128** | 0.040 |
| precision ≥ 0.50 | 0.001 | **0.049** | 0.013 |

## Verdict

**The two-tier specialist is REFUTED.** Its high ROC-AUC (0.64) is a mirage for
a rare-positive task: at every *usable* operating point (precision ≥ 0.30),
the specialist loses badly. Removing family_frac discards real signal —
family_frac in the "rogue zone" (0→0.4) is WEAKER but not USELESS; a gene at
0.35 is still more likely essential than one at 0.02, and the fused model keeps
that residual gradient.

**But the constructive core is VALIDATED.** Adding organism-intrinsic features
to the prior (fused) beats conservation-alone at every usable precision —
at precision ≥ 0.40, fused recovers **2.3× more** rogue essentials than
conservation (0.128 vs 0.055 recall). So the intrinsic features DO carry
rogue-recovery signal; the right way to use them is the simple fused model,
NOT a separate specialist.

## What carried the signal
TIER-2 feature importance: **func_redundancy (isozyme) was the strongest
intrinsic feature**, consistent with the earlier d=0.24 finding. The k-mer
**structural_uniqueness proxy was weak** (not top-8) — the cheap sandbox
proxy for ESM did not validate; the real ESM (fold-level, not k-mer) deserves
one test but with modest expectations.

## The honest ceiling
Even fused, only ~13% of rogue essentials are recovered at 40% precision.
The intrinsic features catch NETWORK-position rogues (sole-provider, complex
member) — a minority. The majority of rogues are ENVIRONMENT-driven (essential
only in a niche), which is not in the genome. That still needs the
Fitness Browser condition-resolved data; no model architecture fixes it.
