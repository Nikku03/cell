# Regulatory-FBA hybrid: regulators as valves, FBA as the oracle

Your valve idea, built correctly this time. v1 failed two ways; both fixed:
- **TF mapping**: TFs aren't in the metabolic model (they're not enzymes). Fixed
  with RegulonDB `GeneProductSet.txt` → genome-wide symbol→b-number (4,780 genes).
  Now **170+ TFs map** and **1,300+ regulated targets** wire in (vs 4 / 11 in v1).
- **Oracle**: pure-graph reachability can't bootstrap cofactors. Replaced with
  **FBA** as the "did the cell still grow" oracle (the irreducible piece).

**Mechanism:** knock out gene g → regulatory closure (any target whose *only*
activator/sigma is now off goes off, iterated) → knock those genes out of the
metabolic model → FBA. g is essential if growth collapses, directly or via its
regulon.

## Result — the first essentiality signal on regulators

| method | called | precision | recall |
|---|---|---|---|
| plain single-gene FBA | 97 | **0.794** | 0.342 |
| regulatory-FBA hybrid | 111 | 0.739 | **0.364** |

- **14 new calls, all regulators** (off-model TFs plain FBA structurally cannot
  reach), **5 truth-essential** → recall +2.2pp, precision −5.5pp.
- Examples caught: `fnr` (regulon of 32 model genes), `phoP` (23), `phoB` (19),
  `oxyR` (15), `arcA` (3). Each becomes essential because its regulon contains
  essential metabolic genes with no alternate activator.

## Two honest caveats that shape the interpretation

**1. The truth labels for regulators are themselves suspect.** Sanity check
confirms the b-number mapping is correct (rpoD/rho/dnaG/eno → essential ✓;
lacZ/thrA → non-essential ✓). But `arcA, fnr, oxyR, phoB, phoP` are labeled
essential here despite being textbook-dispensable in rich aerobic media. The
source field reads **"via essential_families"** — derived from family-level
essentiality, not direct single-knockout data. So the 5 "true positives" are the
model agreeing with a possibly over-called label: **model bias and truth bias may
be aligned**, inflating the apparent win. Regulator validation is shakier than
metabolic validation here.

**2. The regulon rule over-predicts (precision 0.36 on new calls).** Treating a
RegulonDB activation edge as strictly required ("sole activator off → target
off") ignores basal expression and condition-dependence. 9 of 14 new regulator
calls are truth-non-essential. A real TF knockout rarely silences its whole
regulon.

## Honest verdict

The valve framework **does** what nothing else we built could: it extends
essentiality to **regulators**, a gene class plain FBA and conservation are
structurally blind to. That's a genuine proof-of-concept — knocking a regulator
valve can collapse a downstream essential pipe, and the FBA oracle confirms it.

But it is **modest and noisy**: +2.2pp recall, −5.5pp precision, new calls at
0.36 precision, and validated against partly-derived regulator labels. It is not
yet a clean coverage win. To make it one needs (a) a more conservative regulon
rule (basal expression, condition), and (b) gold-standard single-knockout truth
for regulators rather than family-derived labels.

The structural conclusion stands and is now demonstrated, not just argued: the
non-metabolic genome (here, regulators) is reachable only by modeling the extra
layers (regulation), and even then needs better data to call cleanly — the same
signpost toward condition-specific fitness (Wheel 4).
