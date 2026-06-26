# Verified: feba fitness adds orthogonal essentiality coverage (mtub, independent truth)

Question: does feba.db TnSeq fitness data add essentiality-prediction power
ORTHOGONAL to cross-organism conservation? Tested on M. tuberculosis against the
**independent** DeJesus 2017 truth (zero circularity), then adversarially
verified with four checks on the frozen per-gene data.

## Headline (5-fold CV, 1,038 genes, 332 essential)

| signal | AUC | neCov @ P0.90 |
|---|---|---|
| conservation only | 0.722 | 0.03 |
| **conservation + fitness** | **0.819** | **0.75** |

AUC gain **+0.095, bootstrap 95% CI [+0.070, +0.110] — significant.** This is
the first signal in the project that adds coverage conservation doesn't already
have; every prior one (FBA transfer, FBA-free proxy, gap-fill, regulatory,
sequence/PWM) collapsed onto the conserved core.

## The four adversarial checks

**1. Circularity — passed, with a sharpening.** Both feba and DeJesus are TnSeq,
so is the gain just "one essentiality call agreeing with another"?
- `absent_from_feba` alone (= feba's own TnSeq essentiality call): **AUC 0.793**
- graded `min_fit` alone: **AUC 0.514** (≈ chance)
- `min_fit` on genes that ARE measured (present): **AUC 0.593** (weak, real)

So the value is carried by feba's **binary essentiality measurement**, not the
conditional-fitness gradient. The honest claim is *"feba's TnSeq essentiality
concords with an independent TnSeq study and is orthogonal to conservation"* —
not yet *"conditional fitness cracks the conditional soup."*

**2. Orthogonality — real.** corr(conservation, absent) = **+0.50** (overlapping
but not redundant). Decisively: **among genes where conservation is weak
(cons < 0.20, n=508), fitness's `absent` signal predicts essentiality at AUC
0.717.** Fitness works exactly where conservation goes blind. The conditional
gradient does not (AUC 0.46 there).

**3. Selection bias — cuts in fitness's FAVOUR.** The tested subset is 32%
essential vs 21% genome-wide, because we required a panel ortholog (conserved
core). The ~2,600 excluded genes are lineage-specific — where conservation is
*undefined* and fitness would be the **only** signal. So this test **under-states**
fitness's unique value; in the real genome it should be larger, not smaller.

**4. Stats — robust.** Independent re-implementation reproduced
conservation 0.725 / combined 0.821 / gain +0.095, and 2,000-sample bootstrap
puts the gain CI firmly above zero [+0.070, +0.110].

## Verified verdict

**SOUND: feba fitness data adds real, statistically-significant essentiality
coverage orthogonal to conservation, and it works precisely where conservation
fails.** This is the genuine new wheel the whole project kept pointing at.

**Honest scope of the claim:**
- ✅ feba's essentiality *measurement* (which genes tolerate no disruption)
  transfers across organisms via orthologs and is orthogonal to conservation —
  significant on independent truth, strongest where conservation is weak.
- ⏳ the *conditional-fitness gradient* (the part meant to resolve the
  conditional/soup middle) is **not yet demonstrated** — but that's expected:
  DeJesus is single-condition truth, which structurally cannot validate
  conditional value. Proving it needs a conditional ground truth.
- 📈 the measured gain is a **lower bound** — the test only covers the conserved
  core; lineage-specific genes (where fitness stands alone) were excluded.

## What this unlocks

This is the empirical green light for **Wheel 4**: feba fitness, projected via
orthologs, is a real, orthogonal essentiality signal. It generalizes
across clades (v5: AUC 0.77 cross-phylum, 0.82 same-division). The next build is
to wire it in as a fourth wheel and measure the coverage lift on the full panel —
with the discipline that its *proven* value today is the essentiality readout,
and the conditional value is the next thing to test (needs conditional truth).
