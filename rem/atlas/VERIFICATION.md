# Atlas build-order items 2, 3, 4 — build and adversarial verification

Each module was built against predeclared gates, then handed to an independent adversarial
verifier whose only job was to break it. All three verifiers returned SOUND_WITH_CAVEATS with
`reproduces: true`, and all three found real defects. The defects are the point of this file.

---

## Item 3 — Floquet phase-resolved solver (`floquet.py`) — 15 PASS / 1 FAIL / 1 VOID of 17

The strongest of the three. It reproduces the spec's §3.2 table to 5 significant figures, and
it does so with four independent checks the spec does not require: an analytic route with no
matrices (Fano and tail agree to 1.07e-05 and 2.30e-05), phase-resolved distributions against
Poisson(m(t)) to 1.83e-06, and a period-closure check to 4.00e-11.

**The forbidden route was run.** Eigenvector extraction returned garbage at 8 of 17 sweep
points, confirming §3.1's warning empirically rather than by assertion.

### FINDING: the spec's cost number contradicts the spec's mandated pattern

`COST` fails at **8.8 ms against the stated 5.0 ms**, and the diagnosis is arithmetic, not an
implementation defect. §3.1 mandates one `expm` per phase. A single `expm(91×91)` measures
**0.677 ms**, so the 12 mandated calls cost **8.24 ms — already 1.6× the entire budget** before
one matrix product, solve or step. The remaining pipeline is 1.14 ms.

The 5.0 ms is reachable only by **breaking the mandated pattern**: caching `expm` over the 2
*distinct* generators among the 12 phases runs in **1.86 ms**. So either the spec measured a
caching implementation, or its pattern and its cost line describe different code.

### FINDING: the §3.2 table is a 160-sample-per-cycle quadrature

The table reproduces only at `n_sub = 80`. Deviation over the 10 Fano/tail cells: 3.90e-04 at
75, **4.72e-05 at 80**, 2.86e-04 at 85, 9.91e-04 at 100 — a sharp unique minimum, so this is a
forensic determination rather than a fit. At the module's own default of 1000 the last two rows
miss the spec. Worth stating in §3, since T06–T08 are regression tests.

### Verifier caught

- **A phantom 18th gate.** "SPEC 3.2 TABLE reproduction — PASS" was reported but there are
  exactly 17 `_row()` calls and none is for the spec table. It is prose with no coded threshold.
- A `1e-5` bar at `floquet.py:636` appearing nowhere in the predeclaration.
- **Two false "WHAT IT CATCHES" claims.** F-swap's claimed target (recording p after the step
  instead of before) is a *mathematical no-op* on a uniform grid — measured 1.39e-17, no gate
  can detect it. F-control cannot catch truncation error, because its reference is built at the
  same N so truncation cancels exactly.
- A hardcoded string `"<- FASTER: 91x91 is too small to thread"` that was **false in both runs**
  (8.92 vs 7.74 ms; 9.54 vs 7.96 ms).

---

## Item 2 — Aggregate debias (`debias.py`) — was 7/7 PASS, now **5 PASS / 2 FAIL**

### The defect: a different deciding statistic per gate, each one the one that passed

- `G4.1` was decided on the **single declared seed** → 0.3890% → PASS. Across-seed worst of
  1000: **1.2482% → FAIL**.
- `D-scaling` was decided on the **across-seed median** → 0.913 → PASS. Declared seed: **0.655
  → FAIL**.

Swap the statistics and both fail. Neither choice is defensible over the other, so *the choice
itself* was the defect. Standing rule 3 settles it — worst case, not median — and it is now
applied to every stochastic gate in the module. **Both gates now FAIL**, which is the honest
result and is recorded rather than re-tuned.

The verifier explicitly cleared the module of seed shopping: the declared seed sits at the 70th
percentile of |debiased error| and is an *unlucky* draw for three other statistics.

### FINDING: my headline attack on the spec's exponent was inverted

The module claimed the spec's `m^1.06` was "roughly 0.1 high", citing a measured 0.913. That
estimator is **downward-biased**: at m = 1000 the noise is `0.5665·√1000 = 17.91` against a
signal of 79.35 — **22.6%**, not negligible. Across 400 seeds the large-m ratio has median 16.07
[p10 12.60, p90 21.99] against a structural asymptote of exactly 16.00. The honest across-seed
exponent is **a = 1.0016 — above the spec's own table refit of 0.950, not below it.** The attack
on 1.06 is **withdrawn**. The qualitative conclusion is unaffected and firm: scaling is LINEAR,
not sqrt, so item 7's conjunctive cap is justified.

### FINDING: G4.1's pass is ensemble-conditional and was not flagged

The gene-mean ensemble spans only 8.7×. Under a lognormal gene-mean distribution with
`sd_ln = 2.0` — still narrow for a real mRNA atlas — the median debiased error is **1.358% and
the gate fails 63% of the time**; at 2.5 it is 2.690% and fails 79%. "< 1%" is a property of
Rule A *on a near-homogeneous gene set*, not of Rule A.

### Vacuous gates the verifier identified

- **`D-bias` cannot fail.** Its Monte-Carlo sd is 0.0070 pp against a 0.20 pp bar — a 29σ-wide
  bar — and the quantity converges to `exp(σ²/2)` as an identity of numpy's lognormal
  generator. Neither Rule A, nor the gene ensemble, nor the solver enters it.
- **`T12` accepts almost anything.** Its criterion passes for every σ from 0.25 to 1.5 (median
  error 0.72 to 7.67 orders against the spec's 2.0).
- `D-control-b`'s headline "separated by 1.0e+299×" is an artifact of a `1e-300` floor constant.
  The control discriminates in the right direction; that number means nothing.
- `D-control-a` **is** real — the verifier broke it seven ways and it fired on the hard-coded-σ
  bug (−7.688%) while G4.1 still passed at 0.389%.

---

## Item 4 — Uncertainty envelope (`envelope.py`) — 6 FAIL, 1 VOID

### FINDING: §6.4's magnitudes are impossible, not merely unreproduced

This is a proof, not a measurement disagreement. As a single rate `k → 0`, the stationary weight
of protein state n scales as `k^±n`, because n unit-cost events separate state n from state 0
along the cheapest path. Hence

```
|d log10 P(n >= T) / d log10 k|  <=  T
```

so a ×1.2 perturbation can move the tail by **at most `T · log10(1.2) = 18 · 0.079181 = 1.4253`
orders**. §6.4 gives k_translation 3.49, k_mRNA_decay 2.54, k_transcription 2.18,
k_protein_decay 1.79 — **2.45×, 1.78×, 1.53× and 1.26× above that ceiling.** No circuit at
threshold 18 can produce them. Confirmed empirically: driving k_translation down 25× raises the
slope monotonically to 1.2684 and it never crosses the bound.

**What survives, and it is the actionable half:** the *ranking* reproduces at Spearman 0.943,
and the claim that burst-SIZE rates dominate the burst-FREQUENCY rate holds. §6.4's advice is
right; its numbers are not attainable at the stated threshold.

### FINDING: my own diagnosis of three failures was wrong

The module blamed burstiness and concluded §6.1's 2.106 "is not reproducible from section 6
alone". The verifier showed that sweep **confounds burstiness with tail depth** (P(n≥18) moves
4.7× along its own rows) and tested the rival explanation the module never tried — changing only
`k_translation` from 0.583 to 0.40, the one free scale the docstring itself names:

```
                    before (k=0.583)      after (k=0.40)     spec
E1b tail bias          +0.0363 FAIL       +0.0120 PASS      -0.1140
E1d tail IQR            1.5856 FAIL        2.1749 PASS        2.106   (3% off)
E1e IQR ratio           2.7473 FAIL        3.7681 PASS        4.93
E1c mean IQR            0.5772 FAIL        0.5772 FAIL        0.427   (circuit-independent)
```

**§6.1 IS reproducible** at a slightly different circuit; the "not reproducible" claim is
withdrawn. `E1c` stays failed and is genuinely circuit-independent — that one is a real
discrepancy.

### What passed

`E-control` is the strongest control in the three modules: the verifier broke it three ways
(jittered caps, a σ-independent noise floor, a different solve path) and it fired every time,
including reproducing the distinctive "(b) bias moves, IQR does not" signature.

---

## Standing consequences

1. **One deciding statistic, declared before the run, applied to every stochastic gate.** The
   debias module is the counterexample: with per-gate choice, 7/7 passed; with one rule, 5/7.
2. **A gate whose Monte-Carlo sd is 29× narrower than its bar is not a gate.**
3. **Check a claimed sensitivity against its structural ceiling before gating on it.** §6.4's
   magnitudes could have been rejected on paper, without a single solve.
4. **A control's "what it catches" list must be tested by breaking each named thing.** Two of
   floquet's three claims were false, and the module still worked.
