# TB relapse in the macaque model — Maiello, Fortune, Flynn, Lin

**Their paper.** According to PubMed, Maiello P, *et al.*, "Characterizing PET CT patterns and
bacterial dissemination features of tuberculosis relapse in the macaque model", *Infection and
Immunity* 93:e0017725 (2025), PMID 40548727, [DOI](https://doi.org/10.1128/iai.00177-25).

**Their gap, in their own words.**

> "complete sterilization or very low Mtb burden is protective against SIV-induced TB relapse but
> cannot be predicted by PET CT"

> "not every site of persistent Mtb growth after drug treatment is capable of dissemination and
> relapse"

They have built the one model in which the question is answerable — barcoded *Mtb*, scan-matched
lesions, per-lesion burdens, dissemination traced between named sites — and they say plainly that
the imaging cannot deliver the prediction. What follows is what a rare-event engine can compute
from that setup, including the part where the obvious idea turns out to be wrong.

---

## 1. The obvious calculation, and why it fails

The natural question is: *given residual burdens spread across many lesions, what is the
probability that at least one fails to sterilise?* And the natural intuition is that the answer
depends on the **spread** of those burdens, not just their mean — one lesion with a thousand
bacilli being more dangerous than a thousand lesions with one.

**That intuition is false, and REM says so before saying anything else.**

If lesion *i* holds *Nᵢ* residual bacilli and each bacillus independently escapes and regrows with
probability *q*, then

```
P(at least one lesion fails)  =  1 − ∏ᵢ (1 − pᵢ)   where pᵢ = 1 − e^(−q·Nᵢ)
                              =  1 − exp(−q · Σᵢ Nᵢ)
```

The right-hand side contains only the **total**. Holding the mean burden exactly fixed and sweeping
the coefficient of variation of the per-lesion burdens from 0 to 3, the answer moved by
**2.1 × 10⁻¹⁵** — machine noise. Redistributing the same total burden among lesions changes
nothing.

| requested CV | realised CV | P(≥1 lesion fails) |
|---|---|---|
| 0.00 | 0.0000 | 0.213372138933446 |
| 0.50 | 0.4377 | 0.213372138933447 |
| 1.00 | 0.9452 | 0.213372138933447 |
| 2.00 | 1.7892 | 0.213372138933447 |

*(A first run of this test sat at P = 0.999999999962 — pressed against 1, where nothing can move
whether or not the mechanism moves it. That run could not have failed and was rerun at
P ≈ 0.21. The failed configuration is kept in the output.)*

## 2. What does make it depend on more than the mean

Let the per-bacillus escape probability vary between lesions too — which is exactly what their own
paper implies, since lymph nodes "exhibit reduced bacterial killing during drug treatment". Then,
**exactly**:

```
log P(all lesions sterilise)  =  − n · [ E(q)·E(N)  +  Cov(q, N) ]
```

Verified to **2.9 × 10⁻¹⁶** relative error at negative, zero and positive correlation.

The entire "spread" question collapses to **one number: n·Cov(q, N)** — the covariance across
lesions between residual burden and drug escape. It is zero exactly when burden and escape are
uncorrelated, and only then is the mean sufficient.

**A falsifiable directional prediction.** Their paper reports that lymph nodes both kill *Mtb*
poorly under treatment *and* carry multiple barcodes (higher burden). That is positive covariance.
If it holds, then every mean-field estimate of relapse risk — which is to say, every estimate
currently in use — **understates** the risk:

| corr(q,N) | n·Cov | exact P(relapse) | mean-field | log₁₀ ratio |
|---|---|---|---|---|
| −0.63 | −0.308 | 0.3758 | 0.5414 | −0.159 |
| −0.14 | −0.088 | 0.4410 | 0.4883 | −0.044 |
| +0.17 | +0.075 | 0.4730 | 0.4321 | +0.039 |
| +0.83 | +0.281 | 0.5396 | 0.3904 | +0.141 |

The error is exactly zero at zero correlation and grows with |n·Cov|. The magnitude above is set by
the dispersion assumed; the *structure* — sign, and proportionality to n·Cov — is exact.

## 3. A structural degeneracy in the animal-level data

An animal relapses if at least one of its *n* lesions disseminates, each with probability *p*:

```
P(relapse) = 1 − (1 − p)ⁿ
```

The likelihood of their observed **8 of 12** depends on (*n*, *p*) **only through n·log(1−p)**.
Along that curve the log-likelihood is flat to **1.2 × 10⁻¹⁶**:

| n lesions | p per-lesion dissemination | log-likelihood |
|---|---|---|
| 2 | 0.4227 | −7.6381700195 |
| 10 | 0.1040 | −7.6381700195 |
| 60 | 0.0181 | −7.6381700195 |
| 150 | 0.0073 | −7.6381700195 |

**This is a proof, not a power problem.** No number of additional animals separates "many lesions
each rarely disseminating" from "few lesions each often disseminating". Exact Clopper–Pearson 95%
CI on 8/12 is [0.3489, 0.9008], mapping to n·(−log(1−p)) ∈ [0.4291, 2.3102].

**Their barcode assay is the only thing that breaks it.** Their measured 42% median dissemination
fraction *is* p. Feeding it in fixes *n* — the effective number of independently dangerous lesions
per animal. That number is not in the paper and is computable from data they already hold.

## 4. Control

- q = 0 (perfect sterilisation) gives P(≥1 fails) = 0 exactly. ✓
- Independence is an assumption doing real work: independent lesions give 0.920, perfectly
  correlated lesions give 0.265 — a **3.47×** difference. Its size is shown, not assumed small.

---

## What is NOT claimed

- No raw data from this group was in hand. Every input above is quoted from the paper.
- The headline result is a **negative**: the spread of residual burdens does not move relapse
  probability at all under the independent-bacillus model.
- The magnitudes in §2 depend on an assumed dispersion. The **identity** does not.

## UNRETRIEVED — the data request, precisely

| What | Why it matters |
|---|---|
| Per-lesion CFU distribution (plotted, not tabulated in the retrievable text) | Sets Σ Nᵢ, which is the whole of §1 |
| Number of scan-matched granulomas harvested per animal | The n that §3 shows is otherwise unidentifiable |
| Joint per-lesion (burden, drug exposure or killing) | Sets Cov(q,N) — the single number §2 shows the answer turns on |

The third has, as far as we can find, never been measured jointly by anyone. It is the number.

## The question, as a question

*Could the effective number of independently dangerous lesions per animal be extracted from your
existing barcode data — and would the burden-versus-killing covariance across lesions be
measurable in the same necropsies?*

---
*Computation: `rem/atlas/candidates.py` case A. Gates predeclared and committed before running.
Full output with all gate verdicts including failures: `rem/atlas/RESULTS_candidates.txt`.*
