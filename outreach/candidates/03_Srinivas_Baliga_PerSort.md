# PerSort and the purity of a rare gate — Srinivas & Baliga

**Their paper.** According to PubMed, Srinivas V, *et al.*, "PerSort Facilitates Characterization
and Elimination of Persister Subpopulation in Mycobacteria", *mSystems* 5:e01127-20 (2020),
PMID 33262242, [DOI](https://doi.org/10.1128/mSystems.01127-20).

They built a FACS method to isolate translationally dormant mycobacteria and showed that
L-cysteine eliminates them. Three of their stated numbers, taken together, produce an arithmetic
result none of them produces alone.

> "a consistent subpopulation of translationally dormant 'dim' cells was also present, reaching
> about 1% of the population"

> "MSM cultures grown in nutrient-rich conditions were ~5% tolerant to INH and RIF"

> "only ~7% of cells were inappropriately sorted and ... the sorting efficiency for single
> bacterial cells was ~93%"

**Nothing below says any of those numbers is wrong.** Each is correct where it was measured. The
question is only whether the third transfers to a gate on the first.

---

## 1. Efficiency is not purity

A misclassification rate ε applied to a subpopulation at prevalence π gives gate purity

```
purity = π(1−ε) / [ π(1−ε) + (1−π)ε ]
```

**Control first:** at a balanced mixture (π = 0.5, which is how a two-strain sorting control is
usually run), ε = 0.07 returns purity **0.9300** — reproducing their stated 93% exactly. The figure
is right at its own operating point.

**Transfer:**

| ε \ π | 0.004 | 0.01 | 0.05 | 0.50 |
|---|---|---|---|---|
| 0.001 | 0.8005 | 0.9098 | 0.9813 | 0.9990 |
| 0.005 | 0.4442 | 0.6678 | 0.9128 | 0.9950 |
| 0.010 | 0.2845 | 0.5000 | 0.8390 | 0.9900 |
| 0.030 | 0.1149 | 0.2462 | 0.6299 | 0.9700 |
| **0.070** | 0.0507 | **0.1183** | 0.4115 | 0.9300 |

Closed form verified against a 10⁷-cell Monte-Carlo: 0.118321 vs 0.118293 ± 0.000364, agreement
0.08σ.

## 2. The consequence runs in their favour

An impure gate **attenuates** a real difference. At π = 0.01, ε = 0.07:

```
observed difference  =  true difference × 0.1176        (verified, 1.35σ)
```

So any genuine dim-versus-lit difference they measured is **diluted 8.5×**. Their −4.17 log₂ fold
change in 16S rRNA would be a **floor**, not an effect size. The correction makes their result
larger, not smaller.

This cuts both ways and both directions are worth knowing. Which one applies depends entirely on
the false-positive rate of the *dim gate specifically*, which is not the same quantity as overall
sort efficiency.

## 3. A ceiling that needs no model at all

Dim cells are ~1% of the population. ~5% of cells survive 5× MIC INH/RIF. Therefore, **even if
every single dim cell survives**, dim cells are at most **20% of the survivors**.

For dim cells to account for a larger share, their survival advantage over lit cells would have to
be:

| share of survivors | required dim:lit survival ratio |
|---|---|
| 50% | 99× |
| 90% | 891× |
| 99% | 9,801× |

This is an arithmetic identity with two of their published percentages as its only inputs. It says
the translationally dormant subpopulation **cannot by itself be the tolerant subpopulation** unless
the survival ratio is very large. Either that ratio is real and measurable, or a second route to
tolerance exists outside the dim gate.

Both are testable with sorts they already run.

---

## What is NOT claimed

- Not that 93% is wrong. It is right where it was measured.
- Not that their conclusions are wrong. §2 makes their measured effects **larger**.
- The purity figures are only as good as the misclassification rate **for the dim gate**, which is
  a different quantity from overall sort efficiency and is not separately stated.

## UNRETRIEVED — the data request

| What | Why |
|---|---|
| The mixing ratio of the MSM-mEos2 / MSM-mCherry control culture | It is the prevalence at which the 7% applies. Everything in §1 turns on it. |
| The false-positive rate of the **dim gate** specifically (lit cells landing in the dim gate), rather than symmetric two-strain sort efficiency | This is the ε that actually governs purity |
| Per-cell survival of PerSorted dim vs lit at 5× MIC as an absolute ratio | Tests §3 directly |

## The question, as a question

*Given that dim cells are ~1% and tolerance is ~5%, is the dim:lit survival ratio large enough to
close that gap — and if the dim-gate false-positive rate were measured at 1% prevalence rather than
in a balanced mixture, would the measured dim-versus-lit differences turn out to be floors?*

---
*Computation: `rem/atlas/candidates.py` case C. Full output: `rem/atlas/RESULTS_candidates.txt`.*
