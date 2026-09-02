# Loop step 1 — the model proposes. Written and committed BEFORE the truth file was opened.

## What I was given

Summary statistics only — mRNA mean/Fano/skew and protein mean/Fano/skew/quantiles — at six
induction levels. No access to the generating model.

## The three facts that matter

```
              observed / base-model ratio
 induction   mRNA mean   mRNA Fano   prot mean   prot Fano
      0.35       1.000       1.000       1.000       0.916
      0.55       1.000       1.000       1.000       0.931
      0.80       1.000       1.000       1.000       0.945
      1.00       1.000       1.000       1.000       0.952
      1.45       1.000       1.000       1.000       0.965
      2.10       1.000       1.000       1.002       0.977
```

1. **mRNA is untouched.** Mean and Fano match the base model to 1.000 at every condition.
2. **The protein MEAN is untouched.** Also 1.000 everywhere.
3. **The protein FANO is reduced**, by 8.4% at low induction, shrinking to 2.3% at high.

## Ruling out, mechanism by mechanism

| candidate | prediction | verdict |
|---|---|---|
| bursty transcription | raises mRNA Fano | **out** — mRNA identical |
| refractory promoter | changes promoter dwell → changes mRNA Fano | **out** — mRNA identical |
| negative feedback | protein represses transcription → changes mRNA | **out** — mRNA identical |
| constitutive leak | adds transcription → raises mRNA mean | **out** — mRNA identical |
| gene dosage | extra template → raises mRNA mean | **out** — mRNA identical |
| saturating protein decay | protein accumulates → raises protein mean, super-linearly | **out** — protein mean is 1.000 and linear |
| **non-exponential protein decay** | mRNA untouched; protein mean untouched; protein Fano **reduced** | **the only survivor** |

Facts 1 and 2 together are the discriminator. Anything acting upstream of protein moves mRNA.
Anything acting on the protein *rate* moves the protein mean. A mechanism that touches neither
while still moving the variance can only be changing the **shape of the protein degradation
waiting time at fixed mean lifetime**.

## The proposal, with a number attached

**Missing mechanism: protein degradation is multi-step (Erlang / phase-type), not memoryless.**

Quantitative check, which is what turns a story into a proposal. For a species removed by an
Erlang-k waiting time at fixed mean lifetime, the decay's contribution to the Fano factor falls
from 1 to `(k+1)/(2k)`, a deficit of `(k-1)/(2k)`. The observed absolute Fano deficit at the
lowest induction is `4.117 - 3.769 = 0.348`. Solving `(k-1)/(2k) = 0.348` gives

    k = 3.3 steps

and the prediction that the *relative* deficit must shrink as induction rises, because the fixed
0.35-unit deficit is measured against a total Fano that grows from 4.1 to 11.2. Predicted
relative deficit at the highest induction: `0.348 / 11.226 = 3.1%`. Observed: **2.3%**. Same
direction, right magnitude, from one fitted number.

**Committed prediction: `erlang_protein_decay`, with k between 3 and 5.**

## What I expect the solver to say next

Section 7.2 of the spec says waiting-time shape matters only on the step that GATES the rare
event. Protein degradation is on the causal path to a high-protein excursion, so I expect this
to be priced as a real tail effect, not a negligible one — and I expect the direction to be that
the base model **overstates** the upper tail, since removing decay-timing noise narrows the
distribution.
