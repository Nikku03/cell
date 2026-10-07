# The degree baseline leaked the answer, and "counting beats walking" was the leak

## What was wrong

`REM.degv` — and the identical inline `deg` counters in `colab/loop_rem_bipartite_markov.py` (loop
160) and `colab/loop_rem_chemistry.py` (loop 161) — counted each species' reactions over the
**whole graph, including the held-out reaction**.

The walk was correct: `R.operator(j)` deletes reaction `j` from the operator. The degree baseline
was not. And because every case excludes its own seeds from the candidate set, the `+1` that the
held-out reaction contributes lands on **exactly the true targets and nothing else**. The
"baseline" carried a one-sided stamp of the answer.

Three of the six independent verifiers in the merge-design workflow found this in the same run,
without being told to look for it.

## Measured, on 400 DEV cases

| baseline | AUC | MRR |
|---|---|---|
| degree, as shipped (leaky) | 0.7230 ± 0.0125 | 0.0514 |
| degree, honest (`j` deleted) | **0.5867 ± 0.0178** | 0.0506 |
| walk (always correct) | 0.6843 ± 0.0161 | 0.3618 |
| balance | 0.8770 ± 0.0104 | 0.3738 |

Inflation: **+0.1363 AUC**. A verifier's independent sample put it at +0.1293 (21 sem).

## What this overturns

**Loop 160's R5 FAILED because of this bug.** Its finding was "the Markov walk scores 0.6928 against
a degree column at 0.7149 — counting beat walking", and that was recorded in the commit message, in
the loop artefact, and repeated in the chat summary.

Paired on the same 400 cases with the leak closed:

```
honest degree − walk  =  −0.0976,  sem 0.0145  =  −6.7 sem
```

**Walking beats counting.** Loop 160's R5 gate required the walk to exceed degree by more than 0.02;
with the honest baseline the margin is +0.0976, so R5 would have PASSED and loop 160 was 7/7, not
6/7.

The wider claim in that arc — that this repository keeps losing to a popularity column (loops 120,
130, 160) — is not withdrawn for loops 120 and 130, whose confounds were regulator counts and
publication counts and were computed without any held-out deletion. It is withdrawn for loop 160.

## Two further leaks found by the same verifiers, recorded but not yet closed

**Orphaned-by-deletion.** A candidate whose only reaction was the held-out one has degree exactly 0
after deletion, and since seeds are excluded it can only be a true target. 1,018 of 8,428
non-currency species have full degree 1; it fires in ~5% of cases. A degree baseline is unharmed
(0 ranks last) but any LEARNED model can read it as a perfect one-sided label, and must clip it.

**Duplicate reactions.** In 10.8% of cases another reaction survives deletion that maps the same
seeds onto the same targets — 21.5% ignoring direction. On those the walk alone scores AUC 0.9997,
which is memorisation. This inflates the walk-alone baseline, so it makes merge margins measured
against the walk conservative rather than optimistic.

## The fix

`REM.deg_minus(j)` computes the honest degree and `case()` now returns it as `case["degv"]`.
`R.degv_leaky` is kept, named for what it is, so the pre-correction numbers remain reproducible.
`REM.duplicate_survives(j)` reports the memorisation channel.

## Why it survived two loops

Loop 160 gated the walk against the degree column and loop 161 re-used the same column, so both
loops asked "does the walk beat counting" and neither asked "is the counting honest". The deletion
discipline was applied to the object under test and not to its control. A control that is not
subject to the same ablation as the thing it controls is not a control.
