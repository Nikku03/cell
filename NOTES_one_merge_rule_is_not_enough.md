# One merge rule is never enough to call two signals unmergeable

This has now produced a false negative twice in the same session, in two unrelated arms, and both
times the conclusion was recorded in the repository before it was overturned.

## The two cases

**Loop 161, chemistry and connectivity.** S6 fused the Markov walk with the mass-balance score using
an equal-weight RANK sum, measured 0.8053 against the walk's 0.8158, and concluded they substitute.
A six-family design workflow then took the same two signals to **0.9406–0.9996** by fusing in SCORE
space at a frozen scale, and separately measured rank-space fusion at 0.8502. The signals were
independent all along — Spearman **+0.00008** over the full candidate set — and the merge rule was
the entire difference.

**Loop 163b, sequence and structure.** C5 fused ESM-2 embeddings with structural descriptors by
z-scoring both blocks and concatenating them into one cosine k-NN, measured −0.0012, and concluded
structure adds nothing. Loop 163c then found **five rules that beat sequence alone**, all held out:

| rule | vs sequence | sem |
|---|---|---|
| score-space `seq + 0.1·struct` | **+0.0058** | held out both halves |
| learned logistic merge | **+0.0058** | held out both halves |
| RRF k=5 | +0.0046 | 3.5 sem |
| RRF k=60 | +0.0045 | 3.5 sem |
| max(rank) | +0.0043 | 3.3 sem |
| rank product | +0.0006 | 0.4 sem |
| **concatenation (C5's rule)** | **−0.0012** | the only one that lost |
| min(rank) | −0.0594 | −16.7 sem |

Concatenation was the single worst rule that isn't actively perverse. A 480-dimensional ESM block
and a 64-dimensional structure block sharing one cosine distance means structure contributes about
an eighth of the geometry whatever it knows, and per-column z-scoring does not fix a per-BLOCK
imbalance.

## The rule this leaves behind

A negative result about merging is a claim about the RULE unless the rule space has been searched.
Before recording "these signals do not combine", a loop must either try a family of rules — a
weighted sum in score space with the weight held out, a rank fusion, a disjunction, a learned
combiner — or measure the ceiling and show there is nothing to recover.

## Measure the ceiling first, so a null is diagnosable

Loop 163c's M2 gate runs BEFORE any merge rule is tried and reports two numbers:

- the **Spearman** between the two arms' per-case scores
- a **per-case oracle** that picks the better arm using the answer — a ceiling, not a predictor

| | walk vs balance (loop 161) | sequence vs structure (loop 163c) |
|---|---|---|
| Spearman | +0.00008 | +0.5858 |
| oracle gain over best single | +0.07 | +0.0130 (10.4 sem) |
| best merge achieved | +0.064 | +0.0058 |
| fraction of headroom captured | ~91% | ~45% |

Both merges are real. They differ by an order of magnitude, and the ceiling predicted that in
advance. Had M2 come back with an oracle at the best single arm, the null would have been
attributable to the signals rather than to the search — which is the whole point of measuring it
first.

## What this does NOT change

Structure alone scores 0.6993, below a raw 5-mer string lookup at 0.7276, on a benchmark where a
frequency column scores 0.4802. Adding it to sequence buys +0.0058 on 0.7941. That is real and it
is small, and it does not justify ~9,600 AlphaFold downloads for the dark proteome. The correction
here is to the CLAIM "structure adds nothing", not to the decision that followed it.

## One instability worth recording

The learned merge reached the same +0.0058 in both directions with inconsistent coefficients: the
structure score and structure rank terms were `[0.761, 0.426]` in one fold and `[-2.098, 2.107]` in
the other — same magnitude of gain, opposite signs on the structure score. It is finding the gain
through different terms in different folds, which is what a small signal in a correlated pair looks
like, and is a reason to prefer the fixed-weight score-space rule over the learned one here.
