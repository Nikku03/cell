# Loop 173's E9 was gated against a denominator the data cannot have

E9 predeclared, before the numbers, that "more than 99% of above-threshold motif matches sit in
elements that are NOT validated enhancers", citing the futility theorem (Wasserman & Sandelin,
Nat Rev Genet 2004: >99.9% of genomic motif matches are non-functional). It measured 0.8958 and
FAILED.

The gate could not have passed. The futility claim is about the GENOME. E9 evaluated it on the EP
CRISPR benchmark's element set, which is not the genome -- it is a curated candidate pool that the
screen designers pre-filtered to regions they already believed might be enhancers. In that pool
426 of 4,482 elements are validated positives (9.50%), and those elements hold 10.55% of the
element base pairs. So even if matches were distributed perfectly uniformly, the fraction outside
positives would be 0.894, and 0.99 was arithmetically unreachable from the moment the pool was
chosen. This is the "null that cannot move" family in gate_guard.py, wearing a different hat: not a
control that cannot change the statistic, but a THRESHOLD the statistic cannot reach.

## What the same numbers do say, once the denominator is right

Matches: 520,817 above threshold across 736 motifs over 3.04 Mb of element sequence.
  positive elements   9.50% of elements
  positive bp         10.55% of element base pairs
  matches in them     10.42% of all matches

10.42 / 10.55 = 0.99. Motif matches are distributed UNIFORMLY per base pair, with no measurable
enrichment in the elements CRISPR proved are enhancers. That is a stronger statement than the one
E9 tried to make, it is the futility theorem's actual content restated on real ground truth, and it
explains every other failing gate in the loop: there is nothing for a motif-count filter to grip.

The correct genome-scale version of E9 needs a comparison against random genomic windows rather
than against the other members of a pre-filtered pool. That is a different measurement and it is
not made here; the loop's E9 is recorded as FAILED and as MIS-SPECIFIED, and the ratio above is
reported in its place.

## A second, smaller defect in the same loop

`n_frac` is a constant-zero column in every arm: the liftover left no element with any N, so the
column carries no information. It is harmless -- a constant feature cannot change a tree's splits
-- but run_manifest.check_features exists precisely to catch it and was not called on these
frames. Noted rather than quietly deleted, because the arms were already run and reported with it.

---

# A second gate-design defect in the same arc: loop 180's R5 was too easy

E9's threshold could not be reached. R5's could hardly be missed. They are the same class of
mistake from opposite ends, and both are mine.

R5 asked whether the sequence stack's increment over distance falls as promoters become more
CpG-island-like -- the label-free version of "housekeeping genes do not need distal enhancers".
The gate was written as `Spearman(quartile, increment) < 0`. Over FOUR quartile points, a negative
Spearman is close to a coin flip: a null with no structure at all clears it about half the time.

The measurement it passed on:

    Q1 (CpG-poor)  n=50   increment +0.0320
    Q2             n=50   increment -0.0240
    Q3             n=49   increment +0.0041
    Q4 (CpG-rich)  n=50   increment +0.0240
    Spearman = -0.200

That curve is not monotone and its endpoints differ by 0.008. The commit reports it as weak
evidence, which is right, but the GATE said PASS, and a gate that says PASS on this is not
measuring what it was written to measure.

What it should have been: a permutation null -- shuffle the gene-to-quartile assignment a few
hundred times, recompute the Spearman, and require the observed value to sit outside the null's
central mass. That costs nothing and would have returned a p-value instead of a sign.

## The rule this arc keeps re-learning

Before a gate is written down, ask what the statistic does UNDER THE NULL. E9 never asked whether
0.99 was reachable given a pool that is 9.5% positive. R5 never asked how often four points give a
negative Spearman by chance. Neither question needs the data, and both would have been answered in
a minute.

## Loop 187's B6: a gate on a quantity that is identically zero in the data it reads

B6 was written to correct a framing in loop 175. Loop 175 reported human autoregulation as "24
self-loops among 795 curated regulators", 3.0% against E. coli's ~50%, and never asked what chance
gives. B6's suspicion was reasonable: with 55,716 edges over ~1,200 regulators and ~7,500 targets,
chance gives very few self-loops, so 3.0% might be a large ENRICHMENT reported as a small FRACTION.

The rerun returned:

    curated self-loops 0 over 1,177 regulators (0.0%)
    degree-preserving null 0.00 +/- 0.00  ->  z +nan

Zero. And not only in the curated tier -- all 612,133 edges of net_bundle.json.gz, across all three
tiers, contain no a -> a edge at all.

So B6 never measured autoregulation. Loop 175's 24 came from TRRUST v2, read via
colab/data/tf_autoregulation.json; the network B6 reads is CollecTRI/DoRothEA-derived and does not
encode autoregulation as a self-edge. The two numbers are about different files. B6 compared a
count from one source against a null built from another and called the result a correction.

The auto-generated FAIL text is wrong twice over, and is left in the log with this note against it:

    B6 FAIL -- z +nan; the self-loops are what chance gives and loop 175's framing stands

z is not below the bar, it is UNDEFINED -- 0 observed, 0 expected, sd 0, so the statistic is 0/0.
And loop 175's framing is neither confirmed nor refuted by a network that cannot represent the
thing loop 175 counted. The honest verdict is VOID, not FAIL, and the honest next step is to run
this against tf_autoregulation.json, which is the file that actually holds the 24.

## The rule, extended

The earlier version of this rule was: before writing a gate down, ask what the statistic does under
the null. B6 obeyed that and still failed, because it never asked the prior question --

  does the quantity this gate reads EXIST in the file the gate reads it from, and is it non-zero?

One `grep` for a self-edge would have answered it in a second and B6 would have been written
against TRRUST instead. E9 got the denominator wrong, R5 got the null wrong, and B6 got the FILE
wrong. All three are the same failure at different depths: a gate written from the idea rather than
from the data it will touch.

A related consequence worth stating separately: a gate whose statistic can be undefined needs a
third outcome. PASS/FAIL cannot express "the test did not apply", and forcing nan into FAIL puts a
false claim into the record -- here, a claim that loop 175 was right.
