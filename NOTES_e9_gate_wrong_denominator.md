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
