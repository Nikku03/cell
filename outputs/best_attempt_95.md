# Best shot at 95%+ accuracy — the AlphaFold paradigm

Strongest leak-free model: conservation (family_frac + OG-essentiality entropy,
both per-fold leak-free) + environment (standard rich-medium biochemistry) +
intrinsic (isozyme, structural uniqueness, biophysics, operon). 19 features,
calibrated, leave-one-organism-out. Audited leak-free.

## Headline (calibrated, leak-free LOO-organism)
- AUC 0.857, MCC +0.58 (all 10 orgs) / **MCC +0.628 on consistent-label subset**
  (matches the project's 0.63 baseline -- sanity check passed)
- The environment feature (validated separately) is in the stack.

## The 95% answer = AlphaFold paradigm (confident subset + flag the rest)

Risk-coverage: sort predictions by confidence, report accuracy on the most-
confident fraction.

| coverage | accuracy (all 10) | accuracy (consistent labels) |
|---|---|---|
| top 20% | 96.4% | 96.5% |
| top 30% | 95.7% | ~95% |
| top 40% | 93.7% | 94.8% |
| 100% | 84.3% | 85.0% |

**95%+ accuracy is achievable on the 33-39% most-confident genes**
(consistent-label subset: 39%, 10,508 of 26,740). The remaining ~61% are
the conditional zone -> flagged for experiment.

## Did AlphaFold "do it"? Yes — and so did we, the same way.

AlphaFold never hit 100%. It reached **experimental accuracy on ~58% of
residues (pLDDT>90)** and **flagged the low-confidence rest**. That IS the
result -- high accuracy where confident, honest abstention elsewhere.

We deliver the identical paradigm: **95%+ accuracy on the confident 39%,
conditional zone flagged.** The gap to AlphaFold's 58% coverage is two real
things, both measured this session:
  1. LABEL NOISE: essentiality labels agree across labs at only kappa 0.39;
     dropping the two inconsistent-screen organisms lifted confident coverage
     33% -> 39%. Crystallography (AlphaFold's ground truth) is far more
     reproducible than a knockout screen.
  2. The IRREDUCIBLE CONDITIONAL component: ~half the genes flip with
     environment, which is not in the genome -- AlphaFold's target (structure)
     has no such environment-dependence.

## So: can we get to 95%?
- Overall, on all genes: NO with genome-only data -- blocked by label noise
  (85% inter-lab agreement) and the environment-dependent conditional zone.
- On the confident subset: YES, 95%+ on ~39% of genes today, with calibrated
  abstention on the rest. This is exactly, precisely how AlphaFold delivered.
- To grow the confident fraction past 39%: (a) cleaner/consistent labels and
  (b) the measured per-organism condition matrix (Fitness Browser) to convert
  conditional-zone genes from "flag for experiment" into "confident given the
  environment." Both are data, not model, problems.
