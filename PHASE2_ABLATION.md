# Phase 2 feature ablation + the litmus verdict (loo-org-fast, 4 held-out orgs)

All numbers are leave-one-ORGANISM-out, macro across {SB2B, Keio, Phaeo,
pseudo5_N2C3_1}. AUPRC is on the 10:1 downsampled test (≈10x optimistic);
recall@P30 is base-rate-invariant and is the honest headline.

| feature set                              | AUPRC | recall@P30 |
|------------------------------------------|-------|------------|
| conservation + genome context (baseline) | 0.714 | 0.873      |
| + domain (Pfam/TIGRFam) features         | 0.747 | 0.887      |
| + og_cpd prior (LEAKY - bug)             | 0.733 | 0.798      |
| + og_cpd prior (leak-fixed)              | 0.772 | 0.903      |

Locked feature set: family_frac_essential_fold[k] (leak-free per held-out
clade) + numeric gene context (family_n_orgs, paralogs, orphan) + numeric
condition context (MW, concentration, pH, temperature, aerobic) + expGroup
hash + 150 domain multi-hot + n_domains + seed_class + og_cpd_hit_rate +
og_cpd_n + additive_pred.

## Why og_cpd had to be leave-OWN-org-out
exclude-test-org-only computed each TRAIN row's rate including its own org,
so the feature looked strong at fit time but excluded the test org at predict
time -> distribution mismatch -> recall@P30 dropped to 0.798. Computing the
rate over all orgs except BOTH the held-out test org AND the row's own org
makes train and test consistent and leak-free -> recall@P30 0.903.

## Bacitracin litmus verdict (honest)
xgb LOO-organism rank percentile of our validated cluster on the SB2B fold:
  envZ ~10.5%   ompR ~7.5%   pspB ~38.7%

This is NOT a top-1% recovery, and that is the correct result, not a model
failure: envZ/ompR x bacitracin is a strong hit in only ~2 of ~30
bacitracin-tested organisms (PV4, SB2B -- both Shewanella). Its global
(OG x compound) hit rate is ~0.05 vs base rate 0.008. The vulnerability is
clade-specific and rare; no global feature can make a 2/30 signal a top pick
without overfitting. A leave-one-organism-out model ranking it top ~8%
("well above average, not a top pick") is honest.

Separation of claims (do not conflate):
- Phase 1 validated the bacitracin lead DIRECTLY from fitness data (t up to
  16, dose-response, specificity, literature gap). That is the evidence.
- Phase 2 is a GENERAL conditional-vulnerability predictor (recall@P30 0.903).
  It is not, and need not be, the evidence for the bacitracin finding.

## Honest hard bound still to measure
Leave-one-CLADE-out (hold out ALL Shewanella): og_cpd rate for the cluster
-> 0, so the model should have ~no signal for it. That is the true ceiling
on de-novo (not cross-organism-propagated) prediction of a clade-specific
vulnerability, and should be reported alongside the LOO-organism headline.
