# 5-organism cascade test with REAL Phase 2 stream — the honest result

Run on Colab with `--phase2-dir` (real Phase 2 XGBoost predictions from Drive).
This supersedes the sandbox MOCK-phase2 run.

## THE KEY FINDING: Phase 2 is a POOR voter for STRICT essentiality

Phase 2 predicts **conditional vulnerability** (strong_hit under a specific
condition). The cascade test labels are **strict essentiality** (the
`essential` column). These are DIFFERENT TARGETS. Tested against strict
labels, the Phase 2 stream looks terrible:

| organism | phase2 precision | phase2 n_called | base rate |
|----------|-----------------:|----------------:|----------:|
| Keio     | 0.070 | 1,504 / 3,585 | 0.169 |
| Putida   | 0.056 | 3,839 / 4,715 | 0.194 |
| BFirm    | 0.364 | 1,875 | 0.428 |
| Burk376  | 0.243 |   622 | 0.386 |

On Putida, Phase 2 flags 81% of genes as "essential" -- because
max-over-conditions Phase 2 fires for any gene that's a hit under ANY
stress, and almost every gene is conditionally vulnerable somewhere. That
is NOT strict essentiality; it's a different, broader set.

**This empirically re-confirms the project's core thesis**: strict
essentiality (conservation/Phase 0) and conditional vulnerability (Phase 2)
are DIFFERENT phenotypes. You cannot naively combine a strict-essentiality
predictor with a conditional-vulnerability predictor against one label set
and expect agreement to help -- it's a category mismatch.

## CORRECTION of my earlier prediction

I predicted: "real Phase 2 (different blind spot) would lift the cascade
MORE than the mock." WRONG. The mock (derived from conservation) AGREED
with conservation, so it boosted apparent agreement. The real Phase 2
DISAGREES with conservation (it predicts a different target), so as a
strict-essentiality voter it adds noise, not lift. The honest lesson:
cascade agreement requires streams predicting the SAME target.

## What DID work: gold tier on same-target agreement

The UNANIMOUS (gold) tier still hits high precision, driven by
conservation x cooccur agreement (both predict strict essentiality):

| organism | gold precision | gold recall | gold coverage | base rate |
|----------|---------------:|------------:|--------------:|----------:|
| Keio     | 0.734 | 0.524 | 0.121 | 0.169 |
| Putida   | 0.796 | 0.320 | 0.078 | 0.194 |
| BFirm    | 0.900 | 0.200 | 0.095 | 0.428 |
| Burk376  | 0.931 | 0.195 | 0.081 | 0.386 |

IMPORTANT: gold precision correlates with BASE RATE. BFirm/Burk376 have an
atypical ~40% essential rate (Burkholderia, loose RB-TnSeq essentiality
calls), which inflates precision. The TYPICAL-base-rate organisms
(Keio 17%, Putida 19%) give the honest number: **gold tier ~0.73-0.80
precision at ~0.32-0.52 recall, ~8-12% genome coverage.**

## Honest deployment number from this real test

For a typical bacterium (~17-20% essential base rate):
  GOLD tier (same-target stream agreement): ~0.75-0.80 precision,
    ~0.32-0.52 recall of strict essentials, ~8-12% of genome.

This is LOWER than the inflated mock-run numbers (0.92-0.96) because:
  1. the mock phase2 artificially agreed with conservation
  2. BFirm/Burk376's high base rate flattered precision

## Architectural lessons (measured, not theorized)

1. **Cascade agreement works only for same-target streams.** conservation
   x cooccur (both strict essentiality) -> real precision lift. Adding a
   different-target stream (Phase 2 conditional) -> noise.

2. **To use Phase 2 in a cascade, test it against CONDITIONAL labels**, not
   strict essentiality. The right cascade for drug-target discovery is
   conditional-vulnerability streams agreeing on conditional labels -- a
   SEPARATE cascade from the strict-essentiality one.

3. **Two cascades, two products:**
   - strict-essentiality cascade (conservation + cooccur + FBA) -> "what's
     the core essential gene set" -> ~0.75-0.80 gold precision
   - conditional-vulnerability cascade (Phase 2 + atlas + MoA-kernel) ->
     "what gene-drug pairs are adjuvant targets" -> the Phase 2 LOO numbers
     (recall@P30 0.46-0.64)
   These should NOT be mixed against one label set.

## Bottom line

The cascade principle is VALIDATED for same-target streams (gold tier
~0.75-0.80 precision at typical base rates, real measured). The naive
4-stream cascade mixing strict + conditional predictors does NOT work --
and that failure is itself the re-confirmation that strict and conditional
essentiality are different phenotypes requiring different cascades. My
earlier "real Phase 2 lifts more" prediction was falsified by the data.
