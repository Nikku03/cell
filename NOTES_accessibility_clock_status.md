# Status of the accessibility clock

The one result in this project carrying a fourth dimension is loop 191d's finding that promoter
accessibility reaches half its plateau before the mRNA does. This file records what is and is not
established about it, so the number is never quoted without its qualification.

## What is measured

A549 lung carcinoma, 100 nM dexamethasone, ENCODE GGR series, one lab pinned. Promoter DNase
against polyA RNA on a shared grid, 1,310 responding genes, one-sided Wilcoxon p 6.4e-58, holding
inside all three magnitude terciles. Two negative controls pass: CTCF +0.061 and RAD21 -0.022,
architectural factors that should be and are inert to a steroid response.

## What is NOT established

REPLICATION. None. Loop 192 identified the ENCODE dendritic-cell LPS series as a clean candidate --
one lab for all 59 experiments, a graded clock, donor split-half +0.366 -- and measured that it
cannot answer the question: the A549 lead REVERSES when downsampled to that series' four
timepoints. Loop 196 then tested four estimators chosen to fail differently, a level-crossing
interpolation, an increment-weighted mean, an area integral and a parametric exponential fit. All
four recover the lead on eleven points and none on four. The limit is the information in four
timepoints, not the estimator.

THE SIZE OF THE EFFECT. "The A549 lead" is not one number across these loops. Loop 191d reports
+48 min, loop 192 reports +154 and loop 196 reports +101.6 for the same statistic, because each
uses a different grid, replicate set and accessibility baseline convention. Each loop's internal
comparison is like for like; the cross-loop numbers are not, and none of them should be quoted as
"the" lead.

CAUSATION. Nothing perturbs accessibility anywhere in this arc. Accessibility opening before
transcription is equally consistent with chromatin gating transcription and with a third factor
driving both at different lags.

## Search for a denser series (no qualifying public series was found)

Loop 197 broadened loop 192's ENCODE query beyond its treatment-duration filter, which could not
have seen a differentiation series, and checked the broadened query against A549 as a positive
control before believing any absence.

## How this must be quoted

In the census capability table and anywhere else: measured in A549 under dexamethasone,
UNREPLICATED, with the effect size unstable across analysis choices. Not as a general property of
human gene regulation.
