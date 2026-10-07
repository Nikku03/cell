# Kinetics calibration — is CatPred's kcat error learnable?

The kinetics layer assigns each metabolic enzyme a turnover number `kcat`. ~380 enzymes carry a
literature-**measured** kcat (`tier: measured`); the rest are predicted by **CatPred** (a DL model from
substrate SMILES + enzyme sequence), with a small global calibration. The obvious next move: use the measured
subset to *learn a correction* for CatPred that also improves the predicted enzymes. We tested that honestly.
`colab/kcat_calibration_check.py` — 11th recovery-scorecard axis (`kcat_calibration_honest`).

## Result: there is a pattern, but it is not learnable — use CatPred as-is

On the clean in-vitro label set (278 truly-measured enzymes that also have a CatPred prediction):

| step | finding |
|---|---|
| **CatPred raw accuracy** | median **3.30× fold-error**, 37% within 2× — this is near the lab-to-lab kcat scatter floor |
| **residual is systematic?** | weak **regression-to-mean**: `r = −0.31` of residual vs prediction (CatPred slightly compresses the dynamic range). Real, detectable — but weak. Km `r=−0.22`, uncertainty `+0.19`; protein length `~0`. |
| **does a correction generalise?** | **No.** 5-fold CV: global bias 3.30→3.27× (noise), linear de-shrinkage 3.30→**4.51×**, GBM **5.49×**. A 20-seed shrinkage scan puts the **CV-optimal shrinkage at s = 1.0 (CatPred unchanged)** — every damped correction monotonically worsens. |

The residual carries a genuine but weak regression-to-mean signal that is swamped by irreducible measurement
noise. Nothing fit on the measured subset lowers held-out error. **CatPred should be used at its measured
accuracy**, with only the existing minimal global median-bias calibration (`catpred_log10_bias = 0.133`, i.e.
~1.36×, applied to *predictions only* — real measurements are kept as ground truth).

## The trap we avoided: a label-quality artifact

An earlier quick test suggested "recalibration wins ~20% (3.39→2.70×)." That used davidi's
`kcat_measured_per_s`, but only 128/456 of those are truly measured — the rest are EC-class /
network-propagated / prior **fallbacks**. Splitting by davidi's own tier and re-fitting the same recalibration:

| davidi label tier | n | CatPred raw | "recalibrated" | Δ |
|---|---|---|---|---|
| **measured** (real) | 89 | 5.46× | 11.68× | **+6.23 worse** |
| **EC-measured** (real) | 39 | 4.92× | 12.72× | **+7.81 worse** |
| network-propagated (synthetic) | 144 | 2.60× | 1.06× | −1.54 "better" |
| global-prior (synthetic) | 36 | 8.50× | **1.00×** | −7.50 "better" |

The "win" was the model learning to **reproduce the priors** on partly-computed labels. On real measurements
the identical fit is 6–8× *worse*. Had we wired that recalibration into the kinetics layer, it would have
silently corrupted every predicted kcat. The scorecard axis now **locks this out**: it passes only while no
fitted correction beats CatPred-as-is on real measurements.

## Why this is the right answer (in-vitro vs in-vivo)

CatPred predicts **in-vitro** kcat and matches the in-vitro literature labels at 3.3×. Davidi's "measured"
values are **in-vivo effective** kcat (back-derived from flux/abundance) — a different, noisier target (CatPred
raw 5.3× against them). The place cell context *should* help is that in-vivo effective kcat, not in-vitro kcat
— a separate estimator, not a correction to CatPred. That remains open (see `docs/CELLGRAPH.md` scale path).
