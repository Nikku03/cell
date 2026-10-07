# What REM can actually offer each candidate

Matched against the verified list. Graded honestly: three of these are strong, three are
partial, and two are weak enough that leading with REM would waste their time. The weak ones
are marked as such because sending the wrong offer to a good lab costs more than not sending.

**The one-line offer, common to all of them:** *their assays cannot reach the endpoint they
care about.* A time-kill curve or a CFU count bottoms out at 10-100 cells; below that "no
growth" is an inference, not a measurement. Relapse lives entirely underneath. To measure a
1e-9 eradication probability directly takes a billion replicates. The curve **above** the
floor fixes the rates; the rates fix the probability **below** it.

---

## STRONG — the offer is specific, computed, and they can check it

### 1. Nathalie Q. Balaban — Hebrew University
*Nature* 513:418-21 (2014), PMID 25043002, [DOI](https://doi.org/10.1038/nature13469).

Her result gives the **peak**: evolved lag matches the exposure interval. We reproduced that
peak from first principles (4 of 4 exposure durations). The offer is the **shape around it**,
which her paper does not report and which follows immediately from having the exact survival
surface:

```
 exposure   peak lag   selection strength   curvature
   1.5 h      1.48           0.067            -0.050
   2.5 h      2.32           0.200            -0.174
   3.5 h      2.91           0.841            -0.371
   5.0 h      4.57           2.800            -0.704
```

**Selection on lag strengthens 42x as the exposure lengthens.** The testable consequence:
evolved lag should scatter widely across replicate populations at short exposures and cluster
tightly at long ones. That is a **trend across conditions**, not a single number, so it is hard
to match by chance — and her 2014 experiment already ran replicate populations, so the data to
refute it exists.

*Stated honestly in the email:* 3 of the 4 peak-widths are grid-limited rather than measured,
so the width numbers are lower bounds. The selection strengths are not grid-limited and are
the robust half.

### 2. Jérémie Roux and Madalena Chaves — Inria / Université Côte d'Azur
npj Syst Biol Appl (2026), PMID 42443213, [DOI](https://doi.org/10.1038/s41540-026-00782-4).

**The strongest technical match on the list, and the one to think hardest about**, because
they are the nearest neighbour and therefore the likeliest to have done it already.

They fit a mechanistic model to individual single-cell trajectories, **recovering cell-specific
protein abundances**. That is precisely the object our identifiability machinery audits, and we
have a measured result that applies directly to it:

- fitted-parameter error is **anisotropic**. A regressor is pulled hardest along the directions
  the data constrains, so its residual concentrates in the flat directions — the ones that
  barely move the answer. Measured: **isotropic error costs 10.1x more tail spread than fitted
  error of the same nominal magnitude.** So the usual "±X% parameter error → Y% outcome error"
  propagation systematically overstates the damage from a *fit*, and understates it from a
  *guess*.
- the singular spectrum of their fit would name **which recovered abundances their trajectories
  actually determine** and which are sloppy. In our own case that number was a condition number
  of 3.9e4, with the flattest direction being a specific two-parameter combination.

That is an audit of their existing method, not a replacement for it, and it is checkable in an
afternoon on data they already have.

### 3. Françoise Van Bambeke — UCLouvain
Microbiol Spectr 10:e0231321 (2022), PMID 35196815, [DOI](https://doi.org/10.1128/spectrum.02313-21).

**They measure the one thing our own tool says a kill curve cannot determine.** Our
identifiability report on a fitted time-kill curve returns a condition number of 1.9e7, and
names the flattest direction as the **dormant decay rate** — obvious in hindsight, since dormant
cells are exactly the ones a kill curve is not watching die. Van Bambeke's group measures
dormancy *depth* and resuscitation lag directly, single-cell, in intracellular *S. aureus*.

So this is not a service offer, it is a genuine complementarity: their measurement removes our
largest uncertainty, and our solver turns their lag distribution into a relapse probability
they cannot measure. Broad dormancy-depth heterogeneity is also precisely where averaging fails
worst, which makes their system the sharpest available test of the central question.

---

## PARTIAL — the offer is real but needs their data, or an extension we have not built

### 4. Peter K. Sorger and Sabrina L. Spencer
*Nature* 459:428-32 (2009), PMID 19363473, [DOI](https://doi.org/10.1038/nature08012).

They established that fractional killing is non-genetic and that inherited protein state
diverges over a measured memory timescale. Our Floquet result is that when a driver's period is
comparable to that relaxation time, the **tail** is wrong by up to 19x if the periodicity is
averaged away — while the mean is preserved exactly.

The offer: turn their measured memory timescale into a schedule-dependence prediction. The
honest gap: we would need their timescale and a dosing period, and chemotherapy periods are days
to weeks against a memory of roughly one cell cycle, so the ratio may sit outside the regime
where the effect is large. **That ratio decides whether there is anything here, and it should be
computed before the email is sent, not after.**

### 5. Johnjoe McFadden and Suzie Hingley-Wilson — Surrey
FEMS Microbiol Rev 46:fuab042 (2022), PMID 34355746, [DOI](https://doi.org/10.1093/femsre/fuab042).

Their "hunker" review frames heterogeneity qualitatively. Our gap detector turns one of their
claims into a decidable question: given data at six or more conditions, is a two-state model
structurally sufficient, or is a continuum required? It returns a verdict **with its detection
limit printed**, so a null result means something.

TB is also where scheduling matters most in the clinic — intermittent regimens are standard and
relapse is the endpoint. The honest gap: we have no TB parameters, so this is a method plus a
question, not a result.

### 6. Gerard D. Wright — McMaster
Genetics 214:1103-1120 (2020), PMID 32094149, [DOI](https://doi.org/10.1534/genetics.119.302851).

His adjuvant is scored on how *many* persisters rifampin eradicates. FIC index and checkerboard
assays are means. Eradication is a tail. The specific question: **can two combinations with the
same FIC differ in eradication probability** — and if so, is the standard score ranking the
wrong thing? We can compute it from his kill curves.

Caution worth taking seriously: this is the offer most likely to be already known to a
combination chemist. Check the FIC-limitations literature before sending or it reads as naive.

---

## WEAK — say so rather than stretching

### 7. Lori L. Burrows — McMaster
ACS Infect Dis 4:1041-1047 (2018), PMID 29771109, [DOI](https://doi.org/10.1021/acsinfecdis.8b00112).

Biofilm tolerance is **spatially structured** — oxygen and nutrient gradients, penetration
limits. Our machinery is well-mixed and has no spatial dimension at all. The schedule question
applies to dispersal and resuscitation timing if she has that data, but the dominant physics of
her system is the part we do not model. Do not lead with REM here.

### 8. Karen L. Maxwell and Alan R. Davidson — Toronto
Int J Antimicrob Agents 66:107613 (2025), PMID 40930191, [DOI](https://doi.org/10.1016/j.ijantimicag.2025.107613).

Emotionally the best fit on the list — a documented relapse at two weeks with the same isolate
— and **mechanistically the worst**, which is worth being upfront about. Phage *replicate*. The
drug amplifies itself, so the system is predator-prey, not a decaying driver, and our periodic
machinery does not cover that without a model-class extension we have not built or validated.
n = 1 as well.

Honest version of the approach: ask whether they would want the extension built, rather than
implying we already have it.

### 9. York University
Agreed with your own assessment. Zarean et al., Environ Pollut 398:128053 (2026), PMID 41932379,
[DOI](https://doi.org/10.1016/j.envpol.2026.128053) is environmental resistance-**gene**
surveillance on microplastics. There is no single-cell tolerance dynamics, no dosing, no
persister formation. REM has nothing for it. **Do not send.**

One thin thread if you want York specifically: Gerald F. Audette, Department of Chemistry, York
University Toronto — verified via Bragagnolo & Audette, Acta Crystallogr D 80:834-849 (2024),
PMID 39607821, [DOI](https://doi.org/10.1107/S205979832401132X), and Rodriguez & Audette, Struct
Dyn 13:024701 (2026), PMID 41858832, [DOI](https://doi.org/10.1063/4.0001201) on the F-plasmid
T4SS, "central to antibiotic resistance dissemination". Conjugative transfer at low donor
density is a rare stochastic event, so "probability at least one transfer establishes" is a real
question of our type. But it is a different question from the rest of this list and the data
type is structural, not dynamic. Weak, and only if you want the institution covered.

---

## Sending order

Roux/Chaves first, not Balaban — because they are the likeliest to say it has been done, and
that answer is worth more than four polite replies. Balaban second, with the selection-curvature
prediction, because she can refute the whole premise fastest. Van Bambeke third, because that
one is a collaboration rather than a pitch.
