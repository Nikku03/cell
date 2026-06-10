# Environment test via standard condition (rich medium), leak-free

Encoded the STANDARD condition (rich medium) as known biochemistry:
a rich medium provides amino acids, nucleobases, and B-vitamins, so the
biosynthesis genes for those are "rescuable" (less essential); core
machinery (translation/replication/energy/envelope) makes products the
medium can't supply (essential regardless). Feature derived ONLY from
product annotation -- no labels. family_frac stays per-fold leak-free.

## (1) The mechanism is real (univariate, 32,432 genes)

| env category | n | essential |
|---|---|---|
| core (not provided by medium) | 1,918 | **69.7%** |
| rescuable (AA/nt/vitamin biosynth) | 1,739 | **43.5%** |
| other | 28,775 | 23.1% |

Core genes are **1.6×** more essential than rescuable ones. The standard
medium really does rescue biosynthesis -> the environment signal exists
and is biochemically interpretable.

## (2) The surprise: in the CONDITIONAL zone, rescuable genes are MORE essential

| env category | n | essential (conditional zone, family_frac 0.2-0.8) |
|---|---|---|
| nucleotide biosynth | 134 | **80.6%** |
| amino-acid biosynth | 363 | **62.8%** |
| core | 327 | 54.7% |
| other | 7,249 | 44.8% |

Flipped! The biosynthesis genes that land in the conditional zone are
exactly the ones where medium-rescue FAILED in that organism (its niche
lacks the nutrient, or it lacks the importer). That is the
environment x organism interaction -- and a STATIC "rich medium provides
X" rule cannot tell which organisms have access to X.

## (3) Predictive lift (leak-free LOO-organism)

| | overall AUC | overall MCC | conditional AUC | **conditional MCC** |
|---|---|---|---|---|
| family_frac only | 0.7915 | 0.5468 | 0.7304 | 0.2777 |
| + standard-condition feature | **0.8082** | 0.5513 | 0.7338 | **0.3080** |
| lift | **+0.0166** | +0.0045 | +0.0034 | **+0.0303** |

## Verdict (both halves matter)

1. **The standard-condition feature is the first GENUINELY ADDITIVE feature
   in the whole project: +0.017 AUC overall and +0.030 MCC in the conditional
   zone.** Unlike the six prior feature classes (all absorbed by family_frac),
   this one adds signal because it is BIOCHEMISTRY, not phylogeny -- orthogonal
   to conservation.

2. **But it does NOT crack the conditional zone.** A static biochemistry prior
   helps the easy cases (core=essential, biosynthesis=often-not) but can't
   resolve the flips, because the flips depend on the SPECIFIC organism's
   access to the nutrient -- which needs MEASURED per-organism condition data
   (Fitness Browser), not a one-size rule.

The session's central thesis, re-confirmed with new evidence: environment IS
the driver of the conditional zone; encoding the *standard* condition gives a
real, usable lift; fully closing the gap needs the *measured* condition matrix.
