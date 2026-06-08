# Metabolic-context (isozyme backup) as an essentiality signal

Tested whether "is this gene the sole catalyst of its function, or does
the genome encode an isozyme backup?" predicts essentiality -- the
genome-level proxy for the environment/context factor.

Functional identity = normalized GFF product description. Isozyme backup
= another gene in the same genome with the same normalized product
(captures NON-homologous backups that paralog-count misses).

## Result: real signal, correct direction (unlike paralog count)

32,432 genes / 10 organisms; 28,089 with a specific (non-generic) function.

| group | n | % essential |
|---|---|---|
| sole-function (no isozyme) | 16,030 | 33.9% |
| has-isozyme backup | 12,059 | 23.1% |

Cohen's d = +0.239, AUC = 0.542 (more functional copies -> less essential).

Within the AMBIGUOUS families (the hard 47% where essentiality flips):
| group | n | % essential |
|---|---|---|
| sole-function | 5,585 | 43.9% |
| has-isozyme | 5,396 | 37.2% |

## Interpretation

- This is the FIRST metabolic-context feature in the right direction.
  Paralog count went the WRONG way (confounded by family antiquity);
  functional redundancy (isozyme presence) goes the RIGHT way.
- But it is WEAK (AUC 0.54) and does NOT close the ambiguous-zone gap:
  even sole-function ambiguous genes are only 44% essential, so the
  bigger driver of the flips (growth medium / condition) is still
  missing -- and that needs the condition-resolved fitness data we
  don't have.
