# Single-organism test: conservation vs conservation+environment

Test organism: **beril_Methanococcus_JJ** -- the most distant in the
dataset (archaeon; mean family breadth 13.5 vs 24-30 for the bacteria).
Leave-this-organism-out: train on 47 other organisms (177,632 genes),
predict all 1,605 Methanococcus genes (366 essential, 22.8%).

## Head-to-head

| metric | A: conservation | B: + environment | delta |
|---|---|---|---|
| MCC | 0.5057 | 0.4976 | -0.008 |
| precision | 0.530 | 0.512 | -0.018 |
| recall | 0.768 | 0.787 | +0.019 |
| accuracy | 0.792 | 0.781 | -0.011 |

Environment (isozyme-backup) feature shifts the precision/recall
tradeoff toward recall but does not improve net MCC.

## BUT: rogue-essential recovery (what it was designed for)

Rogue essentials = truly essential, but family_frac < 0.3 (invisible
to the conservation prior). 86 of them in this archaeon (23% of its
essential genes -- high, because it's the most distant org).

| | recovered |
|---|---|
| A (conservation only) | 4 / 86 (5%) |
| B (+ environment) | 9 / 86 (10%) |

The environment feature DOUBLED rogue-essential recovery (+5 genes) --
it works on exactly the genes it targets. But it's a precision/recall
trade, not a free win: the extra true positives come with extra false
positives elsewhere, so net MCC is flat-to-slightly-negative.

## Conclusion

- Even on the MOST DISTANT organism, conservation alone reaches
  MCC 0.51 / accuracy 79% -- because the universal core (ribosomes,
  replication) is shared even across domains of life.
- Environment helps the rogue-essentials specifically (5%->10% recovery)
  but trades precision for recall; aggregate metric unchanged.
- No combination reaches 95%. 23% of this organism's essentials are
  rogue (organism/lineage-specific), and even with environment we
  recover only 10% of them -- the rest need the experimental-condition
  data we don't have.
