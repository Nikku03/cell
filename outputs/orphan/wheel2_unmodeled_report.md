# Gap-directed Wheel 2 on unmodeled organisms — verified verdict

**Question:** can the gap-directed Wheel 2 (metabolic-hole identification) extend
to the 35 organisms with no metabolic model, via orthogroup transfer?

**Method:** build a catalog of 364 "hole-OGs" — orthogroups that catalyze an
FBA-essential reaction in ≥1 of 4 BiGG models (iJN1463/iML1515/iEK1008/iYL1228).
An unmodeled gene whose OG is a hole-OG is flagged (ortholog of an essential-
reaction enzyme), inheriting the hole's dimensions (EC/cofactor/direction).

## Verdict (adversarially verified — 3 independent lenses, all agree: SOUND)

**The hole-OG flag does NOT raise the P≥0.90 essential-coverage frontier on
unmodeled organisms. It is redundant with cross-clade conservation (Wheel 2).**

| metric | value | status |
|---|---|---|
| hole-OG precision alone | 0.578 (2.1× pooled / 1.8× mean vs base) | genuine, label-independent ✓ |
| catches | ~26% of essentials | ✓ |
| essCov@P0.90: W2 → W2 + fixed 0.10·hole | 0.459 → 0.461 = **+0.15pp** | ≈ 0 |
| …same, **excluding 1 outlier** (Cup4G11 +16.5pp) | **−0.33pp** | slightly negative |
| honest ceiling (leave-one-org-out CV) | **+0.72pp** | below +2pp materiality bar |
| redundant part (hole ∧ w2≥0.5) precision | 0.823 | the conserved core, already caught |
| orthogonal part (hole ∧ w2<0.5) precision | 0.243 | at/below base rate — uninformative |

## What the verification caught

1. **Leakage audit → clean.** Hole catalog is pure FBA (no labels: 40.8% of
   flagged genes are non-essential; flag varies for 0/14,925 OGs). W2 holds out
   the scored org's entire clade (materially active: 1,497 OGs shift) and is
   per-OG not per-gene.
2. **The +0.15pp is outlier-driven** — carried by one organism; remove it and
   the mean is −0.33pp. Report as "≈0 / slightly negative", never as a positive.
3. **The "+2.0–2.7pp win" is a truth-leak artifact** — it required in-sample
   per-org bonus selection *and* a 5-org subset chosen with the answer key. No
   label-free feature recovers those 5 orgs (corr ≈ 0). Not deployable.

## What survives as real value

The flag is a **~400-gene/organism, 1.8×-enriched, mechanistically annotated
shortlist** (each = ortholog of a known essential-reaction enzyme, with
EC/cofactor/direction dimensions). It is a **hypothesis-generation / triage
tool**, not a coverage-raising predictor.

## Conclusion

Third independent confirmation (after `metabolic_transfer` and the FBA-free
proxy) that for unmodeled organisms, OG-based metabolic signal collapses onto
the conserved core conservation already measures. The ~0.51 essCov@P0.90 ceiling
is a **data** limit; only new condition-specific data (Wheel 4 / feba.db) can
break it.
