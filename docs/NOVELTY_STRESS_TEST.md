# Novelty stress-test: are the cell model's predictions genuinely new *and* checkable?

**Question.** Our synthetic-lethal / co-dependency predictions come from DepMap knockout
correlation (measured) filtered to buffered pairs (both genes individually dispensable, r>0.4).
Do the top calls recover *known* biology (validation) or surface *novel, checkable* biology (worth)?

**Method.** For candidate pairs we queried PubMed (NCBI E-utilities) for co-occurrence
(`"A"[tiab] AND "B"[tiab]`) and co-occurrence in an SL / co-dependency context. Known pairs
were included as positive controls to confirm the check works.

## Headline finding — the method recovers real, drug-relevant complexes from first principles

**KIDINS220 × XPR1** (r=0.843). Flagged as tightly co-dependent from DepMap correlation alone,
with *no annotation* linking them (different pathway labels in our data). Literature:

- 2022 — *"Phosphate dysregulation via the XPR1-KIDINS220 protein complex is a therapeutic
  vulnerability in ovarian cancer."*
- 2025 — *"KIDINS220 and InsP8 safeguard the stepwise regulation of phosphate exporter XPR1."*

A structurally-confirmed complex and a validated druggable cancer vulnerability, rediscovered
from knockout correlation. The scan also re-derived other real complexes the annotation had
split apart — **EMC1/MMGT1** (ER membrane complex), **ARIH2/RNF7** (cullin-RING ligase).

## Positive controls (the check works)

| pair | co-occ | SL-context | note |
|---|---|---|---|
| PAGR1 / PAXIP1 | 5 | 1 | PTIP complex — correctly surfaced its co-dependency paper |
| MDM4 / PPM1D | 7 | 0 | both p53 negative regulators; co-mentioned via the p53 axis |
| BRAF / MAPK1 | 50 | 0 | same MAPK cascade |

## Genuinely novel + checkable candidates

| pair | r | co-occ | why interesting |
|---|---|---|---|
| ATIC × DHODH | 0.671 | 1 | purine × pyrimidine de-novo synthesis co-dependency; DHODH is a validated drug target |
| MAP4K4 × TAOK1 | 0.641 | 0 | two STE20-family kinases in Hippo/stress signaling; plausible redundancy, never co-studied |
| COX14/COX10 × FASTKD5 | ~0.65 | 0 | mito mRNA processing gating OXPHOS complex assembly |

## Honest limits

1. **~50% of top pairs are same-complex** — validation of the method, not novelty.
2. **Zero co-occurrence ≠ novel-and-correct.** The same bucket held genuine unknowns *and*
   real complexes the annotation split (EMC, CRL). Measured co-essentiality is real signal, but
   separating "novel real" from "novel artifact" needs **wet-lab or held-out validation** — not done here.
3. **KIDINS220-XPR1 is retrospective.** It proves the method works; it is not a finding we own.

## Implication for commercial worth

The method demonstrably produces the *kind* of prediction that has value (KIDINS220-XPR1 became a
therapeutic-vulnerability publication). Converting capability into an asset requires finding the
present-day equivalent — strong co-dependency, no prior literature link, plausible mechanism — and
**validating it**. The candidates above are the shortlist to start from.
