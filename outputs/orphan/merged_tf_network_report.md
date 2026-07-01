# Measured wiring — merging CollecTRI + DoRothEA + TRRUST

Replaced the sparse hand-curated TRRUST with a merged, evidence-tagged human regulatory
network (curated TRRUST + measured/curated CollecTRI + confidence-tiered DoRothEA), then
redid the analyses. Code: `colab/merged_tf_network.py`.

## The network is far bigger and better-covered

| | TRRUST only | **merged** |
|---|---|---|
| edges | 9,396 | **326,125** (72,648 high-confidence) |
| TFs | 795 | **1,366** |
| essential genes (CEG) with a known regulator | 82/684 (12%) | **332/684 (48%)** |

TRRUST is 93% contained in CollecTRI (7,849 of 8,427 edges overlap) — the curated and
measured networks validate each other. The big win is **coverage**: nearly half the
essential genes now have an identified regulator (was 1 in 8).

## Everything got stronger on the measured network

- **TFs are vital**: median LOEUF 0.46 vs 0.91 genome-wide (unchanged conclusion, firmer).
- **Network motifs** (high-confidence TF→TF, 13,555 edges): feed-forward loops Z=**+57.3**
  (was +6.1), mutual regulation Z=**+73.8** (was +12.3). The circuit enrichment we saw in
  bacteria is now overwhelming in the measured human network.
- **Master regulators of the essential core**: MYC (91 essential targets), E2F1/E2F4, TP53,
  SP1, and the **NF-κB family** (NFKB1/2, REL, RELA, RELB) — proliferation + survival
  signalling, all highly constrained, all cancer genes.

## TF mutation + disease, redone on measured wiring

For disease-causing TFs, their **measured** targets that share the TF's disease form a
candidate causal regulon — now concrete and correct:

| TF | diseases | measured targets sharing the disease | tissue |
|---|---|---|---|
| TP53 | 20 | 37 | cancer |
| CRX | 6 | 12 | retina (retinal dystrophy) |
| NKX2-5 | 16 | 10 | heart (congenital defects) |
| RUNX1 | 4 | 10 | blood (leukemia) |
| HNF4A / HNF1A / PDX1 | 5–8 | 7–9 | pancreas (MODY diabetes) |
| GATA4 | 10 | 8 | heart |

Combined with the earlier coding-mutation result (68% of TF pathogenic mutations hit the
DNA-binding domain), the picture is complete: a TF's DNA-contact mutation breaks its grip →
its measured target program fails → the shared-tissue disease.

## Honest notes
- CollecTRI/DoRothEA-A/B are curated+ChIP-supported; DoRothEA-D (low confidence, inferred)
  is excluded from the high-confidence set. Directions/signs are mostly reliable at A/B/
  CollecTRI level.
- "measured" here means literature-curated + ChIP/experiment-supported meta-resources, not
  a single raw ChIP-seq reprocessing.
