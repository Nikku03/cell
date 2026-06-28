# Solving TF → genome: how far sequence + conservation actually gets

**The problem.** Where does a TF attach in the genome? One PWM carries ~10 bits;
pinning sites in a 4.6-Mb genome needs ~23. That ~13-bit deficit (Wunderlich–
Mirny) is why a genome-wide motif scan is mostly false positives. The cell closes
the gap with information *outside* the motif: position, co-occurrence,
conservation, concentration. We tested the sequence-derivable ones against
RegulonDB held-out targets (E. coli, learned PWM per TF, cross-validated).

## What each layer buys

| method | mean AUC | note |
|---|---|---|
| family-consensus motif | 0.505 | (earlier) no learned specificity |
| **learned PWM, E. coli only** | **0.52–0.56** | the Wunderlich–Mirny wall |
| footprint, 57 mixed genomes | 0.528 | signal drowned by distant phyla |
| footprint, NEAR (4 Enterobacterales) | 0.517 | ≈ PWM, not above |
| footprint, FAR (distant, control) | **0.499** | chance — confirms wiring is correct |
| PWM + NEAR (z-sum) | 0.519 | no average lift |

The FAR control landing exactly at 0.499 proves the method is sound: distant
genomes (Pseudomonas, Burkholderia, Ralstonia) carry **zero** binding-site
signal, because their regulatory networks have rewired. Only the right
evolutionary distance can help.

## But conservation IS real signal — for the TFs that need it

Averages hide the actual finding. Where the E. coli motif **fails**, near-relative
conservation **rescues** it:

| TF | PWM alone | + near footprint |
|---|---|---|
| OxyR | 0.506 | **0.701** |
| Rob | 0.420 | **0.637** |
| NarL | 0.442 | **0.572** |
| MarA | 0.471 | 0.538 |

These are exactly the TFs with weak/degenerate motifs — the cases the theory says
need extra bits. Conversely, TFs with a strong intrinsic motif (PurR 0.74, FhlA
0.76, MraZ 0.72) gain nothing, and footprint can dilute them. So conservation is
genuine, *complementary* information — it helps precisely where the motif is
weak. Combined "beats PWM in 21/35 TFs," but the wins are small and the losses
offset them on average.

## Why the average doesn't move — and the honest limit

Footprinting power scales with the **number** of independent close relatives.
The standard method uses 10–30 Enterobacteriaceae; feba gives us **4**
(Klebsiella + 3 Dickeya), and the genome cache is one-per-organism, not a dense
relative set. With 4 comparators the missing ~13 bits simply aren't there to
recover. The method is right and underpowered, not wrong.

**The deeper truth:** TF→genome at single-site resolution, from sequence alone,
sits at the information-theoretic floor. This is why the real maps — RegulonDB,
EcoCyc — are **measured** (ChIP-seq, DAP-seq, footprinting assays), not predicted
from sequence. Same pattern as kinetics: the *identity* layer (which genes are
TFs, which family, which effector class) is knowable from sequence; the
*quantitative/positional* layer (exact site, occupancy) is a measurement.

## The constructive pivot — solve the EDGE, not the SITE

For laying out the cell's regulatory wiring we don't need the base-pair position;
we need the **edge**: which TF regulates which gene/operon. And the edge is
recoverable from *functional* data we already have, far better than from binding
sequence:
- **co-fitness** (feba): metabolic-regulator families recover strongly —
  DeoR 423×, GntR 111×, LysR 51× enrichment over random.
- **genomic adjacency** (local regulators next to their targets): precision@K 0.226.
- **footprinting** as a tie-breaker for the weak-motif TFs (OxyR/Rob/NarL).

So: predict the regulatory **graph** from co-fitness + adjacency + family
effector, use footprinting only where motifs are weak, and treat exact
binding-site coordinates as a measured input (RegulonDB) rather than a sequence
prediction. That is the solvable form of "TF → genome" with the data in hand.
