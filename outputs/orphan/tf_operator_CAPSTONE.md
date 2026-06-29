# TF ↔ operator: the complete multidimensional analysis (capstone)

Every dimension we could test, on E. coli / RegulonDB, leak-controlled. The
question: can we predict where a TF binds the genome from sequence + family +
shape + protein + conservation?

## Every dimension, measured

| dimension | method | result (AUC) | verdict |
|---|---|---|---|
| Sequence (direct readout) | learned PWM, both strands, held-out targets | 0.54–0.56 | the Wunderlich-Mirny wall |
| DNA shape (indirect readout) | dinucleotide twist/roll/MGW + electrostatics | +0.00 over PWM | shape = f(sequence), no new bits |
| Architecture (palindrome/dyad) | symmetry of hit window | +0.006 | already inside a good PWM |
| Position / helical phase | distance-to-gene + 10.5bp phase | +0.00 | doesn't discriminate within the promoter window |
| 4-D integrated (all of the above) | logistic fusion, leave-TF-out | 0.546 | ties the plain motif |
| Family (protein class) | family-mate motif transfer | 0.544 vs 0.520 between | small but real |
| Family — best case | AraC/XylS (MarA/SoxS/Rob) | 0.595 | family ≈ operator (marbox) |
| Family — worst case | LysR (OxyR/CysB) | 0.502 (chance) | shares architecture, not sequence |
| Protein recognition code | DBD residue-composition NN transfer | 0.549 ≈ family-mate 0.552 | adds nothing over family |
| corr(DBD-sim, operator-sim) | — | −0.149 | similar domains ≠ similar operators |
| Phylogenetic footprinting | near relatives (4 Enterobacterales) | 0.517 (far-control 0.499) | real but comparator-starved |
| Co-fitness → discover operator | motif from co-fit partners | 0.493 (chance) | cofit = pathway partners, not operator-sharers |

## What it all says

1. **Every DNA-side dimension collapses to ~10 bits.** PWM, shape, architecture
   are transformations of the same operator sequence; stacking them cannot exceed
   the information in that sequence. Measured: 4-D integrated = 0.546 = plain motif.

2. **Position doesn't help once you're already in the promoter window** — its only
   value was restricting the search there, which both targets and non-targets share.

3. **The protein side adds nothing at the level we can extract.** Bulk DBD
   composition does not predict the operator (corr −0.149); DBD-nearest-neighbor
   transfer (0.549) ties a random family-mate (0.552). Specificity is a handful of
   recognition-helix residues; composition washes them out, and no general
   residue→base code exists.

4. **Family is the only sequence-derivable lever that beats random — and only for
   some families.** AraC/XylS and σ54 conserve specificity (family motif ≈
   operator); LysR and most others conserve only architecture.

5. **The signals that carry genuinely independent information are not in this list
   as winners:** measured sites (ChIP/DAP-seq), and many-genome footprinting
   (15–30 relatives, not 4). Those add bits from outside the single sequence.

## The honest bottom line
TF→operator from sequence/structure/family is at the information floor. The
binding signal is intrinsically degenerate (~10 bits) — not poorly represented.
No multidimensional re-encoding changes that; we proved it dimension by dimension.

**The usable pieces for building the regulatory layer:**
- AraC/XylS + σ54 families → predict operator from family motif (the one real win).
- Everything else → take the regulatory EDGE from co-fitness + adjacency + family
  effector (which works), and treat the exact binding SITE as a measured input
  (RegulonDB/ChIP) or a many-genome footprint, not a sequence prediction.

Representation was never the bottleneck. Information is.
