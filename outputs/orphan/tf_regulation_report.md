# TF mutations, network motifs, and how one genome makes many cell types

Three questions on the human regulatory (TF) layer. Code: `colab/tf_mutations.py`,
`colab/tf_motifs_differentiation.py`. Figure: `tf_motifs_differentiation.png`.

## 1) What happens when TF genes are mutated → the DNA-contact breaks

A transcription factor's "active site" is its **DNA-binding domain**. Pathogenic mutations
concentrate there: **620/917 (68%) of TF disease mutations land in the DNA-binding domain**,
even though that domain is a *minority* of each protein.

| TF | pathogenic in DNA-binding | (domain size) | disease |
|---|---|---|---|
| TP53 | 94% | 49% | cancer (R175/R248/R273 hotspots) |
| FOXP3 | 91% | 36% | IPEX (autoimmunity) |
| POU3F4 | 89% | 37% | X-linked deafness |
| HNF1A | 81% | 28% | MODY3 diabetes |
| SOX9 | 79% | 14% | campomelic dysplasia |
| TBX5 | 78% | 35% | Holt-Oram (heart/limb) |
| PAX6 | 77% | 44% | aniridia |
| NKX2-5 | 74% | 19% | congenital heart defects |
*(MECP2 is the exception at 2% — its methyl-CpG-binding domain isn't captured by the
DNA-binding keyword; an annotation gap, not biology.)*

Plus, TFs are **haploinsufficient** (median LOEUF 0.42 vs 0.91 genome-wide) — one broken
copy is enough, so TF mutations cause **dominant developmental disorders and cancer**.
Break the DNA contact of a master regulator and the whole downstream program fails.

## 2) The TF network has the same motifs as bacteria

TRRUST TF→TF subnetwork (795 TFs, 2,056 regulator→regulator edges) vs a degree-preserving
random null (80 randomizations):

| motif | observed | random | enrichment Z |
|---|---|---|---|
| **feed-forward loops** | 1,254 | 881 ± 61 | **+6.1** |
| **mutual regulation** | 103 | 38 ± 5 | **+12.3** |
| autoregulation | 7 pos / 4 neg | — | (TRRUST sparse) |

Same result as our bacterial RegulonDB analysis (Milo/Alon motif theory): feed-forward
loops (signal filtering / delay) and mutual regulation (switches) are strongly
over-represented — evolution reuses the same regulatory circuits in bacteria and humans.
(Autoregulation is under-counted because TRRUST is a sparse curated set; a dense network
like GTRD/ENCODE would recover more.)

## 3) Same genome, different cell types — the answer

Every cell in a body has the **identical genome**. What differs is **which TFs are active**,
and the network is built so that a fixed wiring supports **many stable states (attractors)** —
one per cell type. Two circuit motifs make this work:

- **Mutual-repression toggle switches** (found 103 mutual-regulation pairs; explicit
  mutual-repression toggles include **BCL6↔IRF4** — the *real* switch that decides
  germinal-center B-cell vs plasma-cell fate). A toggle is **bistable**: two stable states,
  each a different cell fate.
- **Feed-forward loops + autoregulation** lock a chosen state in (memory), so a cell stays
  differentiated.

**Demonstration (simulation):** one toggle network (two TFs mutually repressing), started
from 6 different initial conditions, settles into **2 distinct stable fates** — the *same
equations / same genome* reaching different attractors purely by which TF started higher.
Scale this across many switches → a combinatorial number of cell types.

> **Differentiation = the same regulatory network falling into different stable attractors.**
> The genome is the fixed wiring; the cell type is which stable state it settled into.
> Developmental signals nudge a cell toward one attractor; epigenetic marks (DNA
> methylation, histone modifications) then lock it there. No DNA change required.

The figure (`tf_motifs_differentiation.png`) shows the motif enrichment (left) and the
toggle trajectories converging to two fates (right).

## Honest caveats

1. TRRUST is a sparse curated network — motif *directions* are right but counts undercount
   (esp. autoregulation); a dense TF-binding network would strengthen it.
2. The toggle simulation is a canonical 2-node model illustrating the principle, not a
   fitted model of a specific lineage.
3. TF-mutation DNA-binding fraction depends on UniProt domain annotation (MECP2 gap).

## Ties to the rest

TFs are the vital core's control layer: dosage-sensitive (constraint), their mutations hit
the DNA-contact machinery (disease), their circuits are motif-structured (like bacteria),
and their multistability is what turns one genome into every cell type.
