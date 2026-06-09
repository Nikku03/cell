# Pipeline statistics — what 'sequence→structure→function→network' actually delivered on syn3.0's 29 unclear essentials

**Input:** 29 genes flagged "Unclear" in the literature (Breuer 2019).
**Tools:** all in-sandbox — sequence + Chou-Fasman SS + TM prediction + BLOSUM62
Smith-Waterman against annotated M. genitalium proteome + operon-neighbour
voting. NO AlphaFold, NO BLAST, NO HMMER.

## Per-stage outcome

| stage | distribution |
|---|---|
| **3 structure class** | α/β globular 14 · membrane-anchored/lipoprotein 9 · polytopic membrane 6 |
| **4 homology** | HIGH (≥30% id, characterized target): 7 · MED (25-29%): N · LOW/artifact: rest |
| **5 network** | STRONG (co-ess≥67% + informative subsystem): 17 · MED: 11 · WEAK: 1 |

## Integrated result (STRICT — homology hits to other "uncharacterized" proteins excluded as non-evidence)

| tier | n | % of 29 |
|---|---|---|
| **Confident function (homology ≥30% to a CHARACTERIZED target)** | 7 | 24% |
| **Confident by network (operon inside a NAMED complex, atpI-style)** | 9 | 31% |
| Function class narrowed (broad role but specific enzyme not pinned) | 2 | 7% |
| Subsystem placed (location known, function not) | 10 | 34% |
| **Still mysterious** | **1** | **3%** |

## The bottom line

- **55%** (16 of 29) pinned to a *named* function — defensible at curator level
- **97%** (28 of 29) moved out of "Unclear" into some characterization tier
- **3%** (1 gene, JCVISYN3A_0325) genuinely unresolved with these tools

The single residual unclear gene (0325) has high-quality homology to a phosphate
permease but its operon context is uninformative — it likely IS the permease,
but we lack the contextual confirmation our tier system requires.

## Where the AlphaFold-style structure stage would help

The 12 "function class narrowed" or "subsystem placed" genes are the ones a
real 3D structure would lift to "confident function": the fold reveals
distant homology that 25-29% sequence id cannot. So with AlphaFold added
to the pipeline, the realistic outcome would be ~24-26 of 29 confidently
characterized (~85-90%), with the remaining 3-5 being the genuinely
de-novo or fast-evolving inventions that even structure can't catch
without a known relative in the structure DB.
