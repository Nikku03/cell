# Can we use TF binding info to look for operator sequences?

Two modes, tested on E. coli against RegulonDB.

## Mode A — known motif → scan for operators
If we already have a TF's motif (PWM), scanning promoters for matches *is*
looking for operators. It works mechanically but caps at the Wunderlich-Mirny
wall: supervised PWM (learned from real RegulonDB targets) recovers held-out
targets at **mean AUC 0.543**. Useful only when stacked with promoter-window +
family-architecture + ortholog-conservation + sigma constraints.

## Mode B — discover the operator from co-regulated genes (no operator DB)
The attractive idea: for a TF with no known motif, take the genes it
co-regulates (from **co-fitness**), find the shared motif in their promoters =
the operator, then scan with it. Tested: discover motif from each TF's top
co-fit partners (blind to RegulonDB), test whether it recovers held-out
RegulonDB targets.

| signal | mean AUC |
|---|---|
| **co-fit-discovered operator** | **0.493 (chance)** |
| supervised operator (RegulonDB targets) | 0.543 (the wall) |
| co-fit-operator reaches AUC≥0.60 | only **3 / 26 TFs** |

**The bootstrap fails in general** — and the reason is precise and important:

> **Co-fitness partners are not operator-sharers.** Co-fitness captures
> *functional / pathway partnership* (knock out the TF, lose the pathway, same
> conditions hurt). The genes in a TF's pathway are mostly **not** the genes that
> carry its operator. So their promoters share no common motif, and de novo
> discovery returns noise (AUC 0.493).

The 3 that *did* work are the tell: **gadX (0.70), rcsB (0.61), csgD (0.61)** —
acid-resistance / biofilm regulators whose regulons are tight, co-fit strongly,
*and* share a real operator. When the co-regulated set genuinely shares an
operator, the bootstrap recovers it. That's the minority.

## The conclusion

Finding operators needs a set of genes that actually **share the operator** — a
true regulon. We have no way to obtain that set without either already knowing
the regulation (RegulonDB) or measuring it (ChIP-seq). The bootstrap breaks at
**step 1 (getting an operator-sharing set)**, not step 2 (motif discovery is
fine when the set is clean).

So, answering directly: **yes for known-motif TFs (scan, capped at the wall);
mostly no for unknown TFs**, because the functional data we have (co-fitness)
links pathway partners, not operator-sharers. The same boundary as everything
else — we can get the regulatory **edge** (TF→gene) from co-fitness, but the
**operator** (the sequence/site) needs measurement or close-ortholog transfer,
not functional inference.
