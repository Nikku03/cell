# Activator -> target matching via co-fitness, by protein family

An activator and the operon it controls are co-essential under the relevant
condition -> correlated fitness. Co-fitness (feba) is a data-driven functional
link that may recover activator->target edges where sequence/motif fails.

Keio, RegulonDB activators with >=5 in-Keio targets, top-76 cofit partners.

## Overall
recall@cofit 0.109 | precision@K 0.071 | enrichment 33.4x over random.
Co-fitness finds a strongly-enriched FRACTION (~11%) of each regulon.

## By protein family (the answer)
| family | n | mean recall | enrichment |
|---|---|---|---|
| DeoR | 1 | 0.71 | 423x |
| GntR | 3 | 0.23 | 111x |
| LysR | 16 | 0.17 | 51x |
| IclR | 1 | 0.17 | (0 prec@K) |
| Lrp/AsnC | 1 | 0.09 | 3.8x |
| Crp/FNR | 1 | 0.07 | 1.5x |
| Sigma | 5 | 0.06 | 2.8x |
| OmpR (two-component) | 10 | 0.05 | 1.1x |
| AraC, MarR, MerR, LacI, Rrf2 | - | ~0 | ~0 |

## The pattern (biologically coherent)
Co-fitness matches the METABOLIC activator families (LysR/GntR/DeoR/IclR --
local regulators of catabolic operons, co-essential with their targets) at
50-423x enrichment, and FAILS on signaling/global families (OmpR two-component,
sigma, CRP, Fur, MarR) whose stress-response targets don't track the regulator's
fitness.

Same local-vs-global divide every method hits: local/metabolic regulatory edges
recoverable, global/signaling need direct measurement (ChIP).

Complementary to adjacency (positional, 0.51 precision on local pairs):
co-fitness is functional (33x enrichment on metabolic families). Combining the
two should raise recall.
