# Wheel 3 as an assembly engine: laying out the minimal cell, in layers

Your blueprint, built on two pilots: **Putida** (data-rich) and **mtub**
(independent DeJesus truth). Each layer labelled by provenance: genome-derivable
vs needs-measurement.

## What the genome alone lays out

| layer | Putida | mtub | provenance |
|---|---|---|---|
| **L1 scaffold** (universal core) | 262 genes, 176 essential (P 0.67) | 184 genes, 107 essential (P 0.58) | product annotation — generalizes |
| **L2 metabolic skeleton** (FBA) | 194 genes, 124 essential (P 0.64) | 203 genes, 137 essential (P 0.67) | curated model (CarveMe for novel) |
| **L1 ∪ L2 structural core** | 444 genes → **31.4%** of essentials | 380 genes → **31.3%** of essentials | genome-only |
| **L3 regulators** | 273 TFs, effector class annotated | 65 TFs, effector class annotated | family signature — generalizes |
| **L4 edges / motifs** | 546 local edges, **0 FFL** | 130 local edges, **0 FFL** | adjacency only — fails |

Scaffold breakdown (Putida): ribosome 69(59 ess), envelope 51(18), replication
37(19), transcription 36(27), tRNA 26(20), translation 19(14), division 15(10),
energy 9(9). The chassis is recognizable and consistent across both organisms.

## The three honest findings

**1. The genome-only core is real, generalizable, and stops at ~31%.**
Both organisms — including mtub with *independent* truth — land at **31% of
essentials covered** by scaffold + metabolic skeleton. This is the part you can
lay out for any bacterium from its genome: the universal chassis plus the
metabolic skeleton. It is robust (same number, two very different organisms) but
**capped** — the structural core is one-third of the essential cell.

**2. Regulators can be named and chemically typed, but NOT wired.**
We identify every TF by family (273 in Putida, 65 in mtub) and assign each an
**effector class** (LysR→aromatic intermediates, TetR→lipophilic drugs,
FNR/CRP→O2/cAMP, etc.) — that part of your plan works from the genome. But the
**network motifs do not materialize: 0 feed-forward loops in either organism.**
Reason: FFLs require TF→TF→gene chains, and genome-local adjacency almost never
captures them (the global edges where motifs live are exactly the ones the
sequence ceiling — Wunderlich-Mirny — makes unrecoverable). So the regulatory
*layer exists as a parts list with effector chemistry, but not as a wired,
motif-bearing circuit.* That step needs the measured network.

**3. ~69% of essentials need measurement — consistently.**
68.6% (Putida) / 68.7% (mtub) of essential genes fall outside the genome-only
core: lineage-specific essentials, conditional essentials, non-metabolic
machinery the scaffold tagger misses. This is the W4 / measurement frontier, and
its size is stable across organisms.

## The bottom line on "completing the cell like this"

The assembly works and produces a real, layered minimal cell — but it completes
to a **predictable ceiling, not to a whole cell:**

- ✅ **chassis + metabolic skeleton** (~31% of essentials) — laid out from genome, generalizes to any bacterium
- ✅ **regulator parts list + effector chemistry** — identified from genome
- ❌ **regulatory wiring & motifs** — cannot be built de novo (0 FFL); needs the measured network
- ❌ **the other ~69% of essentials** — conditional/lineage-specific; needs fitness measurement (W4)

So the cell you can assemble from a genome alone is the **universal chassis with
its metabolic plumbing and a labelled (but unwired) regulatory parts bin.** The
control circuitry and the conditional two-thirds require measurement — the same
boundary every path in this project has reached, now shown constructively by
trying to build the cell itself.
