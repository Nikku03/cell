# The reaction-chain investigation — can a mechanistic reaction network beat the abstract graph?

A multi-turn hypothesis, tested to exhaustion: **treat each PPI as a reaction with a product, chain the products
(A+B → product → next reaction), bring in the pathway network and its interconnections — and use that to predict what a
knockout does.** The intuition is sound and it's how real mechanistic modeling works. Here is what four measured tests found,
each sharper than the last, and the honest conclusion.

## The four tests

| # | what was tested | data | result |
|---|---|---|---|
| 1 | reaction **co-membership** (undirected: proteins in the same step) enrich for movers | Reactome 15k reactions | apparent 2.03× vs PPI 1.28× — **a size artifact**; size-controlled Wilcoxon **p=0.32** |
| 2 | same, framed as typed vs abstract | — | no advantage once size-matched; and it's the **wrong readout** (reaction predicts flux, not transcription) |
| 3 | **directed product-cascade** reach → essentiality | causal+signaling | cascade reach **+0.13** vs abstract degree **+0.28** — loses |
| 4 | **complete** directed reaction+pathway network → essentiality **and** transcription | Pathway Commons Reactome (119,233 directed edges incl. 52,989 catalysis-precedes) | essentiality: chain out-degree **−0.05** within-network vs abstract degree **+0.31**; transcription: no significant advantage (p=0.50). **Decisive.** |

## The conclusion: it's not graph abstraction, it's the missing bypass model

Even the **complete** directed reaction+pathway network (the strongest version of the idea) does **not** out-predict the plain
abstract graph for knockout effects. The reason is the same every time, and test 3→4 named it precisely: **bypasses /
redundancy.**

A protein can feed a huge downstream product cascade and still be non-essential — *if the cell has an alternative route to
those products.* Raw reaction-chain reachability counts the cascade but not the bypass, so it over-counts impact. Essentiality
isn't "big downstream"; it's "big downstream **with no alternative route**." Plain degree beats the reaction cascade because it
happens to proxy un-bypassable centrality better than cascade-size does.

**The bypass model is only exactly solvable where mass-balance stoichiometry holds — metabolism.** There, "can an alternative
route carry the flux when I block one?" becomes a linear program (FBA). That is why:

- **The reaction-chain idea IS built and validated for metabolism** — `ecFlux` does mutation → flux on the 2,549 local
  metabolic reactions, and it works, because metabolic reactions have the stoichiometric constraints that make bypasses
  computable.
- **The bypass-aware essentiality predictor already exists and works** — essentiality = network centrality **+ paralog
  buffering**, AUC 0.86. The paralog-buffering term *is* a bypass model. The naive reaction cascade lacks it and underperforms.
- **Everywhere else** (signaling, complex assembly), reactions have no mass-balance constraint, so whether a downstream product
  is actually lost when you block one route is not computable from topology — it's the unmeasured coefficient again.

## What is real and what isn't

- **Real:** the reaction chain is now a queryable asset — `outputs/orphan/reaction_network.json` holds 45,635
  `catalysis-precedes` chain edges (A's reaction output feeds B's) + 82,958 typed directed reaction edges, gene-indexed from
  Pathway Commons Reactome. You can trace reaction chains through it. The metabolic reaction network (`generxn`, `ecFlux`) is
  the validated quantitative version. Both are genuine.
- **Not real:** a claim that a mechanistic reaction/pathway network predicts arbitrary knockout effects better than the
  abstract graph. Tested four ways; it does not. The limit was never graph abstraction — it is the bypass/redundancy model,
  which is tractable (FBA) only under mass-balance (metabolism) and unmeasured elsewhere.

## The honest one-line answer

Making the graph a complete mechanistic reaction+pathway network **does not** break the far-field wall, because reaction
topology without a bypass/mass-balance model over-counts impact — and that model is exactly solvable only in metabolism, where
the idea is already built and works (FBA/ecFlux). Beyond metabolism, whether a blocked product is actually lost is a
measurement, not a graph property.
