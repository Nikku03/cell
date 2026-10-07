# Single-cell signalling dynamics — Roux & Chaves

**Their work.** According to PubMed, Fiandaca G, Péré M, Bonhomme S, Chaves M, Roux J, *npj Systems
Biology and Applications* (2026), PMID 42443213,
[DOI](https://doi.org/10.1038/s41540-026-00782-4). No PMC record; full text was **not retrievable**
through the channel used here, so no number from it is quoted below.

## The honest position: this is the nearest neighbour

Of every group on this list, this one is methodologically closest to what REM does. They already
model single-cell heterogeneity in cell-fate decisions with dynamical systems. **The novelty margin
is thin and saying otherwise would be the fastest way to lose the room.**

## What is genuinely different, stated narrowly

REM's distinguishing property is not stochastic modelling — they do that. It is that it solves the
**chemical master equation exactly** and reports a **deep tail** with a self-certifying truncation:
the cap is grown until the answer stops moving, and the observed movement is returned as a
certificate. On a 16-problem sweep this failed **0 of 16** times, where a fixed-offset truncation
rule failed 9 of 16 and a σ-scaled rule failed 4 of 16.

That matters only for questions whose answer lives at P < 10⁻⁶ — "what is the probability that
**zero** cells survive" rather than "what fraction survives". If their questions are about
population fractions and bifurcation structure, the exact tail buys nothing and this should be said
plainly.

## UNRETRIEVED

Everything. No rate constants, no single-cell distributions, no stated gap. The 2026 paper has no
PMC deposit reachable here.

## What would have to be true for an offer to exist

A question of the form *"what is the probability that no cell in this population does X"* where the
answer is smaller than 10⁻⁴ and a mean-field or moment-closure approximation is currently standing
in for it. If their questions are not that shape, there is no offer and the correct move is not to
make one.

## The question, as a question

*Is there a quantity in your system whose answer lives below 10⁻⁴ — a probability of complete
absence rather than a fraction — where a moment closure is currently doing the work?*
