# Cross-kingdom test — the same engine on a HUMAN cell

Ran the exact target-intelligence structure on a human cell (Human-GEM + Hart CEGv2/
NEGv1 reference sets) to answer one question: **does the FBA/topology essentiality
approach that generalized across BACTERIA also work on human?** Reframed as oncology
target discovery — a cell's essential genes are the drug targets.

Code: `colab/target_intel_human.py` · data: Human-GEM (Chalmers), Hart CEG/NEG.

## Headline: it largely does NOT transfer — and the reasons are informative

| metric | bacteria (E. coli) | human (Human-GEM) |
|---|---|---|
| essentiality that is **metabolic** (coverage) | most of it | **17%** (119/684 core-essential) |
| FBA precision (metabolic slice) | 0.455 | **1.00** (30/30, zero FP) |
| FBA recall (metabolic slice) | 0.163 | 0.252 |
| known drug targets recovered as essential | 29 (25 in top 10%) | **0 of 26** |
| condition-dependent essentials (nutrient removal) | many (diauxie, TCA) | **0** (glucose & glutamine removal) |

## The three findings, honestly

**1. 83% of human essentiality is invisible to a metabolic model.** Only 17% of core-
essential genes are even in Human-GEM. Human essentiality is dominated by the
spliceosome, ribosome, cell cycle, and proteasome — non-metabolic machinery a
metabolic model cannot represent. In bacteria the essential core is small, conserved,
and largely metabolic/informational, so FBA + conservation covers most of it; in human
it does not.

**2. On the metabolic slice, FBA is *precise but blind to the drug targets*.**
Precision is a perfect 1.0 (every gene it calls essential is truly essential — better
than E. coli), but it only catches 30 "hard" stoichiometric essentials. All **26 famous
oncology metabolic targets — DHFR, TYMS, RRM1/2, IMPDH1/2, GART, ATIC, PHGDH, GLS, LDHA,
HK2 — are in the model but NONE are FBA-essential.** Why: in a rich culture medium,
nucleotide synthesis is rescued by salvage pathways, and isozyme/route redundancy
buffers single deletions. The real targets are essential *by context and kinetics*
(proliferation rate, lineage, medium), not by stoichiometric necessity.

**3. Cancer's defining metabolic dependencies don't appear as stoichiometric
essentiality.** Removing glucose or glutamine produced **0** new essential genes and did
not even lower the biomass optimum — the in-silico rich medium simply reroutes. The
Warburg effect and glutamine addiction are kinetic/regulatory phenomena, not
constraint-based necessities, so generic FBA can't see them.

## Why bacteria worked and human doesn't — the same principle as everything else

> Essentiality is predictable where it is **topological** (bacteria: small conserved
> core + metabolic necessity in minimal medium). It is **not** predictable from a
> generic stoichiometric model where it is **quantitative / contextual** (human cancer:
> non-metabolic machinery, redundancy, salvage, rich medium, lineage-specific
> dependence).

This is the exact identity-vs-quantity boundary the whole project keeps hitting, now
demonstrated across kingdoms.

## What it would actually take to do human/oncology well (honest)

- **Context-specific models, not generic Human-GEM** — build cancer-cell-line models
  from expression (GIMME/iMAT/tINIT); a lung-cancer cell's essential set ≠ generic.
- **The DepMap CRISPR dependency data directly** — 1000+ cell lines give
  selectively-essential genes (the real oncology targets and the tumor-vs-normal
  selectivity axis). That data — not a stoichiometric model — is the human analog of
  the bacterial fitness screens that carried our best predictions.
- **Non-metabolic essentiality** needs the protein-level/ESM + network layers, since
  83% of it is outside metabolism. (Our W1 protein-essentiality path is the transferable
  piece; the FBA wheel is not.)
- **Kinetic/expression-constrained methods** for Warburg-type dependencies.

## Bottom line

The engine *ran* on a human cell and gave an honest, precise answer on the narrow
metabolic slice — but the bacterial recipe (FBA necessity + conservation) does not
generalize to human essentiality, and in particular does not recover oncology drug
targets from a generic model. For human, the value is not the metabolic wheel; it's the
data-driven layers (DepMap dependencies, protein-level essentiality) that address the
non-metabolic, context-dependent 83%. Good negative result: it maps precisely where the
approach stops working, and why.
