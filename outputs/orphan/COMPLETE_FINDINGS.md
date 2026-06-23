# The bacterial essential-gene system — complete findings

Everything the essential-gene corpus + the E. coli essential network can
currently answer, with the honest caveats. Built from 210,363 labelled genes
across 59 organisms (DEG knockouts, RB-TnSeq fitness, JCVI minimal cells),
orthology, synteny, codon and FBA features.

Artifacts:
- `cell_network_genes.png`, `cell_network_modules.png` — the wired essential genome
- `cell_live.gif` — the cell functioning (flux wave over a cell cycle)
- `cell_die.gif` — knockout → the AND gate flips → death
- `complete_findings.png` — the six quantitative results below
- `explore_findings.png`, `and_gate.png` — earlier panels (Q1–Q3 + correction)

---

## 1. The wiring (E. coli, 605 Keio essentials)

Nodes = essential genes by functional module; edges = operon adjacency
(synteny) + phyletic coupling (orthogroup co-inheritance, variable genes only).

- **336/605 (56%) are universal-core** (present in ≥90% of 48 genomes).
- After the phylogenetic-profiling fix: 1,633 biological edges (215 operon +
  1,418 phyletic). Giant component 253; **82% wired** once the core is counted.
- Mean shortest path between essentials = **5.88 hops** — a tight dependency web.
- **Key caveat:** "other (335)" is an *annotation* bucket, not biology — those
  are essentials our local GFF/OG products couldn't slot into a named module.

## 2. Live functioning (`cell_live.gif`)

A cell-cycle flux wave: membrane import → energy/cofactors → building blocks →
transcription/replication → tRNA/ribosome loading → translation → division →
repeat. **The topology is data-derived; the dynamics are a canonical schematic,
not a kinetic/FBA simulation.** It explains *how* the parts function as a
system; it does not predict rates or viability.

---

## Six quantitative results (`complete_findings.png`)

### 1) Law of essential genome size
Across 59 organisms, the wet-lab minimal cells (syn3a 383, mgen 382, mpne 27*)
sit on a **floor of ~380–460 genes**. Larger genomes carry more essentials in
absolute terms but a *smaller* essential fraction. *(mpne is a tiny partial
dataset.)*

### 2) Essentiality is bimodal, not binary
Of 1,951 orthogroups present in ≥20 genomes: **133 essential in every genome**
(hard core), **379 conserved but never essential** (dispensable shell), ~500
context-dependent (20–80%). Essentiality is a property of *function × context*,
not an intrinsic gene label — which is exactly why cross-clade prediction is
hard.

### 3) The universal irreducible core (130 OGs)
Essential in *every* genome that has them, and **overwhelmingly the translation
apparatus** (ribosomal proteins, aminoacyl-tRNA synthetases, EF-G, GroEL,
release factor). All 130 are functionally annotated — **no dark matter** in the
core. Metabolism dominates the *conditional* middle instead.

### 4) Gene duplication buffers essentiality
Of 18,072 within-organism paralog groups, **73% are entirely dispensable**
(every copy knockout-tolerant). Duplication is the main place the AND gate gets
softened — and where synthetic-lethal pairs hide (individually deletable,
jointly lethal). *Double-knockout data would be needed to confirm specific
pairs; this is a candidate set, not a verified one.*

### 5) The AND gate of viability
Cell viable **iff every essential is present**. For E. coli BW25113 (3,585
genes, 605 essential): random knockouts make the cell **50% dead by k=4**,
95% dead by k≈17. Remove any single essential → death. (`cell_die.gif` shows
this: knock out ribosomal proteins → translation, division, then everything
starves → DEAD.)

### 6) Robust-yet-fragile lives at the *whole-genome* level
Random gene loss is survivable because ~83% of genes are non-essential;
targeted removal (essentials first) kills immediately. The earlier
"essential-graph survival" framing was a **category error** — a still-connected
sub-graph of essentials after a deletion is a *corpse*. Network topology tells
you *how* failures cascade, not *whether* the cell lives.

---

## What this adds up to

A bacterial cell is a **scale-free dependency network built around an inviolable
translational core, padded with a swappable metabolic periphery, and buffered by
gene duplication.** That single sentence explains: why the universal core is
translation (#3), why essentiality is bimodal (#2), why cells tolerate random
but not targeted damage (#6), why duplicates are dispensable (#4), and why
viability is an AND gate (#5).

## Honest limitations
- Heterogeneous "essential" definitions: `beril_*`/`reut` are *fitness-defect*
  calls (20–44% "essential"), not true essentiality (Keio 17%, minimal cells 80%).
- The live/death animations are **explanatory schematics**, not kinetic models.
- FBA features exist for only 3 organisms and have an ID-bridging bug
  (b-number ↔ Keio tag) — flux *sufficiency* is not yet wired to the network.
- Synthetic-lethal pairs are *candidates* from paralogy, not measured.
- Orthogroup essentiality conservation is computed over the 48 genomes with OG
  coverage, not all 59.

## Natural next steps
1. **Quantitative live cell** — drive arrow widths by FBA flux; cycle time by
   predicted growth rate (fix the b-number bridge first).
2. **Verified synthetic lethality** — score the 13,257 dispensable paralog
   groups by co-expression/co-fitness to rank real backup pairs.
3. **Per-species minimal genome design** — universal core (130 OGs) ∪ each
   species' conditional requirements that *complete the AND*.
4. **Annotate the "other" 335** — close the module gap with a PPI/STRING layer,
   turning phyletic correlation into mechanistic interaction edges.
