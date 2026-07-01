# Differentiation simulator — zygote/progenitor → cell fates by TF forcing

Differentiation = the regulatory network's multistability: each stable state (attractor) is
a cell type (Waddington landscape / Kauffman). Start from a progenitor, force master TFs
(hit-and-trial = TF reprogramming — real biology), and the network settles into distinct
cell fates. Code: `colab/differentiation_sim.py`.

## (B) The working model — hematopoiesis (validated toggle-logic network)

Using the canonical Krumsiek Boolean network (11 TFs with the real mutual-repression
circuits), enumerating all 2,048 states gives **5 stable attractors = the real blood cell
fates**: erythrocyte, megakaryocyte, monocyte, granulocyte (+ a silent state).

**Start from a progenitor (GATA2+CEBPA on) and force a TF — you drive differentiation:**

| forced TF(s) (the "hit-and-trial") | fate reached |
|---|---|
| SPI1/PU.1 | granulocyte |
| GATA1 + KLF1 | **erythrocyte** |
| GATA1 + FLI1 | **megakaryocyte** |
| SPI1 + GFI1 | **granulocyte** |
| SPI1 + EGR1 | **monocyte** |

The branch points are **toggle switches** (mutual repression):
- **GATA1 ⊣ PU.1** → erythroid/megakaryocyte vs myeloid (red vs white)
- **KLF1 ⊣ FLI1** → erythrocyte vs megakaryocyte (platelet)
- **EGR1 ⊣ GFI1** → monocyte vs granulocyte

This is exactly the requested experiment — start from a progenitor, change one thing, watch
a specific differentiation emerge — and it reproduces the real blood lineage tree. Forcing
GATA1 alone lands in the bipotent erythroid-megakaryocyte progenitor state; the KLF1/FLI1
toggle then decides the final fate — which is the true biology.

## (A) The honest limit — the raw MEASURED network can't do this yet

Running the full measured aggregate network (CollecTRI) as a naive Boolean model **collapses
to an all-on state** — no distinct fates. Why: the aggregate network is the *superposition of
every cell type's edges*, so it can't distinguish cell types, and without the balanced
mutual-repression logic (and cell-type-specific *active* edges) it just cascades everything on.

**So: differentiation emerges only when the network has the real toggle logic.** We have it,
curated, for hematopoiesis. To get *all* body differentiations by hit-and-trial you need that
logic for every lineage — either curated circuits (as here) or **cell-type-specific active
subnetworks** built from single-cell data. That is precisely the "cell-type activation state"
gap identified in the cell-layout synthesis.

## Answer to "hit-and-trial → all differentiations"
Yes — *within a lineage whose toggle logic we have*, hit-and-trial finds all its fates (all 4
blood cells + the exact branch rules, here). Scaling to the whole body is not more *runs*; it
needs the missing ingredient — the regulatory logic / active network for each lineage. The
simulator is real and correct; the bottleneck is data (cell-type-specific wiring), not the method.

## Honest caveats
- Curated 11-TF blood model (validated literature network), not the whole genome.
- Boolean/logic level (no kinetics) — attractors = fates, transitions = differentiation, but
  no real timing/rates (by design — the kinetics-free map).
- The measured-network collapse is a property of aggregate networks + naive dynamics; a
  cell-type-restricted network with proper logic would behave like the curated one.
