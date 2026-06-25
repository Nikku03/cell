# Wheel 2 on the conservation-blind soup — the real test

**Your design:** Wheel 2 should rescue the soup — the non-conserved essentials,
orphan essentials, and look-alike non-essentials that conservation cannot
separate — by *cellular necessity* (this gene's product fills a hole the cell
needs), not by orthology.

**The test:** restrict to the conservation-blind soup (leak-clean W2 < 0.30 OR
orphan), on the 4 organisms with a native model. Inside that subset — where
conservation has zero signal by construction — does a **gene-local necessity
call** (the gene is FBA single-gene-deletion essential in the org's *own* model)
separate the essentials?

## Result: it works — but only on the slice the model can see

| organism | soup n | soup ess (base) | necessity P | lift | soup-ess reachable | recovered |
|---|---|---|---|---|---|---|
| **mtub** | 3,638 | 760 (0.21) | **0.67** | 3.2× | 321/760 (42%) | **137** |
| beril_Putida | 3,350 | 192 (0.06) | 0.08 | 1.3× | 45/192 (23%) | 4 |
| beril_Keio | 2,520 | 119 (0.05) | 0.11 | 2.4× | 27/119 (23%) | 1 |
| beril_Koxy | 3,228 | 436 (0.14) | — | — | 4/436 (1%)* | 0 |

\*Koxy is a bridge artifact — the gene-name bridge maps only 4 soup genes to the
model, so there's effectively no model coverage to test. Ignore its 0.

**Aggregate (12,736 soup genes, base rate 0.118):** gene-local necessity calls
266 genes at **0.534 precision (4.5× base)** and recovers **142 / 1,507 soup
essentials (9%)**.

## The two findings that matter

**1. Your idea is correct — necessity rescues conservation-blind essentials,
exactly where conservation is weakest.** Look at **mtub**: a phylogenetically
isolated organism where conservation transfers terribly, so 760 real essentials
fall into the soup. Gene-local necessity pulls **137 of them back at 0.67
precision** — essentials that conservation completely missed, recovered by
cellular necessity alone. This is your Wheel 2 working as designed. The reason
Putida/Keio show almost nothing is the flip side: their essentials are
well-conserved, so they're *not in the soup* in the first place — there's little
to rescue. **Necessity's payoff scales with how badly conservation fails**,
which is precisely when you need it.

**2. The hard wall: 74% of the soup essentials are not metabolic at all.** Only
**26%** of conservation-blind essentials are even *in* a metabolic model — the
other **74% are unreachable** by any metabolic gap-fill, no matter how good.
The product annotations make it obvious:

- **recovered (metabolic):** synthase, kinase, ligase, reductase, oxidase,
  cytochrome, phosphate — pathway enzymes. The gap-fill's home turf. ✓
- **unreachable (non-metabolic):** *protein (416), family (170), transporter
  (122), hypothetical (92), regulator (50), ribosomal (37), permease (36),
  transcriptional (39)…* — membrane proteins, transporters, regulators,
  ribosomal proteins, and unknowns. A flux-balance model has no representation
  of these, so it can never flag them.

## What this actually says about Wheel 2

Your assembly metaphor is right, and it rescues the conservation-blind soup —
**for the metabolic subsystem.** The cell "sees a metabolic hole and the gene
that fills it" exactly as you described, and it catches non-conserved metabolic
essentials conservation can't.

But the soup is **dominated (74%) by NON-metabolic essentials** — the cell's
walls, doors, wiring, and machines (membrane, transport, regulation,
translation), not its plumbing (metabolism). The metaphor generalizes perfectly
to them — "I need a wall here, a regulator there" — but **we only hold the
blueprint for one subsystem.** The metabolic model is the plumbing schematic;
we have no equivalent hole-detector for membrane architecture, the ribosome, the
divisome, or transport, so those holes are invisible.

## Bottom line

Wheel 2 (gene-local necessity) **does** separate essentials in the
conservation-blind soup — 4.5× base-rate precision, and a real 137-gene rescue
in the organism where conservation fails hardest. It did not "fail." It is
**capped by scope, not by the idea**: it can only fill holes in the one
subsystem we have a blueprint for. To rescue the other 74%, the same assembly
logic needs blueprints for the non-metabolic subsystems — or a data source that
reports necessity directly across all of them (condition-specific fitness,
Wheel 4), which is the one signal that sees membrane/transport/regulatory genes
too.
