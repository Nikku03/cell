# Kinetics into Wheel 2 — built it, and it does not improve essentiality

**What you asked for, built literally.** Active site → which reaction an enzyme
runs and which metabolites it touches (its partners in the graph); kinetics
(kcat) → how much flux that enzyme can carry; then let it propagate through the
network. That is **enzyme-constrained FBA** (sMOMENT/GECKO), and I wired it onto
Wheel 2 on iJO1366 (E. coli), validated against real **Keio** essentiality.

Mechanism: every gene-associated reaction draws on a shared proteome pool
`sum_j (v_fwd_j + v_rev_j)/kcat_j ≤ P`. A low-kcat (slow) enzyme costs more
budget per unit flux, so a knockout that forces flux onto a slow bypass should
blow the budget and become essential even though a bypass exists topologically.
kcat assigned by EC class (Bar-Even 2011 medians) — the only transferable part
of kinetics; the validated output is essentiality, not the rates.

## Result — kinetics changes nothing for essentiality

Eval universe: 1036 genes both in iJO1366 and in Keio truth (199 essential).

| model | precision | recall | F1 |
|---|---|---|---|
| plain FBA single-gene-deletion | 0.453 | 0.432 | 0.442 |
| **enzyme-constrained (kinetic)** | **0.453** | **0.432** | **0.442** |

**Genes essential ONLY when kinetics is on: 0.** Identical essential set.

Budget sweep — even at a brutally tight pool that cuts WT growth to 80% of max:

| WT-growth kept | new kinetic essentials | continuous AUC (growth-drop vs truth) |
|---|---|---|
| 99% | 0 | 0.703 |
| 95% | 0 | 0.718 |
| 90% | 0 | 0.698 |
| 80% | 0 | 0.703 |

(plain FBA continuous AUC = 0.700). Kinetics neither flips a binary call nor
re-ranks genes — AUC is flat across every budget.

## Why — this is fundamental, not a tuning failure

1. **Essentiality is topological; kinetics is quantitative.** Essential = "is
   there ANY route." Kinetics = "how fast." Tightening the enzyme budget scales
   all fluxes down roughly proportionally — it lowers predicted *growth rate*,
   but it does not selectively destroy the bypass of one specific gene. A gene
   dispensable by topology stays dispensable.

2. **A coarse, transferable kcat can't manufacture a hard bottleneck.** To make
   a NEW essential you need a reaction whose only bypass is so slow that carrying
   the required flux alone exceeds the budget. EC-class medians span ~13–50/s
   (half an order). The genuine outliers that could create a bottleneck (enzymes
   at ~0.01/s) are *per-enzyme* values — and per-enzyme kcat is exactly the
   quantity we already proved does **not** transfer across organisms (6–8 orders
   of spread, Bar-Even 2011). So the kinetics that could help is the kinetics we
   can't get; the kinetics we can get (family priors) doesn't move essentiality.

3. **This reproduces the literature.** GECKO/ecFBA improve prediction of growth
   *rate*, overflow metabolism, and proteome allocation — **not** gene-essentiality
   classification. Our null is the expected result, now measured on our own data.

## The constructive takeaway — kinetics belongs to a different question

Kinetics is the right physics for **graded fitness / growth-rate defect**
(how much slower does the cell grow without this gene), which is the **Wheel 4**
question (feba conditional fitness), *not* the binary Wheel 2 essentiality
question. If the goal is to make kinetics pay off:
- it would predict the *magnitude* of a fitness defect, not the essential/not flip;
- and it would require real measured kcat (the GECKO BRENDA/DLKcat route, needs
  network access), since the transferable family prior provably can't create the
  bottlenecks that would matter.

**Bottom line:** I integrated kinetics into Wheel 2 exactly as described and
measured it against truth. It does not improve essentiality at any budget, for a
structural reason: essentiality is about paths, kinetics is about speed, and the
only kinetics that could change a path is the per-enzyme kcat that doesn't
transfer. Wheel 2 stays topology/FBA; kinetics' real home is graded fitness
(Wheel 4).
