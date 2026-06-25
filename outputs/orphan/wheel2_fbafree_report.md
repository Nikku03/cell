# Can Wheel 2 decide "what's needed" without FBA? — honest verdict

You asked: drop FBA, let Wheel 2 look at the assembled cell + the trucks and
decide what's needed by itself. I built two pure-graph (no-LP) ways to answer
"is this gene needed", on iML1515 vs Keio truth.

| method (no FBA) | precision | recall | vs FBA |
|---|---|---|---|
| **sole-connection** (gene uniquely makes/uses a metabolite → removing it orphans it) | 0.212 | **0.588** | recovers **80%** of FBA's essentials… |
| …but over-calls | (538 flagged, 114 true) | | …buried in 3.7× false positives |
| **network reachability / scope** (removing gene un-reaches a needed precursor) | 0.091 | 0.005 | recovers 4% |
| FBA single-gene-deletion (reference) | 0.794 | 0.397 | — |

## What this shows

**The good news for your instinct:** the graph *does* carry the necessity
signal. Sole-connection reasoning — pure "who uniquely connects to what",
no flux — recovers **80% of the genes FBA calls essential**, and the worked
example is exactly your mechanism: *"DADP would be orphaned without deoxyadenylate
kinase (b0474)"* → b0474 is essential. No optimization needed to see that.

**Why FBA still resists removal:** the two pure-graph methods fail in opposite
directions, and neither is fixable by tuning —

1. **Sole-connection over-calls 3.7×.** It flags every gene that uniquely
   touches a metabolite, but can't tell whether that metabolite *matters* to the
   cell or whether an alternate route covers it. It has the recall, not the
   precision.
2. **Network reachability under-calls 200×.** Even the *intact* cell only
   "reaches" 33 of 64 biomass precursors by pure expansion — the classic
   cofactor-bootstrap problem: a biosynthesis step needs a cofactor that itself
   must be synthesized, and connectivity alone can't break the cycle.

The thing in the middle that gets both right is **steady-state mass balance** —
every metabolite's production must equal its consumption, cofactors must
regenerate in balance. That is precisely what FBA enforces, and precisely what
pure connectivity cannot express. FBA isn't a lazy crutch here; the question
"can the cell still make everything it needs without this gene" *is* a
mass-balance question.

## The honest split of Wheel 2

- **"Which gene fills a known hole?"** — fully FBA-free, already works: the
  dimension filter narrows ~250 same-family candidates to ~7 with the true gene
  retained 100% (wheel2_gapfill). No flux anywhere in that step.
- **"Which holes are non-redundantly needed?"** — this is the necessity
  question, and it needs mass balance. Pure graph gives a recall-heavy proxy
  (sole-connection, 0.59 recall) but can't reach FBA's precision without the
  balance constraint.

## Bottom line

You were right that the *identification* half of Wheel 2 needs no FBA — it never
did. But the *necessity* half ("what's actually needed") is a mass-balance
question, and that's the one irreducible job FBA does. Removing it with pure
connectivity either floods you with false holes (sole-connection) or can't even
build the intact cell (reachability). The graph carries the signal; mass balance
is what ranks it.
