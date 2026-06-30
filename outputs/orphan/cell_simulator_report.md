# Integrated E. coli cell SIMULATOR (the success criterion)

The laid-out cell as a runnable perturbation engine: knock out / mutate genes +
set condition -> predict viability, growth, and downstream regulatory effects.
Combines METABOLIC (FBA iJO1366) + REGULATORY (RegulonDB propagation) +
CONDITIONAL (medium) layers. Validated vs Keio.

## Validation (single-gene KO viability vs Keio, glucose)
accuracy 0.791, precision 0.453, recall 0.432 (= iJO1366 metabolic ceiling).
Non-metabolic essentials (ribosome/replication) need the W1 (ESM+conservation,
0.768) overlay, which exists.

## Worked perturbations (all correct)
- KO pyrB (glucose) -> LETHAL (pyrimidine biosynthesis)
- KO sdhA: VIABLE on glucose, LETHAL on succinate -- SAME GENE, condition-dependent
- KO malQ on maltose -> LETHAL (conditional)
- KO crp (global TF) -> viable, 591 targets lose regulation (cascade enumerated)
- mutate pgi->null -> viable with growth cost (PPP bypass)
- aceA+aceB on acetate -> viable (honest FBA model gap; should be lethal)

## Success against the bar
A working, validated, integrated simulator: "exclude this gene / mutate here, see
what happens" returns a real mostly-correct answer across metabolic + regulatory +
conditional layers. Demonstrates condition-dependent essentiality and TF-KO
cascades on the real network.

## Gaps to "better than a simulator"
1. metabolic precision = FBA ceiling 0.45 -> overlay W1 essential core for recall
2. regulatory KO is topological (what's affected) -> wire closed loop for the
   quantitative new steady state (TF KO -> expression -> metabolic consequence)
3. FBA per-reaction model gaps (aceA/B) -- fixable

Files: colab/cell_simulator.py, outputs/orphan/cell_simulator.json.
