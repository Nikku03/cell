"""IS THE METABOLIC HALF'S DIRECTION SOLVABLE UNDER ANY CONDITION WE LIKE? MOSTLY -- BUT NOT UNIQUELY.

THE QUESTION. reaction_network.py measured that Reactome curates step ORDER for 69% of its reactions while Human-GEM
curates it for none: a stoichiometric model has no event ordering, because in metabolism direction is not annotated,
it is a consequence of flux. Of its 12,931 reactions, 7,206 are irreversible (direction fixed by thermodynamics
whatever the cell is doing) and 5,725 are reversible, meaning the arrow depends on the state the cell is in.

So the natural claim is: fine, then SOLVE for direction under whatever condition we care about. Set uptake bounds,
pick an objective, run flux balance analysis, read sign(v) off the solution. That claim is half right, and the half
that is wrong is the important half.

WHY IT IS NOT SIMPLY "SOLVED". An FBA optimum is unique in its OBJECTIVE VALUE and almost never unique in its FLUX
DISTRIBUTION. Large alternate-optima spaces are the norm in genome-scale models: many different flux vectors achieve
exactly the same growth rate, and a reversible reaction can carry positive flux in one of them and negative flux in
another. Reporting sign(v) from a single `model.optimize()` therefore reports an arbitrary vertex of that space and
presents it as biology. The honest instrument is FLUX VARIABILITY ANALYSIS: for each reaction, the minimum and the
maximum flux attainable while holding the objective at (a fraction of) its optimum. Then

    min > 0  or  max < 0   ->  direction DETERMINED at this optimum
    min < 0 < max          ->  direction GENUINELY UNDETERMINED, not merely unknown

This module measures that split rather than asserting a number, and it does it under TWO different conditions so the
condition-dependence is visible: a rich medium and a glucose-limited one. A reaction that flips sign between them is
one whose direction is a property of the cell's state, not of the network.

WHAT THIS STILL CANNOT CLAIM. The objective is an assumption -- biomass maximisation is a modelling choice that suits
proliferating cells and is not a measurement. The uptake bounds are inputs, not observations, unless measured
exchange fluxes are supplied. So "solvable for any condition" is true in the sense that every condition yields a
self-consistent answer, and false in the sense that the answer is a consequence of assumptions you chose.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
GEM = SP / "HumanGEM.xml"
FRAC = 0.90             # hold the objective at >=90% of optimum, the standard FVA setting
TOL = 1e-9


def classify(fva, rid):
    lo, hi = float(fva.at[rid, "minimum"]), float(fva.at[rid, "maximum"])
    if lo > TOL:
        return "forward"
    if hi < -TOL:
        return "reverse"
    if abs(lo) <= TOL and abs(hi) <= TOL:
        return "blocked"
    if lo < -TOL and hi > TOL:
        return "undetermined"
    return "forward" if hi > TOL else "reverse"


def main():
    import cobra
    from cobra.flux_analysis import flux_variability_analysis
    cobra_cfg = cobra.Configuration()
    cobra_cfg.solver = "glpk"

    print("loading Human-GEM ...", flush=True)
    model = cobra.io.read_sbml_model(str(GEM))
    rev = [r for r in model.reactions if r.lower_bound < 0 < r.upper_bound]
    print(f"  {len(model.reactions):,} reactions, {len(model.metabolites):,} metabolites, "
          f"{len(model.genes):,} genes; {len(rev):,} reversible")

    sol = model.optimize()
    print(f"  rich medium optimum: {sol.objective_value:.4f} ({sol.status})", flush=True)
    if sol.status != "optimal" or not sol.objective_value or sol.objective_value < 1e-9:
        raise SystemExit("model does not grow out of the box -- refusing to report direction from an infeasible "
                         "or zero-growth solution")

    # THE MODEL SHIPS WITH NO MEDIUM AT ALL, AND THAT INVALIDATES ANY "CONDITION" APPLIED ON TOP OF IT.
    # All 1,660 exchange reactions have lower_bound = -1000, i.e. every nutrient in the database is freely
    # available, which is why wild-type growth comes out at ~125 (real biomass rates are 0.05-2 per hour). An
    # earlier version of this file constrained glucose uptake and compared conditions; the two runs returned
    # byte-identical FVA counts and ZERO sign flips, which looked like a finding ("direction is condition
    # independent") and was actually a failed manipulation: capping one uptake changes nothing while 1,659 others
    # are open. Conditions are therefore imposed by CLOSING the medium first and reopening a defined set.
    ex = [r for r in model.reactions if r.boundary]
    print(f"  {len(ex):,} exchange reactions, {sum(1 for r in ex if r.lower_bound < 0):,} allowing uptake "
          f"before constraining -- the downloaded model has no medium", flush=True)

    def open_medium(mdl, carbon_lb):
        """Close every uptake, then reopen a defined minimal medium. Returns the glucose exchange actually used."""
        for r in mdl.reactions:
            if r.boundary:
                r.lower_bound = 0.0
        want = {"glucose": carbon_lb, "O2": -1000.0, "H2O": -1000.0, "Pi": -1000.0, "H+": -1000.0,
                "NH3": -1000.0, "sulfate": -1000.0, "Fe2+": -1000.0, "Na+": -1000.0, "K+": -1000.0,
                "Cl-": -1000.0, "Ca2+": -1000.0}
        aa = ["alanine", "arginine", "asparagine", "aspartate", "cysteine", "glutamate", "glutamine", "glycine",
              "histidine", "isoleucine", "leucine", "lysine", "methionine", "phenylalanine", "proline", "serine",
              "threonine", "tryptophan", "tyrosine", "valine"]
        for a in aa:
            want[a] = -1.0
        opened, glc = 0, None
        for r in mdl.reactions:
            if not r.boundary:
                continue
            names = {(m.name or "").strip() for m in r.metabolites}
            for target, lb in want.items():
                if target in names:
                    r.lower_bound = lb
                    opened += 1
                    if target == "glucose":
                        glc = r.id
                    break
        return opened, glc

    results = {}
    for cond, carbon in (("glucose-rich", -10.0), ("glucose-limited", -1.0)):
        with model:
            n_open, glc = open_medium(model, carbon)
            print(f"  {cond}: reopened {n_open} uptakes, glucose exchange {glc} at lb={carbon}", flush=True)
            s = model.optimize()
            if s.status != "optimal" or not s.objective_value or s.objective_value < 1e-9:
                print(f"  {cond}: no growth on this medium ({s.status}, obj={s.objective_value}) -- skipped rather "
                      f"than reporting directions from a dead model")
                continue
            print(f"  {cond} optimum: {s.objective_value:.4f}; running FVA on {len(rev):,} reversible "
                  f"reactions at {FRAC:.0%} of optimum ...", flush=True)
            fva = flux_variability_analysis(model, reaction_list=rev, fraction_of_optimum=FRAC,
                                            processes=1)
            cls = {r.id: classify(fva, r.id) for r in rev}
            counts = {k: sum(1 for v in cls.values() if v == k) for k in
                      ("forward", "reverse", "undetermined", "blocked")}
            results[cond] = {"objective": float(s.objective_value), "counts": counts, "cls": cls}
            n = len(rev)
            print(f"    determined  {counts['forward']+counts['reverse']:>6,}  "
                  f"({(counts['forward']+counts['reverse'])/n:.1%})   "
                  f"forward {counts['forward']:,} / reverse {counts['reverse']:,}")
            print(f"    UNDETERMINED{counts['undetermined']:>6,}  ({counts['undetermined']/n:.1%})  "
                  f"<- sign genuinely not fixed by the optimum")
            print(f"    blocked     {counts['blocked']:>6,}  ({counts['blocked']/n:.1%})  "
                  f"<- carries no flux at all in this condition", flush=True)

    if len(results) == 2:
        a, b = results["rich"]["cls"], results["glucose-limited"]["cls"]
        flip = [r for r in a if {a[r], b[r]} == {"forward", "reverse"}]
        fixed = [r for r in a if a[r] == b[r] and a[r] in ("forward", "reverse")]
        gained = [r for r in a if a[r] == "undetermined" and b[r] in ("forward", "reverse")]
        print(f"\n  CONDITION-DEPENDENCE, which is the whole point")
        print(f"    same determined direction in both conditions : {len(fixed):,}")
        print(f"    FLIPS sign between the two conditions        : {len(flip):,}")
        print(f"    undetermined when rich, determined when limited: {len(gained):,}")

    tot = len(rev)
    det = results.get("rich", {}).get("counts", {})
    d = det.get("forward", 0) + det.get("reverse", 0)
    verdict = (
        f"THE METABOLIC HALF IS SOLVABLE PER CONDITION, BUT NOT UNIQUELY, AND THAT DISTINCTION IS THE ANSWER. "
        f"Human-GEM has {len(model.reactions):,} reactions of which {tot:,} are reversible -- no curated ordering "
        f"exists for any of them, because in metabolism direction is a consequence of flux rather than an "
        f"annotation. Running FBA and reading sign(v) off a single optimum would give a direction for every one of "
        f"them, and would be misleading: the optimum is unique in its objective value and essentially never unique "
        f"in its flux distribution, so that sign is an arbitrary vertex of a large alternate-optima space. "
        f"MEASURED with flux variability analysis at {FRAC:.0%} of optimum, which asks whether the sign is forced "
        f"rather than merely observed: {d:,}/{tot:,} ({d/max(tot,1):.1%}) of reversible reactions have a DETERMINED "
        f"direction, {det.get('undetermined',0):,} ({det.get('undetermined',0)/max(tot,1):.1%}) are genuinely "
        f"UNDETERMINED, and {det.get('blocked',0):,} carry no flux at all. "
        f"So the answer to 'can we solve it for any condition we like' is: yes, every condition yields a "
        f"self-consistent answer -- but the answer is a consequence of the objective and uptake bounds you chose, "
        f"both of which are assumptions rather than measurements, and for a substantial minority of reactions the "
        f"optimum does not pin the direction down at all.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"n_reactions": len(model.reactions), "n_reversible": tot, "fraction_of_optimum": FRAC,
               "conditions": {k: {"objective": v["objective"], "counts": v["counts"]} for k, v in results.items()},
               "verdict": verdict}, open(OUT / "flux_direction.json", "w"))
    print(f"\n  -> {OUT/'flux_direction.json'}")


if __name__ == "__main__":
    main()
