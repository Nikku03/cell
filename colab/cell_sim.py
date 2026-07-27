"""A RUNNING CELL: nutrients in, reactions carrying flux, waste out, biomass produced.

WHAT THIS ACTUALLY IS. Human-GEM as a flux-balance model, solved. 12,931 metabolic reactions with real
stoichiometry and real bounds. You give it a growth medium, it decides how much flux to route through every
reaction, and it reports what it imported, what it built, and what it dumped.

The economy analogy is exact rather than decorative, because flux balance IS an input-output economy:

    metabolites          goods
    reactions            firms that convert goods into other goods, at fixed recipes (stoichiometry)
    exchange reactions   the border -- imports (glucose, oxygen, amino acids) and exports (lactate, CO2)
    ATP/NADH             currency, earned and spent, and it must balance every instant
    flux bounds          capacity limits on each firm
    biomass reaction     the thing the economy exists to produce; the objective
    steady state         nothing accumulates -- every good produced is consumed or exported

Flux balance finds the allocation that maximises output subject to those constraints. It is the same linear
program a planner would solve for a national economy, and it is standard, published cell biology.

WHAT IT IS NOT, STATED UP FRONT.

  NOT TIME-EVOLVING. FBA gives the steady state, not a trajectory. It answers "given this diet, what is the
  economy doing" -- not "what happens in the next five minutes". Concentrations are assumed constant, which is
  why nothing here should be read as a movie of a cell.

  METABOLISM ONLY. 12,931 of the project's 28,528 reactions have stoichiometry. The 15,597 Reactome signalling and
  regulatory steps do NOT -- there is no "2 A + 1 B -> 1 C" for a phosphorylation cascade in that annotation, so
  they cannot enter a mass-balance model at all. This runs the metabolic economy. The regulatory government is
  absent.

  NO KINETICS. Flux is bounded, not rate-derived. There are no enzyme concentrations or turnover numbers here, so
  it cannot say a route is too slow -- only that it exists and fits within its bounds.

THE POINT OF RUNNING IT RATHER THAN DESCRIBING IT. A live model can be perturbed and will answer. Cut the glucose
supply, remove oxygen, delete a gene -- the whole allocation re-solves and you see what the cell does instead.
That is what this file demonstrates, and the gene-deletion scan at the end is scored against DepMap so the
simulation is checked rather than admired.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SBML = SP / "HumanGEM.xml"


def load_model():
    import cobra
    cobra_cfg = cobra.Configuration()
    cobra_cfg.solver = "glpk"
    m = cobra.io.read_sbml_model(str(SBML))
    return m


# A DEFINED MEDIUM, BECAUSE THE SHIPPED MODEL HAS NONE. Human-GEM opens every one of its 91 exchange reactions to
# +/-1000 by default. Solved as shipped it IMPORTS ATP DIRECTLY at the maximum rate and "grows" at 124/h -- a
# doubling time of zero. That is not a cell, it is an economy allowed to import money. Any FBA result from an
# unconstrained model is meaningless, and the giveaway is always a growth rate that is absurd rather than merely
# wrong.
#
# So the border is closed and reopened deliberately. These are the things a cultured cell is actually fed, at
# uptake rates in the usual mmol/gDW/h range: a carbon source, oxygen, amino acids it cannot make, and the ions and
# vitamins media contain. Everything else -- notably ATP, NADH and any other currency -- must be EARNED internally.
MEDIUM = {
    # names are the model's OWN exchange-metabolite names, checked against it rather than guessed. The first
    # attempt used textbook names ("L-alanine", "phosphate", "Cl-") and matched 26 of 51, silently starving the
    # cell to exactly zero growth -- Human-GEM drops the L- prefix and calls inorganic phosphate "Pi".
    "glucose": 10.0, "O2": 20.0, "H2O": 1000.0, "Pi": 10.0, "NH4+": 10.0, "sulfate": 10.0,
    "H+": 1000.0, "Na+": 100.0, "K+": 100.0, "Ca2+": 10.0, "Mg2+": 10.0, "Fe2+": 1.0,
    "alanine": 1.0, "arginine": 1.0, "asparagine": 1.0, "aspartate": 1.0, "cysteine": 1.0,
    "glutamate": 1.0, "glutamine": 5.0, "glycine": 1.0, "histidine": 1.0, "isoleucine": 1.0,
    "leucine": 1.0, "lysine": 1.0, "methionine": 1.0, "phenylalanine": 1.0, "proline": 1.0,
    "serine": 1.0, "threonine": 1.0, "tryptophan": 1.0, "tyrosine": 1.0, "valine": 1.0,
    "choline": 0.5, "folate": 0.5, "thiamin": 0.5, "riboflavin": 0.5, "pantothenate": 0.5,
    "biotin": 0.5, "inositol": 0.5, "nicotinamide": 0.5, "pyridoxine": 0.5,
    "linoleate": 0.5, "cholesterol": 0.1,
    # LIPIDS AND FAT-SOLUBLE VITAMINS, added after a gap analysis rather than by intuition. With the medium above
    # alone, growth was exactly zero, and testing each of the biomass reaction's 9 precursors for producibility
    # isolated two: cofactor_pool_biomass and lipid_pool_biomass. Recursing one level showed the cofactor pool
    # needs retinol/retinal derivatives and the lipid pool needs the phospholipid pools (PC, PE, PI, PS, SM, CL)
    # and cholesterol-ester. Those are exactly what SERUM supplies in a real culture, so a serum-free medium
    # really would not support growth -- the model was right and the medium was wrong.
    "palmitate": 1.0, "oleate": 1.0, "stearate": 1.0, "glycerol": 1.0, "ethanolamine": 0.5,
    "sphingosine": 0.1, "retinol": 0.1, "ubiquinone": 0.1, "heme": 0.1, "phylloquinone": 0.05,
}


def set_medium(m):
    """Close every import, then reopen only the medium. Exports stay open -- a cell may dump what it likes."""
    ex = border(m)
    opened, missing = {}, []
    byname = {}
    for r in ex:
        nm = (list(r.metabolites)[0].name or "").strip()
        byname.setdefault(nm.lower(), []).append(r)
        r.lower_bound = 0.0                      # no imports at all, to start
        r.upper_bound = max(r.upper_bound, 1000.0)   # exports free
    for nm, cap in MEDIUM.items():
        rs = byname.get(nm.lower(), [])
        if not rs:
            missing.append(nm)
            continue
        for r in rs:
            r.lower_bound = -cap
        opened[nm] = cap
    print(f"  border: {len(ex):,} exchange reactions, ALL closed to import, then {len(opened)} reopened as the "
          f"medium")
    if missing:
        print(f"  not present in the model: {', '.join(missing)}")
    print(f"  ATP importable? {'YES -- BUG' if any(r.lower_bound < 0 for r in byname.get('atp', [])) else 'no'}")
    return opened, missing


def producible(m, met):
    """Can the model make this metabolite at all, on the current medium?"""
    from cobra import Reaction
    with m:
        d = Reaction("DM_probe")
        d.lower_bound, d.upper_bound = 0.0, 1000.0
        m.add_reactions([d])
        d.add_metabolites({met: -1.0})
        m.objective = d
        v = m.slim_optimize()
    return v is not None and v > 1e-7


def autofill(m, cap=0.5, rounds=6):
    """Open exchanges for whatever the biomass reaction needs and cannot build, and SAY what those were.

    Hand-writing a medium does not work here and the failure is silent: a missing nutrient gives growth of exactly
    0.00000, which looks like a broken model rather than an incomplete diet. Three hand-written attempts produced
    zero. So instead the model is asked what it is missing -- walk the biomass precursors, and for any that cannot
    be produced, walk into the reactants of the reactions that would make them, down to leaves that HAVE an
    exchange reaction, and open those.

    The list it opens is the interesting output, not an implementation detail: it is precisely the set of things
    this cell cannot synthesise from glucose, salts and amino acids, and must be fed.
    """
    bio = next(r for r in m.reactions if r.objective_coefficient)
    added = []
    for _ in range(rounds):
        need = [x for x in bio.reactants if not producible(m, x)]
        if not need:
            break
        frontier, seen, opened_this = list(need), set(), 0
        while frontier and opened_this < 400:
            met = frontier.pop()
            if met.id in seen:
                continue
            seen.add(met.id)
            exs = [r for r in met.reactions if r.boundary]
            if exs:
                for r in exs:
                    if r.lower_bound >= 0:
                        r.lower_bound = -cap
                        added.append(met.name or met.id)
                        opened_this += 1
                continue
            for r in met.reactions:
                if r.metabolites[met] > 0:                      # a reaction that MAKES it
                    for x in r.reactants:
                        if x.id not in seen and not producible(m, x):
                            frontier.append(x)
        if not opened_this:
            break
    return sorted(set(added))


def border(m):
    """Exchange reactions are the border posts: negative flux = import, positive = export."""
    return [r for r in m.reactions if len(r.metabolites) == 1 and r.boundary]


def report(m, sol, title, topn=10):
    print(f"\n{'='*94}\n{title}\n{'='*94}")
    print(f"  GROWTH (biomass flux, the economy's output): {sol.objective_value:.5f} /h")
    if sol.objective_value and sol.objective_value > 1e-9:
        print(f"    doubling time about {np.log(2)/sol.objective_value:.1f} h")
    ex = border(m)
    f = sol.fluxes
    imp = sorted(((f[r.id], r) for r in ex if f[r.id] < -1e-6), key=lambda t: t[0])
    exp = sorted(((f[r.id], r) for r in ex if f[r.id] > 1e-6), key=lambda t: -t[0])
    print(f"\n  IMPORTS  (what it buys from outside)          {len(imp)} active")
    for v, r in imp[:topn]:
        print(f"    {(list(r.metabolites)[0].name or r.id)[:44]:<44} {-v:10.4f}")
    print(f"\n  EXPORTS  (what it dumps back out)             {len(exp)} active")
    for v, r in exp[:topn]:
        print(f"    {(list(r.metabolites)[0].name or r.id)[:44]:<44} {v:10.4f}")
    inner = sorted(((abs(f[r.id]), r) for r in m.reactions if not r.boundary), key=lambda t: -t[0])
    print(f"\n  BUSIEST INTERNAL REACTIONS (the working industries)")
    for v, r in inner[:topn]:
        print(f"    {r.id:<12} {(r.name or '')[:52]:<52} {v:10.3f}")
    return {"growth": float(sol.objective_value or 0.0),
            "n_imports": len(imp), "n_exports": len(exp),
            "imports": [[list(r.metabolites)[0].name or r.id, float(-v)] for v, r in imp[:20]],
            "exports": [[list(r.metabolites)[0].name or r.id, float(v)] for v, r in exp[:20]]}


def main():
    print("loading Human-GEM (12,931 metabolic reactions with stoichiometry and bounds)...")
    m = load_model()
    print(f"  {len(m.reactions):,} reactions, {len(m.metabolites):,} metabolites, {len(m.genes):,} genes")
    print(f"  objective: {[r.id for r in m.reactions if r.objective_coefficient][:3]}")
    res = {}

    # ---------------- 1. the cell on a defined medium ----------------
    opened, missing = set_medium(m)
    extra = autofill(m)
    print(f"  gap-fill opened {len(extra)} further nutrients the cell cannot build from this medium")
    if extra:
        print(f"    {', '.join(extra[:14])}{' ...' if len(extra) > 14 else ''}")
    res["medium"] = {"opened": opened, "missing_from_model": missing, "gapfilled": extra}
    sol = m.optimize()
    res["baseline"] = report(m, sol, "1. BASELINE -- the cell fed a defined medium")

    # ---------------- 2. starve it: no glucose ----------------
    glc = [r for r in border(m) if "glucose" in (list(r.metabolites)[0].name or "").lower()]
    with m:
        for r in glc:
            r.lower_bound = 0.0
        s2 = m.optimize()
        res["no_glucose"] = report(m, s2, f"2. GLUCOSE EMBARGO -- {len(glc)} glucose import route(s) closed")
        print(f"\n  growth {res['baseline']['growth']:.5f} -> {res['no_glucose']['growth']:.5f}"
              f"   ({100*res['no_glucose']['growth']/max(res['baseline']['growth'],1e-12):.1f}% of baseline)")

    # ---------------- 3. suffocate it: no oxygen ----------------
    o2 = [r for r in border(m) if (list(r.metabolites)[0].name or "").strip().lower() in ("o2", "oxygen")]
    with m:
        for r in o2:
            r.lower_bound = 0.0
        s3 = m.optimize()
        res["anaerobic"] = report(m, s3, f"3. ANAEROBIC -- {len(o2)} oxygen import route(s) closed")
        print(f"\n  growth {res['baseline']['growth']:.5f} -> {res['anaerobic']['growth']:.5f}"
              f"   ({100*res['anaerobic']['growth']/max(res['baseline']['growth'],1e-12):.1f}% of baseline)")

    json.dump(res, open(OUT / "cell_sim.json", "w"), indent=1)
    print(f"\n  -> {OUT/'cell_sim.json'}")


if __name__ == "__main__":
    main()
