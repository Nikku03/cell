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


# THE MEDIUM: RICH, WITH REALISTIC UPTAKE CAPS. Three attempts at a hand-written MINIMAL medium each produced
# growth of exactly 0.00000, and chasing it down was instructive rather than wasted:
#
#   attempt 1  matched 26 of 51 nutrient names -- Human-GEM drops the L- prefix on amino acids and calls
#              inorganic phosphate "Pi", so the cell starved on a spelling mistake that reads identically to a
#              broken model
#   attempt 2  correct names, still zero. Testing each of the biomass reaction's 9 precursors for producibility
#              isolated two: cofactor_pool_biomass (needs retinol/retinal derivatives) and lipid_pool_biomass
#              (needs the PC/PE/PI/PS/SM/CL phospholipid pools and cholesterol-ester) -- exactly what SERUM
#              supplies in real culture, so the model was right and the medium was wrong
#   attempt 3  automatic gap-fill opened five more (PE-LD pool, PS-LD pool, cobalamin, tocotrienol, lipoic acid)
#              and it was STILL zero, because the blocked set runs deeper than the gap-filler reached
#
# Curating a minimal medium for a genome-scale human model is a research task in its own right, so this uses the
# standard alternative: a RICH medium, which is what serum-containing culture actually is, with every uptake
# capped at a realistic rate instead of the shipped 1000. Two things must still be EARNED and are blocked
# outright -- the currency metabolites, and the four biomass pools themselves, since importing the product is
# the purest form of cheating available to an FBA model.
CURRENCY = {"atp", "adp", "amp", "gtp", "gdp", "gmp", "utp", "udp", "ump", "ctp", "cdp", "cmp",
            "nad+", "nadh", "nadp+", "nadph", "fad", "fadh2", "coa", "acetyl-coa", "ppi",
            "creatine-phosphate", "pep", "dtmp", "dcmp", "dgmp", "damp", "akg", "oaa",
            "succinyl-coa", "glutamyl-glutamate", "1,3-bisphospho-d-glycerate", "carbamoyl-phosphate"}
POOLS = {"MAM10012c", "MAM10013c", "MAM10014c", "MAM10015c"}
GENEROUS = {"glucose": 10.0, "O2": 20.0, "H2O": 1000.0, "H+": 1000.0, "Pi": 10.0, "glutamine": 5.0}
UPTAKE_CAP = 1.0


def set_medium(m):
    ex = border(m)
    nb = nc = 0
    for r in ex:
        met = list(r.metabolites)[0]
        nm = (met.name or "").strip().lower()
        if nm in CURRENCY or met.id in POOLS:
            r.lower_bound = 0.0
            nb += 1
        else:
            r.lower_bound = max(r.lower_bound, -UPTAKE_CAP)
            nc += 1
        r.upper_bound = max(r.upper_bound, 1000.0)
    for nm, cap in GENEROUS.items():
        for r in ex:
            if (list(r.metabolites)[0].name or "").strip().lower() == nm.lower():
                r.lower_bound = -cap
    print(f"  border: {nb} currency/biomass-pool imports BLOCKED (must be earned), "
          f"{nc} others capped at {UPTAKE_CAP} mmol/gDW/h")
    return {"blocked": nb, "capped": nc, "generous": GENEROUS}, []


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
        dt = np.log(2) / sol.objective_value
        flag = "  <- UNREALISTIC; real K562 doubles in ~24 h" if dt < 5 else ""
        print(f"    implied doubling time {dt:.2f} h{flag}")
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
    lac = [(v, r) for v, r in exp if "lactate" in (list(r.metabolites)[0].name or "").lower()]
    if lac:
        o2 = [f[r.id] for r in ex if (list(r.metabolites)[0].name or "").strip() == "O2"]
        print(f"\n  WARBURG CHECK: lactate exported at {lac[0][0]:.1f} while O2 is being consumed at "
              f"{-o2[0] if o2 else 0:.1f}")
        print(f"    fermenting glucose to lactate WITH oxygen available is the defining metabolic phenotype of")
        print(f"    a proliferating cancer line, and K562 is one. The model was never told this.")
    return {"growth": float(sol.objective_value or 0.0),
            "lactate_export": float(lac[0][0]) if lac else 0.0,
            "n_imports": len(imp), "n_exports": len(exp),
            "imports": [[list(r.metabolites)[0].name or r.id, float(-v)] for v, r in imp[:20]],
            "exports": [[list(r.metabolites)[0].name or r.id, float(v)] for v, r in exp[:20]]}


def depmap_k562():
    """DepMap CRISPR gene effect for K562 SPECIFICALLY (ACH-000551), not a cross-line average.

    More negative = the real cell loses fitness when the gene is cut. Around -1 is the median of known-essential
    genes; around 0 is no effect.
    """
    import csv
    f = SP / "CRISPRGeneEffect.csv"
    if not f.exists():
        return {}
    with open(f) as fh:
        rd = csv.reader(fh)
        head = next(rd)
        genes = [h.split(" (")[0] for h in head[1:]]
        for row in rd:
            if row[0].strip() == "ACH-000551":
                return {g: float(v) for g, v in zip(genes, row[1:]) if v not in ("", "NA")}
    return {}


def validate(m, wt_growth):
    """Delete every gene in the model, one at a time, and ask whether the simulation agrees with the real cell.

    THIS IS THE ACCURACY QUESTION, AND IT IS ANSWERABLE IN A WAY THE GROWTH RATE IS NOT. The absolute growth rate
    is meaningless here (uncurated rich medium, no loop removal), but a knockout's RELATIVE effect -- does removing
    this gene stop the cell -- is exactly what DepMap measured in real K562. The model never sees DepMap.

    Reported with the control that matters: predicted-essential genes should be more essential in the real cell
    than genes matched on how many reactions they participate in, because "genes in many reactions are essential"
    would otherwise be the whole result.
    """
    from cobra.flux_analysis import single_gene_deletion
    dep = depmap_k562()
    if not dep:
        print("\n  (no DepMap file -- validation skipped rather than faked)")
        return None
    print(f"\n{'='*94}\n4. ACCURACY vs REAL K562 -- deleting all {len(m.genes):,} model genes in silico\n{'='*94}")
    print(f"  DepMap K562 (ACH-000551) covers {len(dep):,} genes")
    ko = single_gene_deletion(m)
    pred = {}
    for ids, val in zip(ko["ids"], ko["growth"]):
        g = list(ids)[0] if isinstance(ids, (set, frozenset)) else str(ids)
        pred[g] = 0.0 if val is None or np.isnan(val) else float(val)
    ratio = {g: (v / wt_growth if wt_growth > 1e-9 else 0.0) for g, v in pred.items()}
    common = [g for g in ratio if g in dep]
    print(f"  genes in both model and DepMap: {len(common):,}")
    if len(common) < 50:
        return None
    lethal = [g for g in common if ratio[g] < 0.5]
    ok = [g for g in common if ratio[g] >= 0.5]
    ml = float(np.mean([dep[g] for g in lethal])) if lethal else float("nan")
    mo = float(np.mean([dep[g] for g in ok])) if ok else float("nan")
    print(f"\n  model says LETHAL (<50% of wild-type growth) : {len(lethal):>5}   real DepMap mean {ml:+.4f}")
    print(f"  model says TOLERATED                          : {len(ok):>5}   real DepMap mean {mo:+.4f}")
    print(f"  separation                                    : {ml-mo:+.4f}  (negative = correct direction)")
    # AUC: can the model rank real-essential genes above real-nonessential ones?
    y = np.array([1 if dep[g] < -0.5 else 0 for g in common])
    x = np.array([-ratio[g] for g in common])          # higher = more predicted-lethal
    if y.sum() and (1 - y).sum():
        order = np.argsort(x)
        r = np.empty(len(x), float)
        r[order] = np.arange(1, len(x) + 1)
        auc = float((r[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (1 - y).sum()))
        print(f"\n  AUC ranking real-essential genes (DepMap < -0.5): {auc:.4f}   (0.5 = coin flip)")
        print(f"    real-essential among model genes: {int(y.sum()):,}/{len(y):,} ({y.mean():.1%})")
    else:
        auc = None
    # recall/precision of the lethal call
    tp = sum(1 for g in lethal if dep[g] < -0.5)
    prec = tp / max(len(lethal), 1)
    rec = tp / max(int(y.sum()), 1)
    base = y.mean()
    print(f"  lethal-call precision {prec:.3f} vs {base:.3f} base rate ({prec/max(base,1e-9):.2f}x), "
          f"recall {rec:.3f}")
    return {"n_common": len(common), "n_lethal": len(lethal), "depmap_lethal": ml, "depmap_tolerated": mo,
            "separation": ml - mo, "auc": auc, "precision": prec, "recall": rec, "base_rate": float(base)}


def main():
    print("loading Human-GEM (12,931 metabolic reactions with stoichiometry and bounds)...")
    m = load_model()
    print(f"  {len(m.reactions):,} reactions, {len(m.metabolites):,} metabolites, {len(m.genes):,} genes")
    print(f"  objective: {[r.id for r in m.reactions if r.objective_coefficient][:3]}")
    res = {}

    # ---------------- 1. the cell on a defined medium ----------------
    opened, _ = set_medium(m)
    res["medium"] = opened
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

    # ---------------- 4. HOW ACCURATE IS IT? every model gene deleted, scored against real K562 ----------------
    res["validation"] = validate(m, res["baseline"]["growth"])

    json.dump(res, open(OUT / "cell_sim.json", "w"), indent=1)
    print(f"\n  -> {OUT/'cell_sim.json'}")


if __name__ == "__main__":
    main()
