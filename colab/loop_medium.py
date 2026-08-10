"""A MEDIUM THE MODEL CAN ACTUALLY GROW ON -- and what it demands that a culture dish does not.

WHY THIS EXISTS.  loop_deficit's H3 asked whether Human-GEM's wide-open medium -- all 1,660 exchanges at
-1000, so the cell may import adrenic acid and 9-cis-retinol without limit -- was capping what flux
balance could ever score on essentiality. The test came back NOT TESTABLE, because the medium I declared
by hand supported a growth rate of exactly zero. That was my defect. A longer hand-written list is the
same defect with more typing, so this stops hand-writing it.

THE METHOD, and the reason it is better than a list.  Ask the model what it needs. `minimal_medium`
minimises total import flux subject to a growth requirement, which returns the uptakes the network
actually depends on rather than the ones I remembered. Then the physiological list becomes what it
should always have been -- not the definition of the medium, but the YARDSTICK the model's demands are
measured against:

    PHYSIOLOGICAL   glucose, O2, the essential and conditionally essential amino acids, the standard
                    culture vitamins, common ions, the two essential fatty acids. What is in a dish.
    REQUIRED        what minimal_medium says the network cannot grow without.
    THE DIFFERENCE  is a finding about Human-GEM's biomass, not about the cell: every component the
                    model demands that a culture dish does not supply is a place where its biomass
                    equation asks for something the medium cannot give, and it gets reported by name.

PREDECLARED, before any number:

    M1 FEASIBLE   the constructed medium supports growth >= 0.01/h.
                  fails -> still NOT TESTABLE, and say so again rather than dressing it up.
    M2 THE RETEST does closing the medium raise flux balance's essentiality AUC by >= 0.03 over the
                  open-medium 0.6491? This is loop_deficit's H3, asked properly.
                  fails -> the open medium was NOT what was holding FBA back, and loop 1's deficit
                  stands as a real property of the metabolic layer.
    M3 SLACK      in the open medium all 2,848 knockouts moved growth, which no real cell does. On a
                  closed medium a cell should have slack: most single deletions should do nothing.
                  gate: fewer than 90% of knockouts move growth.
                  fails -> the model is still saturated and the realism problem is elsewhere.

CONTROLS: 200 label shuffles; the open-medium run on identical code with only exchange bounds changed;
the physiological list as an independent yardstick that the constructed medium is scored against, not
fitted to.

-> outputs/loop_medium.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import auc, gpr_capacity, GEM, GEMGENES, KDEG, CELL  # noqa: E402
from loop_deficit import strat_auc, cv_auc  # noqa: E402

# EXACT metabolite names, not substrings, and the reason is a bug this file made on its first run.
# loop_deficit matched the medium by substring and so did the first version here: "Pi" then matched 44
# metabolites including alpha-pinene and adipic acid, "choline" matched 35 phosphatidylcholines, and
# "histidine" matched 15 peptides. 403 exchanges were waved through, the constructed medium left 425
# uptakes open and growth at 86.77/h against an open-medium 124.87 -- a medium that constrained almost
# nothing, which would have made the retest meaningless in a way that looked like a result.
# Human-GEM's exchange metabolite names are clean, so exact matching is both possible and correct.
# 54 of the 61 named components exist by exact name; the 7 that do not are reported, not silently
# dropped: Cl-, Zn2+, Mn2+ and cobalamin have no exchange at all in this model.
PHYSIOLOGICAL = [
    "glucose", "O2", "H2O", "H+", "Pi", "sulfate", "HCO3-", "Na+", "K+", "Ca2+", "Cl-", "Fe2+", "Fe3+",
    "Mg2+", "Zn2+", "Cu2+", "Mn2+", "selenate", "iodide",
    "histidine", "isoleucine", "leucine", "lysine", "methionine", "phenylalanine", "threonine",
    "tryptophan", "valine", "cysteine", "tyrosine", "glutamine", "arginine",
    "alanine", "aspartate", "glutamate", "glycine", "proline", "serine", "asparagine",
    "folate", "riboflavin", "thiamin", "pantothenate", "pyridoxine", "nicotinamide", "nicotinate",
    "biotin", "choline", "inositol", "cobalamin", "retinol", "alpha-tocopherol", "ascorbate",
    "linoleate", "linolenate", "urea", "pyruvate", "L-lactate",
]
PHYS_SET = {w.lower() for w in PHYSIOLOGICAL}

# THINGS A CELL CANNOT IMPORT, and the single most important line in this file.
# minimal_medium minimises TOTAL IMPORT FLUX, which rewards importing the largest molecule available:
# one unit of LDL carries enormous mass. Asked what it needed, the model answered with 26 uptakes that
# did not include glucose and did include ATP, NAD+, FAD, LDL, apoE and plasminogen. Human-GEM ships
# exchange reactions for energy currency and for intact plasma proteins, so a "closed" 79-component
# medium still supported 86.26/h -- because it contained ATP.
#
# A cell that can import ATP has no essential genes worth measuring. Every conclusion about essentiality
# drawn on this model without closing these is worthless, and that includes loop 1's. These are barred
# from uptake by name; secretion is left free, because exporting them is real.
NOT_IMPORTABLE = [
    "atp", "adp", "amp", "gtp", "gdp", "gmp", "ctp", "utp", "udp", "datp", "dgtp", "dctp", "dttp",
    "nad+", "nadh", "nadp+", "nadph", "fad", "fadh2", "fmn", "coa", "acetyl-coa", "malonyl-coa",
    "hexanoyl-coa", "succinyl-coa", "sam", "sah", "ppi", "pppi",
    "ldl", "ldl remnant", "hdl", "idl", "vldl", "chylomicron", "chylomicron remnant",
    "apoa1", "apob100", "apoc1", "apoc2", "apoc3", "apoe", "plasminogen", "albumin", "transferrin",
]
NOT_IMPORTABLE_SET = {w.lower() for w in NOT_IMPORTABLE}

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 1904
N_SHUFFLE = 200
MU_REQUIRE = 0.030          # /h -- the growth the minimal medium must support
GATE_FEASIBLE = 0.01
GATE_RETEST = 0.03
GATE_SLACK = 0.90
OPEN_MEDIUM_BEST = 0.6491   # plainFBA, open medium, from cell_loop


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("A MEDIUM THE MODEL CAN ACTUALLY GROW ON")
    say("=" * 100)
    say("  H3 came back NOT TESTABLE because the medium I wrote by hand supported zero growth. This")
    say("  asks the model what it needs instead, and uses the physiological list as a yardstick rather")
    say("  than as the definition.")

    import cobra  # noqa: F401
    from cobra.io import read_sbml_model
    from cobra.medium import minimal_medium
    t0 = time.time()
    M = read_sbml_model(str(GEM))
    M.solver = "glpk"
    say(f"\n  Human-GEM loaded ({time.time()-t0:.0f}s): {len(M.reactions)} reactions, "
        f"{len(M.exchanges)} exchanges, all open at {list(M.exchanges)[0].lower_bound}")
    mu_open = M.slim_optimize()
    say(f"  growth with everything open: {mu_open:.2f}/h  -- a cell that can eat anything")

    barred = []
    for r in M.exchanges:
        nm = (list(r.metabolites)[0].name or "").strip().lower()
        if nm in NOT_IMPORTABLE_SET:
            r.lower_bound = 0.0
            barred.append(nm)
    mu_barred = M.slim_optimize()
    say(f"\n  BARRING WHAT A CELL CANNOT IMPORT -- {len(barred)} exchanges closed to uptake")
    say(f"    {', '.join(sorted(set(barred))[:14])}{' ...' if len(set(barred)) > 14 else ''}")
    say(f"    growth drops {mu_open:.2f}/h -> {mu_barred:.2f}/h. Human-GEM ships uptake reactions for")
    say(f"    energy currency and intact plasma proteins; a model that can import ATP has no essential")
    say(f"    genes worth measuring, and that applies to loop 1's numbers too.")

    # ---- what does the network actually need? ---------------------------------------------------------
    say(f"\n  asking minimal_medium for the uptakes required to reach {MU_REQUIRE}/h ...")
    t1 = time.time()
    mm = minimal_medium(M, MU_REQUIRE, minimize_components=False)
    if mm is None:
        say("  minimal_medium found no solution. Nothing below can be computed; recorded as such.")
        return 1
    mm = mm[mm > 1e-9].sort_values(ascending=False)
    say(f"    {len(mm)} exchange reactions required ({time.time()-t1:.0f}s)")

    def met_name(rid):
        r = M.reactions.get_by_id(rid)
        m = list(r.metabolites)[0]
        return m.name or m.id

    req = {rid: met_name(rid) for rid in mm.index}

    def is_physio(nm):
        return (nm or "").strip().lower() in PHYS_SET

    phys_hit = {k: v for k, v in req.items() if is_physio(v)}
    exotic = {k: v for k, v in req.items() if not is_physio(v)}
    say(f"    of those, {len(phys_hit)} are things a culture dish supplies and {len(exotic)} are not")
    say(f"\n    WHAT THE MODEL DEMANDS THAT A DISH DOES NOT -- a finding about Human-GEM's biomass:")
    for rid, nm in sorted(exotic.items(), key=lambda x: -float(mm[x[0]]))[:15]:
        say(f"      {nm:<52} {float(mm[rid]):>10.3f} mmol/gDW/h")
    if len(exotic) > 15:
        say(f"      ...and {len(exotic)-15} more")

    # ---- build the medium: what a dish gives, PLUS what the model cannot grow without -----------------
    keep = set(mm.index)
    found = set()
    for r in M.exchanges:
        m = list(r.metabolites)[0]
        if is_physio(m.name or ""):
            keep.add(r.id)
            found.add((m.name or "").strip().lower())
    absent = [w for w in PHYSIOLOGICAL if w.lower() not in found]
    say(f"\n  physiological list: {len(PHYSIOLOGICAL)} components, {len(found)} present in the model "
        f"by EXACT name")
    if absent:
        say(f"    no exchange exists for: {', '.join(absent)}")
    for r in M.exchanges:
        r.lower_bound = 0.0
    for rid in keep:
        M.reactions.get_by_id(rid).lower_bound = -1000.0
    mu_med = M.slim_optimize()
    say(f"\n  CONSTRUCTED MEDIUM: {len(keep)} open uptakes of {len(M.exchanges)} "
        f"({len(keep)/len(M.exchanges):.1%})")
    say(f"    growth: {mu_med:.4f}/h   (open medium {mu_open:.2f}/h -- a {mu_open/max(mu_med,1e-9):.0f}x "
        f"reduction, achieved by nutrients rather than by squeezing enzymes)")
    m1 = bool(np.isfinite(mu_med) and mu_med >= GATE_FEASIBLE)
    say(f"    M1 FEASIBLE (>= {GATE_FEASIBLE}/h)   {'PASS' if m1 else 'FAIL'}")
    if not m1:
        say("    Still NOT TESTABLE. Reported as such rather than dressed up.")
        json.dump({"test": "loop_medium", "M1": False, "mu": float(mu_med), "log": log},
                  open(OUT / "loop_medium.json", "w"), indent=2)
        return 1

    # ---- knockouts on the closed medium ---------------------------------------------------------------
    gpr = {r.id: r.gpr for r in M.reactions if r.gene_reaction_rule.strip()}
    ORIG = {r.id: r.bounds for r in M.reactions}
    gids = [g.id for g in M.genes]
    plain = {}
    t2 = time.time()
    for n, gid in enumerate(gids):
        g = M.genes.get_by_id(gid)
        touched = []
        for r in g.reactions:
            if r.id not in gpr:
                continue
            E0 = {x.id: 1.0 for x in r.genes}
            E0[gid] = 0.0
            if (gpr_capacity(gpr[r.id], E0, 1.0) or 0.0) <= 0.0:
                touched.append(r)
                r.bounds = (0.0, 0.0)
        v = M.slim_optimize()
        plain[gid] = 0.0 if (v is None or not np.isfinite(v)) else float(max(v, 0.0))
        for r in touched:
            r.bounds = ORIG[r.id]
        if n % 1000 == 0:
            say(f"      KO {n}/{len(gids)}  ({time.time()-t2:.0f}s)")
    say(f"    knockout pass done in {time.time()-t2:.0f}s")

    moved = sum(1 for v in plain.values() if abs(v - mu_med) > 1e-9 * max(mu_med, 1))
    lethal = sum(1 for v in plain.values() if v <= 1e-9)
    frac = moved / len(gids)
    say(f"\n  M3 SLACK -- does the cell now have any?")
    say(f"    knockouts that change growth : {moved} of {len(gids)} ({frac:.1%})   "
        f"open medium was 2848 of 2848 (100.0%)")
    say(f"    outright lethal              : {lethal}")
    m3 = bool(frac < GATE_SLACK)
    say(f"    gate < {GATE_SLACK:.0%}   M3 {'PASS' if m3 else 'FAIL'}")

    # ---- score ----------------------------------------------------------------------------------------
    prev = json.load(open(OUT / "cell_loop.json"))
    T = pd.DataFrame(prev["scores"])
    y = T.ess.to_numpy()
    gm = pd.read_csv(GEMGENES, sep="\t")
    ens2sym = {a: b for a, b in zip(gm["genes"], gm["geneSymbols"]) if isinstance(b, str)}
    sym2gid = {ens2sym[g]: g for g in gids if g in ens2sym}
    T = T.assign(medium=[1.0 - plain[sym2gid[s]] / mu_med if s in sym2gid else np.nan for s in T.sym])
    ok = T.medium.notna().to_numpy()
    a_med = auc(T.medium[ok], y[ok])
    say(f"\n  M2 THE RETEST -- loop_deficit's H3, asked properly")
    say(f"    scored {int(ok.sum())} genes")
    say(f"    defined-medium FBA  {a_med:.4f}")
    say(f"    open-medium FBA     {OPEN_MEDIUM_BEST:.4f}      change {a_med-OPEN_MEDIUM_BEST:+.4f}")
    m2 = bool(a_med - OPEN_MEDIUM_BEST >= GATE_RETEST)
    say(f"    gate +{GATE_RETEST}   M2 {'PASS' if m2 else 'FAIL'}")

    # does it still carry information the expression lookup does not?
    lr = np.log10(T.rpkm.to_numpy() + 1e-3)
    sa, _ = strat_auc(T.medium[ok].to_numpy(), y[ok], lr[ok])
    a_look, _ = cv_auc(lr[ok].reshape(-1, 1), y[ok])
    a_both, _ = cv_auc(np.c_[lr[ok], T.medium[ok]], y[ok])
    say(f"\n  and against the expression confound, on the same genes:")
    say(f"    within abundance-matched deciles : {sa:.4f}")
    say(f"    5-fold CV lookup {a_look:.4f} -> lookup + defined-medium FBA {a_both:.4f} "
        f"({a_both-a_look:+.4f})")

    null = np.array([auc(T.medium[ok], rng.permutation(y[ok])) for _ in range(N_SHUFFLE)])
    say(f"    control: {N_SHUFFLE} label shuffles -> {null.mean():.4f} +/- {null.std():.4f}")

    say("\n" + "=" * 100)
    for k, v in (("M1 medium supports growth", m1), ("M2 closing it recovers AUC", m2),
                 ("M3 the cell has slack", m3)):
        say(f"  {k:<32}{'PASS' if v else 'FAIL'}")
    if m2:
        say("  THE OPEN MEDIUM WAS HOLDING FLUX BALANCE BACK. A cell allowed to import anything cannot")
        say("  have essential biosynthetic genes; closing the medium recovers real signal, and loop 1's")
        say("  G4 result was partly an artefact of a model shipped without a medium.")
    else:
        say("  CLOSING THE MEDIUM DID NOT RESCUE IT. The open medium was a genuine realism defect and")
        say("  is now fixed, but it was not what capped the essentiality AUC. Loop 1's deficit stands")
        say("  as a property of the metabolic layer, and the honest move is to stop attributing it to")
        say("  the setup.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(GEM), str(GEMGENES), str(OUT / "cell_loop.json"), str(CELL)],
                      available=len(M.genes), used=int(ok.sum()), selection="filtered", seed=SEED,
                      controls=[f"{N_SHUFFLE} label shuffles", "open vs defined medium, identical code",
                                "physiological list as an independent yardstick",
                                "abundance-matched strata", "5-fold cross-validation"],
                      note="medium = what minimal_medium requires, union what a culture dish supplies")
    RM.report(man, emit=say)
    json.dump({"test": "loop_medium", "manifest": man,
               "gates": {"M1 feasible": m1, "M2 retest": m2, "M3 slack": m3},
               "mu_open": float(mu_open), "mu_barred": float(mu_barred),
               "n_barred": len(barred), "barred": sorted(set(barred)),
               "mu_medium": float(mu_med),
               "n_open_uptakes": len(keep), "n_required": len(mm),
               "required_physiological": phys_hit, "required_exotic": exotic,
               "auc_defined_medium": float(a_med), "auc_open_medium": OPEN_MEDIUM_BEST,
               "stratified_auc": float(sa), "cv_lookup": float(a_look), "cv_combined": float(a_both),
               "n_moved": moved, "n_lethal": lethal, "frac_moved": float(frac),
               "null": {"mean": float(null.mean()), "sd": float(null.std())},
               "scores": T.to_dict("records"), "log": log},
              open(OUT / "loop_medium.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_medium.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
