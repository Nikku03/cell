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

WHAT HAPPENED, written after the run against the gates above, unedited.

    M1 FEASIBLE AND LIMITING   PASS.  57 open uptakes of 1,660, uptake capped at 0.2445 mmol/gDW/h
                               per component, growth 0.0300/h -- a 4,162x reduction from the shipped
                               model, achieved by nutrients rather than by squeezing enzymes.
    M2 THE RETEST              PASS.  Defined-medium FBA 0.6850 against open-medium 0.6491, a change
                               of +0.0359 against a +0.03 gate. THE OPEN MEDIUM WAS HOLDING FLUX
                               BALANCE BACK, and loop 1's G4 was partly an artefact of a model
                               shipped without one.
    M3 SLACK                   FAIL -- AND THE FAILURE WAS MINE, NOT THE MODEL'S. Corrected by
                               loop_slack, which audited this gate instead of believing it.
                               The 92% is an artefact of how mu_WT was obtained here: it is the
                               BISECTION'S LAST ITERATE (0.029999790) rather than a re-solve at the
                               final bounds, so every unaffected knockout solves to a marginally higher
                               mu and shows a cost of about -7.2e-07 -- negative, which is impossible.
                               M3's `abs(diff) > 1e-9` counted all 1,525 of those as "moving growth".
                               THE REAL NUMBERS: 14.4% of knockouts have any positive cost and 11.7%
                               cost more than 1% of growth, against 24.3% of the same genes being
                               DepMap dependencies at a matched mark. The model has roughly the right
                               amount of slack. What it does NOT have is the right ordering within the
                               tail (loop_slack P3, Spearman +0.29 inside a null of 0.43).
                               The one-line fix for any future run: re-solve mu_WT at the final bounds,
                               or compare on a relative rather than an absolute threshold.

    and against the expression confound, on the same genes: 0.6507 within abundance-matched deciles,
    and 5-fold CV lookup 0.7761 -> lookup + defined-medium FBA 0.8266 (+0.0504) with a SINGLE
    mechanistic score, where loop 3 needed three to reach +0.0552.

WHAT IT COST TO GET HERE, because the path is the finding as much as the number is.  Six attempts, and
every one failed for a different real reason:
    1  substring matching -- "Pi" matched alpha-pinene, "choline" matched 35 phosphatidylcholines
    2  minimal_medium answered ATP, because it minimises import FLUX and ATP is cheap per unit mass
    3  with ATP barred it answered prothrombin, fibrinogen, haptoglobin -- a blacklist cannot enumerate
       "large molecule", so the rule became mechanical: bar uptake above 40 carbons, from the model's
       own formulas
    4  that barred vitamin B12 too, at 63 carbons, and B12 turned out to be the ONE metabolite this
       network cannot synthesise -- the list said "cobalamin", Human-GEM says "aquacob(III)alamin"
    5  greedy supplementation could not find it from a zero-growth base, because several components
       were missing at once and no single addition helps
    6  and a menu is not a medium: 57 permitted nutrients at 1000 mmol/gDW/h each still gave 46.35/h

THE TWO FINDINGS ABOUT HUMAN-GEM WORTH KEEPING.  It ships uptake reactions for ATP, NAD+, FAD, LDL,
apoE and plasminogen, and for metabolites of up to 9,642,032 carbons; a model that can import its own
energy currency has no essential genes worth measuring. And with a culture dish made free, the network
needs exactly TWO things it cannot make for itself: gamma-tocopherol and lipoic acid. Everything else
in a dish, it can build.

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
    # Vitamin B12, and the NAME is the whole point. This list said "cobalamin"; Human-GEM calls it
    # "aquacob(III)alamin", so it never matched, and that one mismatch is why loop 3's hand-written
    # medium supported exactly zero growth and H3 came back NOT TESTABLE. A greedy probe found it:
    # re-opening this single uptake took the model from 0 to 124.8/h, making it the one metabolite here
    # the network cannot synthesise for itself. It is also 63 carbons, so the >40-carbon bar would have
    # thrown it out too -- the exemption for this list is what stops a rule aimed at Psyllium from
    # discarding a real vitamin.
    "aquacob(III)alamin",
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

# AND A MECHANICAL RULE, because the name list was whack-a-mole.
# With ATP and the lipoproteins barred, minimal_medium simply switched to prothrombin, fibrinogen,
# haptoglobin, starch and glycogenin: the objective minimises total import FLUX, so it will always reach
# for whatever molecule carries the most mass per unit flux, and no blacklist can enumerate that. The
# fix has to come from the model's own data rather than from my memory. Human-GEM carries a formula for
# 1,657 of its 1,660 exchange metabolites and the carbon counts run to 9,642,032 (Psyllium). Nothing a
# cell takes up from a culture medium has more than about forty carbons -- cholesterol is 27, palmitate
# 16 -- so uptake above that is barred, along with the three metabolites carrying no formula at all
# ("steroids", "xenobiotics", "arachidonate derivatives", which are abstractions rather than molecules).
# Anything on the physiological list is exempt, so the rule can never remove a real nutrient.
MAX_UPTAKE_CARBONS = 40

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 1904
N_SHUFFLE = 200
MU_REQUIRE = 0.030          # /h -- the growth the minimal medium must support
GATE_FEASIBLE = 0.01
GATE_LIMITING = 1.0         # a medium that permits 70/h is not a medium
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
    bio = str(M.objective.expression).split("*")[1].split(" ")[0]
    say(f"  growth with everything open: {mu_open:.2f}/h  -- a cell that can eat anything "
        f"(objective {bio})")

    import re as _re

    def carbons(met):
        f = met.formula
        if not f:
            return None
        g = _re.match(r"C(\d*)", f)
        return (int(g.group(1) or 1)) if g else 0

    barred, why = [], {"currency": 0, "too big": 0, "no formula": 0}
    for r in M.exchanges:
        met = list(r.metabolites)[0]
        nm = (met.name or "").strip().lower()
        if nm in PHYS_SET:
            continue
        c = carbons(met)
        tag = None
        if nm in NOT_IMPORTABLE_SET:
            tag = "currency"
        elif c is None:
            tag = "no formula"
        elif c > MAX_UPTAKE_CARBONS:
            tag = "too big"
        if tag:
            r.lower_bound = 0.0
            barred.append(nm)
            why[tag] += 1
    mu_barred = M.slim_optimize()
    say(f"\n  BARRING WHAT A CELL CANNOT IMPORT -- {len(barred)} exchanges closed to uptake")
    say(f"    {why['currency']} energy currency and plasma proteins by name, {why['too big']} with more "
        f"than {MAX_UPTAKE_CARBONS} carbons, {why['no formula']} with no formula at all")
    say(f"    growth {mu_open:.2f}/h -> {mu_barred:.2f}/h")
    say(f"    Human-GEM ships uptake reactions for ATP and for intact plasma proteins, and metabolites")
    say(f"    with up to 9,642,032 carbons. A cell that can import its own energy currency has no")
    say(f"    essential genes worth measuring; loop 1 was run without this bar in place.")

    # ---- what does the network actually need? ---------------------------------------------------------
    say(f"\n  asking minimal_medium for the uptakes required to reach {MU_REQUIRE}/h ...")
    t1 = time.time()
    # MAKE THE PHYSIOLOGICAL NUTRIENTS FREE, which is what fixes the gaming.
    # minimal_medium penalises every import equally, so it reaches for whatever carries the most mass
    # per unit flux: it answered ATP, then prothrombin, then maltohexaose and tripeptides, and never
    # glucose. Greedy supplementation cannot rescue that either -- from a base that does not grow, no
    # SINGLE addition raises growth above zero, because several things are missing at once, so greedy
    # stalls on the first step. The formulation that works is an LP in which uptake of a nutrient a dish
    # supplies costs NOTHING and every other uptake is penalised. Then the solver takes glucose and the
    # amino acids for free and imports something exotic only where the network genuinely cannot do
    # without it -- which is exactly the question being asked.
    from optlang.symbolics import Zero
    bio_rxn = M.reactions.get_by_id(bio)
    old_obj, old_lb = M.objective, bio_rxn.lower_bound
    bio_rxn.lower_bound = MU_REQUIRE                      # growth becomes a constraint, not a goal
    cost = Zero
    for r in M.exchanges:
        if (list(r.metabolites)[0].name or "").strip().lower() not in PHYS_SET:
            cost += r.reverse_variable                    # reverse_variable is the uptake magnitude
    M.objective = M.problem.Objective(cost, direction="min")
    sol = M.optimize()
    exotic_need = {}
    if sol.status == "optimal":
        for r in M.exchanges:
            nm = (list(r.metabolites)[0].name or "").strip().lower()
            if nm not in PHYS_SET and r.reverse_variable.primal > 1e-9:
                exotic_need[r.id] = float(r.reverse_variable.primal)
    M.objective, bio_rxn.lower_bound = old_obj, old_lb
    say(f"    LP status {sol.status}: with a culture dish free, the network still needs "
        f"{len(exotic_need)} non-physiological uptakes to reach {MU_REQUIRE}/h "
        f"({time.time()-t1:.0f}s)")
    if sol.status != "optimal":
        say("    No solution: this model cannot reach that growth rate on a dish plus anything.")
        json.dump({"test": "loop_medium", "M1": False, "infeasible": True,
                   "mu_open": float(mu_open), "mu_barred": float(mu_barred), "log": log},
                  open(OUT / "loop_medium.json", "w"), indent=2)
        return 1
    mm = pd.Series(exotic_need).sort_values(ascending=False) if exotic_need else pd.Series(dtype=float)
    supplement = list(mm.index)

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
    mu_unlimited = M.slim_optimize()
    say(f"\n  CONSTRUCTED MEDIUM: {len(keep)} open uptakes of {len(M.exchanges)} "
        f"({len(keep)/len(M.exchanges):.1%})")

    # A MENU IS NOT A MEDIUM. Restricting WHICH metabolites may be taken up left growth at 46.35/h,
    # because each of the 57 permitted ones could still be imported at 1000 mmol/gDW/h. A dish limits
    # the RATE as well as the list. One constant scales every uptake bound, bisected so growth lands on
    # a 23 h doubling -- fitted to a growth rate and never to the labels, the same discipline kappa was
    # fitted under in loop 1. It is uniform across nutrients, which is crude and is said so here.
    lo, hi = 1e-6, 1000.0
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        for rid in keep:
            M.reactions.get_by_id(rid).lower_bound = -mid
        v = M.slim_optimize()
        if v is None or not np.isfinite(v) or v < MU_REQUIRE:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0001:
            break
    UPTAKE = float(np.sqrt(lo * hi))
    for rid in keep:
        M.reactions.get_by_id(rid).lower_bound = -UPTAKE
    mu_med = M.slim_optimize()
    say(f"    composition alone (each at 1000 mmol/gDW/h): {mu_unlimited:.2f}/h -- a menu, not a medium")
    say(f"    uptake rate capped at {UPTAKE:.4g} mmol/gDW/h per component, bisected to a 23 h doubling")
    say(f"    growth: {mu_med:.4f}/h   (open medium {mu_open:.2f}/h -- a "
        f"{mu_open/max(mu_med,1e-9):.0f}x reduction, achieved by nutrients rather than by squeezing "
        f"enzymes)")
    # A medium has to be feasible AND limiting. Two earlier attempts produced media that supported
    # 86.26/h and 69.60/h -- nominally closed, functionally open -- and a retest on either would have
    # looked like a result while measuring nothing. The upper bound is what makes M1 able to catch that.
    m1 = bool(np.isfinite(mu_med) and GATE_FEASIBLE <= mu_med <= GATE_LIMITING)
    say(f"    M1 FEASIBLE AND LIMITING ({GATE_FEASIBLE} <= mu <= {GATE_LIMITING}/h)   "
        f"{'PASS' if m1 else 'FAIL'}")
    if not m1:
        why_fail = "supports no growth" if mu_med < GATE_FEASIBLE else "does not constrain growth"
        say(f"    The medium {why_fail}, so the retest is NOT TESTABLE. Reported as such rather than")
        say("    dressed up as a negative -- a medium that is not limiting measures nothing.")
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
               "uptake_cap": UPTAKE, "mu_composition_only": float(mu_unlimited),
               "supplement": supplement,
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
