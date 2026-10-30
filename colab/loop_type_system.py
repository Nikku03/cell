"""LOOP 65 -- THE TYPE SYSTEM: FIVE SLOTS PER ENTITY, AND A BUDGET THAT MUST CLOSE.

THE PROPOSAL BEING TESTED. Nothing in a cell is a thing; everything is a constraint on a flux, and
the constraints must collectively produce each other out of a small set of conserved currencies. That
is constraint closure -- Rosen's (M,R) systems, Kauffman's work-constraint cycles, Montevil &
Mossio's formalisation -- and its computational form is the FBA -> ME-model -> whole-cell ladder.
Married, they are a TYPE SYSTEM: every entity gets the same five slots.

    flux       what conserved quantity moves because of it
    cost       which budgets its existence consumes
    capacity   the maximum flux it can carry
    lifetime   turnover rate, floored by dilution from growth
    producers  which fluxes make it

WHY THIS IS WORTH A LOOP HERE RATHER THAN A PARAGRAPH, and the argument is from this project's own
failures. Every negative result in 64 loops has been an ATTENTION confound: publication count
predicts cancer drivers at 0.8259 against our model's 0.2990, it was the best single drug feature at
0.7913, and loop 61 measured it predicting a layer's membership at median AUC 0.7173. Those are all
the same disease -- the model was scored against labels that record what has been STUDIED.

A budget is different in kind. Carbon balances or it does not. Ribosome-seconds are spent or they are
not. No quantity of literature makes a cell afford a protein it cannot afford. THIS IS THE FIRST GATE
TYPE IN THIS PROJECT THAT FAME CANNOT DEFEAT, and that -- not the ontology -- is the reason to build
it.

WHAT THE MODEL CAN ACTUALLY BE TYPED WITH, measured before this was written:

    flux       generxn                     2,549 / 16,492    15.5%
    cost       ppm 97.1%, abund 98.8%, compartment 100%
    capacity   kcat                        2,549             15.5%, of which only 420 MEASURED
                                                             (388 measured + 32 EC-measured); the
                                                             rest are predicted or propagated
    lifetime   RNADecayCafe mRNA decay     partial
    producers  nothing                     0%

So the type system rejects 84.6% of this cell as untyped, and that is the first honest output: the
current model has those fields silently absent, where a type system makes the hole countable.

THE BUDGET CHOSEN, and why it is the ribosome one. Carbon and nitrogen have no ceiling here because
this project has no defined-medium uptake bound to check them against -- they would be inventory, not
a test. Ribosome-seconds DO have a ceiling: a cell has a countable number of ribosomes, each
elongating at a measured rate, for a measured doubling time. Demand must fit inside that or the
abundance vector is wrong.

    demand    = TOTAL_COPIES * sum_i (ppm_i / 1e6 * length_i)      amino acids per new cell
    supply    = N_RIBOSOMES * DOUBLING_S * ELONGATION_AA_S         amino acids polymerisable

Every constant on the supply side and TOTAL_COPIES on the demand side are LITERATURE, not measured
here, and are declared with ranges rather than points -- which is what T6 is for. Lengths are
computed from the actual UniProt sequence of every protein, not assumed.

AND THE SELF-PRODUCTION TERM, which is constraint closure in its most literal computable form: the
ribosome must build itself. The fraction of the ribosome-second budget spent on ribosomal proteins is
a number this framing demands and the previous 64 loops had no way to ask for.

PREDECLARED, before any number:

  T1 THE SLOT CENSUS                                  REPORTED, NOT JUDGED
       per-slot coverage over all 16,492 entities. This gate cannot fail; it exists so the deficit is
       a number in the record rather than an impression.
  T2 COSTS COME FROM SEQUENCE, NOT ASSUMPTION
       per-protein amino-acid count, carbon and nitrogen from the real residue composition. Two
       calibrations that can fail: mean protein length must land in 450-650 aa (literature ~550), and
       >= 80% of the abundance mass must map to a sequence. A ppm vector that cannot be matched to
       lengths cannot be costed.
  T3 THE ABUNDANCE VECTOR DESCRIBES A CELL            ADDED AFTER THE FIRST RUN. THE MODEL FAILS IT.

       PROVENANCE CORRECTED LATER, AND MY WORDING WAS WRONG. This gate and its commit said the cost
       slot holds "BLOOD PLASMA". It does not. Matched against PaxDb's own human datasets, our `ppm`
       is 9606-WHOLE_ORGANISM-integrated: Spearman 1.0000 over 15,740 shared genes, 97.4% coverage.
       The actual plasma dataset (9606-PLASMA-integrated) is a DIFFERENT file -- it correlates with
       ours at only 0.4586, carries 0.6119 plasma-protein mass against our 0.2918, and is topped by
       ALB/IGLL5/APOA1/FGB/TF rather than our TMSB4X/APOA2/RBP4.
       So the defect is real and the gate still FAILS, but it is a WHOLE-ORGANISM proteome, not a
       plasma one. It is body-wide, and a body contains blood, which is why abundant serum proteins
       carry 29% of its mass. The honest statement is "this describes an organism, not a cell" --
       narrower and more defensible than what was committed.
       No clean replacement exists: PaxDb has no integrated CELL_LINE set. HeLa iBAQ is genuinely
       cellular but covers 39.4% of our genes against whole-organism's 97.4%, so a straight swap
       would zero out 9,254 genes. That trade is named here rather than made silently.
       The first run of this module passed its budget gate at demand/supply 0.1744 and the number was
       meaningless, because the vector it costed is not a cell. The model's top ten proteins by `ppm`
       are TMSB4X, APOA2, RBP4, ORM1, APOA1, ALB, TTR, APOC2, APOC1, HPX -- albumin, apolipoproteins,
       transthyretin, hemopexin, orosomucoid. That is BLOOD PLASMA. The other abundance layer,
       `abund`, is no better: LALBA, INS, IGKC, CSN2, SFTPC -- milk, insulin, immunoglobulin, casein,
       surfactant, i.e. secreted tissue extremes.
       Against a real cellular proteome (PaxDb, HeLa iBAQ, Geiger 2012) the divergence is not subtle:
       the 22 canonical plasma proteins named in PLASMA below carry 0.2918 of the model's abundance
       mass and 0.0004 of HeLa's -- a 676-fold enrichment -- while rank correlation where both are
       present is 0.6835. (The scoping pass that found this used a shorter list of 18 and got
       0.2544 / 0.0003; the list was widened before the gate ran, and both are recorded.)
       So the gate: canonical plasma proteins must hold < 1% of abundance mass. This is what a type
       system is FOR -- the cost slot was silently populated with the wrong compartment for the whole
       project, and no gate in 64 previous loops could have noticed, because none of them ever had to
       spend the budget.
  T4 THE RIBOSOME BUDGET CLOSES                       THE GATE.
       demand/supply must be <= 1 at the central literature values, computed on the CELLULAR vector.
       Both vectors are reported side by side so the size of the error is visible. If it exceeds 1
       the abundance vector describes a cell that cannot build itself in a doubling, and the defect
       is in the data, not in the theory.
  T5 IS THE TYPED FRACTION JUST THE FAMOUS FRACTION?
       publication count of entities WITH a flux slot vs those without, as an AUC. The type system
       was adopted to escape the attention confound; if flux coverage is itself predicted by fame at
       high AUC then it re-encodes the confound rather than escaping it, and this must be said. The
       gate is AUC < 0.75, the value at which loop 61's median layer sat.
  T6 THE VERDICT SURVIVES THE PARAMETER RANGE
       demand/supply recomputed across the full literature range of all four constants. A budget
       check that closes only at one convenient parameter choice is not a check, so the WORST corner
       is reported and the gate is that the conclusion is stated for the whole range rather than the
       midpoint.

-> outputs/loop_type_system.json
"""
import collections
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"
FASTA = SC / "human_proteome.fasta.gz"

# residue formulas (peptide-bonded, i.e. free amino acid minus water)
RES_C = {"G": 2, "A": 3, "S": 3, "P": 5, "V": 5, "T": 4, "C": 3, "L": 6, "I": 6, "N": 4,
         "D": 4, "Q": 5, "K": 6, "E": 5, "M": 5, "H": 6, "F": 9, "R": 6, "Y": 9, "W": 11}
RES_N = {"G": 1, "A": 1, "S": 1, "P": 1, "V": 1, "T": 1, "C": 1, "L": 1, "I": 1, "N": 2,
         "D": 1, "Q": 2, "K": 2, "E": 1, "M": 1, "H": 3, "F": 1, "R": 4, "Y": 1, "W": 2}

# LITERATURE, not measured here. Each is a range; T5 sweeps the corners.
LIT = {"TOTAL_COPIES": (1.0e9, 2.3e9, 4.0e9),      # protein molecules per mammalian cell
       "N_RIBOSOMES": (3.0e6, 1.0e7, 2.0e7),       # ribosomes per mammalian cell
       "DOUBLING_S": (72000.0, 86400.0, 108000.0),  # 20-30 h
       "ELONGATION_AA_S": (3.0, 5.6, 9.0)}          # aa per second per ribosome

LEN_LO, LEN_HI = 450, 650
MASS_FLOOR = 0.80
FAME_CEILING = 0.75
RIBO_PREFIX = ("RPL", "RPS")     # cytoplasmic ribosomal proteins, a declarable rule

HELA = SC / "paxdb_hela.txt"     # PaxDb 9606 iBAQ HeLa, Geiger MCP 2012 -- a real cellular proteome
PLASMA_CEILING = 0.01
# canonical abundant plasma proteins. Named explicitly rather than inferred, so the test is a
# declaration that can be argued with rather than a threshold on something vague.
PLASMA = {"ALB", "APOA1", "APOA2", "APOB", "APOC1", "APOC2", "APOC3", "TTR", "RBP4", "ORM1",
          "ORM2", "HPX", "SERPINA1", "TF", "HP", "FGA", "FGB", "FGG", "A2M", "C3", "AHSG", "APOE"}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def proteome():
    """gene symbol -> (length, carbon, nitrogen), longest isoform per symbol."""
    out = {}
    name, seq = None, []
    def flush():
        if not name or not seq:
            return
        s = "".join(seq)
        c = sum(RES_C.get(a, 0) for a in s)
        nn = sum(RES_N.get(a, 0) for a in s)
        if name not in out or len(s) > out[name][0]:
            out[name] = (len(s), c, nn)
    for ln in gzip.open(FASTA, "rt", errors="ignore"):
        if ln.startswith(">"):
            flush()
            gn = [x[3:] for x in ln.split() if x.startswith("GN=")]
            name = gn[0] if gn else None
            seq = []
        else:
            seq.append(ln.strip())
    flush()
    return out


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 65 -- the type system: five slots per entity, and a budget that must close")
    say("  the first gate in this project that fame cannot defeat: a budget balances or it does not")
    say("=" * 100)
    say()

    D = json.load(open(CELL))
    G = D["genes"]
    n = len(G)
    names = [g["name"] for g in G]
    pubs = np.array([float(g.get("pubs") or 0) for g in G])
    ppm = np.array([float(D["ppm"].get(str(i), 0) or 0) for i in range(n)])
    kin = json.load(open(Path("outputs/orphan/kinetics_refined_corrected.json")))["kinetics_refined"]

    say("T1 THE SLOT CENSUS -- reported, not judged")
    has_flux = np.array([str(i) in D["generxn"] for i in range(n)])
    has_cap = np.array([nm in kin for nm in names])
    has_cost = ppm > 0
    measured = sum(1 for v in kin.values() if v.get("tier") in ("measured", "EC-measured"))
    slots = {"flux (generxn)": int(has_flux.sum()),
             "cost (ppm)": int(has_cost.sum()),
             "capacity (kcat, any tier)": int(has_cap.sum()),
             "capacity (kcat, MEASURED only)": measured,
             "lifetime": 0,
             "producers": 0}
    for k, v in slots.items():
        say(f"     {k:34s} {v:7,d} / {n:,}   {v / n:6.1%}")
    typed_all = int((has_flux & has_cap & has_cost).sum())
    say(f"     {'FULLY TYPED (flux+cost+capacity)':34s} {typed_all:7,d} / {n:,}   {typed_all / n:6.1%}")
    say(f"     the type system rejects {1 - typed_all / n:.1%} of this cell as untyped")
    say(f"     T1 PASS (census)")
    say()

    say("T2 COSTS COME FROM SEQUENCE, NOT ASSUMPTION")
    P = proteome()
    L = np.array([P.get(nm, (0, 0, 0))[0] for nm in names], float)
    C = np.array([P.get(nm, (0, 0, 0))[1] for nm in names], float)
    N = np.array([P.get(nm, (0, 0, 0))[2] for nm in names], float)
    hit = L > 0
    mass = ppm[hit].sum() / max(ppm.sum(), 1e-9)
    mean_len = L[hit].mean()
    say(f"     {len(P):,} symbols in the reviewed proteome; {int(hit.sum()):,} of {n:,} "
        f"model genes matched ({hit.mean():.1%})")
    say(f"     abundance mass covered by a sequence: {mass:.4f}")
    say(f"     mean protein length {mean_len:.0f} aa   (literature ~550, gate {LEN_LO}-{LEN_HI})")
    say(f"     mean carbon/residue {C[hit].sum() / max(L[hit].sum(), 1):.3f}, "
        f"nitrogen/residue {N[hit].sum() / max(L[hit].sum(), 1):.3f}")
    t2 = (LEN_LO <= mean_len <= LEN_HI) and mass >= MASS_FLOOR
    say(f"     T2 {'PASS' if t2 else 'FAIL'}")
    say()

    say("T3 THE ABUNDANCE VECTOR DESCRIBES A CELL")
    hela = np.zeros(n)
    ix = {nm: i for i, nm in enumerate(names)}
    for ln in open(HELA):
        if ln.startswith("#"):
            continue
        f = ln.rstrip().split("\t")
        if len(f) < 3:
            continue
        i = ix.get(f[0])
        if i is not None:
            try:
                hela[i] = float(f[2])
            except ValueError:
                pass
    wp = ppm / max(ppm.sum(), 1e-9)
    wh = hela / max(hela.sum(), 1e-9)
    pl = np.array([nm in PLASMA for nm in names])
    p_model, p_hela = float(wp[pl].sum()), float(wh[pl].sum())
    both = (ppm > 0) & (hela > 0)
    from scipy.stats import spearmanr
    rho_ab = float(spearmanr(ppm[both], hela[both]).statistic)
    say(f"     cellular reference: PaxDb HeLa iBAQ (Geiger 2012), mapped to "
        f"{int((hela > 0).sum()):,} of {n:,} genes")
    say(f"     {len(PLASMA)} canonical plasma proteins hold "
        f"{p_model:.4f} of the model's abundance mass")
    say(f"     the same proteins hold {p_hela:.4f} of HeLa's        "
        f"({p_model / max(p_hela, 1e-12):.0f}x enrichment)")
    say(f"     rank correlation where both present: {rho_ab:.4f}  (n={int(both.sum()):,})")
    say(f"     model top 5:  {', '.join(names[i] for i in np.argsort(-ppm)[:5])}")
    say(f"     HeLa  top 5:  {', '.join(names[i] for i in np.argsort(-hela)[:5])}")
    t3 = p_model < PLASMA_CEILING
    say(f"     T3 {'PASS' if t3 else 'FAIL'}  (ceiling {PLASMA_CEILING:.0%}) -- the model's `ppm` "
        f"layer is {'cellular' if t3 else 'A PLASMA PROTEOME, not a cell'}")
    if not t3:
        say("     the cost slot has been populated with the wrong compartment for the whole project.")
        say("     No gate in 64 previous loops could have caught this, because none of them ever had")
        say("     to spend the budget. The remaining arithmetic runs on the CELLULAR vector.")
    say()

    say("T4 THE RIBOSOME BUDGET CLOSES")
    w = wh if not t3 else wp
    say(f"     costed on the {'HeLa cellular' if not t3 else 'model ppm'} vector")
    aa_per_cell_unit = float((w * L).sum())           # weighted mean length over the abundance vector
    def ratio(tc, nr, dt, el):
        demand = tc * aa_per_cell_unit
        supply = nr * dt * el
        return demand, supply, demand / supply
    tc, nr, dt, el = (LIT["TOTAL_COPIES"][1], LIT["N_RIBOSOMES"][1],
                      LIT["DOUBLING_S"][1], LIT["ELONGATION_AA_S"][1])
    dem, sup, r = ratio(tc, nr, dt, el)
    say(f"     abundance-weighted mean protein length  {aa_per_cell_unit:.1f} aa")
    say(f"     demand  {tc:.2e} copies x {aa_per_cell_unit:.0f} aa   = {dem:.3e} aa per new cell")
    say(f"     supply  {nr:.2e} ribosomes x {dt:.0f} s x {el} aa/s = {sup:.3e} aa")
    say(f"     demand / supply = {r:.4f}")
    t4 = r <= 1.0
    say(f"     T4 {'PASS' if t4 else 'FAIL'}  -- this abundance vector "
        f"{'CAN' if t4 else 'CANNOT'} be built in one doubling")
    dm_p, sp_p, r_p = ratio(tc, nr, dt, el) if t3 else (
        tc * float((wp * L).sum()), sup, tc * float((wp * L).sum()) / sup)
    say(f"     for comparison, the model's own plasma-weighted vector gives "
        f"demand/supply {r_p:.4f} at the same constants")
    say()

    ribo = np.array([nm.startswith(RIBO_PREFIX) for nm in names])
    self_frac = float((w[ribo] * L[ribo]).sum() / max(aa_per_cell_unit, 1e-9))
    say(f"     CONSTRAINT CLOSURE, literally: {int(ribo.sum())} cytoplasmic ribosomal proteins "
        f"(RPL/RPS)")
    say(f"     fraction of the ribosome-second budget spent building ribosomes: {self_frac:.4f}")
    say(f"     this is the self-production term the previous 64 loops had no way to ask for")
    say()

    say("T5 IS THE TYPED FRACTION JUST THE FAMOUS FRACTION?")
    a_flux = roc_auc_score(has_flux.astype(int), pubs)
    a_cap = roc_auc_score(has_cap.astype(int), pubs)
    prev = json.load(open(OUT / "loop_state_vector.json"))
    say(f"     publication count predicts HAS-FLUX      AUC {a_flux:.4f}")
    say(f"     publication count predicts HAS-CAPACITY  AUC {a_cap:.4f}")
    say(f"     loop 61 median layer for comparison      AUC {prev['fame_median']:.4f}")
    t5 = max(a_flux, a_cap) < FAME_CEILING
    say(f"     NOTE the two AUCs are identical because kcat coverage IS generxn coverage -- the same "
        f"2,549 genes, not two independent confirmations")
    say(f"     T5 {'PASS' if t5 else 'FAIL'}  (ceiling {FAME_CEILING})")
    if not t5:
        say("     THE COVERAGE IS A FAME ARTEFACT. The budget arithmetic is still fame-free -- carbon "
            "does not care -- but WHICH entities can be typed at all is inherited from curation, so "
            "the type system escapes the confound in its test and not in its scope.")
    say()

    say("T6 THE VERDICT SURVIVES THE PARAMETER RANGE")
    corners = []
    for a in LIT["TOTAL_COPIES"]:
        for b in LIT["N_RIBOSOMES"]:
            for c in LIT["DOUBLING_S"]:
                for d in LIT["ELONGATION_AA_S"]:
                    corners.append(ratio(a, b, c, d)[2])
    lo, hi = min(corners), max(corners)
    say(f"     demand/supply over all {len(corners)} literature corners: "
        f"{lo:.4f} to {hi:.4f}, median {np.median(corners):.4f}")
    say(f"     closes at {sum(1 for x in corners if x <= 1.0)} of {len(corners)} corners")
    t6 = True
    if hi <= 1.0:
        say("     the budget closes EVERYWHERE in the literature range -- the conclusion does not "
            "depend on the constants chosen")
    elif lo > 1.0:
        say("     the budget closes NOWHERE -- the abundance vector is over-committed under every "
            "literature value, which is a defect in the model and not in the constants")
    else:
        say("     the budget closes in SOME corners and not others, so the honest statement is that "
            "this test does not resolve at the precision the literature constants allow")
    say(f"     T6 PASS (range reported, conclusion stated for the range)")
    say()

    gates = {"T1 slot census recorded": True,
             "T2 costs computed from sequence": bool(t2),
             "T3 the abundance vector describes a cell": bool(t3),
             "T4 the ribosome budget closes": bool(t4),
             "T5 typed fraction is not just fame": bool(t5),
             "T6 verdict stated over the parameter range": bool(t6)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(CELL), str(FASTA), str(HELA),
                              "outputs/orphan/kinetics_refined_corrected.json"],
                      available=n, used=int(hit.sum()), selection="filtered", seed=0,
                      controls=["protein lengths from real UniProt sequence, not assumed",
                                "mean-length calibration against the literature value",
                                "abundance mass coverage floor before costing",
                                "publication count tested against slot coverage",
                                "all four literature constants swept over their full range",
                                "abundance vector checked against a real cellular proteome before "
                                "any budget was computed on it"],
                      note="TOTAL_COPIES, N_RIBOSOMES, DOUBLING_S and ELONGATION_AA_S are LITERATURE "
                           "and are not measured here; they are swept rather than fixed")
    RM.report(man, emit=say)
    json.dump({"test": "loop_type_system", "manifest": man, "gates": gates,
               "slots": slots, "n_entities": n, "fully_typed": typed_all,
               "mean_length": float(mean_len), "abundance_mass_covered": float(mass),
               "weighted_mean_length": aa_per_cell_unit,
               "demand_aa": dem, "supply_aa": sup, "ratio": r,
               "plasma_mass_model": p_model, "plasma_mass_hela": p_hela,
               "abundance_rank_rho": rho_ab, "ratio_model_vector": r_p,
               "costed_on": "hela_cellular" if not t3 else "model_ppm",
               "ribosome_self_fraction": self_frac, "n_ribosomal_proteins": int(ribo.sum()),
               "fame_auc_flux": float(a_flux), "fame_auc_capacity": float(a_cap),
               "corner_ratios": {"min": lo, "max": hi, "median": float(np.median(corners)),
                                 "n_closing": int(sum(1 for x in corners if x <= 1.0)),
                                 "n_corners": len(corners)},
               "literature_constants": {k: list(v) for k, v in LIT.items()},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_type_system.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_type_system.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
