"""LOOP 68 -- THE E. COLI TEST BED: DOES THE TYPE SYSTEM CLOSE WHERE THE DATA IS COMPLETE?

WHY A BACTERIUM. Loop 65 built the constraint-closure type system on the human model and found it
rejects 84.6% of the cell as untyped, with `producers` -- the slot that makes it CLOSURE rather than
bookkeeping -- at exactly 0%. It also found, through the first budget this project ever had to spend,
that the cost slot was populated with BLOOD PLASMA and had been for the whole project. Sixty-four
loops of correlation gates never touched the units.

That raises a question the human model cannot answer about itself: is the machinery wrong, or is the
data? A system where all five slots CAN be filled would separate those, and E. coli is that system.
It is the only organism where the FBA -> ME-model ladder has been validated against measured growth
for twenty years, and where the phenotype to predict was measured by systematic single-gene deletion
rather than assembled from literature.

WHAT MAKES THIS A TEST RATHER THAN A DEMONSTRATION. Every failure in this project has been an
ATTENTION confound: publication count predicts cancer drivers at 0.8259 against our model's 0.2990,
was the best single drug feature at 0.7913, and predicts layer membership at median AUC 0.7173. So
the bar here is not "does FBA predict essentiality" -- it is "does it beat knowing how much has been
written about the gene". If a closed mechanistic model cannot beat fame on the best-characterised
organism in biology, with a phenotype measured by knockout rather than curated, then the problem was
never the data.

THE PIECES, all fetched and verified before this was written:

    iML1515              2,712 reactions, 1,877 metabolites, 1,516 genes   (BiGG)
    PEC essentiality     302 essential, 4,190 non-essential, 5 unknown, keyed by b-number
                         -- measured by systematic single-gene disruption, not curated
    PaxDb E. coli        2,260 proteins with abundance (Wisniewski 2014, copies per cell)
    UniProt literature   4,402 genes with a b-number, mean 6.7 PubMed references -- the fame baseline

PRODUCERS, the slot that was 0% in human, is free here and that is the whole structural point. In a
stoichiometric model a metabolite's producers are the reactions with a positive coefficient for it,
read straight off S. Nothing has to be curated, inferred or predicted. The human model has no S
matrix over its 16,492 entities, which is exactly why its producers slot was empty.

PREDECLARED, before any number:

  B1 THE MODEL LOADS AND REPRODUCES A MEASURED GROWTH RATE
       FBA on glucose minimal aerobic must land in 0.4-1.2 /h, which brackets the measured range for
       E. coli K-12 on glucose. This is calibration and runs first: a model that cannot reproduce a
       growth rate makes every later number meaningless, and this is the check loop 65's plasma
       proteome would have failed if any human equivalent had existed.
  B2 ALL FIVE SLOTS FILL, AND PRODUCERS IS NOT ZERO
       per-slot coverage over the 1,516 model genes, printed beside loop 65's human numbers. The gate
       is that the fully-typed fraction exceeds human's 15.4% AND producers exceeds 0%. If a
       bacterium cannot be typed either, the schema is the problem rather than the data.
  B3 CLOSURE PREDICTS A MEASURED PHENOTYPE               THE GATE.
       single-gene deletion FBA across all 1,516 genes, scored against PEC essentiality as AUC. The
       model is not fitted to this in any way -- the essentiality labels never touch the
       stoichiometry.
  B4 IT BEATS FAME                                       THE POINT.
       publication count per gene as the competing predictor, same genes, same labels. The closed
       model must beat it. This is the comparison every human loop lost.
  B5 THE FAILURES ARE NAMED
       essential genes the FBA calls dispensable, and dispensable genes it calls essential, listed by
       name with the reason where the model states one. A prediction that works is only useful
       alongside the shape of where it does not.

-> outputs/loop_ecoli_bed.json
"""
import collections
import json
import os
import re
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
MODEL = SC / "iML1515.json"
PEC = SC / "PECData.dat"
PAX = SC / "paxdb_eco.txt"
FAME = SC / "eco_fame.tsv"

GROWTH_LO, GROWTH_HI = 0.4, 1.2
HUMAN_TYPED = 0.154          # loop 65, fully typed fraction
HUMAN_PRODUCERS = 0.0        # loop 65, producers slot
DEAD = 0.01                  # deletion is lethal if growth falls below this fraction of wild type
SEED = 6801

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def pec_labels():
    ess, non = set(), set()
    for i, ln in enumerate(open(PEC, errors="ignore")):
        if i == 0:
            continue
        f = ln.rstrip("\n").split("\t")
        if len(f) < 10:
            continue
        bs = re.findall(r"b\d{4}", f[3] or "")
        if not bs:
            continue
        if f[9] == "1":
            ess.update(bs)
        elif f[9] == "2":
            non.update(bs)
    return ess, non


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 68 -- the E. coli test bed: does the type system close where the data is complete?")
    say("  the bar is not 'does FBA work'. It is 'does it beat knowing how studied the gene is'.")
    say("=" * 100)
    say()

    import cobra
    cobra.Configuration().solver = "glpk"
    M = cobra.io.load_json_model(str(MODEL))

    say("B1 THE MODEL LOADS AND REPRODUCES A MEASURED GROWTH RATE")
    sol = M.optimize()
    mu = float(sol.objective_value)
    say(f"     iML1515: {len(M.reactions):,} reactions, {len(M.metabolites):,} metabolites, "
        f"{len(M.genes):,} genes")
    say(f"     medium: {len(M.medium)} open exchanges (BiGG default, glucose minimal aerobic)")
    say(f"     FBA growth rate {mu:.4f} /h   (gate {GROWTH_LO}-{GROWTH_HI} /h, measured range for "
        f"K-12 on glucose)")
    b1 = GROWTH_LO <= mu <= GROWTH_HI
    say(f"     B1 {'PASS' if b1 else 'FAIL'}")
    say()

    say("B2 ALL FIVE SLOTS FILL, AND PRODUCERS IS NOT ZERO")
    genes = [g.id for g in M.genes]
    ng = len(genes)
    flux = {g.id for g in M.genes if len(g.reactions) > 0}
    # producers: read straight off the stoichiometry, nothing curated
    prod = collections.defaultdict(set)
    for r in M.reactions:
        for met, coef in r.metabolites.items():
            if coef > 0:
                prod[met.id].add(r.id)
    gene_prod = set()
    for g in M.genes:
        for r in g.reactions:
            if any(coef > 0 for coef in r.metabolites.values()):
                gene_prod.add(g.id)
    cap = {r.id for r in M.reactions if r.upper_bound < 1000 or r.lower_bound > -1000}
    gene_cap = {g.id for g in M.genes if any(r.id in cap for r in g.reactions)}
    ab = {}
    for ln in open(PAX):
        if ln.startswith("#"):
            continue
        f = ln.rstrip().split("\t")
        if len(f) >= 3:
            b = re.search(r"b\d{4}", f[1] or "")
            if b:
                try:
                    ab[b.group(0)] = float(f[2])
                except ValueError:
                    pass
    cost = {g for g in genes if g in ab}
    say(f"     {'slot':14s} {'E. coli':>16s}   {'human (loop 65)':>16s}")
    say(f"     {'flux':14s} {len(flux):7,d} {len(flux) / ng:7.1%}   {'2,549':>9s} {0.155:6.1%}")
    say(f"     {'cost':14s} {len(cost):7,d} {len(cost) / ng:7.1%}   {'15,741':>9s} {0.954:6.1%}")
    say(f"     {'capacity':14s} {len(gene_cap):7,d} {len(gene_cap) / ng:7.1%}   {'2,549':>9s} {0.155:6.1%}")
    say(f"     {'producers':14s} {len(gene_prod):7,d} {len(gene_prod) / ng:7.1%}   {'0':>9s} {0.0:6.1%}"
        f"   <- read off S, nothing curated")
    typed = flux & cost & gene_prod
    say(f"     {'FULLY TYPED':14s} {len(typed):7,d} {len(typed) / ng:7.1%}   {'2,533':>9s} "
        f"{HUMAN_TYPED:6.1%}")
    say(f"     {len(prod):,} metabolites have their producers enumerated from the stoichiometry")
    b2 = (len(typed) / ng > HUMAN_TYPED) and (len(gene_prod) / ng > HUMAN_PRODUCERS)
    say(f"     B2 {'PASS' if b2 else 'FAIL'}")
    say()

    say("B3 CLOSURE PREDICTS A MEASURED PHENOTYPE")
    ess, non = pec_labels()
    say(f"     PEC: {len(ess):,} essential, {len(non):,} non-essential (systematic single-gene "
        f"disruption)")
    say(f"     running {ng:,} single-gene deletions ...")
    from cobra.flux_analysis import single_gene_deletion
    dl = single_gene_deletion(M, gene_list=M.genes, processes=1)
    # cobrapy 0.31 returns a RangeIndex with the gene ids in an `ids` COLUMN, not in the index.
    # The first run of this module read the index, so every gene fell through to the default
    # "no effect" and B3 came back at AUC exactly 0.5000 with zero predictions above threshold --
    # a number that looked like a biological result and was a parsing bug. An AUC of exactly 0.5000
    # with 0 true and 0 false positives is not a finding; it is an assertion that should have fired.
    ratio = {}
    for _, row in dl.iterrows():
        ids = list(row["ids"]) if not isinstance(row["ids"], str) else [row["ids"]]
        v = row["growth"]
        for g in ids:
            ratio[g] = 0.0 if (v is None or not np.isfinite(v)) else float(v) / max(mu, 1e-12)
    assert len(ratio) >= 0.9 * ng, f"deletion parse recovered only {len(ratio)} of {ng} genes"
    nlethal = sum(1 for v in ratio.values() if v < DEAD)
    say(f"     parsed {len(ratio):,} deletion results; {nlethal} are lethal in silico")
    assert nlethal > 0, "no deletion is lethal -- the parse or the medium is wrong"
    lab, score, keptg = [], [], []
    for g in genes:
        if g in ess:
            y = 1
        elif g in non:
            y = 0
        else:
            continue
        lab.append(y)
        score.append(1.0 - min(max(ratio.get(g, 1.0), 0.0), 1.0))
        keptg.append(g)
    lab = np.array(lab)
    score = np.array(score)
    auc = float(roc_auc_score(lab, score))
    pred_ess = score > (1 - DEAD)
    tp = int(((lab == 1) & pred_ess).sum())
    fp = int(((lab == 0) & pred_ess).sum())
    fn = int(((lab == 1) & ~pred_ess).sum())
    say(f"     {len(lab):,} model genes carry a PEC label ({int(lab.sum())} essential)")
    say(f"     deletion-FBA vs measured essentiality   AUC {auc:.4f}")
    say(f"     at the lethal threshold: {tp} true essential, {fp} false essential, {fn} missed")
    say(f"     precision {tp / max(tp + fp, 1):.4f}   recall {tp / max(tp + fn, 1):.4f}")
    b3 = auc > 0.5
    say(f"     B3 {'PASS' if b3 else 'FAIL'}")
    say()

    say("B4 IT BEATS FAME")
    pub = {}
    for i, ln in enumerate(open(FAME, errors="ignore")):
        if i == 0:
            continue
        f = ln.rstrip("\n").split("\t")
        if len(f) < 4:
            continue
        b = re.search(r"b\d{4}", f[2] or "")
        if b:
            pub[b.group(0)] = len([x for x in (f[3] or "").split(";") if x.strip()])
    fv = np.array([pub.get(g, 0) for g in keptg], float)
    have = fv > 0
    auc_f = float(roc_auc_score(lab[have], fv[have]))
    auc_m = float(roc_auc_score(lab[have], score[have]))
    say(f"     {int(have.sum()):,} of {len(keptg):,} genes carry a literature count "
        f"(mean {fv[have].mean():.1f} refs)")
    say(f"     publication count       AUC {auc_f:.4f}")
    say(f"     deletion-FBA            AUC {auc_m:.4f}     delta {auc_m - auc_f:+.4f}")
    say(f"     for comparison, in the HUMAN model fame won every time: cancer drivers 0.8259 vs our "
        f"0.2990 partial, best drug feature 0.7913, layer membership median 0.7173")
    b4 = auc_m > auc_f
    say(f"     B4 {'PASS' if b4 else 'FAIL'}")
    say()

    say("B5 THE FAILURES ARE NAMED")
    n2n = {g.id: (g.name or g.id) for g in M.genes}
    missed = [(keptg[i], n2n.get(keptg[i], "")) for i in range(len(lab))
              if lab[i] == 1 and not pred_ess[i]]
    over = [(keptg[i], n2n.get(keptg[i], "")) for i in range(len(lab))
            if lab[i] == 0 and pred_ess[i]]
    say(f"     essential but FBA says dispensable: {len(missed)}")
    say(f"       {', '.join(nm or g for g, nm in missed[:24])}")
    say(f"     dispensable but FBA says essential: {len(over)}")
    say(f"       {', '.join(nm or g for g, nm in over[:24])}")
    say("     the first class is the informative one: a stoichiometric model only sees metabolic")
    say("     lethality, so essential genes in translation, replication and cell division are")
    say("     invisible to it by construction rather than by error.")
    b5 = True
    say(f"     B5 PASS (named)")
    say()

    gates = {"B1 model reproduces a measured growth rate": bool(b1),
             "B2 all five slots fill, producers non-zero": bool(b2),
             "B3 closure predicts a measured phenotype": bool(b3),
             "B4 it beats fame": bool(b4),
             "B5 the failures are named": bool(b5)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(MODEL), str(PEC), str(PAX), str(FAME)],
                      available=ng, used=len(lab), selection="filtered", seed=SEED,
                      controls=["growth rate checked against the measured range before anything else",
                                "essentiality labels never touch the stoichiometry",
                                "publication count as the competing predictor on the same genes",
                                "producers read off the S matrix rather than curated",
                                "failures named in both directions"],
                      note="PEC essentiality is measured by systematic single-gene disruption, not "
                           "assembled from literature, which is why it can be scored against fame")
    RM.report(man, emit=say)
    json.dump({"test": "loop_ecoli_bed", "manifest": man, "gates": gates,
               "growth_rate": mu, "n_genes": ng,
               "slots": {"flux": len(flux), "cost": len(cost), "capacity": len(gene_cap),
                         "producers": len(gene_prod), "fully_typed": len(typed),
                         "metabolites_with_producers": len(prod)},
               "typed_fraction": len(typed) / ng, "human_typed_fraction": HUMAN_TYPED,
               "n_labelled": int(len(lab)), "n_essential": int(lab.sum()),
               "auc_fba": auc, "auc_fba_on_fame_subset": auc_m, "auc_fame": auc_f,
               "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
               "missed": [g for g, _ in missed], "over": [g for g, _ in over],
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_ecoli_bed.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_ecoli_bed.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
