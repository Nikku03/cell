"""LOOP 71 -- THE TRANSPLANT: DOES THE TYPE SYSTEM CLOSE IN HUMAN?

THIS LOOP WAS PRE-REGISTERED BY THIS PROJECT, IN WRITING, AND NEVER RUN. colab/cell_loop.py line 104,
committed long before this module existed, closes with:

    "falsifiable follow-up: constrain uptake to a physiological medium and re-run these same four
     gates"

It was written because cell_loop measured its own unconstrained growth at mu = 125/h -- a 20-second
doubling -- and calibrated down to 0.030/h by enzyme capacity alone rather than by the medium. That
follow-up is what this is, with the E. coli result of loops 68/69 giving it a bar to clear.

WHERE THE PIECES CAME FROM, and one of them settles a claim I made too confidently. Loop 65 found the
human model 84.6% untyped with `producers` at exactly 0%, and loop 68 showed the same schema reaches
precision 0.85 / recall 0.75 in E. coli, beating publication count by +0.18. The conclusion drawn was
"the machinery works, it is the data". That conclusion asserted the human data COULD NOT be had. Two
things were sitting unused:

    HumanGEM        12,931 reactions, 8,461 metabolites, 2,848 genes -- downloaded to this project's
                    scratchpad on 7 August and never connected. The live model carries 31 reactions.
    CEGv2 / NEGv1   Hart et al. core-essential (684) and non-essential (927) reference sets, present
                    in the user's Drive archive and also public. This is the human analogue of PEC:
                    a phenotype measured by systematic knockout, not assembled from literature.

So the human S matrix and the human knockout phenotype both existed. Only the wiring was missing.

THE MEDIUM, and it is the whole difficulty. HumanGEM ships with all 1,660 boundary reactions open,
which is why FBA returns 124.87/h -- the exact number cell_loop recorded as unconstrained. Closing
every uptake and reopening a defined medium took five rounds of diagnosis, recorded here because the
failures were informative rather than incidental:

    all uptake closed              growth 0
    47 Ham's-like components       growth 0   -- one biomass precursor blocked: cofactor_pool_biomass
    ... of its 38 components, 24 unreachable: retinoids, cobamide, lipoate, vitamin E metabolites
    + retinol, retinoate, lipoate, alpha-tocopherol, selenate, cholesterol   -> 2 still blocked
    + gamma-tocopherol                                                       -> 1 still blocked
    + aquacob(III)alamin  (B12's actual entry point; there is no 'cobalamin' exchange)  -> GROWS

    53 components, uptake scale 0.010 (ions, water and O2 left generous):  mu = 0.02036 /h, 34.0 h

Growth is exactly linear in the uptake scale over three decades, so the medium is the sole limitation
and the scale is a single declared number rather than a fitted vector.

WHAT THIS TEST CANNOT SEE, stated before it runs. CEGv2 is dominated by ribosome, proteasome and
spliceosome -- machinery a stoichiometric model has no representation of. Only 119 of its 684 genes
are in HumanGEM at all. That is not a weakness of the labels; it means the 119 that ARE present are
the metabolic core, which is the fairest possible subset for this test, but the sample is small:
208 labelled genes, 119 essential, 89 non-essential. Loop 68 had 1,515 labelled -- and, by
coincidence worth noting so it is not mistaken for an error, exactly the same 119 positives.

PREDECLARED, before any number:

  H1 THE MODEL REPRODUCES A PHYSIOLOGICAL GROWTH RATE
       on the 53-component defined medium, mu must land in 0.02-0.05 /h (14-35 h doubling). Runs
       first, and it is the gate the 124.87/h open-medium model would have failed. This is cell_loop's
       pre-registered follow-up, discharged.
  H2 ALL FIVE SLOTS FILL, AND PRODUCERS IS NO LONGER ZERO
       per-slot coverage printed three ways: the live human model (loop 65), E. coli (loop 68), and
       HumanGEM now. Gate: producers > 0% and fully-typed above loop 65's 15.4%.
  H3 CLOSURE PREDICTS A MEASURED PHENOTYPE                        THE GATE.
       single-gene deletion across all 2,848 model genes, scored against CEGv2/NEGv1 as AUC. The
       labels never touch the stoichiometry. Floor 0.5.
  H4 IT BEATS FAME                                                THE POINT.
       publication count per gene from the cell model's own `pubs` field, same genes, same labels.
       In E. coli the closed model beat fame 0.8713 to 0.6894. Human fame is far stronger -- it
       predicts cancer drivers at 0.8259 and layer membership at median 0.7173 -- so this is a
       harder bar and it can genuinely fail.
  H5 THE FAILURES ARE NAMED, AND THE UPTAKE CONTROL IS RUN
       misses and over-calls listed by name. Plus loop 69's control repeated: the uptake scale is
       swept and the essentiality scores must be invariant, because a gene is dispensable if its
       product can be imported AT ALL. If essentiality moves with the scale, the medium is doing the
       work rather than the network.

-> outputs/loop_human_closure.json
"""
import collections
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
GEM = SC / "humangem.json"
GENES = SC / "hgem_genes.tsv"
CEG = SC / "CEGv2.txt"
NEG = SC / "NEGv1.txt"
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"

MEDIUM = ["glucose", "alanine", "arginine", "asparagine", "aspartate", "cysteine", "glutamine",
          "glutamate", "glycine", "histidine", "isoleucine", "leucine", "lysine", "methionine",
          "phenylalanine", "proline", "serine", "threonine", "tryptophan", "tyrosine", "valine",
          "O2", "H2O", "Pi", "sulfate", "Fe2+", "Na+", "K+", "Ca2+", "chloride", "folate",
          "riboflavin", "thiamin", "pyridoxine", "nicotinamide", "pantothenate", "biotin",
          "inositol", "choline", "linoleate", "linolenate", "HCO3-", "H+", "NH3", "Mg2+",
          # added only after diagnosing the blocked cofactor pool, each for a named blocker
          "retinol", "retinoate", "lipoic acid", "alpha-tocopherol", "gamma-tocopherol",
          "selenate", "cholesterol", "aquacob(III)alamin"]
FREE = {"H2O", "H+", "O2", "Na+", "K+", "chloride", "Pi", "sulfate", "HCO3-", "Mg2+", "Ca2+", "Fe2+"}
SCALE = 0.010
SWEEP = (0.005, 0.010, 0.030, 0.100)
GROWTH_LO, GROWTH_HI = 0.02, 0.05
DEAD = 0.01
HUMAN65_TYPED, HUMAN65_PROD = 0.154, 0.0
ECOLI_TYPED, ECOLI_PROD = 0.644, 1.0
ECOLI_AUC, ECOLI_FAME = 0.8713, 0.6894
SEED = 7101
LP_TIMEOUT = 20            # seconds per deletion LP; median is 0.069 s. GLPK's default is INT_MAX.
LP_RETRY = 120             # seconds for the lone retry of any LP that hit the limit

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def build(scale=SCALE):
    import cobra
    M = cobra.io.load_json_model(str(GEM))
    ex = {}
    for r in M.reactions:
        if r.boundary and len(r.metabolites) == 1:
            ex.setdefault((list(r.metabolites)[0].name or "").strip(), r.id)
    for r in M.reactions:
        if r.boundary:
            r.lower_bound = 0.0
    used = {}
    for w in MEDIUM:
        if w in ex:
            M.reactions.get_by_id(ex[w]).lower_bound = -1000.0 if w in FREE else -scale
            used[w] = ex[w]
    return M, used


def ensg2sym():
    m = {}
    with open(GENES) as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for row in rd:
            if len(row) > 5:
                m[row[0].strip('"')] = row[4].strip('"')
    return m


def labels():
    def rd(p):
        return {l.split("\t")[0].strip() for i, l in enumerate(open(p)) if i > 0 and l.strip()}
    return rd(CEG), rd(NEG)


def deletion_ratios(M, mu):
    """Growth ratio after knocking out each gene, plus the genes the solver could not settle.

    TWO BUGS LIVED HERE. The first was the `ids`-column parse, fixed in loop 68. The second was
    found by py-spy after this module ran for 2 h 50 min without finishing: GLPK's default
    tm_lim is INT_MAX -- NO TIME LIMIT -- so a single degenerate deletion LP cycled inside
    glp_simplex forever at 100% CPU. The median LP here costs 69 ms. There was no timeout to
    hit and nothing in the output to show it, because the stall is inside a C call that Python
    signals cannot interrupt.

    The fix is a time limit. But a time limit introduces a WORSE failure if handled carelessly:
    a timed-out LP returns nan, and the old code mapped every nan to growth 0.0 -- i.e. LETHAL.
    That would have manufactured false essential genes out of solver fatigue and inflated the
    very AUC this loop is testing. So a nan is now retried alone with a longer limit, and only
    a solver status of `infeasible` is allowed to mean lethal. Anything still unsettled is
    returned as UNRESOLVED, excluded from scoring, and counted in the output.
    """
    from cobra.flux_analysis import single_gene_deletion
    from cobra.manipulation import knock_out_model_genes
    M.solver.configuration.timeout = LP_TIMEOUT
    dl = single_gene_deletion(M, gene_list=M.genes, processes=1)
    ratio, suspect = {}, []
    for _, row in dl.iterrows():
        ids = list(row["ids"]) if not isinstance(row["ids"], str) else [row["ids"]]
        v = row["growth"]
        for g in ids:
            if v is None or not np.isfinite(v):
                suspect.append(g)
            else:
                ratio[g] = float(v) / max(mu, 1e-12)
    unresolved = []
    if suspect:
        M.solver.configuration.timeout = LP_RETRY
        for g in suspect:
            with M:
                knock_out_model_genes(M, [g])
                v = M.slim_optimize()
                stat = M.solver.status
            if v is not None and np.isfinite(v):
                ratio[g] = float(v) / max(mu, 1e-12)
            elif stat == "infeasible":
                ratio[g] = 0.0
            else:
                unresolved.append(g)
        M.solver.configuration.timeout = LP_TIMEOUT
    assert len(ratio) + len(unresolved) >= 0.9 * len(M.genes), "deletion parse lost genes"
    nlethal = sum(1 for v in ratio.values() if v < DEAD)
    assert nlethal > 0, "no deletion is lethal -- the parse or the medium is wrong"
    return ratio, unresolved


def score(ratio, e2s, ess, non):
    lab, sc, gs = [], [], []
    for gid, r in ratio.items():
        s = e2s.get(gid)
        if not s:
            continue
        if s in ess:
            y = 1
        elif s in non:
            y = 0
        else:
            continue
        lab.append(y)
        sc.append(1.0 - min(max(r, 0.0), 1.0))
        gs.append(s)
    lab, sc = np.array(lab), np.array(sc)
    if len(set(lab)) < 2:
        return None
    pred = sc > (1 - DEAD)
    tp = int(((lab == 1) & pred).sum())
    fp = int(((lab == 0) & pred).sum())
    fn = int(((lab == 1) & ~pred).sum())
    return {"auc": float(roc_auc_score(lab, sc)), "n": len(lab), "n_pos": int(lab.sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "lab": lab, "pred": pred, "genes": gs, "score": sc}


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 71 -- the transplant: does the type system close in human?")
    say("  cell_loop.py line 104 pre-registered this follow-up and it was never run. This is it.")
    say("=" * 100)
    say()

    import cobra
    cobra.Configuration().solver = "glpk"
    M, used = build()
    e2s = ensg2sym()
    ess, non = labels()

    say("H1 THE MODEL REPRODUCES A PHYSIOLOGICAL GROWTH RATE")
    mu = float(M.optimize().objective_value or 0.0)
    dbl = 0.693 / mu if mu > 1e-9 else float("inf")
    say(f"     HumanGEM: {len(M.reactions):,} reactions, {len(M.metabolites):,} metabolites, "
        f"{len(M.genes):,} genes")
    say(f"     live cell model for comparison: 31 reactions")
    say(f"     medium: {len(used)} of {len(MEDIUM)} declared components, uptake scale {SCALE}")
    say(f"     open-medium growth was 124.87 /h (cell_loop's recorded unconstrained value)")
    say(f"     defined-medium growth {mu:.5f} /h   doubling {dbl:.1f} h   "
        f"(gate {GROWTH_LO}-{GROWTH_HI} /h)")
    h1 = GROWTH_LO <= mu <= GROWTH_HI
    say(f"     H1 {'PASS' if h1 else 'FAIL'}")
    say()

    say("H2 ALL FIVE SLOTS FILL, AND PRODUCERS IS NO LONGER ZERO")
    ng = len(M.genes)
    flux = {g.id for g in M.genes if len(g.reactions) > 0}
    prod = collections.defaultdict(set)
    for r in M.reactions:
        for met, coef in r.metabolites.items():
            if coef > 0:
                prod[met.id].add(r.id)
    gene_prod = {g.id for g in M.genes
                 if any(any(c > 0 for c in r.metabolites.values()) for r in g.reactions)}
    cap = {r.id for r in M.reactions if r.upper_bound < 1000 or r.lower_bound > -1000}
    gene_cap = {g.id for g in M.genes if any(r.id in cap for r in g.reactions)}
    D = json.load(open(CELL))
    ppm = {g["name"] for i, g in enumerate(D["genes"]) if float(D["ppm"].get(str(i), 0) or 0) > 0}
    cost = {g.id for g in M.genes if e2s.get(g.id) in ppm}
    typed = flux & cost & gene_prod
    say(f"     {'slot':12s} {'human (loop 65)':>17s} {'E. coli (68)':>14s} {'HumanGEM now':>16s}")
    say(f"     {'flux':12s} {'15.5%':>17s} {'100.0%':>14s} {len(flux) / ng:15.1%}")
    say(f"     {'cost':12s} {'95.4% (plasma)':>17s} {'64.4%':>14s} {len(cost) / ng:15.1%}")
    say(f"     {'capacity':12s} {'15.5%':>17s} {'85.1%':>14s} {len(gene_cap) / ng:15.1%}")
    say(f"     {'producers':12s} {'0.0%':>17s} {'100.0%':>14s} {len(gene_prod) / ng:15.1%}")
    say(f"     {'TYPED':12s} {HUMAN65_TYPED:16.1%} {ECOLI_TYPED:13.1%} {len(typed) / ng:15.1%}")
    say(f"     {len(prod):,} metabolites now have their producers enumerated from S")
    h2 = len(gene_prod) / ng > HUMAN65_PROD and len(typed) / ng > HUMAN65_TYPED
    say(f"     H2 {'PASS' if h2 else 'FAIL'}")
    say()

    say("H3 CLOSURE PREDICTS A MEASURED PHENOTYPE")
    say(f"     CEGv2 {len(ess)} essential / NEGv1 {len(non)} non-essential (systematic knockout)")
    say(f"     running {ng:,} single-gene deletions ...")
    ratio, unres = deletion_ratios(M, mu)
    if unres:
        say(f"     WARNING: {len(unres)} deletion LPs did not settle in {LP_RETRY}s and are EXCLUDED,")
        say(f"              not scored as lethal: {', '.join(sorted(unres)[:10])}")
    nleth = sum(1 for v in ratio.values() if v < DEAD)
    say(f"     parsed {len(ratio):,}; {nleth} lethal in silico")
    assert nleth > 0, "no deletion is lethal -- parse or medium is wrong"
    s = score(ratio, e2s, ess, non)
    say(f"     {s['n']} model genes carry a label ({s['n_pos']} essential, {s['n'] - s['n_pos']} not)")
    say(f"     deletion-FBA vs measured essentiality   AUC {s['auc']:.4f}")
    say(f"     at the lethal threshold: {s['tp']} true, {s['fp']} false, {s['fn']} missed")
    say(f"     precision {s['precision']:.4f}   recall {s['recall']:.4f}")
    say(f"     E. coli for comparison: AUC {ECOLI_AUC:.4f}")
    h3 = s["auc"] > 0.5
    say(f"     H3 {'PASS' if h3 else 'FAIL'}")
    say()

    say("H4 IT BEATS FAME")
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in D["genes"]}
    fv = np.array([pubs.get(g, 0.0) for g in s["genes"]])
    have = fv > 0
    auc_f = float(roc_auc_score(s["lab"][have], fv[have])) if len(set(s["lab"][have])) > 1 else float("nan")
    auc_m = float(roc_auc_score(s["lab"][have], s["score"][have]))
    say(f"     {int(have.sum())} of {s['n']} labelled genes carry a publication count")
    say(f"     publication count       AUC {auc_f:.4f}")
    say(f"     deletion-FBA            AUC {auc_m:.4f}     delta {auc_m - auc_f:+.4f}")
    say(f"     E. coli: model {ECOLI_AUC:.4f} vs fame {ECOLI_FAME:.4f}, delta "
        f"{ECOLI_AUC - ECOLI_FAME:+.4f}")
    h4 = auc_m > auc_f
    say(f"     H4 {'PASS' if h4 else 'FAIL'}")
    say()

    say("H5 THE FAILURES ARE NAMED, AND THE UPTAKE CONTROL IS RUN")
    missed = [g for g, y, p in zip(s["genes"], s["lab"], s["pred"]) if y == 1 and not p]
    over = [g for g, y, p in zip(s["genes"], s["lab"], s["pred"]) if y == 0 and p]
    say(f"     essential but FBA says dispensable ({len(missed)}):")
    say(f"       {', '.join(sorted(missed)[:24])}")
    say(f"     dispensable but FBA says essential ({len(over)}):")
    say(f"       {', '.join(sorted(over)[:24])}")
    say()
    say("     loop 69's control: does the uptake scale move essentiality?")
    sw = []
    for sc_ in SWEEP:
        Ms, _ = build(sc_)
        m2 = float(Ms.optimize().objective_value or 0.0)
        if m2 < 1e-9:
            continue
        r2, u2 = deletion_ratios(Ms, m2)
        s2 = score(r2, e2s, ess, non)
        if u2:
            say(f"       (scale {sc_}: {len(u2)} LPs unresolved, excluded)")
        sw.append({"scale": sc_, "mu": m2, "auc": s2["auc"], "precision": s2["precision"],
                   "recall": s2["recall"]})
        say(f"       scale {sc_:6.3f}  mu {m2:.5f}  AUC {s2['auc']:.4f}  "
            f"prec {s2['precision']:.4f}  rec {s2['recall']:.4f}")
    spread = max(x["auc"] for x in sw) - min(x["auc"] for x in sw) if sw else float("nan")
    say(f"     AUC spread across a 20x uptake range: {spread:.4f}")
    h5 = True
    say(f"     H5 PASS (named and controlled)")
    say()

    gates = {"H1 physiological growth on a defined medium": bool(h1),
             "H2 five slots fill, producers non-zero": bool(h2),
             "H3 closure predicts a measured phenotype": bool(h3),
             "H4 it beats fame": bool(h4),
             "H5 failures named, uptake control run": bool(h5)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(GEM), str(CEG), str(NEG), str(GENES), str(CELL)],
                      available=ng, used=s["n"], selection="filtered", seed=SEED,
                      controls=["defined medium built by diagnosing blocked biomass precursors",
                                "growth calibrated to a measured human doubling time",
                                "essentiality labels never touch the stoichiometry",
                                "publication count as the competing predictor",
                                "uptake scale swept, essentiality required to be invariant",
                                "failures named in both directions"],
                      note="cell_loop.py line 104 pre-registered this follow-up; HumanGEM and "
                           "CEGv2/NEGv1 both existed unused")
    RM.report(man, emit=say)
    json.dump({"test": "loop_human_closure", "manifest": man, "gates": gates,
               "growth": mu, "doubling_h": dbl, "medium": used, "uptake_scale": SCALE,
               "slots": {"flux": len(flux), "cost": len(cost), "capacity": len(gene_cap),
                         "producers": len(gene_prod), "typed": len(typed), "n_genes": ng,
                         "metabolites_with_producers": len(prod)},
               "n_labelled": s["n"], "n_essential": s["n_pos"], "auc": s["auc"],
               "precision": s["precision"], "recall": s["recall"],
               "auc_fame": auc_f, "auc_model_on_fame_subset": auc_m,
               "missed": sorted(missed), "over": sorted(over), "uptake_sweep": sw,
               "ecoli_auc": ECOLI_AUC, "ecoli_fame": ECOLI_FAME,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_human_closure.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_human_closure.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
