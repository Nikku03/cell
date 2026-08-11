"""IS BUFFERING PREDICTABLE FROM REDUNDANCY STRUCTURE -- the test that says engineering or theory.

WHY THIS IS THE DECIDING EXPERIMENT.  Five separate results in this project measured the same thing:
near-field mechanism is right and does not reach the far field.

    nexus       predicts interface breaking at 0.77 and states that activity -> phenotype "does NOT compose"
    cell_loop   closes mu = F(mu) to 1.7e-08 and changes prediction by -0.0013
    loop_chain  99.99% correct translation, and two of its three links carry nothing at variant level
    loop_integrate  seven layers reach 0.8309 and only two of them carry it
    loop_recall PPI degree separates the missed genes at 0.6793 and makes the predictor WORSE

And loop_slack measured why it might be: only 11.7% of knockouts cost more than 1% of growth. Cells
buffer. The signal a whole-cell model needs to propagate is the signal evolution spent a very long time
damping out.

So the question that decides how hard a complete cell model is:

    IS BUFFERING PREDICTABLE FROM REDUNDANCY STRUCTURE ALONE?

    predictable and usable  -> the composition gap is ENGINEERING. Model the redundancy, propagate
                              through it, and the far field opens up.
    not predictable         -> it is THEORY, and no amount of data collection closes it.

WHAT BUFFERING IS, MADE MEASURABLE.  A gene is buffered when the model says losing it should hurt and
the cell shrugs. That set already exists and was named: loop_tail found the metabolic model calls 206
genes costly, 158 of which are real dependencies and 48 of which are not. Those 48 ARE the buffered
genes -- mechanistically important, practically dispensable -- and nothing about them was explained.

    group NOT BUFFERED   high metabolic cost, high DepMap dependency      n = 158
    group BUFFERED       high metabolic cost, low DepMap dependency       n = 48
    both have high cost, so whatever separates them is not importance -- it is what covers the loss

THE CIRCULARITY THIS IS BUILT AGAINST, and it would be fatal if ignored.  cell_complete carries two
features that look perfect for this and are poisoned: `sl` (synthetic-lethal pairs) and `codep`
(co-dependency). Both are computed FROM the DepMap dependency matrix -- the same data that defines the
LABEL. A gene that is never a dependency has no dependency variance, so it cannot have strong
co-dependency partners, so codep-count predicts dependency by construction rather than by biology.

Features are therefore split, and only one half counts:

    INDEPENDENT   isozyme count and sole-catalyst reactions (Human-GEM GPR -- model structure)
                  paralogue family size (sequence)
                  PPI degree, complex membership (interaction and complex annotation)
                  expression (proteomics) -- included so it cannot hide behind another feature
    DEPMAP-DERIVED  co-dependency partners, synthetic-lethal partners
                  REPORTED AND NOT COUNTED. They share a source with the label.

PREDECLARED, before any number:

    B1 THE GROUPS ARE MATCHED ON COST   the two groups' metabolic cost distributions must overlap, so
                that what separates them cannot be "one is more costly".
                A design check. Without it this measures importance again and calls it buffering.
    B2 REDUNDANCY SEPARATES THEM   at least one INDEPENDENT feature separates buffered from not-buffered
                above the 95th percentile of 200 label shuffles.
    B3 IT IS NOT EXPRESSION   the best non-expression independent feature also clears the band.
    B4 IT HOLDS OUT OF SAMPLE   5-fold CV over the independent features beats the shuffled band.
                In-sample separation with seven features on 206 genes is nearly free; held out is not.
    B5 IT ACTUALLY FIXES THE PREDICTOR   re-ranking the model's own tail by cost MINUS predicted
                buffering must raise precision at the same number of genes called.
                THIS IS THE ONE THAT DECIDES ENGINEERING VS THEORY, and it is exactly the gate
                loop_recall failed: PPI degree separated the missed genes and still made the predictor
                worse, because it promoted true negatives at the same rate. Separating a group is not
                the same as being able to act on it.

CONTROLS: cost matched between groups; 200 label shuffles; expression carried so it cannot hide;
DepMap-derived features quarantined and reported separately; held-out CV; and the operating point held
fixed when comparing precision.

-> outputs/loop_buffering.json
"""
import collections
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import auc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
SEED = 1904
N_SHUFFLE = 200
DEP_THRESHOLD = 0.10
COST_THRESHOLD = 1e-2

INDEPENDENT = ["isozymes (fewest alternatives)", "sole-catalyst reactions",
               "paralogue family size", "PPI degree", "in a complex", "expression"]
DEPMAP_DERIVED = ["co-dependency partners", "synthetic-lethal partners"]


def family(sym):
    """Gene family by alphabetic prefix: ALDH1A1, ALDH2, ALDH3A1 -> ALDH.

    Better than a fixed character count, which splits ALDH2 from ALDH1A1 and merges unrelated
    four-letter prefixes."""
    m = re.match(r"^([A-Z]+)", sym.upper())
    return m.group(1) if m else sym[:3].upper()


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    say("=" * 100)
    say("IS BUFFERING PREDICTABLE FROM REDUNDANCY STRUCTURE -- engineering or theory")
    say("=" * 100)
    say("  Five results measured the same thing: mechanism is right locally and does not reach the")
    say("  far field. loop_slack measured why -- only 11.7% of knockouts cost >1% of growth. Cells")
    say("  buffer. If that buffering is predictable from redundancy STRUCTURE, the composition gap is")
    say("  engineering. If not, it is theory and no data collection closes it.")

    S = pd.DataFrame(json.load(open(OUT / "loop_medium.json"))["scores"])
    D = json.load(open(CELL))
    genes = D["genes"]
    name2i = {g["name"]: i for i, g in enumerate(genes)}
    ppm = {int(k): float(v) for k, v in D["ppm"].items()}
    g2c = D.get("gene2cplx") or {}
    codep = {int(k): len(v) for k, v in (D.get("codep") or {}).items()}
    sl_count = collections.Counter()
    for rec in (D.get("sl") or []):
        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
            sl_count[int(rec[0])] += 1
            sl_count[int(rec[1])] += 1
    dep = {g["name"]: float(g["dep_frac"]) for g in genes
           if isinstance(g.get("dep_frac"), (int, float))}
    say(f"\n  {len(dep):,} genes with a DepMap dependency; {len(codep):,} with co-dependency; "
        f"{len(sl_count):,} in a synthetic-lethal pair")

    # ---- isozyme redundancy, straight out of the model's own GPR rules ---------------------------
    say("\n  loading Human-GEM for GPR structure (isozymes are metabolic redundancy, and they are")
    say("  written down in the model rather than inferred)")
    import cobra
    cobra_cfg = cobra.Configuration()
    cobra_cfg.solver = "glpk"
    M = cobra.io.read_sbml_model(str(SC / "HumanGEM.xml"))
    ens2sym = {}
    for g in M.genes:
        nm = g.name or g.id
        ens2sym[g.id] = nm
    alts, sole = collections.defaultdict(list), collections.Counter()
    for r in M.reactions:
        gs = [ens2sym.get(x.id, x.id) for x in r.genes]
        if not gs:
            continue
        for s in gs:
            alts[s].append(len(gs) - 1)          # how many OTHER genes can serve this reaction
            if len(gs) == 1:
                sole[s] += 1
    say(f"  {len(alts):,} genes carry at least one reaction; isozyme counts read from "
        f"{len(M.reactions):,} GPR rules")

    fam = collections.Counter(family(g["name"]) for g in genes)

    rows = []
    for _, r in S.iterrows():
        s = r["sym"]
        if s not in dep:
            continue
        c = float(r["medium"])
        if not np.isfinite(c):
            continue
        i = name2i.get(s)
        rows.append({
            "sym": s, "cost": max(c, 0.0), "dep": dep[s],
            "isozymes (fewest alternatives)": float(min(alts[s])) if alts.get(s) else 0.0,
            "sole-catalyst reactions": float(sole.get(s, 0)),
            "paralogue family size": float(fam.get(family(s), 1)),
            "PPI degree": float(genes[i].get("ppi") or 0) if i is not None else 0.0,
            "in a complex": 1.0 if (i is not None and (str(i) in g2c or s in g2c)) else 0.0,
            "expression": float(np.log10((ppm.get(i, 0.0) if i is not None else 0.0) + 1e-3)),
            "co-dependency partners": float(codep.get(i, 0)) if i is not None else 0.0,
            "synthetic-lethal partners": float(sl_count.get(i, 0)) if i is not None else 0.0,
        })
    T = pd.DataFrame(rows)
    tail = T[T.cost > COST_THRESHOLD].reset_index(drop=True)
    tail["real"] = (tail.dep >= DEP_THRESHOLD).astype(int)
    tail["buffered"] = 1 - tail["real"]
    nb, nn = int(tail.buffered.sum()), int(tail.real.sum())
    say(f"\n  the model's own tail: {len(tail)} genes it calls costly")
    say(f"    NOT BUFFERED  {nn}   high cost, and the cell does depend on them")
    say(f"    BUFFERED      {nb}   high cost, and the cell shrugs")

    # ---- B1 the design check ---------------------------------------------------------------------
    cb, cn = tail.cost[tail.buffered == 1], tail.cost[tail.buffered == 0]
    a_cost = auc(tail.cost, tail.buffered.to_numpy())
    nullc = np.array([abs(auc(tail.cost, rng.permutation(tail.buffered.to_numpy())) - 0.5)
                      for _ in range(N_SHUFFLE)])
    b1 = bool(abs(a_cost - 0.5) <= np.percentile(nullc, 95))
    say(f"\n  B1 THE GROUPS ARE MATCHED ON COST -- cost itself separates them at {a_cost:.4f} "
        f"against a shuffled band of {0.5+np.percentile(nullc,95):.4f}")
    say(f"     medians {np.median(cb):.4g} buffered vs {np.median(cn):.4g} not")
    say(f"     B1 {'PASS' if b1 else 'FAIL'}  (if cost separated them, this would be measuring")
    say(f"     importance again and calling it buffering)")

    # ---- B2 / B3 which features separate ----------------------------------------------------------
    y = tail.buffered.to_numpy()
    say(f"\n  B2 REDUNDANCY SEPARATES THEM -- above 0.5 means MORE of it predicts BUFFERED")
    say(f"     {'feature':<34}{'AUC':>9}{'|null| 95th':>14}")
    res = {}
    for f in INDEPENDENT + DEPMAP_DERIVED:
        a = auc(tail[f], y)
        null = np.array([abs(auc(tail[f], rng.permutation(y)) - 0.5) for _ in range(N_SHUFFLE)])
        p95 = float(np.percentile(null, 95))
        sig = bool(abs(a - 0.5) > p95)
        res[f] = {"auc": a, "band95": 0.5 + p95, "significant": sig,
                  "independent": f in INDEPENDENT}
        tag = "" if f in INDEPENDENT else "   <- DepMap-derived, NOT COUNTED"
        say(f"     {f:<34}{a:>9.4f}{0.5+p95:>14.4f}   "
            f"{'SIGNIFICANT' if sig else 'no':<12}{tag}")
    ind = {f: v for f, v in res.items() if v["independent"]}
    b2 = bool(any(v["significant"] for v in ind.values()))
    say(f"     B2 {'PASS' if b2 else 'FAIL'}  (independent features only)")

    ne = {f: v for f, v in ind.items() if f != "expression"}
    best_ne = max(ne, key=lambda f: abs(ne[f]["auc"] - 0.5))
    b3 = bool(ne[best_ne]["significant"])
    say(f"  B3 IT IS NOT EXPRESSION -- best non-expression independent feature is {best_ne} at "
        f"{ne[best_ne]['auc']:.4f}   B3 {'PASS' if b3 else 'FAIL'}")

    # ---- B4 held out -------------------------------------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = tail[INDEPENDENT].to_numpy(float)
    mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")[:, 1]
    a_oof = auc(oof, y)
    null_oof = np.array([abs(auc(oof, rng.permutation(y)) - 0.5) for _ in range(N_SHUFFLE)])
    p95o = float(np.percentile(null_oof, 95))
    b4 = bool(abs(a_oof - 0.5) > p95o)
    say(f"\n  B4 IT HOLDS OUT OF SAMPLE -- 5-fold CV over the {len(INDEPENDENT)} independent features: "
        f"{a_oof:.4f} vs band {0.5+p95o:.4f}   B4 {'PASS' if b4 else 'FAIL'}")

    # ---- B5 does it fix the predictor --------------------------------------------------------------
    say(f"\n  B5 IT ACTUALLY FIXES THE PREDICTOR -- re-rank the model's own tail and call the same")
    say(f"     number of genes, so the operating point cannot drift")
    k = nn
    base_order = np.argsort(-tail.cost.to_numpy())
    prec_base = float(tail.real.to_numpy()[base_order[:k]].mean())
    comb = (tail.cost.to_numpy() - tail.cost.to_numpy().mean()) / (tail.cost.to_numpy().std() + 1e-9) \
        - (oof - oof.mean()) / (oof.std() + 1e-9)
    new_order = np.argsort(-comb)
    prec_new = float(tail.real.to_numpy()[new_order[:k]].mean())
    b5 = bool(prec_new > prec_base)
    say(f"     calling the top {k} of {len(tail)}:")
    say(f"       cost alone                      precision {prec_base:.4f}")
    say(f"       cost minus predicted buffering  precision {prec_new:.4f}   "
        f"({prec_new-prec_base:+.4f})")
    say(f"     loop_tail's headline precision on this tail was 0.7670")
    say(f"     B5 {'PASS' if b5 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"B1 groups matched on cost": b1, "B2 redundancy separates them": b2,
             "B3 not expression": b3, "B4 holds out of sample": b4,
             "B5 fixes the predictor": b5}
    for kk, vv in gates.items():
        say(f"  {kk:<40}{'PASS' if vv else 'FAIL'}")
    if not b1:
        say("  THE DESIGN CHECK FAILED. Cost separates the two groups, so this is measuring importance")
        say("  again rather than buffering, and nothing below it can be read.")
    elif b2 and b4 and b5:
        say("  BUFFERING IS PREDICTABLE AND USABLE, AND THE COMPOSITION GAP IS ENGINEERING.")
        say(f"  Redundancy structure separates buffered from not-buffered out of sample at {a_oof:.4f},")
        say(f"  and using it raises the model's own precision from {prec_base:.4f} to {prec_new:.4f}.")
        say("  That is the first time in this project that naming a missing layer also let us use it.")
    elif b2 and b4:
        say("  PREDICTABLE, NOT USABLE -- AND THAT IS THE SAME WALL AS loop_recall.")
        say(f"  Redundancy separates the groups out of sample at {a_oof:.4f}, and re-ranking by it does")
        say(f"  not raise precision ({prec_base:.4f} -> {prec_new:.4f}). Twice now a real signal about")
        say("  what the model gets wrong has failed to improve the model, which is what a THEORY gap")
        say("  looks like from the inside: the information is present and there is no law that turns")
        say("  it into a prediction.")
    elif b2:
        say("  SEPARABLE IN SAMPLE, NOT OUT OF IT. With six features on a few hundred genes that is")
        say("  close to what overfitting looks like, and the honest reading is that it is not")
        say("  established.")
    else:
        say("  REDUNDANCY STRUCTURE DOES NOT PREDICT BUFFERING. The 48 genes the model over-calls are")
        say("  not distinguished by isozymes, paralogues, complexes or hubs. On this evidence the")
        say("  composition gap is THEORY, not engineering, and collecting more of the same data does")
        say("  not close it.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(OUT / "loop_medium.json"), str(CELL), str(SC / "HumanGEM.xml")],
                      available=len(T), used=len(tail), selection="filtered", seed=SEED,
                      controls=["cost matched between groups as a design check",
                                f"{N_SHUFFLE} label shuffles per feature",
                                "expression carried so it cannot hide",
                                "DepMap-derived features quarantined and reported separately",
                                "5-fold held-out CV",
                                "operating point held fixed when comparing precision"],
                      note="sl and codep are computed from the same DepMap matrix as the label, so "
                           "they are reported and excluded from every gate")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_buffering", "manifest": man, "gates": gates,
               "n_tail": len(tail), "n_buffered": nb, "n_not_buffered": nn,
               "cost_auc": a_cost, "features": res, "oof_auc": a_oof,
               "precision_base": prec_base, "precision_with_buffering": prec_new,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_buffering.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_buffering.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
