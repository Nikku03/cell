"""THE 285 IT MISSES -- what layer would be needed to see them.

WHERE THIS COMES FROM.  loop_tail measured the metabolic model's tail precisely: precision 76.7%,
recall 35.7%. When it calls a gene costly it is right three times in four, and it finds only about a
third of what cells actually depend on. It named the 285 dependencies it misses and observed that many
are outside metabolism entirely -- protein ubiquitination among them.

"Outside metabolism" is a shrug unless it says WHICH layer would cover them. This asks that.

THE QUESTION.  Take the genes the metabolic model MISSES -- real DepMap dependencies that it scores as
costless -- and ask what distinguishes them from genes that are correctly called non-dependencies. If a
feature separates those two groups, that feature is the missing layer, named.

    the missed          DepMap dependency, model cost ~ 0
    the true negatives  not a dependency, model cost ~ 0
    both look identical to the metabolic model, so anything separating them is information it lacks

CANDIDATE LAYERS, each already in cell_complete and none of them metabolic:
    PPI degree            how many protein interactions -- hub-ness
    complex membership    whether the protein sits in an annotated complex
    coexpression partners how many strong co-expression partners
    paralogue buffering   whether a close paralogue exists, which is why many single knockouts do not
                          show a phenotype at all
    expression            the confound that has beaten everything, included so it cannot hide

PREDECLARED, before any number:

    R1 THE MISSED ARE DISTINGUISHABLE   at least one candidate separates missed dependencies from true
                negatives above the 95th percentile of 200 label shuffles.
                fails -> nothing in cell_complete distinguishes them, the missing information is not in
                this container at all, and that is the honest end of this line.
    R2 IT IS NOT JUST EXPRESSION   the best NON-expression candidate also clears the shuffle band.
                fails -> the missing layer is expression, which the model already has access to and
                which loop 25 showed it is already using.
    R3 IT WOULD ACTUALLY HELP   adding the best candidate to the metabolic cost improves recall at
                matched precision on the full scored set.
                THIS IS THE REAL GATE. A feature can separate the missed group and still not improve
                the predictor, because it may also promote true negatives.

CONTROLS: 200 label shuffles; expression included specifically so it cannot hide behind another
feature; precision held fixed when comparing recall, since recall is trivially improvable by lowering
a threshold.

-> outputs/loop_recall.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import auc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
SEED = 1904
N_SHUFFLE = 200
COST_THRESHOLD = 1e-2
DEP_THRESHOLD = 0.10


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("THE 285 IT MISSES -- what layer would be needed to see them")
    say("=" * 100)
    say("  loop_tail measured precision 76.7% and recall 35.7% and said the misses are 'outside")
    say("  metabolism'. That is a shrug unless it names which layer would cover them.")

    lm = OUT / "loop_medium.json"
    if not lm.exists():
        say("  loop_medium.json missing. Recorded as not testable.")
        return 1
    S = pd.DataFrame(json.load(open(lm))["scores"])
    D = json.load(open(CELL))
    genes = D["genes"]
    name2i = {g["name"]: i for i, g in enumerate(genes)}
    ppm = {int(k): float(v) for k, v in D["ppm"].items()}
    coexpr = {int(k): len(v) for k, v in (D.get("coexpr") or {}).items()}
    g2c = D.get("gene2cplx") or {}
    dep = {g["name"]: float(g["dep_frac"]) for g in genes
           if isinstance(g.get("dep_frac"), (int, float))}

    S = S[S["sym"].isin(dep)].copy()
    S["cost"] = S["medium"].astype(float).clip(lower=0.0)
    S["dep"] = [dep[s] for s in S["sym"]]
    S = S[np.isfinite(S.cost)].reset_index(drop=True)

    # paralogue proxy: genes sharing the first four characters of the symbol are usually family members
    fam = {}
    for g in genes:
        fam.setdefault(g["name"][:4], []).append(g["name"])

    rows = []
    for _, r in S.iterrows():
        s = r["sym"]
        i = name2i.get(s)
        rows.append({
            "sym": s, "cost": r["cost"], "dep": r["dep"],
            "PPI degree": float(genes[i].get("ppi") or 0) if i is not None else 0.0,
            "in a complex": 1.0 if str(i) in g2c or s in g2c else 0.0,
            "coexpression partners": float(coexpr.get(i, 0)) if i is not None else 0.0,
            "paralogue family size": float(len(fam.get(s[:4], []))),
            "expression": float(np.log10((ppm.get(i, 0.0) if i is not None else 0.0) + 1e-3)),
        })
    T = pd.DataFrame(rows)
    feats = ["PPI degree", "in a complex", "coexpression partners", "paralogue family size",
             "expression"]

    missed = (T.dep >= DEP_THRESHOLD) & (T.cost <= COST_THRESHOLD)
    tn = (T.dep < DEP_THRESHOLD) & (T.cost <= COST_THRESHOLD)
    sub = T[missed | tn].copy()
    ysub = missed[missed | tn].astype(int).to_numpy()
    say(f"\n  {int(missed.sum()):,} dependencies the model MISSES; {int(tn.sum()):,} correct negatives")
    say(f"  both groups score ~0 in the metabolic model, so anything separating them is information")
    say(f"  the model does not have")

    res = {}
    say(f"\n  {'candidate layer':<26}{'AUC':>9}{'|null| 95th':>14}")
    for f in feats:
        a = auc(sub[f], ysub)
        null = np.array([auc(sub[f], rng.permutation(ysub)) for _ in range(N_SHUFFLE)])
        p95 = float(np.percentile(np.abs(null - 0.5), 95)) + 0.5
        res[f] = {"auc": a, "p95": p95, "beats": bool(abs(a - 0.5) > p95 - 0.5)}
        say(f"  {f:<26}{a:>9.4f}{p95:>14.4f}   {'SIGNIFICANT' if res[f]['beats'] else 'no'}")

    r1 = bool(any(v["beats"] for v in res.values()))
    say(f"\n  R1 THE MISSED ARE DISTINGUISHABLE -- {'PASS' if r1 else 'FAIL'}")
    nonexpr = {f: v for f, v in res.items() if f != "expression"}
    best_ne = max(nonexpr, key=lambda f: abs(nonexpr[f]["auc"] - 0.5))
    r2 = bool(nonexpr[best_ne]["beats"])
    say(f"  R2 IT IS NOT JUST EXPRESSION -- best non-expression layer is {best_ne} at "
        f"{nonexpr[best_ne]['auc']:.4f}   R2 {'PASS' if r2 else 'FAIL'}")

    # ---- R3 would it actually help, at matched precision ----------------------------------------------
    y_full = (T.dep >= DEP_THRESHOLD).astype(int).to_numpy()
    base_tail = T.cost > COST_THRESHOLD
    base_prec = float(y_full[base_tail.to_numpy()].mean()) if base_tail.sum() else 0.0
    base_rec = float(y_full[base_tail.to_numpy()].sum() / max(y_full.sum(), 1))
    say(f"\n  R3 WOULD IT HELP -- baseline metabolic tail: precision {base_prec:.1%}, "
        f"recall {base_rec:.1%}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold

    def oof(cols):
        X = T[list(cols)].to_numpy(float)
        o = np.zeros(len(y_full))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y_full):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            m.fit(X[tr], y_full[tr])
            o[te] = m.predict_proba(X[te])[:, 1]
        return o

    def recall_at(o, prec_target, min_n=20):
        """Maximum recall reachable at precision >= target.

        NOT the first point where precision dips below target -- precision along a ranking is
        noisy and an early dip would cut a good ranker off at its first unlucky gene. Cost alone
        ranks genes exactly as thresholding cost does, so this form is guaranteed to reproduce
        loop_tail's measured 35.7% for the baseline; the early-stop form scored it 2.7%, which
        would have handed the comparison to the challenger for free.
        """
        order = np.argsort(-o)
        ys = y_full[order]
        tp = np.cumsum(ys)
        n = np.arange(1, len(ys) + 1)
        ok = (tp / n >= prec_target) & (n >= min_n)
        if not ok.any():
            return 0.0, 0
        k = int(np.max(np.where(ok)[0]))
        return float(tp[k] / max(y_full.sum(), 1)), int(n[k])

    o_base = oof(["cost"])
    o_plus = oof(["cost", best_ne])
    r_base, n_base = recall_at(o_base, base_prec)
    r_plus, n_plus = recall_at(o_plus, base_prec)
    say(f"    at a matched precision of {base_prec:.1%}:")
    say(f"      metabolic cost alone        recall {r_base:.1%}  ({n_base} genes called)")
    say(f"      cost + {best_ne:<20} recall {r_plus:.1%}  ({n_plus} genes called)")
    r3 = bool(r_plus - r_base >= 0.05)
    say(f"    R3 {'PASS' if r3 else 'FAIL'}  (gate: +5 points of recall)")

    say("\n" + "=" * 100)
    gates = {"R1 the missed are distinguishable": r1, "R2 not just expression": r2,
             "R3 it would actually help": r3}
    for kk, vv in gates.items():
        say(f"  {kk:<40}{'PASS' if vv else 'FAIL'}")
    if r1 and r2 and r3:
        say(f"  THE MISSING LAYER IS NAMED AND IT WOULD WORK: {best_ne}. Adding it lifts recall from")
        say(f"  {r_base:.0%} to {r_plus:.0%} at unchanged precision. That is a concrete next build with")
        say("  a measured payoff, which is what loop_tail's 'outside metabolism' was missing.")
    elif r1 and r2:
        say(f"  THE MISSING LAYER IS NAMED AND IT DOES NOT HELP: {best_ne} separates the missed genes")
        say("  from the true negatives, and adding it to the predictor does not buy recall at matched")
        say("  precision -- because it promotes true negatives at the same rate. Naming the information")
        say("  is not the same as being able to use it, and that distinction is the result.")
    elif r1:
        say("  ONLY EXPRESSION DISTINGUISHES THEM, and the model already has expression. The missing")
        say("  information is not in this container.")
    else:
        say("  NOTHING IN cell_complete DISTINGUISHES THE MISSED FROM THE CORRECTLY IGNORED. The")
        say("  information needed to fix recall is not here at all, which is the honest end of this line.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(lm), str(CELL)], available=len(T), used=int((missed | tn).sum()),
                      selection="filtered", seed=SEED,
                      controls=[f"{N_SHUFFLE} label shuffles per candidate",
                                "expression included so it cannot hide behind another feature",
                                "precision held fixed when comparing recall"],
                      note="compares dependencies the model misses against genes it correctly ignores; "
                           "both score ~0 metabolically, so any separation is information it lacks")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_recall", "manifest": man, "gates": gates,
               "candidates": res, "best_non_expression": best_ne,
               "baseline_precision": base_prec, "baseline_recall": base_rec,
               "recall_base": r_base, "recall_plus": r_plus,
               "n_missed": int(missed.sum()), "n_true_negative": int(tn.sum()), "log": log},
              open(OUT / "loop_recall.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_recall.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
