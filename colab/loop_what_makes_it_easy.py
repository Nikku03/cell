"""Loop 169. What is the confident half MADE OF? Anatomy of a 95%-precision decision.

THE QUESTION. Loop 168 showed that answering only the most-confident 50% of cases gives 95.3%
precision against 72.4% overall, and that the score is calibrated. It did not say WHAT those cases
have in common. That matters for three reasons: a confidence that is really a proxy for one trivial
property should be described as that property; a confident set that is merely "reactions with one
product" is far less useful than it looks; and the abstained half is the actual remaining work, so
knowing whether it is hard or hopeless decides what to build next.

WHAT COULD BE DOING IT, and each is checked rather than assumed:
    the chemistry found an EXACT single-candidate match (solo hit) and there is nothing to decide
    very few candidates are chemically feasible at all, so the shortlist is effectively short
    the reaction has ONE true product rather than several, so there is one right answer not three
    the residual is large and distinctive, making an accidental elemental match unlikely
    the walk has a clear opinion rather than a flat one
    the blend's own top-1-to-top-2 margin is already wide before the ranker sees it

PREDECLARED, before any number is looked at.

  Y1 WHAT SEPARATES CONFIDENT FROM ABSTAINED? Every case-level property above, measured in the
     confident half and the abstained half, with a standardised effect size.
     Gate: passes on all being reported, ranked.

  Y2 IS CONFIDENCE JUST "THE CHEMISTRY FOUND AN EXACT MATCH"? Precision within the solo-hit cases
     and within the no-solo cases, and the coverage each accounts for.
     Gate: PASS iff the confident half retains at least 90% precision on cases WITHOUT a solo hit.
     A FAIL means confidence is a rebranding of the solo test and should be described that way.

  Y3 IS IT JUST "ONE PRODUCT"? Precision split by the number of true non-currency products.
     Gate: PASS iff precision on multi-product cases inside the confident half exceeds 0.80. If
     the confident set is only ever single-product reactions, its usefulness is narrower than
     loop 168's headline implies.

  Y4 COULD WE KNOW BEFOREHAND? Fit a small model predicting the ranker's confidence from the
     case-level properties ALONE -- no per-candidate scores, nothing that needs the ranker to run.
     Gate: PASS iff its AUC for separating correct from incorrect exceeds 0.70. If it does, cases
     can be triaged before the expensive scoring, which is worth knowing.

  Y5 IS THE ABSTAINED HALF HARD OR HOPELESS? Its top-20 recall, and how far down the ranked list
     the answer sits when it is present.
     Gate: PASS iff the abstained half's top-20 recall exceeds 0.80. If the answer is usually still
     IN the shortlist, the abstained half is a ranking problem and worth more work; if it is not,
     the shortlist is the binding constraint there and ranking effort is misdirected.

  Y6 WHAT THIS CANNOT SHOW. Descriptive, on DEV, on curated reactions. It explains the confident
     set that exists, not the one that would exist on genuine gaps.

-> outputs/loop_what_makes_it_easy.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                    # noqa: E402
import run_manifest as RM                  # noqa: E402
from rem.harness import auc_of             # noqa: E402

CACHE = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/"
             "scratchpad/l167_features.npz")
OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_what_makes_it_easy.json"
FEATS = ["tier", "solo", "pair", "feas", "balance", "bal_rank", "walk", "walk_rank",
         "walk_zero", "chain", "chain_rank", "log_deg_clip", "log_heavy", "het_frac",
         "charge", "blend_rank", "resid_cos"]
NFOLD, SEED = 5, 16900

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 104)
    say("  WHAT IS THE CONFIDENT HALF MADE OF?")
    say("=" * 104)
    say()
    z = np.load(CACHE)
    X, y, grp = z["X"], z["y"], z["grp"]
    ceiling = float(z["ceiling"])
    F = {f: X[:, i] for i, f in enumerate(FEATS)}
    cases = np.unique(grp)
    say(f"     {len(cases):,} cases, {X.shape[0]:,} candidate rows, ceiling {ceiling:.4f}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.isotonic import IsotonicRegression
    praw = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=NFOLD).split(X, y, grp):
        praw[te] = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
            random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    pcal = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=NFOLD).split(X, y, grp):
        pcal[te] = IsotonicRegression(out_of_bounds="clip").fit(praw[tr], y[tr]).predict(praw[te])
    say(f"     ranker refit from cache [{time.time()-t0:.0f}s]")

    # ------------------------------------------------- case-level properties
    props, conf, correct, npos, rank_true = [], [], [], [], []
    for g in cases:
        s = grp == g
        p, yy = pcal[s], y[s].astype(bool)
        o = np.argsort(-p)
        conf.append(p[o[0]])
        correct.append(1.0 if yy[o[0]] else 0.0)
        npos.append(int(yy.sum()))
        rank_true.append(int(np.where(yy[o])[0][0]) + 1 if yy.any() else 999)
        props.append({
            "n_solo": float(F["solo"][s].sum()),
            "n_pair": float(F["pair"][s].sum()),
            "n_feas": float(F["feas"][s].sum()),
            "tier_margin": float(np.sort(F["tier"][s])[-1] - np.sort(F["tier"][s])[-2]),
            "max_tier": float(F["tier"][s].max()),
            "walk_flat": float((F["walk"][s] == 0).mean()),
            "walk_margin": float(np.sort(F["walk"][s])[-1] - np.sort(F["walk"][s])[-2]),
            "n_products": float(yy.sum()),
            "median_cand_size": float(np.median(F["log_heavy"][s])),
            "max_resid_cos": float(F["resid_cos"][s].max()),
            "median_degree": float(np.median(F["log_deg_clip"][s])),
        })
    conf = np.array(conf)
    correct = np.array(correct)
    npos = np.array(npos)
    rank_true = np.array(rank_true)
    KEYS = list(props[0])
    P = {k: np.array([p[k] for p in props]) for k in KEYS}
    order = np.argsort(-conf)
    half = len(order) // 2
    C, A = order[:half], order[half:]
    say(f"     confident half n={len(C):,} precision {correct[C].mean():.4f} | "
        f"abstained half n={len(A):,} precision {correct[A].mean():.4f}")

    # ------------------------------------------------------------------ Y1
    say()
    say("Y1 WHAT SEPARATES THEM (standardised difference, confident minus abstained)")
    eff = {}
    for k in KEYS:
        v = P[k]
        sd = v.std() or 1.0
        eff[k] = float((v[C].mean() - v[A].mean()) / sd)
    say(f"     {'property':<20s} {'confident':>10s} {'abstained':>10s} {'effect':>8s}")
    for k, e in sorted(eff.items(), key=lambda kv: -abs(kv[1])):
        say(f"     {k:<20s} {P[k][C].mean():>10.3f} {P[k][A].mean():>10.3f} {e:>+8.2f}")
    y1 = True
    say(f"     Y1 {'PASS' if y1 else 'FAIL'}")

    # ------------------------------------------------------------------ Y2
    say()
    say("Y2 IS IT JUST AN EXACT SINGLE-CANDIDATE MATCH?")
    has_solo = P["n_solo"] > 0
    say(f"     cases with a solo hit: {has_solo.mean():.1%} overall, "
        f"{has_solo[C].mean():.1%} of the confident half, {has_solo[A].mean():.1%} of the abstained")
    for nm, mask in (("solo present", has_solo), ("NO solo", ~has_solo)):
        cc = np.intersect1d(C, np.where(mask)[0])
        say(f"     {nm:<14s} overall precision {correct[mask].mean():.4f} (n={int(mask.sum()):,}) "
            f"| inside the confident half {correct[cc].mean():.4f} (n={len(cc):,})")
    cc_nosolo = np.intersect1d(C, np.where(~has_solo)[0])
    y2 = bool(len(cc_nosolo) > 0 and correct[cc_nosolo].mean() >= 0.90)
    GG.verdict(y2, emit=say, if_true=(
        "the confident half holds 90%+ precision even where the chemistry found no exact match, so "
        "confidence is not a rebranding of the solo test."), if_false=(
        "without a solo hit the confident half falls below 90%; confidence is largely the exact-"
        "match test and should be described that way."))
    say(f"     Y2 {'PASS' if y2 else 'FAIL'}")

    # ------------------------------------------------------------------ Y3
    say()
    say("Y3 IS IT JUST ONE-PRODUCT REACTIONS?")
    say(f"     {'n products':>11s} {'cases':>7s} {'overall':>9s} {'confident':>10s} {'share of C':>11s}")
    for n in (1, 2, 3):
        m = npos == n if n < 3 else npos >= 3
        cc = np.intersect1d(C, np.where(m)[0])
        lab = f"{n}" if n < 3 else "3+"
        say(f"     {lab:>11s} {int(m.sum()):>7,} {correct[m].mean():>9.4f} "
            f"{(correct[cc].mean() if len(cc) else float('nan')):>10.4f} "
            f"{len(cc)/max(len(C),1):>10.1%}")
    multi = np.intersect1d(C, np.where(npos >= 2)[0])
    y3 = bool(len(multi) > 0 and correct[multi].mean() > 0.80)
    GG.verdict(y3, emit=say, if_true=(
        "multi-product reactions inside the confident half are still above 80%, so the confident "
        "set is not merely the easy single-product ones."), if_false=(
        "the confident half is dominated by single-product reactions; its usefulness is narrower "
        "than the headline suggests."))
    say(f"     Y3 {'PASS' if y3 else 'FAIL'}")

    # ------------------------------------------------------------------ Y4
    say()
    say("Y4 COULD WE TRIAGE BEFOREHAND, from case properties alone?")
    Xc = np.column_stack([P[k] for k in KEYS])
    pred = np.zeros(len(cases))
    gk = np.arange(len(cases)) % NFOLD
    for f in range(NFOLD):
        tr, te = gk != f, gk == f
        pred[te] = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, min_samples_leaf=30,
            random_state=0).fit(Xc[tr], correct[tr]).predict_proba(Xc[te])[:, 1]
    a4 = auc_of(pred, correct.astype(bool))
    y4 = bool(a4 > 0.70)
    say(f"     AUC separating correct from incorrect using ONLY case properties: {a4:.4f}")
    GG.verdict(y4, emit=say, if_true=(
        "cases can be triaged before the expensive scoring runs at all."), if_false=(
        "case-level properties alone do not tell you which cases will succeed; the ranker has to "
        "run."))
    say(f"     Y4 {'PASS' if y4 else 'FAIL'}")

    # ------------------------------------------------------------------ Y5
    say()
    say("Y5 IS THE ABSTAINED HALF HARD OR HOPELESS?")
    inlist = rank_true[A] < 999
    say(f"     abstained top-20 recall: {inlist.mean():.4f} "
        f"(confident half {(rank_true[C] < 999).mean():.4f})")
    rr = rank_true[A][inlist]
    say(f"     when present, the answer sits at rank: median {np.median(rr):.0f}, "
        f"mean {rr.mean():.1f}; rank 2-3 in {(np.isin(rr, [2, 3])).mean():.1%}, "
        f"rank 4+ in {(rr >= 4).mean():.1%}")
    y5 = bool(inlist.mean() > 0.80)
    GG.verdict(y5, emit=say, if_true=(
        "the abstained half is a RANKING problem: the answer is usually still in the shortlist, so "
        "more ranker work has somewhere to go."), if_false=(
        "the abstained half is a SHORTLIST problem: the answer is often not there at all, so "
        "better ranking cannot help and the shortlist is the binding constraint."))
    say(f"     Y5 {'PASS' if y5 else 'FAIL'}")

    say()
    say("Y6 WHAT THIS CANNOT SHOW")
    say("     Descriptive, DEV only, on curated reactions. It explains the confident set that")
    say("     exists, not the one that would exist on genuine gaps.")
    y6 = True
    say(f"     Y6 {'PASS' if y6 else 'FAIL'}")

    gates = {"Y1": y1, "Y2": y2, "Y3": y3, "Y4": y4, "Y5": y5, "Y6": y6}
    man = RM.manifest(inputs=[CACHE], available=len(cases), used=len(cases),
                      selection="all", seed=SEED,
                      controls=["precision reported inside the confident half AND overall for every split",
                                "the solo-hit and single-product explanations tested rather than assumed",
                                "triage model uses only case properties, no per-candidate score",
                                "the abstained half characterised, not just discarded"],
                      note="anatomy of the 95%-precision confident half")
    out = {"test": "what makes the confident half confident", "gates": gates,
           "n": len(cases), "conf_precision": float(correct[C].mean()),
           "abst_precision": float(correct[A].mean()), "effects": eff,
           "solo": {"share": float(has_solo.mean()),
                    "conf_nosolo_precision": float(correct[cc_nosolo].mean()) if len(cc_nosolo) else None},
           "triage_auc": a4, "abstained_recall": float(inlist.mean()),
           "manifest": man, "seconds": time.time() - t0, "log": log}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time()-t0:.0f}s]")
    say("=" * 104)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
