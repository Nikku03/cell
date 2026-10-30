"""DOES IT LEARN BIOLOGY, OR LOOK UP SPECIFIC GENE FUNCTIONS?  Leave-one-functional-class-out.

THE QUESTION.  The source side is 696 annotation channels, and the obvious worry is that the model has learned
"genes annotated ribosomal produce the ribosomal response" -- a per-class lookup that would collapse the moment a
knockout's function is absent from training.  That is a different object from a model that has learned structure
transferable ACROSS functions.  The off-annotation test showed 14.6-16.8% of correct calls involve genes sharing
no annotation with the knockout, which argues against pure lookup on the TARGET side.  This asks the same question
on the SOURCE side, which is where the lookup worry actually lives.

THE DESIGN, and the control is what makes it a test rather than a demonstration.

    LOCO      remove EVERY training knockout in functional class c, refit, and score the sealed knockouts that
              belong to c.  The model must predict a class it has never seen perturbed.
    CONTROL   remove the SAME NUMBER of training knockouts drawn at random from all classes, refit, score the
              same sealed knockouts.  This holds the training-set SIZE fixed, so any gap is about WHICH genes
              were removed rather than how many.

Without that control the comparison is worthless: removing a class also removes rows, and a smaller training set
scores worse for reasons that have nothing to do with function.

    LOCO ~= CONTROL   removing a whole function costs no more than removing random genes -> the model transfers
                      across functional boundaries, which is the "learns biology" answer.
    LOCO << CONTROL   the class was carrying its own prediction -> per-class lookup, and the sealed numbers are
                      inflated by every class being represented in training.

CLASSES are defined by k-means on the channel matrix itself -- the annotation geometry the model actually sees,
not an external taxonomy that it might not use.  Cluster count is swept because "how coarse is a function" is
exactly the kind of undeliberated default this project has been burned by twice.

PREDECLARED: this is judged by robustness.Sweeper on (LOCO - CONTROL). HARMFUL or a resolved negative majority
means per-class lookup. NULL or DIRECTIONAL-near-zero means the model transfers across functions. Note the sign
convention: a gap at or above zero is the GOOD outcome here, which is the reverse of every other sweep in this
project, so the ladder's ROBUST label would mean LOCO BEATS the control -- report it as transfer, not as a win.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
N_CLUSTERS = (8, 16, 32)
K_VALUES = (30, 60, 120)
PENALTIES = (1.0, 10.0, 100.0, 1000.0)
MIN_SEALED = 3


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("LEAVE-ONE-FUNCTIONAL-CLASS-OUT: biology, or per-class lookup?")
    report("=" * 100)
    report("  PREDECLARED: LOCO ~= size-matched CONTROL means it transfers across functions.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    gpos = {g: i for i, g in enumerate(genes)}
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05

    sealed, answers = set(), {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists():
            answers[tag] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]
    universe = sorted(set(kos) | sealed)
    urow = {g: i for i, g in enumerate(universe)}

    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)

    cohorts = {}
    for tag in ("", "_holdout"):
        if answers.get(tag):
            nm = tag or "cohort1"
            ks = sorted(answers[tag])
            cohorts[nm] = (ks, {k: set(answers[tag][k]["movers"]) for k in ks})

    from sklearn.cluster import KMeans
    from sklearn.decomposition import NMF

    Xall = Amat[[ki[k] for k in train]].copy()
    Xall[:, tide] = 0.0
    tidx = {k: i for i, k in enumerate(train)}

    sw = Sweeper("LOCO - size-matched CONTROL", axes={"clusters": list(N_CLUSTERS), "K": list(K_VALUES)})
    detail = {}

    for K in K_VALUES:
        nmf = NMF(n_components=K, init="nndsvda", max_iter=400, random_state=0)
        Wall = nmf.fit_transform(Xall).astype(np.float32)
        H = nmf.components_.astype(np.float32)

        def score(w, keys):
            s = A.ridge_apply(chan[[urow[k] for k in keys]], w) @ H
            s[:, tide] = -np.inf
            return s

        def prec(w, keys, truth):
            s = score(w, keys)
            out = []
            for j, k in enumerate(keys):
                sc = s[j].copy()
                if k in gpos:
                    sc[gpos[k]] = -np.inf
                out.append(len({genes[t] for t in np.argsort(-sc)[:A.NPRED]} & truth[k]) / float(A.NPRED))
            return np.array(out)

        def fit_on(sub):
            """Penalty chosen on a TRAIN-ONLY slice.

            The first version of this scored each candidate penalty on the sealed cohorts and kept the best --
            tuning on the test set, which would have inflated every number here and quietly broken the sealed
            protocol that the rest of the project depends on. The validation slice is carved out of `sub` itself,
            so it also shrinks correctly when a class is removed.
            """
            sub = np.asarray(sub)
            r = np.random.default_rng(A.SEED + 3)
            vp = r.permutation(len(sub))
            nv = max(50, int(0.2 * len(sub)))
            vsub, fsub = sub[vp[:nv]], sub[vp[nv:]]
            vkeys = [train[i] for i in vsub]
            vtruth = {k: {genes[i] for i in np.where(Amat[ki[k]] >= A.TAU)[0] if not tide[i]} for k in vkeys}
            best_v, best_p = -1.0, PENALTIES[0]
            for p in PENALTIES:
                w = A.ridge_fit(chan[[urow[train[i]] for i in fsub]], Wall[fsub], p)
                v = float(prec(w, vkeys, vtruth).mean())
                if v > best_v:
                    best_v, best_p = v, p
            # refit on the WHOLE subset at the chosen penalty -- the validation slice was only for choosing it
            return A.ridge_fit(chan[[urow[train[i]] for i in sub]], Wall[sub], best_p)

        for ncl in N_CLUSTERS:
            km = KMeans(n_clusters=ncl, n_init=4, random_state=0).fit(chan[[urow[k] for k in train]])
            lab_tr = km.labels_
            lab_sealed = {nm: km.predict(chan[[urow[k] for k in ks]]) for nm, (ks, _) in cohorts.items()}
            rng = np.random.default_rng(A.SEED + ncl)
            acc = {nm: {"loco": [], "ctrl": []} for nm in cohorts}
            used = 0
            for c in range(ncl):
                inc = np.where(lab_tr == c)[0]
                if len(inc) < 20:
                    continue
                keep = np.where(lab_tr != c)[0]
                w_loco = fit_on(keep)
                drop = rng.choice(len(train), len(inc), replace=False)
                keep2 = np.setdiff1d(np.arange(len(train)), drop)
                w_ctrl = fit_on(keep2)
                for nm, (ks, truth) in cohorts.items():
                    m = lab_sealed[nm] == c
                    if m.sum() < MIN_SEALED:
                        continue
                    sub = [k for k, t in zip(ks, m) if t]
                    acc[nm]["loco"].append(prec(w_loco, sub, truth))
                    acc[nm]["ctrl"].append(prec(w_ctrl, sub, truth))
                used += 1
            cfg = {"clusters": ncl, "K": K}
            for nm in cohorts:
                if not acc[nm]["loco"]:
                    continue
                lo = np.concatenate(acc[nm]["loco"])
                ct = np.concatenate(acc[nm]["ctrl"])
                sw.add(cfg, nm, lo - ct)
                detail[f"K={K}|ncl={ncl}|{nm}"] = {"n": int(len(lo)), "loco": float(lo.mean()),
                                                   "control": float(ct.mean()), "clusters_used": used}
            report(f"  K={K} clusters={ncl}: {used} classes held out | " + " ".join(
                f"{nm} LOCO {detail[f'K={K}|ncl={ncl}|{nm}']['loco']:.4f} vs ctrl "
                f"{detail[f'K={K}|ncl={ncl}|{nm}']['control']:.4f}" for nm in cohorts
                if f"K={K}|ncl={ncl}|{nm}" in detail))

    report(sw.report())
    v = sw.verdict()
    report("\n  READING:")
    if v["verdict"] in ("HARMFUL",):
        report("  Removing a whole function costs MORE than removing the same number of random genes.")
        report("  -> the class was carrying its own prediction: PER-CLASS LOOKUP.")
    elif v["verdict"] == "NULL":
        report("  Removing a whole function costs NO MORE than removing random genes of the same count.")
        report("  -> the model TRANSFERS ACROSS FUNCTIONAL BOUNDARIES. This is the 'learns biology' answer.")
    else:
        report(f"  {v['verdict']}: read the per-cell signs before concluding -- a positive gap means LOCO beat")
        report("  the control, which would be odd and worth explaining rather than celebrating.")

    json.dump({"test": "adrn_loco", "clusters": list(N_CLUSTERS), "K_values": list(K_VALUES),
               "detail": detail, "sweep": v, "log": log},
              open(OUT / "adrn_loco.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'adrn_loco.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
