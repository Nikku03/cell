"""WHAT IS AUC 0.617 ACTUALLY MEASURING?  Decompose it against the floor the raw features already provide.

THE QUESTION.  GeoConv scores 0.617 on interface point-pair discrimination while seeing only each point's K=16
nearest neighbours -- a patch spanning roughly 6 A, about one residue of surface. Before treating 0.617 as
"the model learned something about binding sites", the honest thing is to ask how much of it is available with
NO model at all, from the five per-point features the network is given:

    APBS electrostatic potential | curvature | hydrophobicity | charge | H-bond capability

If a single raw feature, combined by multiplication, already reaches ~0.60, then the entire neural network --
two geodesic convolution layers, a learned Gaussian width, 5,858 parameters, 25 epochs -- is contributing a
couple of points of AUC, and 0.617 means "hydrophobic patches touch hydrophobic patches", which was known
before any of this was built.

WHAT THE TASK REDUCES TO, which matters for reading any number here.  Positives are cross-chain pairs within
4.5 A in the native complex; negatives are pairs beyond 12 A. Both points of a positive pair are therefore IN
the interface, while a negative pair has at least one point outside it. So the task is largely "are BOTH of
these points interface points" -- a per-point property scored through a dot product, not a genuine
complementarity test. A model that merely recognised interface-like patches would do well without ever
learning that two patches FIT.

ARMS, all on the identical eval complexes, points and pairs the architecture A/B used:
    product      f_a * f_b for each feature alone. No learning, no fitting, no parameters.
    similarity   -|f_a - f_b| for each feature alone. Also parameter-free.
    logistic     one logistic regression over all 10 pair features (5 products, 5 abs-diffs), fitted on the
                 TRAIN complexes and evaluated on the same held-out ones. The strongest no-geometry baseline.
    geoconv      0.6171, carried over from dock_arch for reference.

NO COORDINATES ENTER ANY ARM. Every pair score is a function of the two points' FEATURES only -- distance or
position would make the task trivial and measure nothing.

PREDECLARED, before any number:
    best single feature ~ 0.60      -> the network adds almost nothing; 0.617 is local chemistry that was
                                       already in the input, and "GeoConv learns interface structure" is not
                                       a supportable reading.
    best single feature ~ 0.55, logistic ~ 0.58
                                    -> the network adds a real but small margin over hand-combining the same
                                       features.
    all feature arms ~ 0.50         -> the raw features carry nothing pairwise and the network's 0.617 is
                                       genuinely from the geodesic aggregation.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import dock_fft as DF
import dock_poserank as PRK
import dock_rescore as RS
import flex_physics as fp

OUT = A.OUT
N_TRAIN, N_EVAL = 40, 14
N_PTS = 512
SEED = 0
FEATNAMES = ("apbs_potential", "curvature", "hydrophobicity", "charge", "hbond")
GEOCONV_REF = 0.6171          # from dock_arch, same data/split/subsample


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    report("=" * 100)
    report("WHAT IS AUC 0.617 MEASURING? -- decompose it against the parameter-free feature floor")
    report("=" * 100)
    report("  GeoConv sees a ~6 A patch (K=16 neighbours) and 5 per-point features. Before calling 0.617 a")
    report("  learned result, measure how much of it is available with NO model: raw features multiplied.")
    report("  The task is largely 'are BOTH points interface points', since positives are contacts and")
    report("  negatives are far pairs -- not a genuine test that two patches FIT.")
    report("  PREDECLARED: a single raw feature near 0.60 => the network adds almost nothing and 0.617 is")
    report("  local chemistry already present in the input. All arms at 0.50 => the aggregation earned it.")
    report("  NO COORDINATES in any arm -- position would make the task trivial.")

    dockj = json.load(open(OUT / "dock_fft.json"))
    test_pdbs = set(r["pdb"] for r in dockj["complexes"])
    have = sorted(f[:-4] for f in os.listdir(fp.PDBDIR) if f.endswith(".pdb") and "__" not in f)
    rng = np.random.default_rng(SEED)

    pool = {}
    for pdb in have:
        if len(pool) >= N_TRAIN + N_EVAL:
            break
        p = DF.prepare(pdb)
        if p is None or not p["sane"]:
            continue
        chs, ok = {}, True
        for ch in (p["ca"], p["cb"]):
            tag = RS.chain_pdb(pdb, ch)
            r = PRK.cached_surface(tag) if tag else None
            if r is None:
                ok = False
                break
            chs[ch] = r[0]
        if not ok:
            continue
        A_, B_ = chs[p["ca"]], chs[p["cb"]]
        ia = rng.choice(len(A_["pts"]), min(N_PTS, len(A_["pts"])), replace=False)
        ib = rng.choice(len(B_["pts"]), min(N_PTS, len(B_["pts"])), replace=False)
        sa = {k: A_[k][ia] for k in ("pts", "nrm", "feat")}
        sb = {k: B_[k][ib] for k in ("pts", "nrm", "feat")}
        pos, neg = RS.cross_pairs(sa["pts"], sb["pts"])
        if len(pos) < 12 or len(neg) < 12:
            continue
        pool[pdb] = {"fa": sa["feat"], "fb": sb["feat"],
                     "pos": np.array(pos), "neg": np.array(neg)}
    names = sorted(pool)
    ev = [n for n in names if n in test_pdbs][:N_EVAL]
    ev += [n for n in names if n not in test_pdbs and n not in ev][:max(0, N_EVAL - len(ev))]
    tr = [n for n in names if n not in set(ev)][:N_TRAIN]
    report(f"\n  {len(pool)} complexes | train {len(tr)} | eval {len(ev)} (identical split to dock_arch)")

    def pair_arrays(pdbs):
        FA, FB, Y = [], [], []
        for pdb in pdbs:
            c = pool[pdb]
            for arr, lab in ((c["pos"], 1), (c["neg"], 0)):
                FA.append(c["fa"][arr[:, 0]]); FB.append(c["fb"][arr[:, 1]])
                Y.append(np.full(len(arr), lab))
        return np.vstack(FA), np.vstack(FB), np.concatenate(Y)

    FAe, FBe, Ye = pair_arrays(ev)
    FAt, FBt, Yt = pair_arrays(tr)
    report(f"  eval pairs: {int(Ye.sum()):,} contacting / {int((1-Ye).sum()):,} far")

    report(f"\n  PARAMETER-FREE ARMS (one feature, one operation, nothing fitted)")
    report(f"    {'feature':<18}{'product':>10}{'similarity':>13}")
    single = {}
    for i, nm in enumerate(FEATNAMES):
        prod = FAe[:, i] * FBe[:, i]
        simi = -np.abs(FAe[:, i] - FBe[:, i])
        a_p = roc_auc_score(Ye, prod) if np.std(prod) > 1e-12 else float("nan")
        a_s = roc_auc_score(Ye, simi) if np.std(simi) > 1e-12 else float("nan")
        # a feature can be informative in the NEGATIVE direction; report the informative magnitude honestly
        single[nm] = {"product": a_p, "similarity": a_s,
                      "product_flipped": max(a_p, 1 - a_p), "similarity_flipped": max(a_s, 1 - a_s)}
        report(f"    {nm:<18}{a_p:>10.4f}{a_s:>13.4f}")
    report(f"    {'(0.5 = chance; a value BELOW 0.5 is informative with the sign flipped)':<60}")

    best_nm = max(single, key=lambda k: max(single[k]["product_flipped"], single[k]["similarity_flipped"]))
    best_v = max(single[best_nm]["product_flipped"], single[best_nm]["similarity_flipped"])
    report(f"    best single feature, sign-corrected: {best_nm} at {best_v:.4f}")

    Xt = np.hstack([FAt * FBt, np.abs(FAt - FBt)])
    Xe = np.hstack([FAe * FBe, np.abs(FAe - FBe)])
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    lr = LogisticRegression(max_iter=3000).fit((Xt - mu) / sd, Yt)
    a_lr = float(roc_auc_score(Ye, lr.predict_proba((Xe - mu) / sd)[:, 1]))
    report(f"\n  FITTED, NO GEOMETRY: logistic over 10 pair features (5 products + 5 abs-diffs) = {a_lr:.4f}")
    report(f"  GEOCONV (dock_arch, same split): {GEOCONV_REF:.4f}")
    report(f"    network minus best single feature : {GEOCONV_REF - best_v:+.4f}")
    report(f"    network minus fitted logistic     : {GEOCONV_REF - a_lr:+.4f}")

    report("\n  READING")
    if best_v >= 0.58:
        report(f"  0.617 IS MOSTLY THE INPUT. A single raw feature ({best_nm}), multiplied with no model at all,")
        report(f"  reaches {best_v:.4f}. Two geodesic convolution layers, 5,858 parameters and 25 epochs add")
        report(f"  {GEOCONV_REF - best_v:+.4f} on top. What 0.617 means is that interface patches share local")
        report("  chemistry -- which was true before any network existed.")
    elif a_lr >= GEOCONV_REF - 0.02:
        report(f"  0.617 IS HAND-COMBINABLE. A logistic regression on the same features with no geometry at all")
        report(f"  reaches {a_lr:.4f} against the network's {GEOCONV_REF:.4f}. The geodesic aggregation buys")
        report(f"  {GEOCONV_REF - a_lr:+.4f}, so what the network adds over weighting the raw features is small.")
    else:
        report(f"  THE AGGREGATION EARNS IT. Best single feature {best_v:.4f}, fitted logistic {a_lr:.4f}, both")
        report(f"  well under the network's {GEOCONV_REF:.4f}. The K=16 neighbourhood is contributing real")
        report("  structure beyond what the per-point features contain.")
    report(f"\n  Either way 0.617 is a WEAK number in absolute terms: on a task whose negatives are pairs >12 A")
    report("  apart -- the easy negatives -- it ranks the contacting pair first barely 3 times in 5.")

    json.dump({"test": "dock_feature_floor", "n_train": len(tr), "n_eval": len(ev),
               "n_pos": int(Ye.sum()), "n_neg": int((1 - Ye).sum()), "single_feature": single,
               "best_single": best_nm, "best_single_auc": best_v, "logistic_auc": a_lr,
               "geoconv_ref": GEOCONV_REF, "log": log},
              open(OUT / "dock_feature_floor.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'dock_feature_floor.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
