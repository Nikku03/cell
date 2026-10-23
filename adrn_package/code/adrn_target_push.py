"""PUSHING THE TARGET DECODER: is the 59% gap to its ceiling a FEATURE problem or an IRREDUCIBLE one?

WHERE THIS STARTS.  x_j -> B_hat_j reaches 0.1513 against an oracle of 0.3679 -- 41% of the ceiling, the only
axis in this project with unclaimed headroom.  The source side is saturated (nine encoders within 0.008), so the
gap is not the estimator.  It is either the FEATURES or it is irreducible.  Both are testable.

TEST 1 -- FEATURES.  The 696 channels were designed for SOURCES: what makes a knockout impactful.  Nothing in
them describes what makes a gene RESPONSIVE.  Two blocks built earlier for the source side are natively INCOMING
-edge features and belong here instead:

    regulated_by:<TF>   which curated TFs regulate this gene.  On the source side this added nothing (+0.0010,
                        within noise).  For a TARGET it is the direct statement of which programmes can reach it.
    bound_by:<TF>       ChIP occupancy on this gene.  Predicted inert on the source side and was; retested here
                        because occupancy is a property of the target's promoter, not of the knockout.

TEST 2 -- IDENTIFIABILITY, which bounds test 1.  Genes with an IDENTICAL annotation vector must receive an
identical B_hat.  So the cosine between the TRUE B_j of annotation-twins is a hard cap on any decoder: if twins
have unrelated susceptibility, no feature set recovers it, and the remaining 59% is irreducible rather than
missing.  Compared against random gene pairs as the floor.

PREDECLARED.  Features pass only if base+TF beats base past a bootstrap CI on BOTH cohorts.  If twin cosine is
near the random-pair floor, the gap is irreducible and no feature work is worth doing.
"""
from __future__ import annotations
import collections, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
import adrn_ko_channels3 as C3

OUT, SP = A.OUT, A.SP
K, FOLDS, BOOT = 64, 5, 10_000


def main() -> int:
    log = []
    def report(t):
        print(t, flush=True); log.append(t)
    report("=" * 100)
    report("PUSHING THE TARGET DECODER: features or irreducible?")
    report("=" * 100)

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]; genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05
    keep_g = np.where(~tide)[0]; gname = [genes[c] for c in keep_g]

    sealed, answers = set(), {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists(): sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists(): answers[tag] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]
    universe = sorted(set(kos) | sealed | set(gname))
    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    new, _, c_start, p_start = C3.blocks(universe, set(train), report)
    chan_base = np.concatenate([base, extra[:, :tb]], axis=1)
    chan_tf = np.concatenate([chan_base, new[:, :c_start]], axis=1)
    chan_tfchip = np.concatenate([chan_base, new[:, :p_start]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}
    report(f"  target features: base {chan_base.shape[1]} / +TF {chan_tf.shape[1]} / "
           f"+TF+ChIP {chan_tfchip.shape[1]}")

    X = Amat[[ki[k] for k in train]][:, keep_g]
    T_j = X.mean(0, keepdims=True); R = X - T_j; R = R - R.mean(1, keepdims=True)
    from sklearn.decomposition import TruncatedSVD
    B = TruncatedSVD(n_components=K, random_state=0).fit(R).components_.astype(np.float32)
    A_src = (R @ B.T).astype(np.float32)
    w_src = A.ridge_fit(chan_base[[urow[k] for k in train]], A_src, A.PENALTY)

    def gmat(M):
        out = np.zeros((len(gname), M.shape[1]), np.float32)
        idx = [urow[g] for g in gname if g in urow]
        out[[i for i, g in enumerate(gname) if g in urow]] = M[idx]
        return out
    G = {"base": gmat(chan_base), "base_TF": gmat(chan_tf), "base_TF_ChIP": gmat(chan_tfchip)}

    # ---------------- TEST 2: identifiability of B from annotation ----------------
    report(f"\n### IDENTIFIABILITY: do annotation-twins share programme susceptibility?")
    Bn = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-9)
    rr = np.random.default_rng(A.SEED)
    for name, M in G.items():
        groups = collections.defaultdict(list)
        for i in range(len(gname)):
            groups[M[i].tobytes()].append(i)
        twins = [g for g in groups.values() if len(g) > 1]
        cos = []
        for g in twins:
            for a_ in range(min(len(g), 6)):
                for b_ in range(a_ + 1, min(len(g), 6)):
                    cos.append(float(Bn[:, g[a_]] @ Bn[:, g[b_]]))
        ra = rr.integers(0, len(gname), 20000); rb = rr.integers(0, len(gname), 20000)
        rand = float(np.mean([Bn[:, x] @ Bn[:, y] for x, y in zip(ra, rb) if x != y]))
        n_tw = sum(len(g) for g in twins)
        report(f"  {name:<14} {len(twins):>5,} twin groups covering {n_tw:>5,} genes; "
               f"mean cosine(B_twin) {np.mean(cos) if cos else float('nan'):+.4f} "
               f"vs random pairs {rand:+.4f}")

    # ---------------- TEST 1: does the feature block help? ----------------
    D = json.load(open(OUT / "cell_complete.json")); nm = [g["name"] for g in D["genes"]]
    cplx = {}
    for k_, cs in (D.get("gene2cplx") or {}).items():
        try: g = nm[int(k_)]
        except (TypeError, ValueError, IndexError): continue
        lst = sorted(str(c) for c in (cs if isinstance(cs, list) else [cs]))
        if lst: cplx[g] = lst[0]
    del D
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    summary = {}
    for split in ("random", "complex"):
        labels = np.array([f"__{g}" for g in gname]) if split == "random" else \
                 np.array([cplx.get(g, f"__{g}") for g in gname])
        uniq = sorted(set(labels.tolist()))
        rg = np.random.default_rng(A.SEED + 7)
        assign = {uniq[int(u)]: i % FOLDS for i, u in enumerate(rg.permutation(len(uniq)))}
        fold_of = np.array([assign[l] for l in labels])
        arms = {}
        for fname, M in G.items():
            for enc in ("ridge", "knn") + (("gbm",) if fname == "base_TF" else ()):
                Bh = np.zeros_like(B)
                for f in range(FOLDS):
                    te = np.where(fold_of == f)[0]; tr = np.where(fold_of != f)[0]
                    if len(tr) < 100 or not len(te): continue
                    if enc == "ridge":
                        Bh[:, te] = A.ridge_apply(M[te], A.ridge_fit(M[tr], B[:, tr].T, A.PENALTY)).T
                    elif enc == "knn":
                        Bh[:, te] = KNeighborsRegressor(n_neighbors=10).fit(
                            M[tr], B[:, tr].T).predict(M[te]).T
                    else:
                        for j in range(K):
                            Bh[j, te] = HistGradientBoostingRegressor(
                                max_iter=80, random_state=0).fit(M[tr], B[j, tr]).predict(M[te])
                arms[f"{fname}|{enc}"] = Bh
        arms["oracle"] = B
        report(f"\n### {split} gene split")
        for tag, lab in (("", "cohort 1"), ("_holdout", "cohort 2")):
            ans = answers.get(tag)
            if not ans: continue
            cohort = sorted(ans)
            A_hat = A.ridge_apply(chan_base[[urow[k] for k in cohort]], w_src)
            A_true = {}
            for k in cohort:
                row = Amat[ki[k]][keep_g] - T_j.ravel()
                A_true[k] = (row - row.mean()) @ B.T
            res = {a: {"tgt": [], "both": []} for a in arms}
            for f in range(FOLDS):
                te = np.where(fold_of == f)[0]
                if not len(te): continue
                tset = [gname[t] for t in te]
                for j, k in enumerate(cohort):
                    tt = set(ans[k]["movers"]) & set(tset)
                    if len(tt) < 3: continue
                    for a_, Bm in arms.items():
                        for mode, av in (("tgt", A_true[k]), ("both", A_hat[j])):
                            top = np.argsort(-(av @ Bm[:, te]))[:A.NPRED]
                            res[a_][mode].append(len({tset[t] for t in top} & tt) / A.NPRED)
            report(f"  {lab}   {'arm':<22} {'unseen TARGET':>14} {'unseen BOTH':>13}")
            vals = {}
            for a_ in arms:
                v1 = np.array(res[a_]["tgt"]); v2 = np.array(res[a_]["both"])
                vals[a_] = (v1, v2)
                report(f"           {a_:<22} {v1.mean():>14.4f} {v2.mean():>13.4f}")
            report(f"           {'contrast':<30} {'gap':>8} {'95% CI':>22}  verdict")
            out = {}
            for nm_, a_, b_, mi in (("+TF - base (ridge, target)", "base_TF|ridge", "base|ridge", 0),
                                    ("+TF+ChIP - +TF (ridge)", "base_TF_ChIP|ridge", "base_TF|ridge", 0),
                                    ("gbm - ridge (+TF)", "base_TF|gbm", "base_TF|ridge", 0),
                                    ("+TF - base (COMPOSITIONAL)", "base_TF|ridge", "base|ridge", 1)):
                if a_ not in vals or b_ not in vals: continue
                d = vals[a_][mi] - vals[b_][mi]
                if not len(d): continue
                br = np.random.default_rng(0)
                bs = d[br.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
                lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                report(f"           {nm_:<30} {d.mean():>+8.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>22}  "
                       f"{'RESOLVED' if (lo>0 or hi<0) else 'within noise'}")
                out[nm_] = {"gap": float(d.mean()), "ci": [lo, hi]}
            summary[f"{split}|{tag or 'cohort1'}"] = {
                "means": {a: [float(v[0].mean()), float(v[1].mean())] for a, v in vals.items()},
                "contrasts": out}
    json.dump({"test": "adrn_target_push", "summary": summary, "log": log},
              open(OUT / "adrn_target_push.json", "w"), indent=2)
    report(f"\n  -> {OUT/'adrn_target_push.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
