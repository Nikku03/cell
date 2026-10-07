"""SOURCE-SIDE STRUCTURED HOLDOUT: do trees beat ridge when the SOURCE split is hard, as they do for targets?

WHY.  Nine source encoders converged within 0.008 -- but every one of those comparisons used the fixed sealed
cohorts, which are selected by functional-class round-robin, i.e. effectively a RANDOM source split.  On the
TARGET side the encoders also tied on a random gene split and then separated sharply once whole complexes were
held out (gbm - ridge +0.006 random -> +0.024 complex, RESOLVED).  So the source-side convergence may be an
artefact of never having run a hard source split.  This runs one.

CONSTRUCTION.  Group the 4,720 training knockouts by protein complex and by pathway, hold out whole groups, and
refit EVERYTHING per fold: the programme basis B on the training-fold sources only, the encoder on the same, then
decode held-out sources through B_f and score precision@20 against their measured movers.  A random source split
is run alongside so the two are read against each other exactly as on the target side.

PREDECLARED.  If gbm - ridge is within noise on all three splits, the source-side convergence is real and the
target-side non-linearity is specific to susceptibility.  If gbm separates as the split hardens, the earlier
'encoders do not matter' conclusion was an artefact of easy splits and needs stating as such.
"""
from __future__ import annotations
import collections, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT, SP = A.OUT, A.SP
K, FOLDS, BOOT, MIN_SPEC = 64, 5, 10_000, 5


def main() -> int:
    log = []
    def report(t):
        print(t, flush=True); log.append(t)
    report("=" * 100)
    report("SOURCE-SIDE STRUCTURED HOLDOUT: do trees separate from ridge when the split is hard?")
    report("=" * 100)

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]; genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05
    keep_g = np.where(~tide)[0]; gname = [genes[c] for c in keep_g]

    sealed = set()
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists(): sealed |= set(json.loads(f.read_text()))
    train = [k for k in kos if k not in sealed]
    universe = sorted(set(kos) | sealed)
    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}

    Xall = Amat[[ki[k] for k in train]][:, keep_g]
    movers = [{gname[c] for c in np.where(Xall[i] >= A.TAU)[0]} for i in range(len(train))]
    scorable = np.array([len(m) >= MIN_SPEC for m in movers])
    report(f"  {len(train):,} training sources, {int(scorable.sum()):,} with >= {MIN_SPEC} movers "
           f"(only these are scored)")

    D = json.load(open(OUT / "cell_complete.json")); nm = [g["name"] for g in D["genes"]]
    cplx = {}
    for k_, cs in (D.get("gene2cplx") or {}).items():
        try: g = nm[int(k_)]
        except (TypeError, ValueError, IndexError): continue
        lst = sorted(str(c) for c in (cs if isinstance(cs, list) else [cs]))
        if lst: cplx[g] = lst[0]
    del D
    path = {}
    for line in (SP / "ReactomePathways.gmt").read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            for g in p[2:]: path.setdefault(g, p[0])
    grp = {"random": [f"__{k}" for k in train],
           "complex": [cplx.get(k, f"__{k}") for k in train],
           "pathway": [path.get(k, f"__{k}") for k in train]}

    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    rows = np.array([urow[k] for k in train])
    summary = {}
    for split in ("random", "complex", "pathway"):
        labels = np.array(grp[split]); uniq = sorted(set(labels.tolist()))
        rg = np.random.default_rng(A.SEED + 7)
        assign = {uniq[int(u)]: i % FOLDS for i, u in enumerate(rg.permutation(len(uniq)))}
        fold_of = np.array([assign[l] for l in labels])
        per = collections.defaultdict(list)
        for f in range(FOLDS):
            te = np.where((fold_of == f) & scorable)[0]; tr = np.where(fold_of != f)[0]
            if len(tr) < 200 or not len(te): continue
            Xt = Xall[tr]; Tj = Xt.mean(0, keepdims=True)
            Rt = Xt - Tj; Rt = Rt - Rt.mean(1, keepdims=True)
            Bf = TruncatedSVD(n_components=K, random_state=0).fit(Rt).components_.astype(np.float32)
            Af = (Rt @ Bf.T).astype(np.float32)
            Ctr, Cte = chan[rows[tr]], chan[rows[te]]
            preds = {"ridge": A.ridge_apply(Cte, A.ridge_fit(Ctr, Af, A.PENALTY)),
                     "knn": KNeighborsRegressor(n_neighbors=10).fit(Ctr, Af).predict(Cte)}
            gb = np.zeros((len(te), K), np.float32)
            for j in range(K):
                gb[:, j] = HistGradientBoostingRegressor(max_iter=80, random_state=0)\
                    .fit(Ctr, Af[:, j]).predict(Cte)
            preds["gbm"] = gb
            for arm, P in preds.items():
                for r, i in enumerate(te):
                    s = P[r] @ Bf + Tj.ravel()
                    if train[i] in gname: s[gname.index(train[i])] = -np.inf
                    top = np.argsort(-s)[:A.NPRED]
                    per[arm].append(len({gname[t] for t in top} & movers[i]) / A.NPRED)
        report(f"\n### {split} source split  ({len(per['ridge']):,} scored held-out sources)")
        vals = {a: np.array(v) for a, v in per.items()}
        for a in ("gbm", "ridge", "knn"):
            report(f"  {a:<8} precision@20 {vals[a].mean():.4f}")
        out = {}
        for nm_, a_, b_ in (("gbm - ridge", "gbm", "ridge"), ("knn - ridge", "knn", "ridge")):
            d = vals[a_] - vals[b_]
            br = np.random.default_rng(0)
            bs = d[br.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            report(f"  {nm_:<14} {d.mean():>+8.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>22}  "
                   f"{'RESOLVED' if (lo>0 or hi<0) else 'within noise'}")
            out[nm_] = {"gap": float(d.mean()), "ci": [lo, hi], "resolved": bool(lo>0 or hi<0)}
        summary[split] = {"means": {a: float(v.mean()) for a, v in vals.items()},
                          "contrasts": out, "n": int(len(vals["ridge"]))}
    json.dump({"test": "adrn_source_holdout", "summary": summary, "log": log},
              open(OUT / "adrn_source_holdout.json", "w"), indent=2)
    report(f"\n  -> {OUT/'adrn_source_holdout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
