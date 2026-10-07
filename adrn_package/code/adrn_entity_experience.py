"""PREDECLARED TWO-ARM TEST: does ENTITY + EXPERIENCE beat ENTITY alone?

WHY THIS IS A SEPARATE FILE.  In the Evidence Reasoner v1 table, entity+experience (the minus_relationship arm)
reached 0.2965 on the holdout against chan2a's 0.2885 -- the highest number measured in this project. That arm was
NOT in v1's predeclared contrast set; it was found by reading the table afterwards, which is the selection-on-noise
this project keeps catching in other people's work and must not commit itself. So it is re-run here as a
predeclared two-arm comparison, with the combination weights refitted for two stores rather than inherited from a
three-store fit.

THE GATE, fixed before running:
    entity+experience must beat entity_only past a bootstrap CI on BOTH sealed cohorts.

THE CONTROL THAT DECIDES WHETHER IT MEANS ANYTHING.  In v1, retrieving evidence for a COMPLETELY DIFFERENT
knockout tied the real thing -- the retrieval contributed a generic response prior, not query-specific evidence.
So this run repeats wrong_query and random_evidence for the two-store model, and:

    if entity+experience beats entity_only but wrong_query ALSO beats it by the same margin, the gain is a
    better-calibrated generic prior and must be reported as that, not as evidence retrieval.

Retrieval depth K is swept on TRAINING knockouts only; the sealed cohorts see one value.
"""
import collections, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT, SP = A.OUT, A.SP
BOOT, KSWEEP = 10_000, (5, 10, 20, 40)


def main():
    log = []
    def report(t):
        print(t, flush=True); log.append(t)
    report("=" * 100)
    report("PREDECLARED: does ENTITY + EXPERIENCE beat ENTITY alone?")
    report("=" * 100)
    report("  GATE: must beat entity_only on BOTH cohorts past a bootstrap CI.")
    report("  If wrong_query beats entity_only by the same margin, the gain is a generic prior, not evidence.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]; genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05
    cix = np.where(~tide)[0]; cand = [genes[c] for c in cix]

    sealed, answers = set(), {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists(): sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists(): answers[tag] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]
    universe = sorted(set(kos) | sealed)
    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}

    from sklearn.decomposition import NMF
    X = Amat[[ki[k] for k in train]].copy(); X[:, tide] = 0.0
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32); H = nmf.components_.astype(np.float32)
    w_ent = A.ridge_fit(chan[[urow[k] for k in train]], W, A.PENALTY)
    del X
    Ctr = chan[[urow[k] for k in train]]
    Cn = Ctr / (np.linalg.norm(Ctr, axis=1, keepdims=True) + 1e-9)
    Ptr = Amat[[ki[k] for k in train]][:, cix]
    Ptr = Ptr - Ptr.mean(0, keepdims=True)
    rng = np.random.default_rng(A.SEED)

    def stores(k, kret, mode="real", exclude=None):
        ent = (A.ridge_apply(chan[[urow[k]]], w_ent)[0] @ H)[cix]
        src = train[int(rng.integers(0, len(train)))] if mode == "wrong" else k
        cv = chan[urow[src]]; cv = cv / max(np.linalg.norm(cv), 1e-9)
        sim = Cn @ cv
        if exclude is not None: sim[exclude] = -np.inf
        top = (rng.choice(len(train), size=kret, replace=False) if mode == "random"
               else np.argsort(-sim)[:kret])
        return ent, Ptr[top].mean(0)

    def z_(v):
        s = v.std(); return v / (s if s > 0 else 1)

    # ---- sweep K and fit the two-store weight on TRAINING knockouts only ----
    fit_ko = [k for k in train if (Amat[ki[k]][cix] >= A.TAU).sum() >= 5]
    hold = fit_ko[300:400]; fit_ko = fit_ko[:300]
    tpos = {k: i for i, k in enumerate(train)}
    from sklearn.linear_model import LogisticRegression
    best_k, best_v, best_w = None, -1, None
    for kret in KSWEEP:
        Xf, yf = [], []
        for k in fit_ko:
            e, x = stores(k, kret, exclude=tpos.get(k))
            truth = set(np.where(Amat[ki[k]][cix] >= A.TAU)[0].tolist())
            sel = np.array(list(truth) + rng.choice(len(cix), size=min(len(truth)*20, len(cix)),
                                                    replace=False).tolist())
            Xf.append(np.column_stack([z_(e)[sel], z_(x)[sel]]))
            yf.append(np.isin(sel, list(truth)).astype(np.float32))
        lr = LogisticRegression(max_iter=500).fit(np.concatenate(Xf), np.concatenate(yf))
        w = lr.coef_[0]
        hits = []
        for k in hold:
            e, x = stores(k, kret, exclude=tpos.get(k))
            s = w[0]*z_(e) + w[1]*z_(x)
            truth = set(np.where(Amat[ki[k]][cix] >= A.TAU)[0].tolist())
            hits.append(len(set(np.argsort(-s)[:A.NPRED].tolist()) & truth) / A.NPRED)
        v = float(np.mean(hits))
        report(f"    K={kret:>3}  weights ({w[0]:+.3f}, {w[1]:+.3f})  training-holdout precision {v:.4f}")
        if v > best_v: best_k, best_v, best_w = kret, v, w
    report(f"  -> K={best_k}, weights ({best_w[0]:+.3f}, {best_w[1]:+.3f}) selected on TRAINING only")

    ARMS = ("entity_experience", "entity_only", "experience_only", "wrong_query", "random_evidence")
    summary = {}
    for tag, lab in (("", "cohort 1"), ("_holdout", "cohort 2 (holdout)")):
        ans = answers.get(tag)
        if not ans: continue
        cohort = sorted(ans)
        per = {a: [] for a in ARMS}
        for k in cohort:
            truth = set(ans[k]["movers"])
            e, x = stores(k, best_k)
            _, xw = stores(k, best_k, "wrong")
            _, xr = stores(k, best_k, "random")
            S = {"entity_experience": best_w[0]*z_(e) + best_w[1]*z_(x),
                 "entity_only": z_(e),
                 "experience_only": z_(x),
                 "wrong_query": best_w[0]*z_(e) + best_w[1]*z_(xw),
                 "random_evidence": best_w[0]*z_(e) + best_w[1]*z_(xr)}
            for a, s in S.items():
                per[a].append(len({cand[t] for t in np.argsort(-s)[:A.NPRED]} & truth) / A.NPRED)
        report(f"\n### {lab}  ({len(cohort)} knockouts)")
        v = {a: np.array(x) for a, x in per.items()}
        for a in ARMS: report(f"  {a:<20} {v[a].mean():.4f}")
        out = {}
        for nm, a_, b_ in (("GATE ent+exp - entity_only", "entity_experience", "entity_only"),
                           ("wrong_query - entity_only", "wrong_query", "entity_only"),
                           ("ent+exp - wrong_query", "entity_experience", "wrong_query"),
                           ("ent+exp - random_evidence", "entity_experience", "random_evidence")):
            d = v[a_] - v[b_]
            br = np.random.default_rng(0)
            bs = d[br.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            report(f"  {nm:<28} {d.mean():>+8.4f} [{lo:+.4f}, {hi:+.4f}]  "
                   f"{'RESOLVED' if (lo>0 or hi<0) else 'within noise'}")
            out[nm] = {"gap": float(d.mean()), "ci": [lo, hi]}
        summary[tag or "cohort1"] = {"means": {a: float(x.mean()) for a, x in v.items()},
                                     "contrasts": out}
    json.dump({"test": "adrn_entity_experience", "K": best_k, "weights": best_w.tolist(),
               "summary": summary, "log": log},
              open(OUT / "adrn_entity_experience.json", "w"), indent=2)
    report(f"\n  -> {OUT/'adrn_entity_experience.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
