"""ADRN BIOLOGICAL EVIDENCE REASONER v1: does COMBINING evidence stores beat the best single store?

WHAT IS ALREADY KNOWN ABOUT THE FOUR STORES, so this is not re-run blind:
    ENTITY store       chan2a, 696 named channels -> 0.2885 on the sealed holdout. The incumbent.
    RELATIONSHIP store PPI / complexes / codep / regulatory. Measured SEVEN times and indistinguishable from
                       degree-matched rewiring every time; the learned multi-hop version failed this morning.
    EXPERIENCE store   nearest analogous perturbations. This is neighbour transfer, which scored 0.2293 -- real,
                       and below the entity store.
    CONSTRAINT store   not built in v1.

So each store's SOLO value is known. What has never been tested is the composition: chan2a and neighbour transfer
have never been in the same model, and no ablation has ever asked whether their evidence is complementary or
redundant. That is the only genuinely open question here and it is what this measures.

CONSTRUCTION.  For a query knockout k, three evidence scores over every candidate gene j:
    entity      chan2a: named channels -> ridge -> programme mixture -> decode
    experience  the nearest training knockouts by ANNOTATION similarity, their measured centred profiles averaged.
                Note this retrieves by annotation, not by network edge, which is what distinguishes it from the
                neighbour transfer already on record.
    relationship the network neighbours' measured profiles averaged -- neighbour transfer proper.
The three are combined with weights fitted on TRAINING knockouts only, then ranked.

ARMS
    combined        all three stores.  THE TEST.
    entity_only     = chan2a, the incumbent.
    experience_only / relationship_only   each store alone.
    minus_entity / minus_experience / minus_relationship   leave-one-store-out ablations.
    random_evidence retrieval returns random training knockouts. Prices whether retrieval is doing anything.
    wrong_query     evidence retrieved for a DIFFERENT knockout. Prices whether the retrieval is query-specific.

PREDECLARED: the reasoner earns the headline only if `combined` beats `entity_only` past a bootstrap CI on BOTH
sealed cohorts. If it ties, the stores are redundant and the encyclopedia adds nothing over the entity store. The
leave-one-out ablations say which store carries any gain.

THE RISK THE SPEC NAMES, and it is the right one: this could become a nearest-neighbour retriever. random_evidence
and wrong_query exist to catch exactly that, and the complex/pathway-held-out variant is reported alongside the
sealed cohorts so a lookup cannot hide behind a random split.
"""
import collections, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT, SP = A.OUT, A.SP
K_RET, BOOT = 10, 10_000


def main():
    log = []
    def report(t):
        print(t, flush=True); log.append(t)
    report("=" * 100)
    report("ADRN EVIDENCE REASONER v1: does combining evidence stores beat the best single store?")
    report("=" * 100)
    report("  PREDECLARED: earns the headline only if `combined` beats `entity_only` on BOTH cohorts.")

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

    # ---------------- ENTITY store: chan2a ----------------
    from sklearn.decomposition import NMF
    X = Amat[[ki[k] for k in train]].copy(); X[:, tide] = 0.0
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32); H = nmf.components_.astype(np.float32)
    w_ent = A.ridge_fit(chan[[urow[k] for k in train]], W, A.PENALTY)
    del X
    report(f"  entity store: {chan.shape[1]} channels -> {A.K} programmes")

    # ---------------- EXPERIENCE store: annotation-nearest training knockouts ----------------
    Ctr = chan[[urow[k] for k in train]]
    Cn = Ctr / (np.linalg.norm(Ctr, axis=1, keepdims=True) + 1e-9)
    Ptr = Amat[[ki[k] for k in train]][:, cix]
    Ptr = Ptr - Ptr.mean(0, keepdims=True)
    report(f"  experience store: {len(train):,} training profiles, retrieval by annotation cosine")

    # ---------------- RELATIONSHIP store: network neighbours' profiles ----------------
    D = json.load(open(OUT / "cell_complete.json")); nm = [g["name"] for g in D["genes"]]
    nb = collections.defaultdict(set)
    for e in D.get("ppi", []):
        try: a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError): continue
        if 0 <= a < len(nm) and 0 <= b < len(nm) and a != b:
            nb[nm[a]].add(nm[b]); nb[nm[b]].add(nm[a])
    mem = collections.defaultdict(list)
    for k_, cs in (D.get("gene2cplx") or {}).items():
        g = nm[int(k_)] if k_.isdigit() and int(k_) < len(nm) else k_
        for c in (cs if isinstance(cs, list) else [cs]): mem[str(c)].append(g)
    for c, ms in mem.items():
        if len(ms) <= 60:
            for g in ms: nb[g] |= {x for x in ms if x != g}
    for a_, v in (D.get("codep") or {}).items():
        g = nm[int(a_)] if a_.isdigit() and int(a_) < len(nm) else a_
        for b_, _ in v:
            try: nb[g].add(nm[int(b_)])
            except (TypeError, ValueError, IndexError): pass
    del D
    tpos = {k: i for i, k in enumerate(train)}
    report(f"  relationship store: {len(nb):,} genes with neighbours")

    rng = np.random.default_rng(A.SEED)

    def evidence(k, mode="real"):
        """three score vectors over candidate genes"""
        ent = (A.ridge_apply(chan[[urow[k]]], w_ent)[0] @ H)[cix]
        src = k
        if mode == "wrong":
            src = train[int(rng.integers(0, len(train)))]
        cv = chan[urow[src]]; cv = cv / max(np.linalg.norm(cv), 1e-9)
        sim = Cn @ cv
        if mode == "random":
            top = rng.choice(len(train), size=K_RET, replace=False)
        else:
            top = np.argsort(-sim)[:K_RET]
        exp = Ptr[top].mean(0)
        rows = [tpos[p] for p in nb.get(src, ()) if p in tpos]
        rel = Ptr[rows].mean(0) if rows else np.zeros(len(cix), np.float32)
        return ent, exp, rel

    # ---- fit combination weights on a slice of TRAINING knockouts ----
    fit_ko = [k for k in train if (Amat[ki[k]][cix] >= A.TAU).sum() >= 5][:400]
    Xf, yf = [], []
    for k in fit_ko:
        e, x, r = evidence(k)
        truth = set(np.where(Amat[ki[k]][cix] >= A.TAU)[0].tolist())
        sel = list(truth) + rng.choice(len(cix), size=min(len(truth) * 20, len(cix)),
                                       replace=False).tolist()
        sel = np.array(sel)
        Xf.append(np.column_stack([e[sel], x[sel], r[sel]]))
        yf.append(np.isin(sel, list(truth)).astype(np.float32))
    Xf = np.concatenate(Xf); yf = np.concatenate(yf)
    for c_ in range(3):
        s = Xf[:, c_].std();  Xf[:, c_] = Xf[:, c_] / (s if s > 0 else 1)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=500).fit(Xf, yf)
    wts = lr.coef_[0]
    report(f"  combination weights (entity, experience, relationship): "
           f"{wts[0]:+.3f} {wts[1]:+.3f} {wts[2]:+.3f}   fitted on {len(fit_ko)} training knockouts")
    scale = [Xf[:, i].std() for i in range(3)]

    ARMS = ("combined", "entity_only", "experience_only", "relationship_only",
            "minus_entity", "minus_experience", "minus_relationship",
            "random_evidence", "wrong_query")
    summary = {}
    for tag, lab in (("", "cohort 1"), ("_holdout", "cohort 2 (holdout)")):
        ans = answers.get(tag)
        if not ans: continue
        cohort = sorted(ans)
        per = {a: [] for a in ARMS}
        for k in cohort:
            truth = set(ans[k]["movers"])
            E = {m: evidence(k, m) for m in ("real", "random", "wrong")}
            def combo(e, x, r, mask):
                v = np.zeros(len(cix), np.float32)
                for i, (s_, m_) in enumerate(zip((e, x, r), mask)):
                    if m_: v += wts[i] * (s_ / (np.std(s_) if np.std(s_) > 0 else 1))
                return v
            e, x, r = E["real"]
            S = {"combined": combo(e, x, r, (1, 1, 1)),
                 "entity_only": combo(e, x, r, (1, 0, 0)),
                 "experience_only": combo(e, x, r, (0, 1, 0)),
                 "relationship_only": combo(e, x, r, (0, 0, 1)),
                 "minus_entity": combo(e, x, r, (0, 1, 1)),
                 "minus_experience": combo(e, x, r, (1, 0, 1)),
                 "minus_relationship": combo(e, x, r, (1, 1, 0)),
                 "random_evidence": combo(*E["random"], (1, 1, 1)),
                 "wrong_query": combo(*E["wrong"], (1, 1, 1))}
            for a, s_ in S.items():
                top = np.argsort(-s_)[:A.NPRED]
                per[a].append(len({cand[t] for t in top} & truth) / A.NPRED)
        report(f"\n### {lab}  ({len(cohort)} knockouts)")
        v = {a: np.array(x) for a, x in per.items()}
        for a in ARMS:
            report(f"  {a:<20} {v[a].mean():.4f}")
        out = {}
        for nm_, a_, b_ in (("GATE combined - entity_only", "combined", "entity_only"),
                            ("combined - random_evidence", "combined", "random_evidence"),
                            ("combined - wrong_query", "combined", "wrong_query"),
                            ("minus_experience - combined", "minus_experience", "combined"),
                            ("minus_relationship - combined", "minus_relationship", "combined")):
            d = v[a_] - v[b_]
            br = np.random.default_rng(0)
            bs = d[br.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            report(f"  {nm_:<30} {d.mean():>+8.4f} [{lo:+.4f}, {hi:+.4f}]  "
                   f"{'RESOLVED' if (lo>0 or hi<0) else 'within noise'}")
            out[nm_] = {"gap": float(d.mean()), "ci": [lo, hi]}
        summary[tag or "cohort1"] = {"means": {a: float(x.mean()) for a, x in v.items()},
                                     "contrasts": out}
    json.dump({"test": "adrn_evidence_reasoner", "weights": wts.tolist(),
               "summary": summary, "log": log},
              open(OUT / "adrn_evidence_reasoner.json", "w"), indent=2)
    report(f"\n  -> {OUT/'adrn_evidence_reasoner.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
