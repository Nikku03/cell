"""MULTI-HOP PATH ASSEMBLY: can a supervised model learn which paths through the cell's networks matter?

THE LAST UNTESTED FORM OF THE 'ASSEMBLE THE MECHANISM FROM THE BOOK' IDEA.  Path-based assembly has been scored on
the sealed protocol four times and the controls inverted every time:

    mmw_shuffled (shuffled network)   0.0151    <- BEATS the real network
    mmw_walk     (Markov multilayer)  0.0113
    mmw_uniform  (no biology at all)  0.0110
    cascade                           0.0183
    map                               0.0031
    multilayer   (mechanistic rules)  0.0338
    frequency baseline                ~0.12-0.18

and the pair tensor found degree-matched rewiring scored IDENTICALLY to the real graph.  But every one of those
used HAND-DESIGNED propagation -- a walk, a cascade, a rule -- and the pair tensor used only DIRECT edges.  Nobody
has handed a supervised learner 2-hop and 3-hop path counts, personalised PageRank and typed path signatures and
asked it to learn which matter.  That is what this does.

THE CONTROL IS THE WHOLE EXPERIMENT.  Every path feature is recomputed on a DEGREE-MATCHED REWIRED graph: each
edge type is rewired by the configuration model so every gene keeps its exact degree in every layer and only the
wiring moves.  If real paths carry mechanism, real must beat rewired.  If they do not, the topology is
uninformative and no amount of learned assembly recovers anything.

ARMS
    resp_only    the responsiveness prior alone (how often gene j moves across training knockouts). The floor,
                 and effectively the frequency baseline that every path method has lost to.
    resp+paths   the prior plus real-graph path features. THE TEST.
    resp+rewire  the prior plus path features from the degree-matched rewired graph. THE CONTROL.
    paths_only   path features without the prior, to see whether they carry anything at all on their own.

PREDECLARED: passes only if resp+paths beats resp+rewire past a bootstrap CI on BOTH sealed cohorts.  Beating
resp_only while tying resp+rewire would mean the model is using degree, which is what the prior already encodes.
"""
import collections, json, os, sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A

OUT, SP = A.OUT, A.SP
NEG_PER_POS, ALPHA, PPR_IT, BOOT = 20, 0.85, 15, 10_000


def layers(nodes, report):
    D = json.load(open(OUT / "cell_complete.json"))
    nm = [g["name"] for g in D["genes"]]
    ix = {g: i for i, g in enumerate(nodes)}
    def gname(k): return nm[int(k)] if k.isdigit() and int(k) < len(nm) else k
    E = collections.defaultdict(list)
    for e in D.get("ppi", []):
        try: a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError): continue
        if 0 <= a < len(nm) and 0 <= b < len(nm) and nm[a] in ix and nm[b] in ix and a != b:
            E["ppi"].append((ix[nm[a]], ix[nm[b]])); E["ppi"].append((ix[nm[b]], ix[nm[a]]))
    mem = collections.defaultdict(list)
    for k, cs in (D.get("gene2cplx") or {}).items():
        g = gname(k)
        if g in ix:
            for c in (cs if isinstance(cs, list) else [cs]): mem[str(c)].append(ix[g])
    for c, m in mem.items():
        if len(m) <= 60:
            for a in m:
                for b in m:
                    if a != b: E["cplx"].append((a, b))
    for e in D.get("reg", []):
        try:
            s, t = int(e[0]), int(e[1]); w = int(e[2]) if len(e) > 2 else 0
        except (TypeError, ValueError, IndexError): continue
        if 0 <= s < len(nm) and 0 <= t < len(nm) and nm[s] in ix and nm[t] in ix and s != t:
            E["tf" if w else "chip"].append((ix[nm[s]], ix[nm[t]]))
    for a, v in (D.get("codep") or {}).items():
        g = gname(a)
        if g in ix:
            for b, _ in v:
                try: h = nm[int(b)]
                except (TypeError, ValueError, IndexError): continue
                if h in ix: E["codep"].append((ix[g], ix[h]))
    del D
    mats = {}
    n = len(nodes)
    for k, ed in E.items():
        if len(ed) < 100: continue
        r = np.array([x[0] for x in ed]); c = np.array([x[1] for x in ed])
        mats[k] = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n))
        mats[k].data[:] = 1.0
    report(f"  layers: " + " · ".join(f"{k} {m.nnz:,} edges" for k, m in mats.items()))
    return mats


def rewire(mats, seed, report):
    rng = np.random.default_rng(seed)
    out = {}
    for k, m in mats.items():
        co = m.tocoo()
        heads, tails = co.row, co.col.copy()
        rng.shuffle(tails)                       # configuration model: degrees preserved, wiring permuted
        out[k] = sp.csr_matrix((np.ones(len(heads), np.float32), (heads, tails)), shape=m.shape)
    report(f"  degree-matched rewiring built for {len(out)} layers")
    return out


def path_rows(mats, i, comb, colnorm):
    """All path features for source i, as dense rows over every node."""
    n = comb.shape[0]
    e = np.zeros(n, np.float32); e[i] = 1.0
    h1 = comb[i].toarray().ravel()
    h2 = (h1 @ comb)
    h3 = (h2 @ comb)
    p = e.copy()
    for _ in range(PPR_IT):
        p = (1 - ALPHA) * e + ALPHA * (p @ colnorm)
    feats = [h1, np.log1p(h2), np.log1p(h3), np.log1p(p * 1e4)]
    for k, m in mats.items():
        feats.append(m[i].toarray().ravel())
    return np.stack(feats, 1).astype(np.float32)


def main():
    log = []
    def report(t):
        print(t, flush=True); log.append(t)
    report("=" * 100)
    report("MULTI-HOP PATH ASSEMBLY: can a learner find mechanism in the networks?")
    report("=" * 100)
    report("  PREDECLARED: passes only if resp+paths beats resp+rewire on BOTH cohorts.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]; genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05
    cand_ix = np.where(~tide)[0]; cand = [genes[c] for c in cand_ix]

    sealed, answers = set(), {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists(): sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists(): answers[tag] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]

    nodes = sorted(set(cand) | set(kos) | sealed)
    nix = {g: i for i, g in enumerate(nodes)}
    report(f"  graph over {len(nodes):,} nodes; {len(cand):,} candidate response genes")
    real = layers(nodes, report)
    rew = rewire(real, A.SEED, report)

    hit = (Amat[[ki[k] for k in train]] >= A.TAU)
    resp = np.array([hit[:, c].mean() for c in cand_ix], np.float32)
    cand_node = np.array([nix[g] for g in cand])

    def prep(mats):
        comb = sum(mats.values()).tocsr(); comb.data[:] = 1.0
        d = np.asarray(comb.sum(0)).ravel(); d[d == 0] = 1
        return comb, sp.diags(1.0 / d) @ comb

    packs = {"real": prep(real), "rewire": prep(rew)}

    rng = np.random.default_rng(A.SEED)
    tr_kos = [k for k in train if k in nix and hit[train.index(k)].sum() >= 5][:1200]
    report(f"  building training pairs from {len(tr_kos):,} knockouts with >= 5 movers ...")
    Xr, Xw, y = [], [], []
    for n_, k in enumerate(tr_kos):
        row = Amat[ki[k]][cand_ix]
        pos = np.where(row >= A.TAU)[0]
        neg = rng.choice(len(cand), size=min(len(pos) * NEG_PER_POS, len(cand)), replace=False)
        neg = np.setdiff1d(neg, pos)[:len(pos) * NEG_PER_POS]
        sel = np.concatenate([pos, neg])
        for name, (comb, cn) in packs.items():
            f = path_rows(real if name == "real" else rew, nix[k], comb, cn)[cand_node][sel]
            (Xr if name == "real" else Xw).append(np.column_stack([resp[sel], f]))
        y.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
        if n_ % 300 == 0: report(f"    {n_}/{len(tr_kos)}")
    Xr = np.concatenate(Xr); Xw = np.concatenate(Xw); y = np.concatenate(y)
    report(f"  training pairs {len(y):,} ({int(y.sum()):,} positive); {Xr.shape[1]} features")

    from sklearn.ensemble import HistGradientBoostingClassifier
    M = {"resp_only": HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(Xr[:, :1], y),
         "resp+paths": HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(Xr, y),
         "resp+rewire": HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(Xw, y),
         "paths_only": HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(Xr[:, 1:], y)}
    report("  four models fitted")

    summary = {}
    for tag, lab in (("", "cohort 1"), ("_holdout", "cohort 2 (holdout)")):
        ans = answers.get(tag)
        if not ans: continue
        cohort = [k for k in sorted(ans) if k in nix]
        per = {a: [] for a in M}
        for k in cohort:
            truth = set(ans[k]["movers"])
            fr = path_rows(real, nix[k], *packs["real"])[cand_node]
            fw = path_rows(rew, nix[k], *packs["rewire"])[cand_node]
            Fr = np.column_stack([resp, fr]); Fw = np.column_stack([resp, fw])
            for a, mdl in M.items():
                X = {"resp_only": Fr[:, :1], "resp+paths": Fr,
                     "resp+rewire": Fw, "paths_only": Fr[:, 1:]}[a]
                top = np.argsort(-mdl.predict_proba(X)[:, 1])[:A.NPRED]
                per[a].append(len({cand[t] for t in top} & truth) / A.NPRED)
        report(f"\n### {lab}  ({len(cohort)} knockouts)")
        v = {a: np.array(x) for a, x in per.items()}
        for a in ("resp+paths", "resp+rewire", "resp_only", "paths_only"):
            report(f"  {a:<14} {v[a].mean():.4f}")
        out = {}
        for nm_, a_, b_ in (("GATE paths - rewire", "resp+paths", "resp+rewire"),
                            ("paths - resp_only", "resp+paths", "resp_only")):
            d = v[a_] - v[b_]
            br = np.random.default_rng(0)
            bs = d[br.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            report(f"  {nm_:<22} {d.mean():>+8.4f} [{lo:+.4f}, {hi:+.4f}]  "
                   f"{'RESOLVED' if (lo>0 or hi<0) else 'within noise'}")
            out[nm_] = {"gap": float(d.mean()), "ci": [lo, hi]}
        summary[tag or "cohort1"] = {"means": {a: float(x.mean()) for a, x in v.items()},
                                     "contrasts": out}
    json.dump({"test": "adrn_path_assembly", "summary": summary, "log": log},
              open(OUT / "adrn_path_assembly.json", "w"), indent=2)
    report(f"\n  -> {OUT/'adrn_path_assembly.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
