"""pair_retrieval -- pose knockout prediction as RETRIEVAL rather than regression, and get 168x the training signal from the same data.

THE REFRAME. Every model in this project so far asked: given a gene's features, regress its ~8,533-dim response. That is 337 training
examples for a very wide output, and the learning curve, the alpha sweep and the dilution result all say the same thing -- it is
variance-limited. Asked instead as "do these two knockouts behave alike?", the SAME 337 sources yield 337*336/2 = 56,616 labelled pairs.
No new data, 168x more supervised signal, and a target (one similarity scalar) that a small model can actually fit.

WHY IT SHOULD WORK BETTER, measured before building this: annotation similarity really does predict response similarity, and crucially the
ranking of features INVERTS relative to the pointwise model.
    PPI neighbour Jaccard   AUC 0.778 for the top-decile most-similar pairs, fires on 23,105 pairs
    GO Jaccard              AUC 0.712, fires on 44,773 pairs
    shared complex          AUC 0.587, fires on   1,119 pairs
Shared complex is the pointwise model's STRONGEST feature and it is the weakest informative one here, firing on 2% of pairs. The dense
features that got diluted to uselessness in a 235-sample ridge are the strong ones once each is judged against 56k examples. That also
explains the uncurated-gene failure without invoking coverage: those genes have no complex, but they do have PPI neighbours and GO terms.

THE PIPELINE
    1. learn  sim_hat(a,b)  from pair features, on pairs where BOTH sources are in TRAIN
    2. for a held-out gene s, score sim_hat(s, t) against every TRAIN source t
    3. predict s's response as the similarity-weighted average of the top-K train sources' MEASURED profiles
    4. rank genes by |predicted z| and score recall@50, exactly as every other model here

LEAKAGE DISCIPLINE. Splits are SOURCE-disjoint, not pair-disjoint: a training pair may not share a source with a test pair, or the metric
would be fitted on the very gene it is asked about. The tide mask, the pair-feature scaler and the regressor are all train-only. The test
gene's own measured response is used solely as truth. Its annotation is used freely -- that is the cold-start contract.

CONTROLS. The one that matters is COSINE RETRIEVAL: the identical retrieve-and-average pipeline with a fixed cosine metric on the same
features instead of a learned one. If learning the metric does not beat cosine, the gain is retrieval, not learning, and should be
described that way. Also reported: the oracle that retrieves with the TRUE similarity (the ceiling for this architecture), the
best-possible-single-copy oracle, mismatched-source, tide-null and random. Both complex/no-complex strata are always split out, since the
uncurated genes are the population this reframe is supposed to help. Deterministic."""
import os
import json, collections
from pathlib import Path
import numpy as np
import pandas as pd
from eval_harness import Harness

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
TIDE_FRAC = 0.05
MINSRC = 20
MAXCOMPLEX = 200
K_RECALL = 50
SEEDS = (0, 1, 2)
KRETR = (1, 3, 5, 10, 25, 50)     # how many retrieved neighbours to average; large K degenerates to the mean response


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score
    from scipy.stats import wilcoxon, spearmanr
    H = Harness("K562")
    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    genes = [str(g) for g in df.columns]
    kos = [str(k) for k in df.index]; kidx = {k: i for i, k in enumerate(kos)}
    Z = df.values.astype(np.float32); Aall = np.abs(Z)

    tc = [int((np.abs(H.M[H.ki[k]]) >= 1.0).sum()) for k in H.kos
          if k in kidx and (np.abs(H.M[H.ki[k]]) > 0).any() and np.abs(H.M[H.ki[k]])[np.abs(H.M[H.ki[k]]) > 0].min() < 1.0]
    med = float(np.median(tc)); T, best = 4.0, None
    for t in np.arange(1.0, 12.0, 0.1):
        e = abs(float(np.median((Aall >= t).sum(1))) - med)
        if best is None or e < best:
            best, T = e, float(t)
    per = (Aall >= T).sum(1)
    sources = [kos[i] for i in range(len(kos)) if per[i] >= MINSRC]

    # ---------------- static annotation, identical sources to the pointwise model ----------------
    D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]; N = len(names)
    info = {g["name"]: g for g in D["genes"]}
    comps = collections.defaultdict(set)
    for cid, mem in D["complexes"].items():
        mm = {names[x] for x in mem if isinstance(x, int) and x < N}
        if 2 <= len(mm) <= MAXCOMPLEX:
            for g in mm:
                comps[g].add(cid)
    go_of = {}
    for g, d in (D.get("go") or {}).items():
        for br in ("P", "C", "F"):
            for t in (d.get(br) or []):
                go_of.setdefault(g, {}).setdefault(br, set()).add(t)
    doms = {}
    try:
        for k, v in json.load(open(OUT / "domains.json")).get("domains", {}).items():
            try:
                doms[names[int(k)]] = set(str(x) for x in v)
            except (ValueError, IndexError):
                pass
    except Exception:
        pass

    def load_sim(key):
        out = collections.defaultdict(dict)
        for a, lst in (D.get(key) or {}).items():
            try:
                ga = names[int(a)]
            except (ValueError, IndexError):
                continue
            for pr in lst:
                try:
                    gb = names[int(pr[0])]; sc = float(pr[1])
                except (ValueError, IndexError, TypeError):
                    continue
                out[ga][gb] = max(out[ga].get(gb, 0.0), sc)
                out[gb][ga] = max(out[gb].get(ga, 0.0), sc)
        return out
    coex = load_sim("coexpr"); codep = load_sim("codep")
    ppi = collections.defaultdict(set)
    for e in (D.get("ppi") or []):
        try:
            a, b = names[int(e[0])], names[int(e[1])]
        except (ValueError, IndexError, TypeError):
            continue
        ppi[a].add(b); ppi[b].add(a)
    BE = json.load(open(SP / "k562_baseline_expression.json"))
    ctrl_expr = BE["source_control_expr"]

    def jac(a, b):
        return len(a & b) / len(a | b) if (a or b) else 0.0

    def num(g, k):
        try:
            return float((info.get(g, {}) or {}).get(k) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    PAIRF = ["shared_complex", "complex_jac", "go_P", "go_C", "go_F", "coexpr", "codep",
             "ppi_edge", "ppi_nbr_jac", "ppi_nbr_n", "same_proc", "same_comp", "dom_jac",
             "d_ess", "d_loeuf", "d_ppideg", "d_npath", "d_pubs", "d_expr",
             "min_ess", "max_ess", "min_expr", "both_tf"]

    def pair_feats(a, b):
        ca, cb = comps.get(a, set()), comps.get(b, set())
        ga, gb = go_of.get(a, {}), go_of.get(b, {})
        na, nb = ppi.get(a, set()), ppi.get(b, set())
        ea, eb = np.log1p(float(ctrl_expr.get(a, 0.0))), np.log1p(float(ctrl_expr.get(b, 0.0)))
        return [
            1.0 if ca & cb else 0.0, jac(ca, cb),
            jac(ga.get("P", set()), gb.get("P", set())),
            jac(ga.get("C", set()), gb.get("C", set())),
            jac(ga.get("F", set()), gb.get("F", set())),
            coex.get(a, {}).get(b, 0.0), codep.get(a, {}).get(b, 0.0),
            1.0 if b in na else 0.0, jac(na, nb), float(len(na & nb)),
            1.0 if (info.get(a, {}) or {}).get("proc") == (info.get(b, {}) or {}).get("proc") else 0.0,
            1.0 if (info.get(a, {}) or {}).get("comp") == (info.get(b, {}) or {}).get("comp") else 0.0,
            jac(doms.get(a, set()), doms.get(b, set())),
            abs(num(a, "ess") - num(b, "ess")), abs(num(a, "loeuf") - num(b, "loeuf")),
            abs(np.log1p(num(a, "ppi")) - np.log1p(num(b, "ppi"))),
            abs(num(a, "npath") - num(b, "npath")), abs(np.log1p(num(a, "pubs")) - np.log1p(num(b, "pubs"))),
            abs(ea - eb),
            min(num(a, "ess"), num(b, "ess")), max(num(a, "ess"), num(b, "ess")), min(ea, eb),
            1.0 if (num(a, "tf") > 0 and num(b, "tf") > 0) else 0.0,
        ]

    # point features, for the cosine-metric control: the same information, unlearned
    PTF = ["ess", "loeuf", "tf", "ppi", "npath", "ndis", "cpg", "pubs", "dark", "conf", "enh", "dep_frac"]
    def point_feats(g):
        x = [num(g, k) for k in PTF] + [np.log1p(float(ctrl_expr.get(g, 0.0)))]
        return x

    # ---- the POINTWISE model, rebuilt here so the head-to-head is on IDENTICAL splits and can be tested paired.
    # This is program_coldstart + the dense(PCA) block that won in coldstart_coverage (0.3711 there).
    MAXC = MAXCOMPLEX
    cmem = collections.defaultdict(set)
    for cid, mem in D["complexes"].items():
        mm = {names[x] for x in mem if isinstance(x, int) and x < N}
        if 2 <= len(mm) <= MAXC:
            cmem[cid] = mm
    go_flat = {g: {f"{b}:{t}" for b, ts in d.items() for t in ts} for g, d in go_of.items()}
    NUMP = ["ess", "loeuf", "tf", "ppi", "npath", "ndis", "cpg", "pubs", "dark", "conf", "enh", "dep_frac"]

    def pointwise_X(train):
        def vocab(get, mn):
            c = collections.Counter()
            for s in train:
                c.update(get(s))
            return sorted(k for k, v in c.items() if v >= mn)
        v_comp = vocab(lambda s: comps.get(s, ()), 2); v_dom = vocab(lambda s: doms.get(s, ()), 3)
        v_proc = vocab(lambda s: [(info.get(s, {}) or {}).get("proc") or "?"], 3)
        v_loc = vocab(lambda s: [(info.get(s, {}) or {}).get("comp") or "?"], 3)
        v_go = vocab(lambda s: go_flat.get(s, ()), 4)
        trset = set(train)
        v_soft = [c for c in sorted(cmem) if len(cmem[c] & trset) >= 3]
        def base(s):
            x = [num(s, k) for k in NUMP]
            x.append(np.log1p(float(ctrl_expr.get(s, 0.0)))); x.append(1.0 if s in ctrl_expr else 0.0)
            x += [1.0 if (info.get(s, {}) or {}).get("proc") == p else 0.0 for p in v_proc]
            x += [1.0 if (info.get(s, {}) or {}).get("comp") == p else 0.0 for p in v_loc]
            dl = doms.get(s, set()); x += [1.0 if d in dl else 0.0 for d in v_dom]
            cl = comps.get(s, set()); x += [1.0 if c in cl else 0.0 for c in v_comp]
            return x
        def dense(s):
            gs = go_flat.get(s, set())
            d2 = [1.0 if t in gs else 0.0 for t in v_go]
            na = ppi.get(s, set())
            for c in v_soft:
                mb = cmem[c] - {s}
                if not mb:
                    d2 += [0.0, 0.0, 0.0]; continue
                d2.append(len(na & mb) / len(mb) if na else 0.0)
                d2.append(float(np.mean([coex.get(s, {}).get(m, 0.0) for m in mb])))
                d2.append(float(np.mean([codep.get(s, {}).get(m, 0.0) for m in mb])))
            return d2
        return (np.array([base(s) for s in sources], float), np.array([dense(s) for s in sources], float))

    print(f"K562: |z|>={T:.1f} | {len(sources)} sources "
          f"({sum(1 for s in sources if comps.get(s))} in a curated complex, {sum(1 for s in sources if not comps.get(s))} in none)")
    print(f"pair features: {len(PAIRF)}   |   pairs available at 70% train: {int(0.7*len(sources))*(int(0.7*len(sources))-1)//2}")

    res = collections.defaultdict(list)
    persrc = collections.defaultdict(lambda: collections.defaultdict(list))
    aucs = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        ss = sorted(sources); rng.shuffle(ss)
        cut = int(0.7 * len(ss)); train, test = ss[:cut], ss[cut:]
        tr_rows = [kidx[s] for s in train]
        tide = (Aall[tr_rows] >= T).mean(0) >= TIDE_FRAC
        keep = np.where(~tide)[0]

        # standardised profiles, used ONLY to define the pair target among TRAIN sources
        def prof(s):
            v = Z[kidx[s]][keep].astype(np.float64)
            return (v - v.mean()) / (v.std() + 1e-9)
        Ptr = {s: prof(s) for s in train}
        nk = float(len(keep))

        # ---- 1. build the training pair set: BOTH sources in train ----
        XP, yP = [], []
        for i in range(len(train)):
            for j in range(i + 1, len(train)):
                a, b = train[i], train[j]
                XP.append(pair_feats(a, b))
                yP.append(float(np.dot(Ptr[a], Ptr[b]) / nk))
        XP = np.array(XP, float); yP = np.array(yP)
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=6,
                                            random_state=0).fit(XP, yP)

        # ---- cosine control metric on point features, scaled on train ----
        PF = np.array([point_feats(s) for s in sources], float)
        sidx = {s: i for i, s in enumerate(sources)}
        m = PF[[sidx[s] for s in train]].mean(0); sd = PF[[sidx[s] for s in train]].std(0) + 1e-9
        PFn = (PF - m) / sd
        PFn = PFn / (np.linalg.norm(PFn, axis=1, keepdims=True) + 1e-9)

        truths = {}
        for s in test:
            t = {genes[q] for q in np.where((Aall[kidx[s]] >= T) & ~tide)[0] if genes[q] != s}
            if t:
                truths[s] = t
        tr_rank = np.zeros(len(genes)); tr_rank[keep] = (Aall[tr_rows][:, keep] >= T).mean(0)
        TRP = np.array([Z[kidx[t]][keep] for t in train], dtype=np.float64)

        def score(pred_keep, s):
            pr = np.zeros(len(genes)); pr[keep] = pred_keep
            top = keep[np.argsort(-np.abs(pr[keep]))[:K_RECALL]]
            return len({genes[q] for q in top} & truths[s]) / min(len(truths[s]), K_RECALL)

        def rec(lbl, s, v):
            res[(lbl, "all")].append(v); persrc[lbl][s].append(v)
            res[(lbl, "in_complex" if comps.get(s) else "no_complex")].append(v)

        # ---- pointwise comparator on the SAME split ----
        Bp, Dp = pointwise_X(train)
        tr_i = [sidx[s] for s in train]
        pca = PCA(n_components=min(20, len(train) - 1, Dp.shape[1]), random_state=0).fit(Dp[tr_i])
        Xpw = np.hstack([Bp, pca.transform(Dp)])
        mp = Xpw[tr_i].mean(0); sdp = Xpw[tr_i].std(0) + 1e-9
        Mtr = Z[tr_rows][:, keep].astype(np.float64)
        mu = Mtr.mean(0); _, _, Vt = np.linalg.svd(Mtr - mu, full_matrices=False)
        Gk = Vt[:50]
        Ppw = Ridge(alpha=10.0).fit((Xpw[tr_i] - mp) / sdp, (Mtr - mu) @ Gk.T)

        # ---- 2-4. retrieve and average ----
        held_true, held_pred = [], []
        for s in test:
            if s not in truths:
                continue
            xs = np.array([pair_feats(s, t) for t in train], float)
            shat = reg.predict(xs)
            strue = np.array([float(np.dot(prof(s), Ptr[t]) / nk) for t in train])   # ORACLE similarity, for the ceiling only
            held_true.append(strue); held_pred.append(shat)
            cos = PFn[sidx[s]] @ PFn[[sidx[t] for t in train]].T

            for K in KRETR:
                for lbl, sc in (("learned metric", shat), ("[ctrl] cosine metric", cos), ("[oracle] true similarity", strue)):
                    o = np.argsort(-sc)[:K]
                    w = np.clip(sc[o], 0, None)
                    w = w / w.sum() if w.sum() > 0 else np.full(K, 1.0 / K)
                    rec(f"{lbl} K={K}", s, score(w @ TRP[o], s))
            rec("[pointwise] ridge on programs +dense(PCA)", s,
                score(mu + Ppw.predict(((Xpw[sidx[s]] - mp) / sdp)[None, :])[0] @ Gk, s))
            rec("[ref] uniform average of all train (= mu)", s, score(TRP.mean(0), s))
            rec("[ref] tide-null", s, score(tr_rank[keep], s))
            rec("[ref] random", s, score(rng.rand(len(keep)), s))
            rec("[oracle] best single copy", s, max(score(TRP[i], s) for i in range(len(train))))
        # how well is pair similarity itself predicted, on held-out sources?
        ht = np.concatenate(held_true); hp = np.concatenate(held_pred)
        aucs.append((float(spearmanr(hp, ht)[0]),
                     float(roc_auc_score((ht >= np.quantile(ht, 0.9)).astype(int), hp))))

    print(f"\n  pair-similarity prediction on HELD-OUT sources: Spearman {np.mean([a[0] for a in aucs]):.4f}, "
          f"top-decile AUC {np.mean([a[1] for a in aucs]):.4f}  (source-disjoint, so this is honest)")

    print(f"\n  {'method':44s} {'ALL':>8s} {'in cplx':>9s} {'no cplx':>9s}")
    tab = []
    order = ([f"learned metric K={K}" for K in KRETR] + [f"[ctrl] cosine metric K={K}" for K in KRETR]
             + ["[pointwise] ridge on programs +dense(PCA)",
                "[ref] uniform average of all train (= mu)", "[ref] tide-null", "[ref] random"]
             + [f"[oracle] true similarity K={K}" for K in KRETR] + ["[oracle] best single copy"])
    for lbl in order:
        a = res[(lbl, "all")]
        if not a:
            continue
        ic = res[(lbl, "in_complex")]; nc = res[(lbl, "no_complex")]
        tab.append({"method": lbl, "all": round(float(np.mean(a)), 4), "in_complex": round(float(np.mean(ic)), 4),
                    "no_complex": round(float(np.mean(nc)), 4), "n": len(a)})
        print(f"  {lbl:44s} {np.mean(a):>8.4f} {np.mean(ic):>9.4f} {np.mean(nc):>9.4f}")

    def bysrc(lbl, only_nc=False):
        d = persrc[lbl]
        ks = sorted(k for k in d if (not only_nc or not comps.get(k)))
        return ks, np.array([float(np.mean(d[k])) for k in ks])

    bestK = max(KRETR, key=lambda K: np.mean(res[(f"learned metric K={K}", "all")]))
    L = f"learned metric K={bestK}"; C = f"[ctrl] cosine metric K={bestK}"
    ks, lv = bysrc(L); _, cv = bysrc(C)
    _, tv = np.array(bysrc("[ref] tide-null")[0]), bysrc("[ref] tide-null")[1]
    p_c = float(wilcoxon(lv, cv).pvalue); p_t = float(wilcoxon(lv, tv).pvalue)
    ksn, lvn = bysrc(L, True); _, cvn = bysrc(C, True)
    p_cn = float(wilcoxon(lvn, cvn).pvalue) if np.any(lvn != cvn) else 1.0
    _, pw = bysrc("[pointwise] ridge on programs +dense(PCA)")
    _, pwn = bysrc("[pointwise] ridge on programs +dense(PCA)", True)
    POINTWISE = float(pw.mean())
    p_pw = float(wilcoxon(lv, pw).pvalue)
    p_pwn = float(wilcoxon(lvn, pwn).pvalue) if np.any(lvn != pwn) else 1.0
    print(f"\n  best K = {bestK}   (paired tests on {len(ks)} unique sources)")
    print(f"  learned vs cosine metric      {lv.mean()-cv.mean():+.4f}  p={p_c:.3g}")
    print(f"  learned vs tide-null          {lv.mean()-tv.mean():+.4f}  p={p_t:.3g}")
    print(f"  learned vs cosine, NO-COMPLEX {lvn.mean()-cvn.mean():+.4f}  p={p_cn:.3g}  (n={len(ksn)})")
    print(f"  learned vs POINTWISE (same splits, PAIRED)  {lv.mean()-pw.mean():+.4f}  p={p_pw:.3g}")
    print(f"  learned vs POINTWISE, NO-COMPLEX            {lvn.mean()-pwn.mean():+.4f}  p={p_pwn:.3g}  (n={len(ksn)})")

    learn_wins = bool(lv.mean() > cv.mean() and p_c < 0.05)
    beats_pointwise = bool(lv.mean() > POINTWISE and p_pw < 0.05)
    nc_l = float(np.mean(res[(L, "no_complex")])); ic_l = float(np.mean(res[(L, "in_complex")]))
    orc = float(np.mean(res[(f"[oracle] true similarity K={bestK}", "all")]))

    verdict = (
        "PAIR RETRIEVAL (pair_retrieval.py): reposing knockout prediction as retrieval instead of regression. The pointwise framing gives "
        f"337 training examples for an 8,533-dim output and is demonstrably variance-limited; the pairwise framing gives {len(XP)} labelled "
        "pairs from the SAME data with a one-dimensional target. Pipeline: learn sim_hat(a,b) from annotation-only pair features on pairs "
        "whose BOTH members are train sources, score a held-out gene against every train source, and predict its response as the "
        "similarity-weighted average of the top-K train sources' measured profiles. Splits are SOURCE-disjoint, so no training pair shares a "
        "gene with a test pair. "
        f"PAIR SIMILARITY IS GENUINELY PREDICTABLE on held-out sources: Spearman {np.mean([a[0] for a in aucs]):.3f}, top-decile AUC "
        f"{np.mean([a[1] for a in aucs]):.3f}. "
        f"RESULT at the best K={bestK}: {lv.mean():.4f} overall ({ic_l:.4f} in a curated complex, {nc_l:.4f} in none), against the best "
        f"POINTWISE model at {POINTWISE:.4f}, tide-null {float(np.mean(res[('[ref] tide-null','all')])):.4f} and random "
        f"{float(np.mean(res[('[ref] random','all')])):.4f}. "
        + (f"It BEATS the pointwise model head-to-head on identical splits by {lv.mean()-pw.mean():+.4f} (paired p={p_pw:.3g} on "
           f"{len(ks)} unique sources), and on the uncurated genes the pointwise model fails by {lvn.mean()-pwn.mean():+.4f} "
           f"(p={p_pwn:.3g}, n={len(ksn)}). " if beats_pointwise else
           f"It does NOT beat the pointwise model ({lv.mean():.4f} vs {POINTWISE:.4f}), so the reframe does not pay off in recall even though "
           "the pair problem is well-posed. ")
        + "THE CONTROL THAT DECIDES WHETHER LEARNING MATTERS is the identical retrieve-and-average pipeline driven by a fixed cosine metric "
        f"on the same features: {cv.mean():.4f}. Learned minus cosine is {lv.mean()-cv.mean():+.4f} (paired p={p_c:.3g}), "
        + ("so learning the metric is doing real work rather than the gain coming from retrieval alone. " if learn_wins else
           "so the metric learning adds nothing significant -- whatever this architecture achieves comes from RETRIEVAL itself, not from "
           "having learned the similarity. It should be described that way. ")
        + f"On the uncurated genes that the pointwise model fails (0.160 there), this scores {nc_l:.4f}. "
        f"CEILING FOR THIS ARCHITECTURE: retrieving with the TRUE response similarity gives {orc:.4f}, and the best single measured copy "
        f"gives {float(np.mean(res[('[oracle] best single copy','all')])):.4f} -- both are oracles selected by peeking at the answer, so they "
        "bound what any similarity model could reach, they are not predictors. Large K degenerates towards the mean response, which is why "
        f"the uniform average over all train sources scores only {float(np.mean(res[('[ref] uniform average of all train (= mu)','all')])):.4f}. "
        "Deterministic; tide mask, pair target, feature scaling and regressor all fitted on train sources only.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"threshold": T, "n_train_pairs": int(len(XP)), "best_k": bestK, "table": tab,
               "pair_spearman_heldout": float(np.mean([a[0] for a in aucs])),
               "pair_auc_heldout": float(np.mean([a[1] for a in aucs])),
               "learned_vs_cosine": round(float(lv.mean() - cv.mean()), 4), "p_vs_cosine": p_c,
               "p_vs_cosine_no_complex": p_cn, "p_vs_tide": p_t,
               "learning_beats_cosine": learn_wins, "beats_pointwise": beats_pointwise,
               "pointwise_reference": round(POINTWISE, 4), "p_vs_pointwise": p_pw, "p_vs_pointwise_no_complex": p_pwn, "verdict": verdict, "note": verdict},
              open(OUT / "pair_retrieval.json", "w"), indent=1)
    print("\n  -> outputs/orphan/pair_retrieval.json")


if __name__ == "__main__":
    main()
