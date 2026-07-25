"""pair_retrieval_audit -- remove the last selection optimism from the retrieval result, and find out WHERE the remaining gap is.

pair_retrieval reached 0.4047 against a true-similarity ceiling of 0.5480, but two things about that number were still soft:

  1. K=3 WAS CHOSEN BY LOOKING AT THE TEST SWEEP. The margin over the pointwise model is large enough to survive it, but the headline is
     mildly optimistic and should not be. Fixed here with a NESTED split: K is selected on an inner split carved out of TRAIN only, and
     then applied blind to the outer test set. Whatever that produces is the honest number, even if it is lower.

  2. NOBODY KNOWS WHICH HALF OF THE PIPELINE IS LOSING. Retrieval has two stages -- rank the train sources by predicted similarity, then
     average the top-K profiles -- and 0.405 vs 0.548 could come from either. They need completely different fixes, so this measures both:
       RANKING   where does the genuinely most-similar train source land in the learned ranking? If it is usually near the top, ranking
                 is fine. Reported as the median rank of the true-best partner, and the fraction of the time it is in the top 3/10/25.
       WEIGHTING hold the learned TOP-K set fixed but reweight its members by their TRUE similarity. This is an oracle over weights only,
                 with the retrieved set unchanged. If it recovers most of the gap to 0.548, the ranking is adequate and the loss is in how
                 the retrieved profiles are combined; if it recovers little, the retrieved set simply does not contain the right genes and
                 no amount of clever weighting will help.

Same data, splits, tide mask, features and metric as pair_retrieval.py, so every number here is comparable to it. Deterministic."""
import json, collections
from pathlib import Path
import numpy as np
import pandas as pd
from eval_harness import Harness

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
TIDE_FRAC = 0.05
MINSRC = 20
MAXCOMPLEX = 200
K_RECALL = 50
SEEDS = (0, 1, 2)
KRETR = (1, 3, 5, 10, 25, 50)


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from scipy.stats import wilcoxon
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
    ctrl_expr = json.load(open(SP / "k562_baseline_expression.json"))["source_control_expr"]

    def jac(a, b):
        return len(a & b) / len(a | b) if (a or b) else 0.0

    def num(g, k):
        try:
            return float((info.get(g, {}) or {}).get(k) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def pair_feats(a, b):
        ca, cb = comps.get(a, set()), comps.get(b, set())
        ga, gb = go_of.get(a, {}), go_of.get(b, {})
        na, nb = ppi.get(a, set()), ppi.get(b, set())
        ea, eb = np.log1p(float(ctrl_expr.get(a, 0.0))), np.log1p(float(ctrl_expr.get(b, 0.0)))
        return [1.0 if ca & cb else 0.0, jac(ca, cb),
                jac(ga.get("P", set()), gb.get("P", set())), jac(ga.get("C", set()), gb.get("C", set())),
                jac(ga.get("F", set()), gb.get("F", set())),
                coex.get(a, {}).get(b, 0.0), codep.get(a, {}).get(b, 0.0),
                1.0 if b in na else 0.0, jac(na, nb), float(len(na & nb)),
                1.0 if (info.get(a, {}) or {}).get("proc") == (info.get(b, {}) or {}).get("proc") else 0.0,
                1.0 if (info.get(a, {}) or {}).get("comp") == (info.get(b, {}) or {}).get("comp") else 0.0,
                jac(doms.get(a, set()), doms.get(b, set())),
                abs(num(a, "ess") - num(b, "ess")), abs(num(a, "loeuf") - num(b, "loeuf")),
                abs(np.log1p(num(a, "ppi")) - np.log1p(num(b, "ppi"))),
                abs(num(a, "npath") - num(b, "npath")), abs(np.log1p(num(a, "pubs")) - np.log1p(num(b, "pubs"))),
                abs(ea - eb), min(num(a, "ess"), num(b, "ess")), max(num(a, "ess"), num(b, "ess")), min(ea, eb),
                1.0 if (num(a, "tf") > 0 and num(b, "tf") > 0) else 0.0]

    def fit_pairs(tr):
        XP, yP = [], []
        P = {s: pf(s) for s in tr}
        for i in range(len(tr)):
            for j in range(i + 1, len(tr)):
                XP.append(pair_feats(tr[i], tr[j]))
                yP.append(float(np.dot(P[tr[i]], P[tr[j]]) / nk))
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=6,
                                             random_state=0).fit(np.array(XP, float), np.array(yP))

    res = collections.defaultdict(list); persrc = collections.defaultdict(lambda: collections.defaultdict(list))
    ranks = []; chosen = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        ss = sorted(sources); rng.shuffle(ss)
        cut = int(0.7 * len(ss)); train, test = ss[:cut], ss[cut:]
        tr_rows = [kidx[s] for s in train]
        tide = (Aall[tr_rows] >= T).mean(0) >= TIDE_FRAC
        keep = np.where(~tide)[0]; nk = float(len(keep))

        def pf(s):
            v = Z[kidx[s]][keep].astype(np.float64)
            return (v - v.mean()) / (v.std() + 1e-9)

        # ---- 1. NESTED K SELECTION: inner split inside TRAIN only, outer test never consulted ----
        inner_cut = int(0.75 * len(train))
        itr, ite = train[:inner_cut], train[inner_cut:]
        ireg = fit_pairs(itr)
        ITRP = np.array([Z[kidx[t]][keep] for t in itr], dtype=np.float64)
        iscore = {K: [] for K in KRETR}
        for s in ite:
            it = {genes[q] for q in np.where((Aall[kidx[s]] >= T) & ~tide)[0] if genes[q] != s}
            if not it:
                continue
            sh = ireg.predict(np.array([pair_feats(s, t) for t in itr], float))
            for K in KRETR:
                o = np.argsort(-sh)[:K]; w = np.clip(sh[o], 0, None)
                w = w / w.sum() if w.sum() > 0 else np.full(K, 1.0 / K)
                pr = np.zeros(len(genes)); pr[keep] = w @ ITRP[o]
                top = keep[np.argsort(-np.abs(pr[keep]))[:K_RECALL]]
                iscore[K].append(len({genes[q] for q in top} & it) / min(len(it), K_RECALL))
        Kstar = max(KRETR, key=lambda K: np.mean(iscore[K]) if iscore[K] else -1)
        chosen.append(Kstar)

        # ---- outer model ----
        reg = fit_pairs(train)
        TRP = np.array([Z[kidx[t]][keep] for t in train], dtype=np.float64)
        truths = {}
        for s in test:
            t = {genes[q] for q in np.where((Aall[kidx[s]] >= T) & ~tide)[0] if genes[q] != s}
            if t:
                truths[s] = t

        def score(pk, s):
            pr = np.zeros(len(genes)); pr[keep] = pk
            top = keep[np.argsort(-np.abs(pr[keep]))[:K_RECALL]]
            return len({genes[q] for q in top} & truths[s]) / min(len(truths[s]), K_RECALL)

        def rec(lbl, s, v):
            res[lbl].append(v); persrc[lbl][s].append(v)

        for s in test:
            if s not in truths:
                continue
            sh = reg.predict(np.array([pair_feats(s, t) for t in train], float))
            st = np.array([float(np.dot(pf(s), pf(t)) / nk) for t in train])   # true similarity: ORACLE, evaluation only

            def blend(order, w):
                w = np.clip(w, 0, None)
                w = w / w.sum() if w.sum() > 0 else np.full(len(order), 1.0 / len(order))
                return w @ TRP[order]

            ok = np.argsort(-sh)
            rec(f"pre-registered K (nested, K*={Kstar})", s, score(blend(ok[:Kstar], sh[ok[:Kstar]]), s))
            for K in KRETR:
                o = ok[:K]
                rec(f"learned rank, learned weights K={K}", s, score(blend(o, sh[o]), s))
                # WEIGHTING ORACLE: same retrieved set, weights replaced by true similarity
                rec(f"learned rank, ORACLE weights K={K}", s, score(blend(o, st[o]), s))
                ot = np.argsort(-st)[:K]
                rec(f"ORACLE rank, oracle weights K={K}", s, score(blend(ot, st[ot]), s))
            # RANKING QUALITY: where does the genuinely best partner land under the learned metric?
            ranks.append(int(np.where(ok == int(np.argmax(st)))[0][0]))

    n_tr = len(ranks)
    print(f"K562: |z|>={T:.1f} | nested-K chosen per seed: {chosen}")
    print(f"\n  {'method':46s} {'recall@50':>10s}")
    tab = []
    keyrows = [f"pre-registered K (nested, K*={k})" for k in sorted(set(chosen))]
    for lbl in keyrows + [f"learned rank, learned weights K={K}" for K in KRETR] + \
               [f"learned rank, ORACLE weights K={K}" for K in KRETR] + \
               [f"ORACLE rank, oracle weights K={K}" for K in KRETR]:
        v = res.get(lbl)
        if not v:
            continue
        tab.append({"method": lbl, "recall50": round(float(np.mean(v)), 4), "n": len(v)})
        print(f"  {lbl:46s} {np.mean(v):>10.4f}")

    ra = np.array(ranks)
    print(f"\n  RANKING QUALITY -- rank of the TRUE best partner under the learned metric (out of ~235 train sources, {n_tr} test genes)")
    print(f"     median rank {np.median(ra):.0f} | in top 3: {np.mean(ra < 3):.3f} | top 10: {np.mean(ra < 10):.3f} | "
          f"top 25: {np.mean(ra < 25):.3f} | top 50: {np.mean(ra < 50):.3f}")

    bestK = max(KRETR, key=lambda K: np.mean(res[f"learned rank, learned weights K={K}"]))
    lw = float(np.mean(res[f"learned rank, learned weights K={bestK}"]))
    ow = float(np.mean(res[f"learned rank, ORACLE weights K={bestK}"]))
    orc = max(float(np.mean(res[f"ORACLE rank, oracle weights K={K}"])) for K in KRETR)
    pre = float(np.mean(res[f"pre-registered K (nested, K*={chosen[0]})"])) if len(set(chosen)) == 1 else \
        float(np.mean([v for k in set(chosen) for v in res[f"pre-registered K (nested, K*={k})"]]))
    gap = orc - lw
    frac_w = (ow - lw) / gap if gap > 0 else float("nan")
    print(f"\n  DECOMPOSING THE GAP at K={bestK}")
    print(f"     learned rank + learned weights      {lw:.4f}   <- the model")
    print(f"     learned rank + ORACLE weights       {ow:.4f}   (retrieved set unchanged, weights perfected)")
    print(f"     ORACLE rank + oracle weights        {orc:.4f}   <- architecture ceiling")
    print(f"     perfect WEIGHTING recovers {ow-lw:+.4f} of the {gap:.4f} gap = {frac_w:.0%}; the rest is RANKING")
    print(f"\n  PRE-REGISTERED (nested-K) headline: {pre:.4f}   vs the test-selected {lw:.4f} "
          f"({pre-lw:+.4f} of selection optimism)")

    ks = sorted(persrc[f"learned rank, learned weights K={bestK}"])
    a = np.array([np.mean(persrc[f"learned rank, learned weights K={bestK}"][k]) for k in ks])
    b = np.array([np.mean(persrc[f"learned rank, ORACLE weights K={bestK}"][k]) for k in ks])
    p_w = float(wilcoxon(b, a).pvalue) if np.any(a != b) else 1.0

    verdict = (
        "PAIR RETRIEVAL AUDIT (pair_retrieval_audit.py): removing the last selection optimism from the retrieval result and locating the "
        "remaining gap. "
        f"PRE-REGISTERED K: choosing K on an inner split carved out of TRAIN only and applying it blind to the outer test gives {pre:.4f}, "
        f"against {lw:.4f} when K is picked by inspecting the test sweep -- so the selection optimism in the reported headline was "
        f"{lw-pre:+.4f}. The nested procedure chose K*={chosen} across the three seeds. "
        f"DECOMPOSING THE GAP to the {orc:.4f} architecture ceiling: retrieval has two stages and they need different fixes. Holding the "
        f"learned top-{bestK} set fixed and replacing only the weights with true similarity gives {ow:.4f}, recovering {ow-lw:+.4f} of the "
        f"{gap:.4f} gap ({frac_w:.0%}, paired p={p_w:.3g}); the remainder is RANKING -- the retrieved set simply does not contain the right "
        "genes. "
        f"RANKING QUALITY directly: the genuinely most-similar train source lands at median rank {np.median(ra):.0f} of ~235 under the "
        f"learned metric, and is inside the top 10 only {np.mean(ra < 10):.0%} of the time and the top 25 {np.mean(ra < 25):.0%}. "
        + ("So the binding constraint is RETRIEVAL QUALITY, not the averaging step: better weighting of the current retrieved set buys "
           "little, while finding better partners is worth the bulk of the gap. Effort should go into the similarity model and its "
           "features, not into the combination rule. " if frac_w < 0.5 else
           "So the binding constraint is the COMBINATION RULE rather than which genes get retrieved: the right partners are largely being "
           "found and then blended badly, which is a much cheaper problem to fix than improving the metric. ")
        + "Same data, splits, tide mask, features and metric as pair_retrieval.py. The oracle rows are selected by peeking at the answer and "
        "bound what is achievable; they are not predictors. Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"threshold": T, "nested_k_per_seed": chosen, "preregistered_recall": round(pre, 4),
               "test_selected_recall": round(lw, 4), "selection_optimism": round(lw - pre, 4),
               "oracle_weights_recall": round(ow, 4), "architecture_ceiling": round(orc, 4),
               "frac_gap_from_weighting": None if np.isnan(frac_w) else round(float(frac_w), 4),
               "p_oracle_weights": p_w, "median_rank_of_best_partner": float(np.median(ra)),
               "best_partner_in_top10": float(np.mean(ra < 10)), "best_partner_in_top25": float(np.mean(ra < 25)),
               "table": tab, "verdict": verdict, "note": verdict},
              open(OUT / "pair_retrieval_audit.json", "w"), indent=1)
    print("\n  -> outputs/orphan/pair_retrieval_audit.json")


if __name__ == "__main__":
    main()
