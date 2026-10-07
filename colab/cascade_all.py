"""cascade_all — the fair shot at the third arm: predict the GENOME-WIDE knockout cascade (which genes move) by COMBINING every
layer we built into one supervised model, tested on HELD-OUT knockouts. This is the honest kitchen-sink attempt at the wall.

For a knockout of gene G, score every measured gene J by P(J moves). Features per (G,J) pair combine all layers:
  REGULATORY   G->J regulatory edge (signed) ; J in G's TRRUST regulon ; G->J signaling
  PROXIMITY    RWR(G->J) over the multi-layer graph ; PPI-adjacent ; complex co-member ; coexpression ; reaction-connected
  DIRECTION    shortest directed hops G->J
  J-INTRINSIC  J's generic mover-frequency (how often J moves across OTHER knockouts) + J baseline expression  [the control:
               some genes are just responsive; the G-SPECIFIC signal must beat this prior]
  G-INTRINSIC  G is a TF ; G out-degree
Label: J moves under G (|z|>2). Model: gradient-boosted trees, GroupKFold BY KNOCKOUT (predict cascades of unseen knockouts).
Reported vs baselines: J-mover-frequency prior alone, RWR alone, regulon alone -- so we see whether combining everything adds
G-SPECIFIC cascade signal over "predict the usually-responsive genes." Honest: this is the measured far-field wall's fair test.
"""
import os
import json, collections
from pathlib import Path
import numpy as np
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"


def _graph():
    import propagate as pr
    G = pr._load()
    W = pr._transition(G)
    D = json.load(open(OUT / "cell_complete.json"))
    N = G["N"]; names = G["names"]; idx = G["idx"]
    reg = {}                      # (src,dst)->sign
    for e in D["reg"]:
        a, b, s = e[0], e[1], (e[2] if len(e) > 2 else 1)
        if a < N and b < N:
            reg[(a, b)] = s
    sig = set()
    for e in D["sig"]:
        if e[0] < N and e[1] < N:
            sig.add((e[0], e[1]))
    coexpr = {}
    for k, lst in (D.get("coexpr") or {}).items():
        i = int(k)
        for pair in lst:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                coexpr[(i, int(pair[0]))] = float(pair[1]); coexpr[(int(pair[0]), i)] = float(pair[1])
    rxn = set()
    try:
        rn = json.load(open(OUT / "reaction_network.json"))
        for a, b in rn.get("catalysis_precedes", []):
            rxn.add((a, b))
    except Exception:
        pass
    return G, W, reg, sig, coexpr, rxn, D


def load_knockouts(G, min_movers=8, thr=2.0, max_ko=400):
    import h5py
    h = h5py.File(f"{SP}/gwps.h5ad", "r")
    dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
    pert = [p.split("_")[1] if "_" in p else p for p in dec(h["obs"]["gene_transcript"][:])]
    cats = dec(h["var"]["__categories"]["gene_name"][:]); codes = h["var"]["gene_name"][:]
    genes = [cats[c] for c in codes]
    ps = {}
    for i, s in enumerate(pert):
        ps.setdefault(s, i)
    idx = G["idx"]
    KO = {}
    for g in ps:
        if g not in idx:
            continue
        r = np.array(h["X"][ps[g], :], dtype=np.float64)
        r[~np.isfinite(r)] = np.nan; r[np.abs(r) > 1e6] = np.nan
        z = {genes[i]: float(r[i]) for i in range(len(genes)) if np.isfinite(r[i])}
        mv = {x for x in z if abs(z[x]) > thr and x != g}
        if len(mv) >= min_movers:
            KO[g] = {"z": z, "movers": mv, "U": set(z)}
        if len(KO) >= max_ko:
            break
    h.close()
    return KO


def build_matrix(neg_ratio=15):
    G, W, reg, sig, coexpr, rxn, D = _graph()
    idx = G["idx"]; names = G["names"]; ppi = G["ppi"]; tr = G["trrust"]
    KO = load_knockouts(G)
    # J generic mover-frequency (leave-one-out done implicitly by GroupKFold on G, but compute globally then exclude self)
    movefreq = collections.Counter()
    for g, d in KO.items():
        for j in d["movers"]:
            movefreq[j] += 1
    nko = len(KO)
    # J baseline expression (K562 RPKM)
    import csv
    rpkm = {}
    for r in csv.DictReader(open(f"{SP}/rnadecay/AvgKdegs.csv")):
        if r["cell_line"] == "K562":
            try:
                rpkm[r["feature_ID"]] = float(r["avg_RPKM_total"])
            except ValueError:
                pass
    tf = G["tf"]
    rng = np.random.RandomState(0)
    rows = []; ys = []; groups = []
    for g, d in KO.items():
        gi = idx[g]
        p = W @ (np.eye(G["N"])[:, gi] if False else _seed(G["N"], gi))  # not used; RWR below
        pvec = _rwr(W, gi)
        regset = {t[0] for t in tr.get(g, [])}
        Ui = [x for x in d["U"] if x in idx and x != g]
        movers = d["movers"]
        pos = [x for x in Ui if x in movers]
        neg_all = [x for x in Ui if x not in movers]
        neg = list(rng.choice(neg_all, min(len(neg_all), len(pos) * neg_ratio), replace=False)) if neg_all else []
        for J, lab in [(x, 1) for x in pos] + [(x, 0) for x in neg]:
            ji = idx[J]
            feat = [
                1.0 if (gi, ji) in reg else 0.0,
                float(reg.get((gi, ji), 0.0)),
                1.0 if J in regset else 0.0,
                1.0 if (gi, ji) in sig else 0.0,
                float(pvec[ji]),
                1.0 if ji in ppi.get(gi, set()) else 0.0,
                float(coexpr.get((gi, ji), 0.0)),
                1.0 if (gi, ji) in rxn or (ji, gi) in rxn else 0.0,
                (movefreq[J] - (1 if J in movers else 0)) / max(nko - 1, 1),   # leave-self-out generic responsiveness
                np.log10(rpkm.get(J, 0) + 0.1),
                1.0 if g in tf else 0.0,
                float(len(reg_out(reg, gi))),
            ]
            rows.append(feat); ys.append(lab); groups.append(g)
    cols = ["reg_edge", "reg_sign", "trrust_regulon", "signaling", "rwr", "ppi_adj", "coexpr", "reaction",
            "J_movefreq", "J_expr", "G_is_tf", "G_outdeg"]
    return np.array(rows), np.array(ys), np.array(groups), cols, nko


def _seed(n, i):
    e = np.zeros(n); e[i] = 1.0; return e


def _rwr(W, seed, alpha=0.3, iters=40):
    n = W.shape[0]; e = np.zeros(n); e[seed] = 1.0; p = e.copy()
    for _ in range(iters):
        p = (1 - alpha) * (W @ p) + alpha * e
    return p


_REGOUT = {}
def reg_out(reg, gi):
    if not _REGOUT:
        for (a, b) in reg:
            _REGOUT.setdefault(a, set()).add(b)
    return _REGOUT.get(gi, set())


def _auprc(scores, labels):
    o = np.argsort(-scores); y = labels[o]; tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / (tp + fp); rec = tp / max(y.sum(), 1); ap = 0.0; pr = 0.0
    for i in range(len(y)):
        if y[i]:
            ap += prec[i] * (rec[i] - pr); pr = rec[i]
    return ap


def run():
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    X, y, grp, cols, nko = build_matrix()
    base = y.mean()
    uniq = np.unique(grp)
    gkf = GroupKFold(n_splits=min(5, len(uniq)))
    def cv(mask):
        pred = np.zeros(len(y))
        for tr, te in gkf.split(X, y, grp):
            m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.7, eval_metric="logloss", n_jobs=4)
            m.fit(X[tr][:, mask], y[tr]); pred[te] = m.predict_proba(X[te][:, mask])[:, 1]
        return pred
    ci = {c: i for i, c in enumerate(cols)}
    full = cv(list(range(len(cols))))
    prior = cv([ci["J_movefreq"], ci["J_expr"]])                     # generic-responsiveness baseline
    gspec = cv([ci["reg_edge"], ci["reg_sign"], ci["trrust_regulon"], ci["signaling"], ci["rwr"],
                ci["ppi_adj"], ci["coexpr"], ci["reaction"]])         # G-specific only (no J prior)
    res = {"n_knockouts": int(len(uniq)), "n_pairs": int(len(y)), "base_rate": round(float(base), 4),
           "full_AUPRC": round(_auprc(full, y), 4), "full_lift": round(_auprc(full, y) / base, 2),
           "prior_only_AUPRC": round(_auprc(prior, y), 4), "prior_lift": round(_auprc(prior, y) / base, 2),
           "Gspecific_only_AUPRC": round(_auprc(gspec, y), 4), "Gspecific_lift": round(_auprc(gspec, y) / base, 2)}
    # feature importance
    m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=4, eval_metric="logloss").fit(X, y)
    imp = sorted(zip(cols, m.feature_importances_), key=lambda x: -x[1])
    res["top_features"] = [(c, round(float(v), 3)) for c, v in imp[:8]]
    return res


def main():
    print("=" * 100)
    print("CASCADE (the third arm) — predict the genome-wide knockout response by COMBINING all layers, held-out knockouts")
    print("=" * 100)
    r = run()
    print(f"\n  {r['n_knockouts']} knockouts, {r['n_pairs']:,} (G,J) pairs, base mover rate {r['base_rate']:.1%}. GroupKFold BY KNOCKOUT.\n")
    print(f"  {'model':28s} {'AUPRC':>8s} {'lift':>6s}")
    print(f"  {'J-responsiveness prior only':28s} {r['prior_only_AUPRC']:8.4f} {r['prior_lift']:6.2f}   (the control: some genes just move)")
    print(f"  {'G-specific layers only':28s} {r['Gspecific_only_AUPRC']:8.4f} {r['Gspecific_lift']:6.2f}   (regulon/RWR/network, no prior)")
    print(f"  {'FULL (everything combined)':28s} {r['full_AUPRC']:8.4f} {r['full_lift']:6.2f}")
    print(f"\n  top features: {r['top_features']}")
    gain = round(r["full_AUPRC"] - r["prior_only_AUPRC"], 4)
    print(f"\n  => combining everything: AUPRC {r['full_AUPRC']} ({r['full_lift']}x base). The honest question is whether the")
    print(f"     G-SPECIFIC layers add over the generic 'usually-responsive' prior: full - prior = {gain:+.4f} AUPRC.")
    print(f"     Predicting cascades of UNSEEN knockouts is the far-field wall; this is its fair, combined, measured shot.")
    r["full_minus_prior"] = gain
    json.dump(r, open(OUT / "cascade_all.json", "w"), indent=1)
    print("\n  -> outputs/orphan/cascade_all.json")
    return r


if __name__ == "__main__":
    main()
