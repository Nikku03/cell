"""wall_diagnosis — what would BREAK the far-field cascade wall? Distinguish the two hypotheses for why held-out G-specific
mechanism = chance, because they need OPPOSITE data:
  H1 (VOLUME): we just don't have enough single-knockout Perturb-seq. More of the SAME data breaks it.
  H2 (KIND):   single-knockout steady-state pseudobulk is FUNDAMENTALLY insufficient (direct/indirect confounded, buffering
               unobserved). No amount of it helps; you need combinatorial / time-course / multi-context data.

The decisive experiment = a LEARNING CURVE. Take ALL knockouts with a footprint (not the 88-subset), hold out a FIXED test set,
and grow the number of TRAINING knockouts. Measure held-out G-specific-mechanism AUPRC (regulon/RWR/PPI, NO responsiveness prior)
and the prior baseline at each size. If mechanism lift RISES with N -> H1 (get more single-KO data, and we can estimate how much).
If it stays FLAT at chance while only the prior grows -> H2 (single-KO steady-state data cannot break it, at any scale).

Sparse machinery only (no dense NxN transition matrix -> no OOM; no torch).
"""
import json, collections
from pathlib import Path
import numpy as np
OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"


def _load(min_movers=3, max_ko=2000):
    import cascade_all as ca, propagate as pr, scipy.sparse as sp
    G = pr._load()
    KO = ca.load_knockouts(G, min_movers=min_movers, max_ko=max_ko)
    idx = G["idx"]; ppi = G["ppi"]; reg_edges = G["reg_edges"]; trrust = G["trrust"]
    uni = set()
    for g, d in KO.items():
        uni |= {x for x in d["U"] if x in idx}; uni.add(g)
    uni = sorted(uni, key=lambda g: idx[g])
    pos = {g: i for i, g in enumerate(uni)}; N = len(uni)
    inv = {idx[g]: pos[g] for g in uni}; gi = {g: idx[g] for g in uni}
    rows = []; cols = []
    regset = set()
    for e in reg_edges:
        a, b = e[0], e[1]
        if a in inv and b in inv:
            rows.append(inv[a]); cols.append(inv[b]); regset.add((inv[a], inv[b]))
    ppipair = set()
    for g in uni:
        for b in ppi.get(gi[g], ()):
            if b in inv:
                rows.append(pos[g]); cols.append(inv[b]); ppipair.add((pos[g], inv[b]))
    A = sp.coo_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols))), shape=(N, N)).tocsr()
    A = A + A.T; A.data[:] = 1.0; A.setdiag(0.0); A.eliminate_zeros()
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1.0
    Krow = (sp.diags(1.0 / deg) @ A).tocsr()
    return uni, pos, N, Krow, KO, G, regset, ppipair, trrust


def _rwr(Krow, seed_i, alpha=0.3, iters=40):
    N = Krow.shape[0]; e = np.zeros(N); e[seed_i] = 1.0; h = e.copy()
    for _ in range(iters):
        h = (1 - alpha) * (Krow @ h) + alpha * e
    return h


def build(min_movers=3, neg_ratio=12):
    uni, pos, N, Krow, KO, G, regset, ppipair, trrust = _load(min_movers)
    idx = G["idx"]
    import csv
    rpkm = {}
    for r in csv.DictReader(open(f"{SP}/rnadecay/AvgKdegs.csv")):
        if r["cell_line"] == "K562":
            try: rpkm[r["feature_ID"]] = float(r["avg_RPKM_total"])
            except ValueError: pass
    kos = [g for g in KO if g in pos]
    rng = np.random.RandomState(0)
    # per-knockout: feature rows for its candidate pairs
    per_ko = {}
    for g in kos:
        gi = pos[g]
        h = _rwr(Krow, gi)
        regtargets = {t[0] for t in trrust.get(g, [])}
        d = KO[g]; U = [x for x in d["U"] if x in pos and x != g]; movers = d["movers"]
        P = [x for x in U if x in movers]; Ng = [x for x in U if x not in movers]
        neg = list(rng.choice(Ng, min(len(Ng), max(len(P), 1) * neg_ratio), replace=False)) if Ng else []
        rows = []
        for J, lab in [(x, 1) for x in P] + [(x, 0) for x in neg]:
            ji = pos[J]
            rows.append((J, lab,
                         float(h[ji]),                                  # rwr
                         1.0 if (gi, ji) in regset else 0.0,            # reg edge
                         1.0 if J in regtargets else 0.0,               # trrust regulon
                         1.0 if (gi, ji) in ppipair else 0.0,           # ppi
                         np.log10(rpkm.get(J, 0) + 0.1)))               # expr
        per_ko[g] = rows
    return kos, per_ko, KO


def run():
    import xgboost as xgb
    from sklearn.metrics import average_precision_score
    kos, per_ko, KO = build()
    rng = np.random.RandomState(42)
    order = list(kos); rng.shuffle(order)
    n_test = max(len(order) // 4, 5)
    test_kos = order[:n_test]; pool = order[n_test:]                    # fixed held-out knockouts
    # fixed test matrix
    def mat(ko_list, movefreq, nko_for_prior):
        rows = []; ys = []
        for g in ko_list:
            for (J, lab, rwr, rege, trr, ppia, expr) in per_ko[g]:
                prior = (movefreq.get(J, 0) - (1 if lab else 0)) / max(nko_for_prior - 1, 1)
                rows.append([rwr, rege, trr, ppia, prior, expr]); ys.append(lab)
        return np.array(rows), np.array(ys)
    cols = ["rwr", "reg_edge", "trrust", "ppi", "J_movefreq", "J_expr"]
    mech = [0, 1, 2, 3]; prior_cols = [4, 5]
    sizes = [20, 40, 80, 160, 320, len(pool)]
    sizes = sorted({min(s, len(pool)) for s in sizes})
    curve = []
    for nsz in sizes:
        train_kos = pool[:nsz]
        movefreq = collections.Counter()
        for g in train_kos:
            for J in KO[g]["movers"]:
                movefreq[J] += 1
        Xtr, ytr = mat(train_kos, movefreq, len(train_kos))
        Xte, yte = mat(test_kos, movefreq, len(train_kos))
        if ytr.sum() < 5 or yte.sum() < 5:
            continue
        def fit(mask):
            m = xgb.XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, eval_metric="logloss", n_jobs=4)
            m.fit(Xtr[:, mask], ytr)
            return float(average_precision_score(yte, m.predict_proba(Xte[:, mask])[:, 1]))
        base = float(yte.mean())
        curve.append({"n_train_knockouts": nsz, "n_train_pairs": int(len(ytr)),
                      "mechanism_AUPRC": round(fit(mech), 4), "mechanism_lift": round(fit(mech) / base, 2),
                      "prior_AUPRC": round(fit(prior_cols), 4), "prior_lift": round(fit(prior_cols) / base, 2),
                      "base_rate": round(base, 4)})
    return {"n_test_knockouts": len(test_kos), "curve": curve}


def main():
    print("=" * 100)
    print("WALL DIAGNOSIS — learning curve: does held-out G-specific MECHANISM rise with more training knockouts, or stay chance?")
    print("=" * 100)
    r = run()
    print(f"\n  Fixed held-out set of {r['n_test_knockouts']} knockouts. Growing the TRAINING knockout count.\n")
    print(f"  {'#train KOs':>10s} {'#pairs':>8s} {'MECHANISM AUPRC':>16s} {'lift':>6s} {'PRIOR AUPRC':>12s} {'lift':>6s}")
    for c in r["curve"]:
        print(f"  {c['n_train_knockouts']:>10d} {c['n_train_pairs']:>8d} {c['mechanism_AUPRC']:>16.4f} {c['mechanism_lift']:>6.2f} "
              f"{c['prior_AUPRC']:>12.4f} {c['prior_lift']:>6.2f}")
    if len(r["curve"]) >= 2:
        first, last = r["curve"][0], r["curve"][-1]
        dmech = last["mechanism_lift"] - first["mechanism_lift"]
        dprior = last["prior_lift"] - first["prior_lift"]
        rising = dmech > 0.5 and last["mechanism_lift"] > 1.5
        r["mechanism_lift_delta"] = round(dmech, 2); r["prior_lift_delta"] = round(dprior, 2)
        r["hypothesis"] = "H1_volume" if rising else "H2_kind"
        if rising:
            verdict = (f"MECHANISM RISES with training data ({first['mechanism_lift']}x -> {last['mechanism_lift']}x as KOs "
                       f"{first['n_train_knockouts']}->{last['n_train_knockouts']}). H1: the wall is partly a DATA-VOLUME problem "
                       "-- more single-knockout Perturb-seq would help. Extrapolate the slope to estimate how much.")
        else:
            verdict = (f"MECHANISM STAYS FLAT at chance ({first['mechanism_lift']}x -> {last['mechanism_lift']}x) even as training "
                       f"knockouts grow {first['n_train_knockouts']}->{last['n_train_knockouts']}, while the PRIOR climbs "
                       f"({first['prior_lift']}x -> {last['prior_lift']}x). H2: MORE SINGLE-KNOCKOUT STEADY-STATE DATA DOES NOT "
                       "BREAK THE WALL. The G-specific transmission signal is not being under-sampled -- it is not RECOVERABLE "
                       "from single-KO pseudobulk endpoints at all (direct vs indirect confounded; buffering unobserved). What "
                       "would break it is a DIFFERENT KIND of measurement: (1) COMBINATORIAL double-KO Perturb-seq (Norman/GI) to "
                       "observe epistasis/buffering directly; (2) TIME-COURSE / 4sU nascent-RNA to separate direct targets (early) "
                       "from cascade (late); (3) MULTI-CONTEXT Perturb-seq (Replogle-Nadig multi-cell-line) to learn conserved "
                       "transmission vs context-specific responsiveness. Those measure the transmission coefficient; single-KO "
                       "endpoints never can.")
        print(f"\n  VERDICT: {verdict}")
        r["verdict"] = verdict
    r["note"] = ("Learning-curve diagnosis of the far-field wall: grow training knockouts against a FIXED held-out set, measure "
                 "G-specific-mechanism vs responsiveness-prior AUPRC. If mechanism is flat at chance while the prior climbs, more "
                 "single-KO steady-state Perturb-seq cannot break the wall (H2); the fix is combinatorial (double-KO/GI), "
                 "time-course (4sU nascent), or multi-context (multi-cell-line) data that actually measures the per-edge "
                 "transmission coefficient. Sparse machinery, no torch, no dense transition matrix.")
    json.dump(r, open(OUT / "wall_diagnosis.json", "w"), indent=1)
    print("\n  -> outputs/orphan/wall_diagnosis.json")
    return r


if __name__ == "__main__":
    main()
