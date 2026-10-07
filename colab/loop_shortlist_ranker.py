"""Loop 167. How much of the 93% ceiling can a learned re-ranker of the top-20 actually reach?

THE QUESTION, STATED EXACTLY. The merged score's top-20 contains the answer 92.67% of the time on
DEV, and it puts it FIRST 50.73% of the time. Re-ranking inside the top-20 cannot exceed 92.67% --
an answer outside the shortlist is unreachable by any ordering of the shortlist -- so 0.9267 is a
hard ceiling and the question is what fraction of the 0.4194 gap between 0.5073 and it is reachable.

WHY A LEARNED RANKER AND WHY NOW. Every merge in this arc has been a hand-weighted sum:
T + 0.9*max(zw, rb), or a fitted scalar on a score-space blend. Ranking 20 candidates with rich
per-candidate features is a far smaller problem than ranking 8,428, and DEV supplies 5,582 cases at
20 rows each. The design workflow's family B did reach hit@1 0.765 with a gradient-boosted ranker --
but its own verifier found a leak, and W2 closes that leak before anything is fitted.

THE LEAK, AND HOW IT IS CLOSED. A candidate whose ONLY reaction was the held-out one has j-deleted
degree exactly 0, and because a case excludes its own seeds from candidacy such a candidate can only
be a TRUE TARGET. 1,018 of 8,428 non-currency species have full degree 1, and the verifier measured
the indicator firing in 5.5% of cases with every flagged candidate a true positive. A tree model
reads that as a free label. Every degree-derived feature here is CLIPPED at 1, so degree 0 and
degree 1 are indistinguishable to the model, and W2 gates that the clip actually removes the signal
by measuring the raw indicator's AUC before and after.

PREDECLARED, before any number is looked at.

  W1 THE CEILING. Recall of the merged top-20 on DEV, and the arithmetic that hit@1 cannot exceed it.
     Gate: passes on being reported. Every number below is quoted as a fraction of this, so a gain
     cannot be read as progress toward 100% when the reachable maximum is 0.9267.

  W2 THE ORPHAN LEAK IS CLOSED. The indicator 1[j-deleted degree == 0] scored on the shortlist, and
     the same after clipping.
     Gate: PASS iff the clipped feature set contains no feature whose single-feature AUC on the
     shortlist exceeds 0.60 while being computable only from the held-out reaction's absence.

  W3 DOES THE LEARNED RANKER BEAT THE HAND-WEIGHTED MERGE? Case-grouped 5-fold CV over DEV, so no
     case appears in both fit and score.
     Gate: hit@1 improves by more than 3 sem.

  W4 HOW MUCH OF THE CEILING. Report hit@1 as a fraction of the 0.9267 reachable maximum, for the
     blend and for the ranker.
     Gate: passes on being reported.

  W5 WHICH FEATURES CARRY IT, and does any single one dominate? Permutation importance on the fitted
     model.
     Gate: passes on being reported. If one feature carries everything then this is that feature and
     not a learned merge.

  W6 WHAT THIS CANNOT SHOW. DEV only -- TEST was read once by loop 162. The ceiling is a property of
     the shortlist, so a better shortlist moves it and this loop does not attempt one. And a ranker
     fitted on curated reactions inherits whatever regularities curation introduced.

-> outputs/loop_shortlist_ranker.json
"""
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse, stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                    # noqa: E402
import run_manifest as RM                  # noqa: E402
from rem.harness import REM, auc_of        # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_shortlist_ranker.json"
CMAX, HASH_SEED = 6, 90210
BLOCK_SCALE, ALPHA, NITER = 0.9, 0.15, 60
TOPN, NFOLD, SEED = 20, 5, 16700
DEG_CLIP = 1

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 104)
    say("  RE-RANKING THE TOP-20: how much of the 92.67% ceiling is reachable?")
    say("=" * 104)
    say()

    R = REM()
    NC = len(R.noncur)
    Ei = np.rint(R.Enc).astype(np.int64)
    nz = (Ei != 0).any(1)
    heavy = R.Enc.sum(1) - R.Enc[:, list(map(str, R.elements)).index("H")]
    rngh = np.random.default_rng(HASH_SEED)
    h1 = rngh.integers(-(2 ** 62), 2 ** 62, Ei.shape[1], dtype=np.int64)
    h2 = rngh.integers(-(2 ** 62), 2 ** 62, Ei.shape[1], dtype=np.int64)

    def keys(V, h):
        with np.errstate(over="ignore"):
            return (V.astype(np.int64) * h).sum(1)
    own1, own2 = Counter(), Counter()
    for c in range(1, CMAX + 1):
        V = c * Ei[nz]
        for a, b in zip(keys(V, h1), keys(V, h2)):
            own1[int(a)] += 1
            own2[int(b)] += 1

    def tier_parts(cs):
        r0 = cs["residual"]
        r = np.rint(r0).astype(np.int64)
        solo = np.zeros(NC)
        pair = np.zeros(NC)
        feas = np.zeros(NC)
        if np.abs(r0 - r).max() > 1e-6 or (r < 0).any():
            return solo, pair, feas
        d1, d2 = Counter(), Counter()
        for s in cs["seeds"]:
            k = R.ncmap[int(s)]
            if not nz[k]:
                continue
            for c in range(1, CMAX + 1):
                d1[int(keys((c * Ei[k])[None, :], h1)[0])] += 1
                d2[int(keys((c * Ei[k])[None, :], h2)[0])] += 1
        feas = (((r - Ei) >= 0).all(1) & nz).astype(float)
        for c in range(1, CMAX + 1):
            L = r - c * Ei
            ok = (L >= 0).all(1) & nz
            solo[ok & (L == 0).all(1)] = 1.0
            if not ok.any():
                continue
            k1, k2 = keys(L, h1), keys(L, h2)
            for i in np.where(ok)[0]:
                a, b = int(k1[i]), int(k2[i])
                if own1[a] - d1.get(a, 0) > 0 and own2[b] - d2.get(b, 0) > 0:
                    pair[i] = 1.0
        return solo, pair, feas

    def restricted(cs, cols):
        keepsp = set(int(R.noncur[c]) for c in cols) | set(cs["seeds"])
        rxs = set()
        for i in keepsp:
            rxs |= R.sp_rx[i]
        rxs.discard(cs["j"])
        if not rxs:
            return np.zeros(len(cols))
        spl, rxl = sorted(keepsp), sorted(rxs)
        si = {v: k for k, v in enumerate(spl)}
        ri = {v: k + len(spl) for k, v in enumerate(rxl)}
        n = len(spl) + len(rxl)
        src, dst = [], []
        for j in rxl:
            rv = R.rev[j] == 1
            for i in R.react_of[j]:
                if i in si:
                    src += [si[i]]
                    dst += [ri[j]]
                    if rv:
                        src += [ri[j]]
                        dst += [si[i]]
            for i in R.prod_of[j]:
                if i in si:
                    src += [ri[j]]
                    dst += [si[i]]
                    if rv:
                        src += [si[i]]
                        dst += [ri[j]]
        if not src:
            return np.zeros(len(cols))
        A = sparse.csr_matrix((np.ones(len(src)), (dst, src)), shape=(n, n))
        col = np.asarray(A.sum(0)).ravel()
        col[col == 0] = 1.0
        P = A @ sparse.diags(1.0 / col)
        e = np.zeros(n)
        for s in cs["seeds"]:
            if s in si:
                e[si[s]] = 1.0
        if e.sum() == 0:
            return np.zeros(len(cols))
        e /= e.sum()
        p = e.copy()
        for _ in range(NITER):
            p = (1 - ALPHA) * (P @ p) + ALPHA * e
        return np.array([p[si[int(R.noncur[c])]] for c in cols])

    def r01(v):
        return (stats.rankdata(v, "average") - 1) / max(len(v) - 1, 1)

    FEATS = ["tier", "solo", "pair", "feas", "balance", "bal_rank", "walk", "walk_rank",
             "walk_zero", "chain", "chain_rank", "log_deg_clip", "log_heavy", "het_frac",
             "charge", "blend_rank", "resid_cos"]
    say(f"     features: {len(FEATS)} per candidate, degree clipped at {DEG_CLIP}")

    X, y, grp, base_rank = [], [], [], []
    orphan_ind = []
    n_ok, recall20, blend_hit1 = 0, [], []
    for t, j in enumerate(R.dev):
        cs = R.case(j)
        if cs is None:
            continue
        m = ~cs["excl"]
        if cs["pos"][m].sum() == 0 or (~cs["pos"][m]).sum() == 0:
            continue
        solo, pair, feas = tier_parts(cs)
        T = 8 * solo + 4 * pair + 1 * feas
        w = R.walk(R.operator(cs["j"]), cs["seeds"])[:R.NS][R.noncur]
        zw = np.zeros(NC)
        mm = w > 0
        nn = int(mm.sum())
        if nn == 1:
            zw[mm] = 1.0
        elif nn > 1:
            zw[mm] = 0.001 + 0.999 * (stats.rankdata(w[mm], "average") - 1) / (nn - 1)
        bal = R.balance_score(cs["residual"])
        rb = r01(bal)
        blend = T + BLOCK_SCALE * np.maximum(zw, rb) + 1e-6 * rb
        blend[cs["excl"]] = -np.inf
        order = np.argsort(-blend, kind="stable")
        cols = order[:TOPN]
        recall20.append(1.0 if cs["pos"][cols].any() else 0.0)
        blend_hit1.append(1.0 if cs["pos"][cols[0]] else 0.0)
        ch = restricted(cs, cols)
        dg = np.maximum(cs["degv"][cols], DEG_CLIP)
        res = cs["residual"]
        rn = np.linalg.norm(res)
        cosr = (R.Enc[cols] @ res) / (np.maximum(np.linalg.norm(R.Enc[cols], axis=1), 1e-9)
                                      * max(rn, 1e-9))
        F = np.column_stack([
            T[cols], solo[cols], pair[cols], feas[cols], bal[cols], rb[cols],
            w[cols], r01(w)[cols], (w[cols] == 0).astype(float),
            ch, r01(ch), np.log(dg), np.log1p(heavy[cols]),
            R.Enc[cols][:, [list(map(str, R.elements)).index(e) for e in ("N", "O")]].sum(1)
            / np.maximum(heavy[cols], 1),
            R.charge[R.noncur][cols], np.arange(len(cols), dtype=float) / TOPN, cosr])
        X.append(F)
        y.append(cs["pos"][cols].astype(int))
        grp.append(np.full(len(cols), n_ok))
        orphan_ind.append((cs["degv"][cols] == 0).astype(float))
        n_ok += 1
        if n_ok % 500 == 0:
            say(f"     {n_ok:,}/{len(R.dev):,} [{time.time()-t0:.0f}s]")
    X = np.vstack(X)
    y = np.concatenate(y)
    grp = np.concatenate(grp)
    orph = np.concatenate(orphan_ind)
    ceiling = float(np.mean(recall20))
    b1 = float(np.mean(blend_hit1))
    say(f"     {n_ok:,} cases, {X.shape[0]:,} rows, {int(y.sum()):,} positives")

    # ------------------------------------------------------------------ W1
    say()
    say("W1 THE CEILING")
    say(f"     merged top-{TOPN} recall on DEV: {ceiling:.4f}")
    say(f"     the blend puts it FIRST: {b1:.4f}")
    say(f"     reachable gap: {ceiling - b1:+.4f}. hit@1 cannot exceed {ceiling:.4f} by any "
        f"ordering of this shortlist.")
    w1 = True
    say(f"     W1 {'PASS' if w1 else 'FAIL'}")

    # ------------------------------------------------------------------ W2
    say()
    say("W2 THE ORPHAN LEAK")
    a_raw = auc_of(orph, y.astype(bool))
    dcol = X[:, FEATS.index("log_deg_clip")]
    a_clip = auc_of(-dcol, y.astype(bool))
    say(f"     1[j-deleted degree == 0] fires on {int(orph.sum()):,} of {len(orph):,} shortlist "
        f"rows; of those {float(y[orph > 0].mean() if orph.sum() else 0):.1%} are true positives")
    say(f"     its AUC as a lone feature: {a_raw:.4f}")
    say(f"     the CLIPPED log-degree feature the model sees: AUC {a_clip:.4f}")
    leaky = [f for k, f in enumerate(FEATS)
             if f.startswith("log_deg") and auc_of(-X[:, k], y.astype(bool)) > 0.60]
    w2 = bool(len(leaky) == 0)
    GG.verdict(w2, emit=say, if_true=(
        "clipping at 1 makes degree 0 and degree 1 indistinguishable, so the one-sided label "
        "indicator is not available to the model."), if_false=(
        f"a degree feature still exceeds 0.60 alone ({leaky}); the leak is not closed."))
    say(f"     W2 {'PASS' if w2 else 'FAIL'}")

    # ------------------------------------------------------------------ W3
    say()
    say("W3 THE LEARNED RANKER, case-grouped 5-fold")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=NFOLD).split(X, y, grp):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                             max_leaf_nodes=31, min_samples_leaf=40,
                                             random_state=0)
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    hit1_m, hit1_b = [], []
    for g in range(n_ok):
        s = grp == g
        p, yy = pred[s], y[s].astype(bool)
        hit1_m.append(1.0 if yy[np.argmax(p)] else 0.0)
        hit1_b.append(1.0 if yy[0] else 0.0)
    hm, hb = np.array(hit1_m), np.array(hit1_b)
    d3 = float((hm - hb).mean())
    s3 = float((hm - hb).std() / np.sqrt(len(hm)))
    w3 = bool(d3 > 3 * s3)
    say(f"     blend hit@1 {hb.mean():.4f} -> ranker hit@1 {hm.mean():.4f} = {d3:+.4f} "
        f"sem {s3:.4f} ({d3/s3:+.1f} sem)")
    GG.verdict(w3, emit=say, if_true=(
        "a learned re-ranker of 20 candidates beats the hand-weighted merge."), if_false=(
        "the learned ranker does not beat the hand-weighted merge on the shortlist."))
    say(f"     W3 {'PASS' if w3 else 'FAIL'}")

    # ------------------------------------------------------------------ W4
    say()
    say("W4 FRACTION OF THE CEILING")
    say(f"     ceiling                    {ceiling:.4f}   (100%)")
    say(f"     hand-weighted blend        {hb.mean():.4f}   ({hb.mean()/ceiling:.1%} of ceiling)")
    say(f"     learned re-ranker          {hm.mean():.4f}   ({hm.mean()/ceiling:.1%} of ceiling)")
    say(f"     remaining gap to ceiling   {ceiling - hm.mean():+.4f}")
    w4 = True
    say(f"     W4 {'PASS' if w4 else 'FAIL'}")

    # ------------------------------------------------------------------ W5
    say()
    say("W5 PERMUTATION IMPORTANCE (drop in hit@1 when one feature is shuffled within case)")
    rng = np.random.default_rng(SEED)
    tr, te = next(iter(GroupKFold(n_splits=NFOLD).split(X, y, grp)))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                         min_samples_leaf=40, random_state=0).fit(X[tr], y[tr])

    def h1_of(P, idx):
        out = []
        for g in np.unique(grp[idx]):
            s = grp[idx] == g
            out.append(1.0 if y[idx][s].astype(bool)[np.argmax(P[s])] else 0.0)
        return float(np.mean(out))
    p0 = clf.predict_proba(X[te])[:, 1]
    b0 = h1_of(p0, te)
    imp = {}
    for k, f in enumerate(FEATS):
        Xp = X[te].copy()
        Xp[:, k] = rng.permutation(Xp[:, k])
        imp[f] = b0 - h1_of(clf.predict_proba(Xp)[:, 1], te)
    for f, v in sorted(imp.items(), key=lambda kv: -kv[1])[:10]:
        say(f"     {f:<16s} {v:+.4f}")
    top = max(imp.values())
    w5 = True
    GG.verdict(top < 0.5 * b0, emit=say, if_true="no single feature carries the ranker.",
               if_false="one feature dominates; this is that feature, not a learned merge.")
    say(f"     W5 {'PASS' if w5 else 'FAIL'}")

    say()
    say("W6 WHAT THIS CANNOT SHOW")
    say(f"     DEV only. TEST was read once by loop 162 and is not read here.")
    say(f"     {ceiling:.4f} is a property of the SHORTLIST, not of the method. A better shortlist")
    say("     moves the ceiling and this loop does not attempt one.")
    say("     A ranker fitted on curated reactions inherits whatever regularities curation left.")
    w6 = True
    say(f"     W6 {'PASS' if w6 else 'FAIL'}")

    gates = {"W1": w1, "W2": w2, "W3": w3, "W4": w4, "W5": w5, "W6": w6}
    man = RM.manifest(inputs=[Path("colab/data/rem_bipartite.npz"),
                              Path("colab/data/rem_chem.npz")],
                      available=len(R.dev), used=n_ok, selection="all", seed=SEED,
                      controls=["the orphan-by-deletion leak clipped and the clip gated at W2",
                                "case-grouped folds so no case is in both fit and score",
                                "the ceiling reported first, so gains are read as a fraction of what is reachable",
                                "permutation importance, so a single dominant feature cannot hide",
                                "DEV only; TEST untouched"],
                      note="learned re-ranking of the merged top-20 against its own recall ceiling")
    out = {"test": "shortlist re-ranker", "gates": gates, "n": n_ok,
           "ceiling": ceiling, "blend_hit1": float(hb.mean()), "ranker_hit1": float(hm.mean()),
           "delta": [d3, s3], "importance": imp,
           "leak": {"raw_auc": a_raw, "clipped_auc": a_clip},
           "manifest": man, "seconds": time.time() - t0, "log": log}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time()-t0:.0f}s]")
    say("=" * 104)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
