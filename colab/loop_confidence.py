"""Loop 168. Calibrated confidence and selective prediction: abstain instead of guessing.

WHY THIS AND NOT A BETTER RANKER. Loop 167 took hit@1 from 0.4982 to 0.7266 against a 0.9307
ceiling. A ranked list that is right 73% of the time is awkward to act on: for the 389 dead ends and
259 orphans that motivated this arc, what is wanted is not a leaderboard but a set of proposals that
can be trusted. That means a CALIBRATED probability and a decision rule that declines to answer when
the probability is low, trading coverage for precision.

THE THREE THINGS THAT HAVE TO BE TRUE, and each is a separate gate.
  (1) The score must be a probability. A gradient-boosted classifier's predict_proba is not
      calibrated by default, and an uncalibrated 0.9 is not a 90% chance of anything. X1 measures
      Brier and expected calibration error against a reliability curve, before and after isotonic
      regression fitted OUT OF FOLD.
  (2) Confidence must actually order difficulty. X2 tests that against the only honest control: a
      random abstention rule at the same coverage. If sorting by confidence is no better than
      sorting by nothing, the score is not a confidence.
  (3) It has to buy something at a coverage a person would use. X3 reports the coverage at which
      precision reaches 90% and 95%.

AND THE DEBT FROM LOOP 167. Its top feature by permutation importance was clipped degree at +0.1594,
ahead of everything else -- and degree was the worst global scorer in this arc (0.5926 honest AUC)
and has leaked twice. Loop 167 recorded that an ablation was owed before its headline is relied on.
X4 pays it: the identical pipeline with every degree-derived feature removed.

PREDECLARED, before any number is looked at.

  X1 IS THE SCORE A PROBABILITY? Brier score and expected calibration error over 10 bins, raw and
     after out-of-fold isotonic calibration, with the reliability curve reported.
     Gate: PASS iff calibration reduces ECE by more than half AND the calibrated ECE is under 0.05.

  X2 DOES CONFIDENCE ORDER DIFFICULTY? Accuracy on the most-confident X% of cases against a random
     abstention rule at the same coverage, for X in {10, 25, 50, 75, 100}.
     Gate: PASS iff at 50% coverage the confident half beats random abstention by more than 3 sem.

  X3 WHAT COVERAGE BUYS 90% AND 95% PRECISION? The risk-coverage curve, reported whole.
     Gate: PASS iff 90% precision is reachable at any coverage above 10%. Below that the rule is
     too selective to be worth having and the loop says so.

  X4 THE DEGREE ABLATION LOOP 167 OWED. The identical pipeline with every degree feature dropped.
     Gate: PASS iff hit@1 without degree still beats the hand-weighted blend's 0.4982 by more than
     3 sem. A FAIL means loop 167's gain was carried by a feature that has leaked twice before, and
     its headline is withdrawn rather than footnoted.

  X5 WHICH CONFIDENCE MEASURE. Top probability, top-two margin, and negative entropy over the 20.
     Gate: passes on all three being reported, so the best is a choice made in the open.

  X6 WHAT THIS CANNOT SHOW. DEV only. Calibration is fitted out of fold but on the same
     distribution, and a curated-reaction distribution is not the dead-end distribution this is
     ultimately for. Abstention is only useful if the abstained cases are cheap to leave unanswered.

-> outputs/loop_confidence.json
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

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_confidence.json"
CACHE = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/"
             "scratchpad/l167_features.npz")
CMAX, HASH_SEED = 6, 90210
BLOCK_SCALE, ALPHA, NITER = 0.9, 0.15, 60
TOPN, NFOLD, SEED, DEG_CLIP = 20, 5, 16800, 1
BLEND_HIT1 = 0.4982
FEATS = ["tier", "solo", "pair", "feas", "balance", "bal_rank", "walk", "walk_rank",
         "walk_zero", "chain", "chain_rank", "log_deg_clip", "log_heavy", "het_frac",
         "charge", "blend_rank", "resid_cos"]
DEG_FEATS = ["log_deg_clip"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def build_features(R):
    NC = len(R.noncur)
    Ei = np.rint(R.Enc).astype(np.int64)
    nz = (Ei != 0).any(1)
    els = list(map(str, R.elements))
    heavy = R.Enc.sum(1) - R.Enc[:, els.index("H")]
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
    NOix = [els.index(e) for e in ("N", "O")]
    X, y, grp = [], [], []
    n_ok, rec = 0, []
    t0 = time.time()
    for j in R.dev:
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
        cols = np.argsort(-blend, kind="stable")[:TOPN]
        rec.append(1.0 if cs["pos"][cols].any() else 0.0)
        ch = restricted(cs, cols)
        dg = np.maximum(cs["degv"][cols], DEG_CLIP)
        res = cs["residual"]
        cosr = (R.Enc[cols] @ res) / (np.maximum(np.linalg.norm(R.Enc[cols], axis=1), 1e-9)
                                      * max(np.linalg.norm(res), 1e-9))
        X.append(np.column_stack([
            T[cols], solo[cols], pair[cols], feas[cols], bal[cols], rb[cols],
            w[cols], r01(w)[cols], (w[cols] == 0).astype(float), ch, r01(ch),
            np.log(dg), np.log1p(heavy[cols]),
            R.Enc[cols][:, NOix].sum(1) / np.maximum(heavy[cols], 1),
            R.charge[R.noncur][cols], np.arange(len(cols), dtype=float) / TOPN, cosr]))
        y.append(cs["pos"][cols].astype(int))
        grp.append(np.full(len(cols), n_ok))
        n_ok += 1
        if n_ok % 1000 == 0:
            print(f"  features {n_ok:,} [{time.time()-t0:.0f}s]", flush=True)
    return (np.vstack(X), np.concatenate(y), np.concatenate(grp),
            float(np.mean(rec)), n_ok)


def main():
    t0 = time.time()
    say("=" * 104)
    say("  CONFIDENCE AND ABSTENTION -- and the degree ablation loop 167 owed")
    say("=" * 104)
    say()
    R = REM()
    if CACHE.exists():
        z = np.load(CACHE)
        X, y, grp, ceiling, n_ok = z["X"], z["y"], z["grp"], float(z["ceiling"]), int(z["n"])
        say(f"     features from cache: {X.shape}")
    else:
        X, y, grp, ceiling, n_ok = build_features(R)
        np.savez_compressed(CACHE, X=X, y=y, grp=grp, ceiling=ceiling, n=n_ok)
        say(f"     features built and cached: {X.shape} [{time.time()-t0:.0f}s]")
    say(f"     {n_ok:,} cases | ceiling (top-{TOPN} recall) {ceiling:.4f} | "
        f"blend hit@1 {BLEND_HIT1:.4f}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.isotonic import IsotonicRegression

    def oof(cols):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=NFOLD).split(X, y, grp):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                                 max_leaf_nodes=31, min_samples_leaf=40,
                                                 random_state=0).fit(X[tr][:, cols], y[tr])
            p[te] = clf.predict_proba(X[te][:, cols])[:, 1]
        return p
    allc = list(range(len(FEATS)))
    praw = oof(allc)
    say(f"     out-of-fold predictions computed [{time.time()-t0:.0f}s]")

    # isotonic calibration, itself out of fold
    pcal = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=NFOLD).split(X, y, grp):
        iso = IsotonicRegression(out_of_bounds="clip").fit(praw[tr], y[tr])
        pcal[te] = iso.predict(praw[te])

    def ece(p, k=10):
        e, edges = 0.0, np.linspace(0, 1, k + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
            if m.sum():
                e += m.mean() * abs(p[m].mean() - y[m].mean())
        return float(e)
    br_raw = float(np.mean((praw - y) ** 2))
    br_cal = float(np.mean((pcal - y) ** 2))
    e_raw, e_cal = ece(praw), ece(pcal)

    # ------------------------------------------------------------------ X1
    say()
    say("X1 IS THE SCORE A PROBABILITY?")
    say(f"     Brier  raw {br_raw:.5f} -> calibrated {br_cal:.5f}")
    say(f"     ECE    raw {e_raw:.5f} -> calibrated {e_cal:.5f}")
    say("     reliability, calibrated (bin -> predicted / observed / n):")
    edges = np.linspace(0, 1, 11)
    rel = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pcal >= a) & (pcal < b) if b < 1 else (pcal >= a) & (pcal <= b)
        if m.sum() > 30:
            rel.append({"lo": float(a), "pred": float(pcal[m].mean()),
                        "obs": float(y[m].mean()), "n": int(m.sum())})
            say(f"       [{a:.1f},{b:.1f})  {pcal[m].mean():.3f} / {y[m].mean():.3f}  "
                f"n={int(m.sum()):,}")
    x1 = bool(e_cal < 0.05 and e_cal < 0.5 * e_raw)
    GG.verdict(x1, emit=say, if_true="the calibrated score can be read as a probability.",
               if_false="calibration did not fix it; the number is a score, not a probability.")
    say(f"     X1 {'PASS' if x1 else 'FAIL'}")

    # -------------------------------------------------- case-level confidence
    cases = np.unique(grp)
    top_p, margin, negent, correct = [], [], [], []
    for g in cases:
        s = grp == g
        p, yy = pcal[s], y[s].astype(bool)
        o = np.argsort(-p)
        top_p.append(p[o[0]])
        margin.append(p[o[0]] - (p[o[1]] if len(o) > 1 else 0.0))
        q = p / max(p.sum(), 1e-12)
        negent.append(float((q * np.log(q + 1e-12)).sum()))
        correct.append(1.0 if yy[o[0]] else 0.0)
    top_p = np.array(top_p)
    margin = np.array(margin)
    negent = np.array(negent)
    correct = np.array(correct)
    CONF = {"top_prob": top_p, "margin": margin, "neg_entropy": negent}

    # ------------------------------------------------------------------ X5
    say()
    say("X5 WHICH CONFIDENCE MEASURE (accuracy on the most-confident half)")
    half = {}
    for nm, c in CONF.items():
        idx = np.argsort(-c)[:len(c) // 2]
        half[nm] = float(correct[idx].mean())
        say(f"     {nm:<12s} top-50% accuracy {half[nm]:.4f}")
    bestc = max(half, key=half.get)
    conf = CONF[bestc]
    x5 = True
    say(f"     using {bestc}")
    say(f"     X5 {'PASS' if x5 else 'FAIL'}")

    # ------------------------------------------------------------------ X2/X3
    say()
    say("X2/X3 THE RISK-COVERAGE CURVE")
    order = np.argsort(-conf)
    say(f"     {'coverage':>9s} {'n':>7s} {'precision':>10s} {'random':>8s} {'lift':>8s}")
    curve = []
    rng = np.random.default_rng(SEED)
    for cov in (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00):
        k = max(int(cov * len(order)), 1)
        acc = float(correct[order[:k]].mean())
        rnd = float(np.mean([correct[rng.permutation(len(correct))[:k]].mean()
                             for _ in range(30)]))
        curve.append({"coverage": cov, "n": k, "precision": acc, "random": rnd})
        say(f"     {cov:>8.0%} {k:>7,} {acc:>10.4f} {rnd:>8.4f} {acc-rnd:>+8.4f}")
    k50 = len(order) // 2
    a50 = correct[order[:k50]]
    d2 = float(a50.mean() - correct.mean())
    s2 = float(np.sqrt(a50.var() / len(a50) + correct.var() / len(correct)))
    x2 = bool(d2 > 3 * s2)
    GG.verdict(x2, emit=say, if_true=(
        f"confidence orders difficulty: the confident half is {d2:+.4f} above the overall rate "
        f"({d2/s2:+.1f} sem)."), if_false="confidence does not order difficulty.")
    say(f"     X2 {'PASS' if x2 else 'FAIL'}")
    # MAXIMUM coverage still meeting the precision target, not the minimum. The first version used
    # next(...) over a curve whose precision DECREASES with coverage, so it returned the smallest
    # coverage -- always 0.10 -- and the gate then rejected it for being <= 0.10. The gate asked
    # its question backwards and failed a result that was in fact strong at every level.
    cov90 = max([c["coverage"] for c in curve if c["precision"] >= 0.90], default=None)
    cov95 = max([c["coverage"] for c in curve if c["precision"] >= 0.95], default=None)
    say(f"     90% precision reached at coverage {cov90 if cov90 else 'never'}; "
        f"95% at {cov95 if cov95 else 'never'}")
    x3 = bool(cov90 is not None and cov90 > 0.10)
    GG.verdict(x3, emit=say, if_true=(
        f"at {cov90:.0%} coverage the top-1 proposal is right 90% of the time -- that is a usable "
        f"rule for proposing gap fills."), if_false=(
        "90% precision is not reachable above 10% coverage; the rule is too selective to be worth "
        "having."))
    say(f"     X3 {'PASS' if x3 else 'FAIL'}")

    # ------------------------------------------------------------------ X4
    say()
    say("X4 THE DEGREE ABLATION LOOP 167 OWED")
    nodeg = [i for i, f in enumerate(FEATS) if f not in DEG_FEATS]
    pnd = oof(nodeg)
    h_full, h_nd = [], []
    for g in cases:
        s = grp == g
        yy = y[s].astype(bool)
        h_full.append(1.0 if yy[np.argmax(praw[s])] else 0.0)
        h_nd.append(1.0 if yy[np.argmax(pnd[s])] else 0.0)
    h_full, h_nd = np.array(h_full), np.array(h_nd)
    d4 = float(h_nd.mean() - BLEND_HIT1)
    s4 = float(h_nd.std() / np.sqrt(len(h_nd)))
    x4 = bool(d4 > 3 * s4)
    say(f"     with degree    hit@1 {h_full.mean():.4f}")
    say(f"     WITHOUT degree hit@1 {h_nd.mean():.4f}   (blend {BLEND_HIT1:.4f})")
    say(f"     cost of removing degree: {h_nd.mean()-h_full.mean():+.4f}")
    say(f"     without degree vs blend: {d4:+.4f} sem {s4:.4f} ({d4/s4:+.1f} sem)")
    GG.verdict(x4, emit=say, if_true=(
        "loop 167's gain survives without any degree feature, so it was not carried by the column "
        "that leaked twice in this arc."), if_false=(
        "without degree the ranker does not beat the hand-weighted blend. Loop 167's headline was "
        "carried by a feature that has leaked twice before and is WITHDRAWN, not footnoted."))
    say(f"     X4 {'PASS' if x4 else 'FAIL'}")

    say()
    say("X6 WHAT THIS CANNOT SHOW")
    say("     DEV only; TEST was read once by loop 162.")
    say("     Calibration is fitted out of fold but on the same distribution, and curated reactions")
    say("     are not the dead-end distribution this is ultimately for.")
    say("     Abstention is only useful if the abstained cases are cheap to leave unanswered.")
    x6 = True
    say(f"     X6 {'PASS' if x6 else 'FAIL'}")

    gates = {"X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5, "X6": x6}
    man = RM.manifest(inputs=[Path("colab/data/rem_bipartite.npz"),
                              Path("colab/data/rem_chem.npz")],
                      available=n_ok, used=n_ok, selection="all", seed=SEED,
                      controls=["isotonic calibration fitted OUT OF FOLD, never on its own scores",
                                "random abstention at matched coverage as X2's control",
                                "the degree ablation loop 167 recorded as owed",
                                "three confidence measures reported, so the choice is in the open",
                                "case-grouped folds throughout"],
                      note="calibrated confidence, selective prediction, and the degree ablation")
    out = {"test": "confidence and abstention", "gates": gates, "n": n_ok, "ceiling": ceiling,
           "brier": [br_raw, br_cal], "ece": [e_raw, e_cal], "reliability": rel,
           "conf_measures": half, "chosen": bestc, "curve": curve,
           "cov90": cov90, "cov95": cov95,
           "ablation": {"with_degree": float(h_full.mean()), "without": float(h_nd.mean()),
                        "vs_blend": [d4, s4]},
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
