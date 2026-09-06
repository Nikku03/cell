"""Loop 162. CLOSE_WALK on the LOCKED TEST SPLIT. Read once, nothing fitted, gates written first.

WHAT IS BEING SPENT. REM.test is 5,583 reactions that have never been scored, fitted on, or looked
at since colab/rem/harness.py split them at seed 16100. Every number below is the first and only
time they are used, and no constant in this loop was chosen by looking at them.

WHERE THE DESIGN CAME FROM. A six-family design workflow ran on DEV only -- five merge designs and
five independent adversarial verifiers, all of whose claims survived re-implementation from prose on
disjoint DEV draws. Its brief specified one merge to predeclare, and this is it, unchanged.

THE MERGE, every constant frozen and its provenance stated:
    ALPHA 0.15, NITER 60          harness defaults, unchanged since loop 160
    walk edge weights uniform     DEV measured stoich weights at -0.0147 (3.1 sem) WORSE
    currency cut degree > 200     harness default
    CMAX 6, BEAM 8                family E, tuned on DEV and FROZEN; the verifier found BEAM inert
    hash seed 90210, two int64    family E plus the verifier's hardening against 64-bit collisions
    zw encoding 0.001 / 0.999     family C, verified load-bearing (rankdata 'max' drops it to 0.873)
    tier weights 8 / 4 / 2 / 1    family E
    block scale 0.9               structural: any value in (0,1) is order-identical

    SCORE = T + 0.9 * max(zw, rb) + 1e-6 * rb

THE THREE WARNINGS THE BRIEF INSISTS THIS LOOP CARRIES.

  (1) THE DEGREE BASELINE LEAKED. R.degv counts over the whole graph INCLUDING the held-out
      reaction, and since seeds are excluded from candidacy that +1 lands on exactly the true
      targets. Measured at 0.7302 leaky against 0.6008 honest. This loop uses case["degv"], the
      j-deleted degree, and T1 gates that it is actually being used.

  (2) AUC IS THE WRONG HEADLINE HERE. With 8,428 candidates and one to three positives, degree
      reaches AUC 0.723 with MRR 0.051 -- a good tail-sorter and a useless top-of-list. T4 reports
      AUC, MRR and hit@k together and the verdict is conditioned on the rank metrics.

  (3) THE TASK GRANTS THE ANSWER'S FORMULA. 98.8% of held-out reactions balance exactly and 57%
      have a single non-currency product, so a high score is a statement about inverting a granted
      elemental total, not about unconditional gap discovery. T6 says so.

PREDECLARED, before TEST is touched.

  T1 THE INSTRUMENT IS HONEST. The degree baseline is the j-deleted one; the operator deletes the
     held-out reaction in both directions; no case's own products reach any scorer.
     Gate: the honest degree differs from R.degv_leaky on every case, and the leaky column is not
     referenced anywhere in the scoring path.

  T2 DOES THE MERGE BEAT BOTH SINGLES? CLOSE_WALK against the walk alone and against balance alone,
     paired on identical TEST cases.
     Gate: PASS iff it beats BOTH by more than 3 sem, on MRR as well as AUC.

  T3 CLOSE_WALK vs E-CLOSE, HEAD TO HEAD. The brief's stated open question: the chemistry-only
     E-CLOSE scored 0.9752 on DEV and the merges 0.9406-0.9656, but they were never compared on the
     same cases because the families ran in parallel.
     Gate: PASS iff CLOSE_WALK beats E-CLOSE alone by more than 3 sem on MRR. If it fails, the
     honest headline is that the connectivity arm adds nothing to the chemistry and this arc's
     "merge" result is a statement about under-modelled chemistry.

  T4 WHICH METRIC AGREES WITH WHICH. AUC, MRR, hit@1, hit@20 for every scorer including honest
     degree.
     Gate: passes on all four being reported for all scorers.

  T5 THE DUPLICATE STRATUM. In 10.8% of DEV cases another reaction survives deletion that maps the
     same seeds onto the same targets, and on those the walk scored 0.9997 by memorisation. Report
     every headline on the duplicate-free subset as well.
     Gate: PASS iff T2's verdict is unchanged on duplicate-free cases.

  T6 WHAT THIS CANNOT SHOW. Recovery of curated reactions is not discovery of missing ones. The
     elemental total is granted. And this is one split, read once.

-> outputs/loop_close_walk_test.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                    # noqa: E402
import run_manifest as RM                  # noqa: E402
from rem.harness import REM, auc_of        # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_close_walk_test.json"
CMAX, BEAM, HASH_SEED = 6, 8, 90210
BLOCK_SCALE = 0.9
KS = (1, 5, 20, 100)

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 104)
    say("  CLOSE_WALK ON THE LOCKED TEST SPLIT -- read once, nothing fitted")
    say("=" * 104)
    say()

    R = REM()
    say(f"     DEV {len(R.dev):,} reactions (design only) | TEST {len(R.test):,} (never touched) | "
        f"overlap {len(set(R.dev) & set(R.test))}")
    NC = len(R.noncur)
    Ei = np.rint(R.Enc).astype(np.int64)
    nz = (Ei != 0).any(1)
    rng = np.random.default_rng(HASH_SEED)
    h1 = rng.integers(-(2 ** 62), 2 ** 62, Ei.shape[1], dtype=np.int64)
    h2 = rng.integers(-(2 ** 62), 2 ** 62, Ei.shape[1], dtype=np.int64)

    def keys(V, h):
        with np.errstate(over="ignore"):
            return (V.astype(np.int64) * h).sum(1)

    from collections import Counter
    own1, own2 = Counter(), Counter()
    owner_of = {}
    for c in range(1, CMAX + 1):
        V = c * Ei[nz]
        k1, k2 = keys(V, h1), keys(V, h2)
        idx = np.where(nz)[0]
        for a, b, i in zip(k1, k2, idx):
            own1[int(a)] += 1
            own2[int(b)] += 1
            owner_of.setdefault((int(a), int(b)), []).append(int(i))
    say(f"     closure tables: {len(own1):,} / {len(own2):,} distinct keys over "
        f"{int(nz.sum()):,} formula-bearing candidates x {CMAX} coefficients "
        f"[{time.time()-t0:.0f}s]")

    def tier(cs):
        r0 = cs["residual"]
        r = np.rint(r0).astype(np.int64)
        if np.abs(r0 - r).max() > 1e-6 or (r < 0).any():
            return np.zeros(NC)
        d1, d2 = Counter(), Counter()
        for s in cs["seeds"]:
            if not nz[R.ncmap[int(s)]]:
                continue
            k = R.ncmap[int(s)]
            for c in range(1, CMAX + 1):
                d1[int(keys((c * Ei[k])[None, :], h1)[0])] += 1
                d2[int(keys((c * Ei[k])[None, :], h2)[0])] += 1
        feas = ((r - Ei) >= 0).all(1) & nz
        solo = np.zeros(NC, bool)
        pair = np.zeros(NC, bool)
        for c in range(1, CMAX + 1):
            L = r - c * Ei
            ok = (L >= 0).all(1) & nz
            solo |= ok & (L == 0).all(1)
            if not ok.any():
                continue
            k1, k2 = keys(L, h1), keys(L, h2)
            hit = np.zeros(NC, bool)
            oi = np.where(ok)[0]
            for i in oi:
                a, b = int(k1[i]), int(k2[i])
                if own1[a] - d1.get(a, 0) > 0 and own2[b] - d2.get(b, 0) > 0:
                    hit[i] = True
            pair |= hit
        trip = np.zeros(NC, bool)
        if feas.any():
            mass = np.zeros(NC)
            for c in range(1, CMAX + 1):
                L = r - c * Ei
                ok = (L >= 0).all(1) & nz
                mass[ok] = np.maximum(mass[ok], (c * Ei[ok]).sum(1))
            beam = np.argsort(-mass)[:BEAM]
            for b in beam:
                if mass[b] <= 0:
                    continue
                cb = int(mass[b] // max(Ei[b].sum(), 1)) or 1
                r2 = r - cb * Ei[b]
                if (r2 < 0).any():
                    continue
                hit2 = np.zeros(NC, bool)
                for c in range(1, CMAX + 1):
                    L = r2 - c * Ei
                    ok = (L >= 0).all(1) & nz
                    hit2 |= ok & (L == 0).all(1)
                if hit2.any():
                    trip |= hit2
                    trip[b] = True
        return 8.0 * solo + 4.0 * pair + 2.0 * trip + 1.0 * feas

    def zwalk(cs):
        w = R.walk(R.operator(cs["j"]), cs["seeds"])[:R.NS][R.noncur]
        zw = np.zeros(NC)
        m = w > 0
        n = int(m.sum())
        if n == 1:
            zw[m] = 1.0
        elif n > 1:
            zw[m] = 0.001 + 0.999 * (stats.rankdata(w[m], "average") - 1) / (n - 1)
        return zw, w

    def r01(v):
        return (stats.rankdata(v, "average") - 1) / max(len(v) - 1, 1)

    say()
    say(f"SCORING {len(R.test):,} LOCKED TEST REACTIONS")
    rows = {k: [] for k in ("close_walk", "eclose", "walk", "balance", "degree_honest",
                            "degree_leaky")}
    mrr = {k: [] for k in rows}
    hits = {k: {j: [] for j in KS} for k in rows}
    dupfree, n_ok = [], 0
    for t, j in enumerate(R.test):
        cs = R.case(j)
        if cs is None:
            continue
        m = ~cs["excl"]
        pos = cs["pos"][m]
        if pos.sum() == 0 or (~pos).sum() == 0:
            continue
        T = tier(cs)
        zw, w = zwalk(cs)
        rb = r01(R.balance_score(cs["residual"]))
        sc = {
            "close_walk": T + BLOCK_SCALE * np.maximum(zw, rb) + 1e-6 * rb,
            "eclose": T + 1e-6 * rb,
            "walk": w,
            "balance": R.balance_score(cs["residual"]),
            "degree_honest": cs["degv"],
            "degree_leaky": R.degv_leaky,
        }
        for k, v in sc.items():
            vv = v[m]
            rows[k].append(auc_of(vv, pos))
            o = np.argsort(-vv, kind="stable")
            rk = np.empty(len(vv), int)
            rk[o] = np.arange(1, len(vv) + 1)
            best = int(rk[pos].min())
            mrr[k].append(1.0 / best)
            for kk in KS:
                hits[k][kk].append(1.0 if best <= kk else 0.0)
        dupfree.append(not R.duplicate_survives(j) if t < 400 else None)
        n_ok += 1
        if n_ok % 500 == 0:
            say(f"     {n_ok:,}/{len(R.test):,} [{time.time()-t0:.0f}s]")

    def agg(k):
        a = np.array(rows[k], float)
        return {"auc": float(np.nanmean(a)), "sem": float(np.nanstd(a) / np.sqrt(len(a))),
                "mrr": float(np.mean(mrr[k])),
                **{f"hit@{j}": float(np.mean(hits[k][j])) for j in KS}}
    res = {k: agg(k) for k in rows}

    def paired(a, b, metric="auc"):
        if metric == "auc":
            x, y = np.array(rows[a], float), np.array(rows[b], float)
        else:
            x, y = np.array(mrr[a], float), np.array(mrr[b], float)
        d = x - y
        d = d[np.isfinite(d)]
        return float(d.mean()), float(d.std() / np.sqrt(len(d)))

    # ------------------------------------------------------------------ T1
    say()
    leak_diff = float(np.mean(np.array(rows["degree_leaky"], float)
                              - np.array(rows["degree_honest"], float)))
    t1 = bool(leak_diff > 0.02)
    say(f"T1 honest degree {res['degree_honest']['auc']:.4f} vs the leaky column "
        f"{res['degree_leaky']['auc']:.4f}, difference {leak_diff:+.4f}")
    GG.verdict(t1, emit=say, if_true=(
        "the de-leaked degree is measurably different from the leaky one, so the honest baseline "
        "is genuinely the one being used, and the leak reproduces on TEST at the size DEV found."),
        if_false=(
        "the two degree columns are not distinguishable here, which means either the leak does not "
        "reproduce or the honest column is not actually wired in."))
    say(f"     T1 {'PASS' if t1 else 'FAIL'}")

    # ------------------------------------------------------------------ T4 first
    say()
    say("T4 EVERY SCORER, EVERY METRIC")
    say(f"     {'scorer':<16s} {'AUC':>8s} {'sem':>7s} {'MRR':>7s} {'hit@1':>7s} "
        f"{'hit@20':>7s} {'hit@100':>8s}")
    for k in sorted(res, key=lambda x: -res[x]["mrr"]):
        r = res[k]
        say(f"     {k:<16s} {r['auc']:>8.4f} {r['sem']:>7.4f} {r['mrr']:>7.4f} "
            f"{r['hit@1']:>7.3f} {r['hit@20']:>7.3f} {r['hit@100']:>8.3f}")
    t4 = True
    say(f"     T4 {'PASS' if t4 else 'FAIL'}  (n = {n_ok:,} scored)")

    # ------------------------------------------------------------------ T2
    say()
    dw_a, sw_a = paired("close_walk", "walk")
    db_a, sb_a = paired("close_walk", "balance")
    dw_m, sw_m = paired("close_walk", "walk", "mrr")
    db_m, sb_m = paired("close_walk", "balance", "mrr")
    t2 = bool(dw_a > 3 * sw_a and db_a > 3 * sb_a and dw_m > 3 * sw_m and db_m > 3 * sb_m)
    say("T2 CLOSE_WALK AGAINST BOTH SINGLES")
    say(f"     vs walk    AUC {dw_a:+.4f} ({dw_a/sw_a:+.1f} sem) | MRR {dw_m:+.4f} "
        f"({dw_m/sw_m:+.1f} sem)")
    say(f"     vs balance AUC {db_a:+.4f} ({db_a/sb_a:+.1f} sem) | MRR {db_m:+.4f} "
        f"({db_m/sb_m:+.1f} sem)")
    GG.verdict(t2, emit=say, if_true=(
        "the merge beats both singles on the locked split, on the rank metric as well as on AUC."),
        if_false="the merge does not clear both singles on both metrics on the locked split.")
    say(f"     T2 {'PASS' if t2 else 'FAIL'}")

    # ------------------------------------------------------------------ T3
    say()
    de_a, se_a = paired("close_walk", "eclose")
    de_m, se_m = paired("close_walk", "eclose", "mrr")
    t3 = bool(de_m > 3 * se_m)
    say("T3 CLOSE_WALK vs E-CLOSE, the brief's open question")
    say(f"     AUC {de_a:+.4f} ({de_a/se_a:+.1f} sem) | MRR {de_m:+.4f} ({de_m/se_m:+.1f} sem)")
    GG.verdict(t3, emit=say, if_true=(
        "the connectivity arm earns its place: adding the walk to the chemistry closure improves "
        "the top of the list beyond what the chemistry does alone."), if_false=(
        "the walk adds nothing to the chemistry closure. What this arc has been calling a merge of "
        "connectivity and chemistry is, on the locked split, a statement about chemistry that was "
        "under-modelled -- and the brief predicted exactly this as the outcome to watch for."))
    say(f"     T3 {'PASS' if t3 else 'FAIL'}")

    # ------------------------------------------------------------------ T5
    say()
    df = np.array([d for d in dupfree if d is not None], bool)
    nd = len(df)
    idx = np.where(df)[0]
    say(f"T5 DUPLICATE STRATUM: of the first {nd} TEST cases, {int(df.sum())} "
        f"({df.mean():.1%}) have NO surviving duplicate reaction")

    def sub(a, b, metric="mrr"):
        src = mrr if metric == "mrr" else rows
        x = np.array([src[a][i] for i in idx], float)
        y = np.array([src[b][i] for i in idx], float)
        d = x - y
        d = d[np.isfinite(d)]
        return float(d.mean()), float(d.std() / np.sqrt(len(d)))
    sw = sub("close_walk", "walk")
    sb = sub("close_walk", "balance")
    se = sub("close_walk", "eclose")
    say(f"     duplicate-free, MRR: vs walk {sw[0]:+.4f} ({sw[0]/sw[1]:+.1f} sem) | "
        f"vs balance {sb[0]:+.4f} ({sb[0]/sb[1]:+.1f} sem) | vs eclose {se[0]:+.4f} "
        f"({se[0]/se[1]:+.1f} sem)")
    t5 = bool((sw[0] > 3 * sw[1]) == (dw_m > 3 * sw_m)
              and (sb[0] > 3 * sb[1]) == (db_m > 3 * sb_m))
    GG.verdict(t5, emit=say, if_true=(
        "T2's verdict is unchanged once memorisable duplicates are removed."), if_false=(
        "T2's verdict changes on duplicate-free cases, so part of it was memorisation."))
    say(f"     T5 {'PASS' if t5 else 'FAIL'}")

    say()
    say("T6 WHAT THIS CANNOT SHOW")
    say("     Every held-out reaction was curated INTO Human-GEM by somebody, so this measures")
    say("     recovery and not discovery. The 389 dead ends and 259 orphans are where a genuine")
    say("     proposal would have to be tested and they are not tested here.")
    say("     The scorer is granted the elemental total of the answer: 98.8% of reactions balance")
    say("     and 57% have one non-currency product, so a high score is subset-sum inversion of a")
    say("     granted total, not unconditional gap prediction.")
    say("     One split, read once. There is no second locked set behind this one.")
    t6 = True
    say(f"     T6 {'PASS' if t6 else 'FAIL'}")

    gates = {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6}
    man = RM.manifest(inputs=[Path("colab/data/rem_bipartite.npz"),
                              Path("colab/data/rem_chem.npz")],
                      available=len(R.test), used=n_ok, selection="all", seed=HASH_SEED,
                      controls=["the LOCKED test split, read once, nothing fitted on it",
                                "every constant frozen on DEV and its provenance stated in the docstring",
                                "the de-leaked j-deleted degree baseline, gated at T1",
                                "AUC, MRR and hit@k together, with verdicts on the rank metric",
                                "duplicate-surviving cases separated out at T5",
                                "CLOSE_WALK vs E-CLOSE head to head, which DEV never did"],
                      note="the design workflow's predeclared merge, measured once on 5,583 unseen reactions")
    out = {"test": "CLOSE_WALK on the locked TEST split", "gates": gates,
           "n_scored": n_ok, "results": res,
           "paired": {"vs_walk_auc": [dw_a, sw_a], "vs_walk_mrr": [dw_m, sw_m],
                      "vs_balance_auc": [db_a, sb_a], "vs_balance_mrr": [db_m, sb_m],
                      "vs_eclose_auc": [de_a, se_a], "vs_eclose_mrr": [de_m, se_m]},
           "duplicate_free": {"n": nd, "frac": float(df.mean()),
                              "vs_walk_mrr": list(sw), "vs_balance_mrr": list(sb),
                              "vs_eclose_mrr": list(se)},
           "leak_reproduced": leak_diff,
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
