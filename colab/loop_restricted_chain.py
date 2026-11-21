"""Loop 166. Run the Markov chain INSIDE the chemistry shortlist, not across the whole graph.

WHAT LOOP 162 DOES, AND THE GAP IN IT. Its merge is one shot:

    SCORE = T + 0.9 * max(zw, rb) + 1e-6 * rb

where zw is a restart walk over the ENTIRE bipartite graph, rank-normalised and then averaged into
the chemistry tier. The walk therefore spends its probability mass before the chemistry has narrowed
anything: H+ touches 2,722 reactions and H2O 2,020, so a walk seeded anywhere flows into them, which
is exactly why the walk alone scores AUC 0.7008 on TEST while the chemistry closure scores 0.9731.

THE PROPOSAL. Let the chemistry pick a shortlist first, then run the chain restricted to it, and use
the chain only to ORDER what chemistry already believes. This is not the same as masking the
full-graph walk to the shortlist: a restricted chain renormalises, so probability that would have
drained into a hub outside the shortlist is redistributed among the candidates instead of lost. U3
separates those two, because if masking is as good as restricting then the shortlist is doing all
the work and the chain is not.

THERE IS PRECEDENT FOR EXPECTING THIS TO WORK. Loop 160 measured the same walk at AUC 0.6928 over
all 8,428 candidates and 0.8158 restricted to a 2-step neighbourhood of 68. Narrowing the candidate
set was worth +0.12 to the walk. The chemistry closure's top-20 contains the answer 88.5% of the
time on TEST, which is a far better shortlist than a 2-step neighbourhood, whose recall loop 161
measured at 52.6%.

ON THE SPLIT. TEST has been read once, by loop 162. This loop runs on DEV. A second TEST read is
worth spending only if DEV shows a clear win, and it would be the second read of a set whose value
comes from being read once -- U6 says so rather than treating the split as reusable.

PREDECLARED, before any number is looked at.

  U1 THE SHORTLIST IS WORTH BUILDING ON. Recall of the chemistry tier's top-N for N in
     {5, 10, 20, 50, 100}: how often the true answer is inside it at all.
     Gate: PASS iff top-20 recall on DEV exceeds 0.80. Below that, re-ranking a shortlist cannot
     beat scoring the whole space and the rest of the loop is void.

  U2 DOES THE RESTRICTED CHAIN BEAT LOOP 162's ONE-SHOT BLEND? Same cases, same chemistry tier,
     the chain run inside the top-N instead of blended from the whole graph.
     Gate: PASS iff hit@1 improves by more than 3 sem. hit@1 is the target the proposal names, and
     MRR and hit@20 are reported beside it so a gain at the top that costs the tail is visible.

  U3 IS IT THE CHAIN OR JUST THE SHORTLIST? Three scorers on identical shortlists: the restricted
     chain, the full-graph walk MASKED to the shortlist, and the chemistry tier alone.
     Gate: PASS iff the restricted chain beats the masked walk by more than 3 sem. A FAIL means
     restriction adds nothing over filtering and the honest description is a chemistry shortlist
     with a cosmetic re-rank.

  U4 WHERE IS N, AND IS IT STABLE? Sweep N over {5, 10, 20, 50, 100, 500} and report the curve.
     Gate: passes on the whole curve being reported. A method that works at exactly one N is a
     tuned number and the curve makes that visible.

  U5 WHAT DOES IT COST AT THE TAIL? hit@20, hit@100 and MRR for every variant.
     Gate: PASS iff the best variant does not lose more than 3 sem on hit@20 relative to loop 162's
     blend. Trading the tail for the top is a real trade and it has to be priced.

  U6 WHAT THIS CANNOT SHOW. DEV only. A second TEST read would be the second read of a set whose
     value is that it was read once, and this loop does not take it.

-> outputs/loop_restricted_chain.json
"""
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse, stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                    # noqa: E402
import run_manifest as RM                  # noqa: E402
from rem.harness import REM, auc_of        # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_restricted_chain.json"
CMAX, BEAM, HASH_SEED = 6, 8, 90210
BLOCK_SCALE, ALPHA, NITER = 0.9, 0.15, 60
NS_GRID = (5, 10, 20, 50, 100, 500)
N_DEV, SAMPLE_SEED = 1500, 16600
U1_RECALL = 0.80

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 104)
    say("  THE CHAIN INSIDE THE SHORTLIST -- cascade, not blend. DEV only.")
    say("=" * 104)
    say()

    R = REM()
    NC = len(R.noncur)
    Ei = np.rint(R.Enc).astype(np.int64)
    nz = (Ei != 0).any(1)
    rng = np.random.default_rng(HASH_SEED)
    h1 = rng.integers(-(2 ** 62), 2 ** 62, Ei.shape[1], dtype=np.int64)
    h2 = rng.integers(-(2 ** 62), 2 ** 62, Ei.shape[1], dtype=np.int64)

    def keys(V, h):
        with np.errstate(over="ignore"):
            return (V.astype(np.int64) * h).sum(1)
    own1, own2 = Counter(), Counter()
    for c in range(1, CMAX + 1):
        V = c * Ei[nz]
        for a, b in zip(keys(V, h1), keys(V, h2)):
            own1[int(a)] += 1
            own2[int(b)] += 1

    def tier(cs):
        r0 = cs["residual"]
        r = np.rint(r0).astype(np.int64)
        if np.abs(r0 - r).max() > 1e-6 or (r < 0).any():
            return np.zeros(NC)
        d1, d2 = Counter(), Counter()
        for s in cs["seeds"]:
            k = R.ncmap[int(s)]
            if not nz[k]:
                continue
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
            for i in np.where(ok)[0]:
                a, b = int(k1[i]), int(k2[i])
                if own1[a] - d1.get(a, 0) > 0 and own2[b] - d2.get(b, 0) > 0:
                    pair[i] = True
        return 8.0 * solo + 4.0 * pair + 1.0 * feas

    # --------------------------------------------------- restricted chain
    sp_rx = R.sp_rx

    def restricted_walk(cs, cols):
        """Markov chain on the subgraph induced by seeds + shortlist + their reactions.

        Probability that would drain into a hub outside the shortlist is renormalised among the
        candidates instead of leaving the system -- which is what makes this different from masking
        a full-graph walk."""
        keepsp = set(int(R.noncur[c]) for c in cols) | set(cs["seeds"])
        rxs = set()
        for i in keepsp:
            rxs |= sp_rx[i]
        rxs.discard(cs["j"])
        if not rxs:
            return np.zeros(len(cols))
        spl = sorted(keepsp)
        rxl = sorted(rxs)
        si = {v: k for k, v in enumerate(spl)}
        ri = {v: k + len(spl) for k, v in enumerate(rxl)}
        n = len(spl) + len(rxl)
        src, dst = [], []
        for j in rxl:
            rev = R.rev[j] == 1
            for i in R.react_of[j]:
                if i in si:
                    src.append(si[i])
                    dst.append(ri[j])
                    if rev:
                        src.append(ri[j])
                        dst.append(si[i])
            for i in R.prod_of[j]:
                if i in si:
                    src.append(ri[j])
                    dst.append(si[i])
                    if rev:
                        src.append(si[i])
                        dst.append(ri[j])
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

    rngs = np.random.default_rng(SAMPLE_SEED)
    dev = [int(x) for x in rngs.choice(R.dev, size=min(N_DEV, len(R.dev)), replace=False)]
    say(f"     {len(dev):,} DEV reactions sampled at seed {SAMPLE_SEED} "
        f"(TEST untouched by this loop)")

    def r01(v):
        return (stats.rankdata(v, "average") - 1) / max(len(v) - 1, 1)

    recall = {N: [] for N in NS_GRID}
    METRICS = ["blend162", "tier_only"] + [f"chain{N}" for N in NS_GRID] \
        + [f"masked{N}" for N in NS_GRID]
    hit1 = {k: [] for k in METRICS}
    hit20 = {k: [] for k in METRICS}
    hit100 = {k: [] for k in METRICS}
    mrr = {k: [] for k in METRICS}
    n_ok = 0
    for t, j in enumerate(dev):
        cs = R.case(j)
        if cs is None:
            continue
        m = ~cs["excl"]
        pos = cs["pos"][m]
        if pos.sum() == 0 or (~pos).sum() == 0:
            continue
        T = tier(cs)
        w = R.walk(R.operator(cs["j"]), cs["seeds"])[:R.NS][R.noncur]
        zw = np.zeros(NC)
        mm = w > 0
        nn = int(mm.sum())
        if nn == 1:
            zw[mm] = 1.0
        elif nn > 1:
            zw[mm] = 0.001 + 0.999 * (stats.rankdata(w[mm], "average") - 1) / (nn - 1)
        rb = r01(R.balance_score(cs["residual"]))
        blend = T + BLOCK_SCALE * np.maximum(zw, rb) + 1e-6 * rb

        def record(name, score):
            v = score[m]
            o = np.argsort(-v, kind="stable")
            rk = np.empty(len(v), int)
            rk[o] = np.arange(1, len(v) + 1)
            b = int(rk[pos].min())
            hit1[name].append(1.0 if b == 1 else 0.0)
            hit20[name].append(1.0 if b <= 20 else 0.0)
            hit100[name].append(1.0 if b <= 100 else 0.0)
            mrr[name].append(1.0 / b)
        record("blend162", blend)
        record("tier_only", T + 1e-6 * rb)

        base = T + 1e-6 * rb
        order = np.argsort(-base, kind="stable")
        for N in NS_GRID:
            cols = order[:N]
            recall[N].append(1.0 if cs["pos"][cols].any() else 0.0)
            pr = restricted_walk(cs, cols)
            sc = base.copy()
            if pr.max() > 0:
                sc[cols] = base[cols] + BLOCK_SCALE * r01(pr)
            record(f"chain{N}", sc)
            sm = base.copy()
            wc = w[cols]
            if wc.max() > 0:
                sm[cols] = base[cols] + BLOCK_SCALE * r01(wc)
            record(f"masked{N}", sm)
        n_ok += 1
        if n_ok % 250 == 0:
            say(f"     {n_ok:,}/{len(dev):,} [{time.time()-t0:.0f}s]")

    def agg(k):
        return {"hit@1": float(np.mean(hit1[k])), "hit@20": float(np.mean(hit20[k])),
                "hit@100": float(np.mean(hit100[k])), "mrr": float(np.mean(mrr[k])),
                "sem1": float(np.std(hit1[k]) / np.sqrt(len(hit1[k])))}
    res = {k: agg(k) for k in METRICS}

    def pd(a, b, d):
        x, y = np.array(d[a], float), np.array(d[b], float)
        z = x - y
        return float(z.mean()), float(z.std() / np.sqrt(len(z)))

    # ------------------------------------------------------------------ U1
    say()
    say("U1 SHORTLIST RECALL -- is the answer even in the chemistry top-N?")
    for N in NS_GRID:
        say(f"     top-{N:<4d} recall {np.mean(recall[N]):.4f}")
    u1 = bool(np.mean(recall[20]) > U1_RECALL)
    GG.verdict(u1, emit=say, if_true=(
        "the chemistry shortlist is worth re-ranking: the answer is usually inside it."),
        if_false=("the shortlist misses the answer too often for re-ranking to beat scoring the "
                  "whole space; the rest of this loop is void."))
    say(f"     U1 {'PASS' if u1 else 'FAIL'}")

    # ------------------------------------------------------------------ U4 (curve first)
    say()
    say("U4 THE CURVE")
    say(f"     {'variant':<14s} {'hit@1':>7s} {'hit@20':>7s} {'hit@100':>8s} {'MRR':>7s}")
    for k in ["tier_only", "blend162"] + [f"masked{N}" for N in NS_GRID] \
            + [f"chain{N}" for N in NS_GRID]:
        r = res[k]
        say(f"     {k:<14s} {r['hit@1']:>7.4f} {r['hit@20']:>7.4f} {r['hit@100']:>8.4f} "
            f"{r['mrr']:>7.4f}")
    u4 = True
    say(f"     U4 {'PASS' if u4 else 'FAIL'}   (n = {n_ok:,})")

    # ------------------------------------------------------------------ U2
    best = max(NS_GRID, key=lambda N: res[f"chain{N}"]["hit@1"])
    d2, s2 = pd(f"chain{best}", "blend162", hit1)
    u2 = bool(d2 > 3 * s2)
    say()
    say(f"U2 best restricted chain (N={best}) vs loop 162's blend, hit@1: "
        f"{res[f'chain{best}']['hit@1']:.4f} vs {res['blend162']['hit@1']:.4f} = "
        f"{d2:+.4f} sem {s2:.4f} ({d2/s2:+.1f} sem)")
    dm, sm_ = pd(f"chain{best}", "blend162", mrr)
    say(f"     MRR {dm:+.4f} ({dm/sm_:+.1f} sem)")
    GG.verdict(u2, emit=say, if_true=(
        "running the chain inside the shortlist beats blending it from the whole graph."),
        if_false=("the cascade does not beat the one-shot blend at the top of the list."))
    say(f"     U2 {'PASS' if u2 else 'FAIL'}")

    # ------------------------------------------------------------------ U3
    d3, s3 = pd(f"chain{best}", f"masked{best}", hit1)
    u3 = bool(d3 > 3 * s3)
    say()
    say(f"U3 restricted chain vs the SAME shortlist re-ranked by the masked full-graph walk: "
        f"{res[f'chain{best}']['hit@1']:.4f} vs {res[f'masked{best}']['hit@1']:.4f} = "
        f"{d3:+.4f} sem {s3:.4f} ({d3/s3:+.1f} sem)")
    GG.verdict(u3, emit=say, if_true=(
        "restricting the CHAIN does work that filtering the candidates does not -- renormalising "
        "the probability that would have drained into hubs is the mechanism."), if_false=(
        "restriction adds nothing over masking. The shortlist is doing the work and the honest "
        "description is a chemistry shortlist with a cosmetic re-rank."))
    say(f"     U3 {'PASS' if u3 else 'FAIL'}")

    # ------------------------------------------------------------------ U5
    d5, s5 = pd(f"chain{best}", "blend162", hit20)
    u5 = bool(d5 > -3 * s5)
    say()
    say(f"U5 THE TAIL: hit@20 {res[f'chain{best}']['hit@20']:.4f} vs blend "
        f"{res['blend162']['hit@20']:.4f} = {d5:+.4f} ({d5/s5:+.1f} sem)")
    GG.verdict(u5, emit=say, if_true="the top is not bought at the tail's expense.",
               if_false="the gain at rank 1 is paid for by losing answers from the top 20.")
    say(f"     U5 {'PASS' if u5 else 'FAIL'}")

    say()
    say("U6 WHAT THIS CANNOT SHOW")
    say("     DEV only. TEST was read once by loop 162 and is not read here; a second read would be")
    say("     the second read of a set whose whole value is that it was read once.")
    say("     N is chosen from the curve above, on DEV, so N itself is a fitted quantity.")
    u6 = True
    say(f"     U6 {'PASS' if u6 else 'FAIL'}")

    gates = {"U1": u1, "U2": u2, "U3": u3, "U4": u4, "U5": u5, "U6": u6}
    man = RM.manifest(inputs=[Path("colab/data/rem_bipartite.npz"),
                              Path("colab/data/rem_chem.npz")],
                      available=len(R.dev), used=n_ok, selection="random", seed=SAMPLE_SEED,
                      controls=["DEV only; TEST untouched by this loop",
                                "the masked full-graph walk on identical shortlists, so U3 separates restriction from filtering",
                                "the whole N curve reported, not the best point",
                                "hit@20 priced at U5 so a top-of-list gain cannot hide a tail loss",
                                "shortlist recall gated first, so re-ranking is only attempted where it can help"],
                      note="cascade: chemistry shortlists, then a Markov chain restricted to it")
    out = {"test": "restricted chain inside the chemistry shortlist", "gates": gates,
           "n": n_ok, "recall": {N: float(np.mean(recall[N])) for N in NS_GRID},
           "results": res, "best_N": best,
           "u2": [d2, s2], "u3": [d3, s3], "u5": [d5, s5],
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
