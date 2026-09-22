"""IS THIS A LOOKUP WITH EXTRA STEPS?  And if so, what is the binding constraint?

THE CHALLENGE THIS ANSWERS.  "It is nothing novel -- you can get the same information with a few clicks." The
synthesis in UPDATES.md already says every success here is retrieval and every failure is inference, but that was
an argument from a pattern of results, not a measurement. This measures it.

THE TENSION IT ALSO RESOLVES.  I have claimed both of these, and they cannot both be true:

    "we are data-limited"    but the scaling curve went FLAT at 4,720 perturbations (-0.0009 per doubling), so
                             more examples of the same kind buy nothing
    "we are model-limited"   but a 42k-parameter closed-form ridge ties or beats a 0.63M transformer
                             (0.2016 vs 0.2167 before; today 0.2567 ridge vs 0.2420 flat vs 0.2304 multiplex)

If neither more data nor more capacity moves it, the constraint is somewhere else, and there are two candidates
this test can tell apart.

THE ARMS.

    frequency        the movement-frequency prior. The floor.
    nn1 / nn5 / nn10 COPY the measured response of the 1, 5 or 10 nearest TRAINING genes in annotation space.
                     This is literally the "few clicks" model: find the most similar gene someone already
                     measured, and report what happened to it. No fitting, no parameters.
    ridge            the incumbent ADRN.
    nn_perm          nearest neighbours chosen by a PERMUTED similarity -- the control for "any neighbour will do".

    nn1_ORACLE       copy the response of the training gene whose TRUE RESPONSE is most similar. This reads the
    nn5_ORACLE       answer to choose the neighbour, so it is an oracle and a denominator, never a competitor.
                     It is the upper bound on what ANY retrieval system could achieve from this training set.

WHAT THE COMPARISONS MEAN -- this is the whole point of the design.

    ridge ~ nn_annot                    the model IS a lookup. The fitting adds nothing a nearest-neighbour
                                        query would not give you, and "a few clicks" is literally correct.
    ridge >> nn_annot                   the model interpolates beyond retrieval; it is doing something a lookup
                                        cannot, whatever else is wrong with it.
    nn_ORACLE >> nn_annot               FEATURE-LIMITED. The training set CONTAINS a gene with the right answer,
                                        and the annotation features cannot find it. More data will not help;
                                        better gene descriptors would.
    nn_ORACLE ~ nn_annot ~ ridge        DATA-LIMITED in the way that matters: the training set does not contain
                                        a better answer for these genes at all, so no retrieval and no
                                        representation over this set can do better. Only new biology helps.

PREDECLARED before running: judged on robustness.Sweeper over prediction depth x similarity metric, on the two
sealed 200-knockout cohorts. Both contrasts (ridge - best_nn, oracle - best_nn) carried separately.

Note the sign convention: a LOW ridge - nn gap is the uncomfortable result here, not a good one.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
NPREDS = (10, 20, 50)
METRICS = ("cosine", "jaccard")
PENALTIES = (1.0, 10.0, 100.0, 1000.0)
KNN = (1, 5, 10)


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("IS THIS A LOOKUP WITH EXTRA STEPS?  ridge vs nearest-measured-gene, and the retrieval oracle")
    report("=" * 100)
    report("  PREDECLARED: ridge ~ nn  => it is a lookup. oracle >> nn => feature-limited.")
    report("               oracle ~ nn ~ ridge => the training set has no better answer; only new biology helps.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    gj = {g: j for j, g in enumerate(genes)}
    Aabs = np.abs(z["M"]).astype(np.float32)
    tide = (Aabs >= A.TAU).mean(0) >= 0.05

    sealed, answers = set(), {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists():
            answers[tag or "cohort1"] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]
    train_rows = np.array([ki[k] for k in train])
    universe = sorted(set(kos) | sealed)
    urow = {g: i for i, g in enumerate(universe)}
    tr_u = np.array([urow[k] for k in train])
    report(f"\n  {len(kos):,} perturbations; sealed {len(sealed)}; training pool {len(train):,}")

    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1).astype(np.float32)

    # response profiles, tide-masked, used as the retrieval TARGET and by the oracle
    R = Aabs.copy()
    R[:, tide] = 0.0
    Rtr = R[train_rows]

    def sim_matrix(metric, Q, B):
        if metric == "cosine":
            q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
            b = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
            return q @ b.T
        inter = (Q > 0).astype(np.float32) @ (B > 0).astype(np.float32).T
        a = (Q > 0).sum(1, keepdims=True)
        c = (B > 0).sum(1, keepdims=True).T
        return inter / (a + c - inter + 1e-9)

    freq = (Aabs[train_rows] >= A.TAU).mean(0)
    freq[tide] = -np.inf

    def topset(v, n):
        return {genes[t] for t in np.argsort(-v)[:n]}

    def hit(v, movers, n, k):
        v = v.copy()
        v[tide] = -np.inf
        if k in gj:
            v[gj[k]] = -np.inf
        return len(topset(v, n) & set(movers)) / float(n)

    # ---- ridge, penalty tuned on a TRAIN-ONLY split ----
    from sklearn.decomposition import NMF
    X = Aabs[train_rows].copy()
    X[:, tide] = 0.0
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=300, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)
    del X
    rv = np.random.default_rng(A.SEED + 3).permutation(len(tr_u))
    nv = max(50, len(rv) // 5)
    best_v, best_p = -1e18, PENALTIES[0]
    for pen in PENALTIES:
        w = A.ridge_fit(chan[tr_u[rv[nv:]]], W[rv[nv:]], pen)
        e = -float(np.mean((A.ridge_apply(chan[tr_u[rv[:nv]]], w) - W[rv[:nv]]) ** 2))
        if e > best_v:
            best_v, best_p = e, pen
    wr = A.ridge_fit(chan[tr_u], W, best_p)
    report(f"  ridge fitted, penalty {best_p} (chosen on a train-only split)")

    sw_lookup = Sweeper("ridge - best nearest-neighbour", axes={"npred": list(NPREDS), "metric": list(METRICS)})
    sw_oracle = Sweeper("retrieval ORACLE - best nearest-neighbour",
                        axes={"npred": list(NPREDS), "metric": list(METRICS)})
    cells = {}
    for metric in METRICS:
        for npred in NPREDS:
            arms = {}
            for tag, ans in answers.items():
                keys = [k for k in sorted(ans) if k in urow]
                Q = chan[[urow[k] for k in keys]]
                S = sim_matrix(metric, Q, chan[tr_u])
                rngp = np.random.default_rng(A.SEED + 9)
                Sp = S[:, rngp.permutation(S.shape[1])]
                # ORACLE similarity: how close is the TRUE response, which is the answer we are predicting
                Qr = np.array([R[ki[k]] for k in keys], np.float32)
                So = sim_matrix("cosine", Qr, Rtr)
                pr = A.ridge_apply(chan[[urow[k] for k in keys]], wr)
                for i, k in enumerate(keys):
                    mv = ans[k]["movers"]
                    arms.setdefault("frequency", []).append(hit(freq.copy(), mv, npred, k))
                    arms.setdefault("ridge", []).append(hit(pr[i] @ H, mv, npred, k))
                    for kk in KNN:
                        idx = np.argsort(-S[i])[:kk]
                        arms.setdefault(f"nn{kk}", []).append(hit(Rtr[idx].mean(0), mv, npred, k))
                        oidx = np.argsort(-So[i])[:kk]
                        arms.setdefault(f"nn{kk}_ORACLE", []).append(hit(Rtr[oidx].mean(0), mv, npred, k))
                    pidx = np.argsort(-Sp[i])[:5]
                    arms.setdefault("nn_perm", []).append(hit(Rtr[pidx].mean(0), mv, npred, k))
            arms = {n: np.array(v) for n, v in arms.items()}
            bn = max(KNN, key=lambda kk: arms[f"nn{kk}"].mean())
            bo = max(KNN, key=lambda kk: arms[f"nn{kk}_ORACLE"].mean())
            cfg, tg = {"npred": npred, "metric": metric}, f"n{npred}|{metric}"
            sw_lookup.add(cfg, tg, arms["ridge"] - arms[f"nn{bn}"])
            sw_oracle.add(cfg, tg, arms[f"nn{bo}_ORACLE"] - arms[f"nn{bn}"])
            cells[tg] = {"best_nn": f"nn{bn}", "best_oracle": f"nn{bo}",
                         **{n: float(v.mean()) for n, v in arms.items()}}
            report(f"  {metric:>8} N={npred:>2}: freq {arms['frequency'].mean():.4f} | "
                   f"nn1 {arms['nn1'].mean():.4f} | nn5 {arms['nn5'].mean():.4f} | "
                   f"nn10 {arms['nn10'].mean():.4f} | nn_perm {arms['nn_perm'].mean():.4f} | "
                   f"ridge {arms['ridge'].mean():.4f} | ORACLE {arms[f'nn{bo}_ORACLE'].mean():.4f}")

    for nm, s in (("lookup", sw_lookup), ("oracle", sw_oracle)):
        report(s.report())
        if s.inert_axes() or s.duplicate_cells():
            report(f"  GUARD {nm}: inert axes {s.inert_axes()}, duplicate cells {s.duplicate_cells()}")

    gl = float(np.mean([c["ridge"] - c[c["best_nn"]] for c in cells.values()]))
    go = float(np.mean([c[c["best_oracle"] + "_ORACLE"] - c[c["best_nn"]] for c in cells.values()]))
    mr = float(np.mean([c["ridge"] for c in cells.values()]))
    mn = float(np.mean([c[c["best_nn"]] for c in cells.values()]))
    mo = float(np.mean([c[c["best_oracle"] + "_ORACLE"] for c in cells.values()]))
    mf = float(np.mean([c["frequency"] for c in cells.values()]))
    report(f"\n  MEANS  frequency {mf:.4f} | best nn {mn:.4f} | ridge {mr:.4f} | retrieval ORACLE {mo:.4f}")
    report(f"         ridge - nn {gl:+.4f}   |   oracle - nn {go:+.4f}")

    report("\n  READING")
    lookup_pos = sum(1 for c in sw_lookup._cells.values() if c["lo"] > 0)
    oracle_pos = sum(1 for c in sw_oracle._cells.values() if c["lo"] > 0)
    if lookup_pos <= len(sw_lookup._cells) / 2:
        report("  The ridge does NOT beat copying the nearest measured gene. The fitting, the channels and the")
        report("  basis add nothing a nearest-neighbour query would not give you. It is a lookup with extra")
        report("  steps, and the 'few clicks' criticism is literally correct.")
    else:
        report(f"  The ridge beats nearest-neighbour retrieval by {gl:+.4f}, resolved in {lookup_pos} of")
        report(f"  {len(sw_lookup._cells)} cells. It is doing something a lookup cannot -- interpolating across")
        report("  measured genes rather than copying one.")
    if oracle_pos > len(sw_oracle._cells) / 2:
        report(f"  The retrieval ORACLE beats annotation retrieval by {go:+.4f}. The training set CONTAINS a gene")
        report("  with a much better answer and the annotation cannot find it: FEATURE-LIMITED. More screening")
        report("  of the same kind will not help; better gene descriptors would.")
    else:
        report(f"  The retrieval oracle is only {go:+.4f} above annotation retrieval. Even choosing the best")
        report("  neighbour WITH THE ANSWER IN HAND barely helps -- the training set does not contain a better")
        report("  answer for these genes. Neither features nor architecture is the constraint. Only new biology")
        report("  is, which is exactly the thing that cannot be got from data already on disk.")

    json.dump({"test": "adrn_is_it_lookup", "npreds": list(NPREDS), "metrics": list(METRICS), "knn": list(KNN),
               "cells": cells, "means": {"frequency": mf, "best_nn": mn, "ridge": mr, "oracle": mo},
               "gap_ridge_minus_nn": gl, "gap_oracle_minus_nn": go,
               "sweeps": {"lookup": sw_lookup.verdict(), "oracle": sw_oracle.verdict()}, "log": log},
              open(OUT / "adrn_is_it_lookup.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'adrn_is_it_lookup.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
