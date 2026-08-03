"""TARGET DECODER + COMPOSITIONAL COLD START: can we predict how an UNSEEN gene responds?

WHAT IS NEW.  Every result so far held the target side fixed: the responding genes were always ones the model had
seen respond thousands of times.  Only the SOURCE (the knocked-out gene) was ever held out.  This holds out the
TARGETS, and then both at once.

    B_j  is gene j's programme-susceptibility vector: how gene j responds to each of the K response programmes.
    For genes in training it is read off the factorisation.  For an unseen gene it has to be PREDICTED from
    annotation:  x_j -> B_hat_j.

THREE REGIMES, and the third is the real test.
    unseen source            done already -- that is chan2a, 0.2885.
    unseen TARGET            A_i known, B_hat_j predicted from the target's annotation.
    unseen source AND target Y_hat_ij = A_hat_i^T B_hat_j.  Neither end was ever observed.  This is
                             compositional generalisation and nothing in this project has tested it.

SPLITS.  Genes are held out in blocks, by two rules: random, and grouped by protein COMPLEX so that no complex
member of a held-out gene remains in training.  A model that survives random and collapses on complex was
recognising the complex, not the biology.

ARMS on the target side, all scored on the identical held-out gene sets.
    oracle_B    the true B_j.  An ORACLE and a denominator, never a competitor -- it is the ceiling for the
                target decoder, since it is what a perfect x_j -> B_j map would produce.
    ridge_B     B_hat_j from the 696 named channels by ridge.  THE TEST.
    knn_B       B_j of the 10 nearest genes in annotation space.  A lookup, and it tied the ridge on the source
                side, so it is the baseline to beat.
    marginal_B  the mean B over training genes -- no gene-specific information at all.  The floor.
    shuffled_B  annotation permuted across genes.  Must lose.

PREDECLARED.  The target decoder passes only if ridge_B or knn_B beats marginal_B past a bootstrap CI, on BOTH
sealed cohorts, under BOTH split rules.  Compositional cold start passes only if the source+target arm beats the
same floor.  If the decoder ties the marginal, then programme susceptibility is not predictable from annotation
and the model can only ever speak about genes it has already watched respond.
"""
from __future__ import annotations
import collections, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT, SP = A.OUT, A.SP
K = 64
FOLDS = 5
BOOT = 10_000


def main() -> int:
    log: list[str] = []
    def report(t):
        print(t, flush=True); log.append(t)
    report("=" * 100)
    report("TARGET DECODER AND COMPOSITIONAL COLD START")
    report("=" * 100)
    report("  PREDECLARED: passes only if ridge_B or knn_B beats marginal_B on BOTH cohorts, BOTH splits.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]; genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}; gi = {g: i for i, g in enumerate(genes)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05
    keep_g = np.where(~tide)[0]
    gname = [genes[c] for c in keep_g]

    sealed, answers = set(), {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists(): sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists(): answers[tag] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]

    # channels over the union of knockouts and TARGET genes -- the target decoder needs x_j for every gene
    universe = sorted(set(kos) | sealed | set(gname))
    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}
    report(f"  channels {chan.shape[1]} over {len(universe):,} entities "
           f"({sum(1 for g in gname if g in urow):,} of {len(gname):,} target genes covered)")

    X = Amat[[ki[k] for k in train]][:, keep_g]
    T_j = X.mean(0, keepdims=True); R = X - T_j
    R = R - R.mean(1, keepdims=True)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=K, random_state=0).fit(R)
    B = svd.components_.astype(np.float32)              # K x genes
    A_src = (R @ B.T).astype(np.float32)                # sources x K
    report(f"  programme basis K={K} on {R.shape[0]:,} training sources x {R.shape[1]:,} non-tide genes")

    # source encoder (ridge on channels), so the source side is cold too
    src_rows = np.array([urow[k] for k in train])
    w_src = A.ridge_fit(chan[src_rows], A_src, A.PENALTY)

    # gene groupings for the two split rules
    D = json.load(open(OUT / "cell_complete.json"))
    nm = [g["name"] for g in D["genes"]]
    cplx = {}
    for k_, cs in (D.get("gene2cplx") or {}).items():
        try: g = nm[int(k_)]
        except (TypeError, ValueError, IndexError): continue
        lst = sorted(str(c) for c in (cs if isinstance(cs, list) else [cs]))
        if lst: cplx[g] = lst[0]
    del D
    grp = {"random": np.array([f"__{g}" for g in gname]),
           "complex": np.array([cplx.get(g, f"__{g}") for g in gname])}

    have = np.array([g in urow for g in gname])
    Xg = np.zeros((len(gname), chan.shape[1]), np.float32)
    Xg[have] = chan[[urow[g] for g in gname if g in urow]]

    from sklearn.neighbors import KNeighborsRegressor
    summary = {}
    for split in ("random", "complex"):
        labels = grp[split]; uniq = sorted(set(labels.tolist()))
        rr = np.random.default_rng(A.SEED + 7)
        assign = {uniq[int(u)]: i % FOLDS for i, u in enumerate(rr.permutation(len(uniq)))}
        fold_of = np.array([assign[l] for l in labels])
        Bhat = {a: np.zeros_like(B) for a in ("ridge_B", "knn_B", "marginal_B", "shuffled_B")}
        pmt = Xg[np.random.default_rng(A.SEED + 3).permutation(len(Xg))]
        for f in range(FOLDS):
            te = np.where(fold_of == f)[0]; tr = np.where(fold_of != f)[0]
            tr = tr[have[tr]]
            if len(tr) < 100 or not len(te): continue
            wB = A.ridge_fit(Xg[tr], B[:, tr].T, A.PENALTY)
            Bhat["ridge_B"][:, te] = A.ridge_apply(Xg[te], wB).T
            kn = KNeighborsRegressor(n_neighbors=10).fit(Xg[tr], B[:, tr].T)
            Bhat["knn_B"][:, te] = kn.predict(Xg[te]).T
            Bhat["marginal_B"][:, te] = B[:, tr].mean(1, keepdims=True)
            wS = A.ridge_fit(pmt[tr], B[:, tr].T, A.PENALTY)
            Bhat["shuffled_B"][:, te] = A.ridge_apply(pmt[te], wS).T
        Bhat["oracle_B"] = B

        report(f"\n### {split} gene split ({FOLDS} folds, {len(set(fold_of))} used)")
        for tag, label in (("", "cohort 1"), ("_holdout", "cohort 2 (holdout)")):
            ans = answers.get(tag)
            if not ans: continue
            cohort = sorted(ans)
            A_hat = A.ridge_apply(chan[[urow[k] for k in cohort]], w_src)
            A_true = {}
            for k in cohort:
                row = Amat[ki[k]][keep_g] - T_j.ravel()
                A_true[k] = (row - row.mean()) @ B.T
            res = {a: {"seen_src": [], "cold_both": []} for a in Bhat}
            for f in range(FOLDS):
                te = np.where(fold_of == f)[0]
                if not len(te): continue
                for j, k in enumerate(cohort):
                    truth = set(ans[k]["movers"])
                    tset = [gname[t] for t in te]
                    tt = truth & set(tset)
                    if len(tt) < 3: continue
                    for arm, Bm in Bhat.items():
                        for mode, av in (("seen_src", A_true[k]), ("cold_both", A_hat[j])):
                            s = av @ Bm[:, te]
                            top = np.argsort(-s)[:A.NPRED]
                            res[arm][mode].append(len({tset[t] for t in top} & tt) / A.NPRED)
            report(f"  {label}: {'arm':<12} {'unseen TARGET':>14} {'unseen BOTH':>13}")
            vals = {}
            for arm in ("oracle_B", "ridge_B", "knn_B", "marginal_B", "shuffled_B"):
                a1 = np.array(res[arm]["seen_src"]); a2 = np.array(res[arm]["cold_both"])
                vals[arm] = (a1, a2)
                report(f"  {'':<10} {arm:<12} {a1.mean():>14.4f} {a2.mean():>13.4f}")
            report(f"  {'':<10} {'contrast':<28} {'gap':>8} {'95% CI':>22}  verdict")
            out = {}
            for nm_, a_, b_, mi in (("ridge_B - marginal (target)", "ridge_B", "marginal_B", 0),
                                    ("knn_B - marginal (target)", "knn_B", "marginal_B", 0),
                                    ("ridge_B - shuffled", "ridge_B", "shuffled_B", 0),
                                    ("COMPOSITIONAL ridge_B - marg", "ridge_B", "marginal_B", 1)):
                d = vals[a_][mi] - vals[b_][mi]
                if not len(d): continue
                br = np.random.default_rng(0)
                bs = d[br.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
                lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                report(f"  {'':<10} {nm_:<28} {d.mean():>+8.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>22}  "
                       f"{'RESOLVED' if (lo>0 or hi<0) else 'within noise'}")
                out[nm_] = {"gap": float(d.mean()), "ci": [lo, hi], "resolved": bool(lo>0 or hi<0)}
            summary[f"{split}|{tag or 'cohort1'}"] = {
                "means": {a: [float(v[0].mean()), float(v[1].mean())] for a, v in vals.items()},
                "contrasts": out, "n": int(len(vals["ridge_B"][0]))}
    json.dump({"test": "adrn_target_decoder", "K": K, "folds": FOLDS,
               "summary": summary, "log": log},
              open(OUT / "adrn_target_decoder.json", "w"), indent=2)
    report(f"\n  -> {OUT/'adrn_target_decoder.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
