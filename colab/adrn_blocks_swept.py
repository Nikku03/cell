"""The block ablation, re-run through the ROBUSTNESS HARNESS: every gate swept over its representation rank.

This supersedes the gate section of adrn_source_blocks.py.  That script reported chan2a+ppi > chan2a at +0.0140 /
+0.0152 from a single configuration, and the rank sweep in adrn_ppi_replicate then showed the effect lived in one
of six cells -- the one that had been run first.  `SVD_K = 128` was an undeliberated default carrying a finding.

Here the sweep is structural: `robustness.Sweeper` refuses an axis with fewer than two values and raises on a
verdict from a single cell, so the same mistake cannot be made by forgetting.  Each block is swept over the one
knob that was previously fixed by accident:

    ppi     truncated-SVD rank of the adjacency          {64, 128, 256}
    codep   representation of the DepMap profile         {svd64, svd256, raw 1100-d}

and each is judged by the ladder in robustness.py -- ROBUST / DIRECTIONAL / FRAGILE / NULL / HARMFUL -- rather
than by whether some cell somewhere cleared a CI.

WHAT THIS RE-TESTS.  codep passed its original gate at the raw representation only.  That is exactly the position
PPI was in before the sweep, so codep gets the same treatment before it is trusted.  If codep is ROBUST it has
earned the claim made for it; if it is FRAGILE it has not, and the honest summary of the day changes again.

KNOWN REMAINING UNSWEPT DEFAULT, recorded rather than quietly left: the NMF programme count `A.K = 60`. It sits
upstream of every arm, so sweeping it multiplies the whole run by the number of values and is not done here. It
is the next candidate, and until it is swept every number below is conditional on K=60.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from adrn_source_blocks import rewire, load_esm, load_codep
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
PPI_RANKS = (64, 128, 256)
CODEP_REPS = ("svd64", "svd256", "raw")
PENALTIES = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
VAL_FRAC = 0.2


def main():
    log = []

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("BLOCK ABLATION THROUGH THE ROBUSTNESS HARNESS -- every gate swept over its representation rank")
    report("=" * 100)

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05

    sealed, answers = set(), {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists():
            answers[tag] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]
    universe = sorted(set(kos) | sealed)
    urow = {g: i for i, g in enumerate(universe)}
    tr = np.array([urow[k] for k in train])

    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    P, pmiss = load_codep(universe, report)

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] if isinstance(g, dict) else str(g) for g in D["genes"]]
    n = len(names)
    edges = np.array([(a, b) for a, b in D["ppi"] if a < n and b < n], np.int64)
    nidx = {g: i for i, g in enumerate(names)}
    rows_u = np.array([nidx.get(g, -1) for g in universe])
    report(f"  ppi {len(edges):,} edges; codep {P.shape}")

    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD, NMF

    def graph_embed(e, k, seed=None):
        ee = rewire(e, n, seed) if seed is not None else e
        M = sparse.coo_matrix((np.ones(len(ee) * 2), (np.r_[ee[:, 0], ee[:, 1]], np.r_[ee[:, 1], ee[:, 0]])),
                              shape=(n, n)).tocsr()
        Z = TruncatedSVD(n_components=k, random_state=0).fit_transform(M)
        out = np.zeros((len(universe), k), np.float32)
        ok = rows_u >= 0
        out[ok] = Z[rows_u[ok]]
        return out

    X = Amat[[ki[k] for k in train]].copy()
    X[:, tide] = 0.0
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)

    def prep(M, miss=None):
        M = M.astype(np.float32).copy()
        if miss is not None and miss.any():
            M[miss] = M[tr[~miss[tr]]].mean(0)
        mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-6
        return (M - mu) / sd

    cz = prep(chan)

    def score(src, w, keys):
        s = A.ridge_apply(src[[urow[k] for k in keys]], w) @ H
        s[:, tide] = -np.inf
        return s

    def prec(s, keys, truth):
        out = []
        for j, k in enumerate(keys):
            sc = s[j].copy()
            if k in genes:
                sc[genes.index(k)] = -np.inf
            out.append(len({genes[t] for t in np.argsort(-sc)[:A.NPRED]} & truth[k]) / float(A.NPRED))
        return np.array(out)

    nval = int(len(train) * VAL_FRAC)
    vp = np.random.default_rng(A.SEED + 1).permutation(len(train))
    val, fit = [train[i] for i in vp[:nval]], [train[i] for i in vp[nval:]]
    tidx = {k: i for i, k in enumerate(train)}
    Wfit = W[[tidx[k] for k in fit]]
    vtruth = {k: {genes[i] for i in np.where(Amat[ki[k]] >= A.TAU)[0] if not tide[i]} for k in val}
    frows = np.array([urow[k] for k in fit])

    def fit_arm(src):
        cand = sorted(((prec(score(src, A.ridge_fit(src[frows], Wfit, p), val), val, vtruth).mean(), p)
                       for p in PENALTIES), reverse=True)
        return A.ridge_fit(src[tr], W, cand[0][1])

    cohorts = {}
    for tag in ("", "_holdout"):
        if answers.get(tag):
            nm = tag or "cohort1"
            ks = sorted(answers[tag])
            cohorts[nm] = (ks, {k: set(answers[tag][k]["movers"]) for k in ks})

    wc = fit_arm(chan)
    baseline = {nm: prec(score(chan, wc, ks), ks, truth) for nm, (ks, truth) in cohorts.items()}
    for nm, v in baseline.items():
        report(f"  chan2a baseline {nm}: {v.mean():.4f}")

    sw_ppi_ctrl = Sweeper("chan2a+ppi - chan2a+ppi_rw", axes={"rank": list(PPI_RANKS)})
    sw_ppi_base = Sweeper("chan2a+ppi - chan2a", axes={"rank": list(PPI_RANKS)})
    sw_codep = Sweeper("chan2a+codep - chan2a", axes={"rep": list(CODEP_REPS)})

    for cfg in sw_ppi_ctrl.configs():
        k = cfg["rank"]
        real = np.concatenate([cz, prep(graph_embed(edges, k))], 1)
        rw = np.concatenate([cz, prep(graph_embed(edges, k, seed=A.SEED))], 1)
        wr, ww = fit_arm(real), fit_arm(rw)
        for nm, (ks, truth) in cohorts.items():
            pr = prec(score(real, wr, ks), ks, truth)
            pw = prec(score(rw, ww, ks), ks, truth)
            sw_ppi_ctrl.add(cfg, nm, pr - pw)
            sw_ppi_base.add(cfg, nm, pr - baseline[nm])
        report(f"  ppi rank {k:>3} done")

    for cfg in sw_codep.configs():
        rep = cfg["rep"]
        if rep == "raw":
            Pk = P
        else:
            Pk = TruncatedSVD(n_components=int(rep[3:]), random_state=0).fit_transform(
                np.nan_to_num(P.astype(np.float32)))
        src = np.concatenate([cz, prep(Pk, pmiss if rep == "raw" else None)], 1)
        w = fit_arm(src)
        for nm, (ks, truth) in cohorts.items():
            sw_codep.add(cfg, nm, prec(score(src, w, ks), ks, truth) - baseline[nm])
        report(f"  codep {rep} done")

    out = {}
    for sw in (sw_codep, sw_ppi_ctrl, sw_ppi_base):
        report(sw.report())
        out[sw.name] = sw.verdict()

    report("\n  SUMMARY")
    for name, v in out.items():
        report(f"    {name:<34} {v['verdict']:<12} "
               f"(+{v['n_resolved_pos']}/-{v['n_resolved_neg']} of {v['n_cells']} cells)")
    report("\n  Conditional on the NMF programme count K=60, which remains unswept and sits upstream of every arm.")

    json.dump({"test": "adrn_blocks_swept", "ppi_ranks": list(PPI_RANKS), "codep_reps": list(CODEP_REPS),
               "baseline": {k: float(v.mean()) for k, v in baseline.items()},
               "sweeps": out, "log": log},
              open(OUT / "adrn_blocks_swept.json", "w"), indent=2)
    report(f"\n  -> {OUT / 'adrn_blocks_swept.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
