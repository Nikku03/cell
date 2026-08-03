"""The block ablation through the ROBUSTNESS HARNESS: every gate swept over BOTH undeliberated defaults.

This supersedes the gate section of adrn_source_blocks.py.  That script reported chan2a+ppi > chan2a at +0.0140 /
+0.0152 from a single configuration; the rank sweep in adrn_ppi_replicate then showed the effect lived in one of
six cells -- the one that had been run first.  `SVD_K = 128` was an undeliberated default carrying a finding.

TWO AXES, because there were two such defaults and sweeping only the one that already burned me would repeat the
mistake at the next level up:

    rank / rep   the block's own representation size.   ppi {64,128,256} SVD components;
                 codep {svd64, svd256, raw 1150-d}.
    K            the NMF programme count, `A.K = 60`.   {30, 60, 120}.  This one sits UPSTREAM of every arm --
                 chan2a, ppi and codep are all decoded through the same H -- so every number this project has
                 reported on the sealed cohorts is conditional on K=60, and none of them has ever been checked
                 against another value.

9 configs x 2 cohorts = 18 cells per block.  `robustness.Sweeper` fixes the verdict ladder before the numbers
arrive and raises on a single-cell verdict, so the failure mode cannot recur by forgetting.

WHAT IS REALLY ON TRIAL HERE IS codep, not ppi.  ppi has already been corrected twice.  codep is the one surviving
positive claim of the day (+0.0213 / +0.0235 over chan2a; 0.2410 / 0.2818 standing alone) and it was measured at
exactly ONE representation -- raw -- which is precisely the position ppi occupied before the sweep exposed it.

COST NOTE: the rewired graph is built ONCE and embedded at each rank, rather than rewired per rank as the first
draft did.  That is both ~3x cheaper and cleaner -- every rank then sees the same degree-matched null instead of
three different draws, so rank-to-rank differences are about the rank and not about which swap sequence ran.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from adrn_source_blocks import rewire, load_codep
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
PPI_RANKS = (64, 128, 256)
CODEP_REPS = ("svd64", "svd256", "raw")
K_VALUES = (30, 60, 120)
PENALTIES = (1.0, 10.0, 100.0, 1000.0)
VAL_FRAC = 0.2


def main():
    log = []
    t_start = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("BLOCK ABLATION x ROBUSTNESS HARNESS -- swept over representation rank AND the NMF programme count K")
    report("=" * 100)
    report(f"  axes: rank/rep {PPI_RANKS} / {CODEP_REPS}   x   K {K_VALUES}   = 9 configs x 2 cohorts per block")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    gpos = {g: i for i, g in enumerate(genes)}
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

    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD, NMF

    def adj(e):
        return sparse.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                                 shape=(n, n)).tocsr()

    def take(Z, k):
        out = np.zeros((len(universe), k), np.float32)
        ok = rows_u >= 0
        out[ok] = Z[rows_u[ok]]
        return out

    report("  building the degree-matched rewiring ONCE (shared by every rank) ...")
    t0 = time.time()
    e_rw = rewire(edges, n, A.SEED)
    report(f"    rewired in {time.time() - t0:.0f}s")
    Areal, Arw = adj(edges), adj(e_rw)

    def prep(M, miss=None):
        M = M.astype(np.float32).copy()
        if miss is not None and miss.any():
            M[miss] = M[tr[~miss[tr]]].mean(0)
        mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-6
        return (M - mu) / sd

    cz = prep(chan)
    kmax = max(PPI_RANKS)
    Zr = TruncatedSVD(n_components=kmax, random_state=0).fit_transform(Areal)
    Zw = TruncatedSVD(n_components=kmax, random_state=0).fit_transform(Arw)
    PPI_SRC = {k: (np.concatenate([cz, prep(take(Zr, kmax)[:, :k])], 1),
                   np.concatenate([cz, prep(take(Zw, kmax)[:, :k])], 1)) for k in PPI_RANKS}
    Praw = np.nan_to_num(P.astype(np.float32))
    CODEP_SRC = {}
    for rep in CODEP_REPS:
        Pk = Praw if rep == "raw" else TruncatedSVD(n_components=int(rep[3:]),
                                                    random_state=0).fit_transform(Praw)
        CODEP_SRC[rep] = np.concatenate([cz, prep(Pk, pmiss if rep == "raw" else None)], 1)
    report(f"  source matrices built ({time.time() - t_start:.0f}s elapsed)")

    cohorts = {}
    for tag in ("", "_holdout"):
        if answers.get(tag):
            nm = tag or "cohort1"
            ks = sorted(answers[tag])
            cohorts[nm] = (ks, {k: set(answers[tag][k]["movers"]) for k in ks})

    nval = int(len(train) * VAL_FRAC)
    vp = np.random.default_rng(A.SEED + 1).permutation(len(train))
    val, fitk = [train[i] for i in vp[:nval]], [train[i] for i in vp[nval:]]
    tidx = {k: i for i, k in enumerate(train)}
    vtruth = {k: {genes[i] for i in np.where(Amat[ki[k]] >= A.TAU)[0] if not tide[i]} for k in val}
    frows = np.array([urow[k] for k in fitk])

    sw_ppi_ctrl = Sweeper("chan2a+ppi - chan2a+ppi_rw", axes={"rank": list(PPI_RANKS), "K": list(K_VALUES)})
    sw_ppi_base = Sweeper("chan2a+ppi - chan2a", axes={"rank": list(PPI_RANKS), "K": list(K_VALUES)})
    sw_codep = Sweeper("chan2a+codep - chan2a", axes={"rep": list(CODEP_REPS), "K": list(K_VALUES)})
    absolute = {}

    X = Amat[[ki[k] for k in train]].copy()
    X[:, tide] = 0.0
    for K in K_VALUES:
        t0 = time.time()
        nmf = NMF(n_components=K, init="nndsvda", max_iter=400, random_state=0)
        W = nmf.fit_transform(X).astype(np.float32)
        H = nmf.components_.astype(np.float32)
        Wfit = W[[tidx[k] for k in fitk]]

        def score(src, w, keys):
            s = A.ridge_apply(src[[urow[k] for k in keys]], w) @ H
            s[:, tide] = -np.inf
            return s

        def prec(s, keys, truth):
            out = []
            for j, k in enumerate(keys):
                sc = s[j].copy()
                if k in gpos:
                    sc[gpos[k]] = -np.inf
                out.append(len({genes[t] for t in np.argsort(-sc)[:A.NPRED]} & truth[k]) / float(A.NPRED))
            return np.array(out)

        def fit_arm(src):
            cand = sorted(((prec(score(src, A.ridge_fit(src[frows], Wfit, p), val), val, vtruth).mean(), p)
                           for p in PENALTIES), reverse=True)
            return A.ridge_fit(src[tr], W, cand[0][1])

        wc = fit_arm(chan)
        baseline = {nm: prec(score(chan, wc, ks), ks, truth) for nm, (ks, truth) in cohorts.items()}
        absolute[f"K={K}|chan2a"] = {nm: float(v.mean()) for nm, v in baseline.items()}
        report(f"\n  K={K}: chan2a " + "  ".join(f"{nm} {v.mean():.4f}" for nm, v in baseline.items()))

        for rank in PPI_RANKS:
            real, rw = PPI_SRC[rank]
            wr, ww = fit_arm(real), fit_arm(rw)
            cfg = {"rank": rank, "K": K}
            for nm, (ks, truth) in cohorts.items():
                pr = prec(score(real, wr, ks), ks, truth)
                pw = prec(score(rw, ww, ks), ks, truth)
                sw_ppi_ctrl.add(cfg, nm, pr - pw)
                sw_ppi_base.add(cfg, nm, pr - baseline[nm])
            absolute[f"K={K}|ppi{rank}"] = {nm: float(prec(score(real, wr, ks), ks, truth).mean())
                                            for nm, (ks, truth) in cohorts.items()}
        for rep in CODEP_REPS:
            src = CODEP_SRC[rep]
            w = fit_arm(src)
            cfg = {"rep": rep, "K": K}
            for nm, (ks, truth) in cohorts.items():
                sw_codep.add(cfg, nm, prec(score(src, w, ks), ks, truth) - baseline[nm])
            absolute[f"K={K}|codep_{rep}"] = {nm: float(prec(score(src, w, ks), ks, truth).mean())
                                              for nm, (ks, truth) in cohorts.items()}
        report(f"  K={K} done in {time.time() - t0:.0f}s")

    out = {}
    for sw in (sw_codep, sw_ppi_ctrl, sw_ppi_base):
        report(sw.report())
        out[sw.name] = sw.verdict()

    report("\n  SUMMARY")
    for name, v in out.items():
        report(f"    {name:<34} {v['verdict']:<12} "
               f"(+{v['n_resolved_pos']}/-{v['n_resolved_neg']} of {v['n_cells']} cells, "
               f"sign-consistent {v['sign_consistent']})")
    report(f"\n  total {time.time() - t_start:.0f}s")

    json.dump({"test": "adrn_blocks_swept", "ppi_ranks": list(PPI_RANKS), "codep_reps": list(CODEP_REPS),
               "K_values": list(K_VALUES), "absolute": absolute, "sweeps": out, "log": log},
              open(OUT / "adrn_blocks_swept.json", "w"), indent=2)
    report(f"\n  -> {OUT / 'adrn_blocks_swept.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
