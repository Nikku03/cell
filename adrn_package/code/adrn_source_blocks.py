"""WHICH SOURCE-SIDE DATA BLOCKS ACTUALLY ADD ANYTHING? annotation vs sequence vs co-dependency vs PPI.

THE REQUEST behind this was "use PPI and non-PPI, and pathways and other data". Two of those are already spent,
and saying so is part of the answer rather than an evasion:

  * PATHWAYS ARE THE INCUMBENT, not an addition. chan2a's 696 channels are 120 pathways + 250 GO + 120 complexes
    + 120 Pfam + 120 domains + compartments + lesion classes. Adding pathways means adding what is already there.
  * PPI DEGREE IS ALREADY A CHANNEL (`ppi_degree`, adrn_ko_conjunctions.py:254). And eight previous network tests
    in this project were indistinguishable from degree-matched rewiring, once with the shuffle winning outright
    (0.0151 vs 0.0113). The standing evidence says degree is the *only* thing the graph carries.

So the PPI arm here is not "does PPI help" -- it is the sharper question the prior work forces: **does PPI
NEIGHBOURHOOD STRUCTURE add anything beyond the degree chan2a already has?** That is why the decisive arm is not
`ppi` against `chan2a` but `ppi` against its own DEGREE-MATCHED REWIRING. A rewired graph preserves every node's
degree exactly while destroying who is wired to whom. If the real graph does not beat its rewired twin, the
neighbourhood is noise and the eight prior results were right.

BLOCKS (each standardised on training statistics before concatenation):

    chan2a    696 curated annotation channels. The incumbent.
    esm       mean-pooled ESM-2. Learned from ~250M sequences, exists for every gene, differs between the
              annotation twins that chan2a cannot separate.
    codep     DepMap gene-effect profile across ~1,100 cell lines. MEASURED, dense, and untested as a wholesale
              source side -- only `dep_frac`/`ess` summaries reached the Tier-B channels. This is the one block
              that is neither curation nor graph.
    ppi       PPI adjacency row -> truncated SVD. Neighbourhood identity, not degree.
    ppi_rw    the same construction on a degree-matched rewired graph. THE CONTROL.

GATES -- every one requires the sign on BOTH sealed cohorts past a 95% bootstrap CI over knockouts:

    G_esm     chan2a+esm    > chan2a
    G_codep   chan2a+codep  > chan2a
    G_ppi     chan2a+ppi    > chan2a
    G_ppiCTRL chan2a+ppi    > chan2a+ppi_rw     <- if this fails, G_ppi passing means only "degree, again"
    G_all     chan2a+all    > chan2a

A block that passes its own gate but fails against its control has not earned its place. That distinction is the
entire reason this file exists rather than a concatenate-everything script.

FAIRNESS: ridge penalty tuned PER ARM on a train-only split, since sparse binary channels, dense embeddings and
1,100-dim measured profiles do not share a fair penalty. Sealed cohorts are never touched during tuning.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT, SP = A.OUT, A.SP
MODEL = "esm2_t12_35M_UR50D"
PENALTIES = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
VAL_FRAC, BOOT, SVD_K = 0.2, 10_000, 128


def load_esm(universe):
    files = sorted(SP.glob(f"esm_{MODEL}.shard*.npz")) or [SP / f"esm_{MODEL}.npz"]
    emb = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        for g, v in zip([str(x) for x in z["genes"]], z["E"]):
            emb[g] = v
    dim = len(next(iter(emb.values())))
    E = np.zeros((len(universe), dim), np.float32)
    miss = np.array([g not in emb for g in universe])
    for i, g in enumerate(universe):
        if g in emb:
            E[i] = emb[g]
    return E, miss


def load_codep(universe, report):
    import pandas as pd
    df = pd.read_csv(SP / "CRISPRGeneEffect.csv", index_col=0)
    sym = [c.split(" (")[0] for c in df.columns]
    V = df.to_numpy(np.float32)                      # cell lines x genes
    col = np.nanmean(V, axis=0)
    bad = ~np.isfinite(col)
    col[bad] = 0.0
    ix = np.where(np.isnan(V))
    V[ix] = np.take(col, ix[1])
    pos = {}
    for j, s in enumerate(sym):
        pos.setdefault(s, j)
    P = np.zeros((len(universe), V.shape[0]), np.float32)
    miss = np.array([g not in pos for g in universe])
    for i, g in enumerate(universe):
        if g in pos:
            P[i] = V[:, pos[g]]
    report(f"  codep: {V.shape[0]} cell lines x {len(set(sym)):,} genes; "
           f"missing for {int(miss.sum()):,} of {len(universe):,} universe genes")
    return P, miss


def rewire(edges, n_nodes, seed, passes=10):
    """Degree-preserving double-edge swap: every node keeps its exact degree, who-links-to-whom is destroyed."""
    e = edges.copy()
    rng = np.random.default_rng(seed)
    m = len(e)
    seen = {(min(a, b), max(a, b)) for a, b in e}
    for _ in range(passes * m):
        i, j = rng.integers(0, m, 2)
        if i == j:
            continue
        a, b = e[i]
        c, d = e[j]
        if len({a, b, c, d}) < 4:
            continue
        n1, n2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if n1 in seen or n2 in seen:
            continue
        seen.discard((min(a, b), max(a, b)))
        seen.discard((min(c, d), max(c, d)))
        e[i] = (a, d)
        e[j] = (c, b)
        seen.add(n1)
        seen.add(n2)
    return e


def ppi_blocks(universe, report):
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD
    D = json.load(open(OUT / "cell_complete.json"))
    # cell_complete's `genes` is a list of RECORDS, not symbols -- adrn_ko_conjunctions reads `names` the same
    # way (it does `names[a]` on dicts only after extracting `name`).  The first draft here treated them as
    # strings and died on `unhashable type: dict`.
    names = [g["name"] if isinstance(g, dict) else str(g) for g in D["genes"]]
    idx = {g: i for i, g in enumerate(names)}
    edges = np.array([(a, b) for a, b in D["ppi"] if a < len(names) and b < len(names)], np.int64)
    n = len(names)
    report(f"  ppi: {len(edges):,} edges over {n:,} genes")
    rows = np.array([idx.get(g, -1) for g in universe])

    def embed(e, tag):
        M = sparse.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                              shape=(n, n)).tocsr()
        Z = TruncatedSVD(n_components=SVD_K, random_state=0).fit_transform(M)
        out = np.zeros((len(universe), SVD_K), np.float32)
        ok = rows >= 0
        out[ok] = Z[rows[ok]]
        deg = np.zeros(len(universe))
        deg[ok] = np.asarray(M[rows[ok]].sum(1)).ravel()
        report(f"    {tag}: mean degree over universe {deg.mean():.1f}")
        return out

    real = embed(edges, "real")
    rw = embed(rewire(edges, n, A.SEED), "rewired")
    return real, rw


def main():
    log = []

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("SOURCE-SIDE BLOCKS: annotation vs sequence vs co-dependency vs PPI (with degree-matched control)")
    report("=" * 100)
    report("  PREDECLARED: a block must beat chan2a on BOTH cohorts; PPI must ALSO beat its rewired twin.")

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
    E, emiss = load_esm(universe)
    P, pmiss = load_codep(universe, report)
    PPI, PPIRW = ppi_blocks(universe, report)
    report(f"  esm missing for {int(emiss.sum()):,} universe genes (training mean substituted)")

    def prep(M, miss=None):
        M = M.astype(np.float32).copy()
        if miss is not None and miss.any():
            M[miss] = M[tr[~miss[tr]]].mean(0)
        mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-6
        return (M - mu) / sd

    B = {"chan2a": chan, "esm": prep(E, emiss), "codep": prep(P, pmiss),
         "ppi": prep(PPI), "ppi_rw": prep(PPIRW)}
    cz = prep(chan)
    SRC = {
        "chan2a": chan,
        "chan2a+esm": np.concatenate([cz, B["esm"]], 1),
        "chan2a+codep": np.concatenate([cz, B["codep"]], 1),
        "chan2a+ppi": np.concatenate([cz, B["ppi"]], 1),
        "chan2a+ppi_rw": np.concatenate([cz, B["ppi_rw"]], 1),
        "chan2a+all": np.concatenate([cz, B["esm"], B["codep"], B["ppi"]], 1),
        "esm_only": B["esm"],
        "codep_only": B["codep"],
    }

    from sklearn.decomposition import NMF
    X = Amat[[ki[k] for k in train]].copy()
    X[:, tide] = 0.0
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)

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
    val = [train[i] for i in vp[:nval]]
    fit = [train[i] for i in vp[nval:]]
    tidx = {k: i for i, k in enumerate(train)}
    Wfit = W[[tidx[k] for k in fit]]
    vtruth = {k: {genes[i] for i in np.where(Amat[ki[k]] >= A.TAU)[0] if not tide[i]} for k in val}
    frows = np.array([urow[k] for k in fit])
    fitted = {}
    for name, src in SRC.items():
        cand = sorted(((prec(score(src, A.ridge_fit(src[frows], Wfit, p), val), val, vtruth).mean(), p)
                       for p in PENALTIES), reverse=True)
        fitted[name] = A.ridge_fit(src[tr], W, cand[0][1])
        report(f"  penalty {name:<16} -> {cand[0][1]:>7.1f}  (val {cand[0][0]:.4f})")

    _, inv, cnt = np.unique(chan, axis=0, return_inverse=True, return_counts=True)
    twin_of = cnt[inv] > 1

    per, means = {}, {}
    for tag in ("", "_holdout"):
        ans = answers.get(tag)
        if not ans:
            continue
        name = tag or "cohort1"
        cohort = sorted(ans)
        truth = {k: set(ans[k]["movers"]) for k in cohort}
        per[name] = {a: prec(score(src, fitted[a], cohort), cohort, truth) for a, src in SRC.items()}
        means[name] = {a: float(v.mean()) for a, v in per[name].items()}
        tw = np.array([twin_of[urow[k]] for k in cohort])
        report(f"\n  === {name} ({len(cohort)} knockouts, {int(tw.sum())} with an exact channel twin) ===")
        for a in SRC:
            report(f"  {a:<16} {per[name][a].mean():>8.4f}")

    report(f"\n  {'gate':<10} {'contrast':<34} {'cohort1':>22} {'cohort2':>22}  verdict")
    GATES = {"G_esm": ("chan2a+esm", "chan2a"), "G_codep": ("chan2a+codep", "chan2a"),
             "G_ppi": ("chan2a+ppi", "chan2a"), "G_ppiCTRL": ("chan2a+ppi", "chan2a+ppi_rw"),
             "G_all": ("chan2a+all", "chan2a")}
    out = {}
    for g, (a, b) in GATES.items():
        row, ok = [], True
        for name in per:
            d = per[name][a] - per[name][b]
            br = np.random.default_rng(0)
            bs = d[br.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            row.append((float(d.mean()), lo, hi))
            ok &= lo > 0
        cells = "  ".join(f"{m:+.4f} [{lo:+.4f},{hi:+.4f}]" for m, lo, hi in row)
        report(f"  {g:<10} {a + ' - ' + b:<34} {cells}  {'PASS' if ok else 'fail'}")
        out[g] = {"contrast": f"{a} - {b}", "cells": row, "pass": bool(ok)}

    if out.get("G_ppi", {}).get("pass") and not out.get("G_ppiCTRL", {}).get("pass"):
        report("\n  NOTE: PPI beat chan2a but NOT its degree-matched rewiring -- that is degree, which chan2a"
               " already carries as `ppi_degree`. It has not earned a place.")

    # strata for whichever block won by the most, since the twin story predicts a larger gain on twinned knockouts
    best = max((k for k in ("G_esm", "G_codep", "G_ppi") if out[k]["pass"]),
               key=lambda k: min(c[0] for c in out[k]["cells"]), default=None)
    strat = {}
    if best:
        arm = GATES[best][0]
        report(f"\n  twin strata for {arm} (the twin hypothesis predicts a LARGER gain on twinned knockouts)")
        for name in per:
            cohort = sorted(answers["" if name == "cohort1" else "_holdout"])
            tw = np.array([twin_of[urow[k]] for k in cohort])
            strat[name] = {}
            for lab, m in (("twinned", tw), ("unique", ~tw)):
                if m.sum() >= 10:
                    d = (per[name][arm] - per[name]["chan2a"])[m]
                    strat[name][lab] = {"n": int(m.sum()), "gain": float(d.mean())}
                    report(f"    {name:<9} {lab:<9} n={int(m.sum()):>3}  gain {d.mean():+.4f}")

    json.dump({"test": "adrn_source_blocks", "means": means, "gates": out, "strata": strat,
               "best_block": best, "log": log},
              open(OUT / "adrn_source_blocks.json", "w"), indent=2)
    report(f"\n  -> {OUT / 'adrn_source_blocks.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
