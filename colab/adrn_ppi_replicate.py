"""REPLICATE the PPI-beats-rewiring result on an INDEPENDENT edge set, at several embedding ranks.

WHAT IS BEING REPLICATED.  adrn_source_blocks found chan2a+ppi > chan2a+ppi_rw (+0.0155 / +0.0325), i.e. PPI
neighbourhood structure carries signal beyond degree.  That contradicts eight earlier network tests in this
project, all of which tied degree-matched rewiring.  adrn_ppi_verify already showed the control is sound (degree
exactly preserved, 98.2% of edges moved) and the edges do not leak (co-response ratio 1.05x).  What it could NOT
show is whether the finding is specific to that particular edge list or that particular embedding rank.

TWO WAYS A SURVIVING RESULT CAN STILL BE AN ARTEFACT, and this addresses both:

  1. EDGE-SET SPECIFIC.  `cell_complete.json`'s 191,447 edges are an accumulated mixture with NO provenance
     metadata -- the file's `_meta` is empty, so what went into it cannot be audited.  BioPlex is a single
     assay with a stated method (AP-MS, HA-FLAG bait immunoprecipitation, LC-MS/MS, CompPASS-plus) and a
     published probability per edge.  If the effect is real it must survive on edges whose origin is known.
  2. RANK SPECIFIC.  128 components was an unexamined default.  An effect that appears only at one truncation is
     a property of that truncation.  Ranks 64 / 128 / 256 are run for both edge sets.

PREDECLARED GATE.  The BioPlex arm must beat its own degree-matched rewiring on BOTH sealed cohorts at MORE THAN
ONE rank.  A single (source, rank) cell passing is what an artefact looks like.

Reported alongside: the same sweep on the original mixed edge set, so that "replicates" and "was rank-specific
all along" are distinguishable rather than conflated.

BIOPLEX FILTERING: pInt >= 0.75, the floor already used elsewhere in this repo (bioplex.json `pint_floor`), so
the threshold is inherited rather than chosen to suit this test.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from adrn_source_blocks import rewire

OUT, SP = A.OUT, A.SP
RANKS = (64, 128, 256)
PENALTIES = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
VAL_FRAC, BOOT, PINT = 0.2, 10_000, 0.75
BIOPLEX = ("BioPlex_293T_Network_10K_Dec_2019.tsv", "BioPlex_HCT116_Network_5.5K_Dec_2019.tsv")


def bioplex_edges(names, report):
    idx = {g: i for i, g in enumerate(names)}
    seen, kept, dropped = set(), [], 0
    for fn in BIOPLEX:
        p = SP / fn
        if not p.exists():
            report(f"  MISSING {fn}")
            continue
        with open(p) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                try:
                    if float(row["pInt"]) < PINT:
                        continue
                except (ValueError, KeyError):
                    continue
                a, b = idx.get(row["SymbolA"].strip('"')), idx.get(row["SymbolB"].strip('"'))
                if a is None or b is None or a == b:
                    dropped += 1
                    continue
                k = (min(a, b), max(a, b))
                if k not in seen:
                    seen.add(k)
                    kept.append(k)
    report(f"  bioplex: {len(kept):,} distinct edges at pInt>={PINT} ({dropped:,} rows dropped: gene not in "
           f"the cell's symbol space or self-loop)")
    return np.array(kept, np.int64)


def main():
    log = []

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("REPLICATE the PPI result: independent edge set (BioPlex AP-MS) x embedding rank")
    report("=" * 100)
    report(f"  PREDECLARED: bioplex must beat its rewired twin on BOTH cohorts at >1 of ranks {RANKS}.")

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

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] if isinstance(g, dict) else str(g) for g in D["genes"]]
    n = len(names)
    mixed = np.array([(a, b) for a, b in D["ppi"] if a < n and b < n], np.int64)
    bp = bioplex_edges(names, report)
    report(f"  mixed (cell_complete): {len(mixed):,} edges")
    ov = len({(min(a, b), max(a, b)) for a, b in mixed} & {(a, b) for a, b in bp})
    report(f"  overlap between the two edge sets: {ov:,} "
           f"({ov / max(len(bp), 1):.1%} of bioplex) -- they are related but not the same graph")

    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD, NMF
    rows_u = np.array([{g: i for i, g in enumerate(names)}.get(g, -1) for g in universe])

    def embed(e, k, seed=None):
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

    def prep(M):
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
    val = [train[i] for i in vp[:nval]]
    fit = [train[i] for i in vp[nval:]]
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

    base_res = {}
    wc = fit_arm(chan)
    for nm, (ks, truth) in cohorts.items():
        base_res[nm] = prec(score(chan, wc, ks), ks, truth)
        report(f"  chan2a baseline {nm}: {base_res[nm].mean():.4f}")

    results, rows_out = {}, []
    for sname, E in (("bioplex", bp), ("mixed", mixed)):
        for k in RANKS:
            real = np.concatenate([cz, prep(embed(E, k))], 1)
            rw = np.concatenate([cz, prep(embed(E, k, seed=A.SEED))], 1)
            wr, ww = fit_arm(real), fit_arm(rw)
            cells, ok_ctrl, ok_base = [], True, True
            for nm, (ks, truth) in cohorts.items():
                pr = prec(score(real, wr, ks), ks, truth)
                pw = prec(score(rw, ww, ks), ks, truth)
                d = pr - pw
                br = np.random.default_rng(0)
                bs = d[br.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
                lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                db = pr - base_res[nm]
                bs2 = db[br.integers(0, len(db), size=(BOOT, len(db)))].mean(1)
                lo2 = float(np.percentile(bs2, 2.5))
                cells.append({"cohort": nm, "real": float(pr.mean()), "rw": float(pw.mean()),
                              "vs_rw": [float(d.mean()), lo, hi], "vs_chan2a": [float(db.mean()), lo2]})
                ok_ctrl &= lo > 0
                ok_base &= lo2 > 0
            results[f"{sname}|{k}"] = {"cells": cells, "beats_rewired": bool(ok_ctrl),
                                       "beats_chan2a": bool(ok_base)}
            c1, c2 = cells[0], cells[1] if len(cells) > 1 else cells[0]
            rows_out.append((sname, k, c1, c2, ok_ctrl, ok_base))
            report(f"  {sname:<8} rank {k:>3}  real {c1['real']:.4f}/{c2['real']:.4f}  "
                   f"rw {c1['rw']:.4f}/{c2['rw']:.4f}  "
                   f"vs_rw {c1['vs_rw'][0]:+.4f}/{c2['vs_rw'][0]:+.4f}  "
                   f"{'BEATS RW' if ok_ctrl else 'ties rw'}  {'>chan2a' if ok_base else '~chan2a'}")

    npass = sum(1 for s, k, *_ , ok, _ in rows_out if s == "bioplex" and ok)
    gate = npass > 1
    report(f"\n  bioplex beats its rewired twin on both cohorts at {npass} of {len(RANKS)} ranks")
    report(f"  GATE: {'PASS -- replicates on an independent, provenance-clean edge set' if gate else 'FAIL'}")
    mpass = sum(1 for s, k, *_, ok, _ in rows_out if s == "mixed" and ok)
    report(f"  (mixed edge set for comparison: {mpass} of {len(RANKS)} ranks)")

    json.dump({"test": "adrn_ppi_replicate", "ranks": list(RANKS), "pint": PINT,
               "n_bioplex_edges": int(len(bp)), "n_mixed_edges": int(len(mixed)), "overlap": int(ov),
               "chan2a": {k: float(v.mean()) for k, v in base_res.items()},
               "results": results, "bioplex_ranks_passing": npass, "gate_pass": bool(gate), "log": log},
              open(OUT / "adrn_ppi_replicate.json", "w"), indent=2)
    report(f"\n  -> {OUT / 'adrn_ppi_replicate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
