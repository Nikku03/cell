"""DOES A LEARNED SEQUENCE REPRESENTATION BEAT, OR ADD TO, THE CURATED ANNOTATION SOURCE SIDE?

THE QUESTION THIS ANSWERS.  The off-annotation test showed the model can name genes no database links to the
knockout, so the downstream machinery uses data-driven structure.  What it cannot do is tell apart two knockouts
with the same annotation vector -- 99 of 200 cohort-1 knockouts have an EXACT channel twin, and the twin ceiling
(0.536 / 0.621) is where that binds.  A protein sequence differs between twins and, unlike a GO term, was not
written down by a curator: ESM-2's representation is learned from ~250M sequences.  So this is the direct test of
whether the ceiling is the ARCHITECTURE or the VOCABULARY.

ARMS -- same sealed cohorts, same NPRED, same scorer, same NMF programmes.  Only the source-side matrix changes.

    chan2a        the incumbent: 696 curated annotation channels.
    esm           mean-pooled ESM-2 embedding only.  No annotation at all.  This is the arm that matters for
                  unannotated genes, because it is the one that still has an input when the databases are empty.
    chan2a_esm    both, concatenated after per-block standardisation.
    esm_shuffled  ESM rows permuted across genes.  LIVENESS: if this ties `esm`, the embedding is contributing a
                  constant offset rather than gene-specific information and the arm tests nothing.
    frequency     rank by how often a gene moves in training.  The floor every arm must clear.

PREDECLARED GATES, all requiring the sign on BOTH sealed cohorts past a 95% bootstrap CI over knockouts:

    G1  chan2a_esm > chan2a      sequence carries signal the annotation does not.  THE headline question.
    G2  esm > frequency          sequence alone is a usable source side -- the unannotated-gene capability.
    G3  esm > esm_shuffled       liveness.  Fails => G1/G2 are uninterpretable, whatever they say.

THE STRATIFIED READOUT, which is the mechanistic point rather than the headline.  Knockouts are split by whether
they HAVE an exact channel twin in the universe.  If sequence helps because it separates twins, the gain must be
larger on the twinned stratum.  If the gain is flat across strata, sequence is adding generic signal and the twin
story is wrong -- worth knowing either way, and it is the kind of claim that is easy to assert and rarely checked.

FAIRNESS.  Annotation channels are sparse and binary; ESM is dense and continuous.  A single shared ridge penalty
would silently favour one geometry, so the penalty is tuned PER ARM on a train-only validation split and the
sealed cohorts are never touched during tuning.

MISSING SEQUENCES.  125 of 5,120 knockout genes have no reviewed human sequence (renamed symbols such as
AARS -> AARS1, plus clone identifiers).  Those rows get the TRAINING MEAN embedding, so they carry no
gene-specific information rather than a fabricated one, and the count landing in each cohort is reported.
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
VAL_FRAC, BOOT = 0.2, 10_000


def load_esm(kos):
    """gene -> embedding, merged across shards. Returns (matrix aligned to kos, missing mask)."""
    files = sorted(SP.glob(f"esm_{MODEL}.shard*.npz")) or [SP / f"esm_{MODEL}.npz"]
    emb = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        for g, v in zip([str(x) for x in z["genes"]], z["E"]):
            emb[g] = v
    dim = len(next(iter(emb.values())))
    E = np.zeros((len(kos), dim), np.float32)
    miss = np.zeros(len(kos), bool)
    for i, g in enumerate(kos):
        if g in emb:
            E[i] = emb[g]
        else:
            miss[i] = True
    return E, miss, len(files)


def main():
    log = []

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("ESM-2 SEQUENCE SOURCE SIDE vs CURATED ANNOTATION CHANNELS")
    report("=" * 100)
    report("  PREDECLARED  G1 chan2a_esm>chan2a | G2 esm>frequency | G3 esm>esm_shuffled -- both cohorts, 95% CI")

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
    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan_all = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}

    E_all, miss_all, nshard = load_esm(universe)
    report(f"  ESM embeddings merged from {nshard} file(s): {E_all.shape}, "
           f"missing {int(miss_all.sum()):,} of {len(universe):,}")

    tr_rows = np.array([urow[k] for k in train])
    # missing rows take the TRAINING mean, computed before any standardisation
    if miss_all.any():
        E_all[miss_all] = E_all[tr_rows[~miss_all[tr_rows]]].mean(0)
    mu, sd = E_all[tr_rows].mean(0), E_all[tr_rows].std(0) + 1e-6
    E_z = (E_all - mu) / sd
    Cmu, Csd = chan_all[tr_rows].mean(0), chan_all[tr_rows].std(0) + 1e-6
    Cz = (chan_all - Cmu) / Csd

    rr = np.random.default_rng(A.SEED)
    perm = rr.permutation(len(universe))
    SRC = {"chan2a": chan_all,
           "esm": E_z,
           "chan2a_esm": np.concatenate([Cz, E_z], axis=1),
           "esm_shuffled": E_z[perm]}

    from sklearn.decomposition import NMF
    X = Amat[[ki[k] for k in train]].copy()
    X[:, tide] = 0.0
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)
    report(f"  {A.K} programmes fitted on {len(train):,} training knockouts")

    def score(src, w, keys):
        mix = A.ridge_apply(src[[urow[k] for k in keys]], w)
        s = mix @ H
        s[:, tide] = -np.inf
        return s

    def prec(s, keys, truth):
        hits = []
        for j, k in enumerate(keys):
            sc = s[j].copy()
            if k in genes:
                sc[genes.index(k)] = -np.inf
            top = np.argsort(-sc)[:A.NPRED]
            hits.append(len({genes[t] for t in top} & truth[k]) / float(A.NPRED))
        return np.array(hits)

    # ---- per-arm penalty tuning on a TRAIN-ONLY validation split
    nval = int(len(train) * VAL_FRAC)
    vperm = np.random.default_rng(A.SEED + 1).permutation(len(train))
    val = [train[i] for i in vperm[:nval]]
    fit = [train[i] for i in vperm[nval:]]
    tidx = {k: i for i, k in enumerate(train)}
    Wfit = W[[tidx[k] for k in fit]]
    vtruth = {k: set(np.where(Amat[ki[k]] >= A.TAU)[0]) for k in val}
    vtruth = {k: {genes[i] for i in v if not tide[i]} for k, v in vtruth.items()}
    best = {}
    for name, src in SRC.items():
        rows = np.array([urow[k] for k in fit])
        cand = []
        for p in PENALTIES:
            w = A.ridge_fit(src[rows], Wfit, p)
            cand.append((prec(score(src, w, val), val, vtruth).mean(), p))
        cand.sort(reverse=True)
        best[name] = cand[0][1]
        report(f"  penalty {name:<14} -> {cand[0][1]:>7.1f}  (val {cand[0][0]:.4f})")

    fitted = {}
    for name, src in SRC.items():
        fitted[name] = A.ridge_fit(src[tr_rows], W, best[name])

    freq = (Amat[[ki[k] for k in train]] >= A.TAU).mean(0)
    freq[tide] = -np.inf
    freq_top = {genes[t] for t in np.argsort(-freq)[:A.NPRED]}

    # twin strata: does a knockout share an EXACT channel vector with any other gene in the universe?
    _, inv, cnt = np.unique(chan_all, axis=0, return_inverse=True, return_counts=True)
    twin_of = cnt[inv] > 1

    per, means, strat = {}, {}, {}
    for tag in ("", "_holdout"):
        ans = answers.get(tag)
        if not ans:
            continue
        name = tag or "cohort1"
        cohort = sorted(ans)
        truth = {k: set(ans[k]["movers"]) for k in cohort}
        res = {}
        for arm, src in SRC.items():
            res[arm] = prec(score(src, fitted[arm], cohort), cohort, truth)
        res["frequency"] = np.array([len(freq_top & truth[k]) / float(A.NPRED) for k in cohort])
        per[name] = res
        means[name] = {a: float(v.mean()) for a, v in res.items()}
        tw = np.array([twin_of[urow[k]] for k in cohort])
        nmiss = int(sum(miss_all[urow[k]] for k in cohort))
        report(f"\n  === {name} ({len(cohort)} knockouts; {int(tw.sum())} have an exact channel twin; "
               f"{nmiss} lack a sequence) ===")
        for a in ("chan2a_esm", "chan2a", "esm", "esm_shuffled", "frequency"):
            report(f"  {a:<14} {res[a].mean():>8.4f}")
        strat[name] = {}
        for lab, m in (("twinned", tw), ("unique", ~tw)):
            if m.sum() >= 10:
                d = (res["chan2a_esm"] - res["chan2a"])[m]
                strat[name][lab] = {"n": int(m.sum()), "chan2a": float(res["chan2a"][m].mean()),
                                    "chan2a_esm": float(res["chan2a_esm"][m].mean()), "gain": float(d.mean())}
                report(f"    {lab:<9} n={int(m.sum()):>3}  chan2a {res['chan2a'][m].mean():.4f} -> "
                       f"chan2a_esm {res['chan2a_esm'][m].mean():.4f}   gain {d.mean():+.4f}")

    report(f"\n  {'gate':<6} {'contrast':<28} {'cohort1':>22} {'cohort2':>22}  verdict")
    gates = {"G1": ("chan2a_esm", "chan2a"), "G2": ("esm", "frequency"), "G3": ("esm", "esm_shuffled")}
    out = {}
    for g, (a, b) in gates.items():
        row, ok = [], True
        for name in per:
            d = per[name][a] - per[name][b]
            br = np.random.default_rng(0)
            bs = d[br.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            row.append((float(d.mean()), lo, hi))
            ok &= lo > 0
        cells = "  ".join(f"{m:+.4f} [{lo:+.4f},{hi:+.4f}]" for m, lo, hi in row)
        report(f"  {g:<6} {a + ' - ' + b:<28} {cells}  {'PASS' if ok else 'fail'}")
        out[g] = {"contrast": f"{a} - {b}", "cells": row, "pass": bool(ok)}

    json.dump({"test": "adrn_esm_source", "model": MODEL, "penalties": best, "means": means,
               "strata": strat, "gates": out, "log": log},
              open(OUT / "adrn_esm_source.json", "w"), indent=2)
    report(f"\n  -> {OUT / 'adrn_esm_source.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
