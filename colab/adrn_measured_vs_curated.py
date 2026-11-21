"""MEASURED vs CURATED: for a gene nobody perturbed, which kind of descriptor works?

THE PATTERN THIS TESTS.  Sorting every result in this project by what the feature IS, rather than by what it was
called, gives one clean split:

    MEASUREMENTS that worked          xline +0.058/+0.071, codep +0.021/+0.024, Norman composition +0.4326
    CURATION that failed at cold start  annotation (twin ceiling 0.5365), ESM-2, network topology, ChIP binding,
                                        response columns, and every similarity function learnable over annotation

Six curated representations, six failures. Every measured one has helped.

AND ONE OF THEM WAS NEVER LABELLED.  `codep` is DepMap essentiality across ~1,100 cell lines. Unlike `xline`,
which needs the gene measured in ANOTHER Perturb-seq screen and whose whole gain sits on already-measured genes
(+0.0666 measured vs -0.0045 cold), DepMap is a DIFFERENT EXPERIMENT covering ~18,000 genes -- thousands of them
never Perturb-seq'd. Its +0.021 is therefore ALREADY a cold-start gain, and nobody ever said so. That makes it an
existence proof: a measurement made for an unrelated purpose transfers where six descriptions did not.

THE QUESTION.  If measurements transfer and descriptions do not, then for a gene with no perturbation data the
right descriptor is OTHER ASSAYS OF THAT GENE, not a better ontology. So race them.

    curated       the named channels + Tier A: pathways, GO, complexes, Pfam, domains, compartments. Curation.
    measured      Tier B (DepMap dep_frac/ess, abundance, ppm, FBA) PLUS the full DepMap essentiality profile
                  across every cell line, SVD-reduced. All of it is somebody's experiment.
    both          concatenated.
    measured_perm the DepMap profile permuted across genes -- every gene keeps a real profile, somebody else's.
    curated_perm  the same for the annotation block.
    frequency     the floor.

Both blocks are cold-start by construction: the sealed cohorts' Perturb-seq rows are hidden from everything, so
neither block can be reading the answer.

PREDECLARED, before any number:
    measured >= curated, resolved      -> for unmeasured genes the path is OTHER ASSAYS, not better annotation,
                                          and codep's +0.021 is the floor of that path rather than a curiosity.
    measured ~ 0 over its permutation  -> only a perturbation predicts a perturbation, and the honest advice is
                                          triage-then-screen rather than predict.
    measured < curated                 -> annotation is better than I think and the six failures are about the
                                          top-20 task specifically, not about curation.

LEAK CONTROL.  The DepMap SVD rotation is fitted on TRAINING genes only and sealed genes are projected onto it.
DepMap is an independent assay -- it contains no Perturb-seq response -- but the rotation still must not be
chosen with the sealed genes' help.

Swept over basis rank x prediction depth on robustness.Sweeper, both sealed cohorts.
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
K_VALUES = (30, 60, 120)
NPREDS = (10, 20, 50)
PENALTIES = (1.0, 10.0, 100.0, 1000.0)
DEP_SVD = 128


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("MEASURED vs CURATED descriptors, for genes whose perturbation is unseen")
    report("=" * 100)
    report("  PREDECLARED: measured >= curated => the path for unmeasured genes is OTHER ASSAYS.")
    report("               measured ~ its permutation => only a perturbation predicts a perturbation.")

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
    report(f"\n  {len(kos):,} perturbations; sealed {len(sealed)}; train {len(train):,}")

    base, base_names = A.build_channels(universe, set(train), report)
    extra, extra_names, tb = C.extra_channels(universe, set(train), report)
    curated = np.concatenate([base, extra[:, :tb]], axis=1).astype(np.float32)
    tierb = extra[:, tb:].astype(np.float32)
    report(f"  curated block {curated.shape[1]} dims (named + Tier A) | Tier B measured {tierb.shape[1]} dims")

    # ---- the full DepMap essentiality profile: a different experiment, on ~18,000 genes ----
    dep_path = SP / "CRISPRGeneEffect.csv"
    depmat = np.zeros((len(universe), 0), np.float32)
    if dep_path.exists():
        import csv as _csv
        with open(dep_path, newline="") as fh:
            rd = _csv.reader(fh)
            hdr = next(rd)
            sym = [h.split(" (")[0] for h in hdr[1:]]
            want = {g: i for i, g in enumerate(sym) if g in urow}
            cols = sorted(want.values())
            colgene = [sym[c] for c in cols]
            rows = []
            for r in rd:
                v = np.array([r[1 + c] for c in cols], dtype=object)
                rows.append(np.array([np.nan if x == "" else float(x) for x in v], np.float32))
        D = np.vstack(rows)                                   # cell lines x genes
        report(f"  DepMap {D.shape[0]:,} cell lines x {D.shape[1]:,} genes matched to our universe "
               f"({100*D.shape[1]/len(universe):.1f}% of it)")
        colmean = np.nanmean(D, axis=0)
        colmean[~np.isfinite(colmean)] = 0.0
        idx = np.where(~np.isfinite(D))
        D[idx] = np.take(colmean, idx[1])
        P = np.zeros((len(universe), D.shape[0]), np.float32)  # gene x cell-line profile
        for j, g in enumerate(colgene):
            P[urow[g]] = D[:, j]
        have = np.array([np.abs(P[urow[g]]).sum() > 0 for g in universe])
        for tag, ans in answers.items():
            n = sum(1 for k in ans if k in urow and have[urow[k]])
            report(f"    sealed {tag}: {n}/{len(ans)} have a DepMap profile")
        mu = P[tr_u].mean(0, keepdims=True)
        _, S, Vt = np.linalg.svd(P[tr_u] - mu, full_matrices=False)
        depmat = ((P - mu) @ Vt[:DEP_SVD].T).astype(np.float32)
        report(f"  DepMap profile reduced -> {depmat.shape[1]} dims, rotation on training genes only "
               f"({100*float((S[:DEP_SVD]**2).sum()/(S**2).sum()):.1f}% of training variance)")
    else:
        report("  CRISPRGeneEffect.csv absent -- the measured block is Tier B only, which is 9 dims and a much")
        report("  weaker test. Stated rather than silently degraded.")

    def nz(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    measured = nz(np.concatenate([tierb, depmat], 1)) if depmat.size else nz(tierb)
    curated_n = nz(curated)
    rp = np.random.default_rng(A.SEED + 31)
    FEATS = {"curated": curated_n, "measured": measured,
             "both": nz(np.concatenate([curated, tierb, depmat], 1) if depmat.size
                        else np.concatenate([curated, tierb], 1)),
             "measured_perm": measured[rp.permutation(len(measured))],
             "curated_perm": curated_n[rp.permutation(len(curated_n))]}
    report(f"  measured block {measured.shape[1]} dims | curated block {curated_n.shape[1]} dims")

    freq = (Aabs[train_rows] >= A.TAU).mean(0)
    freq[tide] = -np.inf

    def hit(v, movers, n, k):
        v = np.asarray(v, np.float64).copy()
        v[tide] = -np.inf
        if k in gj:
            v[gj[k]] = -np.inf
        return len({genes[t] for t in np.argsort(-v)[:n]} & set(movers)) / float(n)

    from sklearn.decomposition import NMF
    sw = Sweeper("measured - curated", axes={"K": list(K_VALUES), "npred": list(NPREDS)})
    swp = Sweeper("measured - measured_perm", axes={"K": list(K_VALUES), "npred": list(NPREDS)})
    cells = {}
    for K in K_VALUES:
        X = Aabs[train_rows].copy()
        X[:, tide] = 0.0
        nmf = NMF(n_components=K, init="nndsvda", max_iter=300, random_state=0)
        W = nmf.fit_transform(X).astype(np.float32)
        H = nmf.components_.astype(np.float32)
        del X
        fitted = {}
        for nm, F in FEATS.items():
            rv = np.random.default_rng(A.SEED + 6).permutation(len(tr_u))
            nv = max(50, len(rv) // 5)
            bv, bp = -1e18, PENALTIES[0]
            for pen in PENALTIES:
                w = A.ridge_fit(F[tr_u[rv[nv:]]], W[rv[nv:]], pen)
                e = -float(np.mean((A.ridge_apply(F[tr_u[rv[:nv]]], w) - W[rv[:nv]]) ** 2))
                if e > bv:
                    bv, bp = e, pen
            fitted[nm] = A.ridge_fit(F[tr_u], W, bp)
        report(f"\n  K={K}")
        for npred in NPREDS:
            arms = {}
            for tag, ans in answers.items():
                keys = [k for k in sorted(ans) if k in urow]
                pred = {nm: A.ridge_apply(FEATS[nm][[urow[k] for k in keys]], fitted[nm]) @ H
                        for nm in FEATS}
                for i, k in enumerate(keys):
                    mv = ans[k]["movers"]
                    arms.setdefault("frequency", []).append(hit(freq, mv, npred, k))
                    for nm in FEATS:
                        arms.setdefault(nm, []).append(hit(pred[nm][i], mv, npred, k))
            arms = {n: np.array(v) for n, v in arms.items()}
            cfg, tg = {"K": K, "npred": npred}, f"K{K}|n{npred}"
            sw.add(cfg, tg, arms["measured"] - arms["curated"])
            swp.add(cfg, tg, arms["measured"] - arms["measured_perm"])
            cells[tg] = {n: float(v.mean()) for n, v in arms.items()}
            report(f"    N={npred:>2}: freq {arms['frequency'].mean():.4f} | curated {arms['curated'].mean():.4f} "
                   f"| measured {arms['measured'].mean():.4f} | both {arms['both'].mean():.4f} | "
                   f"m_perm {arms['measured_perm'].mean():.4f} | c_perm {arms['curated_perm'].mean():.4f}")

    for nm, s in (("measured-curated", sw), ("measured-perm", swp)):
        report(s.report())
        if s.inert_axes() or s.duplicate_cells():
            report(f"  GUARD {nm}: inert axes {s.inert_axes()}, duplicate cells {s.duplicate_cells()}")

    mm = {n: float(np.mean([c[n] for c in cells.values()])) for n in list(FEATS) + ["frequency"]}
    report(f"\n  MEANS  " + " | ".join(f"{n} {mm[n]:.4f}" for n in
                                       ("frequency", "curated", "measured", "both",
                                        "measured_perm", "curated_perm")))
    gmc = mm["measured"] - mm["curated"]
    gmp = mm["measured"] - mm["measured_perm"]
    npos = sum(1 for c in sw._cells.values() if c["lo"] > 0)
    ppos = sum(1 for c in swp._cells.values() if c["lo"] > 0)
    report("\n  READING")
    if ppos <= len(swp._cells) / 2:
        report(f"  The measured block does not beat its own permutation ({gmp:+.4f}). Other assays of a gene")
        report("  carry nothing about how it responds to being knocked out. Only a perturbation predicts a")
        report("  perturbation, and the honest advice for unmeasured genes is triage-then-screen.")
    elif npos > len(sw._cells) / 2:
        report(f"  MEASURED beats CURATED by {gmc:+.4f}, resolved in {npos}/{len(sw._cells)} cells, and beats")
        report(f"  its own permutation by {gmp:+.4f}. For a gene nobody has perturbed, the descriptor that")
        report("  works is OTHER EXPERIMENTS on it, not a better ontology. codep's +0.021 was the floor of")
        report("  this path, and the path is real.")
    else:
        report(f"  Measured is live (beats its permutation by {gmp:+.4f}) but does not beat curated")
        report(f"  ({gmc:+.4f}). Both carry something; neither dominates. The useful arm is `both`")
        report(f"  ({mm['both']:.4f} vs curated {mm['curated']:.4f}), and the honest claim is complementarity,")
        report("  not replacement.")

    json.dump({"test": "adrn_measured_vs_curated", "K_values": list(K_VALUES), "npreds": list(NPREDS),
               "dep_svd": DEP_SVD, "cells": cells, "means": mm,
               "gap_measured_minus_curated": gmc, "gap_measured_minus_perm": gmp,
               "sweeps": {"measured_curated": sw.verdict(), "measured_perm": swp.verdict()}, "log": log},
              open(OUT / "adrn_measured_vs_curated.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'adrn_measured_vs_curated.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
