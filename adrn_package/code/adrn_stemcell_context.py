"""DOES A PLURIPOTENT CONTEXT TRANSFER DIFFERENTLY FROM A CANCER LINE? The developmental-origin hypothesis, tested.

THE IDEA.  Every cell descends from a zygote, so a shared developmental core should exist and a model that learns
it should transfer across cell types.  Part of that is already confirmed here: the DepMap variance budget puts
86.5% of gene-effect variance in the SHARED gene marginal and only 15.1% in the (gene, context) interaction.  The
common core is real and it is most of the biology.

WHAT WAS MISSING, AND WHY THIS FILE EXISTS.  The multi-line test used six terminal CANCER lines -- aneuploid,
epigenetically drifted, decades in culture, and none of them developmentally upstream of any other.  There was no
pluripotent context anywhere in the training pool, so "propagate from the common origin" had never been tested
against the alternative that origin does not matter.

RENGE (Ishikawa 2023, GSE213069) supplies exactly the missing context: human iPSC, 23 pluripotency-TF CRISPR
knockouts, genome-wide scRNA-seq at FOUR time points, day2 to day5.  It is already on disk.  It has been used in
this project once, to ask whether time-resolution recovers mechanism (it did not -- held-out lift flat at ~1.2x).
It has NEVER been used as a TRAINING CONTEXT for the sealed K562 protocol, which is what is done here.

THREE TESTS.

  A  PER-SOURCE TRANSFER.  Fit the channels -> loadings map on ONE context only and score it on the sealed K562
     cohorts.  This ranks contexts by how well they transfer into K562, and the ranking is what the developmental
     hypothesis makes a prediction about.  K562 is erythroleukaemia -- HAEMATOPOIETIC.  So:
         if lineage governs transfer   Jurkat (T-cell, haematopoietic) should lead the cancer lines
         if origin governs transfer    hiPSC (pluripotent, upstream of everything) should lead or be competitive
         if neither                    the ranking should track sample size and nothing else
     The third outcome is the null and it is a real possibility.

  B  ADD hiPSC TO THE POOL.  Pooled and banked arms with the pluripotent context included, against the existing
     mlk562 (0.2885) and mlbank.  Prices whether a developmentally distinct context adds anything the cancer
     lines could not.

  C  DOES THE TIMEPOINT MATTER.  Each hiPSC day is fitted separately.  Early days sit closer to the direct
     regulon and late days to the differentiation cascade, so if developmental state carries transferable
     information the days should not be interchangeable.

HONEST BOUNDS, up front.  RENGE has 23 knockouts against K562's 4,720 -- roughly 0.5% of the pool.  It CANNOT move
the pooled number by volume, and any pooled result here is a statement about kind, not quantity.  Test A is the
one with real power, because each source is judged on its own and sample size is reported alongside.
"""

from __future__ import annotations

import collections
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT = A.OUT
SP = A.SP
LINES = ("HCT116", "HepG2", "Jurkat", "RPE1", "Melanoma")
DAYS = ("day2", "day3", "day4", "day5")
LINEAGE = {"Jurkat": "haematopoietic", "HCT116": "endoderm", "HepG2": "endoderm",
           "RPE1": "neuroectoderm", "Melanoma": "neural crest", "hiPSC": "PLURIPOTENT",
           "hiPSC_day2": "PLURIPOTENT", "hiPSC_day3": "PLURIPOTENT",
           "hiPSC_day4": "PLURIPOTENT", "hiPSC_day5": "PLURIPOTENT"}


def load_pickle_line(name, gene_index, n_genes):
    path = SP / f"nlz_{name}.pkl"
    if not path.exists():
        return [], np.zeros((0, n_genes), np.float32)
    KO = pickle.load(open(path, "rb"))
    kos = sorted(KO)
    M = np.zeros((len(kos), n_genes), np.float32)
    for i, k in enumerate(kos):
        for g, zval in KO[k].items():
            j = gene_index.get(g)
            if j is not None:
                M[i, j] = abs(float(zval))
    return kos, M


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 106)
    report("PLURIPOTENT vs CANCER CONTEXT: does developmental origin govern transfer into K562?")
    report("=" * 106)
    report("  PREDECLARED: if lineage governs, Jurkat (haematopoietic, like K562) leads the cancer lines.")
    report("  If ORIGIN governs, hiPSC leads or is competitive. If neither, the ranking tracks sample size.")

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    gi = {g: i for i, g in enumerate(genes)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05

    sealed: set[str] = set()
    answers: dict[str, dict] = {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists():
            answers[tag] = json.loads(af.read_text())
    train = [k for k in kos if k not in sealed]
    universe = sorted(set(kos) | sealed)

    from sklearn.decomposition import NMF
    X = Amat[[ki[k] for k in train]].copy()
    X[:, tide] = 0.0
    report(f"\n  K562 training pool {len(train):,}; frozen NMF basis ({A.K} components) ...")
    nmf = NMF(n_components=A.K, init="nndsvda", max_iter=400, random_state=0)
    W_k = nmf.fit_transform(X).astype(np.float32)
    H = nmf.components_.astype(np.float32)
    del X
    project = np.linalg.pinv(H.T).astype(np.float32)

    base, _ = A.build_channels(universe, set(train), report)
    extra, _, tb = C.extra_channels(universe, set(train), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}
    report(f"  channels {chan.shape[1]}")

    # ---------------- every source context, projected onto the frozen basis ----------------
    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {}     # name -> (rows into chan, loadings)
    sources["K562"] = (np.array([urow[k] for k in train]), W_k)
    for line in LINES:
        lkos, M = load_pickle_line(line, gi, len(genes))
        if not lkos:
            continue
        M[:, tide] = 0.0
        keep = [i for i, k in enumerate(lkos) if k in urow]
        if len(keep) < 20:
            continue
        sources[line] = (np.array([urow[lkos[i]] for i in keep]),
                         (project @ M[keep].T).T.astype(np.float32))

    import renge_timecourse as R
    hip_rows, hip_load = [], []
    for day in DAYS:
        try:
            gnames, _, resp, ncells = R.responses(day)
        except Exception as exc:
            report(f"    [warn] hiPSC {day} unavailable ({exc})")
            continue
        if not resp:
            continue
        col = {g: j for j, g in enumerate(gnames)}
        kos_d = sorted(resp)
        M = np.zeros((len(kos_d), len(genes)), np.float32)
        for i, k in enumerate(kos_d):
            v = resp[k]
            for g, j in gi.items():
                c = col.get(g)
                if c is not None:
                    M[i, j] = abs(float(v[c]))
        M[:, tide] = 0.0
        keep = [i for i, k in enumerate(kos_d) if k in urow]
        if len(keep) < 5:
            report(f"    [warn] hiPSC {day}: only {len(keep)} knockouts map to the channel universe")
            continue
        rows = np.array([urow[kos_d[i]] for i in keep])
        load = (project @ M[keep].T).T.astype(np.float32)
        sources[f"hiPSC_{day}"] = (rows, load)
        hip_rows.append(rows)
        hip_load.append(load)
        report(f"    hiPSC {day}: {len(keep)} knockouts usable of {len(kos_d)}")
    if hip_rows:
        sources["hiPSC"] = (np.concatenate(hip_rows), np.concatenate(hip_load))

    report(f"\n  contexts available: {len(sources)}")

    # ---------------- scoring helper ----------------
    def score(weights) -> dict[str, float]:
        out = {}
        for tag in ("", "_holdout"):
            ans = answers.get(tag)
            if not ans:
                continue
            cohort = sorted(ans)
            mixes = A.ridge_apply(chan[[urow[k] for k in cohort]], weights)
            hits = []
            for j, k in enumerate(cohort):
                s = (mixes[j] @ H).copy()
                s[tide] = -np.inf
                if k in gi:
                    s[gi[k]] = -np.inf
                top = np.argsort(-s)[:A.NPRED]
                hits.append(len({genes[i] for i in top} & set(ans[k]["movers"])) / A.NPRED)
            out[tag or "cohort1"] = float(np.mean(hits))
        return out

    # ---------------- TEST A: one context at a time ----------------
    report(f"\n### A. PER-SOURCE TRANSFER into the sealed K562 cohorts")
    report(f"  {'source':<14} {'lineage':<16} {'n':>6} {'cohort1':>9} {'cohort2':>9}")
    per_source = {}
    for name in sorted(sources, key=lambda s: -len(sources[s][0])):
        rows, load = sources[name]
        w = A.ridge_fit(chan[rows], load, A.PENALTY)
        r = score(w)
        per_source[name] = {"n": int(len(rows)), "lineage": LINEAGE.get(name, "erythroid (self)"), **r}
        report(f"  {name:<14} {LINEAGE.get(name, 'erythroid (self)'):<16} {len(rows):>6,} "
               f"{r.get('cohort1', float('nan')):>9.4f} {r.get('_holdout', float('nan')):>9.4f}")

    # ---------------- TEST B: pooled and banked, with and without hiPSC ----------------
    report(f"\n### B. POOLING, with and without the pluripotent context")
    k_rows, k_load = sources["K562"]
    cancer = [s for s in sources if s in LINES]
    hip = [s for s in sources if s == "hiPSC"]
    combos = {
        "K562 only": ["K562"],
        "K562 + cancer lines": ["K562"] + cancer,
        "K562 + hiPSC": ["K562"] + hip,
        "K562 + cancer + hiPSC": ["K562"] + cancer + hip,
    }
    report(f"  {'pool':<26} {'n':>7} {'cohort1':>9} {'cohort2':>9}  {'banked c1':>10} {'banked c2':>10}")
    pooled = {}
    for label, members in combos.items():
        members = [m for m in members if m in sources]
        rows = np.concatenate([sources[m][0] for m in members])
        load = np.concatenate([sources[m][1] for m in members])
        w = A.ridge_fit(chan[rows], load, A.PENALTY)
        plain = score(w)
        resid = k_load - A.ridge_apply(chan[k_rows], w)
        bank = A.ridge_fit(chan[k_rows], resid, A.PENALTY)
        mixes_banked = {}
        for tag in ("", "_holdout"):
            ans = answers.get(tag)
            if not ans:
                continue
            cohort = sorted(ans)
            design = chan[[urow[k] for k in cohort]]
            m = A.ridge_apply(design, w) + A.ridge_apply(design, bank)
            hits = []
            for j, k in enumerate(cohort):
                s = (m[j] @ H).copy()
                s[tide] = -np.inf
                if k in gi:
                    s[gi[k]] = -np.inf
                top = np.argsort(-s)[:A.NPRED]
                hits.append(len({genes[i] for i in top} & set(ans[k]["movers"])) / A.NPRED)
            mixes_banked[tag or "cohort1"] = float(np.mean(hits))
        pooled[label] = {"n": int(len(rows)), "plain": plain, "banked": mixes_banked}
        report(f"  {label:<26} {len(rows):>7,} {plain.get('cohort1', 0):>9.4f} "
               f"{plain.get('_holdout', 0):>9.4f}  {mixes_banked.get('cohort1', 0):>10.4f} "
               f"{mixes_banked.get('_holdout', 0):>10.4f}")

    report(f"\n### C. DOES THE hiPSC TIMEPOINT MATTER (per-day transfer, from test A)")
    for day in DAYS:
        k = f"hiPSC_{day}"
        if k in per_source:
            report(f"  {day}: n={per_source[k]['n']:>3}  cohort1 {per_source[k].get('cohort1', 0):.4f}  "
                   f"cohort2 {per_source[k].get('_holdout', 0):.4f}")

    json.dump({"test": "adrn_stemcell_context", "per_source": per_source, "pooled": pooled,
               "lineage": LINEAGE, "log": log},
              open(OUT / "adrn_stemcell_context.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_stemcell_context.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
