"""E2GMAP -- USE 3 of the AlphaGenome 2 data: which DNA switch controls which gene, in each cell type.

From the memory bank (rem/atlas/_cache/ag2bank): ENCODE Registry v4 element activity per biosample (2.35M elements,
1,678 human biosamples) and ENCODE RNA-seq per biosample. No 3D contact data (not held), so contact is a power law in
distance, as in the ABC model's contact-free variant (Fulco et al. 2019; Nasser et al. 2021).

SCORES for an element E (an interval) and gene g in cell type c
  dist      (|d| + 5 kb)^-0.87, d = element midpoint to TSS (the ABC power law).                       [baseline]
  abc       a(E,c) * dist / sum over active elements E' within 5 Mb of g of a(E',c) * dist(E',g), where a is
            the most active class of any registry element overlapping E among c's biosamples: PLS/pELS/dELS 1.0,
            CA-H3K4me3/CA-CTCF/CA-TF/CA-only 0.5, TF-only 0.25, inactive 0.
  coact     Pearson correlation, across ENCODE biosamples with both element calls and RNA-seq, between E's activity
            and g's log1p TPM (registry elements overlapping E; the most active class per biosample).
  final     logistic regression on [log abc, log dist, a(E,c), coact, coact x a(E,c)], fitted on the DEV set only.
DEV: the ENCODE-rE2G K562 CRISPRi benchmark (training set; 10,356 pairs, 471 regulated) -- may be looked at.
EXAM: the ENCODE-rE2G held-out benchmark (4,378 pairs, 190 regulated; WTC11, K562, HCT116, Jurkat, GM12878; pairs not
in the ENCODE-rE2G training data) -- scored once (`exam` locks).
GATES, PREDECLARED (AUPRC = average precision, unweighted; 2,000 bootstrap resamples of EXAM pairs, stratified by
cell type; BETTER if the 2.5th percentile of the AUPRC difference is > 0)
  E1  PRIMARY: final vs dist.
  E2  final vs abc (does co-activity across cell types add?).
  Descriptive: AUPRC per cell type and by distance (< 10 kb, 10-100 kb, > 100 kb); published references from the
  AlphaGenome 2 paper on the same held-out set (different weighting: by direct-effect probability): AG2 0.78,
  AG1 0.73, Borzoi ensemble 0.61, ENCODE-rE2G 0.57.
THE MAP (memory bank layer, local): for the cell types of our screens and benchmarks (K562, HepG2, Jurkat, HCT116,
HEK293, H1, H9, WTC11, GM12878, retinal pigment epithelial cell), every active element within 1 Mb of each protein-
coding / lncRNA TSS scored with `final`; links kept where final >= the DEV threshold of 50% precision.
NOT: no contact maps; element activity is a class, not a signal level; co-activity mixes tissues and cell lines.

Run: python3 -P rem/atlas/e2gmap.py dev | exam | build
"""

from __future__ import annotations
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
sys.path.insert(0, HERE)
_s = importlib.util.spec_from_file_location("ag2bank", os.path.join(HERE, "ag2bank.py"))
ag = importlib.util.module_from_spec(_s); _s.loader.exec_module(ag)
AG = ag.BANK
CC = os.path.join(AG, "ccre", "human")
OUTD = R("outputs", "e2gmap")
CACHE = os.path.join(HERE, "_cache", "e2gmap")
DEVF = os.path.join(AG, "enhancer_gene", "EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz")
EXAMF = os.path.join(AG, "enhancer_gene", "EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz")
CELL_TERM = {"K562": "K562", "WTC11": "WTC11", "HCT116": "HCT116", "Jurkat": "Jurkat, Clone E6-1", "GM12878": "GM12878"}
MAP_CELLS = ["K562", "HepG2", "Jurkat, Clone E6-1", "HCT116", "HEK293", "H1", "H9", "WTC11", "GM12878", "retinal pigment epithelial cell"]
CLASS_ACT = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.25, 0.0])        # PLS pELS dELS CA-H3K4me3 CA-CTCF CA-TF CA-only TF-only Low
FEATS = ["log_abc", "log_dist", "activity", "coact", "coact_x_activity"]


class Registry:
    def __init__(self):
        r = np.load(os.path.join(CC, "registry.npz"))
        self.chrom = r["chrom"]; self.start = r["start"].astype(np.int64); self.end = r["end"].astype(np.int64)
        self.mid = (self.start + self.end) // 2
        self.meta = json.load(open(os.path.join(CC, "meta.json")))["biosamples"]
        self.by = {}
        for c in np.unique(self.chrom):
            ix = np.nonzero(self.chrom == c)[0]; o = ix[np.argsort(self.start[ix])]
            self.by[c] = (o, self.start[o], self.end[o], self.mid[o])

    def activity(self, term):
        """per-element activity (0-1) in a biosample term: most active class over its biosamples."""
        cls = np.full(len(self.chrom), 8, np.uint8)
        for b in self.meta:
            if b["biosample"] == term:
                s = np.load(os.path.join(CC, "shards", b["annotation"] + ".npz"))
                cls[s["idx"]] = np.minimum(cls[s["idx"]], np.minimum(s["cls"], 8))
        return CLASS_ACT[cls]

    def overlap(self, chrom, s, e):
        if chrom not in self.by:
            return np.array([], np.int64)
        o, st, en, _ = self.by[chrom]
        hi = np.searchsorted(st, e); lo = np.searchsorted(st, s - 20000)
        k = np.arange(lo, hi); k = k[(en[k] > s) & (st[k] < e)]
        return o[k]


def dist_contact(d):
    return (np.abs(d) + 5000.0) ** -0.87


def coactivity_inputs(reg):
    """Biosample terms with both element calls and RNA-seq; activity matrix rows are built lazily per element."""
    rna = np.load(os.path.join(AG, "bulk", "encode_rna_human.npz")); m = json.load(open(os.path.join(AG, "bulk", "encode_rna_human_meta.json")))["meta"]
    X = rna["X"].astype(np.float32); gcol = {g: i for i, g in enumerate(rna["cols"].tolist())}
    terms = sorted(set(m["biosample"]) & {b["biosample"] for b in reg.meta})
    E = np.stack([X[[i for i, b in enumerate(m["biosample"]) if b == t]].mean(0) for t in terms])   # terms x genes
    A = np.stack([reg.activity(t) for t in terms]).astype(np.float32)                                # terms x elements
    return terms, E, gcol, A


def _z(M):
    M = M - M.mean(0, keepdims=True); return M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-9)


def features(df, reg, act_by_term, coact_pack):
    terms, E, gcol, A = coact_pack
    Ez = _z(E); n = len(df)
    F = np.zeros((n, len(FEATS)), np.float64)
    for cell, term in CELL_TERM.items():
        rows = np.nonzero((df["CellType"] == cell).to_numpy())[0]
        if not len(rows):
            continue
        act = act_by_term[term]
        for i in rows:
            r = df.iloc[i]
            ov = reg.overlap(r["chrom"], int(r["chromStart"]), int(r["chromEnd"]))
            a = float(act[ov].max()) if len(ov) else 0.0
            mid = (int(r["chromStart"]) + int(r["chromEnd"])) // 2; tss = int(r["startTSS"])
            dc = dist_contact(mid - tss)
            o, st, en, md = reg.by.get(r["chrTSS"], (None,) * 4)
            denom = 1e-12
            if o is not None:
                lo, hi = np.searchsorted(md, tss - 5_000_000), np.searchsorted(md, tss + 5_000_000)
                k = o[lo:hi]; denom += float((act[k] * dist_contact(reg.mid[k] - tss)).sum())
            abc = a * dc / (denom + a * dc)
            ca = 0.0
            g = str(r["measuredGeneEnsemblId"]).split(".")[0]
            if len(ov) and g in gcol:
                av = A[:, ov].max(1)
                if av.std() > 0:
                    ca = float(np.dot((av - av.mean()) / (np.linalg.norm(av - av.mean()) + 1e-9), Ez[:, gcol[g]]))
            F[i] = [math.log(abc + 1e-6), math.log(dc), a, ca, ca * a]
    return F


def _ap(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def _load(path):
    import pandas as pd
    return pd.read_csv(path, sep="\t")


def _prep(path, tag):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"features_{tag}.npz")
    df = _load(path)
    if os.path.exists(p):
        return df, np.load(p)["F"]
    reg = Registry(); pack = coactivity_inputs(reg)
    act = {t: reg.activity(t) for t in set(CELL_TERM.values())}
    F = features(df, reg, act, pack); np.savez(p, F=F, terms=np.array(pack[0]))
    print(f"{tag}: features for {len(df)} pairs; co-activity over {len(pack[0])} biosample terms", flush=True)
    return df, F


def _fit(F, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, C=1.0).fit(F, y)


def dev():
    df, F = _prep(DEVF, "dev"); y = df["Regulated"].astype(bool).to_numpy()
    m = _fit(F, y); pf = m.predict_proba(F)[:, 1]
    out = dict(pairs=len(y), positives=int(y.sum()), auprc=dict(dist=_ap(y, F[:, 1]), abc=_ap(y, F[:, 0]), coact=_ap(y, F[:, 3]),
                                                                  activity=_ap(y, F[:, 2]), final_in_sample=_ap(y, pf)),
               coef=dict(zip(FEATS, m.coef_[0].round(3).tolist())), intercept=float(m.intercept_[0]))
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, pf); ok = np.nonzero(pr[:-1] >= 0.5)[0]
    out["threshold_precision_50"] = float(th[ok[0]]) if len(ok) else None
    os.makedirs(OUTD, exist_ok=True); json.dump(out, open(os.path.join(OUTD, "dev.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def exam():
    lock = os.path.join(OUTD, "exam_lock.json")
    if os.path.exists(lock):
        print("EXAM already scored; refusing."); return
    dd, Fd = _prep(DEVF, "dev"); yd = dd["Regulated"].astype(bool).to_numpy(); m = _fit(Fd, yd)
    df, F = _prep(EXAMF, "exam"); y = df["Regulated"].astype(bool).to_numpy()
    s = dict(final=m.predict_proba(F)[:, 1], abc=F[:, 0], dist=F[:, 1])
    au = {k: _ap(y, v) for k, v in s.items()}
    rng = np.random.default_rng(3); cells = df["CellType"].to_numpy(); groups = [np.nonzero(cells == c)[0] for c in np.unique(cells)]
    diffs = {"E1": [], "E2": []}
    for _ in range(2000):
        b = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        if y[b].sum() == 0:
            continue
        f = _ap(y[b], s["final"][b]); diffs["E1"].append(f - _ap(y[b], s["dist"][b])); diffs["E2"].append(f - _ap(y[b], s["abc"][b]))
    V = {k: dict(diff=au["final"] - au["dist" if k == "E1" else "abc"], ci_2_5=float(np.percentile(v, 2.5)),
                 ci_97_5=float(np.percentile(v, 97.5)), verdict="BETTER" if np.percentile(v, 2.5) > 0 else "NO DIFFERENCE")
         for k, v in diffs.items()}
    dd_ = np.abs(df["distanceToTSS"].to_numpy())
    desc = {"by_cell_type": {c: {k: _ap(y[cells == c], v[cells == c]) for k, v in s.items()} for c in np.unique(cells) if y[cells == c].any()},
            "by_distance": {lab: {k: _ap(y[msk], v[msk]) for k, v in s.items()} for lab, msk in
                            (("<10kb", dd_ < 1e4), ("10-100kb", (dd_ >= 1e4) & (dd_ < 1e5)), (">100kb", dd_ >= 1e5)) if y[msk].any()},
            "published_reference_weighted": dict(AG2=0.78, AG1=0.73, Borzoi_ensemble=0.61, ENCODE_rE2G=0.57)}
    res = dict(auprc=au, verdicts=V, descriptive=desc, pairs=len(y), positives=int(y.sum()), scored=time.strftime("%Y-%m-%d %H:%M"))
    os.makedirs(OUTD, exist_ok=True); json.dump(res, open(lock, "w"), indent=1)
    L = ["=" * 97, "E2GMAP -- switch-to-gene links from the memory bank, held-out CRISPRi benchmark (scored once)", "=" * 97,
         f"  {len(y)} pairs, {int(y.sum())} regulated; AUPRC: final {au['final']:.3f}, abc {au['abc']:.3f}, distance {au['dist']:.3f}"]
    for k, lab in (("E1", "PRIMARY final vs distance"), ("E2", "final vs abc (co-activity adds?)")):
        v = V[k]; L.append(f"  {k} {lab:34s} diff {v['diff']:+.3f}, 95% CI [{v['ci_2_5']:+.3f}, {v['ci_97_5']:+.3f}] -> {v['verdict']}")
    L.append("  by cell type: " + "; ".join(f"{c} final {x['final']:.2f} / dist {x['dist']:.2f}" for c, x in desc["by_cell_type"].items()))
    L.append("  by distance: " + "; ".join(f"{c} final {x['final']:.2f} / dist {x['dist']:.2f}" for c, x in desc["by_distance"].items()))
    L.append("  published on this set (weighted by direct-effect probability, not comparable exactly): AG2 0.78, AG1 0.73, Borzoi 0.61, ENCODE-rE2G 0.57")
    open(os.path.join(HERE, "RESULTS_e2gmap.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


def build():
    """The map layer: for MAP_CELLS, active elements within 1 Mb of each gene TSS, scored with the DEV-fitted model."""
    t0 = time.time()
    dd, Fd = _prep(DEVF, "dev"); m = _fit(Fd, dd["Regulated"].astype(bool).to_numpy())
    thr = json.load(open(os.path.join(OUTD, "dev.json")))["threshold_precision_50"]
    reg = Registry(); terms, E, gcol, A = coactivity_inputs(reg); Ez = _z(E)
    genes = [g for g in json.load(open(os.path.join(AG, "raw", "gencode46_tss.json"))) if g[0] in gcol]
    out_dir = os.path.join(AG, "e2g_map"); os.makedirs(out_dir, exist_ok=True); rep = {}
    Ac = A - A.mean(0, keepdims=True); An = Ac / (np.linalg.norm(Ac, axis=0, keepdims=True) + 1e-9)
    for term in MAP_CELLS:
        act = reg.activity(term); E_, G_, S_ = [], [], []
        for gid, name, gt, ch, tss in genes:
            if ch not in reg.by:
                continue
            o, st, en, md = reg.by[ch]
            lo, hi = np.searchsorted(md, tss - 5_000_000), np.searchsorted(md, tss + 5_000_000)
            k5 = o[lo:hi]; denom = float((act[k5] * dist_contact(reg.mid[k5] - tss)).sum()) + 1e-12
            lo, hi = np.searchsorted(md, tss - 1_000_000), np.searchsorted(md, tss + 1_000_000)
            k = o[lo:hi]; k = k[act[k] > 0]
            if not len(k):
                continue
            a = act[k]; dc = dist_contact(reg.mid[k] - tss); abc = a * dc / denom
            ca = An[:, k].T @ Ez[:, gcol[gid]]
            X = np.c_[np.log(abc + 1e-6), np.log(dc), a, ca, ca * a]
            p = m.predict_proba(X)[:, 1]; keep = p >= thr
            E_.append(k[keep]); G_.append(np.full(int(keep.sum()), gcol[gid], np.int32)); S_.append(p[keep].astype(np.float16))
        e, g, s = np.concatenate(E_), np.concatenate(G_), np.concatenate(S_)
        safe = term.replace(" ", "_").replace(",", "")
        np.savez_compressed(os.path.join(out_dir, f"{safe}.npz"), element=e.astype(np.uint32), gene=g, score=s)
        rep[term] = dict(links=int(len(e)), genes=int(len(np.unique(g))), elements=int(len(np.unique(e))))
        print(f"map {term}: {rep[term]} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(dict(cells=rep, threshold=thr, genes_index="encode_rna_human cols (Ensembl)", element_index="ccre/human/registry.npz"),
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    json.dump(rep, open(os.path.join(OUTD, "map_report.json"), "w"), indent=1)


if __name__ == "__main__":
    {"dev": dev, "exam": exam, "build": build}[sys.argv[1]]()
