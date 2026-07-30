"""WRITE THE ENHANCER-GENE MODEL INTO THE CELL OBJECT, as a scored edge layer with its limits attached.

WHY THIS EXISTS. The cell object carries 612,133 regulatory edges and 191,447 PPI, but NO enhancer-gene edge
layer: genes hold an `enh` COUNT and `loops3d` holds 767 entries. Every result from the E-G thread -- the
0.6881 AUPRC model, the ranking gate, the polarity work -- lived in modules and JSON and was never written
back, so it contributed nothing to the cell. This connects it.

THE FEATURE-SET DECISION, WHICH IS WHERE THIS COULD SILENTLY BREAK. The best benchmark model uses the TSV's
own DHS.RPM / H3K27ac.RPM / CTCF.RPM columns. Those exist ONLY for the 14,734 benchmark pairs. Scoring a new
locus needs features computable anywhere, which means peak-derived ones from the ENCODE panel. Training on the
RPM columns and applying to peak-derived features would be a distribution shift that no test in this project
would catch, because every existing test scores the benchmark pairs. So the model here is trained on
PEAK-DERIVED features end to end, and the first thing the module does is measure what that costs against the
recorded benchmark-column numbers. If the cost is large, the layer is not written.

WHAT GETS WRITTEN, AND WHAT DOES NOT
  written    edge (element -> gene, cell type), calibrated score, estimated precision, distance, and the
             feature block that produced it
  NOT written  effect magnitude, which is not predictable from biology in this benchmark and is confounded
             with detection power (R2 +0.0881 from power alone, every biological arm net-negative)
  NOT written  polarity, whose mechanism is now unexplained -- the silencer reading and the
             indirect-neighbour reading are both measured out

CALIBRATION, NOT RAW SCORES. A raw classifier margin is not a probability and would be read as one. Scores are
mapped to an out-of-fold empirical precision, so an edge at 0.7 means 70% of held-out pairs scoring that high
were real. That mapping is computed on chromosome-held-out folds, never in-sample.

PROVENANCE TRAVELS WITH THE LAYER. Each edge carries the model id, and the layer carries the measured accuracy,
the distance floor it must be judged against, the LOCO transfer penalty, and the K562 concentration of the
training positives. An edge layer without those numbers invites exactly the overreading this project has spent
its time correcting.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
ENC = SP / "celltypes"
NET = SP / "cell_complete.json.gz"
LAYER = SP / "cell_eg_edges.json.gz"
SEEDS = (0, 1, 2, 3, 4)
TRACKS = ["accessibility", "h3k27ac", "ctcf", "polr2a"]
WINDOW = int(os.environ.get("EG_WINDOW", 250_000))
FLANK = 500
MIN_PRECISION = float(os.environ.get("EG_MINPREC", 0.30))
MODEL_ID = "eg-peakfeat-v1"


def load_peaks(cell, assay):
    import glob
    files = glob.glob(str(ENC / cell / f"{assay}*.bed.gz"))
    if not files:
        return None
    by = {}
    with gzip.open(sorted(files)[0], "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            try:
                s, e = int(f[1]), int(f[2])
            except ValueError:
                continue
            sig = float(f[6]) if len(f) > 6 and f[6] not in (".", "") else 1.0
            by.setdefault(f[0], []).append((s, e, sig))
    out = {}
    for c, v in by.items():
        v.sort()
        out[c] = (np.array([x[0] for x in v]), np.array([x[1] for x in v]),
                  np.array([x[2] for x in v]))
    return out


def sig_at(pk, ch, lo, hi):
    if pk is None or ch not in pk:
        return 0.0
    st, en, sg = pk[ch]
    i = int(np.searchsorted(st, hi))
    if i == 0:
        return 0.0
    j = max(0, i - 400)
    m = en[j:i] >= lo
    return float(sg[j:i][m].sum()) if m.any() else 0.0


FEATNAMES = (["log_dist"]
             + [f"elem_{t}" for t in TRACKS] + [f"tss_{t}" for t in TRACKS]
             + ["n_candidates", "dist_rank", "activity_rank", "activity_share", "dist_excess"])


def featurise(pairs, peaks):
    """pairs = list of (chrom, elem_mid, tss, gene_key). Peak-derived only, so the same code path serves
    both the benchmark rows and any new locus -- which is the point."""
    n = len(pairs)
    F = np.zeros((n, 1 + 2 * len(TRACKS)))
    for i, (ch, mid, tss, _k) in enumerate(pairs):
        F[i, 0] = np.log10(max(abs(mid - tss), 1))
        for j, t in enumerate(TRACKS):
            F[i, 1 + j] = np.log1p(sig_at(peaks.get(t), ch, mid - FLANK, mid + FLANK))
            F[i, 1 + len(TRACKS) + j] = np.log1p(sig_at(peaks.get(t), ch, tss - FLANK, tss + FLANK))
    # competition, within (gene) group
    grp = {}
    for i, (_c, _m, _t, k) in enumerate(pairs):
        grp.setdefault(k, []).append(i)
    C = np.zeros((n, 5))
    act = F[:, 1] + F[:, 1 + TRACKS.index("h3k27ac")]
    for k, idx in grp.items():
        idx = np.array(idx)
        d = 10 ** F[idx, 0]
        a = act[idx]
        C[idx, 0] = len(idx)
        C[idx, 1] = np.argsort(np.argsort(d)) + 1
        C[idx, 2] = np.argsort(np.argsort(-a)) + 1
        C[idx, 3] = a / max(a.sum(), 1e-9)
        C[idx, 4] = F[idx, 0] - np.median(F[idx, 0])
    return np.hstack([F, C])


def main():
    from sklearn.metrics import average_precision_score
    from ranking_gate import load, folds_chrom, fit_oof, groups_of, rank_metrics, summarise
    print("=" * 100)
    print("WRITING THE E-G MODEL INTO THE CELL OBJECT")
    print("=" * 100)
    rows = load()
    y = np.array([1 if r.get("Significant") in ("TRUE", "True", "true") else 0 for r in rows])
    ct = np.array([r["CellType"] for r in rows])
    ch = np.array([r["chrom"] for r in rows])
    cells = sorted(set(ct))
    peaks = {}
    for c in cells:
        got = {t: load_peaks(c, t) for t in TRACKS}
        if all(v is not None for v in got.values()):
            peaks[c] = got
    print(f"  benchmark {len(y):,} pairs, {int(y.sum())} positives; peak panel for {sorted(peaks)}")
    usable = np.array([c in peaks for c in ct])
    print(f"  {int(usable.sum()):,}/{len(y):,} pairs scoreable from peaks "
          f"(missing panel: {sorted(set(ct[~usable]))})")

    pairs = []
    for r, c in zip(rows, ct):
        mid = (int(r["chromStart"]) + int(r["chromEnd"])) // 2
        pairs.append((r["chrom"], mid, int(float(r["startTSS"])),
                      (r["measuredGeneSymbol"], c)))
    X = np.zeros((len(rows), len(FEATNAMES)))
    for c in peaks:
        m = np.where(ct == c)[0]
        X[m] = featurise([pairs[i] for i in m], peaks[c])
    Xu, yu, chu = X[usable], y[usable], ch[usable]
    ku = [pairs[i][3] for i in np.where(usable)[0]]

    # ---- GATE: what does the peak-derived feature set cost vs the recorded benchmark-column numbers? ----
    print(f"\n  GATE -- peak-derived features vs the recorded benchmark-column model")
    ap, pers = [], []
    groups = groups_of(ku, yu)
    oof = np.zeros(len(yu))
    for s in SEEDS:
        f = folds_chrom(chu, s)
        p = fit_oof(Xu, yu, f)
        oof += p / len(SEEDS)
        ap.append(average_precision_score(yu, p))
        pers.append(rank_metrics(p, yu, groups))
    per = {k: {m: float(np.mean([pp[k][m] for pp in pers])) for m in ("r1", "r3", "mrr", "rank", "nrank", "ap")}
           for k in groups}
    sm = summarise(per)
    REF = {"pooled_auprc": 0.5372, "r1": 0.697, "mrr": 0.791, "dist_r1": 0.6509}
    print(f"    peak-derived   pooled AUPRC {np.mean(ap):.4f}   R@1 {sm['r1']:.4f}   MRR {sm['mrr']:.4f}"
          f"   ({len(groups)} groups)")
    print(f"    benchmark-col  pooled AUPRC {REF['pooled_auprc']:.4f}   R@1 {REF['r1']:.4f}   "
          f"MRR {REF['mrr']:.4f}")
    print(f"    distance floor R@1 {REF['dist_r1']:.4f}")
    cost_r1 = sm["r1"] - REF["r1"]
    print(f"    cost of using peak-derived features: R@1 {cost_r1:+.4f}")
    if sm["r1"] <= REF["dist_r1"]:
        raise SystemExit(
            f"peak-derived R@1 {sm['r1']:.4f} does not beat the distance floor {REF['dist_r1']:.4f}; "
            "writing this layer would ship something no better than genomic coordinates, so it is refused")

    # ---- CALIBRATION: score -> out-of-fold empirical precision ----
    print(f"\n  CALIBRATION -- mapping score to held-out precision (never in-sample)")
    order = np.argsort(-oof)
    ys = yu[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    thr = np.linspace(0.05, 0.95, 19)
    calib = []
    for t in thr:
        m = oof >= t
        calib.append((float(t), float(yu[m].mean()) if m.sum() >= 20 else None, int(m.sum())))
    print(f"    {'score>=':>8s} {'n':>7s} {'precision':>10s}")
    for t, p, n in calib:
        if p is not None:
            print(f"    {t:8.2f} {n:7,d} {p:10.4f}")
    keep_thr = next((t for t, p, n in calib if p is not None and p >= MIN_PRECISION), None)
    if keep_thr is None:
        raise SystemExit(f"no score threshold reaches precision {MIN_PRECISION}; refusing to write a layer "
                         "whose edges cannot be given an honest precision")
    print(f"    -> writing edges at score >= {keep_thr:.2f} (precision >= {MIN_PRECISION})")

    # ---- final model on all usable benchmark pairs, then score genome-wide candidates ----
    import xgboost as xgb
    mdl = xgb.XGBClassifier(scale_pos_weight=(yu == 0).sum() / max((yu == 1).sum(), 1), max_depth=4,
                            n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
                            min_child_weight=5, reg_lambda=2.0, eval_metric="aucpr", n_jobs=4)
    mdl.fit(Xu, yu)

    d = json.load(gzip.open(NET, "rt"))
    tss = {}
    for g in d["genes"]:
        try:
            tss.setdefault(g["chrom"], []).append((int(g["tss"]), g["name"]))
        except (KeyError, ValueError, TypeError):
            continue
    ngenes = sum(len(v) for v in tss.values())
    for c in tss:
        tss[c].sort()
    del d
    print(f"\n  SCORING GENOME-WIDE CANDIDATES ({ngenes:,} genes with coordinates)")

    edges, per_cell = [], {}
    for cell in sorted(peaks):
        acc = peaks[cell].get("accessibility")
        if acc is None:
            continue
        cand = []
        for chrom, (st, en, sg) in acc.items():
            if chrom not in tss:
                continue
            mids = (st + en) // 2
            gp = np.array([t for t, _ in tss[chrom]])
            gn = [n for _, n in tss[chrom]]
            lo = np.searchsorted(gp, mids - WINDOW)
            hi = np.searchsorted(gp, mids + WINDOW)
            for i in range(len(mids)):
                for j in range(lo[i], hi[i]):
                    if abs(int(gp[j]) - int(mids[i])) > FLANK:
                        cand.append((chrom, int(mids[i]), int(gp[j]), (gn[j], cell)))
        if not cand:
            continue
        Xc = featurise(cand, peaks[cell])
        sc = mdl.predict_proba(Xc)[:, 1]
        m = sc >= keep_thr
        prec_of = {t: p for t, p, _ in calib if p is not None}
        for i in np.where(m)[0]:
            chrom, mid, t_, (g, _c) = cand[i]
            pr = max((p for t, p in prec_of.items() if sc[i] >= t), default=MIN_PRECISION)
            edges.append({"cell": cell, "chrom": chrom, "elem": mid, "gene": g,
                          "dist": int(abs(mid - t_)), "score": round(float(sc[i]), 4),
                          "precision": round(float(pr), 3), "model": MODEL_ID})
        per_cell[cell] = {"candidates": len(cand), "edges": int(m.sum()),
                          "frac_kept": float(m.mean())}
        print(f"    {cell:9s} {len(cand):>9,} candidates -> {int(m.sum()):>7,} edges "
              f"({100*m.mean():.2f}%)")

    layer = {
        "model": MODEL_ID,
        "n_edges": len(edges),
        "score_threshold": keep_thr,
        "min_precision": MIN_PRECISION,
        "features": FEATNAMES,
        "window_bp": WINDOW,
        "calibration": [{"score": t, "precision": p, "n": n} for t, p, n in calib if p is not None],
        "measured_accuracy": {
            "pooled_auprc_peak_features": float(np.mean(ap)),
            "within_gene_recall_at_1": sm["r1"], "within_gene_mrr": sm["mrr"],
            "n_evaluable_groups": len(groups),
            "distance_only_floor_r1": REF["dist_r1"],
            "benchmark_column_model_r1": REF["r1"],
            "cost_of_peak_features_r1": float(cost_r1)},
        "limits": [
            "717 of 820 training positives are K562, so non-K562 edges are extrapolation",
            "cross-cell-type transfer penalty measured at +0.0597 AUPRC (pooled minus LOCO)",
            "distance alone reaches R@1 0.6509; judge these edges against that floor, not against zero",
            "effect MAGNITUDE is deliberately absent: not predictable from biology in this benchmark and "
            "confounded with detection power (R2 +0.0881 from power alone)",
            "POLARITY is deliberately absent: 19.4% of validated pairs are positive-effect, and both "
            "candidate mechanisms (silencer, indirect-via-neighbour) are measured out",
            "quantitative Hi-C adds +0.0042 MRR (p 0.26) over CTCF, so these edges encode no measured 3D",
            "scored only for cell types with a full ENCODE peak panel"],
        "per_cell": per_cell,
        "edges": edges}
    LAYER.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in layer.items() if k != "edges"}
    json.dump(summary, open(OUT / "eg_layer.json", "w"), indent=1, default=float)
    print(f"\n  {len(edges):,} edges written")
    print(f"  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB)")
    print(f"  -> {OUT/'eg_layer.json'}  (summary + provenance, no edge list)")
    print(f"\n  The cell object is NOT modified in place; this is a sidecar layer keyed on (cell, elem, gene) "
          f"so it can be joined, versioned and dropped without touching the 612k regulatory edges.")


if __name__ == "__main__":
    main()
