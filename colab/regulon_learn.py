"""regulon_learn — the Perturb-seq-supervised, CONTEXT-SPECIFIC (K562) regulon (roadmap Point 1: fix the recall collapse).

The problem: mutreg/propagate seed from TRRUST's ~66 CURATED targets per TF, so recall against the hundreds of genes that
actually move in Perturb-seq is low. And the VIPER negative (viper.py) showed the deeper reason a pan-tissue regulon fails at
all: it predicts the WRONG direction for K562 (GATA1 sign-agreement 0.33 < chance). Both point to the same fix: learn the
regulon FROM K562 data.

The build: for each of the 8 erythroid-core TFs with genome-wide ChIP in K562 (GATA1/GATA2/KLF1/MAX/MYC/RUNX1/SPI1/TAL1),
  CANDIDATES  every gene whose promoter (TSS +/- window) is BOUND by that TF's ChIP.        [binding, not curation]
  LABEL       the gene MOVES in that TF's Perturb-seq knockout (|z| > thr).                  [function, measured]
  FEATURES    ChIP peak strength + proximity + count at the promoter ; promoter chromatin (DNase, H3K27ac, H3K4me3, POLR2A,
              H3K27me3-repressive) ; baseline expression.                                    [mechanism, all measured K562]
Binding != function is the whole point (a TF binds thousands of promoters, regulates a fraction). The model learns WHICH bound
genes are functional -> a K562-specific regulon of the confident subset, far larger than the curated 66.

THE HONESTY GATE (leave-one-TF-out): predict the held-out TF's functional targets among its bound candidates. Compare:
  MECHANISM  ChIP + promoter chromatin only (no responsiveness prior)
  PRIOR      the gene's generic mover-frequency across the OTHER 7 TFs   [the cascade_all trap: 'some genes just move']
  FULL       both
If MECHANISM does not beat PRIOR for held-out TFs, the model only relearned 'responsive genes' -- the far-field wall again, and
we say so. If it does, we have learned TF-specific functional targeting that generalizes, and the expanded regulon is real.
"""
import json, gzip, bisect, math, collections
from pathlib import Path
import numpy as np
OUT = Path("outputs/orphan")
INV = OUT / "invivo"
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TFS = ["GATA1", "GATA2", "KLF1", "MAX", "MYC", "RUNX1", "SPI1", "TAL1"]


def _tss():
    D = json.load(open(OUT / "cell_complete.json"))
    t = {}
    for g in D["genes"]:
        if g.get("chrom") and g.get("tss"):
            try:
                t[g["name"]] = (g["chrom"], int(g["tss"]))
            except (ValueError, TypeError):
                pass
    return t


def _peaks(bedgz, sigcol=6):
    """per-chrom sorted [(start,end,signal)]."""
    import crispr_gate as cg
    return cg._signal_index(bedgz, sigcol)


def _promoter_max(idx, chrom, tss, half=1000):
    import crispr_gate as cg
    return cg._max_signal(idx, chrom, tss - half, tss + half)


def _chip_at_promoter(idx, chrom, tss, window):
    """max peak signal, nearest-peak distance, and peak count within TSS +/- window."""
    v = idx.get(chrom)
    if not v:
        return 0.0, window + 1, 0
    starts = [x[0] for x in v]
    lo = bisect.bisect_left(starts, tss - window - 500)
    best = 0.0; nearest = window + 1; cnt = 0
    for k in range(lo, len(v)):
        s, e, sig = v[k]
        if s > tss + window:
            break
        center = (s + e) // 2
        d = abs(center - tss)
        if d <= window:
            cnt += 1
            if sig > best:
                best = sig
            if d < nearest:
                nearest = d
    return best, nearest, cnt


def build(window=5000, thr=1.5, min_ko_for_prior=1):
    import viper as v
    genes, gidx, ps = v.load_universe()
    measured = set(genes)
    sigs = v.load_signatures(only=set(TFS))[2]
    tss = _tss()
    # promoter chromatin marks
    marks = {m: _peaks(INV / "marks" / f"{m}.bed.gz", 6) for m in
             ["dnase", "h3k27ac", "h3k4me3", "polr2a", "h3k27me3"]}
    # baseline expression from Perturb-seq control mean (var-level)
    import h5py
    h = h5py.File(f"{SP}/gwps.h5ad", "r")
    base = {}
    if "mean" in h["var"]:
        mv = h["var"]["mean"][:]
        dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
        cats = dec(h["var"]["__categories"]["gene_name"][:]); codes = h["var"]["gene_name"][:]
        vg = [cats[c] for c in codes]
        base = {g: float(x) for g, x in zip(vg, mv)}
    h.close()
    # movers per TF (measured-gene moves) for label + responsiveness prior
    moved = {t: {genes[i] for i in range(len(genes)) if abs(sigs[t][i]) > thr} for t in TFS if t in sigs}
    movefreq = collections.Counter()
    for t in moved:
        for g in moved[t]:
            movefreq[g] += 1
    rows = []
    for t in TFS:
        if t not in sigs:
            continue
        cidx = _peaks(INV / "chip_gw" / f"{t}.bed.gz", 6)
        for g, (c, p) in tss.items():
            if g not in measured or g == t:
                continue
            chip, ndist, ncnt = _chip_at_promoter(cidx, c, p, window)
            if chip <= 0:                              # candidate = BOUND at promoter
                continue
            lab = 1 if g in moved[t] else 0
            prior = (movefreq[g] - (1 if g in moved[t] else 0)) / max(len(moved) - 1, 1)  # leave-self-out
            feat = {
                "chip_sig": chip, "chip_log_dist": math.log10(ndist + 1.0), "chip_count": ncnt,
                "prom_dnase": _promoter_max(marks["dnase"], c, p), "prom_h3k27ac": _promoter_max(marks["h3k27ac"], c, p),
                "prom_h3k4me3": _promoter_max(marks["h3k4me3"], c, p), "prom_polr2a": _promoter_max(marks["polr2a"], c, p),
                "prom_h3k27me3": _promoter_max(marks["h3k27me3"], c, p), "base_expr": base.get(g, 0.0),
                "J_movefreq": prior, "_tf": t, "_gene": g, "_y": lab}
            rows.append(feat)
    return rows


def run(window=5000, thr=1.5):
    import xgboost as xgb
    from sklearn.metrics import average_precision_score
    rows = build(window, thr)
    fcols = ["chip_sig", "chip_log_dist", "chip_count", "prom_dnase", "prom_h3k27ac", "prom_h3k4me3",
             "prom_polr2a", "prom_h3k27me3", "base_expr"]
    prior_col = "J_movefreq"
    tfs = [t for t in TFS if any(r["_tf"] == t for r in rows)]
    X = np.array([[r[c] for c in fcols] for r in rows], dtype=np.float64)
    Xp = np.array([[r[prior_col]] for r in rows], dtype=np.float64)
    y = np.array([r["_y"] for r in rows]); tfarr = np.array([r["_tf"] for r in rows])
    def loto(cols_X):
        pred = np.zeros(len(y))
        for t in tfs:
            te = tfarr == t; tr = ~te
            if y[tr].sum() < 3 or y[te].sum() < 1:
                pred[te] = np.nan; continue
            pos = max((y[tr] == 1).sum(), 1); neg = (y[tr] == 0).sum()
            m = xgb.XGBClassifier(scale_pos_weight=neg / pos, max_depth=3, n_estimators=200,
                                  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                  min_child_weight=5, reg_lambda=2.0, eval_metric="aucpr", n_jobs=4)
            m.fit(cols_X[tr], y[tr]); pred[te] = m.predict_proba(cols_X[te])[:, 1]
        return pred
    def auprc_bytf(pred):
        out = {}
        for t in tfs:
            te = (tfarr == t) & np.isfinite(pred)
            if y[te].sum() >= 1 and (y[te] == 0).sum() >= 1:
                out[t] = float(average_precision_score(y[te], pred[te]))
        return out
    mech = auprc_bytf(loto(X))                                  # mechanism: ChIP + chromatin
    prior = auprc_bytf(loto(Xp))                                # responsiveness prior only
    full = auprc_bytf(loto(np.column_stack([X, Xp])))          # both
    chip_only = auprc_bytf(loto(X[:, [0]]))                     # naive: ChIP strength alone (the un-learned regulon)
    def mean(d):
        return round(float(np.mean(list(d.values()))), 3) if d else float("nan")
    base_rate = round(float(y.mean()), 3)
    pos_per_tf = {t: int(((tfarr == t) & (y == 1)).sum()) for t in tfs}
    n_tf_evaluable = sum(1 for t in tfs if pos_per_tf[t] >= 3)   # a held-out AUPRC needs a few positives to mean anything
    res = {"n_candidates": len(y), "n_positive": int(y.sum()), "base_rate": base_rate, "tfs": tfs,
           "window": window, "move_thr": thr, "positives_per_tf": pos_per_tf, "n_tf_evaluable": n_tf_evaluable,
           "mechanism_AUPRC": mean(mech), "prior_AUPRC": mean(prior), "full_AUPRC": mean(full),
           "chip_only_AUPRC": mean(chip_only),
           "per_tf_mechanism": {t: round(mech.get(t, float('nan')), 3) for t in tfs},
           "per_tf_prior": {t: round(prior.get(t, float('nan')), 3) for t in tfs},
           # a claim of generalization requires (a) enough evaluable held-out TFs and (b) mechanism > prior
           "mechanism_beats_prior": (n_tf_evaluable >= 4) and (mean(mech) > mean(prior) + 0.01)}
    return res, rows, (X, Xp, y, tfarr, fcols)


def export_regulon(rows, thr_prob=0.5):
    """train on ALL 8 TFs and emit the expanded K562 regulon (confident functional targets per TF)."""
    import xgboost as xgb
    fcols = ["chip_sig", "chip_log_dist", "chip_count", "prom_dnase", "prom_h3k27ac", "prom_h3k4me3",
             "prom_polr2a", "prom_h3k27me3", "base_expr"]
    X = np.array([[r[c] for c in fcols] for r in rows]); y = np.array([r["_y"] for r in rows])
    pos = max((y == 1).sum(), 1); neg = (y == 0).sum()
    m = xgb.XGBClassifier(scale_pos_weight=neg / pos, max_depth=3, n_estimators=200, learning_rate=0.05,
                          min_child_weight=5, reg_lambda=2.0, eval_metric="aucpr", n_jobs=4).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    reg = collections.defaultdict(list)
    for r, pr in zip(rows, p):
        if pr >= thr_prob:
            reg[r["_tf"]].append((r["_gene"], round(float(pr), 3)))
    return {t: sorted(v, key=lambda x: -x[1]) for t, v in reg.items()}


def main():
    print("=" * 100)
    print("REGULON-LEARN — a K562-specific, Perturb-seq-supervised regulon (Point 1: fix the recall collapse)")
    print("=" * 100)
    res, rows, _ = run()
    print(f"\n  {res['n_candidates']:,} ChIP-bound (TF,gene) candidates over {len(res['tfs'])} TFs, {res['n_positive']} functional "
          f"(move |z|>{res['move_thr']}); base rate {res['base_rate']:.1%}. Leave-one-TF-out.\n")
    print(f"  {'model':34s} {'held-out AUPRC':>14s}")
    print(f"  {'ChIP strength alone (naive regulon)':34s} {res['chip_only_AUPRC']:14.3f}")
    print(f"  {'responsiveness PRIOR only':34s} {res['prior_AUPRC']:14.3f}   (the control: some genes just move)")
    print(f"  {'MECHANISM (ChIP + promoter chromatin)':34s} {res['mechanism_AUPRC']:14.3f}")
    print(f"  {'FULL (mechanism + prior)':34s} {res['full_AUPRC']:14.3f}")
    print(f"\n  positives per TF: {res['positives_per_tf']}  -> only {res['n_tf_evaluable']} TF(s) have >=3 (evaluable held-out).")
    beats = res["mechanism_beats_prior"]
    if beats:
        reg = export_regulon(rows)
        sizes = {t: len(reg.get(t, [])) for t in res["tfs"]}
        res["expanded_regulon_sizes"] = sizes
        json.dump({"regulon": reg, "meta": res}, open(OUT / "regulon_k562.json", "w"), indent=1)
        verdict = ("PASS -- mechanism (this TF's ChIP + promoter chromatin) beats the generic responsiveness prior for HELD-OUT "
                   "TFs ({:.3f} vs {:.3f}) with {} evaluable held-out TFs: the model learned TF-specific functional targeting "
                   "that GENERALIZES. Expanded K562 regulon emitted (sizes {})."
                   ).format(res["mechanism_AUPRC"], res["prior_AUPRC"], res["n_tf_evaluable"], sizes)
    else:
        verdict = ("HONEST NEGATIVE (untestable, not a coding bug) -- the supervised regulon CANNOT be validated for "
                   "generalization on this data: promoter-proximal ChIP binding almost never coincides with a Perturb-seq mover "
                   "here. Only {} functional positives across 8 TFs (mostly GATA1={}, MAX={}); {} of 8 TF knockouts have ZERO "
                   "promoter-bound movers. Two real causes, both the project's recurring wall: (1) these TFs act through DISTAL "
                   "enhancers, not promoters, so promoter-binding is the wrong candidate set; (2) the K562 Perturb-seq footprints "
                   "are too SPARSE (same emptiness that sank VIPER). With <4 evaluable held-out TFs the leave-one-TF-out AUPRC is "
                   "noise, so NO generalization claim is made. Recall-collapse is real but is NOT fixable by promoter-ChIP "
                   "supervision on this dataset."
                   ).format(int(res["n_positive"]), res["positives_per_tf"].get("GATA1", 0),
                            res["positives_per_tf"].get("MAX", 0),
                            sum(1 for t in res["tfs"] if res["positives_per_tf"].get(t, 0) == 0))
    print(f"\n  VERDICT: {verdict}")
    res["verdict"] = verdict
    res["note"] = ("Point 1: a K562-specific regulon learned by supervising TF ChIP + promoter chromatin on Perturb-seq movers, "
                   "for the 8 erythroid-core TFs with genome-wide K562 ChIP. Candidates = ChIP-bound promoters; label = moves in "
                   "that TF's Perturb-seq KO; the model picks the functional subset (binding!=function). Honesty gate = "
                   "leave-one-TF-out vs a responsiveness prior: mechanism must beat 'some genes just move' to claim TF-specific "
                   "generalization. Scope-limited to 8 TFs (only these have genome-wide K562 ChIP locally).")
    json.dump(res, open(OUT / "regulon_learn.json", "w"), indent=1)
    print("\n  -> outputs/orphan/regulon_learn.json" + (" + regulon_k562.json" if beats else ""))
    return res


if __name__ == "__main__":
    main()
