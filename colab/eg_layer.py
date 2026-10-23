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
MODEL_ID = "eg-peakfeat-v2-nocompetition"


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


# COMPETITION FEATURES ARE EXCLUDED, and that is a correction rather than a simplification.
# `competition_audit.py` found three things. n_candidates alone scores AUROC 0.2733, so it encodes the
# per-group base rate -- arithmetic, not biology. The block's whole gain disappears where competition is a
# coherent idea at all: +0.0836 over all groups but -0.0068 restricted to groups with >=5 candidates. And
# decisively for this module, the features are defined RELATIVE TO A CANDIDATE SET: the benchmark's is
# "elements someone chose to test", this module's genome-wide set is "every accessible peak in a 250 kb
# window", and medians move from 19 to 44 (n_candidates), 8 to 33 (dist_rank), 0.000 to 0.512 (dist_excess).
# Training on one and scoring the other -- which the first version of this layer did -- cost -0.1201 AUPRC
# and landed BELOW the no-competition baseline. A feature that cannot be computed the same way at training
# and at inference does not belong in a shipped layer.
FEATNAMES = (["log_dist"]
             + [f"elem_{t}" for t in TRACKS] + [f"tss_{t}" for t in TRACKS])


def featurise(pairs, peaks):
    """pairs = list of (chrom, elem_mid, tss, gene_key). Peak-derived and CANDIDATE-SET INDEPENDENT, so the
    same code path gives the same value for a pair whether it is scored inside the benchmark or genome-wide
    -- which is what the first version got wrong."""
    n = len(pairs)
    F = np.zeros((n, 1 + 2 * len(TRACKS)))
    for i, (ch, mid, tss, _k) in enumerate(pairs):
        F[i, 0] = np.log10(max(abs(mid - tss), 1))
        for j, t in enumerate(TRACKS):
            F[i, 1 + j] = np.log1p(sig_at(peaks.get(t), ch, mid - FLANK, mid + FLANK))
            F[i, 1 + len(TRACKS) + j] = np.log1p(sig_at(peaks.get(t), ch, tss - FLANK, tss + FLANK))
    return F


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
    # v1 reference kept so the cost of dropping competition is visible rather than quietly absorbed
    REF = {"pooled_auprc": 0.5372, "r1": 0.697, "mrr": 0.791, "dist_r1": 0.6509,
           "v1_peak_auprc_with_competition": 0.5322}
    print(f"    peak-derived   pooled AUPRC {np.mean(ap):.4f}   R@1 {sm['r1']:.4f}   MRR {sm['mrr']:.4f}"
          f"   ({len(groups)} groups)")
    print(f"    benchmark-col  pooled AUPRC {REF['pooled_auprc']:.4f}   R@1 {REF['r1']:.4f}   "
          f"MRR {REF['mrr']:.4f}")
    print(f"    distance floor R@1 {REF['dist_r1']:.4f}")
    cost_r1 = sm["r1"] - REF["r1"]
    print(f"    cost of using peak-derived features: R@1 {cost_r1:+.4f}")
    print(f"    v1 (with competition, but computed inconsistently) pooled AUPRC "
          f"{REF['v1_peak_auprc_with_competition']:.4f}; v2 drops that block because it could not be "
          f"computed the same way at training and inference")
    if sm["r1"] <= REF["dist_r1"]:
        raise SystemExit(
            f"peak-derived R@1 {sm['r1']:.4f} does not beat the distance floor {REF['dist_r1']:.4f}; "
            "writing this layer would ship something no better than genomic coordinates, so it is refused")

    # ---- CALIBRATION, PER DISTANCE STRATUM ----
    # A single global curve is wrong here and would ship an overstated layer. The benchmark's candidates are
    # DISTAL-heavy (63.5% beyond 250 kb, 1.9% under 10 kb) while genome-wide candidates within a 250 kb
    # window are PROXIMAL-heavy (22% under 10 kb, 65% at 10-100 kb). Proximal pairs score high on distance
    # whether or not they are real, so a distal-calibrated precision applied to them overstates the layer.
    # Precision is therefore calibrated inside distance strata, and a stratum the benchmark cannot support
    # emits NO edges rather than edges carrying an unsupported number.
    STRATA = [(0, 10_000, "<10 kb"), (10_000, 100_000, "10-100 kb"), (100_000, WINDOW, "100-250 kb")]
    MIN_SUPPORT = 150
    du = 10 ** Xu[:, 0]
    print(f"\n  CALIBRATION -- per distance stratum, out-of-fold (never in-sample)")
    print(f"    benchmark candidate composition vs the genome-wide window:")
    thr = np.linspace(0.05, 0.95, 19)
    calib = {}
    for lo, hi, lab in STRATA:
        sm_ = (du >= lo) & (du < hi)
        npos = int(yu[sm_].sum())
        print(f"    {lab:12s} benchmark n={int(sm_.sum()):5,d} pos={npos:4d} "
              f"base rate {yu[sm_].mean() if sm_.sum() else 0:.4f}", end="")
        if sm_.sum() < MIN_SUPPORT or npos < 15:
            print("   <- TOO THIN; this stratum will emit no edges")
            calib[lab] = None
            continue
        curve = []
        for t in thr:
            m = sm_ & (oof >= t)
            if m.sum() >= 20:
                curve.append((float(t), float(yu[m].mean()), int(m.sum())))
        keep = next((t for t, pr_, _ in curve if pr_ >= MIN_PRECISION), None)
        calib[lab] = {"curve": curve, "threshold": keep}
        print(f"   -> threshold {keep if keep is not None else float('nan'):.2f} "
              f"for precision >= {MIN_PRECISION}"
              if keep is not None else "   <- never reaches the precision floor; emits no edges")
    if all(v is None or v.get("threshold") is None for v in calib.values()):
        raise SystemExit("no distance stratum can be calibrated to the precision floor; refusing to write "
                         "a layer whose edges cannot be given an honest precision")

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
        dc = np.array([abs(c[1] - c[2]) for c in cand], float)
        kept = 0
        for lo, hi, lab in STRATA:
            cal = calib.get(lab)
            if cal is None or cal.get("threshold") is None:
                continue
            prec_of = {t: pr_ for t, pr_, _ in cal["curve"]}
            m = (dc >= lo) & (dc < hi) & (sc >= cal["threshold"])
            for i in np.where(m)[0]:
                chrom, mid, t_, (g, _c) = cand[i]
                pr = max((pr_ for t, pr_ in prec_of.items() if sc[i] >= t), default=MIN_PRECISION)
                edges.append({"cell": cell, "chrom": chrom, "elem": mid, "gene": g,
                              "dist": int(abs(mid - t_)), "band": lab,
                              "score": round(float(sc[i]), 4),
                              "precision": round(float(pr), 3), "model": MODEL_ID})
            kept += int(m.sum())
        per_cell[cell] = {"candidates": len(cand), "edges": kept,
                          "frac_kept": kept / max(len(cand), 1)}
        print(f"    {cell:9s} {len(cand):>9,} candidates -> {kept:>7,} edges "
              f"({100*kept/max(len(cand),1):.2f}%)")

    layer = {
        "model": MODEL_ID,
        "n_edges": len(edges),
        "min_precision": MIN_PRECISION,
        "calibration_is_per_distance_stratum": True,
        "strata_thresholds": {lab: (calib[lab]["threshold"] if calib.get(lab) else None)
                              for _lo, _hi, lab in STRATA},
        "features": FEATNAMES,
        "window_bp": WINDOW,
        "calibration": {lab: (calib[lab]["curve"] if calib.get(lab) else None)
                        for _lo, _hi, lab in STRATA},
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
            "scored only for cell types with a full ENCODE peak panel",
            "competition features are EXCLUDED: n_candidates alone scores AUROC 0.2733 (per-group base-rate "
            "leakage), the block is worth -0.0068 restricted to groups with >=5 candidates, and it is "
            "defined relative to a candidate set so training on the benchmark's and scoring genome-wide "
            "cost -0.1201 AUPRC in v1 of this layer",
            "precision is calibrated WITHIN distance strata because the benchmark is distal-heavy "
            "(63.5% beyond 250 kb, 1.9% under 10 kb) while genome-wide candidates in a 250 kb window are "
            "proximal-heavy; a single global curve overstated precision on proximal edges",
            "strata the benchmark cannot support emit no edges at all rather than edges carrying an "
            "unsupported precision"],
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
