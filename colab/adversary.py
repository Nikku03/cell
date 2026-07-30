"""WHAT DOES 0.688 ACTUALLY BEAT? A coordinate-only adversary, within-gene ranking, and a leakage audit.

WHY THIS COMES BEFORE ANY NEW FEATURE. The E-G model reads 0.6881 pooled AUPRC against a published
ENCODE-rE2G figure of 0.66 on the same benchmark. Neither number means much until two questions are answered,
and neither has been:

  1 How much of it is obtainable from COORDINATES ALONE -- chromosome, element position, TSS position,
    distance, gene identity, cell type? No biology, no assay, no ChIP. If a coordinate-only model reaches
    most of 0.688, then almost everything attributed to accessibility, H3K27ac, Pol II and CTCF is a
    restatement of where things sit in the genome.
  2 Does the model answer the question anyone actually has? Global AUPRC pools every pair in the benchmark, so
    it can be won by ordering distance strata correctly. The practical question is per gene: GIVEN this
    promoter and its candidate enhancers, WHICH one regulates it. That is a ranking problem inside a gene, and
    a global average-precision can look strong while the within-gene ranking is near random.

THE LEAKAGE QUESTION, SETTLED RATHER THAN ASSERTED. A reasonable objection is that gene identity leaks: if a
gene appears in training and test, a model can memorise its positive rate, and `missed.py` found
`other_positives_same_gene` to be the strongest single discriminator of missed positives (p 3.6e-19). But a
gene has ONE TSS on ONE chromosome, so chromosome-grouped folds are automatically gene-disjoint -- the
objection is already answered by the existing CV scheme. Rather than assert that, this module demonstrates it:
gene identity is offered as a feature under BOTH random folds and chromosome folds. Under random folds it
should help (that is leakage). Under chromosome folds it should do nothing. If it helps under chromosome
folds, something is wrong with the fold construction and every number this project has reported is suspect.

WHAT THIS MODULE DOES NOT CLAIM. It does not test whether Hi-C helps, whether activity features help, or
whether the model is good. It establishes the FLOOR that any biological claim has to clear, and the metric
that claim should be reported in. The features it uses are deliberately stupid.
"""
import csv
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
CR = SP / "crispr"
HELDOUT = "EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"
TRAINING = "EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz"
SEEDS = (0, 1, 2, 3, 4)


def load():
    def rd(f):
        p = CR / f
        if not p.exists():
            raise SystemExit(f"benchmark absent at {p}; run expand_cells.py first")
        return list(csv.DictReader(gzip.open(p, "rt"), delimiter="\t"))
    tr, ho = rd(TRAINING), rd(HELDOUT)
    for r in tr:
        r["CellType"] = "K562"
    return tr + ho


def truthy(v):
    return v in ("TRUE", "True", "true", "1")


def build(rows):
    """Coordinates and identity only. No assay, no ChIP, no sequence, no biology of any kind."""
    chroms = sorted({r["chrom"] for r in rows})
    ci = {c: i for i, c in enumerate(chroms)}
    cells = sorted({r["CellType"] for r in rows})
    ce = {c: i for i, c in enumerate(cells)}
    genes = sorted({r["measuredGeneSymbol"] for r in rows})
    gidx = {g: i for i, g in enumerate(genes)}
    X, gene_id, y, ch, ct, key = [], [], [], [], [], []
    for r in rows:
        s, e = float(r["chromStart"]), float(r["chromEnd"])
        tss = float(r["startTSS"]) if r.get("startTSS") else np.nan
        mid = 0.5 * (s + e)
        d = abs(mid - tss) if np.isfinite(tss) else np.nan
        X.append([ci[r["chrom"]], mid, tss, e - s,
                  np.log10(max(d, 1.0)) if np.isfinite(d) else -1.0,
                  1.0 if np.isfinite(d) and mid > tss else 0.0,       # element upstream or downstream
                  ce[r["CellType"]]])
        gene_id.append(gidx[r["measuredGeneSymbol"]])
        y.append(1 if truthy(r.get("Significant", "")) else 0)
        ch.append(r["chrom"])
        ct.append(r["CellType"])
        key.append((r["measuredGeneSymbol"], r["CellType"]))
    return (np.array(X), np.array(gene_id, float).reshape(-1, 1), np.array(y),
            np.array(ch), np.array(ct), key,
            ["chrom_idx", "elem_mid", "tss", "elem_width", "log_dist", "is_downstream", "celltype_idx"])


def folds_chrom(ch, seed):
    """Chromosome-grouped: a chromosome lands wholly in one fold, so genes cannot straddle folds."""
    u = sorted(set(ch))
    rng = np.random.default_rng(seed)
    a = rng.permutation(len(u))
    m = {c: a[i] % 5 for i, c in enumerate(u)}
    return np.array([m[c] for c in ch])


def folds_random(n, seed):
    rng = np.random.default_rng(seed)
    return rng.permutation(n) % 5


def oof(X, y, f):
    import xgboost as xgb
    p = np.zeros(len(y))
    for k in range(5):
        tr, te = f != k, f == k
        m = xgb.XGBClassifier(scale_pos_weight=(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1),
                              max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.6, min_child_weight=5, reg_lambda=2.0,
                              eval_metric="aucpr", n_jobs=4)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def within_gene(scores, y, key, min_cand=2):
    """Recall@1, MRR and per-gene AP, macro-averaged over (gene, cell type) groups with a positive.

    This is the metric that matches the question a user has: given one promoter and its candidates, which
    enhancer regulates it. Groups with no positive are excluded -- they cannot contribute a rank -- and the
    number excluded is returned, because silently dropping most of the data would flatter the metric.
    """
    from sklearn.metrics import average_precision_score
    g = {}
    for i, k in enumerate(key):
        g.setdefault(k, []).append(i)
    r1, mrr, aps, used, skipped = [], [], [], 0, 0
    for k, idx in g.items():
        idx = np.array(idx)
        if len(idx) < min_cand or y[idx].sum() == 0:
            skipped += 1
            continue
        used += 1
        order = idx[np.argsort(-scores[idx])]
        yy = y[order]
        r1.append(float(yy[0]))
        mrr.append(1.0 / (int(np.argmax(yy == 1)) + 1))
        aps.append(average_precision_score(y[idx], scores[idx]))
    return {"n_groups_used": used, "n_groups_skipped": skipped,
            "recall_at_1": float(np.mean(r1)) if r1 else np.nan,
            "mrr": float(np.mean(mrr)) if mrr else np.nan,
            "macro_ap": float(np.mean(aps)) if aps else np.nan}


def main():
    from sklearn.metrics import average_precision_score
    print("=" * 100)
    print("WHAT DOES THE MODEL ACTUALLY BEAT? -- coordinate-only adversary + within-gene ranking")
    print("=" * 100)
    rows = load()
    X, gid, y, ch, ct, key, names = build(rows)
    print(f"  {len(y):,} pairs, {int(y.sum())} positives, base rate {y.mean():.4f}")
    print(f"  cell types: " + ", ".join(f"{c} ({int(y[ct == c].sum())}+/{int((ct == c).sum())})"
                                        for c in sorted(set(ct))))
    print(f"  coordinate-only features: {names}")
    ng = len({k for k in key})
    print(f"  {ng:,} (gene, cell type) groups")

    R = {"n_pairs": int(len(y)), "n_pos": int(y.sum()), "base_rate": float(y.mean()),
         "features": names, "n_groups": ng}

    ARMS = {
        "distance only": X[:, [4]],
        "coordinates only": X,
        "coordinates + gene identity": np.hstack([X, gid]),
    }

    for scheme in ("chromosome-grouped", "random"):
        print(f"\n  {scheme.upper()} FOLDS")
        print(f"    {'arm':30s} {'pooled AUPRC':>13s} {'Recall@1':>9s} {'MRR':>7s} {'macro AP':>9s}")
        R[scheme] = {}
        for tag, M in ARMS.items():
            ap, rk = [], []
            for s in SEEDS:
                f = folds_chrom(ch, s) if scheme == "chromosome-grouped" else folds_random(len(y), s)
                p = oof(M, y, f)
                ap.append(average_precision_score(y, p))
                rk.append(within_gene(p, y, key))
            mean = {k: float(np.mean([r[k] for r in rk])) for k in
                    ("recall_at_1", "mrr", "macro_ap")}
            R[scheme][tag] = {"auprc": float(np.mean(ap)), "auprc_sd": float(np.std(ap)), **mean,
                              "n_groups_used": rk[0]["n_groups_used"],
                              "n_groups_skipped": rk[0]["n_groups_skipped"]}
            print(f"    {tag:30s} {np.mean(ap):13.4f} {mean['recall_at_1']:9.4f} "
                  f"{mean['mrr']:7.4f} {mean['macro_ap']:9.4f}")
        gl = R[scheme]["coordinates + gene identity"]["auprc"] - R[scheme]["coordinates only"]["auprc"]
        R[scheme]["gene_identity_gain"] = float(gl)
        print(f"    gene identity adds {gl:+.4f}")

    # ---- the leakage verdict ----
    lc = R["chromosome-grouped"]["gene_identity_gain"]
    lr = R["random"]["gene_identity_gain"]
    print("\n  LEAKAGE AUDIT -- does gene identity buy anything it should not?")
    print(f"    under random folds        {lr:+.4f}   <- a gene can appear in train and test; "
          f"memorising its positive rate is possible")
    print(f"    under chromosome folds    {lc:+.4f}   <- a gene has one TSS on one chromosome, so folds "
          f"are gene-disjoint by construction")
    if lc > 0.01:
        print("    ** WARNING: gene identity helps even under chromosome folds. The folds are not "
              "gene-disjoint and every number this project reports is suspect.")
        R["leakage_verdict"] = "FAIL -- chromosome folds are not gene-disjoint"
    else:
        R["leakage_verdict"] = (f"PASS -- gene identity is worth {lc:+.4f} under chromosome folds "
                                f"against {lr:+.4f} under random folds, so the existing CV scheme "
                                f"already blocks gene-level memorisation")
    print(f"    -> {R['leakage_verdict']}")

    # ---- the floor ----
    floor = R["chromosome-grouped"]["coordinates only"]
    dist = R["chromosome-grouped"]["distance only"]
    print("\n" + "=" * 100)
    print(f"  THE FLOOR ANY BIOLOGICAL CLAIM MUST CLEAR (chromosome folds, no assay data at all)")
    print(f"    distance alone     AUPRC {dist['auprc']:.4f}   Recall@1 {dist['recall_at_1']:.4f}   "
          f"MRR {dist['mrr']:.4f}")
    print(f"    all coordinates    AUPRC {floor['auprc']:.4f}   Recall@1 {floor['recall_at_1']:.4f}   "
          f"MRR {floor['mrr']:.4f}")
    print(f"    within-gene metrics use {floor['n_groups_used']:,} groups and skip "
          f"{floor['n_groups_skipped']:,} with no positive or <2 candidates")
    print(f"\n  The published comparison is 0.66 and this project reads 0.6881 POOLED. Against a "
          f"coordinate-only floor of {floor['auprc']:.4f}, the biology is worth "
          f"{0.6881 - floor['auprc']:+.4f} of pooled AUPRC.")
    print("  Report the within-gene numbers alongside it: pooled AUPRC can be won by ordering distance "
          "strata, which is not the question anyone asks of an E-G model.")
    R["floor"] = {"auprc": floor["auprc"], "recall_at_1": floor["recall_at_1"], "mrr": floor["mrr"],
                  "biology_headroom_vs_0.6881": float(0.6881 - floor["auprc"])}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "adversary.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'adversary.json'}")


if __name__ == "__main__":
    main()
