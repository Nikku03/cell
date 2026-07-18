"""halflife — the mRNA-decay layer (A4), tested HONESTLY against measured half-lives. See FIRST_PRINCIPLES.md sec 3.

Predict per-gene mRNA decay rate ln(k_deg) in K562 from SEQUENCE features (the log-additive model: codon composition ->
codon optimality, 3'UTR length / ARE / GC / m6A, 5'UTR length / uAUG), and measure held-out R^2 / Spearman against real
SLAM-seq/TimeLapse-seq half-lives. This is the layer that turns a transcription-RATE change into a steady-state mRNA LEVEL
([mRNA]_ss = k_txn / k_deg), and it is the first of the post-transcriptional layers where LOSING IS POSSIBLE -- the published
sequence-only ceiling is R^2 ~ 0.5-0.6 (Saluki, Agarwal & Kelley 2022) against denoised consensus data.

GROUND TRUTH: RNAdecayCafe (Zenodo 15785218), uniformly-reprocessed pulse-label NR-seq half-lives; K562 (our cell line),
avg_log_kdeg per gene. FEATURES: GENCODE v46 pc_transcripts (headers give CDS/UTR3/UTR5 coords); longest-CDS transcript per
gene. Codon optimality is learned IN-FOLD (61-codon composition as features) so there is no CSC leakage. CV is GroupKFold by
chromosome (paralog-safe). Reported vs baselines (mean / 3'UTR-length-only / codon-only) and the Saluki ceiling.
"""
import gzip, json, re, subprocess
from pathlib import Path
import numpy as np
OUT = Path("outputs/orphan")
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/rnadecay")
GENCODE = SCR / "pc.fa.gz"
KDEG = SCR / "AvgKdegs.csv"
GENCODE_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.pc_transcripts.fa.gz"
KDEG_URL = "https://zenodo.org/api/records/15785218/files/AvgKdegs_genes_v1.csv/content"

BASES = "ACGT"
CODONS = [a + b + c for a in BASES for b in BASES for c in BASES]
SENSE = [c for c in CODONS if c not in ("TAA", "TAG", "TGA")]      # 61 sense codons
DRACH = re.compile(r"(?=[AGT][AG]AC[ACT])")                        # m6A consensus


def _ensure():
    if not GENCODE.exists():
        SCR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-s", "-L", GENCODE_URL, "-o", str(GENCODE)], check=True)
    if not KDEG.exists():
        subprocess.run(["curl", "-s", "-L", KDEG_URL, "-o", str(KDEG)], check=True)


def parse_gencode():
    """longest-CDS transcript per gene -> {gene: (cds_seq, utr3_seq, utr5_seq)}."""
    best = {}
    name = None; parts = None; seq = []
    def flush():
        if not parts:
            return
        # header fields: ENST|ENSG|OTT|OTT|txname|GENE|len|(UTR5:a-b)|CDS:a-b|(UTR3:a-b)
        gene = parts[5]
        cds = utr3 = utr5 = None
        for f in parts:
            if f.startswith("CDS:"):
                a, b = f[4:].split("-"); cds = (int(a), int(b))
            elif f.startswith("UTR3:"):
                a, b = f[5:].split("-"); utr3 = (int(a), int(b))
            elif f.startswith("UTR5:"):
                a, b = f[5:].split("-"); utr5 = (int(a), int(b))
        if cds is None:
            return
        s = "".join(seq)
        clen = cds[1] - cds[0] + 1
        if gene not in best or clen > best[gene][0]:
            cds_s = s[cds[0] - 1:cds[1]]
            utr3_s = s[utr3[0] - 1:utr3[1]] if utr3 else ""
            utr5_s = s[utr5[0] - 1:utr5[1]] if utr5 else ""
            best[gene] = (clen, cds_s, utr3_s, utr5_s)
    with gzip.open(GENCODE, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                parts = line[1:].strip().split("|"); seq = []
            else:
                seq.append(line.strip())
        flush()
    return {g: (c, u3, u5) for g, (cl, c, u3, u5) in best.items()}


def features(cds, utr3, utr5):
    """log-additive sequence features (FIRST_PRINCIPLES.md sec3)."""
    f = {}
    # codon composition (61-dim frequencies) -> codon optimality is learned in-fold
    n = len(cds) // 3
    counts = {c: 0 for c in SENSE}
    for i in range(0, n * 3, 3):
        cod = cds[i:i + 3]
        if cod in counts:
            counts[cod] += 1
    tot = max(sum(counts.values()), 1)
    for c in SENSE:
        f["cod_" + c] = counts[c] / tot
    f["log_cds_len"] = np.log10(max(len(cds), 1))
    f["log_utr3_len"] = np.log10(max(len(utr3), 1))
    f["log_utr5_len"] = np.log10(max(len(utr5), 1))
    L3 = max(len(utr3), 1)
    f["utr3_are_per_kb"] = 1000.0 * len(re.findall("(?=ATTTA)", utr3)) / L3
    f["utr3_gc"] = (utr3.count("G") + utr3.count("C")) / L3 if utr3 else 0.0
    f["utr3_m6a_per_kb"] = 1000.0 * len(DRACH.findall(utr3)) / L3
    f["cds_gc"] = (cds.count("G") + cds.count("C")) / max(len(cds), 1)
    f["utr5_uaug"] = utr5.count("ATG")                                 # uORF proxy
    return f


def build():
    _ensure()
    tx = parse_gencode()
    # K562 target
    import csv
    y = {}
    with open(KDEG) as fh:
        for r in csv.DictReader(fh):
            if r["cell_line"] == "K562":
                try:
                    v = float(r["avg_log_kdeg"]); rpkm = float(r["avg_RPKM_total"])
                except ValueError:
                    continue
                if np.isfinite(v) and rpkm > 0.5:                       # expressed genes only (cleaner truth)
                    y[r["feature_ID"]] = v
    # chrom for GroupKFold
    D = json.load(open(OUT / "cell_complete.json"))
    chrom = {g["name"]: g.get("chrom", "chrNA") for g in D["genes"]}
    genes = sorted(set(tx) & set(y))
    rows = [features(*tx[g]) for g in genes]
    cols = list(rows[0].keys())
    X = np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)
    yv = np.array([y[g] for g in genes])
    grp = np.array([chrom.get(g, "chrNA") for g in genes])
    return genes, cols, X, yv, grp


def _ridge_cv(X, y, grp, alpha=10.0, cols=None, feat_mask=None):
    """GroupKFold-by-chromosome ridge; returns pooled out-of-fold predictions."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    if feat_mask is not None:
        X = X[:, feat_mask]
    pred = np.zeros_like(y)
    uniq = np.unique(grp)
    k = min(5, len(uniq))
    gkf = GroupKFold(n_splits=k)
    for tr, te in gkf.split(X, y, grp):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return pred


def _xgb_cv(X, y, grp):
    """same features, gradient-boosted trees (non-linear) -> what ceiling do these first-principles features reach?"""
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    pred = np.zeros_like(y)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(grp))))
    for tr, te in gkf.split(X, y, grp):
        m = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.7, n_jobs=4)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def _metrics(y, p):
    from scipy.stats import spearmanr, pearsonr
    ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return {"R2": round(1 - ss_res / ss_tot, 4), "pearson_r": round(float(pearsonr(y, p)[0]), 4),
            "spearman_r": round(float(spearmanr(y, p)[0]), 4)}


def run():
    genes, cols, X, y, grp = build()
    codon_idx = [i for i, c in enumerate(cols) if c.startswith("cod_")]
    utr3len_idx = [cols.index("log_utr3_len")]
    seq_idx = [i for i, c in enumerate(cols) if not c.startswith("cod_")]     # non-codon seq features
    res = {"n_genes": len(genes), "n_features": len(cols),
           "target": "K562 avg_log_kdeg (RNAdecayCafe, uniformly-reprocessed SLAM/TimeLapse-seq)"}
    res["full_model_ridge"] = _metrics(y, _ridge_cv(X, y, grp, cols=cols))
    res["baseline_utr3_length_only"] = _metrics(y, _ridge_cv(X, y, grp, feat_mask=utr3len_idx))
    res["baseline_codon_only"] = _metrics(y, _ridge_cv(X, y, grp, feat_mask=codon_idx))
    res["baseline_seqfeat_no_codon"] = _metrics(y, _ridge_cv(X, y, grp, feat_mask=seq_idx))
    try:
        res["full_model_xgboost"] = _metrics(y, _xgb_cv(X, y, grp))
    except Exception as e:
        res["full_model_xgboost"] = {"error": str(e)[:80]}
    # feature directions (fit on all, standardized) for interpretation
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X); m = Ridge(alpha=10.0).fit(sc.transform(X), y)
    order = np.argsort(-np.abs(m.coef_))
    res["top_features"] = [(cols[i], round(float(m.coef_[i]), 3)) for i in order[:12] if not cols[i].startswith("cod_")][:8]
    return res


def main():
    print("=" * 96)
    print("HALF-LIFE (A4) — predict measured mRNA decay ln(k_deg) in K562 from sequence, held-out")
    print("=" * 96)
    r = run()
    print(f"\n  {r['n_genes']} K562 genes (expressed), {r['n_features']} sequence features. Chromosome-GroupKFold, ridge.")
    print(f"  ground truth: {r['target']}")
    print(f"\n  {'model':32s} {'R2':>8s} {'pearson':>8s} {'spearman':>9s}")
    for k in ["baseline_utr3_length_only", "baseline_codon_only", "baseline_seqfeat_no_codon", "full_model_ridge", "full_model_xgboost"]:
        m = r.get(k, {})
        if "R2" in m:
            print(f"  {k:32s} {m['R2']:8.3f} {m['pearson_r']:8.3f} {m['spearman_r']:9.3f}")
    print(f"\n  top non-codon features (standardized ridge coef; + = faster decay/shorter t-half):")
    for f, c in r["top_features"]:
        print(f"    {f:20s} {c:+.3f}")
    print(f"\n  HONEST CONTEXT: sequence-only ceiling is R2 ~ 0.5-0.6 (Saluki deep CNN+RNN on full sequence, denoised consensus);")
    print(f"  their interpretable Lasso 'genetic features' ~ r 0.67. This is a linear model on hand-built first-principles")
    print(f"  features -> a floor, not the ceiling. R2>0 held-out = the decay layer genuinely predicts measured half-life.")
    r["ceiling_note"] = ("sequence-only ceiling R2~0.5-0.6 (Saluki, Agarwal & Kelley 2022, deep CNN+RNN, denoised consensus); "
                         "interpretable Lasso ~r0.67. This linear first-principles model is a floor. Absolute half-life needs "
                         "measurement; sequence predicts ranking/direction. Ground truth RNAdecayCafe K562 (Zenodo 15785218).")
    json.dump(r, open(OUT / "halflife.json", "w"), indent=1)
    print("\n  -> outputs/orphan/halflife.json")
    return r


if __name__ == "__main__":
    main()
