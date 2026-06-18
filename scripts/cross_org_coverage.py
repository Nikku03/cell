"""Risk-coverage / precision-at-coverage for the cross-organism predictor.

Mirrors the LOO setup from cross_org_protection.py but instead of just AUPRC/AUC,
ranks held-out predictions and asks:
  - at what COVERAGE (top X% of confident calls) does precision cross 0.5 / 0.7 / 0.9 ?
  - precision and recall vs coverage (the AlphaFold abstention curve)
  - bacteria-pool vs archaea-pool separately (we already saw archaea collapse)

Per feature set: conservation_only and all_combined (protection_only is too weak).

Outputs:
  outputs/orphan/cross_org_coverage.png
  outputs/orphan/cross_org_coverage_summary.json
  outputs/orphan/cross_org_coverage_scores.csv  -- per-gene held-out score
"""
import re, json, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")

ORGS = {
    "beril_RalstoniaGMI1000": "GCF_000009125.1",
    "beril_RalstoniaBSBF1503":"GCF_001587155.1",
    "beril_RalstoniaPSI07":   "GCF_000283475.1",
    "beril_Dda3937":          "GCF_000147055.1",
    "beril_HerbieS":          "GCF_000143225.1",
    "beril_Magneto":          "GCF_000009985.1",
    "beril_Methanococcus_JJ": "GCF_002945325.1",
    "beril_Methanococcus_S2": "GCF_000011585.1",
}
ARCHAEA = {"beril_Methanococcus_JJ", "beril_Methanococcus_S2"}
MOB = re.compile(r"transposase|integrase|insertion sequence|IS\d|phage|recombinase|"
                 r"mobile element", re.I)

labels = defaultdict(dict)
for r in csv.DictReader(open(DATA / "labels" / "labels.csv")):
    labels[r["organism"]][r["locus_tag"]] = int(r["essential"])
orth = defaultdict(dict)
for r in csv.DictReader(open(DATA / "labels" / "orthology_features.csv")):
    orth[r["organism"]][r["locus_tag"]] = r


def read_contigs(acc):
    d = {}; cur = None; buf = []
    for line in (DATA / "genome_cache" / acc / "genome.fna").read_text().splitlines():
        if line.startswith(">"):
            if cur: d[cur] = "".join(buf).upper()
            cur = line[1:].split()[0]; buf = []
        else: buf.append(line.strip())
    if cur: d[cur] = "".join(buf).upper()
    return d


def oriC_of_contig(seq, win=10000):
    n = len(seq); skew = []
    for i in range(0, n, win):
        w = seq[i:i+win]; g, c = w.count("G"), w.count("C")
        skew.append((g - c) / max(1, g + c))
    cum = np.cumsum(skew)
    return int(np.argmin(cum)) * win, int(np.argmax(cum)) * win


def build_org_features(org, acc):
    contigs = read_contigs(acc)
    cds = defaultdict(list)
    for line in (DATA / "genome_cache" / acc / "genomic.gff").read_text().splitlines():
        if "\tCDS\t" not in line: continue
        c = line.split("\t")
        m = re.search(r"locus_tag=([^;\n]+)", c[8])
        if not m: continue
        prod = re.search(r"product=([^;\n]+)", c[8])
        cds[c[0]].append((int(c[3]), int(c[4]), c[6], m.group(1),
                          prod.group(1) if prod else ""))
    rows = []
    for contig, lst in cds.items():
        seq = contigs.get(contig)
        if not seq or len(seq) < 20000: continue
        L = len(seq); oriC, terC = oriC_of_contig(seq)
        mob = [(a+b)//2 for a, b, st, lt, p in lst if MOB.search(p or "")]
        for s, e, strand, lt, prod in lst:
            if lt not in labels[org]: continue
            o = orth[org].get(lt)
            mid = (s + e) // 2
            d = abs(mid - oriC) % L; d = min(d, L - d)
            oriC_frac = d / (L / 2)
            delta = (mid - oriC) % L
            leading = int((strand == "+") == (delta < L / 2))
            if mob:
                md = min(min(abs(mid-mp), L-abs(mid-mp)) for mp in mob) / 1e3
            else:
                md = 500.0
            gene = seq[s-1:e]
            gc = (gene.count("G") + gene.count("C")) / max(1, len(gene))
            cons = float(o["family_frac_essential_leakfree"]) if o and o["family_frac_essential_leakfree"] not in ("","nan") else np.nan
            npar = float(o["n_paralogs_in_genome"]) if o and o["n_paralogs_in_genome"] not in ("",) else 0.0
            rows.append(dict(
                organism=org, locus_tag=lt, essential=labels[org][lt],
                leading=leading, oriC_frac=round(oriC_frac, 4),
                mobile_dist_kb=round(min(md, 500), 2), n_paralogs=npar,
                gene_len_aa=(e - s + 1)//3, gc=round(gc, 4),
                conservation=cons,
            ))
    return rows


print("building features per organism ...")
all_rows = []
for org, acc in ORGS.items():
    r = build_org_features(org, acc)
    all_rows += r
    print(f"  {org:<26} genes={len(r):<5} essential={sum(x['essential'] for x in r)}")
df = pd.DataFrame(all_rows)
df["conservation"] = df["conservation"].fillna(0.0)
print(f"total: {len(df):,} genes\n")


def fit_predict(Xtr, ytr, Xte, iters=500, lr=0.5, l2=2.0):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1
    Xtr = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]
    Xte = np.c_[np.ones(len(Xte)), (Xte-mu)/sd]
    w = np.zeros(Xtr.shape[1]); ytr = ytr.astype(float)
    pos = ytr.mean() or 1e-6
    cw = np.where(ytr == 1, 0.5/pos, 0.5/(1-pos))
    for _ in range(iters):
        p = 1/(1+np.exp(-Xtr@w))
        g = Xtr.T@((p-ytr)*cw)/len(Xtr); g[1:] += l2*w[1:]/len(Xtr)
        w -= lr*g
    return 1/(1+np.exp(-Xte@w))


FEATURESETS = {
    "conservation_only": ["conservation"],
    "all_combined": ["leading","oriC_frac","mobile_dist_kb","n_paralogs",
                     "gene_len_aa","gc","conservation"],
}

print("LOO scoring (held-out predictions) ...")
scores = {fs: np.zeros(len(df)) for fs in FEATURESETS}
df = df.reset_index(drop=True)
for held in ORGS:
    te_mask = (df.organism == held).to_numpy()
    tr = df[~te_mask]; te = df[te_mask]
    for fs, cols in FEATURESETS.items():
        s = fit_predict(tr[cols].to_numpy(float), tr.essential.to_numpy(),
                        te[cols].to_numpy(float))
        scores[fs][te_mask] = s

for fs in FEATURESETS:
    df[f"score_{fs}"] = scores[fs]

df.to_csv(OUT / "cross_org_coverage_scores.csv", index=False)


def coverage_curve(score, y, cov_grid):
    """Rank-by-score-desc, then for each coverage fraction k, report
    precision (TP/k) and recall (TP/total_pos)."""
    o = np.argsort(-score)
    ys = np.asarray(y)[o]
    n = len(ys); pos = ys.sum()
    out = []
    for c in cov_grid:
        k = max(1, int(round(c * n)))
        tp = ys[:k].sum()
        out.append((c, k, tp / k, tp / max(1, pos)))
    return out  # list of (coverage, k, precision, recall)


def coverage_at_precision(score, y, thresholds=(0.5, 0.7, 0.9)):
    """For each precision threshold, find the max coverage where precision >= t.
    Returns dict {t: (coverage, recall)} or None if never reached."""
    o = np.argsort(-score)
    ys = np.asarray(y)[o].astype(float)
    n = len(ys); pos = ys.sum() or 1
    tp = np.cumsum(ys); k = np.arange(1, n+1)
    prec = tp / k; rec = tp / pos; cov = k / n
    out = {}
    for t in thresholds:
        ok = prec >= t
        # require at least 20 calls so we are not measuring noise
        ok = ok & (k >= 20)
        if not ok.any():
            out[t] = None
        else:
            i = np.where(ok)[0].max()
            out[t] = (float(cov[i]), float(rec[i]), int(k[i]))
    return out


cov_grid = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                     0.60, 0.70, 0.80, 0.90, 1.00])

# Per-org and pooled (bacteria vs archaea) curves
POOLS = {
    "bacteria_pool": [o for o in ORGS if o not in ARCHAEA],
    "archaea_pool":  list(ARCHAEA),
    "all_pool":      list(ORGS),
}

summary = {"per_org": {}, "pools": {}, "featuresets": list(FEATURESETS)}

print("\n=== per-organism precision @ coverage (all_combined) ===")
print(f"  {'organism':<26} base | top10% top25% top50% | cov@P=0.7  cov@P=0.9")
for org in ORGS:
    m = (df.organism == org).to_numpy()
    y = df.essential[m].to_numpy(); base = float(y.mean())
    s = df["score_all_combined"][m].to_numpy()
    curve = coverage_curve(s, y, cov_grid)
    cap = coverage_at_precision(s, y)
    p10 = next(p for c,k,p,r in curve if abs(c-0.10) < 1e-6)
    p25 = next(p for c,k,p,r in curve if abs(c-0.25) < 1e-6)
    p50 = next(p for c,k,p,r in curve if abs(c-0.50) < 1e-6)
    c7 = cap[0.7]; c9 = cap[0.9]
    f7 = f"{c7[0]*100:5.1f}% (R={c7[1]:.2f})" if c7 else "  never"
    f9 = f"{c9[0]*100:5.1f}% (R={c9[1]:.2f})" if c9 else "  never"
    print(f"  {org:<26} {base:.2f} | {p10:.2f}  {p25:.2f}  {p50:.2f} | {f7}  {f9}")
    summary["per_org"][org] = {
        "base_rate": round(base, 3),
        "curve_all_combined": [(round(c,2), int(k), round(p,3), round(r,3))
                               for c,k,p,r in curve],
        "cov_at_precision_all_combined": {str(t): cap[t] for t in cap},
    }

print("\n=== pooled precision @ coverage ===")
print(f"  {'pool':<16} {'fs':<20} | top10% top25% top50% | cov@P=0.7   cov@P=0.9")
for pool_name, members in POOLS.items():
    summary["pools"][pool_name] = {}
    m = df.organism.isin(members).to_numpy()
    y = df.essential[m].to_numpy(); base = float(y.mean())
    for fs in FEATURESETS:
        s = df[f"score_{fs}"][m].to_numpy()
        curve = coverage_curve(s, y, cov_grid)
        cap = coverage_at_precision(s, y, (0.5, 0.7, 0.9))
        p10 = next(p for c,k,p,r in curve if abs(c-0.10) < 1e-6)
        p25 = next(p for c,k,p,r in curve if abs(c-0.25) < 1e-6)
        p50 = next(p for c,k,p,r in curve if abs(c-0.50) < 1e-6)
        c7 = cap[0.7]; c9 = cap[0.9]
        f7 = f"{c7[0]*100:5.1f}% (R={c7[1]:.2f})" if c7 else "  never  "
        f9 = f"{c9[0]*100:5.1f}% (R={c9[1]:.2f})" if c9 else "  never  "
        print(f"  {pool_name:<16} {fs:<20} | {p10:.2f}  {p25:.2f}  {p50:.2f} | {f7}  {f9}")
        summary["pools"][pool_name][fs] = {
            "base_rate": round(base, 3),
            "n": int(m.sum()),
            "curve": [(round(c,2), int(k), round(p,3), round(r,3))
                      for c,k,p,r in curve],
            "cov_at_precision": {str(t): cap[t] for t in cap},
        }

(OUT / "cross_org_coverage_summary.json").write_text(json.dumps(summary, indent=2))


# ---- Figure ----
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Panel A: precision-vs-coverage curve, per-org (all_combined)
ax = axes[0, 0]
cmap_bac = plt.cm.Blues(np.linspace(0.4, 0.9, len([o for o in ORGS if o not in ARCHAEA])))
cmap_arc = plt.cm.Oranges(np.linspace(0.5, 0.9, len(ARCHAEA)))
ib = ia = 0
for org in ORGS:
    m = (df.organism == org).to_numpy()
    y = df.essential[m].to_numpy()
    s = df["score_all_combined"][m].to_numpy()
    o = np.argsort(-s); ys = y[o].astype(float)
    k = np.arange(1, len(ys)+1)
    prec = np.cumsum(ys) / k; cov = k / len(ys)
    base = y.mean()
    if org in ARCHAEA:
        c = cmap_arc[ia]; ia += 1; ls = "--"
    else:
        c = cmap_bac[ib]; ib += 1; ls = "-"
    ax.plot(cov, prec, color=c, lw=1.6, ls=ls,
            label=org.replace("beril_",""))
    ax.axhline(base, color=c, lw=0.6, alpha=0.25)
ax.axhline(0.7, color="k", ls=":", lw=0.8); ax.axhline(0.9, color="k", ls=":", lw=0.8)
ax.set_xlabel("coverage (fraction reported, sorted by confidence)")
ax.set_ylabel("precision at coverage")
ax.set_title("A. Precision vs coverage per organism (all_combined)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(fontsize=7, ncol=2, loc="lower left")
ax.grid(alpha=0.3)

# Panel B: bacteria pool vs archaea pool, both featuresets
ax = axes[0, 1]
colors = {"conservation_only": "#3498DB", "all_combined": "#C0392B"}
for pool_name, members in [("bacteria_pool", POOLS["bacteria_pool"]),
                            ("archaea_pool", POOLS["archaea_pool"])]:
    m = df.organism.isin(members).to_numpy()
    y = df.essential[m].to_numpy(); base = y.mean()
    for fs in FEATURESETS:
        s = df[f"score_{fs}"][m].to_numpy()
        o = np.argsort(-s); ys = y[o].astype(float)
        k = np.arange(1, len(ys)+1)
        prec = np.cumsum(ys) / k; cov = k / len(ys)
        ls = "-" if pool_name == "bacteria_pool" else "--"
        ax.plot(cov, prec, color=colors[fs], lw=2.0, ls=ls,
                label=f"{pool_name.split('_')[0]} | {fs}")
    ax.axhline(base, color="gray" if pool_name=="bacteria_pool" else "gray",
               lw=0.6, alpha=0.25 if pool_name=="bacteria_pool" else 0.5)
ax.axhline(0.7, color="k", ls=":", lw=0.8)
ax.axhline(0.9, color="k", ls=":", lw=0.8)
ax.set_xlabel("coverage"); ax.set_ylabel("precision")
ax.set_title("B. Bacteria vs archaea pool (solid=bacteria, dashed=archaea)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)

# Panel C: recall vs coverage (bacteria pool, both featuresets)
ax = axes[1, 0]
for pool_name, members, ls in [("bacteria", POOLS["bacteria_pool"], "-"),
                                 ("archaea",  POOLS["archaea_pool"],  "--")]:
    m = df.organism.isin(members).to_numpy()
    y = df.essential[m].to_numpy(); pos = y.sum() or 1
    for fs in FEATURESETS:
        s = df[f"score_{fs}"][m].to_numpy()
        o = np.argsort(-s); ys = y[o].astype(float)
        cov = np.arange(1, len(ys)+1) / len(ys)
        rec = np.cumsum(ys) / pos
        ax.plot(cov, rec, color=colors[fs], lw=2.0, ls=ls,
                label=f"{pool_name} | {fs}")
ax.plot([0,1],[0,1], "k:", lw=0.6, label="random")
ax.set_xlabel("coverage"); ax.set_ylabel("recall (of true essentials)")
ax.set_title("C. Recall vs coverage")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

# Panel D: bar — coverage achievable at precision 0.7 / 0.9 per org
ax = axes[1, 1]
orgs_short = [o.replace("beril_","") for o in ORGS]
cov7 = []; cov9 = []
for org in ORGS:
    m = (df.organism == org).to_numpy()
    y = df.essential[m].to_numpy()
    s = df["score_all_combined"][m].to_numpy()
    cap = coverage_at_precision(s, y, (0.7, 0.9))
    cov7.append(cap[0.7][0]*100 if cap[0.7] else 0)
    cov9.append(cap[0.9][0]*100 if cap[0.9] else 0)
xb = np.arange(len(ORGS)); w_ = 0.4
ax.bar(xb-w_/2, cov7, w_, label="coverage @ precision ≥ 0.7", color="#16A085", alpha=0.9)
ax.bar(xb+w_/2, cov9, w_, label="coverage @ precision ≥ 0.9", color="#8E44AD", alpha=0.9)
for i,(c7,c9) in enumerate(zip(cov7, cov9)):
    if c7>0: ax.text(i-w_/2, c7, f"{c7:.0f}%", ha="center", va="bottom", fontsize=7)
    if c9>0: ax.text(i+w_/2, c9, f"{c9:.0f}%", ha="center", va="bottom", fontsize=7)
ax.set_xticks(xb); ax.set_xticklabels(orgs_short, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("coverage (% of genes reportable)")
ax.set_title("D. Coverage at which precision crosses 0.7 / 0.9 (all_combined)")
ax.legend(fontsize=8); ax.set_ylim(0, 60); ax.grid(alpha=0.3, axis="y")

fig.suptitle("Risk-coverage analysis: at what coverage can the cross-org predictor hit high precision?",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT / "cross_org_coverage.png", dpi=140, bbox_inches="tight")
print(f"\nwrote cross_org_coverage.png + summary + scores csv")
