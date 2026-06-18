"""Cross-organism essential-gene prediction from PROTECTION features.

For 8 organisms (genomes available locally) compute per-gene:
  PROTECTION features:
    leading_strand   (1 if co-oriented with replication, per-contig GC-skew oriC)
    oriC_dist_frac   (0=at origin .. 1=at terminus; per replichore)
    mobile_dist_kb   (distance to nearest transposase/phage/integrase CDS)
    n_paralogs       (in-genome family copies)
    gene_len_aa
    gc               (gene GC fraction)
  CONSERVATION feature (the known strong baseline):
    conservation     (family_frac_essential_leakfree)
  LABEL: essential (0/1)

Then LEAVE-ONE-ORGANISM-OUT: train on 7, predict the held-out org. Compare
three feature sets:
  (A) protection-only   -- do the 'safety' features alone predict essentiality?
  (B) conservation-only -- the known baseline
  (C) all combined      -- does protection ADD anything over conservation?

Pure NumPy logistic regression (sandbox-runnable). Reports AUPRC + AUC per org.

Outputs:
  outputs/orphan/cross_org_protection.csv          -- per-org per-featureset metrics
  outputs/orphan/cross_org_protection.png
  outputs/orphan/cross_org_protection_summary.json
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
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def rc(s): return s.translate(COMP)[::-1]

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
MOB = re.compile(r"transposase|integrase|insertion sequence|IS\d|phage|recombinase|"
                 r"mobile element", re.I)

# ---- labels + orthology (load once, index by org) ----
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
    """oriC = GC-skew cumulative minimum; return position and terminus."""
    n = len(seq)
    skew = []
    for i in range(0, n, win):
        w = seq[i:i+win]; g, c = w.count("G"), w.count("C")
        skew.append((g - c) / max(1, g + c))
    cum = np.cumsum(skew)
    return int(np.argmin(cum)) * win, int(np.argmax(cum)) * win


def build_org_features(org, acc):
    contigs = read_contigs(acc)
    # parse CDS per contig
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
        L = len(seq)
        oriC, terC = oriC_of_contig(seq)
        mob = [ (a+b)//2 for a, b, st, lt, p in lst if MOB.search(p or "") ]
        for s, e, strand, lt, prod in lst:
            if lt not in labels[org]: continue
            o = orth[org].get(lt)
            mid = (s + e) // 2
            # dist to oriC (shorter arc), normalized by half-genome
            d = abs(mid - oriC) % L; d = min(d, L - d)
            oriC_frac = d / (L / 2)
            # leading strand
            delta = (mid - oriC) % L
            leading = int((strand == "+") == (delta < L / 2))
            # mobile distance
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
    e = sum(x["essential"] for x in r)
    print(f"  {org:<26} genes={len(r):<5} essential={e}")
    all_rows += r
df = pd.DataFrame(all_rows)
df["conservation"] = df["conservation"].fillna(0.0)
print(f"\ntotal genes: {len(df):,}  across {df.organism.nunique()} orgs")

# ---- pure-numpy logistic regression + metrics ----
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
    return 1/(1+np.exp(-Xte@w)), w

def auprc(score, y):
    o = np.argsort(-score); ys = np.asarray(y)[o]
    tp = np.cumsum(ys); k = np.arange(1, len(ys)+1)
    prec = tp/k; rec = tp/max(1, ys.sum())
    rec0 = np.concatenate([[0], rec])
    return float(np.sum(prec*np.diff(rec0)))

def auc(score, y):
    y = np.asarray(y); o = np.argsort(score); yr = y[o]
    n1 = y.sum(); n0 = len(y)-n1
    if n1 == 0 or n0 == 0: return float("nan")
    ranks = np.argsort(np.argsort(score)) + 1
    return float((ranks[y==1].sum() - n1*(n1+1)/2) / (n1*n0))

FEATURESETS = {
    "protection_only": ["leading","oriC_frac","mobile_dist_kb","n_paralogs","gene_len_aa","gc"],
    "conservation_only": ["conservation"],
    "all_combined": ["leading","oriC_frac","mobile_dist_kb","n_paralogs","gene_len_aa","gc","conservation"],
}

print("\nLEAVE-ONE-ORGANISM-OUT prediction (AUPRC / AUC):")
results = []
for held in ORGS:
    te = df[df.organism == held]; tr = df[df.organism != held]
    yte = te.essential.to_numpy()
    base = yte.mean()
    row = {"organism": held, "n": len(te), "base_rate": round(base, 3)}
    line = f"  {held:<26} base={base:.2f} | "
    for fs, cols in FEATURESETS.items():
        pred, _ = fit_predict(tr[cols].to_numpy(float), tr.essential.to_numpy(),
                              te[cols].to_numpy(float))
        ap = auprc(pred, yte); ac = auc(pred, yte)
        row[f"{fs}_auprc"] = round(ap, 3); row[f"{fs}_auc"] = round(ac, 3)
        line += f"{fs.split('_')[0]}: AP {ap:.2f}/AUC {ac:.2f}  "
    results.append(row); print(line)

res = pd.DataFrame(results)
print("\n=== MEAN across organisms ===")
for fs in FEATURESETS:
    print(f"  {fs:<20} AUPRC {res[f'{fs}_auprc'].mean():.3f}   "
          f"AUC {res[f'{fs}_auc'].mean():.3f}")

# feature importance: train on ALL, standardized coefficients
cols_all = FEATURESETS["all_combined"]
X = df[cols_all].to_numpy(float); y = df.essential.to_numpy()
_, w = fit_predict(X, y, X)
coef = dict(zip(cols_all, w[1:]))
print("\nstandardized logistic coefficients (trained on all orgs):")
for c, v in sorted(coef.items(), key=lambda x: -abs(x[1])):
    print(f"  {c:<16} {v:+.3f}")

res.to_csv(OUT / "cross_org_protection.csv", index=False)
(OUT / "cross_org_protection_summary.json").write_text(json.dumps({
    "n_orgs": len(ORGS), "n_genes": int(len(df)),
    "mean_auprc": {fs: round(float(res[f"{fs}_auprc"].mean()), 3) for fs in FEATURESETS},
    "mean_auc":   {fs: round(float(res[f"{fs}_auc"].mean()), 3) for fs in FEATURESETS},
    "coefficients": {k: round(float(v), 3) for k, v in coef.items()},
    "per_org": results,
}, indent=2))

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
orgs_short = [o.replace("beril_","") for o in res.organism]
x = np.arange(len(res)); w_ = 0.26
ax = axes[0]
for i, (fs, col) in enumerate([("protection_only","#E67E22"),
                               ("conservation_only","#3498DB"),
                               ("all_combined","#C0392B")]):
    ax.bar(x + (i-1)*w_, res[f"{fs}_auc"], w_, label=fs.replace("_"," "), color=col, alpha=0.88)
ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="random (AUC 0.5)")
ax.set_xticks(x); ax.set_xticklabels(orgs_short, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("AUC (held-out organism)")
ax.set_title("Cross-organism essentiality prediction — AUC by feature set")
ax.legend(fontsize=8); ax.set_ylim(0, 1)

ax = axes[1]
means = [res[f"{fs}_auprc"].mean() for fs in FEATURESETS]
aucs  = [res[f"{fs}_auc"].mean() for fs in FEATURESETS]
labels_fs = ["protection\nonly","conservation\nonly","all\ncombined"]
xb = np.arange(3)
ax.bar(xb-0.2, means, 0.4, label="mean AUPRC", color="#16A085", alpha=0.9)
ax.bar(xb+0.2, aucs, 0.4, label="mean AUC", color="#8E44AD", alpha=0.9)
for i,(m,a) in enumerate(zip(means,aucs)):
    ax.text(i-0.2, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(i+0.2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(xb); ax.set_xticklabels(labels_fs)
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.set_ylabel("mean across 8 organisms"); ax.set_ylim(0,1)
ax.set_title("Mean performance — does protection add to conservation?")
ax.legend(fontsize=9)

fig.suptitle("Predicting essential genes across 8 organisms from protection + conservation features",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT / "cross_org_protection.png", dpi=140, bbox_inches="tight")
print(f"\nwrote cross_org_protection.png + csv + summary")
