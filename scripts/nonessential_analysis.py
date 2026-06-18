"""Flip the lens: predict NON-ESSENTIAL genes. Everything we did, mirrored.

Predicting the majority class is a different problem:
  - base rate flips to ~0.72, so high-precision calls are reachable at high
    coverage -- the AlphaFold-paradigm win we could NOT get for essentials.
  - the features flip sign: what POSITIVELY marks a gene as safe-to-lose?
  - the hard zone inverts: for essentials it was the ROGUE zone (essential but
    not conserved). For non-essentials it is the BUFFERED zone (conserved /
    looks-essential but actually dispensable -- redundancy-protected).

Arenas (6 bacteria + 2 archaea, from cross_org_coverage_scores.csv):
  A  RISK-COVERAGE for non-essential, LOO -- precision@coverage, coverage@P0.9/0.95
  B  MINIMAL MODEL -- conservation + paralogs + length predict non-ess?
                       (and the flipped standardized coefficients)
  C  BUFFERED hard zone -- conserved (cons>=0.5) but non-essential genes:
                       how many, are they paralog-rich (redundancy explanation)?
  D  per-organism coverage@P0.9 (the high-coverage win) bacteria vs archaea

Outputs:
  outputs/orphan/nonessential_analysis.png
  outputs/orphan/nonessential_buffered_GMI1000.csv
  outputs/orphan/nonessential_analysis_summary.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/user/cell/outputs/orphan")
ARCHAEA = {"beril_Methanococcus_JJ", "beril_Methanococcus_S2"}

df = pd.read_csv(OUT / "cross_org_coverage_scores.csv")
df["noness"] = 1 - df.essential
bac = df[~df.organism.isin(ARCHAEA)].reset_index(drop=True)
BACT = list(dict.fromkeys(bac.organism))
ALLORGS = list(dict.fromkeys(df.organism))

FEATS_ALL = ["leading","oriC_frac","mobile_dist_kb","n_paralogs","gene_len_aa","gc","conservation"]
FEATS_3   = ["conservation","n_paralogs","gene_len_aa"]


def fit_predict(Xtr, ytr, Xte, it=600, lr=0.5, l2=2.0, ret_w=False):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1
    Xtr = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]
    Xte = np.c_[np.ones(len(Xte)), (Xte-mu)/sd]
    w = np.zeros(Xtr.shape[1]); ytr = ytr.astype(float)
    pos = ytr.mean() or 1e-6; cw = np.where(ytr == 1, 0.5/pos, 0.5/(1-pos))
    for _ in range(it):
        p = 1/(1+np.exp(-Xtr@w)); g = Xtr.T@((p-ytr)*cw)/len(Xtr)
        g[1:] += l2*w[1:]/len(Xtr); w -= lr*g
    pred = 1/(1+np.exp(-Xte@w))
    return (pred, w) if ret_w else pred


def auc(score, y):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y)-n1
    if n1 == 0 or n0 == 0: return float("nan")
    ranks = np.argsort(np.argsort(score)) + 1
    return float((ranks[y==1].sum() - n1*(n1+1)/2)/(n1*n0))

def auprc(score, y):
    o = np.argsort(-score); ys = np.asarray(y)[o]
    tp = np.cumsum(ys); k = np.arange(1, len(ys)+1)
    prec = tp/k; rec = tp/max(1, ys.sum())
    return float(np.sum(prec*np.diff(np.concatenate([[0], rec]))))

def cov_at_prec(score, y, t, mc=20):
    o = np.argsort(-score); ys = np.asarray(y)[o].astype(float)
    k = np.arange(1, len(ys)+1); prec = np.cumsum(ys)/k; cov = k/len(ys)
    rec = np.cumsum(ys)/max(1, ys.sum())
    ok = (prec >= t) & (k >= mc)
    if not ok.any(): return None
    i = np.where(ok)[0].max()
    return float(cov[i]), float(rec[i])


# ---- LOO predict NON-ESSENTIAL, both feature sets, all organisms ----
df["pred_noness_all"] = np.nan
df["pred_noness_3"] = np.nan
for held in ALLORGS:
    m = (df.organism == held).to_numpy()
    tr = df[~m]
    df.loc[m, "pred_noness_all"] = fit_predict(
        tr[FEATS_ALL].to_numpy(float), tr.noness.to_numpy(), df[m][FEATS_ALL].to_numpy(float))
    df.loc[m, "pred_noness_3"] = fit_predict(
        tr[FEATS_3].to_numpy(float), tr.noness.to_numpy(), df[m][FEATS_3].to_numpy(float))

# ---- A: risk-coverage non-essential, bacteria pool ----
bp = df[~df.organism.isin(ARCHAEA)]
ap = df[df.organism.isin(ARCHAEA)]
y_b = bp.noness.to_numpy(); s_b = bp.pred_noness_all.to_numpy()
base_b = y_b.mean()
print(f"BACTERIA non-essential base rate: {base_b:.3f}")
o = np.argsort(-s_b); ys = y_b[o]
for c in (0.25, 0.50, 0.70, 0.90):
    k = int(c*len(ys)); print(f"  precision@top{int(c*100)}% non-ess = {ys[:k].mean():.3f}")
covA = {}
for t in (0.9, 0.95, 0.99):
    r = cov_at_prec(s_b, y_b, t)
    covA[t] = r
    print(f"  coverage@P>={t}: " + (f"{r[0]*100:.1f}% (recall {r[1]:.2f})" if r else "never"))

# ---- B: minimal model AUC/AUPRC + flipped coefficients ----
print("\nMINIMAL MODEL (non-essential), cross-org LOO mean over 6 bacteria:")
rows_b = []
for held in BACT:
    m = (bac.organism == held)
    te = bac[m]; tr = bac[~m]; yte = te.noness.to_numpy()
    pa = fit_predict(tr[FEATS_ALL].to_numpy(float), tr.noness.to_numpy(), te[FEATS_ALL].to_numpy(float))
    p3 = fit_predict(tr[FEATS_3].to_numpy(float), tr.noness.to_numpy(), te[FEATS_3].to_numpy(float))
    pc = fit_predict(tr[["conservation"]].to_numpy(float), tr.noness.to_numpy(), te[["conservation"]].to_numpy(float))
    rows_b.append(dict(organism=held,
        auc_all=auc(pa,yte), auprc_all=auprc(pa,yte),
        auc_3=auc(p3,yte), auprc_3=auprc(p3,yte),
        auc_cons=auc(pc,yte), auprc_cons=auprc(pc,yte)))
mb = pd.DataFrame(rows_b)
for c in ["all","3","cons"]:
    print(f"  {c:<5} AUC {mb[f'auc_{c}'].mean():.3f}  AUPRC {mb[f'auprc_{c}'].mean():.3f}")
# coefficients on all bacteria
_, w = fit_predict(bac[FEATS_ALL].to_numpy(float), bac.noness.to_numpy(),
                   bac[FEATS_ALL].to_numpy(float), ret_w=True)
coef = sorted(zip(FEATS_ALL, w[1:]), key=lambda t: -t[1])
print("\nnon-essential coefficients (standardized, + => marks SAFE-to-lose):")
for n, v in coef: print(f"  {n:<16} {v:+.3f}")

# ---- C: BUFFERED hard zone (conserved but non-essential) ----
buf = bac[(bac.noness == 1) & (bac.conservation >= 0.5)]
ess = bac[bac.essential == 1]
ne_un = bac[(bac.noness == 1) & (bac.conservation < 0.5)]
print(f"\nBUFFERED zone (non-ess & cons>=0.5): n={len(buf)} "
      f"({len(buf)/ (bac.noness==1).sum()*100:.1f}% of non-essentials)")
print(f"  mean paralogs:  buffered {buf.n_paralogs.mean():.2f} | "
      f"essential {ess.n_paralogs.mean():.2f} | other-noness {ne_un.n_paralogs.mean():.2f}")
print(f"  paralog>=1 frac: buffered {(buf.n_paralogs>=1).mean():.2f} | "
      f"essential {(ess.n_paralogs>=1).mean():.2f} | other-noness {(ne_un.n_paralogs>=1).mean():.2f}")

# name buffered genes in GMI1000
atlas = pd.read_csv(OUT / "gene_dot_atlas_GMI1000.csv")
gmi = bac[bac.organism == "beril_RalstoniaGMI1000"].merge(
    atlas[["locus_tag","product"]], on="locus_tag", how="left")
bufg = gmi[(gmi.noness == 1) & (gmi.conservation >= 0.5)].sort_values("n_paralogs", ascending=False)
bufg[["locus_tag","product","conservation","n_paralogs","gene_len_aa"]].to_csv(
    OUT / "nonessential_buffered_GMI1000.csv", index=False)
print(f"\n  GMI1000 buffered genes: {len(bufg)} (top paralog-rich):")
for _, r in bufg.head(10).iterrows():
    print(f"   {r['locus_tag']:<13} par={int(r['n_paralogs']):<3} cons={r['conservation']:.2f} "
          f"{str(r['product'])[:46]}")

# ---- D: per-org coverage@P0.9 non-essential ----
covD = {}
for org in ALLORGS:
    m = df.organism == org
    r = cov_at_prec(df[m].pred_noness_all.to_numpy(), df[m].noness.to_numpy(), 0.9)
    covD[org] = r[0] if r else 0.0

# ---- figure ----
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# A: risk-coverage curves, non-ess, bacteria vs archaea
ax = axes[0, 0]
for pool, mask, c, lab in [("bacteria", ~df.organism.isin(ARCHAEA), "#2980B9","bacteria"),
                           ("archaea",  df.organism.isin(ARCHAEA),  "#E67E22","archaea")]:
    sub = df[mask]; y = sub.noness.to_numpy(); s = sub.pred_noness_all.to_numpy()
    o = np.argsort(-s); ys = y[o].astype(float)
    k = np.arange(1, len(ys)+1); prec = np.cumsum(ys)/k; cov = k/len(ys)
    ax.plot(cov, prec, color=c, lw=2, label=f"{lab} (base {y.mean():.2f})")
    ax.axhline(y.mean(), color=c, ls=":", lw=0.8, alpha=0.5)
ax.axhline(0.9, color="k", ls="--", lw=0.8); ax.axhline(0.95, color="k", ls="--", lw=0.6)
ax.set_xlabel("coverage (fraction called non-essential)"); ax.set_ylabel("precision")
ax.set_title("A. Risk-coverage for NON-ESSENTIAL prediction (all_combined)")
ax.set_xlim(0,1); ax.set_ylim(0.5,1.01); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# B: minimal model bars + coefficients
ax = axes[0, 1]
sets = ["cons","3","all"]; labels_s = ["conservation\nonly","three core\n(cons+parlg+len)","all 7"]
aucs = [mb[f"auc_{s}"].mean() for s in sets]; aprs = [mb[f"auprc_{s}"].mean() for s in sets]
x = np.arange(3)
ax.bar(x-0.2, aucs, 0.4, label="AUC", color="#2980B9")
ax.bar(x+0.2, aprs, 0.4, label="AUPRC", color="#16A085")
for i,(a,p) in enumerate(zip(aucs,aprs)):
    ax.text(i-0.2,a,f"{a:.2f}",ha="center",va="bottom",fontsize=9)
    ax.text(i+0.2,p,f"{p:.2f}",ha="center",va="bottom",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels_s, fontsize=9); ax.set_ylim(0,1)
ax.axhline(base_b, color="gray", ls="--", lw=0.8, label=f"base {base_b:.2f}")
ax.set_ylabel("non-essential, cross-org LOO (6 bact)")
ax.set_title("B. Minimal model for non-essential\n(AUPRC high: it's the majority class)")
ax.legend(fontsize=8)

# C: buffered zone -- paralog distribution
ax = axes[1, 0]
groups = [("essential", ess.n_paralogs), ("buffered\n(cons>=.5, non-ess)", buf.n_paralogs),
          ("other non-ess", ne_un.n_paralogs)]
data = [g[1].clip(upper=8).to_numpy() for g in groups]
parts = ax.violinplot(data, showmeans=True)
ax.set_xticks([1,2,3]); ax.set_xticklabels([g[0] for g in groups], fontsize=9)
ax.set_ylabel("in-genome paralogs (clipped at 8)")
ax.set_title(f"C. BUFFERED hard zone: conserved-but-dispensable\n"
             f"buffered paralogs {buf.n_paralogs.mean():.1f} vs essential {ess.n_paralogs.mean():.1f}")
ax.grid(alpha=0.3, axis="y")

# D: per-org coverage@P0.9 non-essential
ax = axes[1, 1]
orgs = [o.replace("beril_","") for o in ALLORGS]
vals = [covD[o]*100 for o in ALLORGS]
cols = ["#E67E22" if o in ARCHAEA else "#2980B9" for o in ALLORGS]
ax.bar(range(len(orgs)), vals, color=cols, alpha=0.88)
for i,v in enumerate(vals):
    ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(orgs))); ax.set_xticklabels(orgs, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("% genes confidently called non-essential")
ax.set_title("D. Coverage @ precision>=0.9 (non-essential)\nblue=bacteria orange=archaea")
ax.set_ylim(0, max(vals)*1.25 if max(vals)>0 else 1)

fig.suptitle("Flipping the lens: predicting NON-ESSENTIAL genes (the majority-class problem)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT / "nonessential_analysis.png", dpi=140, bbox_inches="tight")

(OUT / "nonessential_analysis_summary.json").write_text(json.dumps({
    "bacteria_noness_base": round(float(base_b), 3),
    "A_coverage_at_precision": {str(t): (None if covA[t] is None else
        {"coverage": round(covA[t][0],3), "recall": round(covA[t][1],3)}) for t in covA},
    "A_precision_at_top": {
        f"top{int(c*100)}": round(float(np.sort(s_b)[::-1][:int(c*len(s_b))].mean()
            if False else y_b[np.argsort(-s_b)][:int(c*len(y_b))].mean()), 3)
        for c in (0.25,0.5,0.7,0.9)},
    "B_minimal_model_mean": {c: {"auc": round(float(mb[f'auc_{c}'].mean()),3),
        "auprc": round(float(mb[f'auprc_{c}'].mean()),3)} for c in ["cons","3","all"]},
    "B_coefficients": {n: round(float(v),3) for n,v in coef},
    "C_buffered": {"n": int(len(buf)),
        "pct_of_nonessentials": round(float(len(buf)/(bac.noness==1).sum()),3),
        "mean_paralogs_buffered": round(float(buf.n_paralogs.mean()),2),
        "mean_paralogs_essential": round(float(ess.n_paralogs.mean()),2),
        "mean_paralogs_other_noness": round(float(ne_un.n_paralogs.mean()),2)},
    "D_coverage_P90_per_org": {o: round(float(covD[o]),3) for o in ALLORGS},
}, indent=2))
print("\nwrote nonessential_analysis.png + buffered csv + summary")
