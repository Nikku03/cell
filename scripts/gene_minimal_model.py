"""Minimal model: is essentiality just conservation + redundancy + gene length?

The dot atlas showed essentials and their 'opposite' differ only on conservation,
paralog count and length -- not on motif/RBS/GC grammar. Test it directly.

TWO arenas:
  (1) WITHIN GMI1000 (5-fold CV) -- here we also have the non-negotiable
      sequence channels (motif TGCTG density, RBS strength, mobile distance):
        - three_core      : conservation + n_paralogs + length_aa
        - non_negotiables : motif_density + rbs_strength + mobile_dist_kb
        - full_14         : everything from the dot atlas
  (2) CROSS-ORG LOO over 6 bacteria (the honest transfer test, from
      cross_org_coverage_scores.csv):
        - conservation_only
        - three_core       : conservation + n_paralogs + gene_len_aa
        - position_geom    : leading + oriC_frac + mobile_dist_kb + gc
        - all_seven        : everything available cross-org

Outputs:
  outputs/orphan/gene_minimal_model.png
  outputs/orphan/gene_minimal_model_summary.json
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


def fit_predict(Xtr, ytr, Xte, it=600, lr=0.5, l2=2.0):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1
    Xtr = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]
    Xte = np.c_[np.ones(len(Xte)), (Xte-mu)/sd]
    w = np.zeros(Xtr.shape[1]); ytr = ytr.astype(float)
    pos = ytr.mean() or 1e-6
    cw = np.where(ytr == 1, 0.5/pos, 0.5/(1-pos))
    for _ in range(it):
        p = 1/(1+np.exp(-Xtr@w))
        g = Xtr.T@((p-ytr)*cw)/len(Xtr); g[1:] += l2*w[1:]/len(Xtr)
        w -= lr*g
    return 1/(1+np.exp(-Xte@w))


def auc(score, y):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y)-n1
    if n1 == 0 or n0 == 0: return float("nan")
    ranks = np.argsort(np.argsort(score)) + 1
    return float((ranks[y==1].sum() - n1*(n1+1)/2) / (n1*n0))


def auprc(score, y):
    o = np.argsort(-score); ys = np.asarray(y)[o]
    tp = np.cumsum(ys); k = np.arange(1, len(ys)+1)
    prec = tp/k; rec = tp/max(1, ys.sum())
    return float(np.sum(prec*np.diff(np.concatenate([[0], rec]))))


# ---------- (1) within GMI1000, 5-fold CV ----------
df = pd.read_csv(OUT / "gene_dot_atlas_GMI1000.csv")
y = df.essential.to_numpy()
SETS_IN = {
    "three_core":      ["conservation", "n_paralogs", "length_aa"],
    "non_negotiables": ["motif_density", "rbs_strength", "mobile_dist_kb"],
    "full_14":         ["motif_density","rbs_strength","mobile_dist_kb","length_aa",
                        "gc","leading","oriC_frac","n_paralogs","conservation",
                        "atg_start","tga_stop","m10_hit","m35_hit","deep_operon"],
}
rng = np.random.default_rng(0)
fold = rng.integers(0, 5, len(df))
print("=== WITHIN GMI1000 (5-fold CV) ===")
within = {}
for name, cols in SETS_IN.items():
    preds = np.zeros(len(df))
    for f in range(5):
        te = fold == f; tr = ~te
        preds[te] = fit_predict(df.loc[tr, cols].to_numpy(float), y[tr],
                                df.loc[te, cols].to_numpy(float))
    a = auc(preds, y); ap = auprc(preds, y)
    within[name] = {"auc": round(a, 3), "auprc": round(ap, 3)}
    print(f"  {name:<18} AUC {a:.3f}  AUPRC {ap:.3f}  ({len(cols)} feats)")

# ---------- (2) cross-org LOO over 6 bacteria ----------
sc = pd.read_csv(OUT / "cross_org_coverage_scores.csv")
sc = sc[~sc.organism.isin(ARCHAEA)].reset_index(drop=True)
BACT = list(dict.fromkeys(sc.organism))
SETS_LOO = {
    "conservation_only": ["conservation"],
    "three_core":        ["conservation", "n_paralogs", "gene_len_aa"],
    "position_geom":     ["leading", "oriC_frac", "mobile_dist_kb", "gc"],
    "all_seven":         ["leading","oriC_frac","mobile_dist_kb","n_paralogs",
                          "gene_len_aa","gc","conservation"],
}
print("\n=== CROSS-ORG LOO (6 bacteria) ===")
print(f"  {'organism':<24} " + " ".join(f"{k.split('_')[0][:6]:>7}" for k in SETS_LOO))
loo_rows = []
for held in BACT:
    m = (sc.organism == held).to_numpy()
    tr = sc[~m]; te = sc[m]; yte = te.essential.to_numpy()
    row = {"organism": held}
    line = f"  {held.replace('beril_',''):<24} "
    for name, cols in SETS_LOO.items():
        pred = fit_predict(tr[cols].to_numpy(float), tr.essential.to_numpy(),
                           te[cols].to_numpy(float))
        a = auc(pred, yte); row[name] = round(a, 3)
        line += f"{a:>7.3f} "
    loo_rows.append(row); print(line)
loo = pd.DataFrame(loo_rows)
print("\n  MEAN AUC:")
loo_mean = {}
for name in SETS_LOO:
    loo_mean[name] = round(float(loo[name].mean()), 3)
    print(f"    {name:<20} {loo_mean[name]:.3f}")

# ---------- figure ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# A: within-GMI1000 bars
ax = axes[0]
names = list(SETS_IN); aucs = [within[n]["auc"] for n in names]
aprs = [within[n]["auprc"] for n in names]
x = np.arange(len(names))
ax.bar(x-0.2, aucs, 0.4, label="AUC", color="#2980B9")
ax.bar(x+0.2, aprs, 0.4, label="AUPRC", color="#16A085")
for i,(a,p) in enumerate(zip(aucs,aprs)):
    ax.text(i-0.2,a,f"{a:.2f}",ha="center",va="bottom",fontsize=9)
    ax.text(i+0.2,p,f"{p:.2f}",ha="center",va="bottom",fontsize=9)
ax.axhline(0.5,color="gray",ls="--",lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(["three core\n(cons+parlg+len)","non-negotiables\n(motif+rbs+mobile)","full 14"],fontsize=9)
ax.set_ylim(0,1); ax.set_ylabel("within-GMI1000 5-fold CV")
ax.set_title("A. Three core features ≈ full model;\nmotif+RBS+mobile alone barely beat chance")
ax.legend(fontsize=9)

# B: cross-org LOO grouped bars
ax = axes[1]
orgs = [o.replace("beril_","") for o in loo.organism]
xx = np.arange(len(orgs)); wbar = 0.2
cols_plot = [("conservation_only","#95A5A6"),("three_core","#C0392B"),
             ("position_geom","#E67E22"),("all_seven","#2C3E50")]
for i,(name,c) in enumerate(cols_plot):
    ax.bar(xx+(i-1.5)*wbar, loo[name], wbar, label=name.replace("_"," "), color=c, alpha=0.9)
ax.axhline(0.5,color="gray",ls="--",lw=0.8)
ax.set_xticks(xx); ax.set_xticklabels(orgs,rotation=40,ha="right",fontsize=8)
ax.set_ylim(0,1); ax.set_ylabel("AUC (held-out organism)")
ax.set_title(f"B. Cross-org LOO: three_core {loo_mean['three_core']:.2f} vs all_seven {loo_mean['all_seven']:.2f}")
ax.legend(fontsize=8, loc="lower left")

# C: the 3-feature dot plot where separation IS visible
ax = axes[2]
ne = df.essential==0
ax.scatter(df.conservation[ne], df.length_aa[ne], s=6, c="#BDC3C7", alpha=0.4,
           label="non-essential")
ax.scatter(df.conservation[~ne], df.length_aa[~ne], s=8, c="#C0392B", alpha=0.5,
           label="essential")
ax.set_yscale("log")
ax.set_xlabel("conservation (family_frac_essential)")
ax.set_ylabel("gene length (aa, log)")
ax.set_title("C. The three-core space: dots DO separate\n(size∝paralogs not shown; cons on x, length on y)")
ax.legend(fontsize=9)

fig.suptitle("Is essentiality just conservation + redundancy + gene length? (vs the full / sequence-grammar models)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(OUT / "gene_minimal_model.png", dpi=140, bbox_inches="tight")

(OUT / "gene_minimal_model_summary.json").write_text(json.dumps({
    "within_GMI1000_5fold": within,
    "cross_org_LOO_mean_auc": loo_mean,
    "cross_org_LOO_per_org": loo_rows,
    "takeaway": "three_core (conservation+paralogs+length) recovers nearly all of "
                "the full model; motif+RBS+mobile sequence grammar alone is weak.",
}, indent=2))
print("\nwrote gene_minimal_model.png + summary")
