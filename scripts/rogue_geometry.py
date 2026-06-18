"""Forbidden-configuration test: can replication geometry find ROGUE essentials?

Logic chain:
  - The cell avoids the lagging-strand x long-gene configuration (head-on
    replication-transcription collision risk; we measured lagging-Q4 essentials
    at 19% vs leading-Q4 at 30%).
  - So a gene that IS essential while sitting in that 'forbidden' slot is doing
    something the cell could not design away -- a candidate CONDITIONAL/rogue
    essential.
  - Rogue zone = essential AND conservation < 0.1 (conservation cannot find
    these by construction; that was the whole Project-A wall).

Tests (6 bacteria, from cross_org_coverage_scores.csv):
  T1  among ESSENTIALS, is the rogue fraction (cons<0.1) higher in the forbidden
      config (lagging & long) than in the safe config (leading & short)?
  T2  within LOW-CONSERVATION genes (cons<0.1, where conservation = 0 signal),
      does the forbidden config enrich for essentials over base?
  T3  can a GEOMETRY-ONLY model (no conservation) recover rogue essentials
      LOO across organisms -- precision/recall vs the random base rate?
  T4  name the GMI1000 rogue-essential candidates in the forbidden slot.

Outputs:
  outputs/orphan/rogue_geometry.png
  outputs/orphan/rogue_geometry_candidates_GMI1000.csv
  outputs/orphan/rogue_geometry_summary.json
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
df = df[~df.organism.isin(ARCHAEA)].reset_index(drop=True)
BACT = list(dict.fromkeys(df.organism))

# per-organism length quartiles -> "long" = top quartile
df["long"] = False
for o in BACT:
    m = df.organism == o
    q75 = df.loc[m, "gene_len_aa"].quantile(0.75)
    df.loc[m, "long"] = df.loc[m, "gene_len_aa"] >= q75
df["lagging"] = df.leading == 0
df["forbidden"] = df.lagging & df.long          # the configuration the cell avoids
df["safe"] = (~df.lagging) & (~df.long)         # leading + short
df["rogue"] = (df.essential == 1) & (df.conservation < 0.1)
df["lowcons"] = df.conservation < 0.1

base = df.essential.mean()
print(f"6 bacteria: {len(df)} genes, base essential {base:.3f}, "
      f"{df.rogue.sum()} rogue essentials, {df.lowcons.sum()} low-cons genes\n")

# ---- T1: rogue fraction among essentials, by configuration ----
print("T1  among ESSENTIALS, fraction that are rogue (cons<0.1) by config:")
ess = df[df.essential == 1]
configs = {
    "safe (leading+short)":  ess.safe,
    "leading+long":          (~ess.lagging) & ess.long,
    "lagging+short":         ess.lagging & (~ess.long),
    "forbidden (lagging+long)": ess.forbidden,
}
t1 = {}
for name, mask in configs.items():
    sub = ess[mask]
    frac = (sub.conservation < 0.1).mean() if len(sub) else float("nan")
    t1[name] = {"n": int(len(sub)), "rogue_frac": round(float(frac), 3)}
    print(f"  {name:<28} n={len(sub):<5} rogue_frac={frac:.3f}")

# ---- T2: within low-conservation genes, essential rate by config ----
print("\nT2  within LOW-CONSERVATION genes (cons<0.1), essential rate by config:")
lc = df[df.lowcons]
lc_base = lc.essential.mean()
print(f"  low-cons base essential rate: {lc_base:.3f}  (n={len(lc)})")
t2 = {"lowcons_base": round(float(lc_base), 3), "n": int(len(lc))}
for name, col in [("safe (leading+short)", "safe"),
                  ("forbidden (lagging+long)", "forbidden")]:
    sub = lc[lc[col]]
    r = sub.essential.mean() if len(sub) else float("nan")
    enr = r / lc_base if lc_base else float("nan")
    t2[name] = {"n": int(len(sub)), "ess_rate": round(float(r), 3),
                "enrichment": round(float(enr), 3)}
    print(f"  {name:<28} n={len(sub):<5} ess_rate={r:.3f}  ({enr:+.2f}x lowcons-base)")

# ---- T3: geometry-only LOO recovery of rogue essentials ----
GEOM = ["leading", "oriC_frac", "mobile_dist_kb", "n_paralogs", "gene_len_aa", "gc"]
def fit_predict(Xtr, ytr, Xte, it=600, lr=0.5, l2=2.0):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1
    Xtr = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]
    Xte = np.c_[np.ones(len(Xte)), (Xte-mu)/sd]
    w = np.zeros(Xtr.shape[1]); ytr = ytr.astype(float)
    pos = ytr.mean() or 1e-6; cw = np.where(ytr == 1, 0.5/pos, 0.5/(1-pos))
    for _ in range(it):
        p = 1/(1+np.exp(-Xtr@w)); g = Xtr.T@((p-ytr)*cw)/len(Xtr)
        g[1:] += l2*w[1:]/len(Xtr); w -= lr*g
    return 1/(1+np.exp(-Xte@w))

# train on low-cons genes of other orgs, predict low-cons genes of held org
df["geom_score"] = np.nan
for held in BACT:
    tr = df[(df.organism != held) & df.lowcons]
    te_mask = (df.organism == held) & df.lowcons
    te = df[te_mask]
    if len(te) < 10 or tr.essential.nunique() < 2: continue
    pred = fit_predict(tr[GEOM].to_numpy(float), tr.essential.to_numpy(),
                       te[GEOM].to_numpy(float))
    df.loc[te_mask, "geom_score"] = pred

lc2 = df[df.lowcons & df.geom_score.notna()]
def rec_at_prec(score, y, t, mc=10):
    o = np.argsort(-score); ys = np.asarray(y)[o].astype(float)
    k = np.arange(1, len(ys)+1); prec = np.cumsum(ys)/k; rec = np.cumsum(ys)/max(1, ys.sum())
    ok = (prec >= t) & (k >= mc)
    return float(rec[np.where(ok)[0].max()]) if ok.any() else 0.0
def auc(score, y):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y)-n1
    if n1 == 0 or n0 == 0: return float("nan")
    ranks = np.argsort(np.argsort(score)) + 1
    return float((ranks[y==1].sum() - n1*(n1+1)/2)/(n1*n0))
yv = lc2.essential.to_numpy(); sv = lc2.geom_score.to_numpy()
geom_auc = auc(sv, yv)
o = np.argsort(-sv); ys = yv[o]
top10 = ys[:int(0.1*len(ys))].mean(); top30 = ys[:int(0.3*len(ys))].mean()
print(f"\nT3  GEOMETRY-ONLY rogue recovery (low-cons genes, LOO, n={len(lc2)}):")
print(f"  base rate {lc2.essential.mean():.3f}  AUC {geom_auc:.3f}")
print(f"  precision@top10% = {top10:.3f}  ({top10/lc2.essential.mean():+.2f}x base)")
print(f"  precision@top30% = {top30:.3f}  ({top30/lc2.essential.mean():+.2f}x base)")
r_at_50 = rec_at_prec(sv, yv, max(0.5, 1.5*lc2.essential.mean()))
print(f"  recall@P>=50%   = {r_at_50:.3f}")

# ---- T4: name the GMI1000 rogue-essential candidates in the forbidden slot ----
atlas = pd.read_csv(OUT / "gene_dot_atlas_GMI1000.csv")
g = df[df.organism == "beril_RalstoniaGMI1000"].merge(
    atlas[["locus_tag", "product"]], on="locus_tag", how="left")
cand = g[(g.essential == 1) & (g.forbidden) & (g.conservation < 0.1)].copy()
cand = cand.sort_values("gene_len_aa", ascending=False)
cand_out = cand[["locus_tag", "product", "gene_len_aa", "leading",
                 "oriC_frac", "mobile_dist_kb", "n_paralogs", "conservation"]]
cand_out.to_csv(OUT / "rogue_geometry_candidates_GMI1000.csv", index=False)
print(f"\nT4  GMI1000 rogue-essential candidates in forbidden slot: {len(cand)}")
for _, r in cand.head(15).iterrows():
    print(f"  {r['locus_tag']:<13} {str(r['product'])[:52]:<52} {int(r['gene_len_aa'])}aa")

# ---- figure ----
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# A: T1 rogue fraction among essentials by config
ax = axes[0, 0]
names = list(configs); fracs = [t1[n]["rogue_frac"] for n in names]
ns = [t1[n]["n"] for n in names]
cols = ["#27AE60", "#7FB3D5", "#F39C12", "#C0392B"]
ax.bar(range(len(names)), fracs, color=cols, alpha=0.88)
for i,(f,n) in enumerate(zip(fracs, ns)):
    ax.text(i, f, f" {f:.2f}\n n={n}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(["safe\nlead+short","lead+long","lag+short","forbidden\nlag+long"], fontsize=9)
ax.set_ylabel("rogue fraction among essentials (cons<0.1)")
ax.set_ylim(0, max(fracs)*1.35)
ax.set_title("T1. Are forbidden-slot essentials more often rogue?")

# B: T2 essential rate within low-cons by config
ax = axes[0, 1]
labels_b = ["low-cons\nbase", "safe\nlead+short", "forbidden\nlag+long"]
vals_b = [t2["lowcons_base"], t2["safe (leading+short)"]["ess_rate"],
          t2["forbidden (lagging+long)"]["ess_rate"]]
ns_b = [t2["n"], t2["safe (leading+short)"]["n"], t2["forbidden (lagging+long)"]["n"]]
ax.bar(range(3), vals_b, color=["#7F8C8D","#27AE60","#C0392B"], alpha=0.88)
for i,(v,n) in enumerate(zip(vals_b, ns_b)):
    ax.text(i, v, f" {v:.2f}\n n={n}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(labels_b, fontsize=9)
ax.set_ylabel("essential rate (low-conservation genes only)")
ax.set_ylim(0, max(vals_b)*1.35)
ax.set_title("T2. Within rogue-prone (low-cons) genes,\ndoes forbidden config enrich essentials?")

# C: T3 risk-coverage on low-cons subset, geometry vs random
ax = axes[1, 0]
o = np.argsort(-sv); ys2 = yv[o].astype(float)
cov = np.arange(1, len(ys2)+1)/len(ys2); prec = np.cumsum(ys2)/np.arange(1, len(ys2)+1)
ax.plot(cov, prec, color="#2980B9", lw=2, label="geometry-only model")
ax.axhline(lc2.essential.mean(), color="gray", ls="--", lw=1,
           label=f"random (base {lc2.essential.mean():.2f})")
ax.set_xlabel("coverage (low-conservation genes, ranked)")
ax.set_ylabel("precision"); ax.set_ylim(0, max(0.5, prec[:int(0.1*len(prec))].max()*1.1))
ax.set_title(f"T3. Geometry recovering rogue essentials\nAUC {geom_auc:.2f} (low-cons LOO)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# D: candidate table as text
ax = axes[1, 1]; ax.axis("off")
lines = [f"T4. GMI1000 rogue-essential candidates",
         f"    (essential + lagging + long + cons<0.1): {len(cand)} genes\n"]
for _, r in cand.head(16).iterrows():
    p = str(r["product"])[:40]
    lines.append(f"{r['locus_tag']:<12} {p:<40} {int(r['gene_len_aa'])}aa")
ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
        fontsize=8, transform=ax.transAxes)

fig.suptitle("Can the 'forbidden' replication configuration (lagging + long) flag rogue essentials?",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT / "rogue_geometry.png", dpi=140, bbox_inches="tight")

(OUT / "rogue_geometry_summary.json").write_text(json.dumps({
    "n_bacteria": len(BACT), "n_genes": int(len(df)),
    "base_essential": round(float(base), 3),
    "n_rogue": int(df.rogue.sum()),
    "T1_rogue_frac_among_essentials_by_config": t1,
    "T2_lowcons_essential_rate_by_config": t2,
    "T3_geometry_lowcons_LOO": {
        "n": int(len(lc2)), "base": round(float(lc2.essential.mean()), 3),
        "auc": round(float(geom_auc), 3),
        "precision_top10": round(float(top10), 3),
        "precision_top30": round(float(top30), 3)},
    "T4_n_candidates_GMI1000": int(len(cand)),
}, indent=2))
print("\nwrote rogue_geometry.png + candidates csv + summary")
