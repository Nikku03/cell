"""Combined essential-gene coverage across three channels (bacteria only).

The three channels are different in KIND, and conflating them is the usual error:

  DETECTORS (predict essentiality, give a per-gene score):
    1. conservation  (leak-free family essentiality fraction)
    2. protection    (leading strand, oriC dist, mobile-element dist, paralogs,
                      gene length, GC)  -- the recent cross-org features

  ANNOTATOR (does NOT detect essentiality; explains what a gene DOES):
    3. structure + context  (Project A: Foldseek/AFDB folds + confound-guarded
                             co-inheritance partners)  -- de-orphans DARK genes

So we answer two honest questions:

  (Q1) DETECTION coverage -- at a fixed high-precision operating point
       (precision >= 0.7, the same bar as the risk-coverage analysis), what
       fraction of an organism's TRUE essentials does each detector flag, and
       what is the union / overlap?  Computed for all 6 bacteria, LOO.

  (Q2) ANNOTATION overlay -- of the essentials that are DARK (no known
       function), how many can the Project A structure+context channel explain?
       Computed for GMI1000 (the one organism with the full Project A atlas).

Operating point: each detector is thresholded at the score that yields
precision = 0.7 on the held-out organism (matched precision => fair recall Venn).

Outputs:
  outputs/orphan/combined_coverage_venn.png
  outputs/orphan/combined_coverage_venn_summary.json
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
df = df[~df.organism.isin(ARCHAEA)].reset_index(drop=True)   # bacteria only
BACT = list(dict.fromkeys(df.organism))

PROT_COLS = ["leading","oriC_frac","mobile_dist_kb","n_paralogs","gene_len_aa","gc"]


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


# protection-only LOO scores (bacteria-only training pool)
df["score_protection_only"] = 0.0
for held in BACT:
    m = (df.organism == held).to_numpy()
    tr = df[~m]; te = df[m]
    s = fit_predict(tr[PROT_COLS].to_numpy(float), tr.essential.to_numpy(),
                    te[PROT_COLS].to_numpy(float))
    df.loc[m, "score_protection_only"] = s
# refit conservation-only on bacteria-only pool too (the CSV's was trained
# with archaea in the pool; redo for a clean bacteria-only comparison)
df["score_cons_bact"] = 0.0
for held in BACT:
    m = (df.organism == held).to_numpy()
    tr = df[~m]; te = df[m]
    s = fit_predict(tr[["conservation"]].to_numpy(float), tr.essential.to_numpy(),
                    te[["conservation"]].to_numpy(float))
    df.loc[m, "score_cons_bact"] = s


def threshold_at_precision(score, y, target=0.7, min_calls=20):
    """Return the score threshold giving precision >= target at max coverage."""
    o = np.argsort(-score); ys = np.asarray(y)[o].astype(float)
    k = np.arange(1, len(ys)+1); tp = np.cumsum(ys); prec = tp / k
    ok = (prec >= target) & (k >= min_calls)
    if not ok.any():
        return None
    i = np.where(ok)[0].max()
    return float(score[o][i])


def flagged_set(score, thr):
    if thr is None:
        return np.zeros(len(score), dtype=bool)
    return score >= thr


print("=== DETECTION coverage @ precision>=0.7 (bacteria, LOO) ===")
print(f"  {'organism':<24} essΣ | cons  prot  union | cons-only prot-only  both")
per_org = {}
agg = dict(ess=0, cons=0, prot=0, union=0, consonly=0, protonly=0, both=0,
           ess_caught=0)
for org in BACT:
    m = (df.organism == org).to_numpy()
    sub = df[m]
    y = sub.essential.to_numpy().astype(bool)
    sc = sub["score_cons_bact"].to_numpy()
    sp = sub["score_protection_only"].to_numpy()
    tc = threshold_at_precision(sc, y, 0.7)
    tp = threshold_at_precision(sp, y, 0.7)
    fc = flagged_set(sc, tc) & y      # true essentials caught by conservation
    fp = flagged_set(sp, tp) & y      # true essentials caught by protection
    union = fc | fp
    ne = int(y.sum())
    consN = int(fc.sum()); protN = int(fp.sum()); uN = int(union.sum())
    both = int((fc & fp).sum())
    consonly = int((fc & ~fp).sum()); protonly = int((fp & ~fc).sum())
    print(f"  {org.replace('beril_',''):<24} {ne:<4} | "
          f"{consN/ne:.2f}  {protN/ne:.2f}  {uN/ne:.2f} | "
          f"{consonly:<9} {protonly:<10} {both}")
    per_org[org] = dict(
        n_essential=ne, cons_recall=round(consN/ne,3), prot_recall=round(protN/ne,3),
        union_recall=round(uN/ne,3),
        cons_only=consonly, prot_only=protonly, both=both,
        union_n=uN, missed=int((~union).sum()),
    )
    for kk,vv in [("ess",ne),("cons",consN),("prot",protN),("union",uN),
                  ("consonly",consonly),("protonly",protonly),("both",both)]:
        agg[kk]+=vv

print("\n=== POOLED across 6 bacteria ===")
ess=agg["ess"]
print(f"  total essentials                : {ess}")
print(f"  conservation detects (P>=0.7)   : {agg['cons']:5d}  ({agg['cons']/ess:.1%})")
print(f"  protection   detects (P>=0.7)   : {agg['prot']:5d}  ({agg['prot']/ess:.1%})")
print(f"  UNION detects                   : {agg['union']:5d}  ({agg['union']/ess:.1%})")
print(f"    both channels                 : {agg['both']:5d}")
print(f"    conservation-only             : {agg['consonly']:5d}")
print(f"    protection-only (unique add)  : {agg['protonly']:5d}  <- what protection buys")
print(f"  MISSED by both detectors        : {ess-agg['union']:5d}  ({1-agg['union']/ess:.1%})")

# ---------------- Q2: annotation overlay on GMI1000 ----------------
print("\n=== ANNOTATION overlay (Project A structure+context), GMI1000 ===")
atlas = pd.read_parquet(OUT / "atlas_beril_RalstoniaGMI1000.parquet")
ess = atlas[atlas.essential == 1]
ne = len(ess)
annotated   = int((ess.function_tier == "annotated").sum())
dark        = int(ess.function_tier.isin(["live_ghost","unknown","context_inferred"]).sum())
ctx_rescue  = int((ess.function_tier == "context_inferred").sum())
ctx_partner = int((ess.context_partner_named == 1).sum())
# structure: the local atlas is the PRE-structure merge (fold_hit_named all 0);
# the full Project A capstone run added 60 fold-named de-orphans across all dark
# genes (552 dark -> 96 de-orphaned: 60 structure + 34 context + 2 phenotype).
# We report the LOCAL, reproducible context rescue here and cite the capstone
# structure figure separately rather than fabricating per-gene fold membership.
print(f"  total essentials                : {ne}")
print(f"  already annotated (known func)  : {annotated}  ({annotated/ne:.1%})")
print(f"  DARK essentials (no annotation) : {dark}  ({dark/ne:.1%})")
print(f"    of which de-orphaned by context: {ctx_rescue}")
print(f"  (essentials with a named context partner, incl. annotated: {ctx_partner})")
print(f"  NOTE: full Project A run adds structure (Foldseek) de-orphaning on top;")
print(f"        capstone: 552 dark genes -> 96 de-orphaned (17%) genome-wide.")

summary = {
    "scope": "bacteria only (6 organisms), leave-one-organism-out",
    "operating_point": "precision >= 0.7 per held-out organism",
    "detection": {
        "per_org": per_org,
        "pooled": {
            "total_essentials": int(agg["ess"]),
            "conservation_detects": int(agg["cons"]),
            "protection_detects": int(agg["prot"]),
            "union_detects": int(agg["union"]),
            "both": int(agg["both"]),
            "conservation_only": int(agg["consonly"]),
            "protection_only_unique": int(agg["protonly"]),
            "missed_by_both": int(agg["ess"]-agg["union"]),
            "conservation_recall": round(agg["cons"]/agg["ess"],3),
            "protection_recall": round(agg["prot"]/agg["ess"],3),
            "union_recall": round(agg["union"]/agg["ess"],3),
        },
    },
    "annotation_overlay_GMI1000": {
        "total_essentials": ne,
        "annotated": annotated,
        "dark": dark,
        "dark_deorphaned_by_context_local": ctx_rescue,
        "note": "structure (Foldseek/AFDB) per-gene fold membership lives in the "
                "Project A Colab run, not local; capstone reports 552 dark -> 96 "
                "(17%) de-orphaned genome-wide (60 structure + 34 context + 2 pheno).",
    },
}
(OUT / "combined_coverage_venn_summary.json").write_text(json.dumps(summary, indent=2))

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: detection Venn (pooled) drawn manually as two overlapping circles
ax = axes[0]; ax.set_aspect("equal"); ax.axis("off")
both=agg["both"]; co=agg["consonly"]; po=agg["protonly"]; miss=agg["ess"]-agg["union"]
from matplotlib.patches import Circle
c1=Circle((-0.35,0),0.62,alpha=0.45,color="#3498DB")
c2=Circle(( 0.35,0),0.62,alpha=0.45,color="#E67E22")
ax.add_patch(c1); ax.add_patch(c2)
ax.text(-0.75,0,f"conservation\nonly\n{co}",ha="center",va="center",fontsize=11)
ax.text(0.75,0,f"protection\nonly\n{po}",ha="center",va="center",fontsize=11)
ax.text(0,0,f"both\n{both}",ha="center",va="center",fontsize=11,fontweight="bold")
ax.text(0,-1.0,f"missed by both detectors: {miss}",ha="center",fontsize=10,color="#C0392B")
ax.set_xlim(-1.6,1.6); ax.set_ylim(-1.3,1.0)
ax.set_title(f"A. Detection @ P≥0.7 (6 bacteria, n={agg['ess']} essentials)\n"
             f"union recall = {agg['union']/agg['ess']:.0%}", fontsize=11)

# Panel B: per-org stacked recall (cons-only / both / prot-only / missed)
ax = axes[1]
orgs=[o.replace("beril_","") for o in BACT]
x=np.arange(len(BACT))
co=[per_org[o]["cons_only"] for o in BACT]
bo=[per_org[o]["both"] for o in BACT]
po=[per_org[o]["prot_only"] for o in BACT]
ms=[per_org[o]["missed"] for o in BACT]
tot=[per_org[o]["n_essential"] for o in BACT]
co=np.array(co)/tot; bo=np.array(bo)/tot; po=np.array(po)/tot; ms=np.array(ms)/tot
ax.bar(x,bo,color="#27AE60",label="both detectors")
ax.bar(x,co,bottom=bo,color="#3498DB",label="conservation-only")
ax.bar(x,po,bottom=bo+co,color="#E67E22",label="protection-only (unique)")
ax.bar(x,ms,bottom=bo+co+po,color="#D5D8DC",label="missed by both")
ax.set_xticks(x); ax.set_xticklabels(orgs,rotation=40,ha="right",fontsize=8)
ax.set_ylabel("fraction of true essentials"); ax.set_ylim(0,1)
ax.set_title("B. Per-organism detection coverage @ P≥0.7")
ax.legend(fontsize=8,loc="upper right")

# Panel C: GMI1000 annotation overlay (essential catalog composition)
ax = axes[2]
sizes=[annotated, ctx_rescue, dark-ctx_rescue]
labs=[f"annotated\n{annotated}",
      f"dark→context\n{ctx_rescue}",
      f"dark, unrescued\n(local) {dark-ctx_rescue}"]
cols=["#95A5A6","#16A085","#C0392B"]
ax.pie(sizes,labels=labs,colors=cols,autopct=lambda p:f"{p:.0f}%",
       startangle=90,textprops={"fontsize":9})
ax.set_title(f"C. GMI1000 essential catalog (n={ne})\nstructure adds ~17% de-orphan genome-wide (capstone)")

fig.suptitle("Combined essential-gene coverage: two detectors (conservation, protection) + one annotator (AlphaFold/context)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(OUT / "combined_coverage_venn.png", dpi=140, bbox_inches="tight")
print("\nwrote combined_coverage_venn.png + summary")
