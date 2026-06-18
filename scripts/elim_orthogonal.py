"""Eliminate non-essentials on THEIR OWN terms, not by inverted essentiality.

The previous loop killed 99% of rogue essentials because its 'non-essential'
classifier was the essentiality axis read backwards (conservation-dominated):
'confident non-essential' == 'confidently not essential-looking', and a rogue
essential (low conservation, single-copy, long) looks exactly like that.

The fix: eliminate a gene ONLY on POSITIVE evidence of dispensability that is
orthogonal to the essentiality axis:
   redundancy   -- a paralog backup exists in the genome (n_paralogs)
   mobility     -- sits in an accessory / mobile-element neighborhood
Crucially, rogue essentials are paralog-POOR and not mobile-adjacent, so a
dispensability-evidence filter should SPARE them.

Compare three eliminators (LOO over 6 bacteria), each thresholded to non-ess
precision >= 0.90 on TRAINING data:
   A  essentiality-axis  : all 7 features (what we did) -- 'essential terms'
   B  dispensability     : n_paralogs + mobile_dist_kb only -- 'non-essential terms'
   C  conservation-only  : the pure inverted-essentiality baseline

Metrics per eliminator: how many killed, kill precision, and -- the key number --
how many ROGUE essentials (cons<0.1) and BUFFERED genes (cons>=0.5 non-ess) got
killed. Then run the essential predictor on B's residual and check rogue survival.

Outputs:
  outputs/orphan/elim_orthogonal.png
  outputs/orphan/elim_orthogonal_summary.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/user/cell/outputs/orphan")
ARCHAEA = {"beril_Methanococcus_JJ","beril_Methanococcus_S2"}
df = pd.read_csv(OUT / "cross_org_coverage_scores.csv")
df = df[~df.organism.isin(ARCHAEA)].reset_index(drop=True)
BACT = list(dict.fromkeys(df.organism))
df["noness"] = 1 - df.essential
df["rogue"] = (df.essential==1) & (df.conservation<0.1)
df["buffered"] = (df.essential==0) & (df.conservation>=0.5)

ELIMS = {
    "A_essentiality_axis": ["leading","oriC_frac","mobile_dist_kb","n_paralogs","gene_len_aa","gc","conservation"],
    "B_dispensability":    ["n_paralogs","mobile_dist_kb"],
    "C_conservation_only": ["conservation"],
}

def fit_predict(Xtr,ytr,Xte,it=600,lr=0.5,l2=2.0):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    Xtr=np.c_[np.ones(len(Xtr)),(Xtr-mu)/sd]; Xte=np.c_[np.ones(len(Xte)),(Xte-mu)/sd]
    w=np.zeros(Xtr.shape[1]); ytr=ytr.astype(float)
    pos=ytr.mean() or 1e-6; cw=np.where(ytr==1,0.5/pos,0.5/(1-pos))
    for _ in range(it):
        p=1/(1+np.exp(-Xtr@w)); g=Xtr.T@((p-ytr)*cw)/len(Xtr)
        g[1:]+=l2*w[1:]/len(Xtr); w-=lr*g
    return 1/(1+np.exp(-Xtr@w)), 1/(1+np.exp(-Xte@w))

def thr_at_prec(s,y,t=0.9,mn=20):
    o=np.argsort(-s); ys=np.asarray(y)[o].astype(float); ss=np.asarray(s)[o]
    k=np.arange(1,len(ys)+1); prec=np.cumsum(ys)/k
    ok=(prec>=t)&(k>=mn)
    return float(ss[np.where(ok)[0].max()]) if ok.any() else np.inf

def auc(s,y):
    y=np.asarray(y); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return float("nan")
    r=np.argsort(np.argsort(s))+1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

# run each eliminator LOO, record per-gene kill flag
rogue_total = int(df.rogue.sum()); buf_total = int(df.buffered.sum())
summary = {}
print(f"pool: {len(df)} genes | {int(df.essential.sum())} ess | "
      f"{rogue_total} rogue | {buf_total} buffered\n")
for name, feats in ELIMS.items():
    killed = np.zeros(len(df), bool)
    for held in BACT:
        m=(df.organism==held).to_numpy(); tr=df[~m]; te=df[m]
        ptr,pte = fit_predict(tr[feats].to_numpy(float), tr.noness.to_numpy(), te[feats].to_numpy(float))
        thr = thr_at_prec(ptr, tr.noness.to_numpy(), 0.9)
        killed[m] = pte >= thr
    df[f"kill_{name}"] = killed
    nk=int(killed.sum())
    kill_prec = float(df.noness[killed].mean()) if nk else float("nan")  # frac truly non-ess
    ess_killed = int((killed & (df.essential==1)).sum())
    rogue_killed = int((killed & df.rogue).sum())
    buf_killed = int((killed & df.buffered).sum())
    summary[name] = dict(
        killed=nk, kill_frac=round(nk/len(df),3),
        kill_precision=round(kill_prec,3),
        essentials_killed=ess_killed,
        essentials_killed_frac=round(ess_killed/int(df.essential.sum()),3),
        rogue_killed=rogue_killed, rogue_spared=rogue_total-rogue_killed,
        rogue_spared_frac=round((rogue_total-rogue_killed)/rogue_total,3),
        buffered_killed=buf_killed,
        buffered_caught_frac=round(buf_killed/buf_total,3))
    print(f"{name}")
    print(f"   killed {nk} ({nk/len(df):.0%})  kill-precision {kill_prec:.3f}")
    print(f"   essentials killed {ess_killed} ({ess_killed/int(df.essential.sum()):.1%} of ess)")
    print(f"   ROGUE spared {rogue_total-rogue_killed}/{rogue_total} "
          f"({(rogue_total-rogue_killed)/rogue_total:.0%})   buffered caught "
          f"{buf_killed}/{buf_total} ({buf_killed/buf_total:.0%})\n")

# ---- essential prediction on B's residual, check rogue survival ----
FE = ELIMS["A_essentiality_axis"]
resB = ~df["kill_B_dispensability"].to_numpy()
df["ess_score_resB"]=np.nan
for held in BACT:
    te_mask=(df.organism==held).to_numpy() & resB
    tr_mask=(df.organism!=held).to_numpy() & resB
    if te_mask.sum()<20: continue
    _,pte=fit_predict(df[tr_mask][FE].to_numpy(float), df[tr_mask].essential.to_numpy(),
                      df[te_mask][FE].to_numpy(float))
    df.loc[df.index[te_mask],"ess_score_resB"]=pte
rb=df[resB & df.ess_score_resB.notna()]
yb=rb.essential.to_numpy(); sb=rb.ess_score_resB.to_numpy()
o=np.argsort(-sb); ys=yb[o]
covB=None
k=np.arange(1,len(ys)+1); pc=np.cumsum(ys)/k
ok=(pc>=0.7)&(k>=20)
if ok.any(): covB=float(k[np.where(ok)[0].max()])/len(ys)
print("=== essential predictor on B's residual ===")
print(f"  residual size {len(rb)}  base ess {yb.mean():.3f}  AUC {auc(sb,yb):.3f}")
print(f"  precision@top10% {ys[:int(0.1*len(ys))].mean():.3f}  coverage@P0.7 {covB if covB else 0:.2f}")
rogueB = df[resB & df.rogue]
print(f"  rogue essentials surviving into residual: {len(rogueB)}/{rogue_total} "
      f"({len(rogueB)/rogue_total:.0%})  (vs 1% with eliminator A)")

# ---- figure ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
names=list(ELIMS); short=["A\nessentiality\naxis","B\ndispensability\n(parlg+mobile)","C\nconservation\nonly"]

# panel 1: rogue spared vs buffered caught vs kill coverage
ax=axes[0]
x=np.arange(3); w=0.26
ax.bar(x-w, [summary[n]["kill_frac"] for n in names], w, label="genes killed (frac)", color="#7F8C8D")
ax.bar(x,   [summary[n]["rogue_spared_frac"] for n in names], w, label="rogue ESSENTIALS spared", color="#27AE60")
ax.bar(x+w, [summary[n]["buffered_caught_frac"] for n in names], w, label="buffered caught", color="#8E44AD")
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9); ax.set_ylim(0,1.05)
for i,n in enumerate(names):
    ax.text(i-w, summary[n]["kill_frac"], f"{summary[n]['kill_frac']:.0%}", ha="center", va="bottom", fontsize=8)
    ax.text(i, summary[n]["rogue_spared_frac"], f"{summary[n]['rogue_spared_frac']:.0%}", ha="center", va="bottom", fontsize=8)
ax.set_title("Eliminator comparison\n(want: high rogue-spared AND high kill)")
ax.legend(fontsize=8)

# panel 2: essentials wrongly killed
ax=axes[1]
ek=[summary[n]["essentials_killed_frac"] for n in names]
ax.bar(x, ek, color=["#C0392B","#27AE60","#C0392B"], alpha=0.85)
for i,v in enumerate(ek): ax.text(i,v,f"{v:.0%}",ha="center",va="bottom",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
ax.set_ylabel("fraction of ALL essentials wrongly killed")
ax.set_title("Collateral damage: essentials lost\n(lower is better)")
ax.set_ylim(0, max(ek)*1.3)

# panel 3: kill precision vs coverage tradeoff
ax=axes[2]
for i,n in enumerate(names):
    ax.scatter(summary[n]["kill_frac"], summary[n]["rogue_spared_frac"], s=160,
               c=["#C0392B","#27AE60","#E67E22"][i])
    ax.annotate(short[i].replace("\n"," "), (summary[n]["kill_frac"], summary[n]["rogue_spared_frac"]),
                fontsize=8, xytext=(5,5), textcoords="offset points")
ax.set_xlabel("coverage: fraction of genes confidently eliminated")
ax.set_ylabel("rogue essentials spared (fraction)")
ax.set_title("The tradeoff: eliminate more OR spare rogue?")
ax.set_xlim(0,0.7); ax.set_ylim(0,1.05); ax.grid(alpha=0.3)

fig.suptitle("Eliminating non-essentials on their OWN terms (dispensability evidence) vs inverted essentiality",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(OUT/"elim_orthogonal.png", dpi=140, bbox_inches="tight")

summary["essential_on_B_residual"]={
    "residual_n":int(len(rb)),"base_ess":round(float(yb.mean()),3),
    "auc":round(float(auc(sb,yb)),3),
    "coverage_at_p70":round(covB,3) if covB else 0.0,
    "rogue_surviving":int(len(rogueB)),
    "rogue_surviving_frac":round(len(rogueB)/rogue_total,3)}
(OUT/"elim_orthogonal_summary.json").write_text(json.dumps(summary,indent=2))
print("\nwrote elim_orthogonal.png + summary")
