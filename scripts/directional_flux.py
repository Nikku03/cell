"""Directional flux: is a gene being RETAINED in its lineage or DIMINISHING?

Static breadth (how many organisms have the gene) was flat (8,388 genomes added
nothing). This tests the DIRECTIONAL version of the trajectory idea:

  retention = fraction of the focal organism's SISTER lineage that still carries
              the gene's OG (within-lineage retention)
  breadth   = fraction of ALL organisms with the OG (cross-lineage prevalence)

The signature of a DIMINISHING gene: broad across distant lineages but ABSENT in
close sisters (breadth high, retention low) -> the lineage is shedding it.
The signature of a CORE gene: retained everywhere (both high).

Leak-clean: uses presence/absence only (known from genomes at deploy time), not
essentiality labels. Tests:
  1. does retention predict essentiality (AUC)?
  2. KEY: at the SAME breadth, does retention add signal? (stratified)
  3. are 'diminishing' genes (broad but lost in sisters) less essential?

Outputs:
  outputs/orphan/directional_flux.png
  outputs/orphan/directional_flux.json
"""
import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/user/cell/outputs/orphan")
DATA = Path("/home/user/cell/data/drive_import")

SISTERS = {
 "beril_RalstoniaGMI1000": ["beril_RalstoniaBSBF1503","beril_RalstoniaPSI07","beril_RalstoniaUW163"],
 "beril_RalstoniaBSBF1503":["beril_RalstoniaGMI1000","beril_RalstoniaPSI07","beril_RalstoniaUW163"],
 "beril_RalstoniaPSI07":   ["beril_RalstoniaGMI1000","beril_RalstoniaBSBF1503","beril_RalstoniaUW163"],
 "beril_Dda3937":          ["beril_Ddia6719","beril_DdiaME23"],
 "beril_HerbieS":          ["beril_Cup4G11","beril_BFirm","beril_Burk376"],
 "beril_Magneto":          ["beril_azobra","beril_Smeli","beril_PS"],
}
FOCAL = list(SISTERS)

# labels
labels={}
for r in csv.DictReader(open(DATA/"labels"/"labels.csv")):
    if r["organism"] in FOCAL: labels[(r["organism"],r["locus_tag"])]=int(r["essential"])
# orthology: og -> set of orgs; gene -> og
og_orgs=defaultdict(set); gene_og={}; all_orgs=set()
for r in csv.DictReader(open(DATA/"labels"/"orthology_features.csv")):
    if r["og_id"]:
        og_orgs[r["og_id"]].add(r["organism"]); all_orgs.add(r["organism"])
        gene_og[(r["organism"],r["locus_tag"])]=r["og_id"]
NALL=len(all_orgs)
# conservation (leak-free) for stratify reference
cons={}
for r in csv.DictReader(open(DATA/"labels"/"orthology_features.csv")):
    v=r.get("family_frac_essential_leakfree","")
    cons[(r["organism"],r["locus_tag"])]=float(v) if v not in("","nan") else 0.0
print(f"{NALL} organisms in orthology, {len(labels)} labeled focal genes")

rows=[]
for (org,lt),y in labels.items():
    og=gene_og.get((org,lt))
    if not og: continue
    present=og_orgs[og]
    sis=SISTERS[org]
    retention=sum(1 for s in sis if s in present)/len(sis)        # within-lineage
    breadth=len(present)/NALL                                      # cross-lineage
    # cross-lineage excluding own sisters+self (distant prevalence)
    distant=len([o for o in present if o not in sis and o!=org])/max(1,NALL-len(sis)-1)
    diminishing=distant - retention   # high = broad-but-lost-in-sisters
    rows.append(dict(org=org, lt=lt, essential=y, conservation=cons.get((org,lt),0.0),
                     retention=retention, breadth=breadth, distant=distant,
                     diminishing=diminishing))
df=pd.DataFrame(rows)
print(f"{len(df)} genes, essential rate {df.essential.mean():.3f}")

def auc(score,y):
    y=np.asarray(y); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return float('nan')
    r=np.argsort(np.argsort(score))+1; return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

print("\nsingle-feature essentiality AUC:")
for f in ["breadth","retention","distant","diminishing","conservation"]:
    print(f"  {f:<14} AUC {auc(df[f].values, df.essential.values):.3f}")

# KEY TEST: within breadth strata, does retention add signal?
print("\nSTRATIFIED: within each breadth bin, essential rate by retention level")
df["bbin"]=pd.qcut(df.breadth, 4, labels=["b1","b2","b3","b4"], duplicates="drop")
df["rbin"]=pd.cut(df.retention, [-0.01,0.01,0.5,0.99,1.01], labels=["lost","minority","most","all"])
piv=df.pivot_table(index="bbin", columns="rbin", values="essential", aggfunc="mean", observed=True)
print(piv.round(3).to_string())
# within the highest-breadth bin, lost vs retained
hb=df[df.bbin=="b4"]
lost=hb[hb.retention<=0.01]; kept=hb[hb.retention>=0.99]
if len(lost)>20 and len(kept)>20:
    print(f"\n  among BROADEST genes (b4): retained-in-sisters ess {kept.essential.mean():.3f} "
          f"(n={len(kept)}) vs lost-in-sisters ess {lost.essential.mean():.3f} (n={len(lost)})")

# does retention add OVER breadth? simple 2-feature logistic vs breadth-only
def fit_auc(cols):
    X=df[cols].to_numpy(float); y=df.essential.to_numpy()
    mu=X.mean(0); sd=X.std(0); sd[sd==0]=1; Xz=(X-mu)/sd; Xa=np.c_[np.ones(len(Xz)),Xz]
    w=np.zeros(Xa.shape[1])
    for _ in range(400):
        p=1/(1+np.exp(-Xa@w)); g=Xa.T@(p-y)/len(y); w-=0.5*g
    return auc(1/(1+np.exp(-Xa@w)), y)
print(f"\nbreadth only            AUC {fit_auc(['breadth']):.3f}")
print(f"breadth + retention     AUC {fit_auc(['breadth','retention']):.3f}")
print(f"breadth + retention + diminishing AUC {fit_auc(['breadth','retention','diminishing']):.3f}")

summary=dict(n_genes=int(len(df)),
    single_auc={f:round(auc(df[f].values,df.essential.values),3) for f in
                ["breadth","retention","distant","diminishing","conservation"]},
    breadth_only_auc=round(fit_auc(['breadth']),3),
    breadth_plus_retention_auc=round(fit_auc(['breadth','retention']),3),
    breadth_plus_flux_auc=round(fit_auc(['breadth','retention','diminishing']),3))
json.dump(summary, open(OUT/"directional_flux.json","w"), indent=2)

# figure
fig,ax=plt.subplots(1,2,figsize=(14,5.5))
import numpy as np
pivn=piv.reindex(columns=["lost","minority","most","all"])
im=ax[0].imshow(pivn.values, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=df.essential.max())
ax[0].set_xticks(range(len(pivn.columns))); ax[0].set_xticklabels(pivn.columns)
ax[0].set_yticks(range(len(pivn.index))); ax[0].set_yticklabels([f"breadth {b}" for b in pivn.index])
for i in range(pivn.shape[0]):
    for j in range(pivn.shape[1]):
        v=pivn.values[i,j]
        if v==v: ax[0].text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=9)
ax[0].set_xlabel("sister-lineage retention ->"); ax[0].set_title("Essential rate by breadth x retention\n(does direction matter at fixed breadth?)")
fig.colorbar(im,ax=ax[0],label="essential rate")
# AUC bars
labels_b=["breadth\nonly","+retention","+flux"]
vals=[summary["breadth_only_auc"],summary["breadth_plus_retention_auc"],summary["breadth_plus_flux_auc"]]
ax[1].bar(range(3),vals,color=["#7F8C8D","#16A085","#C0392B"])
for i,v in enumerate(vals): ax[1].text(i,v,f"{v:.3f}",ha="center",va="bottom")
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(labels_b); ax[1].set_ylim(0,1)
ax[1].set_title("Does directional flux add over static breadth?")
fig.suptitle("Directional flux: retained vs diminishing genes (the trajectory idea)",fontweight="bold")
fig.tight_layout(); fig.savefig(OUT/"directional_flux.png",dpi=140,bbox_inches="tight")
print("\nwrote directional_flux.png + .json")
