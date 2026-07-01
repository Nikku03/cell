"""Plot the integrated human cell: (A) genome ideogram with the essential core, master TFs,
and disease genes laid out; (B) the vital-core signature (every layer agrees)."""
import csv, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT=Path("outputs/orphan")
rows=list(csv.DictReader(open(OUT/"integrated_cell_human.csv")))
meta=json.load(open(OUT/"integrated_cell_human.json")); CS=meta["chrom_sizes"]
def f(v):
    try: return float(v)
    except: return None
CHROMS=[f"chr{i}" for i in range(1,23)]+["chrX","chrY"]
yof={c:i for i,c in enumerate(CHROMS)}
fig=plt.figure(figsize=(15,11)); gs=fig.add_gridspec(2,1,height_ratios=[2.4,1])
ax=fig.add_subplot(gs[0])
maxlen=max(CS.values())
for c in CHROMS:
    y=yof[c]; ax.add_patch(plt.Rectangle((0,y-0.28),CS[c]/1e6,0.56,fc="#eceff1",ec="#b0bec5",lw=.5,zorder=1))
# plot genes
ess=[]; tf=[]; dis=[]
for r in rows:
    c=r["chrom"]; 
    if c not in yof or not r["tss"]: continue
    x=float(r["tss"])/1e6; y=yof[c]
    if r["essential"]=="1": ess.append((x,y))
    if r["lineage_master"]: tf.append((x,y,r["gene"]))
    nd=f(r["n_diseases"]) or 0
    if nd>=5: dis.append((x,y))
if dis: ax.scatter([x for x,y in dis],[y+0.13 for x,y in dis],s=3,c="#ff9800",alpha=.35,zorder=2,label=f"disease genes (n={len(dis)})")
if ess: ax.scatter([x for x,y in ess],[y for x,y in ess],s=8,c="#e53935",alpha=.7,zorder=3,label=f"essential core (n={len(ess)})")
for x,y,g in tf:
    ax.scatter([x],[y-0.13],s=55,c="#1e88e5",marker="*",zorder=4,edgecolor="k",lw=.3)
    ax.annotate(g,(x,y-0.13),fontsize=5.5,ha="center",va="top",xytext=(0,-4),textcoords="offset points")
ax.scatter([],[],s=55,c="#1e88e5",marker="*",edgecolor="k",lw=.3,label="lineage master TFs")
ax.set_yticks(range(len(CHROMS))); ax.set_yticklabels([c.replace("chr","") for c in CHROMS],fontsize=8)
ax.set_xlabel("position (Mb)"); ax.set_ylim(-1,len(CHROMS)); ax.invert_yaxis()
ax.set_title("The integrated human cell, laid out on the genome (zygote = same DNA, all layers overlaid)")
ax.legend(loc="lower right",fontsize=8,framealpha=.9); ax.set_xlim(-2,maxlen/1e6+2)
# Panel B: vital-core signature
axb=fig.add_subplot(gs[1])
def grp(lab):
    return [r for r in rows if r["essential"]==lab]
E=grp("1"); N=grp("0")
def mean(rs,k,inv=False):
    v=[f(r[k]) for r in rs if f(r[k]) is not None]
    m=np.mean(v) if v else 0; return m
metrics=[("constraint\n(1-LOEUF)",lambda rs:1-mean(rs,"loeuf")),("PPI hubness\n(interactions)",lambda rs:mean(rs,"ppi_degree")),
    ("CpG-island\npromoter %",lambda rs:100*mean(rs,"cpg_promoter")),("pathways\nper gene",lambda rs:mean(rs,"n_pathways")),
    ("regulators\nper gene",lambda rs:mean(rs,"regulators_in"))]
labels=[m[0] for m in metrics]; ev=[m[1](E) for m in metrics]; nv=[m[1](N) for m in metrics]
# normalize each metric to essential=1 for display
disp_e=[1]*len(metrics); disp_n=[nv[i]/ev[i] if ev[i] else 0 for i in range(len(metrics))]
x=np.arange(len(metrics)); w=0.38
axb.bar(x-w/2,disp_e,w,color="#e53935",label="essential core")
axb.bar(x+w/2,disp_n,w,color="#90a4ae",label="non-essential")
for i in range(len(metrics)):
    axb.text(x[i]-w/2,1.02,f"{ev[i]:.1f}",ha="center",fontsize=7)
    axb.text(x[i]+w/2,disp_n[i]+0.02,f"{nv[i]:.1f}",ha="center",fontsize=7)
axb.set_xticks(x); axb.set_xticklabels(labels,fontsize=8); axb.set_ylabel("relative to essential core")
axb.set_title("Vital-core signature: every layer independently flags the essential genes")
axb.legend(fontsize=8); axb.grid(alpha=.2,axis="y")
plt.tight_layout(); plt.savefig(OUT/"integrated_cell_human.png",dpi=130)
print("wrote integrated_cell_human.png")
print(f"  plotted {len(ess)} essential, {len(tf)} master TFs, {len(dis)} disease genes on the genome")
