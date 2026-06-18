"""Per-gene dot atlas: essentials vs the 'opposite', with the non-essentials split.

Every gene in Ralstonia GMI1000 becomes ONE dot. Feature vector per gene
(non-negotiable: motifs, RBS, mobile-elements; plus more):

  motif_density   top 5-mer TGCTG (+rc) per kb of the gene        [MOTIF]
  rbs_strength    5 - mismatch(GGAGG) upstream  (higher = stronger) [RBS]
  mobile_dist_kb  distance to nearest transposase/IS/phage         [MOBILE]
  length_aa       gene length
  gc              gene GC fraction
  leading         co-oriented with replication (1/0)
  oriC_frac       0 origin .. 1 terminus
  n_paralogs      in-genome family copies
  conservation    family_frac_essential_leakfree
  atg_start       start codon ATG (1/0)
  tga_stop        stop codon TGA (1/0)
  m10_hit         strong -10 Pribnow box (1/0)
  m35_hit         strong -35 box (1/0)
  deep_operon     both neighbours same strand, gaps < 50 bp (1/0)

Then:
  (1) fit the essentiality axis (logistic) -> project every gene -> the genes at
      the ANTI-essential extreme are 'the opposite of essential genes'.
  (2) PCA of all genes (dots), colour essential vs non-essential.
  (3) SPLIT the non-essentials with k-means (k=4); profile each cluster on the
      non-negotiable channels.
  (4) the headline dot plot: x=mobile_dist, y=motif_density, size=RBS,
      colour=class -- all three non-negotiables in one view.

Outputs:
  outputs/orphan/gene_dot_atlas_GMI1000.png
  outputs/orphan/gene_dot_atlas_GMI1000.csv
  outputs/orphan/gene_dot_atlas_GMI1000_summary.json
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
ORG = "beril_RalstoniaGMI1000"; ACC = "GCF_000009125.1"
COMP = str.maketrans("ACGTNacgtn","TGCANtgcan")
def rc(s): return s.translate(COMP)[::-1]
MOB = re.compile(r"transposase|integrase|insertion sequence|IS\d|phage|"
                 r"recombinase|mobile element", re.I)
M10="TATAAT"; M35="TTGACA"; SD="GGAGG"; MOTIF="TGCTG"; MRC=rc(MOTIF)
def mm(a,b): return sum(1 for x,y in zip(a,b) if x!=y)
def best(region,cons):
    k=len(cons); b=k+1
    for i in range(len(region)-k+1):
        d=mm(region[i:i+k],cons)
        if d<b: b=d
    return b

ess={r["locus_tag"]:int(r["essential"])
     for r in csv.DictReader(open(DATA/"labels"/"labels.csv")) if r["organism"]==ORG}
orth={r["locus_tag"]:r
      for r in csv.DictReader(open(DATA/"labels"/"orthology_features.csv")) if r["organism"]==ORG}

contigs={}; cur=None; buf=[]
for line in (DATA/"genome_cache"/ACC/"genome.fna").read_text().splitlines():
    if line.startswith(">"):
        if cur: contigs[cur]="".join(buf).upper()
        cur=line[1:].split()[0]; buf=[]
    else: buf.append(line.strip())
if cur: contigs[cur]="".join(buf).upper()

cds=defaultdict(list)
for line in (DATA/"genome_cache"/ACC/"genomic.gff").read_text().splitlines():
    if "\tCDS\t" not in line: continue
    c=line.split("\t")
    m=re.search(r"locus_tag=([^;\n]+)",c[8])
    if not m: continue
    prod=re.search(r"product=([^;\n]+)",c[8])
    cds[c[0]].append((int(c[3]),int(c[4]),c[6],m.group(1),prod.group(1) if prod else ""))
for k in cds: cds[k].sort()

def oriC_terC(seq,win=10000):
    n=len(seq); sk=[]
    for i in range(0,n,win):
        w=seq[i:i+win]; g,cc=w.count("G"),w.count("C"); sk.append((g-cc)/max(1,g+cc))
    cum=np.cumsum(sk); return int(np.argmin(cum))*win,int(np.argmax(cum))*win

rows=[]
for ctg,lst in cds.items():
    seq=contigs.get(ctg)
    if not seq or len(seq)<20000: continue
    L=len(seq); oriC,terC=oriC_terC(seq)
    mobs=sorted({(a+b)//2 for a,b,st,lt,p in lst if MOB.search(p or "")})
    for i,(s,e,st,lt,prod) in enumerate(lst):
        if lt not in ess: continue
        mid=(s+e)//2; gene=seq[s-1:e]; laa=(e-s+1)//3
        gc=(gene.count("G")+gene.count("C"))/max(1,len(gene))
        motif=(gene.count(MOTIF)+gene.count(MRC))/(len(gene)/1000)
        if st=="+":
            up=seq[max(0,s-1-30):s-1]; startc=seq[s-1:s+2]; stopc=seq[e-3:e]
        else:
            up=rc(seq[e:min(L,e+30)]); startc=rc(seq[e-3:e]); stopc=rc(seq[s-1:s+2])
        rbs_mm = best(up[-20:-5],SD) if len(up)>=20 else 5
        rbs_str = 5-rbs_mm
        m10 = int(best(up[-18:-4],M10)<=1) if len(up)>=18 else 0
        m35 = int(best(up[-30:-14],M35)<=1) if len(up)>=30 else 0
        delta=(mid-oriC)%L; leading=int((st=="+")==(delta<L/2))
        d=abs(mid-oriC)%L; d=min(d,L-d); oriC_frac=d/(L/2)
        md=(min(min(abs(mid-mp),L-abs(mid-mp)) for mp in mobs)/1e3) if mobs else 500.0
        up_same=(i>0 and lst[i-1][2]==st); dn_same=(i<len(lst)-1 and lst[i+1][2]==st)
        up_gap=(s-lst[i-1][1]) if i>0 else 999
        dn_gap=(lst[i+1][0]-e) if i<len(lst)-1 else 999
        deep=int(up_same and dn_same and up_gap<50 and dn_gap<50)
        o=orth.get(lt)
        cons=float(o["family_frac_essential_leakfree"]) if o and o["family_frac_essential_leakfree"] not in ("","nan") else 0.0
        npar=float(o["n_paralogs_in_genome"]) if o and o["n_paralogs_in_genome"]!="" else 0.0
        rows.append(dict(locus_tag=lt,contig=ctg,mid=mid,essential=ess[lt],product=prod,
            motif_density=round(motif,3), rbs_strength=rbs_str, mobile_dist_kb=round(min(md,500),2),
            length_aa=laa, gc=round(gc,4), leading=leading, oriC_frac=round(oriC_frac,4),
            n_paralogs=npar, conservation=round(cons,4), atg_start=int(startc=="ATG"),
            tga_stop=int(stopc=="TGA"), m10_hit=m10, m35_hit=m35, deep_operon=deep))
df=pd.DataFrame(rows)
print(f"{len(df)} genes  ({df.essential.sum()} essential, {(df.essential==0).sum()} non-essential)")

FEATS=["motif_density","rbs_strength","mobile_dist_kb","length_aa","gc","leading",
       "oriC_frac","n_paralogs","conservation","atg_start","tga_stop","m10_hit",
       "m35_hit","deep_operon"]
X=df[FEATS].to_numpy(float)
mu=X.mean(0); sd=X.std(0); sd[sd==0]=1; Xz=(X-mu)/sd

# (1) essentiality axis via logistic (class-weighted)
def logit(Xtr,ytr,it=800,lr=0.4,l2=2.0):
    Xa=np.c_[np.ones(len(Xtr)),Xtr]; w=np.zeros(Xa.shape[1]); ytr=ytr.astype(float)
    pos=ytr.mean() or 1e-6; cw=np.where(ytr==1,0.5/pos,0.5/(1-pos))
    for _ in range(it):
        p=1/(1+np.exp(-Xa@w)); g=Xa.T@((p-ytr)*cw)/len(Xa); g[1:]+=l2*w[1:]/len(Xa); w-=lr*g
    return w
w=logit(Xz,df.essential.to_numpy())
df["ess_axis"]=np.c_[np.ones(len(Xz)),Xz]@w
coef=sorted(zip(FEATS,w[1:]),key=lambda t:-t[1])
print("\nessentiality axis (standardized logistic weights):")
for n,v in coef: print(f"  {n:<16} {v:+.3f}")

# the OPPOSITE of essential genes = anti-essential extreme (lowest axis), among non-ess
ne=df[df.essential==0].copy()
thr=ne.ess_axis.quantile(0.15)
opp=ne[ne.ess_axis<=thr]
print(f"\nThe OPPOSITE set: {len(opp)} most anti-essential non-essential genes (bottom 15%)")
print("  profile vs essentials (mean):")
for f in ["motif_density","rbs_strength","mobile_dist_kb","length_aa","gc","n_paralogs","conservation"]:
    print(f"   {f:<16} opposite={opp[f].mean():8.2f}   essential={df[df.essential==1][f].mean():8.2f}")

# (2) PCA (numpy SVD) of all genes
U,S,Vt=np.linalg.svd(Xz-Xz.mean(0),full_matrices=False)
PC=U[:,:2]*S[:2]
df["pc1"]=PC[:,0]; df["pc2"]=PC[:,1]
ev=(S**2)/np.sum(S**2)
print(f"\nPCA explained var: PC1 {ev[0]:.1%}  PC2 {ev[1]:.1%}")
opp=df[(df.essential==0)&(df.ess_axis<=thr)]   # refresh with pc columns

# (3) split non-essentials with k-means (k=4), pure numpy
def kmeans(Xk,k=4,iters=60,seed=0):
    rng=np.random.default_rng(seed); cent=Xk[rng.choice(len(Xk),k,replace=False)]
    for _ in range(iters):
        d=((Xk[:,None,:]-cent[None,:,:])**2).sum(2); lab=d.argmin(1)
        new=np.array([Xk[lab==j].mean(0) if (lab==j).any() else cent[j] for j in range(k)])
        if np.allclose(new,cent): break
        cent=new
    return lab,cent
neZ=Xz[(df.essential==0).to_numpy()]
klab,_=kmeans(neZ,k=4,seed=1)
ne_idx=df.index[df.essential==0]
df["ne_cluster"]=-1
df.loc[ne_idx,"ne_cluster"]=klab
print("\nNon-essential split into 4 clusters (mean non-negotiables + size):")
print(f"  {'cluster':<8}{'n':>5}{'motif':>8}{'rbs':>6}{'mobile_kb':>11}{'len_aa':>8}{'gc':>7}{'parlg':>7}")
cluster_profiles=[]
for j in range(4):
    sub=df[df.ne_cluster==j]
    prof=dict(cluster=int(j),n=int(len(sub)),
        motif=round(float(sub.motif_density.mean()),2),
        rbs=round(float(sub.rbs_strength.mean()),2),
        mobile_kb=round(float(sub.mobile_dist_kb.mean()),1),
        len_aa=round(float(sub.length_aa.mean()),0),
        gc=round(float(sub.gc.mean()),3),
        paralogs=round(float(sub.n_paralogs.mean()),2),
        leading=round(float(sub.leading.mean()),2))
    cluster_profiles.append(prof)
    print(f"  {j:<8}{len(sub):>5}{sub.motif_density.mean():>8.2f}{sub.rbs_strength.mean():>6.2f}"
          f"{sub.mobile_dist_kb.mean():>11.1f}{sub.length_aa.mean():>8.0f}{sub.gc.mean():>7.3f}"
          f"{sub.n_paralogs.mean():>7.2f}")

# ---- figure ----
fig,axes=plt.subplots(2,2,figsize=(17,14))

# A: PCA dots, essential vs non-essential
ax=axes[0,0]
nemask=df.essential==0
ax.scatter(df.pc1[nemask],df.pc2[nemask],s=7,c="#BDC3C7",alpha=0.5,label="non-essential")
ax.scatter(df.pc1[~nemask],df.pc2[~nemask],s=9,c="#C0392B",alpha=0.6,label="essential")
ax.scatter(opp.pc1,opp.pc2,s=14,facecolors="none",edgecolors="#2980B9",lw=0.7,label="'opposite' (anti-ess)")
ax.set_xlabel(f"PC1 ({ev[0]:.0%})"); ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
ax.set_title("A. Every gene a dot (PCA of 14 features)"); ax.legend(fontsize=8)

# B: the essentiality-axis histogram, essential vs non-essential, opposite shaded
ax=axes[0,1]
bins=np.linspace(df.ess_axis.min(),df.ess_axis.max(),50)
ax.hist(df.ess_axis[~nemask],bins,alpha=0.6,color="#C0392B",label="essential",density=True)
ax.hist(df.ess_axis[nemask],bins,alpha=0.5,color="#BDC3C7",label="non-essential",density=True)
ax.axvline(thr,color="#2980B9",ls="--",lw=1.2,label="anti-ess cutoff (15%)")
ax.set_xlabel("essentiality axis (logistic projection)"); ax.set_ylabel("density")
ax.set_title("B. The essentiality axis -- 'opposite' genes sit at the left tail")
ax.legend(fontsize=8)

# C: THE non-negotiable dot plot -- x mobile_dist, y motif, size rbs, colour class
ax=axes[1,0]
for mask,col,lab in [(nemask,"#BDC3C7","non-essential"),(~nemask,"#C0392B","essential")]:
    sub=df[mask]
    ax.scatter(sub.mobile_dist_kb,sub.motif_density,
               s=4+(sub.rbs_strength)**1.6, c=col, alpha=0.45,
               edgecolors="none",label=f"{lab} (size=RBS)")
ax.set_xscale("symlog")
ax.set_xlabel("mobile-element distance (kb, symlog)  [NON-NEGOTIABLE]")
ax.set_ylabel("motif TGCTG density per kb  [NON-NEGOTIABLE]")
ax.set_title("C. The three non-negotiables: motifs (y) x mobile (x) x RBS (dot size)")
ax.legend(fontsize=8,loc="upper right")

# D: non-essential clusters in PCA space + essentials faint
ax=axes[1,1]
ax.scatter(df.pc1[~nemask],df.pc2[~nemask],s=6,c="#E74C3C",alpha=0.15,label="essential (bg)")
cols=["#2980B9","#27AE60","#E67E22","#8E44AD"]
for j in range(4):
    sub=df[df.ne_cluster==j]
    ax.scatter(sub.pc1,sub.pc2,s=8,c=cols[j],alpha=0.6,
               label=f"NE cluster {j} (n={len(sub)})")
ax.set_xlabel(f"PC1 ({ev[0]:.0%})"); ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
ax.set_title("D. Non-essentials split into 4 clusters (k-means)")
ax.legend(fontsize=8)

fig.suptitle(f"Gene dot atlas -- {ORG.replace('beril_','')}: essentials, their opposite, and the split non-essentials",
             fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT/"gene_dot_atlas_GMI1000.png",dpi=140,bbox_inches="tight")
df.to_csv(OUT/"gene_dot_atlas_GMI1000.csv",index=False)

(OUT/"gene_dot_atlas_GMI1000_summary.json").write_text(json.dumps({
    "organism":ORG,"n_genes":int(len(df)),
    "n_essential":int(df.essential.sum()),"n_nonessential":int((df.essential==0).sum()),
    "essentiality_axis_weights":{n:round(float(v),3) for n,v in coef},
    "pca_explained_var":[round(float(ev[0]),3),round(float(ev[1]),3)],
    "opposite_set":{
        "n":int(len(opp)),
        "mean_motif":round(float(opp.motif_density.mean()),3),
        "mean_rbs":round(float(opp.rbs_strength.mean()),3),
        "mean_mobile_kb":round(float(opp.mobile_dist_kb.mean()),2),
        "mean_len_aa":round(float(opp.length_aa.mean()),1),
        "mean_paralogs":round(float(opp.n_paralogs.mean()),2),
    },
    "nonessential_clusters":cluster_profiles,
},indent=2))
print("\nwrote gene_dot_atlas_GMI1000.png + csv + summary")
