"""SHARK-style test: does physicochemical k-mer similarity predict essentiality
where sequence-identity conservation (family_frac) is blind (the rogue zone)?

SHARK (reduced-alphabet k-mer homology approximation) detects similarity between
divergent sequences via physicochemical character rather than exact residues.
Test for our problem: represent each protein by reduced-alphabet 3-mer
frequencies (6 physicochemical classes -> 216 features) + global scalars, then a
LEAVE-ONE-ORGANISM-OUT physicochemical nearest-neighbour vote on essentiality.

Decisive comparisons:
  - physicochemical-NN AUC vs conservation (family_frac) AUC, overall
  - the same, restricted to the ROGUE ZONE (essential & conservation<0.1) where
    family_frac is blind by construction. If physicochem-NN beats chance HERE,
    SHARK-style recovers signal homology missed.

Pure-Python/NumPy, sandbox-runnable (no GPU, no network).
Output: outputs/orphan/shark_physicochem_test.json
"""
import csv, re, glob, os
from pathlib import Path
import numpy as np

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")
ORGS = ["beril_RalstoniaGMI1000","beril_RalstoniaBSBF1503","beril_RalstoniaPSI07",
        "beril_Dda3937","beril_HerbieS","beril_Magneto"]

# 6-class physicochemical reduced alphabet
GROUP = {}
for aa,g in [("AVLIMC",0),("FWY",1),("STNQ",2),("KRH",3),("DE",4),("GP",5)]:
    for a in aa: GROUP[a]=g
NG=6; K=3; DIM=NG**K   # 216 reduced 3-mer features

acc={r["organism"]:r["accession"] for r in csv.DictReader(open(DATA/"labels"/"genome_cache_manifest.csv")) if r.get("accession")}
labels={(r["organism"],r["locus_tag"]):int(r["essential"]) for r in csv.DictReader(open(DATA/"labels"/"labels.csv"))}
cons={}
for r in csv.DictReader(open(DATA/"labels"/"orthology_features.csv")):
    v=r["family_frac_essential_leakfree"]
    cons[(r["organism"],r["locus_tag"])]=float(v) if v not in("","nan") else 0.0

def parse_faa(p):
    d={};cur=None;buf=[]
    for line in open(p):
        if line.startswith(">"):
            if cur:d[cur]="".join(buf)
            cur=line[1:].split()[0];buf=[]
        else:buf.append(line.strip())
    if cur:d[cur]="".join(buf)
    return d

def profile(seq):
    seq=seq.rstrip("*")
    v=np.zeros(DIM,np.float32); n=0
    codes=[GROUP.get(a,-1) for a in seq]
    for i in range(len(codes)-K+1):
        a,b,c=codes[i],codes[i+1],codes[i+2]
        if a<0 or b<0 or c<0: continue
        v[a*36+b*6+c]+=1; n+=1
    if n: v/=n
    return v

# build per-gene profiles for the 6 bacteria
X=[]; y=[]; org=[]; cn=[]
for o in ORGS:
    a=acc.get(o)
    if not a: continue
    gff=DATA/"genome_cache"/a/"genomic.gff"; faa=DATA/"genome_cache"/a/"protein.faa"
    if not (gff.exists() and faa.exists()): continue
    lt2pid={}
    for line in open(gff):
        if "\tCDS\t" not in line: continue
        m=re.search(r"locus_tag=([^;\n]+)",line); p=re.search(r"protein_id=([^;\n]+)",line)
        if m and p: lt2pid[m.group(1)]=p.group(1)
    seqs=parse_faa(faa)
    for (oo,lt),e in labels.items():
        if oo!=o: continue
        pid=lt2pid.get(lt)
        if not pid or pid not in seqs: continue
        X.append(profile(seqs[pid])); y.append(e); org.append(o); cn.append(cons.get((o,lt),0.0))
X=np.array(X); y=np.array(y); org=np.array(org); cn=np.array(cn)
# L2-normalize profiles for cosine
Xn=X/ (np.linalg.norm(X,axis=1,keepdims=True)+1e-9)
print(f"{len(X)} proteins with physicochemical profiles, essential rate {y.mean():.3f}")

def auc(s,yy):
    yy=np.asarray(yy); n1=yy.sum(); n0=len(yy)-n1
    if n1==0 or n0==0: return float("nan")
    r=np.argsort(np.argsort(s))+1; return float((r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0))

# LEAVE-ONE-ORGANISM-OUT physicochemical NN vote
KNN=25
pred=np.full(len(X),np.nan)
for held in ORGS:
    te=org==held; tr=~te
    if te.sum()<50: continue
    Xt=Xn[tr]; yt=y[tr]; Xe=Xn[te]
    # cosine sim in chunks
    for s in range(0,Xe.shape[0],512):
        sims=Xe[s:s+512]@Xt.T               # [chunk, n_tr]
        idx=np.argpartition(-sims,KNN,axis=1)[:,:KNN]
        pred[np.where(te)[0][s:s+512]] = yt[idx].mean(1)
ev=~np.isnan(pred)
print("\n=== physicochemical k-mer NN vote (LOO-organism) ===")
print(f"  overall AUC: {auc(pred[ev],y[ev]):.3f}   (conservation overall AUC: {auc(cn[ev],y[ev]):.3f})")
# rogue zone: essential & conservation<0.1 vs non-essential & conservation<0.1
rogue_mask = ev & (cn<0.1)
print(f"\n  ROGUE ZONE (conservation<0.1, n={rogue_mask.sum()}, ess rate {y[rogue_mask].mean():.3f}):")
print(f"    physicochem-NN AUC: {auc(pred[rogue_mask],y[rogue_mask]):.3f}")
print(f"    conservation AUC:   {auc(cn[rogue_mask],y[rogue_mask]):.3f}  (blind by construction)")
# does physicochem ADD to conservation? logistic of [cons, physichem_nn]
def fit_auc(cols, mask):
    Xc=np.c_[cols].T[mask]; yy=y[mask]
    mu=Xc.mean(0);sd=Xc.std(0);sd[sd==0]=1;Z=np.c_[np.ones(mask.sum()),(Xc-mu)/sd]
    w=np.zeros(Z.shape[1])
    for _ in range(300):
        p=1/(1+np.exp(-Z@w)); g=Z.T@(p-yy)/len(yy); w-=0.5*g
    return auc(1/(1+np.exp(-Z@w)),yy)
print(f"\n  combined (conservation + physicochem-NN) overall AUC: {fit_auc([cn,pred],ev):.3f}")

import json
json.dump(dict(n=int(ev.sum()),
  physicochem_nn_auc=round(auc(pred[ev],y[ev]),3),
  conservation_auc=round(auc(cn[ev],y[ev]),3),
  rogue_n=int(rogue_mask.sum()),
  rogue_physicochem_auc=round(auc(pred[rogue_mask],y[rogue_mask]),3),
  rogue_conservation_auc=round(auc(cn[rogue_mask],y[rogue_mask]),3),
  combined_auc=round(fit_auc([cn,pred],ev),3)),
  open(OUT/"shark_physicochem_test.json","w"), indent=2)
print("\nwrote shark_physicochem_test.json")
