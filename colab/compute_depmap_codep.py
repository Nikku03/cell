"""Co-essentiality from DepMap: correlate every gene's CRISPR knockout profile across 1,100 cell
lines. Genes with correlated dependency are functional partners; a partner that's individually
non-essential yet co-dependent is a synthetic-lethal candidate. -> depmap_codep.json {gene:[[partner,r],...]}"""
import csv, json, numpy as np
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
genes=None; rows=[]
with open(H/"CRISPRGeneEffect.csv") as f:
    rd=csv.reader(f); genes=[g.split(" (")[0] for g in next(rd)[1:]]
    for r in rd: rows.append([float(x) if x else np.nan for x in r[1:]])
X=np.array(rows,dtype=np.float32)                     # lines x genes
# standardize columns, NaN->0
mu=np.nanmean(X,axis=0); sd=np.nanstd(X,axis=0); sd[sd==0]=1
Xs=(X-mu)/sd; Xs[np.isnan(Xs)]=0
n=Xs.shape[0]
# per-gene mean effect to know if a partner is individually essential
frac_dep=np.nanmean(X< -0.5,axis=0)
gi={g:i for i,g in enumerate(genes)}
codep={}
CH=1000
for s in range(0,len(genes),CH):
    e=min(s+CH,len(genes))
    C=(Xs[:,s:e].T @ Xs)/n                              # (chunk x allgenes) correlations
    for k in range(e-s):
        i=s+k; row=C[k]; row[i]=-9
        top=np.argpartition(-row,12)[:12]
        top=top[np.argsort(-row[top])]
        part=[[genes[j],round(float(row[j]),3)] for j in top if row[j]>0.2][:8]
        if part: codep[genes[i]]=part
json.dump(codep,open(OUT/"depmap_codep.json","w"))
# synthetic-lethal candidates: co-dependent pairs where BOTH are individually dispensable (buffered)
sl=[]
for g,parts in codep.items():
    if frac_dep[gi[g]]<0.3:
        for p,r in parts:
            if r>0.4 and frac_dep[gi[p]]<0.3 and g<p: sl.append([g,p,r])
sl.sort(key=lambda x:-x[2])
json.dump(sl[:2000],open(OUT/"depmap_sl.json","w"))
print("co-dependency: genes with partners",len(codep),"| SL candidate pairs",len(sl))
print("examples SL:",sl[:5])
