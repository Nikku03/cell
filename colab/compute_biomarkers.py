"""Biomarkers of target sensitivity (#4/#5). For each candidate target, correlate its DepMap
dependency profile across cell lines with (a) CCLE expression and (b) CCLE damaging mutations.
-> biomarkers.json {target: {expr:[[gene,r]], mut:[[gene,delta]]}}   (Chronos<0 = dependent, so
negative expr-corr / negative mut-delta = SENSITIVITY biomarker; positive = resistance)."""
import csv, json, numpy as np
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
def load(fn):
    with open(H/fn) as f:
        rd=csv.reader(f); genes=[g.split(" (")[0] for g in next(rd)[1:]]; lines=[]; M=[]
        for r in rd:
            lines.append(r[0]); M.append([float(x) if x else np.nan for x in r[1:]])
    return lines, genes, np.array(M,dtype=np.float32)
dl,dg,D=load("CRISPRGeneEffect.csv"); print("dep",D.shape)
el,eg,E=load("CCLE_expression.csv"); print("expr",E.shape)
ml,mg,MU=load("CCLE_mutations_damaging.csv"); print("mut",MU.shape)
di={a:i for i,a in enumerate(dl)}
# targets = priority + white-space genes present in the dependency matrix
tg=set()
for f in ["ti_target_priority.json","ti_whitespace.json"]:
    for r in json.load(open(OUT/f)): tg.add(r["gene"])
dgi={g:i for i,g in enumerate(dg)}
targets=[g for g in tg if g in dgi]; print("targets",len(targets))
tcol=np.array([dgi[g] for g in targets])
def zc(A):  # standardize columns, NaN->0
    mu=np.nanmean(A,0); sd=np.nanstd(A,0); sd[sd==0]=1; Z=(A-mu)/sd; Z[np.isnan(Z)]=0; return Z
# --- expression biomarkers ---
sh=[a for a in el if a in di]; ei=[el.index(a) for a in sh]; dexp=[di[a] for a in sh]
Ez=zc(E[ei]); Dz=zc(D[np.ix_(dexp,tcol)]); n=len(sh)
Ecorr=(Dz.T @ Ez)/n           # targets x exprgenes
# --- mutation biomarkers ---
shm=[a for a in ml if a in di]; mi=[ml.index(a) for a in shm]; dmut=[di[a] for a in shm]
Mb=np.nan_to_num(MU[mi])>0    # boolean mutated
Dm=D[np.ix_(dmut,tcol)]; Dm=np.nan_to_num(Dm, nan=np.nanmean(Dm))
cnt=Mb.sum(0); keep=cnt>=15
mutsum=(Mb.T.astype(np.float32) @ Dm)              # mutgenes x targets (sum dep in mutant lines)
mutmean=mutsum/np.maximum(cnt[:,None],1)
totmean=Dm.mean(0)
wtmean=(Dm.sum(0)[None,:]-mutsum)/np.maximum((len(shm)-cnt)[:,None],1)
delta=mutmean-wtmean                                # mutgenes x targets
bm={}
for k,g in enumerate(targets):
    ec=Ecorr[k]; order=np.argsort(ec)               # most negative first = sensitivity
    expr=[[eg[j],round(float(ec[j]),2)] for j in list(order[:5])+list(order[-2:]) if abs(ec[j])>0.25 and eg[j]!=g]
    dl_=delta[:,k].copy(); dl_[~keep]=0; mo=np.argsort(dl_)
    mut=[[mg[j],round(float(dl_[j]),2)] for j in list(mo[:5])+list(mo[-2:]) if abs(dl_[j])>0.2]
    if expr or mut: bm[g]={"expr":expr,"mut":mut}
json.dump(bm,open(OUT/"biomarkers.json","w"))
print("biomarkers for",len(bm),"targets")
# validation: MDM2 should be TP53-mutation RESISTANT (positive delta); BRAF sensitive to BRAF mut
for chk in ["MDM2","BRAF"]:
    if chk in bm: print(chk,"mut-biomarkers:",bm[chk]["mut"][:4])
