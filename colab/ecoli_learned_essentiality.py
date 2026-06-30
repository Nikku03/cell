"""Flagship learned essentiality: fuse ESM protein embedding + conservation + FBA
necessity + feba fitness + network features into one classifier (5-fold OOF) for
E. coli. Beat the naive OR-fusion (precision 0.51 / recall 0.36) and report the
coverage at precision>=0.90.
"""
import csv, sqlite3, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"
torch.manual_seed(0); np.random.seed(0)
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models")
con=sqlite3.connect("data/external_data/feba/feba.db"); cur=con.cursor()
# ESM
d=np.load(OUT/"ecoli_esm.npz",allow_pickle=True); esm_b=[str(x) for x in d["b"]]; ESM=d["E"].astype(np.float32)
esm_idx={b:i for i,b in enumerate(esm_b)}
# Keio truth
b_of={r["locus_tag"]:r["b_number"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
ess_b={}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"]=="beril_Keio":
        b=b_of.get(r["locus_tag"])
        if b: ess_b[b]=int(r["essential"])
# conservation + fitness (feba Keio)
b2kl={sn:l for l,sn in cur.execute("SELECT locusId,sysName FROM Gene WHERE orgId='Keio' AND sysName LIKE 'b%'")}
north=defaultdict(set)
for o2,l1 in cur.execute("SELECT orgId2,locusId1 FROM Ortholog WHERE orgId1='Keio'"): north[l1].add(o2)
Ppanel=cur.execute("SELECT COUNT(DISTINCT orgId2) FROM Ortholog WHERE orgId1='Keio'").fetchone()[0]
minfit={}
for lid,fit in cur.execute("SELECT locusId,fit FROM GeneFitness WHERE orgId='Keio'"):
    if fit is None: continue
    if lid not in minfit or fit<minfit[lid]: minfit[lid]=fit
def conservation(b): kl=b2kl.get(b); return len(north.get(kl,()))/Ppanel if kl else 0.0
def fitfeat(b): kl=b2kl.get(b); return minfit.get(kl,0.0) if kl else 0.0
# FBA lethal (glucose) + metabolic
m=cobra.io.read_sbml_model(str(MODELS/"iJO1366.xml.gz")); model_b={g.id for g in m.genes}
wt=m.slim_optimize(); df=single_gene_deletion(m,processes=1); fba_lethal=set()
for _,row in df.iterrows():
    ids=row["ids"]; b=list(ids)[0] if isinstance(ids,(set,frozenset)) else ids
    gr=row["growth"]
    if gr is None or (isinstance(gr,float) and np.isnan(gr)) or gr<0.01*wt: fba_lethal.add(b)
# network features
name2b={r["gene_name"].lower():r["gene_id"] for r in csv.DictReader(open(REG/"TRN.csv")) if r.get("gene_name") and r.get("gene_id")}
regsize=defaultdict(int); indeg=defaultdict(int)
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); regsize[p[0].lower()]+=1;
    tb=name2b.get(p[1].lower())
    if tb: indeg[tb]+=1
b2name={b:n for n,b in name2b.items()}

# assemble
genes=[b for b in ess_b if b in esm_idx]
y=np.array([ess_b[b] for b in genes])
def extra(b):
    nm=b2name.get(b,"")
    return [conservation(b),1.0 if b in fba_lethal else 0.0,1.0 if b in model_b else 0.0,
            fitfeat(b),1.0 if nm in regsize else 0.0,np.log1p(regsize.get(nm,0)),np.log1p(indeg.get(b,0))]
Xe=np.array([extra(b) for b in genes],np.float32)
Xfull=np.hstack([ESM[[esm_idx[b] for b in genes]],Xe])
print(f"genes {len(genes)}  essential {y.sum()} ({y.mean():.2%})  feat dim {Xfull.shape[1]}")

def auc(p,yy):
    p=np.asarray(p);yy=np.asarray(yy)
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(len(p)); mm=yy==1
    return float((r[mm].sum()-mm.sum()*(mm.sum()-1)/2)/(mm.sum()*(~mm).sum()))
class Net(nn.Module):
    def __init__(s,dd,h=128): super().__init__(); s.f=nn.Sequential(nn.Linear(dd,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,x): return s.f(x).squeeze(-1)
def cv(X,y,ep=40):
    idx=np.arange(len(X)); np.random.shuffle(idx); oof=np.zeros(len(X))
    for f in np.array_split(idx,5):
        tr=np.setdiff1d(idx,f); mu=X[tr].mean(0); sd=X[tr].std(0); sd[sd<1e-9]=1
        Xn=(X-mu)/sd; net=Net(X.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
        p1=y[tr].mean(); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-p1)/max(p1,1e-6)))
        Xt=torch.from_numpy(Xn).float(); yt=torch.from_numpy(y).float()
        for e in range(ep):
            net.train(); perm=torch.from_numpy(tr)[torch.randperm(len(tr))]
            for i in range(0,len(perm),256):
                j=perm[i:i+256]; ls=lf(net(Xt[j]),yt[j]); opt.zero_grad(); ls.backward(); opt.step()
        net.eval()
        with torch.no_grad(): oof[f]=torch.sigmoid(net(Xt[torch.from_numpy(f)])).numpy()
    return oof
def cov_at_p(s,y,t=0.90):
    o=np.argsort(-s); ys=y[o]; cum=np.cumsum(ys); n=np.arange(1,len(ys)+1); prec=cum/n
    en=0
    for i in range(len(prec)-1,-1,-1):
        if prec[i]>=t: en=i+1; break
    o2=np.argsort(s); yl=y[o2]; c2=np.cumsum(1-yl); n2=np.arange(1,len(yl)+1); p2=c2/n2
    nn_=0
    for i in range(len(p2)-1,-1,-1):
        if p2[i]>=t: nn_=i+1; break
    return en/len(y),nn_/len(y)

print("\n=== learned essentiality (5-fold OOF, vs Keio) ===")
res={}
for name,X in [("conservation only",Xe[:,:1]),("non-ESM features",Xe),("ESM only",ESM[[esm_idx[b] for b in genes]]),("FULL (ESM+all)",Xfull)]:
    oof=cv(X,y); a=auc(oof,y); ec,nc=cov_at_p(oof,y,0.90)
    res[name]=dict(auc=round(a,3),essCov=round(ec,3),neCov=round(nc,3),total=round(ec+nc,3))
    print(f"  {name:<20} AUC={a:.3f}  cov@90%: ess {100*ec:.0f}% + ne {100*nc:.0f}% = {100*(ec+nc):.0f}%")
import json
json.dump(res,open(OUT/"ecoli_learned_essentiality.json","w"),indent=2)
print("\nwrote outputs/orphan/ecoli_learned_essentiality.json")
