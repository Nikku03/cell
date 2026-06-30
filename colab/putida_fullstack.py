"""Generalization proof: run the full-stack essentiality fusion on a SECOND
organism, P. putida (non-E.coli), with its own data. conservation + FBA(iJN1463)
+ feba fitness -> learned fusion, 5-fold OOF, vs beril_Putida truth.
"""
import csv, sqlite3, warnings
from collections import defaultdict
import numpy as np, torch, torch.nn as nn
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"; torch.manual_seed(0); np.random.seed(0)
# truth
ess={}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"]=="beril_Putida": ess[r["locus_tag"]]=int(r["essential"])
# conservation (leak-free)
cons={}
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r["organism"]=="beril_Putida":
        v=r.get("family_frac_essential_leakfree","")
        if v not in ("","NA"):
            try:
                fv=float(v)
                if fv==fv: cons[r["locus_tag"]]=fv
            except: pass
# FBA iJN1463
m=cobra.io.read_sbml_model("data/external_data/bigg_models/iJN1463.xml.gz"); model_g={g.id for g in m.genes}
wt=m.slim_optimize(); df=single_gene_deletion(m,processes=1); leth=set()
for _,row in df.iterrows():
    i=row["ids"]; g=list(i)[0] if isinstance(i,(set,frozenset)) else i; gr=row["growth"]
    if gr is None or (isinstance(gr,float) and np.isnan(gr)) or gr<0.01*wt: leth.add(g)
print(f"iJN1463: {len(model_g)} genes, WT {wt:.2f}, FBA-lethal {len(leth)}")
# feba fitness (Putida)
con=sqlite3.connect("data/external_data/feba/feba.db");cur=con.cursor()
minfit={}
for lid,fit in cur.execute("SELECT locusId,fit FROM GeneFitness WHERE orgId='Putida'"):
    if fit is None: continue
    if lid not in minfit or fit<minfit[lid]: minfit[lid]=fit
print(f"feba Putida fitness genes: {len(minfit)}")
# assemble (genes with truth)
genes=[g for g in ess]
y=np.array([ess[g] for g in genes])
def feat(g):
    return [cons.get(g,0.0),1.0 if g in leth else 0.0,1.0 if g in model_g else 0.0,minfit.get(g,0.0)]
X=np.nan_to_num(np.array([feat(g) for g in genes],np.float32))
print(f"genes {len(genes)}  essential {y.sum()} ({y.mean():.2%})  (have conservation {sum(1 for g in genes if g in cons)}, fitness {sum(1 for g in genes if g in minfit)})")
def auc(p,yy):
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(len(p)); mm=yy==1
    if mm.sum()==0 or mm.sum()==len(yy): return float('nan')
    return float((r[mm].sum()-mm.sum()*(mm.sum()-1)/2)/(mm.sum()*(~mm).sum()))
class Net(nn.Module):
    def __init__(s,dd,h=48): super().__init__(); s.f=nn.Sequential(nn.Linear(dd,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,1))
    def forward(s,x): return s.f(x).squeeze(-1)
def cv(X,y):
    idx=np.arange(len(X)); np.random.shuffle(idx); oof=np.zeros(len(X))
    for f in np.array_split(idx,5):
        tr=np.setdiff1d(idx,f); mu=X[tr].mean(0); sd=X[tr].std(0); sd[sd<1e-9]=1; Xn=(X-mu)/sd
        net=Net(X.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4)
        p1=y[tr].mean(); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-p1)/max(p1,1e-6)))
        Xt=torch.from_numpy(Xn).float(); yt=torch.from_numpy(y).float()
        for e in range(60):
            perm=torch.from_numpy(tr)[torch.randperm(len(tr))]
            for i in range(0,len(perm),256):
                j=perm[i:i+256]; ls=lf(net(Xt[j]),yt[j]); opt.zero_grad(); ls.backward(); opt.step()
        with torch.no_grad(): oof[f]=torch.sigmoid(net(Xt[torch.from_numpy(f)])).numpy()
    return oof
def cov(s,y,t=.9):
    o=np.argsort(-s); ys=y[o]; pr=np.cumsum(ys)/np.arange(1,len(ys)+1); en=0
    for i in range(len(pr)-1,-1,-1):
        if pr[i]>=t: en=i+1;break
    o2=np.argsort(s); yl=y[o2]; p2=np.cumsum(1-yl)/np.arange(1,len(yl)+1); ne=0
    for i in range(len(p2)-1,-1,-1):
        if p2[i]>=t: ne=i+1;break
    return en/len(y),ne/len(y)
print("\n=== P. putida full-stack essentiality (5-fold OOF, vs truth) ===")
import json; res={}
for name,cols in [("conservation only",[0]),("genome-only (cons+FBA)",[0,1,2]),("data-rich (+fitness)",[0,1,2,3])]:
    oof=cv(X[:,cols],y); a=auc(oof,y); ec,nc=cov(oof,y)
    res[name]=dict(auc=round(a,3),total_cov=round(ec+nc,3))
    print(f"  {name:<24} AUC={a:.3f}  cov@90%: ess {100*ec:.0f}% + ne {100*nc:.0f}% = {100*(ec+nc):.0f}%")
json.dump(dict(organism="P. putida KT2440",n=len(genes),ess_rate=round(float(y.mean()),3),results=res),
          open("outputs/orphan/putida_fullstack.json","w"),indent=2)
print("\nwrote outputs/orphan/putida_fullstack.json")
