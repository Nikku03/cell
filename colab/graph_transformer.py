"""Transformer+GNN on the integrated cell. Honest head-to-head: MLP (features only) vs GCN
(graph message-passing) vs GAT (attention on the graph = transformer+GNN). Task: predict
essentiality from integrated features + regulatory/PPI network, 5-fold OOF. Then score the
dark genome. LGNN/dynamics head deferred (needs trajectory data)."""
import csv, gzip, warnings, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings("ignore"); torch.manual_seed(0); np.random.seed(0)
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
rows=list(csv.DictReader(open(OUT/"integrated_cell_human.csv")))
def fv(v,d=0.0):
    try: return float(v)
    except: return d
FEATS=["loeuf","is_tf","regulon_out","regulators_in","ppi_degree","n_pathways","cpg_promoter","enhancers","n_diseases"]
genes=[r["gene"] for r in rows]; idx={g:i for i,g in enumerate(genes)}; N=len(genes)
X=np.array([[fv(r[f]) for f in FEATS] for r in rows],np.float32)
# loeuf missing -> median; log some
X=np.nan_to_num(X,nan=0.0)
for j,f in enumerate(FEATS):
    if f in ("regulon_out","regulators_in","ppi_degree","n_pathways","enhancers","n_diseases"):
        X[:,j]=np.log1p(X[:,j])
mu=X.mean(0); sd=X.std(0); sd[sd<1e-9]=1; X=(X-mu)/sd
y=np.array([1 if r["essential"]=="1" else (0 if r["essential"]=="0" else -1) for r in rows])
lab=np.where(y>=0)[0]
print(f"{N} genes, {len(FEATS)} features, {len(lab)} labeled (ess {int((y==1).sum())}, non {int((y==0).sum())})")
# edges: regulatory (collectri) + PPI (string physical >=700)
E=set()
for r in csv.DictReader(open(H/"collectri.tsv"),delimiter="\t"):
    a,b=r["source_genesymbol"],r["target_genesymbol"]
    if a in idx and b in idx: E.add((idx[a],idx[b])); E.add((idx[b],idx[a]))
ensp2sym={}
for l in gzip.open(H/"string_aliases.txt.gz","rt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=3 and p[2]=="Ensembl_HGNC_symbol": ensp2sym[p[0]]=p[1]
for l in gzip.open(H/"string_physical.txt.gz","rt"):
    if l.startswith("protein1"): continue
    a,b,s=l.split()
    if int(s)<700: continue
    ga=ensp2sym.get(a); gb=ensp2sym.get(b)
    if ga in idx and gb in idx: E.add((idx[ga],idx[gb])); E.add((idx[gb],idx[ga]))
ei=np.array(list(E)).T if E else np.zeros((2,0),int)
src=torch.tensor(ei[0]); dst=torch.tensor(ei[1])
print(f"graph edges (undirected pairs x2): {ei.shape[1]}")
Xt=torch.from_numpy(X); yt=torch.from_numpy(y.astype(np.float32))
deg=torch.zeros(N).index_add_(0,dst,torch.ones(len(dst))).clamp(min=1)
def gconv(h,W):  # mean-aggregate neighbors
    m=h[src]; agg=torch.zeros(h.shape[0],h.shape[1]).index_add_(0,dst,m)/deg.unsqueeze(1)
    return torch.relu(agg@W)
class Net(nn.Module):
    def __init__(s,kind,d=32):
        super().__init__(); s.kind=kind
        s.fin=nn.Linear(len(FEATS),d)
        s.W1=nn.Parameter(torch.randn(d,d)*0.1); s.W2=nn.Parameter(torch.randn(d,d)*0.1)
        s.att=nn.Linear(2*d,1) if kind=="GAT" else None
        s.out=nn.Sequential(nn.ReLU(),nn.Linear(d,1))
    def gat(s,h,W):
        hp=torch.relu(h@W); e=F.leaky_relu(s.att(torch.cat([hp[src],hp[dst]],1)).squeeze(1))
        e=e-e.max(); a=torch.exp(e)
        Z=torch.zeros(h.shape[0]).index_add_(0,dst,a).clamp(min=1e-9)
        a=a/Z[dst]
        agg=torch.zeros_like(hp).index_add_(0,dst,hp[src]*a.unsqueeze(1))
        return torch.relu(agg)
    def forward(s,x):
        h=torch.relu(s.fin(x))
        if s.kind=="MLP":
            h=torch.relu(h@s.W1); h=torch.relu(h@s.W2)
        elif s.kind=="GCN":
            h=gconv(h,s.W1); h=gconv(h,s.W2)
        else:  # GAT = attention on graph (transformer+GNN)
            h=s.gat(h,s.W1); h=s.gat(h,s.W2)
        return s.out(h).squeeze(1)
def auc(p,yy):
    o=np.argsort(p);r=np.empty(len(p));r[o]=np.arange(len(p));m=yy==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))
def run(kind):
    rng=np.random.default_rng(0); li=lab.copy(); rng.shuffle(li); folds=np.array_split(li,5)
    oof=np.full(N,np.nan)
    for k in range(5):
        te=folds[k]; tr=np.concatenate([folds[j] for j in range(5) if j!=k])
        net=Net(kind); opt=torch.optim.AdamW(net.parameters(),lr=5e-3,weight_decay=1e-4)
        p1=y[tr].mean(); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-p1)/max(p1,1e-6)))
        trt=torch.tensor(tr)
        for ep in range(150):
            net.train(); opt.zero_grad(); out=net(Xt); ls=lf(out[trt],yt[trt]); ls.backward(); opt.step()
        net.eval()
        with torch.no_grad(): pr=torch.sigmoid(net(Xt)).numpy()
        oof[te]=pr[te]
    a=auc(oof[lab],y[lab]); return a,oof
print("\n=== 5-fold OOF AUC (predict essentiality) ===")
res={}
for kind in ["MLP","GCN","GAT"]:
    t0=time.time(); a,oof=run(kind); res[kind]=(a,oof); print(f"  {kind:4s} (features{'+graph' if kind!='MLP' else ' only'}{'+attention' if kind=='GAT' else ''}): AUC {a:.3f}  ({time.time()-t0:.0f}s)",flush=True)
# best model -> score dark genes (no essentiality label, low disease/study)
best=max(res,key=lambda k:res[k][0]); _,oof=res[best]
# retrain best on all labeled, score all
net=Net(best); opt=torch.optim.AdamW(net.parameters(),lr=5e-3,weight_decay=1e-4)
p1=y[lab].mean(); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-p1)/max(p1,1e-6))); trt=torch.tensor(lab)
for ep in range(150): net.train();opt.zero_grad();ls=lf(net(Xt)[trt],yt[trt]);ls.backward();opt.step()
net.eval()
with torch.no_grad(): allp=torch.sigmoid(net(Xt)).numpy()
dark=[i for i in range(N) if y[i]<0 and fv(rows[i]["n_diseases"])==0 and fv(rows[i]["n_pathways"])<=1]
darktop=sorted(dark,key=lambda i:-allp[i])[:15]
print(f"\nbest model: {best}. Top dark-genome genes predicted essential:")
print("  "+", ".join(f"{genes[i]}({allp[i]:.2f})" for i in darktop))
import json
json.dump(dict(auc={k:round(v[0],3) for k,v in res.items()},best=best,edges=int(ei.shape[1]),
    labeled=len(lab),top_dark_essential=[genes[i] for i in darktop]),open(OUT/"graph_transformer.json","w"),indent=2)
print("\nwrote graph_transformer.json")
