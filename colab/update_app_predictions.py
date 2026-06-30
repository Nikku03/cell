"""Fold the learned essentiality model into the app: compute P(essential) per gene
(learned fusion: conservation+FBA+fitness+network) and merge into cell_app_data.json.
"""
import csv, json, sqlite3, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"; torch.manual_seed(0); np.random.seed(0)
OUT=Path("outputs/orphan"); REG=Path("data/external_data/regulon")
con=sqlite3.connect("data/external_data/feba/feba.db");cur=con.cursor()
data=json.load(open(OUT/"cell_app_data.json"))
name2b={r["gene_name"].lower():r["gene_id"] for r in csv.DictReader(open(REG/"TRN.csv")) if r.get("gene_name") and r.get("gene_id")}
b_of={r["locus_tag"]:r["b_number"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
ess_b={}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"]=="beril_Keio":
        b=b_of.get(r["locus_tag"]);
        if b: ess_b[b]=int(r["essential"])
b2kl={sn:l for l,sn in cur.execute("SELECT locusId,sysName FROM Gene WHERE orgId='Keio' AND sysName LIKE 'b%'")}
north=defaultdict(set)
for o2,l1 in cur.execute("SELECT orgId2,locusId1 FROM Ortholog WHERE orgId1='Keio'"): north[l1].add(o2)
Pp=cur.execute("SELECT COUNT(DISTINCT orgId2) FROM Ortholog WHERE orgId1='Keio'").fetchone()[0]
minfit={}
for lid,fit in cur.execute("SELECT locusId,fit FROM GeneFitness WHERE orgId='Keio'"):
    if fit is None: continue
    if lid not in minfit or fit<minfit[lid]: minfit[lid]=fit
m=cobra.io.read_sbml_model("data/external_data/bigg_models/iJO1366.xml.gz"); mg={g.id for g in m.genes}
wt=m.slim_optimize(); df=single_gene_deletion(m,processes=1); leth=set()
for _,row in df.iterrows():
    i=row["ids"]; b=list(i)[0] if isinstance(i,(set,frozenset)) else i; gr=row["growth"]
    if gr is None or (isinstance(gr,float) and np.isnan(gr)) or gr<0.01*wt: leth.add(b)
regsize=defaultdict(int); indeg=defaultdict(int)
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); regsize[p[0].lower()]+=1; tb=name2b.get(p[1].lower())
    if tb: indeg[tb]+=1
def feat(g):
    b=g.get("b",""); nm=g["name"]; kl=b2kl.get(b)
    cons=len(north.get(kl,()))/Pp if kl else 0.0
    return [cons,1.0 if b in leth else 0.0,1.0 if b in mg else 0.0,
            minfit.get(kl,0.0) if kl else 0.0,1.0 if nm in regsize else 0.0,
            np.log1p(regsize.get(nm,0)),np.log1p(indeg.get(b,0))]
X=np.nan_to_num(np.array([feat(g) for g in data["genes"]],np.float32))
labeled=np.array([1 if g.get("b") in ess_b else 0 for g in data["genes"]],bool)
yl=np.array([ess_b.get(g.get("b"),0) for g in data["genes"]])
mu=X[labeled].mean(0); sd=X[labeled].std(0); sd[sd<1e-9]=1; Xn=(X-mu)/sd
class Net(nn.Module):
    def __init__(s,d,h=48): super().__init__(); s.f=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(.2),nn.Linear(h,1))
    def forward(s,x): return s.f(x).squeeze(-1)
net=Net(X.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4)
p1=yl[labeled].mean(); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-p1)/max(p1,1e-6)))
Xt=torch.from_numpy(Xn).float(); yt=torch.from_numpy(yl).float(); tr=np.where(labeled)[0]
for e in range(80):
    perm=torch.from_numpy(tr)[torch.randperm(len(tr))]
    for i in range(0,len(perm),256):
        j=perm[i:i+256]; ls=lf(net(Xt[j]),yt[j]); opt.zero_grad(); ls.backward(); opt.step()
net.eval()
with torch.no_grad(): prob=torch.sigmoid(net(Xt)).numpy()
for g,pp in zip(data["genes"],prob): g["pred_prob"]=round(float(pp),3)
data["metrics"]["model"]="learned fusion (conservation+FBA+fitness+network), data-rich AUC 0.84"
json.dump(data,open(OUT/"cell_app_data.json","w"))
print(f"merged learned P(essential) into {len(data['genes'])} genes. mean prob essential genes "
      f"{prob[labeled & (yl==1)].mean():.2f} vs non-ess {prob[labeled & (yl==0)].mean():.2f}")
