"""Does "active part + kinetics" do just fine for cross-org gene essentiality?

Operationalize the user's two invariants and test on the SAME clean held-out
truth used before (mtub / DeJesus 2017), train on the 8 feba pilots.

  ACTIVE PART  = Pfam-domain composition (multi-hot over a vocab built ONLY from
                 training orgs). The domain is the functional/active module and is
                 cross-organism by construction. This is the conserved invariant.
  KINETICS     = sequence-implied flexibility / dynamics descriptors (fold-coupled,
                 the PARTIALLY-conserved part). NOTE: real kcat is NOT conserved
                 (Bar-Even 2011, spans 6-8 orders), so we do not pretend to have it;
                 these are honest dynamics proxies, not turnover numbers.

Ablation (cross-org mtub DeJesus AUC), each trained the same way:
   conservation | ESM | active(Pfam) | kinetics | active+kinetics | +ESM | +cons | ALL
Answer: do active+kinetics MATCH/BEAT ESM+conservation (0.768)?  Are they orthogonal?
"""
import csv, json, sqlite3
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import torch, torch.nn as nn
INP=Path("/home/user/cell/data/external_data/transformer_inputs"); OUT=Path("/home/user/cell/outputs/orphan")
torch.manual_seed(0); np.random.seed(0)
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db"); cur=con.cursor()
PILOT=["Putida","Keio","Koxy","Caulo","Btheta","MR1","PV4","Dda3937"]
LAB="/home/user/cell/data/drive_import/labels/labels.csv"

# ---- sequence machinery (translate genes from scaffolds) ----
CODON={'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L','ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V','TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P','ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A','TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E','TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R','AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
def translate(nt):
    s=''.join(CODON.get(nt[i:i+3],'X') for i in range(0,len(nt)-2,3))
    return s[:-1] if s.endswith('*') else s
def rc(s):
    t={'A':'T','T':'A','G':'C','C':'G','N':'N'}; return ''.join(t.get(c,'N') for c in reversed(s))

# Vihinen flexibility scale & Kyte-Doolittle hydropathy
FLEX={'A':0.357,'R':0.529,'N':0.463,'D':0.511,'C':0.346,'Q':0.493,'E':0.497,'G':0.544,
      'H':0.323,'I':0.462,'L':0.365,'K':0.466,'M':0.295,'F':0.314,'P':0.509,'S':0.507,
      'T':0.444,'W':0.305,'Y':0.420,'V':0.386}
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
    'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
def kinetics_feats(seq):
    if not seq: seq="M"
    n=len(seq); cnt=Counter(seq)
    def fr(s): return sum(cnt[a] for a in s)/n
    flex=np.mean([FLEX.get(a,0.42) for a in seq])
    kd=np.mean([KD.get(a,0.0) for a in seq])
    # net charge (D,E -1; K,R +1; H +0.1), aromatic, small/flexible (G,S,A), rigid (P), CYS (disulfide/rigidity)
    charge=(fr('KR')-fr('DE'))
    return np.array([np.log1p(n), flex, kd, charge, fr('FWY'), fr('GS'),
                     fr('P'), fr('C'), fr('AGS'), fr('ILVM')], np.float32)

def org_sequences(fb, g_keys):
    scaf={r[0]:r[1] for r in cur.execute("SELECT scaffoldId,sequence FROM ScaffoldSeq WHERE orgId=?",(fb,))}
    coord={lid:(sc,b,e,st) for lid,sc,b,e,st in
           cur.execute("SELECT locusId,scaffoldId,begin,end,strand FROM Gene WHERE orgId=? AND type=1",(fb,))}
    seqs=[]
    for k in g_keys:
        c=coord.get(k)
        if not c: seqs.append("M"); continue
        sc,b,e,st=c; s=scaf.get(sc,""); nt=s[b-1:e]
        if st=='-': nt=rc(nt)
        seqs.append((translate(nt) or "M")[:600])
    return seqs

def org_pfam(fb):
    d=defaultdict(set)
    for lid,dom in cur.execute("SELECT locusId,domainId FROM GeneDomain WHERE orgId=? AND domainDb='PFam'",(fb,)):
        d[lid].add(dom)
    return d

# ---- labels ----
def feba_label(fb,g_keys):
    seen=set(); minf={}
    for lid,fit in cur.execute("SELECT locusId,fit FROM GeneFitness WHERE orgId=?",(fb,)):
        if fit is None: continue
        seen.add(lid)
        if lid not in minf or fit<minf[lid]: minf[lid]=fit
    return np.array([1 if ((k not in seen) or (minf.get(k,0.0)<-3.0)) else 0 for k in g_keys])

# ---- load pilots (ESM + keys), build pfam vocab from TRAIN orgs only ----
print("loading pilots...",flush=True)
pilot={}
domcount=Counter()
for fb in PILOT:
    d=np.load(INP/f"{fb}_esm.npz",allow_pickle=True)
    gk=[str(k) for k in d["g_keys"]]
    pfam=org_pfam(fb)
    for k in gk:
        for dm in pfam.get(k,()): domcount[dm]+=1
    pilot[fb]=dict(gk=gk, esm=d["G_esm"].astype(np.float32), pfam=pfam)
VOCAB=[dm for dm,_ in domcount.most_common(400)]; vidx={dm:i for i,dm in enumerate(VOCAB)}
print(f"  Pfam vocab (top, train-only): {len(VOCAB)}")
def pfam_vec(pfam,k):
    v=np.zeros(len(VOCAB),np.float32)
    for dm in pfam.get(k,()):
        if dm in vidx: v[vidx[dm]]=1.0
    return v

# build train matrices
Xesm=[];Xpf=[];Xkin=[];Y=[]
for fb in PILOT:
    gk=pilot[fb]["gk"]; pfam=pilot[fb]["pfam"]
    seqs=org_sequences(fb,gk)
    Xesm.append(pilot[fb]["esm"])
    Xpf.append(np.stack([pfam_vec(pfam,k) for k in gk]))
    Xkin.append(np.stack([kinetics_feats(s) for s in seqs]))
    Y.append(feba_label(fb,gk))
    print(f"  {fb:<8} {len(gk)} genes  ess={Y[-1].sum()}",flush=True)
Xesm=np.concatenate(Xesm); Xpf=np.concatenate(Xpf); Xkin=np.concatenate(Xkin); Y=np.concatenate(Y)

# ---- mtub eval (DeJesus truth) ----
m=np.load(INP/"MycoTube_esm.npz",allow_pickle=True); mk=[str(k) for k in m["g_keys"]]
mpfam=org_pfam("MycoTube"); mseq=org_sequences("MycoTube",mk)
Mesm=m["G_esm"].astype(np.float32)
Mpf=np.stack([pfam_vec(mpfam,k) for k in mk]); Mkin=np.stack([kinetics_feats(s) for s in mseq])
dej={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open(LAB))
     if r["organism"]=="mtub" and r["source"]=="DeJesus 2017 (MTBseq)"}

# conservation via feba Ortholog projection of panel labels
lab=defaultdict(dict)
for r in csv.DictReader(open(LAB)):
    if r["organism"]=="mtub": continue
    lab[r["organism"].replace("beril_","")][r["locus_tag"]]=int(r["essential"])
orth=defaultdict(list)
for o2,l2,l1 in cur.execute("SELECT orgId2,locusId2,locusId1 FROM Ortholog WHERE orgId1='MycoTube'"):
    orth[l1].append((o2,l2))
cons={}
for g in mk:
    vals=[lab[o][l] for o,l in orth.get(g,[]) if o in lab and l in lab[o]]
    if vals: cons[g]=np.mean(vals)

# ---- net + train ----
class Net(nn.Module):
    def __init__(s,d,h=256): super().__init__(); s.f=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,x): return s.f(x).squeeze(-1)
def auc(p,y):
    p=np.asarray(p,float); y=np.asarray(y)
    if y.sum()==0 or y.sum()==len(y): return float('nan')
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(len(p)); mm=y==1; pos=mm.sum()
    return float((r[mm].sum()-pos*(pos-1)/2)/(pos*(~mm).sum()))
def train_pred(Xtr,ytr,Xte,epochs=30):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-9]=1
    Xn=(Xtr-mu)/sd; Xen=(Xte-mu)/sd
    net=Net(Xtr.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    p1=ytr.mean(); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-p1)/max(p1,1e-6)))
    Xt=torch.from_numpy(Xn).float(); yt=torch.from_numpy(ytr).float()
    for ep in range(epochs):
        idx=torch.randperm(len(Xt)); net.train()
        for i in range(0,len(idx),2048):
            j=idx[i:i+2048]; ls=lf(net(Xt[j]),yt[j]); opt.zero_grad(); ls.backward(); opt.step()
    net.eval()
    with torch.no_grad(): return torch.sigmoid(net(torch.from_numpy(Xen).float())).numpy()

# eval set: genes with DeJesus truth AND conservation AND (all features available)
common=[i for i,k in enumerate(mk) if k in dej and k in cons]
y=np.array([dej[mk[i]] for i in common])
print(f"\neval set (mtub DeJesus & conservation): n={len(common)} ess={int(y.sum())} ({y.mean():.1%})")
cvec=np.array([cons[mk[i]] for i in common])

FE={"ESM":(Xesm,Mesm),"active":(Xpf,Mpf),"kinetics":(Xkin,Mkin)}
def block(names):
    Xtr=np.concatenate([FE[n][0] for n in names],1)
    Xte=np.concatenate([FE[n][1][common] for n in names],1)
    return train_pred(Xtr,Y,Xte)
def z(a): s=a.std(); return (a-a.mean())/(s if s>1e-9 else 1)

results={}
# conservation alone (no training needed)
results["conservation"]=auc(cvec,y)
for combo in [["active"],["kinetics"],["active","kinetics"],["ESM"],
              ["ESM","active"],["ESM","active","kinetics"]]:
    name="+".join(combo); results[name]=auc(block(combo),y)
# add conservation (z-sum fusion with the model score)
for combo in [["active","kinetics"],["ESM"],["ESM","active","kinetics"]]:
    pred=block(combo); name="+".join(combo)+"+cons"
    results[name]=auc(z(pred)+z(cvec),y)

print("\n=== cross-org mtub (DeJesus) AUC ===")
for k in ["conservation","active","kinetics","active+kinetics","ESM",
          "ESM+active","ESM+active+kinetics",
          "active+kinetics+cons","ESM+cons","ESM+active+kinetics+cons"]:
    if k in results: print(f"  {k:<28} {results[k]:.3f}")
json.dump(dict(n=len(common),ess=int(y.sum()),pfam_vocab=len(VOCAB),results={k:round(v,3) for k,v in results.items()}),
          open(OUT/"active_kinetics.json","w"),indent=2)
print(f"\nwrote {OUT}/active_kinetics.json")
