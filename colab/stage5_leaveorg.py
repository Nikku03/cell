"""STAGE 5 (part A+C): the real generalization tests.

A) LEAVE-ONE-ORG-OUT: train ESM-XATTN(+stress) on 7 pilot orgs, predict the
   held org's conditional fitness. Metrics on held org: R_dc (double-centered),
   defect_AUC. This is the "new organism" test -- where ortholog transfer always
   collapsed to conservation. Sequence(ESM)+condition based, so a genuine test.
C) HELD-OUT-CHEMICAL: within an org, hold out whole stress chemicals; does
   +stress still beat media-only on those NOVEL chemicals? (memorize vs generalize)

Streamed to avoid OOM. Global condition vocab shared across orgs.
"""
import sqlite3, json, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch, torch.nn as nn
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db"); cur=con.cursor()
INP=Path("/home/user/cell/data/external_data/transformer_inputs"); OUT=Path("/home/user/cell/outputs/orphan")
torch.manual_seed(0); np.random.seed(0)
PILOT=["Putida","Keio","Koxy","Caulo","Btheta","MR1","PV4","Dda3937"]

# global vocab (media + stress chemicals) -- same as Stage 4
media_comps=defaultdict(set)
for m,c in cur.execute("SELECT media,compound FROM MediaComponents"): media_comps[m].add(c)
cc=Counter()
for s in media_comps.values():
    for c in s: cc[c]+=1
TOPCOMP=[c for c,_ in cc.most_common(200)]
chemcnt=Counter()
for fb in PILOT:
    for row in cur.execute("SELECT condition_1,condition_2,condition_3,condition_4 FROM Experiment WHERE orgId=?",(fb,)):
        for x in row:
            if x and x not in ("None",""): chemcnt[x]+=1
TOPCHEM=[c for c,_ in chemcnt.most_common(400)]
comp2i={c:i for i,c in enumerate(TOPCOMP)}; chem2i={c:200+i for i,c in enumerate(TOPCHEM)}
AERO=200+len(TOPCHEM); VOCAB=AERO+1; PAD=VOCAB; MAXC=80
chem_id_set=set(chem2i.values())
print(f"vocab {VOCAB} (200 media + {len(TOPCHEM)} chem + aero)")

def org_cond_tokens(fb):
    out={}
    for row in cur.execute("SELECT expName,media,aerobic,condition_1,condition_2,condition_3,condition_4 FROM Experiment WHERE orgId=?",(fb,)):
        exp,media,aer=row[0],row[1],row[2]
        toks=[comp2i[c] for c in media_comps.get(media,set()) if c in comp2i]
        if aer=="aerobic": toks.append(AERO)
        toks+=[chem2i[x] for x in row[3:] if x in chem2i]
        out[exp]=np.array(toks if toks else [PAD-1],dtype=np.int64)
    return out

def load(fb):
    d=np.load(INP/f"{fb}.npz",allow_pickle=True)
    esm=np.load(INP/f"{fb}_esm.npz",allow_pickle=True)["G_esm"].astype(np.float32)
    c_keys=[str(k) for k in d["c_keys"]]
    ct=org_cond_tokens(fb)
    TOK=np.full((len(c_keys),MAXC),PAD,np.int64)
    for i,ck in enumerate(c_keys):
        a=ct.get(ck,np.array([PAD-1]))[:MAXC]; TOK[i,:len(a)]=a
    return dict(Gesm=esm, gi=d["gi"].astype(np.int64), ci=d["ci"].astype(np.int64),
                fit=d["fit"].astype(np.float32), t=d["t"].astype(np.float32), TOK=TOK,
                nG=esm.shape[0], nC=len(c_keys))
DATA={fb:load(fb) for fb in PILOT}
DIM=DATA[PILOT[0]]["Gesm"].shape[1]

class XAttn(nn.Module):
    def __init__(s,g,D=96,h=160):
        super().__init__(); s.ge=nn.Sequential(nn.Linear(g,256),nn.ReLU(),nn.Linear(256,D))
        s.emb=nn.Embedding(VOCAB+1,D,padding_idx=PAD); s.q=nn.Linear(D,D); s.k=nn.Linear(D,D); s.v=nn.Linear(D,D)
        s.h=nn.Sequential(nn.Linear(2*D,h),nn.ReLU(),nn.Dropout(0.1),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1)); s.D=D
    def forward(s,gv,comp):
        mask=comp!=PAD; g=s.ge(gv); E=s.emb(comp); q=s.q(g).unsqueeze(1); k=s.k(E); v=s.v(E)
        sc=(q*k).sum(-1)/(s.D**0.5); sc=sc.masked_fill(~mask,-1e9); w=torch.softmax(sc,-1).unsqueeze(-1)
        return s.h(torch.cat([g,(w*v).sum(1)],-1)).squeeze(-1)

def pear(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); a=a-a.mean(); b=b-b.mean()
    return float((a*b).sum()/(np.sqrt((a*a).sum()*(b*b).sum())+1e-12))
def auc(s,y):
    s=np.asarray(s,float); y=np.asarray(y)
    if y.sum()==0 or y.sum()==len(y): return float('nan')
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); p=y==1; pos=p.sum()
    return float((r[p].sum()-pos*(pos-1)/2)/(pos*(~p).sum()))

def gather(rows):
    """rows: list of (fb, idx_array). returns gv, comp, fit, t."""
    gv=[]; comp=[]; fit=[]; tt=[]
    for fb,idx in rows:
        d=DATA[fb]
        gv.append(d["Gesm"][d["gi"][idx]]); comp.append(d["TOK"][d["ci"][idx]])
        fit.append(d["fit"][idx]); tt.append(d["t"][idx])
    return np.concatenate(gv),np.concatenate(comp),np.concatenate(fit),np.concatenate(tt)

def train_pred(train_rows, eval_rows, epochs=2, bs=8192, sample=1_800_000):
    net=XAttn(DIM); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5); lf=nn.HuberLoss(delta=1.0,reduction='none')
    # flatten train index pool
    pool=[]
    for fb,idx in train_rows:
        for j in idx: pool.append((fb,j))
    pool=np.array(pool,dtype=object)
    for ep in range(epochs):
        sel=np.random.permutation(len(pool))
        if sample and len(sel)>sample: sel=sel[:sample]
        net.train()
        for i in range(0,len(sel),bs):
            chunk=pool[sel[i:i+bs]]
            by=defaultdict(list)
            for fb,j in chunk: by[fb].append(j)
            rows=[(fb,np.array(v)) for fb,v in by.items()]
            gv,comp,fit,t=gather(rows)
            gv=torch.from_numpy(gv); comp=torch.from_numpy(comp); y=torch.from_numpy(fit); w=torch.from_numpy(np.clip(np.abs(t),0,10))
            ls=(lf(net(gv,comp),y)*w).mean(); opt.zero_grad(); ls.backward(); opt.step()
    # predict eval
    net.eval(); preds=[]
    with torch.no_grad():
        for fb,idx in eval_rows:
            d=DATA[fb]
            for i in range(0,len(idx),bs):
                s=idx[i:i+bs]
                gv=torch.from_numpy(d["Gesm"][d["gi"][s]]); comp=torch.from_numpy(d["TOK"][d["ci"][s]])
                preds.append(net(gv,comp).numpy())
    return np.concatenate(preds)

# ===== A) leave-one-org-out =====
print(f"\n{'='*60}\n(A) LEAVE-ONE-ORG-OUT (new-organism conditional prediction)\n{'='*60}")
res=[]
for held in ["Keio","Putida","Caulo"]:
    tr=[(fb, np.arange(len(DATA[fb]["fit"]))) for fb in PILOT if fb!=held]
    d=DATA[held]; ev=[(held, np.arange(len(d["fit"])))]
    t0=time.time(); preds=train_pred(tr,ev);
    # double-center using held org's OWN gene/cond means (these are computable for a real new org from its raw data)
    nG=d["nG"]; nC=d["nC"]; gm=np.zeros(nG); gc=np.zeros(nG); cm=np.zeros(nC); ccount=np.zeros(nC)
    for gix,cix,f in zip(d["gi"],d["ci"],d["fit"]): gm[gix]+=f; gc[gix]+=1; cm[cix]+=f; ccount[cix]+=1
    gm/=np.maximum(gc,1); cm/=np.maximum(ccount,1); glob=d["fit"].mean()
    yt=d["fit"]; gmv=gm[d["gi"]]; cmv=cm[d["ci"]]
    dc_true=yt-gmv-cmv+glob; dc_pred=preds-gmv-cmv+glob
    R_dc=pear(dc_pred,dc_true); ydef=(dc_true<-1).astype(int); aucd=auc(-dc_pred,ydef)
    R_agg=pear(preds,yt)
    res.append(dict(test="leave_org",held=held,R_agg=round(R_agg,3),R_dc=round(R_dc,3),defect_AUC=round(aucd,3),n=len(yt)))
    print(f"  held {held:<8} R_agg={R_agg:.3f}  R_dc={R_dc:.3f}  defect_AUC={aucd:.3f}  ({time.time()-t0:.0f}s)",flush=True)

# ===== C) held-out-CHEMICAL (within Btheta, the stress-rich org) =====
print(f"\n{'='*60}\n(C) HELD-OUT-CHEMICAL (novel-stress generalization, Btheta)\n{'='*60}")
fb="Btheta"; d=DATA[fb]
# which conditions involve a held-out chemical?
exp_keys=[str(k) for k in np.load(INP/f"{fb}.npz",allow_pickle=True)["c_keys"]]
# map each condition index -> set of chemical token ids it contains
cond_chems=[set(int(x) for x in d["TOK"][i] if int(x) in chem_id_set) for i in range(d["nC"])]
all_chems=sorted(set().union(*cond_chems)) if cond_chems else []
np.random.shuffle(all_chems)
held_chems=set(all_chems[:max(1,len(all_chems)//5)])  # hold 20% of chemicals
ev_cond=np.array([any(c in held_chems for c in cond_chems[i]) for i in range(d["nC"])])
ev_mask=ev_cond[d["ci"]]; tr_mask=~ev_mask
tr=[(fb, np.nonzero(tr_mask)[0])]; ev=[(fb, np.nonzero(ev_mask)[0])]
preds=train_pred(tr,ev,sample=None)
evi=np.nonzero(ev_mask)[0]; nG=d["nG"]; nC=d["nC"]
gm=np.zeros(nG); gc=np.zeros(nG); cm=np.zeros(nC); ccount=np.zeros(nC)
for gix,cix,f in zip(d["gi"][tr_mask],d["ci"][tr_mask],d["fit"][tr_mask]): gm[gix]+=f; gc[gix]+=1
gm/=np.maximum(gc,1)
# for held chemicals, cond-mean unknown from train; use held org cond-mean computed on eval (allowed: a real org measures its own cond means)
for cix,f in zip(d["ci"][ev_mask],d["fit"][ev_mask]): cm[cix]+=f; ccount[cix]+=1
cm/=np.maximum(ccount,1); glob=d["fit"][tr_mask].mean()
yt=d["fit"][evi]; gmv=gm[d["gi"][evi]]; cmv=cm[d["ci"][evi]]
dc_true=yt-gmv-cmv+glob; dc_pred=preds-gmv-cmv+glob
R_dc=pear(dc_pred,dc_true); ydef=(dc_true<-1).astype(int); aucd=auc(-dc_pred,ydef)
res.append(dict(test="held_chemical",held="Btheta",n_held_chems=len(held_chems),R_dc=round(R_dc,3),defect_AUC=round(aucd,3),n=len(yt)))
print(f"  Btheta novel-chemical ({len(held_chems)} held): R_dc={R_dc:.3f}  defect_AUC={aucd:.3f}  n_eval={len(yt)}",flush=True)

json.dump(res, open(OUT/"stage5_leaveorg.json","w"), indent=2)
print("\nwrote outputs/orphan/stage5_leaveorg.json")
