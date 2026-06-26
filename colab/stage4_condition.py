"""STAGE 4: compositional condition encoder with TOKENIZED STRESSES.

Stage 3 found the condition encoder collapsed 118 stress chemicals to 1 bit.
Fix: each condition = a SET of tokens = {media compounds} U {stress/nutrient
chemicals from condition_1..4} U {aerobic}. The XATTN gene query attends over
this richer set -> can learn gene<->stress couplings.

Gene side: ESM embeddings (Stage 3, frozen). Re-run the gate:
  R_dc (double-centered residual), defect_AUC (target >0.6), heldgene_R.
Head-to-head: media-only condition tokens (Stage 3) vs media+stress (Stage 4).
"""
import sqlite3, json, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch, torch.nn as nn
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db"); cur=con.cursor()
INP=Path("/home/user/cell/data/external_data/transformer_inputs"); OUT=Path("/home/user/cell/outputs/orphan")
torch.manual_seed(0); np.random.seed(0)
ORGS=["Btheta","Keio","Putida"]

# ---- build a GLOBAL token vocab: media compounds + condition chemicals ----
media_comps=defaultdict(set)
for m,c in cur.execute("SELECT media,compound FROM MediaComponents"): media_comps[m].add(c)
comp_count=Counter()
for s in media_comps.values():
    for c in s: comp_count[c]+=1
TOPCOMP=[c for c,_ in comp_count.most_common(200)]
# condition chemicals across pilot
cond_chem=Counter()
for fb in ORGS:
    for row in cur.execute("SELECT condition_1,condition_2,condition_3,condition_4 FROM Experiment WHERE orgId=?",(fb,)):
        for x in row:
            if x and x not in ("None",""): cond_chem[x]+=1
TOPCHEM=[c for c,_ in cond_chem.most_common(400)]
# vocab: 0..199 media, 200..(200+len(TOPCHEM)-1) chemicals, then aerobic token
comp2i={c:i for i,c in enumerate(TOPCOMP)}
chem2i={c:200+i for i,c in enumerate(TOPCHEM)}
AERO=200+len(TOPCHEM); VOCAB=AERO+1; PAD=VOCAB
print(f"vocab: 200 media + {len(TOPCHEM)} stress/nutrient chemicals + aerobic = {VOCAB} (PAD={PAD})")

def cond_tokens(fb):
    """expName -> (media_only_tokens, media+stress_tokens)"""
    out={}
    for row in cur.execute("SELECT expName,media,aerobic,condition_1,condition_2,condition_3,condition_4 FROM Experiment WHERE orgId=?",(fb,)):
        exp,media,aer=row[0],row[1],row[2]
        base=[comp2i[c] for c in media_comps.get(media,set()) if c in comp2i]
        if aer=="aerobic": base.append(AERO)
        chems=[chem2i[x] for x in row[3:] if x in chem2i]
        media_only=base if base else [PAD-1]
        full=base+chems if (base+chems) else [PAD-1]
        out[exp]=(np.array(media_only,dtype=np.int64), np.array(full,dtype=np.int64))
    return out

def load(fb):
    d=np.load(INP/f"{fb}.npz",allow_pickle=True)
    esm=np.load(INP/f"{fb}_esm.npz",allow_pickle=True)["G_esm"].astype(np.float32)
    c_keys=[str(k) for k in d["c_keys"]]
    ct=cond_tokens(fb)
    MAXC=80
    c0=np.full((len(c_keys),MAXC),PAD,np.int64)  # media-only
    c1=np.full((len(c_keys),MAXC),PAD,np.int64)  # media+stress
    for i,ck in enumerate(c_keys):
        mo,fu=ct.get(ck,(np.array([PAD-1]),np.array([PAD-1])))
        mo=mo[:MAXC]; fu=fu[:MAXC]; c0[i,:len(mo)]=mo; c1[i,:len(fu)]=fu
    return dict(Gesm=esm, gi=d["gi"].astype(np.int64), ci=d["ci"].astype(np.int64),
                fit=d["fit"].astype(np.float32), t=d["t"].astype(np.float32),
                tok_media=c0, tok_full=c1, nG=esm.shape[0], nC=len(c_keys))

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

def train(d,tokkey,mask_tr,mask_ev,epochs=4,bs=8192):
    net=XAttn(d["Gesm"].shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5); lf=nn.HuberLoss(delta=1.0,reduction='none')
    TOK=d[tokkey]; tri=np.nonzero(mask_tr)[0]
    for ep in range(epochs):
        np.random.shuffle(tri); net.train()
        for i in range(0,len(tri),bs):
            sel=tri[i:i+bs]
            gv=torch.from_numpy(d["Gesm"][d["gi"][sel]]); comp=torch.from_numpy(TOK[d["ci"][sel]])
            y=torch.from_numpy(d["fit"][sel]); w=torch.from_numpy(np.clip(np.abs(d["t"][sel]),0,10))
            ls=(lf(net(gv,comp),y)*w).mean(); opt.zero_grad(); ls.backward(); opt.step()
    net.eval(); evi=np.nonzero(mask_ev)[0]; preds=[]
    with torch.no_grad():
        for i in range(0,len(evi),bs):
            sel=evi[i:i+bs]
            gv=torch.from_numpy(d["Gesm"][d["gi"][sel]]); comp=torch.from_numpy(TOK[d["ci"][sel]])
            preds.append(net(gv,comp).numpy())
    return np.concatenate(preds),evi

res=[]
print(f"\n{'org':<8}{'cond_enc':<14}{'R_dc':>7}{'defect_AUC':>11}")
for fb in ORGS:
    d=load(fb); nC=d["nC"]; nG=d["nG"]
    hold=np.zeros(nC,bool); hold[np.random.choice(nC,int(nC*0.2),replace=False)]=True
    tr=~hold[d["ci"]]; ev=hold[d["ci"]]
    gm=np.zeros(nG); gc=np.zeros(nG); cm=np.zeros(nC); cc=np.zeros(nC)
    for gix,cix,f in zip(d["gi"][tr],d["ci"][tr],d["fit"][tr]):
        gm[gix]+=f; gc[gix]+=1; cm[cix]+=f; cc[cix]+=1
    gm/=np.maximum(gc,1); cm/=np.maximum(cc,1); glob=d["fit"][tr].mean()
    for tokkey,enc in [("tok_media","ESM-XATTN media-only"),("tok_full","ESM-XATTN +stress")]:
        preds,evi=train(d,tokkey,tr,ev)
        yt=d["fit"][evi]; gmv=gm[d["gi"][evi]]; cmv=cm[d["ci"][evi]]
        dc_true=yt-gmv-cmv+glob; dc_pred=preds-gmv-cmv+glob
        R_dc=pear(dc_pred,dc_true); ydef=(dc_true<-1).astype(int); aucd=auc(-dc_pred,ydef)
        res.append(dict(org=fb,cond_enc=enc,R_dc=round(R_dc,3),defect_AUC=round(aucd,3),n_def=int(ydef.sum())))
        print(f"  {fb:<6}{enc:<22}{R_dc:>7.3f}{aucd:>11.3f}",flush=True)

print("\nMEAN by condition encoder:")
for enc in ["ESM-XATTN media-only","ESM-XATTN +stress"]:
    rs=[r for r in res if r["cond_enc"]==enc]
    print(f"  {enc:<22} R_dc={np.mean([r['R_dc'] for r in rs]):.3f}  defect_AUC={np.mean([r['defect_AUC'] for r in rs]):.3f}")
json.dump(res, open(OUT/"stage4_condition.json","w"), indent=2)
print("\nwrote outputs/orphan/stage4_condition.json")
