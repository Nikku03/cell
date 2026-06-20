"""Can co-occurrence channels resolve genes the TRANSFORMER abstains on?

The transformer at brightness>=0.85 confidently calls ~57% of genes cross-clade
at ~92% accuracy. The other ~43% are abstained -- can the 4 co-occurrence
channels (backup, presence-cooccur, co-essentiality, phyletic) call any of
THOSE at deployable precision, and how much does that raise total coverage?

Pipeline:
  1. Train the AF-style transformer leave-one-CLADE-out (5 major clades).
  2. For each held-clade gene, get model probability + brightness (calibrated
     confidence head).
  3. Define transformer-abstained = brightness < 0.85.
  4. For each abstained gene, compute the 4 cooccur channels leak-clean
     (training orgs = orgs NOT in the held clade).
  5. Apply strict promotion: >=2 channels agree -> call.
  6. Combined coverage = transformer-confident + cooccur-rescued; precision
     = weighted over the union.

Output: outputs/orphan/transformer_plus_cooccur_results.json
"""
import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")
rng = np.random.default_rng(0)

# ---- load transformer cache ----
Z = np.load(OUT/"af_msa_cache.npz", allow_pickle=True)
foc=Z["foc"]; msa=Z["msa"]; msa_org=Z["msa_org"]; mask=Z["msa_mask"]; y=Z["y"]
clade=Z["clade"]; orgs=list(Z["orgs"]); meta_org=Z["meta_org"]; lt=Z["lt"]
N=len(y); print(f"cache: {N} genes, {len(orgs)} orgs")

# ---- orthology / labels / cooccur prebuilt ----
gene_og={}; og_present=defaultdict(set)
for r in csv.DictReader(open(DATA/"labels"/"orthology_features.csv")):
    if r["og_id"]:
        gene_og[(r["organism"],r["locus_tag"])]=r["og_id"]
        og_present[r["og_id"]].add(r["organism"])
labels={(r["organism"],r["locus_tag"]):int(r["essential"])
        for r in csv.DictReader(open(DATA/"labels"/"labels.csv"))}
all_orgs=sorted({o for og in og_present.values() for o in og})
O=len(all_orgs); org_idx={o:i for i,o in enumerate(all_orgs)}
all_ogs=sorted(og_present)
G=len(all_ogs); og_idx2={og:j for j,og in enumerate(all_ogs)}
Y=np.full((O,G), np.nan, np.float32)
for (o,l),e in labels.items():
    og=gene_og.get((o,l))
    if og: Y[org_idx[o], og_idx2[og]]=e
P=np.zeros((O,G), np.float32)
for og,orgs_p in og_present.items():
    j=og_idx2[og]
    for o in orgs_p: P[org_idx[o],j]=1.0
cofeat={}
for r in csv.DictReader(open(DATA/"labels"/"cooccurrence_features.csv")):
    try: cofeat[(r["organism"],r["locus_tag"])]=float(r["cooccur_essential_neighbor_frac"])
    except: pass

# Per-gene OG idx + per-gene org-idx in orthology space
cache_to_orth=np.full(len(orgs), -1, dtype=np.int32)
for ci,name in enumerate(orgs):
    if str(name) in org_idx: cache_to_orth[ci]=org_idx[str(name)]
gene_og_j=np.full(N,-1,dtype=np.int32); gene_org_i=np.full(N,-1,dtype=np.int32)
gene_org_name=np.empty(N,dtype=object); gene_lt=np.empty(N,dtype=object)
for i in range(N):
    name=str(orgs[meta_org[i]])
    key=(name,str(lt[i]))
    og=gene_og.get(key)
    if og is not None and og in og_idx2: gene_og_j[i]=og_idx2[og]
    gene_org_i[i]=cache_to_orth[meta_org[i]]
    gene_org_name[i]=name; gene_lt[i]=str(lt[i])

# Candidate partner sets
p_sum=P.sum(0); cand_p=(p_sum>=5)&(p_sum<=O-5); cand_p_idx=np.where(cand_p)[0]
n_lab=(~np.isnan(Y)).sum(0); mean_e=np.nanmean(Y,axis=0)
cand_e=(n_lab>=10)&(mean_e>0.05)&(mean_e<0.95); cand_e_idx=np.where(cand_e)[0]
Pc=P[:,cand_p_idx]-P[:,cand_p_idx].mean(0); Pn=Pc/(Pc.std(0)+1e-9)

# ---- transformer (single-block) ----
H=12; HID=24
def init(Df):
    s=0.3
    return dict(Wq=rng.normal(0,s,(Df,H)).astype(np.float32),
        Wk=rng.normal(0,s,(6,H)).astype(np.float32),
        Wv=rng.normal(0,s,(6,H)).astype(np.float32),
        W1=rng.normal(0,s,(Df+H,HID)).astype(np.float32), b1=np.zeros(HID,np.float32),
        w2=rng.normal(0,s,HID).astype(np.float32), b2=np.float32(0.0),
        wc=rng.normal(0,s,HID).astype(np.float32), bc=np.float32(0.0))
def sig(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
def fwd(P_, fb, mb, kb):
    q=fb@P_["Wq"]; Kx=mb@P_["Wk"]; Vx=mb@P_["Wv"]
    s=np.einsum("bkh,bh->bk",Kx,q)/np.sqrt(H); s=np.where(kb>0,s,-1e9)
    s=s-s.max(1,keepdims=True); e=np.exp(s)*(kb>0); a=e/(e.sum(1,keepdims=True)+1e-9)
    ctx=np.einsum("bk,bkh->bh",a,Vx); z=np.concatenate([fb,ctx],1)
    h1=z@P_["W1"]+P_["b1"]; hid=np.maximum(h1,0)
    return hid@P_["w2"]+P_["b2"], hid@P_["wc"]+P_["bc"], dict(fb=fb,mb=mb,kb=kb,q=q,Kx=Kx,Vx=Vx,a=a,z=z,h1=h1,hid=hid)
def bwd(P_, c, dl, dc):
    g={k:np.zeros_like(v) for k,v in P_.items()}
    hid=c["hid"]; z=c["z"]
    g["w2"]=hid.T@dl; g["b2"]=dl.sum(); g["wc"]=hid.T@dc; g["bc"]=dc.sum()
    dhid=np.outer(dl,P_["w2"])+np.outer(dc,P_["wc"]); dh1=dhid*(c["h1"]>0)
    g["W1"]=z.T@dh1; g["b1"]=dh1.sum(0); dz=dh1@P_["W1"].T
    Df=c["fb"].shape[1]; dfb=dz[:,:Df]; dctx=dz[:,Df:]
    a=c["a"]; Vx=c["Vx"]; Kx=c["Kx"]; q=c["q"]; mb=c["mb"]
    da=np.einsum("bh,bkh->bk",dctx,Vx); dV=np.einsum("bk,bh->bkh",a,dctx)
    ds=(a*(da-(a*da).sum(1,keepdims=True)))/np.sqrt(H)
    dK=np.einsum("bk,bh->bkh",ds,q); dq=np.einsum("bk,bkh->bh",ds,Kx)
    g["Wq"]=c["fb"].T@dq; g["Wk"]=np.einsum("bkd,bkh->dh",mb,dK); g["Wv"]=np.einsum("bkd,bkh->dh",mb,dV)
    return g
def mask_msa(held_cl):
    held_cache_orgs=[ci for ci in range(len(orgs))
                     if (meta_org==ci).any() and str(clade[np.where(meta_org==ci)[0][0]])==held_cl]
    out=msa.copy(); bad=np.isin(msa_org, held_cache_orgs)
    out[...,0]=np.where(bad,0,out[...,0]); out[...,1]=np.where(bad,0,out[...,1])
    return out

def train_eval(held_cl, epochs=6, bs=2048, lr=3e-3):
    P_=init(foc.shape[1]); m={k:np.zeros_like(v) for k,v in P_.items()}; v={k:np.zeros_like(v) for k,v in P_.items()}
    msa_m=mask_msa(held_cl)
    train_idx=np.where(clade!=held_cl)[0]; test_idx=np.where(clade==held_cl)[0]
    pos=y[train_idx].mean() or 1e-6; t=0
    for ep in range(epochs):
        perm=rng.permutation(train_idx)
        for i in range(0,len(perm),bs):
            bi=perm[i:i+bs]
            lo,clo,c=fwd(P_, foc[bi], msa_m[bi], mask[bi])
            p=sig(lo); yb=y[bi]; cw=np.where(yb==1,0.5/pos,0.5/(1-pos))
            dl=(p-yb)*cw/len(bi)
            corr=((p>0.5).astype(np.float32)==yb).astype(np.float32)
            cc=sig(clo); dc=(cc-corr)/len(bi)
            g=bwd(P_,c,dl,dc); t+=1
            for kk in P_:
                m[kk]=0.9*m[kk]+0.1*g[kk]; v[kk]=0.999*v[kk]+0.001*(g[kk]**2)
                mh=m[kk]/(1-0.9**t); vh=v[kk]/(1-0.999**t)
                P_[kk]-=lr*mh/(np.sqrt(vh)+1e-8)
    lo,clo,_=fwd(P_, foc[test_idx], msa_m[test_idx], mask[test_idx])
    return sig(lo), sig(clo), test_idx

# ---- cooccur channels for one gene ----
def channels(j, oi, train_mask, org_name, lt_str):
    out=dict(backup=0, presence=0, coess=0, phyl=0)
    if j<0 or oi<0: return out
    e_full=Y[:,j]; e_tr=e_full[train_mask]
    v=~np.isnan(e_tr)
    if v.sum()>=8 and e_tr[v].std()>=0.1:
        et=e_tr[v]; et_z=(et-et.mean())/(et.std()+1e-9)
        Pn_tr=Pn[train_mask][v]
        Pnc=Pn_tr-Pn_tr.mean(0); Pnz=Pnc/(Pnc.std(0)+1e-9)
        corr=(et_z@Pnz)/len(et_z)
        if j in cand_p_idx:
            pos=np.searchsorted(cand_p_idx, j)
            if pos<len(cand_p_idx) and cand_p_idx[pos]==j: corr[pos]=2.0
        # backup
        anti=-corr; bp=int(np.argmax(anti)); sc=float(anti[bp])
        if sc>=0.7:
            pres=int(P[oi, cand_p_idx[bp]]==1.0)
            out["backup"]=-1 if pres else +1
        # presence
        bp2=int(np.argmax(corr)); sc2=float(corr[bp2])
        if sc2>=0.7:
            pres=int(P[oi, cand_p_idx[bp2]]==1.0)
            out["presence"]=+1 if pres else -1
        # coess
        Y_tr=Y[train_mask][:,cand_e_idx]
        Ymean=np.nanmean(Y_tr,axis=0); Ymean[np.isnan(Ymean)]=0.5
        Yimp=np.where(np.isnan(Y_tr),Ymean[None,:],Y_tr)
        Yv=Yimp[v]; Yvc=Yv-Yv.mean(0); Yvz=Yvc/(Yvc.std(0)+1e-9)
        corr_e=(et_z@Yvz)/len(et_z)
        if j in cand_e_idx:
            pos=np.searchsorted(cand_e_idx, j)
            if pos<len(cand_e_idx) and cand_e_idx[pos]==j: corr_e[pos]=-2.0
        bp3=int(np.argmax(corr_e)); sc3=float(corr_e[bp3])
        if sc3>=0.7:
            pe=Y[oi, cand_e_idx[bp3]]
            if not np.isnan(pe): out["coess"]=+1 if pe>=0.5 else -1
    phyl=cofeat.get((org_name,lt_str))
    if phyl is not None:
        if phyl<=0.10: out["phyl"]=-1
        elif phyl>=0.80: out["phyl"]=+1
    return out

# ---- main ----
MAJOR=["pseudomonas","ralstonia","shewanella","dickeya","burkholderia"]
print("\ntraining transformer LOCO + scoring cooccur on abstained genes...")
T_corr=0; T_called=0  # transformer-confident counts
C_corr=0; C_called=0  # cooccur-rescued counts
N_pooled=0

per_clade=[]
for cl in MAJOR:
    if cl not in set(clade): continue
    p, cf, te=train_eval(cl)
    yt=y[te]
    hi=cf>=0.85
    call_t=(p>0.5).astype(int)
    correct_t=(call_t==yt).astype(int)
    # transformer-confident metrics
    Tn=int(hi.sum()); Tc=int(correct_t[hi].sum())
    # cooccur rescue on abstained (brightness < 0.85)
    abst=~hi
    held_orgs=set(np.where(np.isin(np.array([cache_to_orth[ci] for ci in range(len(orgs))]),
                                    [cache_to_orth[ci] for ci in range(len(orgs))
                                     if (meta_org==ci).any() and str(clade[np.where(meta_org==ci)[0][0]])==cl]))[0].tolist())
    train_mask=np.ones(O, bool)
    held_cache_orgs=[ci for ci in range(len(orgs))
                     if (meta_org==ci).any() and str(clade[np.where(meta_org==ci)[0][0]])==cl]
    for ci in held_cache_orgs:
        oi=cache_to_orth[ci]
        if oi>=0: train_mask[oi]=False

    Cn=0; Cc=0
    for k,idx_in_test in enumerate(np.where(abst)[0]):
        i_full=te[idx_in_test]
        j=int(gene_og_j[i_full]); oi=int(gene_org_i[i_full])
        if j<0 or oi<0: continue
        ch=channels(j, oi, train_mask, str(gene_org_name[i_full]), str(gene_lt[i_full]))
        votes=[v_ for v_ in ch.values() if v_!=0]
        if len(votes)<2: continue
        s=sum(votes)
        if abs(s)<2: continue
        pred=1 if s>0 else 0
        truth=int(yt[idx_in_test])
        Cn+=1; Cc+=int(pred==truth)
    pc=dict(clade=cl, n_test=int(len(te)),
            transformer_hi=Tn, transformer_acc=round(Tc/Tn,3) if Tn else 0,
            cooccur_rescued=Cn, cooccur_acc=round(Cc/Cn,3) if Cn else 0,
            combined_n=Tn+Cn, combined_acc=round((Tc+Cc)/(Tn+Cn),3) if (Tn+Cn) else 0,
            combined_coverage=round((Tn+Cn)/len(te),3))
    per_clade.append(pc)
    print(f"  {cl:<12} n={len(te):<6} | transformer-hi {Tn} ({Tn/len(te):.0%})@{Tc/Tn:.0%} | "
          f"cooccur +{Cn} ({Cn/abst.sum():.0%} of abst)@{Cc/Cn if Cn else 0:.0%} | "
          f"combined cov {(Tn+Cn)/len(te):.0%} @ {(Tc+Cc)/(Tn+Cn) if (Tn+Cn) else 0:.0%}")
    T_called+=Tn; T_corr+=Tc; C_called+=Cn; C_corr+=Cc; N_pooled+=len(te)

print(f"\n=== POOLED (5 major clades, {N_pooled} genes) ===")
print(f"  transformer-confident (brightness>=0.85): {T_called}/{N_pooled} = {T_called/N_pooled:.3f} coverage at {T_corr/T_called:.3f} accuracy")
print(f"  + cooccur rescued (>=2 votes agree):       +{C_called} at {C_corr/C_called:.3f} accuracy")
total=T_called+C_called; tcorr=T_corr+C_corr
print(f"  COMBINED:                                  {total}/{N_pooled} = {total/N_pooled:.3f} coverage at {tcorr/total:.3f} accuracy")
print(f"  delta vs transformer alone:                +{(total-T_called)/N_pooled*100:.1f}pp coverage, "
      f"{(tcorr/total - T_corr/T_called)*100:+.2f}pp accuracy")

json.dump(dict(
  n_pooled=int(N_pooled),
  transformer=dict(called=int(T_called), correct=int(T_corr),
                   coverage=round(T_called/N_pooled,3), accuracy=round(T_corr/T_called,3)),
  cooccur_added=dict(called=int(C_called), correct=int(C_corr),
                     accuracy=round(C_corr/C_called,3) if C_called else None),
  combined=dict(called=int(total), correct=int(tcorr),
                coverage=round(total/N_pooled,3), accuracy=round(tcorr/total,3)),
  delta_pp=dict(coverage=round((total-T_called)/N_pooled*100,2),
                accuracy=round((tcorr/total - T_corr/T_called)*100,2)),
  per_clade=per_clade
), open(OUT/"transformer_plus_cooccur_results.json","w"), indent=2)
print("\nwrote transformer_plus_cooccur_results.json")
