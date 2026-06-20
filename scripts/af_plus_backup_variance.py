"""Combine backup-mining + variance flag with the AF-style transformer model.

Builds two new per-gene features over the 179,237-gene cache:
  backup_present   : presence of the gene's best anti-correlated backup OG (0/1)
                     -- leak-clean (anti-corr score computed from OTHER organisms)
  ess_variance     : std of essentiality across other-org members of the gene's OG
                     -- leak-clean (excludes the gene's own organism)

Then trains the same single-block ortholog-MSA attention + confidence head model,
leave-one-CLADE-out (the strict regime that produced the original 0.85 AUC /
78% coverage), and compares:
  baseline (7 focal features)  vs  +backup+variance (9 focal features)

Reports per-clade and pooled: AUC, two-sided coverage@P>=0.90, brightness
(high-confidence acc>=0.85) coverage.

Outputs:
  outputs/orphan/af_plus_backup_variance.json
"""
import json, csv
from pathlib import Path
from collections import defaultdict
import numpy as np

OUT = Path("/home/user/cell/outputs/orphan")
DATA = Path("/home/user/cell/data/drive_import")
rng = np.random.default_rng(0)

# ---- load cache ----
Z = np.load(OUT/"af_msa_cache.npz", allow_pickle=True)
foc=Z["foc"]; msa=Z["msa"]; msa_org=Z["msa_org"]; mask=Z["msa_mask"]; y=Z["y"]
clade=Z["clade"]; orgs=list(Z["orgs"]); meta_org=Z["meta_org"]; lt=Z["lt"]
N=len(y); print(f"cache: {N} genes")

# ---- build OG, presence, essentiality matrices ----
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
G=len(all_ogs); og_idx={og:j for j,og in enumerate(all_ogs)}
Y_mat=np.full((O,G), np.nan, np.float32)
for (o,l),e in labels.items():
    og=gene_og.get((o,l))
    if og: Y_mat[org_idx[o], og_idx[og]]=e
P_mat=np.zeros((O,G), np.float32)
for og,orgs_p in og_present.items():
    j=og_idx[og]
    for o in orgs_p: P_mat[org_idx[o],j]=1.0
print(f"matrices: O={O} orgs, G={G} OGs")

# ---- per-gene OG (-1 if no OG) AND map cache org idx -> orthology org idx ----
# Cache uses one index space (over 59 cache orgs); orthology Y/P matrices use another (48 labeled-in-orthology orgs).
# Build a bridge by organism name.
cache_to_orth=np.full(len(orgs), -1, dtype=np.int32)
for ci,name in enumerate(orgs):
    name=str(name)
    if name in org_idx:
        cache_to_orth[ci]=org_idx[name]
print(f"cache orgs mapped to orthology: {int((cache_to_orth>=0).sum())}/{len(orgs)}")

gene_og_idx=np.full(N, -1, dtype=np.int32)
gene_orth_org=np.full(N, -1, dtype=np.int32)
for i in range(N):
    name=str(orgs[meta_org[i]])
    key=(name, str(lt[i]))
    og=gene_og.get(key)
    if og is not None and og in og_idx: gene_og_idx[i]=og_idx[og]
    gene_orth_org[i]=cache_to_orth[meta_org[i]]
print(f"genes with OG: {int((gene_og_idx>=0).sum())}/{N}; with mapped org: {int((gene_orth_org>=0).sum())}/{N}")

# ---- LOO-clade features (variance + backup) ----
# Variance: per held clade, per OG, std over orgs NOT in held clade
# Backup: per held clade, per OG, best anti-correlated OG over orgs NOT in held
clades=sorted(set(clade))
clade_of_org=np.full(O, "?", dtype=object)
for ci in range(len(orgs)):
    oi=cache_to_orth[ci]
    if oi<0: continue
    where=np.where(meta_org==ci)[0]
    if where.size: clade_of_org[oi]=str(clade[where[0]])
print(f"clades: {len(clades)} | orgs with mapped clade: {int((clade_of_org!='?').sum())}/{O}")

# Candidate backups: OGs present in 5..O-5 orgs
p_sum=P_mat.sum(0)
cand_mask=(p_sum>=5)&(p_sum<=O-5)
cand_idx=np.where(cand_mask)[0]
print(f"candidate backups: {len(cand_idx)}")

def loo_features(held_clade):
    """Compute (variance[G], backup_og[G], backup_score[G]) using orgs NOT in held clade."""
    tr_org_mask = clade_of_org != held_clade
    Y_tr = Y_mat[tr_org_mask]      # [O_tr, G]
    P_tr = P_mat[tr_org_mask][:, cand_idx]    # [O_tr, n_cand]
    valid = ~np.isnan(Y_tr)
    n_lab = valid.sum(0)
    s  = np.where(valid, Y_tr, 0).sum(0)
    s2 = np.where(valid, Y_tr*Y_tr, 0).sum(0)
    mean = np.where(n_lab>0, s/np.maximum(n_lab,1), 0)
    var  = np.where(n_lab>0, s2/np.maximum(n_lab,1) - mean**2, 0)
    var_g = np.sqrt(np.maximum(var,0))
    var_g[n_lab<3] = 0.0   # mark unreliable
    fam_majority = (mean>=0.5).astype(np.float32)
    fam_majority[n_lab<3] = 0.5

    # Backup mining: for OGs with variable essentiality, find best anti-corr backup
    backup_og = np.full(G, -1, np.int32)
    backup_score = np.zeros(G, np.float32)
    backup_present_mat = np.zeros((O,G), np.float32)  # will fill for backup of each OG
    # Restrict targets to OGs with enough labeled training orgs (>=8) and variable
    target_mask = (n_lab>=8) & (mean>0.1) & (mean<0.9)
    target_js = np.where(target_mask)[0]

    # Standardize candidate presence over training organisms
    P_tr_c = P_tr - P_tr.mean(0)
    P_tr_std = P_tr_c.std(0)+1e-9
    P_tr_z = P_tr_c / P_tr_std

    for j in target_js:
        e_tr_full = Y_tr[:, j]
        v = ~np.isnan(e_tr_full)
        if v.sum()<8: continue
        et = e_tr_full[v]
        if et.std()<0.1: continue
        et_z = (et - et.mean())/(et.std()+1e-9)
        Pv = P_tr_z[v]
        Pv_c = Pv - Pv.mean(0)
        Pv_s = Pv_c.std(0)+1e-9
        Pv_z = Pv_c / Pv_s
        corr = (et_z @ Pv_z) / len(et_z)   # [n_cand]
        anticorr = -corr
        # ban self-match (target OG is at position cand_idx==j if present)
        if j in cand_idx:
            pos = np.searchsorted(cand_idx, j)
            if cand_idx[pos]==j: anticorr[pos] = -2.0
        bp = int(np.argmax(anticorr)); sc = float(anticorr[bp])
        if sc<0.4: continue
        backup_og[j] = cand_idx[bp]
        backup_score[j] = sc
    # Backup-presence broadcast to ALL orgs (including held) so we can index per gene
    bp_g = np.zeros(G, np.float32)   # will be filled per gene from P_mat[org, backup]
    return var_g, fam_majority, backup_og, backup_score

# Per-gene feature for held-clade: variance(gene.og) and backup-present(gene.org, backup_of_gene.og)
print("\ncomputing LOO-clade features for each major clade...")
MAJOR=["pseudomonas","ralstonia","shewanella","dickeya","burkholderia"]
all_var=np.zeros(N,np.float32); all_bp=np.zeros(N,np.float32); all_bs=np.zeros(N,np.float32)
all_fm=np.full(N,0.5,np.float32)
for cl in MAJOR:
    if cl not in set(clade): continue
    var_g, fm_g, bog_g, bs_g = loo_features(cl)
    # apply to genes whose clade==cl (those are TEST genes; features were trained on others)
    te_mask = (clade==cl)
    for i in np.where(te_mask)[0]:
        j = gene_og_idx[i]
        if j<0: continue
        all_var[i] = var_g[j]; all_fm[i] = fm_g[j]; all_bs[i] = bs_g[j]
        b_og = bog_g[j]
        oi = gene_orth_org[i]
        if b_og>=0 and oi>=0:
            all_bp[i] = P_mat[oi, b_og]
    print(f"  {cl}: test genes={int(te_mask.sum())} | with backup={(all_bs[te_mask]>0).sum()} ({(all_bs[te_mask]>0).mean():.0%})")

# ---- model: identical to af_train.py (single-head attention + brightness) ----
H=12; HID=24
def init(Df):
    s=0.3
    return dict(Wq=rng.normal(0,s,(Df,H)).astype(np.float32),
        Wk=rng.normal(0,s,(6,H)).astype(np.float32),
        Wv=rng.normal(0,s,(6,H)).astype(np.float32),
        W1=rng.normal(0,s,(Df+H,HID)).astype(np.float32), b1=np.zeros(HID,np.float32),
        w2=rng.normal(0,s,HID).astype(np.float32), b2=np.float32(0.0),
        wc=rng.normal(0,s,HID).astype(np.float32), bc=np.float32(0.0))
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

def forward(P, fb, mb, kb):
    q=fb@P["Wq"]; Kx=mb@P["Wk"]; Vx=mb@P["Wv"]
    s=np.einsum("bkh,bh->bk",Kx,q)/np.sqrt(H); s=np.where(kb>0,s,-1e9)
    s=s-s.max(1,keepdims=True); e=np.exp(s)*(kb>0); a=e/(e.sum(1,keepdims=True)+1e-9)
    ctx=np.einsum("bk,bkh->bh",a,Vx); z=np.concatenate([fb,ctx],1)
    h1=z@P["W1"]+P["b1"]; hid=np.maximum(h1,0)
    return hid@P["w2"]+P["b2"], hid@P["wc"]+P["bc"], dict(fb=fb,mb=mb,kb=kb,q=q,Kx=Kx,Vx=Vx,a=a,z=z,h1=h1,hid=hid)
def backward(P, c, dl, dc):
    g={k:np.zeros_like(v) for k,v in P.items()}
    hid=c["hid"]; z=c["z"]
    g["w2"]=hid.T@dl; g["b2"]=dl.sum(); g["wc"]=hid.T@dc; g["bc"]=dc.sum()
    dhid=np.outer(dl,P["w2"])+np.outer(dc,P["wc"]); dh1=dhid*(c["h1"]>0)
    g["W1"]=z.T@dh1; g["b1"]=dh1.sum(0); dz=dh1@P["W1"].T
    Df=c["fb"].shape[1]; dfb=dz[:,:Df]; dctx=dz[:,Df:]
    a=c["a"]; Vx=c["Vx"]; Kx=c["Kx"]; q=c["q"]; mb=c["mb"]
    da=np.einsum("bh,bkh->bk",dctx,Vx); dV=np.einsum("bk,bh->bkh",a,dctx)
    ds=(a*(da-(a*da).sum(1,keepdims=True)))/np.sqrt(H)
    dK=np.einsum("bk,bh->bkh",ds,q); dq=np.einsum("bk,bkh->bh",ds,Kx)
    g["Wq"]=c["fb"].T@dq; g["Wk"]=np.einsum("bkd,bkh->dh",mb,dK); g["Wv"]=np.einsum("bkd,bkh->dh",mb,dV)
    return g

def mask_msa(held_clade):
    # msa_org is in CACHE org index space; identify cache orgs in held clade
    held_cache_orgs=[ci for ci in range(len(orgs))
                     if (meta_org==ci).any() and str(clade[np.where(meta_org==ci)[0][0]])==held_clade]
    out=msa.copy(); bad=np.isin(msa_org, held_cache_orgs)
    out[...,0]=np.where(bad,0,out[...,0]); out[...,1]=np.where(bad,0,out[...,1])
    return out

def auc(s,yy):
    yy=np.asarray(yy); n1=yy.sum(); n0=len(yy)-n1
    if n1==0 or n0==0: return float("nan")
    r=np.argsort(np.argsort(s))+1; return float((r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0))
def cov_at_p(score, yy, t, mn=50):
    conf=np.maximum(score, 1-score); call=(score>0.5).astype(int); corr=(call==yy).astype(float)
    o=np.argsort(-conf); cc=corr[o]; k=np.arange(1,len(cc)+1); pr=np.cumsum(cc)/k
    ok=(pr>=t)&(k>=mn); return float(k[np.where(ok)[0].max()]/len(cc)) if ok.any() else 0.0

def train_eval(foc_in, held_clade, epochs=6, bs=2048, lr=3e-3):
    P=init(foc_in.shape[1]); m={k:np.zeros_like(v) for k,v in P.items()}; v={k:np.zeros_like(v) for k,v in P.items()}
    held_orgs=set(np.where(clade_of_org==held_clade)[0])
    msa_m=mask_msa(held_clade)
    train_idx=np.where(clade!=held_clade)[0]
    test_idx=np.where(clade==held_clade)[0]
    pos=y[train_idx].mean() or 1e-6; t=0
    for ep in range(epochs):
        perm=rng.permutation(train_idx)
        for i in range(0,len(perm),bs):
            bi=perm[i:i+bs]
            lo,clo,c=forward(P, foc_in[bi], msa_m[bi], mask[bi])
            p=sigmoid(lo); yb=y[bi]
            cw=np.where(yb==1,0.5/pos,0.5/(1-pos))
            dl=(p-yb)*cw/len(bi)
            corr=((p>0.5).astype(np.float32)==yb).astype(np.float32)
            cc=sigmoid(clo); dc=(cc-corr)/len(bi)
            g=backward(P,c,dl,dc); t+=1
            for kk in P:
                m[kk]=0.9*m[kk]+0.1*g[kk]; v[kk]=0.999*v[kk]+0.001*(g[kk]**2)
                mh=m[kk]/(1-0.9**t); vh=v[kk]/(1-0.999**t)
                P[kk]-=lr*mh/(np.sqrt(vh)+1e-8)
    lo,clo,_=forward(P, foc_in[test_idx], msa_m[test_idx], mask[test_idx])
    return sigmoid(lo), sigmoid(clo), test_idx

# ---- build two feature sets ----
foc_base = foc.astype(np.float32)  # 7
foc_aug  = np.concatenate([foc_base, all_var.reshape(-1,1), all_bp.reshape(-1,1)], axis=1).astype(np.float32)  # 9
print(f"\nbaseline foc dims {foc_base.shape[1]} | augmented {foc_aug.shape[1]}")

# ---- evaluate ----
def evaluate(label, foc_in):
    print(f"\n=== {label} (leave-one-CLADE-out) ===")
    pooled_p=[]; pooled_y=[]; per_clade=[]
    for cl in MAJOR:
        if cl not in set(clade): continue
        p, cf, te=train_eval(foc_in, cl)
        yt=y[te]
        a=auc(p,yt); cov=cov_at_p(p,yt,0.9)
        # brightness >=0.85 coverage
        hi=cf>=0.85; hi_acc=float(((p>0.5).astype(int)==yt)[hi].mean()) if hi.any() else float("nan")
        per_clade.append((cl,len(te),a,cov,float(hi.mean()),hi_acc))
        pooled_p.append(p); pooled_y.append(yt)
        print(f"  {cl:<12} n={len(te):<6} AUC={a:.3f}  cov@P0.90={cov:.3f}  hi-conf={hi.mean():.0%}@{hi_acc:.2%}")
    pp=np.concatenate(pooled_p); py=np.concatenate(pooled_y)
    pa=auc(pp,py); pc=cov_at_p(pp,py,0.9)
    print(f"  POOLED      n={len(py):<6} AUC={pa:.3f}  cov@P0.90={pc:.3f}")
    return dict(per_clade=[dict(clade=c,n=n,auc=round(a,3),cov_p90=round(cv,3),
                                hi_conf_frac=round(hf,3),hi_conf_acc=round(ha,3)) for c,n,a,cv,hf,ha in per_clade],
                pooled_auc=round(pa,3), pooled_cov_p90=round(pc,3),
                pooled_n=int(len(py)))

results=dict(baseline=evaluate("BASELINE (7 features)", foc_base),
             augmented=evaluate("AUGMENTED (+variance +backup)", foc_aug))

# Δ
b=results["baseline"]; a=results["augmented"]
print(f"\n=== DELTA ===")
print(f"  AUC:        {b['pooled_auc']:.3f}  ->  {a['pooled_auc']:.3f}  (Δ {a['pooled_auc']-b['pooled_auc']:+.3f})")
print(f"  cov@P0.90:  {b['pooled_cov_p90']:.3f}  ->  {a['pooled_cov_p90']:.3f}  (Δ {a['pooled_cov_p90']-b['pooled_cov_p90']:+.3f})")
results["delta"]=dict(auc=round(a["pooled_auc"]-b["pooled_auc"],3),
                       cov_p90=round(a["pooled_cov_p90"]-b["pooled_cov_p90"],3))
json.dump(results, open(OUT/"af_plus_backup_variance.json","w"), indent=2)
print("\nwrote af_plus_backup_variance.json")
