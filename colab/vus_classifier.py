"""VUS pathogenicity classifier: fuse functional-site + burial + population constraint +
ESM variant-effect into one score. Train on ClinVar pathogenic vs benign (5-fold CV),
then score Variants of Uncertain Significance (VUS)."""
import json, urllib.request, re, time, gzip, warnings, socket
socket.setdefaulttimeout(35)
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
warnings.filterwarnings("ignore")
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")
SC=Path("/tmp/claude-0/-home-user-cell/558875a8-e44f-59af-a869-11202bf854d3/scratchpad")
AA3={'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G','His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}
GENES=["LDLR","PAH","TP53","VHL","RB1","PTEN","HBB","G6PD","HNF1A","GATA4","TBX5",
 "SOX9","KCNQ1","RET","MLH1","STK11"]
# gene -> ACC + sequence
acc={}; seqs={}
def parse():
    sym=None; buf=[]
    for l in open(SC/"human.faa"):
        if l.startswith(">"):
            if sym and buf: seqs[sym]=max(seqs.get(sym,""),"".join(buf),key=len)
            buf=[]; m=re.match(r">\w+\|(\w+)\|",l); gn=[t[3:] for t in l.split() if t.startswith("GN=")]
            sym=gn[0] if gn else None
            if m and sym: acc.setdefault(sym,m.group(1))
        else: buf.append(l.strip())
    if sym and buf: seqs[sym]=max(seqs.get(sym,""),"".join(buf),key=len)
parse()
loeuf={}
with gzip.open(H/"gnomad_constraint.txt.bgz","rt") as f:
    hh=f.readline().split("\t"); gi,li=hh.index("gene"),hh.index("oe_lof_upper")
    for l in f:
        p=l.split("\t")
        try: v=float(p[li])
        except: continue
        if p[gi] not in loeuf or v<loeuf[p[gi]]: loeuf[p[gi]]=v
def hj(url,data=None):
    req=urllib.request.Request(url,data=(json.dumps(data).encode() if data else None),
        headers={"Content-Type":"application/json"} if data else {})
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(req,timeout=30))
        except Exception: time.sleep(2)
    return None
def af(ac):
    api=hj(f"https://alphafold.ebi.ac.uk/api/prediction/{ac}")
    if not api: return None
    try: txt=urllib.request.urlopen(api[0]["pdbUrl"],timeout=30).read().decode()
    except: return None
    ca={}
    for ln in txt.splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip()=="CA":
            ca[int(ln[22:26])]=(float(ln[30:38]),float(ln[38:46]),float(ln[46:54]),float(ln[60:66]))
    return ca
def features_uni(ac):
    d=hj(f"https://rest.uniprot.org/uniprotkb/{ac}.json?fields=ft_act_site,ft_binding,ft_dna_bind,ft_disulfid")
    anchor=set()
    for f in (d or {}).get("features",[]):
        loc=f["location"]
        try: s=int(loc["start"]["value"]); e=int(loc["end"]["value"])
        except: continue
        if f["type"] in("Active site","Binding site","DNA binding","Disulfide bond"): anchor|=set(range(s,e+1))
    return anchor
def clinvar(sym):
    d=hj("https://gnomad.broadinstitute.org/api",{"query":'{gene(gene_symbol:"%s",reference_genome:GRCh38){clinvar_variants{clinical_significance major_consequence hgvsp}}}'%sym})
    out=[]
    for v in ((d or {}).get("data",{}).get("gene") or {}).get("clinvar_variants",[]):
        if v.get("major_consequence")!="missense_variant": continue
        m=re.search(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})",v.get("hgvsp") or "")
        if not m or m.group(1) not in AA3 or m.group(3) not in AA3: continue
        sig=(v.get("clinical_significance") or "").lower()
        lab=1 if ("pathogenic" in sig and "conflicting" not in sig) else (0 if ("benign" in sig and "conflicting" not in sig) else None)
        out.append((int(m.group(2)),AA3[m.group(1)],AA3[m.group(3)],lab))
    return out

print("loading ESM-2 8M ...",flush=True)
tok=AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
mdl=AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D").eval(); torch.set_num_threads(8)
MASK=tok.mask_token_id; AATOK={a:tok.convert_tokens_to_ids(a) for a in AA3.values()}
@torch.no_grad()
def esm_llr(seq,variants):  # variants: list of (pos,wt,mut) -> dict pos-> logsoftmax vector
    seq=seq[:1022]; ids=tok(seq,return_tensors="pt")["input_ids"][0]
    pos=sorted({p for p,_,_ in variants if 1<=p<=len(seq)})
    lp={}
    for i in range(0,len(pos),24):
        js=pos[i:i+24]; batch=ids.repeat(len(js),1)
        for k,p in enumerate(js): batch[k,p]=MASK
        out=mdl(input_ids=batch).logits; ls=torch.log_softmax(out,-1)
        for k,p in enumerate(js): lp[p]=ls[k,p]
    return lp

rows=[]
for sym in GENES:
    ac=acc.get(sym); seq=seqs.get(sym)
    if not ac or not seq: print(f"{sym}: skip (no acc/seq)"); continue
    ca=af(ac); anchor=features_uni(ac); vs=clinvar(sym)
    if not ca or not vs: print(f"{sym}: skip (no struct/variants)"); continue
    rn=sorted(ca); xyz=np.array([ca[r][:3] for r in rn])
    contacts={r:int((np.sqrt(((xyz-xyz[i])**2).sum(1))<10).sum())-1 for i,r in enumerate(rn)}
    lp=esm_llr(seq,[(p,w,m) for p,w,m,_ in vs])
    npath=sum(1 for *_,l in vs if l==1); nben=sum(1 for *_,l in vs if l==0); nvus=sum(1 for *_,l in vs if l is None)
    for pos,wt,mut,lab in vs:
        if pos not in ca or pos not in lp: continue
        v=lp[pos]; llr=float(v[AATOK[mut]]-v[AATOK[wt]])   # <0 = damaging
        atf=1.0 if pos in anchor else 0.0
        d=99.0
        if anchor:
            p3=np.array(ca[pos][:3]); dd=[np.sqrt(((p3-np.array(ca[a][:3]))**2).sum()) for a in anchor if a in ca]
            if dd: d=min(dd)
        rows.append(dict(gene=sym,pos=pos,var=f"{wt}{pos}{mut}",label=lab,
            esm_llr=llr, burial=contacts[pos], plddt=ca[pos][3], at_func=atf,
            dist_func=min(d,30.0), loeuf=loeuf.get(sym,1.0)))
    print(f"  {sym:8s} path={npath} benign={nben} vus={nvus}",flush=True)
    time.sleep(0.3)

FEAT=["esm_llr","burial","plddt","at_func","dist_func","loeuf"]
lab=np.array([r["label"] if r["label"] is not None else -1 for r in rows])
X=np.array([[r[f] for f in FEAT] for r in rows],float)
tr=lab>=0; ntr=int(tr.sum())
print(f"\ntotal variants {len(rows)}: labeled {ntr} (path {int((lab==1).sum())}, benign {int((lab==0).sum())}), VUS {int((lab==-1).sum())}")
mu=X[tr].mean(0); sd=X[tr].std(0); sd[sd<1e-9]=1; Xn=(X-mu)/sd
def fit(Xt,yt,l2=1.0,it=500,lr=0.15):
    Xa=np.hstack([Xt,np.ones((len(Xt),1))]); w=np.zeros(Xa.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-Xa@w)); g=Xa.T@(p-yt)/len(yt)+l2*np.r_[w[:-1],0]/len(yt); w-=lr*g
    return w
def pred(w,Xt): return 1/(1+np.exp(-(np.hstack([Xt,np.ones((len(Xt),1))])@w)))
def auc(p,y):
    o=np.argsort(p);r=np.empty(len(p));r[o]=np.arange(len(p));m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))
# 5-fold OOF on labeled
idx=np.where(tr)[0]; rng=np.random.default_rng(0); rng.shuffle(idx); folds=np.array_split(idx,5)
oof=np.zeros(len(rows)); 
for k in range(5):
    te=folds[k]; trn=np.concatenate([folds[j] for j in range(5) if j!=k])
    w=fit(Xn[trn],lab[trn]); oof[te]=pred(w,Xn[te])
ytr=lab[tr]; print(f"\n=== FUSION classifier 5-fold OOF AUC = {auc(oof[tr],ytr):.3f} ===")
# ablations: each feature alone (direction-corrected via AUC)
print("single-feature AUC:")
for j,f in enumerate(FEAT):
    a=auc(Xn[tr][:,j],ytr); a=max(a,1-a); print(f"  {f:10s}: {a:.3f}")
# full-fit weights + score VUS
w=fit(Xn[tr],ytr); 
print("weights:",{f:round(float(w[j]),2) for j,f in enumerate(FEAT)})
vus=np.where(lab==-1)[0]; vs_score=pred(w,Xn[vus])
for r,s in zip([rows[i] for i in vus],vs_score): r["path_score"]=round(float(s),3)
nlp=int((vs_score>=0.5).sum())
print(f"\nVUS scored: {len(vus)}; likely pathogenic (>=0.5): {nlp} ({100*nlp/max(len(vus),1):.0f}%), likely benign: {len(vus)-nlp}")
top=sorted([rows[i] for i in vus],key=lambda r:-r["path_score"])[:12]
print("top VUS flagged likely-pathogenic:")
for r in top: print(f"  {r['gene']:7s} {r['var']:10s} score={r['path_score']:.2f} (esm_llr {r['esm_llr']:.1f} burial {r['burial']} atFunc {int(r['at_func'])})")
import csv
with open(OUT/"vus_scored.csv","w",newline="") as f:
    wtr=csv.DictWriter(f,fieldnames=["gene","var","label","path_score"]+FEAT); wtr.writeheader()
    for r in rows: wtr.writerow({k:r.get(k) for k in ["gene","var","label","path_score"]+FEAT})
json.dump(dict(n_variants=len(rows),labeled=ntr,path=int((lab==1).sum()),benign=int((lab==0).sum()),
    vus=int((lab==-1).sum()),oof_auc=round(auc(oof[tr],ytr),3),
    single_feature_auc={f:round(max(auc(Xn[tr][:,j],ytr),1-auc(Xn[tr][:,j],ytr)),3) for j,f in enumerate(FEAT)},
    weights={f:round(float(w[j]),3) for j,f in enumerate(FEAT)},
    vus_likely_pathogenic=nlp,top_vus=[{k:t[k] for k in("gene","var","path_score")} for t in top]),
    open(OUT/"vus_classifier.json","w"),indent=2)
print("\nwrote vus_classifier.json + vus_scored.csv")
