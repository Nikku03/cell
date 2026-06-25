"""Non-essential coverage (neCov) at P>=0.90 for the non-essential class.

Mirror of fba_free_coverage.py but for the NON-essential side: how many true
non-essentials can we confidently flag (genes safe to delete) at high precision?
Score = LOW combined (w1+w2[+w3]) means non-essential; sort ascending, sweep.
"""
import csv, json, time, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
cobra.Configuration().solver = "glpk"
OUT = Path("outputs/orphan"); MODELS = Path("data/external_data/bigg_models")

class Scaler:
    def fit(self,X): self.mu=X.mean(0); self.sd=X.std(0); self.sd[self.sd<1e-9]=1; return self
    def transform(self,X): return (X-self.mu)/self.sd
class Logit:
    def __init__(self,C=1.0,max_iter=400,class_weight=None): self.C=C; self.max_iter=max_iter; self.cw=class_weight
    def fit(self,X,y):
        n,d=X.shape; Xb=np.c_[X,np.ones(n)]; w=np.zeros(d+1)
        cw=np.where(y==1,0.5/max(y.mean(),1e-6),0.5/max(1-y.mean(),1e-6)) if self.cw=="balanced" else np.ones(n)
        lr=0.1
        for it in range(self.max_iter):
            z=np.clip(Xb@w,-30,30); p=1/(1+np.exp(-z))
            w-=lr*(Xb.T@((p-y)*cw)/n+(w/self.C)*np.r_[np.ones(d),0])
            if it%80==79: lr*=0.5
        self.w=w; return self
    def predict_proba(self,X):
        Xb=np.c_[X,np.ones(len(X))]; z=np.clip(Xb@self.w,-30,30); p=1/(1+np.exp(-z)); return np.c_[1-p,p]

ALL=list(csv.DictReader(open("data/drive_import/labels/labels.csv")))
clade_of={r["organism"]:r["clade"] for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
og_of={(r["organism"],r["locus_tag"]):r["og_id"] for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")) if r.get("og_id")}
P=np.load(OUT/"af_torch_preds_aug.npz",allow_pickle=True)
_cl,_lt,_p=P["clade"],P["lt"],P["p"]
w1={(str(_cl[i]),str(_lt[i])):float(_p[i]) for i in range(len(_p)) if not np.isnan(_p[i])}
def cons_rate(held):
    n=defaultdict(int); e=defaultdict(int)
    for r in ALL:
        if clade_of.get(r["organism"])==held: continue
        og=og_of.get((r["organism"],r["locus_tag"]))
        if og: n[og]+=1; e[og]+=int(r["essential"])
    return {og:e[og]/n[og] for og in n if n[og]>=5}

RICH=["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly","his__L","ile__L","leu__L","lys__L","met__L","phe__L","pro__L","ser__L","thr__L","trp__L","tyr__L","val__L","ade","gua","ura","thymd","cytd","ins","adn","gsn","uri","thm","ribflv","nac","pnto__R","pydx","4abz","btn","fol"]
def fba_run(path,label):
    M=cobra.io.read_sbml_model(str(path))
    for x in RICH:
        rid=f"EX_{x}_e"
        if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
    g0=M.slim_optimize(); print(f"  [{label}] ...",flush=True)
    de=single_gene_deletion(M); out={}
    for _,row in de.iterrows():
        gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
        out[gid]=int(row["growth"]<0.01*g0 or np.isnan(row["growth"]))
    return out,M
print("FBA labels...",flush=True); t0=time.time()
fba_pu,_=fba_run(MODELS/"iJN1463.xml.gz","Pu"); fba_ec,_=fba_run(MODELS/"iML1515.xml.gz","Ec")
fba_mt,_=fba_run(MODELS/"iEK1008.xml.gz","Mt"); fba_ko,M_kox=fba_run(MODELS/"iYL1228.xml.gz","Ko")
print(f"  ({time.time()-t0:.0f}s)",flush=True)
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
mtub_lts={r["locus_tag"] for r in ALL if r["organism"]=="mtub"}
koxy_annot={r["locus_tag"]:r["product"] for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")) if r["organism"]=="beril_Koxy" and r.get("product")}
koxy_tok=defaultdict(list)
for lt,prod in koxy_annot.items():
    for tk in re.findall(r"[A-Za-z0-9]{4,}",prod): koxy_tok[tk.lower()].append(lt)
kpn_to_koxy={}
for g in M_kox.genes:
    gn=(g.name or "").strip()
    if gn and g.id!=gn:
        mm=koxy_tok.get(gn.lower(),[])
        if len(mm)==1: kpn_to_koxy[g.id]=mm[0]
fba_truth={}
for g,e in fba_pu.items(): fba_truth[("beril_Putida",g)]=e
for b,e in fba_ec.items():
    if b in b_to_keio: fba_truth[("beril_Keio",b_to_keio[b])]=e
for g,e in fba_mt.items():
    if g in mtub_lts: fba_truth[("mtub",g)]=e
for g,e in fba_ko.items():
    if g in kpn_to_koxy: fba_truth[("beril_Koxy",kpn_to_koxy[g])]=e
real_fba_score={k:float(v) for k,v in fba_truth.items()}

codon={(r["organism"],r["locus_tag"]):r for r in csv.DictReader(open("data/drive_import/labels/codon_features.csv"))}
orth={(r["organism"],r["locus_tag"]):r for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv"))}
coocc={(r["organism"],r["locus_tag"]):r for r in csv.DictReader(open("data/drive_import/labels/cooccurrence_features.csv"))}
reg={(r["organism"],r["locus_tag"]):r for r in csv.DictReader(open("data/drive_import/labels/regulator_features.csv"))}
annot={(r["organism"],r["locus_tag"]):(r.get("product") or "").lower() for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))}
KEYS=["synthase","synthetase","ligase","decarboxylase","dehydrogenase","transferase","reductase","hydrolase","kinase","phosphatase","isomerase","biosynth","ribosom","trna","rrna","metabolism","cofactor","cell wall","peptidoglycan","membrane","atp","gtp","transport","permease","abc","secretion","regulator","sensor","histidine kinase","conjugat","hypothetical","unknown"]
def fv(o,lt):
    f=[]; c=codon.get((o,lt))
    f+=[float(c["cai"]) if c and c["cai"] else 0, float(c["gc"]) if c and c["gc"] else 0, float(c["gc3"]) if c and c["gc3"] else 0,
        np.log1p(float(c["cds_length"])) if c and c["cds_length"] else 0,
        np.log1p(abs(float(c["intergenic_prev"]))) if c and c["intergenic_prev"] else 0,
        np.log1p(abs(float(c["intergenic_next"]))) if c and c["intergenic_next"] else 0,
        int(c["same_strand_prev"]=="True") if c else 0, int(c["same_strand_next"]=="True") if c else 0]
    h=orth.get((o,lt))
    f+=[int(h["own_fold"]) if h and h["own_fold"] else 0, np.log1p(float(h["n_paralogs_in_genome"])) if h else 0,
        np.log1p(float(h["family_size_total"])) if h else 0, np.log1p(float(h["family_n_organisms"])) if h else 0,
        int(h["is_orphan"]=="True") if h else 1]
    k=coocc.get((o,lt))
    f+=[float(k["cooccur_max_jaccard"]) if k and k["cooccur_max_jaccard"] else 0,
        np.log1p(float(k["cooccur_n_neighbors_50"])) if k else 0, np.log1p(float(k["cooccur_n_neighbors_80"])) if k else 0]
    g=reg.get((o,lt))
    f+=[int(g["is_regulator"]=="True") if g else 0, int(g["is_signaling"]=="True") if g else 0,
        int(g["is_transporter"]=="True") if g else 0, int(g["is_conditional"]=="True") if g else 0]
    p=annot.get((o,lt),""); f+=[int(kw in p) for kw in KEYS]
    return f
X,y,keys=[],[],[]
for (o,lt),e in fba_truth.items(): X.append(fv(o,lt)); y.append(e); keys.append((o,lt))
X=np.array(X); y=np.array(y); sc=Scaler().fit(X)
mf=Logit(class_weight="balanced").fit(sc.transform(X),y)
truth_by={(r["organism"],r["locus_tag"]):int(r["essential"]) for r in ALL}
all_keys=list(truth_by.keys()); X60=np.array([fv(o,lt) for (o,lt) in all_keys])
proxy_score={k:float(s) for k,s in zip(all_keys,mf.predict_proba(sc.transform(X60))[:,1])}
print(f"[proxy] scored {len(proxy_score)}\n",flush=True)

def build_table(org):
    held=clade_of.get(org,"?"); rate=cons_rate(held); out=[]
    for (o,lt),tv in truth_by.items():
        if o!=org: continue
        og=og_of.get((o,lt)); w1v=w1.get((held,lt),np.nan); w2v=rate.get(og,np.nan) if og else np.nan
        prx=proxy_score.get((o,lt),np.nan); rfb=real_fba_score.get((o,lt),np.nan)
        if np.isnan(w1v) or np.isnan(w2v) or np.isnan(prx): continue
        out.append((lt,tv,w1v,w2v,prx,rfb))
    return out

def cov_ne_at_precision(scores, truths, target_p=0.90, min_calls=10):
    """Non-essential side: LOW score => non-essential. Sort ASC, sweep.
    precision_ne = (#true non-ess in call)/calls; neCov = TP_ne/total_non_ess."""
    s=np.array(scores); y=np.array(truths)         # y: 1=essential
    order=np.argsort(s)                            # ascending: most non-essential first
    yo=y[order]; ne=(yo==0).astype(int)
    tp=np.cumsum(ne); n=np.arange(1,len(yo)+1); prec=tp/n
    total_ne=(y==0).sum()
    valid=(prec>=target_p)&(n>=min_calls)
    if not valid.any(): return (0.0,0.0,0)
    k=np.max(np.where(valid)[0])
    return (float(prec[k]), float(tp[k]/max(total_ne,1)), int(n[k]+0))

print("=== MODELED orgs: neCov @ P_ne>=0.90 (proxy vs real FBA) ===")
print(f"{'organism':<15}{'tot_NE':>8}  | REAL: P neCov calls | PROXY: P neCov calls")
modeled4=["beril_Putida","beril_Keio","mtub","beril_Koxy"]; rows_mod=[]
for org in modeled4:
    T=build_table(org); T=[r for r in T if not np.isnan(r[5])]
    if not T: continue
    tot_ne=sum(1 for r in T if r[1]==0)
    scR=[(r[2]+r[3]+r[5])/3 for r in T]; scP=[(r[2]+r[3]+r[4])/3 for r in T]; yT=[r[1] for r in T]
    pR,cR,nR=cov_ne_at_precision(scR,yT); pP,cP,nP=cov_ne_at_precision(scP,yT)
    rows_mod.append(dict(organism=org,tot_ne=tot_ne,real_P=pR,real_neCov=cR,real_calls=nR,prox_P=pP,prox_neCov=cP,prox_calls=nP))
    print(f"  {org:<13}{tot_ne:>8}  | {pR:.3f} {cR:.3f} {nR:>5} | {pP:.3f} {cP:.3f} {nP:>5}")

print("\n=== UNMODELED orgs: neCov @ P_ne>=0.90 (2-wheel vs 3-wheel proxy) ===")
print(f"{'organism':<22}{'tot_NE':>8}  | 2wheel P neCov | 3wheel P neCov  d_neCov")
MODELED15={"beril_Putida","beril_Keio","ecoli_BW25113_tradis","mtub","beril_Koxy","beril_WCS417","beril_PS","beril_psRCH2","beril_SyringaeB728a","beril_SyringaeB728a_mexBdelta","beril_pseudo1_N1B4","beril_pseudo3_N2E3","beril_pseudo5_N2C3_1","beril_pseudo6_N2E2","beril_pseudo13_GW456_L13"}
unmodeled=sorted({o for (o,_) in truth_by}-MODELED15); rows_un=[]
for org in unmodeled:
    T=build_table(org)
    if len(T)<50: continue
    tot_ne=sum(1 for r in T if r[1]==0)
    if tot_ne<10: continue
    s2=[(r[2]+r[3])/2 for r in T]; s3=[(r[2]+r[3]+r[4])/3 for r in T]; yT=[r[1] for r in T]
    p2,c2,n2=cov_ne_at_precision(s2,yT); p3,c3,n3=cov_ne_at_precision(s3,yT)
    rows_un.append(dict(organism=org,n=len(T),tot_ne=tot_ne,w2_P=p2,w2_neCov=c2,w3_P=p3,w3_neCov=c3,delta_neCov=round(c3-c2,3)))
rows_un.sort(key=lambda r:-r["n"])
for r in rows_un[:25]:
    print(f"  {r['organism']:<20}{r['tot_ne']:>8}  | {r['w2_P']:.3f} {r['w2_neCov']:.3f} | {r['w3_P']:.3f} {r['w3_neCov']:.3f}  {r['delta_neCov']:+.3f}")
m2=np.mean([r["w2_neCov"] for r in rows_un]); m3=np.mean([r["w3_neCov"] for r in rows_un])
mp2=np.mean([r["w2_P"] for r in rows_un]); mp3=np.mean([r["w3_P"] for r in rows_un])
print(f"\n  MEAN over {len(rows_un)} unmodeled orgs (@ P_ne>=0.90):")
print(f"    2-wheel: P={mp2:.3f}  neCov={m2:.3f}")
print(f"    3-wheel(proxy): P={mp3:.3f}  neCov={m3:.3f}   delta_neCov={(m3-m2)*100:+.1f}pp")
json.dump(dict(modeled=rows_mod,unmodeled=rows_un,mean_2wheel_neCov=float(m2),mean_3wheel_neCov=float(m3),mean_2wheel_P=float(mp2),mean_3wheel_P=float(mp3)),open(OUT/"fba_free_necoverage.json","w"),indent=2)
print(f"\nwrote outputs/orphan/fba_free_necoverage.json")
