"""Triage workflow: how big is the leftover after confident calls + W2 family filter?

Step 1: at P>=0.90 in both directions, what fraction of each organism gets
        confidently called (essential OR non-essential)?
Step 2: the leftover = the SOUP. Inside the soup, how many genes have weak
        W2 family signal -- meaning:
          - no orthogroup (orphan / hypothetical)
          - OR family_n_organisms < threshold (clade-specific)
          - OR very low OG conservation rate
        These can be REMOVED from the experimental candidate pool: they're
        accessory/novelty genes and ess base-rate among them is low.
Step 3: shortlist = soup minus weak-W2 genes. Measure (n, ess_rate, lift).
        That's the experimental hit-and-trial set.
"""
import csv, json, time, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
cobra.Configuration().solver = "glpk"
OUT = Path("outputs/orphan"); MODELS = Path("data/external_data/bigg_models")

# --- numpy logit (same as before) ---
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
            w-=lr*(Xb.T@((p-y)*cw)/n + (w/self.C)*np.r_[np.ones(d),0])
            if it%80==79: lr*=0.5
        self.w=w; return self
    def predict_proba(self,X):
        Xb=np.c_[X,np.ones(len(X))]; z=np.clip(Xb@self.w,-30,30); p=1/(1+np.exp(-z)); return np.c_[1-p,p]

ALL=list(csv.DictReader(open("data/drive_import/labels/labels.csv")))
clade_of={r["organism"]:r["clade"] for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
og_of={(r["organism"],r["locus_tag"]):r["og_id"] for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")) if r.get("og_id")}
orth_row={(r["organism"],r["locus_tag"]):r for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv"))}
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

# proxy: load via same FBA + feature pipeline (abridged)
RICH=["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly","his__L","ile__L","leu__L","lys__L","met__L","phe__L","pro__L","ser__L","thr__L","trp__L","tyr__L","val__L","ade","gua","ura","thymd","cytd","ins","adn","gsn","uri","thm","ribflv","nac","pnto__R","pydx","4abz","btn","fol"]
def fba_run(path,label):
    M=cobra.io.read_sbml_model(str(path))
    for x in RICH:
        rid=f"EX_{x}_e"
        if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
    g0=M.slim_optimize()
    de=single_gene_deletion(M); out={}
    for _,row in de.iterrows():
        gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
        out[gid]=int(row["growth"]<0.01*g0 or np.isnan(row["growth"]))
    print(f"  [{label}] {sum(out.values())}/{len(out)} ess",flush=True)
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

codon={(r["organism"],r["locus_tag"]):r for r in csv.DictReader(open("data/drive_import/labels/codon_features.csv"))}
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
    h=orth_row.get((o,lt))
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
X,y=[],[]
for (o,lt),e in fba_truth.items(): X.append(fv(o,lt)); y.append(e)
X=np.array(X); y=np.array(y); sc=Scaler().fit(X)
mf=Logit(class_weight="balanced").fit(sc.transform(X),y)
truth_by={(r["organism"],r["locus_tag"]):int(r["essential"]) for r in ALL}
all_keys=list(truth_by.keys()); X60=np.array([fv(o,lt) for (o,lt) in all_keys])
proxy_score={k:float(s) for k,s in zip(all_keys,mf.predict_proba(sc.transform(X60))[:,1])}
print(f"[proxy] scored {len(proxy_score)}\n",flush=True)

# ---- triage workflow ----
# For each unmodeled org:
#   build per-gene table (W1, W2, proxy, truth, OG, family_size, is_orphan, family_n_orgs)
#   3-wheel score = mean(W1,W2,proxy)
#   Tier A = top-K confidently essential at P>=0.90
#   Tier C = top-K confidently NON-essential at P_ne>=0.90
#   SOUP   = remainder
#   In the soup, filter weak W2: drop if (no OG) OR (is_orphan) OR (family_n_organisms < 6)
#   SHORTLIST = soup minus weak-W2; report n, ess_rate, base_rate, lift

def build_table(org):
    held=clade_of.get(org,"?"); rate=cons_rate(held); out=[]
    for (o,lt),tv in truth_by.items():
        if o!=org: continue
        og=og_of.get((o,lt)); w1v=w1.get((held,lt),np.nan); w2v=rate.get(og,np.nan) if og else np.nan
        prx=proxy_score.get((o,lt),np.nan)
        if np.isnan(w1v) or np.isnan(w2v) or np.isnan(prx): continue
        h=orth_row.get((o,lt))
        fam_n_orgs=int(float(h["family_n_organisms"])) if (h and h["family_n_organisms"]) else 0
        is_orphan=(h is not None and h.get("is_orphan")=="True")
        out.append(dict(lt=lt,truth=tv,w1=w1v,w2=w2v,prx=prx,og=og,
                        fam_n_orgs=fam_n_orgs, is_orphan=is_orphan))
    return out

def split_at_precision(table, ess_target=0.90, ne_target=0.90):
    s=np.array([(r["w1"]+r["w2"]+r["prx"])/3 for r in table])
    y=np.array([r["truth"] for r in table])
    # tier A (essential): sort DESC, sweep
    od=np.argsort(-s); ya=y[od]
    cum=np.cumsum(ya); n=np.arange(1,len(ya)+1); p_ess=cum/n
    valid=p_ess>=ess_target
    kA=int(np.max(np.where(valid)[0])+1) if valid.any() else 0
    A_idx=set(od[:kA].tolist())
    # tier C (non-ess): sort ASC, sweep
    oa=np.argsort(s); yc=y[oa]; ne=(yc==0).astype(int)
    cumN=np.cumsum(ne); p_ne=cumN/n
    validN=p_ne>=ne_target
    kC=int(np.max(np.where(validN)[0])+1) if validN.any() else 0
    C_idx=set(oa[:kC].tolist())
    return A_idx, C_idx

MODELED15={"beril_Putida","beril_Keio","ecoli_BW25113_tradis","mtub","beril_Koxy","beril_WCS417","beril_PS","beril_psRCH2","beril_SyringaeB728a","beril_SyringaeB728a_mexBdelta","beril_pseudo1_N1B4","beril_pseudo3_N2E3","beril_pseudo5_N2C3_1","beril_pseudo6_N2E2","beril_pseudo13_GW456_L13"}

print("=== Triage funnel per unmodeled organism ===")
print(f"{'organism':<22}{'N':>5}{'baseR':>7}"
      f"{'A':>6}{'A_P':>6}{'C':>5}{'C_P':>5}"
      f"{'soup':>6}{'soupR':>7}"
      f"{'drop':>5}{'short':>6}{'shortR':>8}{'lift':>6}", flush=True)
rows=[]
for org in sorted({o for (o,_) in truth_by} - MODELED15):
    T=build_table(org)
    if len(T)<200: continue
    N=len(T); base=sum(r["truth"] for r in T)/N
    A,C = split_at_precision(T, 0.90, 0.90)
    overlap = A & C
    if overlap: C = C - A  # essential trumps
    soup_idx = set(range(N)) - A - C
    soup = [T[i] for i in soup_idx]
    soup_ess = sum(r["truth"] for r in soup)
    soupR = soup_ess/len(soup) if soup else 0
    # weak W2 filter: orphan OR no OG OR family_n_orgs < 6
    def weak(r): return r["is_orphan"] or (not r["og"]) or r["fam_n_orgs"]<6
    dropped = [r for r in soup if weak(r)]
    short = [r for r in soup if not weak(r)]
    short_ess = sum(r["truth"] for r in short)
    shortR = short_ess/len(short) if short else 0
    drop_ess = sum(r["truth"] for r in dropped)
    A_correct = sum(T[i]["truth"] for i in A)
    A_P = A_correct/len(A) if A else 0
    C_correct = sum(1-T[i]["truth"] for i in C)
    C_P = C_correct/len(C) if C else 0
    rows.append(dict(organism=org, N=N, base=base,
                     A_n=len(A), A_P=A_P, C_n=len(C), C_P=C_P,
                     soup_n=len(soup), soup_R=soupR,
                     drop_n=len(dropped), drop_R=drop_ess/max(len(dropped),1),
                     short_n=len(short), short_R=shortR,
                     lift=shortR/max(base,1e-6)))
    print(f"  {org:<20}{N:>5}{base:>7.2f}"
          f"{len(A):>6}{A_P:>6.2f}{len(C):>5}{C_P:>5.2f}"
          f"{len(soup):>6}{soupR:>7.2f}"
          f"{len(dropped):>5}{len(short):>6}{shortR:>8.2f}{shortR/max(base,1e-6):>6.1f}x")

# aggregate
N_tot=sum(r["N"] for r in rows); A_tot=sum(r["A_n"] for r in rows); C_tot=sum(r["C_n"] for r in rows)
soup_tot=sum(r["soup_n"] for r in rows); short_tot=sum(r["short_n"] for r in rows)
ess_tot=sum(r["N"]*r["base"] for r in rows)
short_ess_tot=sum(r["short_n"]*r["short_R"] for r in rows)
A_ess_tot=sum(r["A_n"]*r["A_P"] for r in rows)
print(f"\n=== Aggregate over {len(rows)} unmodeled organisms ===")
print(f"  Total genes:           {N_tot:>7}")
print(f"  Tier A (essential):    {A_tot:>7}  ({A_tot/N_tot*100:.1f}%)  precision~{A_ess_tot/A_tot:.2f}")
print(f"  Tier C (non-ess):      {C_tot:>7}  ({C_tot/N_tot*100:.1f}%)")
print(f"  Soup (uncalled):       {soup_tot:>7}  ({soup_tot/N_tot*100:.1f}%)")
print(f"  + dropped (weak W2):   {soup_tot-short_tot:>7}  ({(soup_tot-short_tot)/N_tot*100:.1f}% of total)")
print(f"  -> shortlist for wet:  {short_tot:>7}  ({short_tot/N_tot*100:.1f}% of total)")
print(f"  Shortlist ess rate:    {short_ess_tot/short_tot:.3f}  vs base {ess_tot/N_tot:.3f}  ({(short_ess_tot/short_tot)/(ess_tot/N_tot):.1f}x)")
print(f"  Essentials in shortlist: {int(short_ess_tot)}  /  {int(ess_tot-A_ess_tot)} essentials remaining after Tier A "
      f"= {short_ess_tot/max(ess_tot-A_ess_tot,1)*100:.0f}% recovery")

json.dump(rows, open(OUT/"triage_funnel.json","w"), indent=2)
print(f"\nwrote outputs/orphan/triage_funnel.json")
