"""Coverage analysis: what fraction of true essentials does the 3-wheel-with-
proxy actually catch, at matched precision?

The Step-3 result showed TTT-with-proxy precision 1.000 but only 2-5 calls per
organism, vs real-FBA TTT precision 0.957 with 83-112 calls. That's a
threshold artefact: proxy probabilities are concentrated, the 0.5 cut is too
strict. Real test: at MATCHED precision targets, who covers more essentials?

For each org we sweep proxy threshold and pick the operating point that hits
the target precision when combined with W1+W2 (also sweep their thresholds
the same way), then report essential-coverage (essCov).
"""
import csv, json, time, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
cobra.Configuration().solver = "glpk"

OUT = Path("outputs/orphan")
MODELS = Path("data/external_data/bigg_models")

# --- numpy logistic regression (matches fba_free_wheel.py) ---
class Scaler:
    def fit(self, X):
        self.mu = X.mean(0); self.sd = X.std(0); self.sd[self.sd<1e-9] = 1.0; return self
    def transform(self, X): return (X - self.mu)/self.sd
class Logit:
    def __init__(self, C=1.0, max_iter=400, class_weight=None):
        self.C=C; self.max_iter=max_iter; self.class_weight=class_weight
    def fit(self, X, y):
        n,d=X.shape; Xb=np.c_[X, np.ones(n)]; w=np.zeros(d+1)
        cw = np.where(y==1, 0.5/max(y.mean(),1e-6), 0.5/max(1-y.mean(),1e-6)) if self.class_weight=="balanced" else np.ones(n)
        lr=0.1
        for it in range(self.max_iter):
            z=np.clip(Xb@w,-30,30); p=1/(1+np.exp(-z))
            g=Xb.T@((p-y)*cw)/n + (w/self.C)*np.r_[np.ones(d),0]
            w -= lr*g
            if it%80==79: lr *= 0.5
        self.w=w; return self
    def predict_proba(self, X):
        Xb=np.c_[X, np.ones(len(X))]; z=np.clip(Xb@self.w,-30,30); p=1/(1+np.exp(-z))
        return np.c_[1-p, p]

# ---- shared ----
ALL = list(csv.DictReader(open("data/drive_import/labels/labels.csv")))
clade_of = {r["organism"]:r["clade"] for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
og_of = {(r["organism"], r["locus_tag"]):r["og_id"]
         for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")) if r.get("og_id")}
P = np.load(OUT/"af_torch_preds_aug.npz", allow_pickle=True)
_cl,_lt,_p = P["clade"], P["lt"], P["p"]
w1 = {(str(_cl[i]), str(_lt[i])): float(_p[i]) for i in range(len(_p)) if not np.isnan(_p[i])}

def cons_rate(held):
    n=defaultdict(int); e=defaultdict(int)
    for r in ALL:
        if clade_of.get(r["organism"])==held: continue
        og=og_of.get((r["organism"], r["locus_tag"]))
        if og: n[og]+=1; e[og]+=int(r["essential"])
    return {og: e[og]/n[og] for og in n if n[og]>=5}

# ---- run FBA + bridges (same as before) ----
RICH = ["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly",
        "his__L","ile__L","leu__L","lys__L","met__L","phe__L","pro__L","ser__L",
        "thr__L","trp__L","tyr__L","val__L","ade","gua","ura","thymd","cytd",
        "ins","adn","gsn","uri","thm","ribflv","nac","pnto__R","pydx","4abz","btn","fol"]
def fba_run(path,label):
    M=cobra.io.read_sbml_model(str(path))
    for x in RICH:
        rid=f"EX_{x}_e"
        if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
    g0=M.slim_optimize()
    print(f"  [{label}] g0={g0:.2f} ...", flush=True)
    de=single_gene_deletion(M); out={}
    for _,row in de.iterrows():
        gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
        out[gid]=int(row["growth"]<0.01*g0 or np.isnan(row["growth"]))
    print(f"    {sum(out.values())}/{len(out)} ess", flush=True)
    return out, M

print("Running FBA labels...", flush=True); t0=time.time()
fba_pu,_=fba_run(MODELS/"iJN1463.xml.gz","iJN1463")
fba_ec,_=fba_run(MODELS/"iML1515.xml.gz","iML1515")
fba_mt,_=fba_run(MODELS/"iEK1008.xml.gz","iEK1008")
fba_ko,M_kox=fba_run(MODELS/"iYL1228.xml.gz","iYL1228")
print(f"  ({time.time()-t0:.0f}s)\n", flush=True)

b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
mtub_lts={r["locus_tag"] for r in ALL if r["organism"]=="mtub"}
koxy_annot={r["locus_tag"]:r["product"]
            for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))
            if r["organism"]=="beril_Koxy" and r.get("product")}
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

# also build a real-FBA score per (org, lt) for the modeled orgs
real_fba_score={k:float(v) for k,v in fba_truth.items()}

# ---- features (52, same set as fba_free_wheel.py) ----
codon={(r["organism"],r["locus_tag"]):r for r in csv.DictReader(open("data/drive_import/labels/codon_features.csv"))}
orth={(r["organism"],r["locus_tag"]):r for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv"))}
coocc={(r["organism"],r["locus_tag"]):r for r in csv.DictReader(open("data/drive_import/labels/cooccurrence_features.csv"))}
reg={(r["organism"],r["locus_tag"]):r for r in csv.DictReader(open("data/drive_import/labels/regulator_features.csv"))}
annot={(r["organism"],r["locus_tag"]):(r.get("product") or "").lower()
       for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))}
KEYS=["synthase","synthetase","ligase","decarboxylase","dehydrogenase",
      "transferase","reductase","hydrolase","kinase","phosphatase",
      "isomerase","biosynth","ribosom","trna","rrna","metabolism",
      "cofactor","cell wall","peptidoglycan","membrane","atp","gtp",
      "transport","permease","abc","secretion","regulator","sensor",
      "histidine kinase","conjugat","hypothetical","unknown"]
def fv(o,lt):
    f=[]
    c=codon.get((o,lt))
    f.append(float(c["cai"]) if c and c["cai"] else 0)
    f.append(float(c["gc"]) if c and c["gc"] else 0)
    f.append(float(c["gc3"]) if c and c["gc3"] else 0)
    f.append(np.log1p(float(c["cds_length"])) if c and c["cds_length"] else 0)
    f.append(np.log1p(abs(float(c["intergenic_prev"]))) if c and c["intergenic_prev"] else 0)
    f.append(np.log1p(abs(float(c["intergenic_next"]))) if c and c["intergenic_next"] else 0)
    f.append(int(c["same_strand_prev"]=="True") if c else 0)
    f.append(int(c["same_strand_next"]=="True") if c else 0)
    h=orth.get((o,lt))
    f.append(int(h["own_fold"]) if h and h["own_fold"] else 0)
    f.append(np.log1p(float(h["n_paralogs_in_genome"])) if h else 0)
    f.append(np.log1p(float(h["family_size_total"])) if h else 0)
    f.append(np.log1p(float(h["family_n_organisms"])) if h else 0)
    f.append(int(h["is_orphan"]=="True") if h else 1)
    k=coocc.get((o,lt))
    f.append(float(k["cooccur_max_jaccard"]) if k and k["cooccur_max_jaccard"] else 0)
    f.append(np.log1p(float(k["cooccur_n_neighbors_50"])) if k else 0)
    f.append(np.log1p(float(k["cooccur_n_neighbors_80"])) if k else 0)
    g=reg.get((o,lt))
    f.append(int(g["is_regulator"]=="True") if g else 0)
    f.append(int(g["is_signaling"]=="True") if g else 0)
    f.append(int(g["is_transporter"]=="True") if g else 0)
    f.append(int(g["is_conditional"]=="True") if g else 0)
    p=annot.get((o,lt),"")
    for kw in KEYS: f.append(int(kw in p))
    return f

# train final proxy on all 4 modeled orgs
X,y,keys=[],[],[]
for (o,lt),e in fba_truth.items():
    X.append(fv(o,lt)); y.append(e); keys.append((o,lt))
X=np.array(X); y=np.array(y)
sc=Scaler().fit(X)
mf=Logit(class_weight="balanced",max_iter=400,C=1.0).fit(sc.transform(X), y)
truth_by={(r["organism"],r["locus_tag"]):int(r["essential"]) for r in ALL}
all_keys=list(truth_by.keys())
X60=np.array([fv(o,lt) for (o,lt) in all_keys])
proxy_score={k:float(s) for k,s in zip(all_keys, mf.predict_proba(sc.transform(X60))[:,1])}
print(f"[proxy] trained, scored {len(proxy_score)} genes\n", flush=True)

# ---- coverage analysis ----
# Build per-org score table: w1, w2, proxy, real_fba (where available), truth
def build_table(org):
    held=clade_of.get(org,"?"); rate=cons_rate(held); out=[]
    for (o,lt),tv in truth_by.items():
        if o!=org: continue
        og=og_of.get((o,lt))
        w1v=w1.get((held,lt), np.nan)
        w2v=rate.get(og, np.nan) if og else np.nan
        prx=proxy_score.get((o,lt), np.nan)
        rfb=real_fba_score.get((o,lt), np.nan)  # NaN if no model
        if np.isnan(w1v) or np.isnan(w2v) or np.isnan(prx): continue
        out.append((lt, tv, w1v, w2v, prx, rfb))
    return out

def coverage_at_precision(scores, truths, target_p=0.90, min_calls=10):
    """Sort by score DESC; sweep cutoff to find largest call set with precision>=target.
    Returns (precision, recall=essCov, n_calls). recall = TP / total_pos."""
    s=np.array(scores); y=np.array(truths)
    order=np.argsort(-s)
    yo=y[order]
    tp = np.cumsum(yo)
    n  = np.arange(1, len(yo)+1)
    prec = tp / n
    total_pos = y.sum()
    # find largest k where prec[k-1] >= target_p and n>=min_calls
    valid = (prec >= target_p) & (n >= min_calls)
    if not valid.any():
        return (0.0, 0.0, 0)
    k = np.max(np.where(valid)[0])
    return (float(prec[k]), float(tp[k]/max(total_pos,1)), int(n[k]+0))

def consensus_call(table, thr_w1, thr_w2, thr_w3, use_proxy=True):
    """Return (n_call, n_correct, n_ess_total). All 3 wheels >= threshold = essential."""
    calls=correct=0; ess_total=sum(r[1] for r in table)
    for (lt,tv,w1v,w2v,prx,rfb) in table:
        w3v = prx if use_proxy else rfb
        if np.isnan(w3v): continue
        if w1v>=thr_w1 and w2v>=thr_w2 and w3v>=thr_w3:
            calls+=1
            if tv==1: correct+=1
    return calls, correct, ess_total

# ---- modeled orgs: REAL-FBA vs PROXY at matched-precision-target ----
print("=== MODELED orgs: essCov @ matched precision ===")
print(f"{'organism':<15}{'tot_ess':>8}  | REAL-FBA: P recall calls | PROXY-W3': P recall calls")
modeled4=["beril_Putida","beril_Keio","mtub","beril_Koxy"]
rows_mod=[]
for org in modeled4:
    T = build_table(org)
    if not T: continue
    # for real-FBA we need rfb present; filter
    T_rf = [r for r in T if not np.isnan(r[5])]
    if not T_rf: continue
    ess_tot = sum(r[1] for r in T_rf)
    # Combined score = mean of normalized signals (each already in [0,1]); use this as a SINGLE rankable score per pipeline
    def combo(r, mode):
        w1v,w2v=r[2],r[3]; w3=r[4] if mode=="proxy" else r[5]
        if np.isnan(w3): return -1
        return (w1v + w2v + w3) / 3
    scR=[combo(r,"real") for r in T_rf]; yR=[r[1] for r in T_rf]
    scP=[combo(r,"proxy") for r in T_rf]; yP=[r[1] for r in T_rf]
    pR,rR,nR = coverage_at_precision(scR, yR, 0.90)
    pP,rP,nP = coverage_at_precision(scP, yP, 0.90)
    rows_mod.append(dict(organism=org, ess_total=ess_tot,
                         real_P=pR, real_essCov=rR, real_calls=nR,
                         prox_P=pP, prox_essCov=rP, prox_calls=nP))
    print(f"  {org:<13}{ess_tot:>8}  | {pR:.3f}  {rR:.3f}  {nR:>5}     "
          f"  | {pP:.3f}  {rP:.3f}  {nP:>5}")

print("\n=== UNMODELED orgs: essCov of 3-wheel (proxy) vs 2-wheel @ P>=0.90 ===")
print(f"{'organism':<22}{'tot_ess':>8}  | 2-wheel: P essCov | 3-wheel(proxy): P essCov  d_essCov")
MODELED15={"beril_Putida","beril_Keio","ecoli_BW25113_tradis","mtub","beril_Koxy",
           "beril_WCS417","beril_PS","beril_psRCH2","beril_SyringaeB728a",
           "beril_SyringaeB728a_mexBdelta","beril_pseudo1_N1B4","beril_pseudo3_N2E3",
           "beril_pseudo5_N2C3_1","beril_pseudo6_N2E2","beril_pseudo13_GW456_L13"}
unmodeled = sorted({o for (o,_) in truth_by} - MODELED15)
rows_un=[]
for org in unmodeled:
    T = build_table(org)
    if len(T)<50: continue
    ess_tot=sum(r[1] for r in T)
    if ess_tot<10: continue
    # 2-wheel score = mean(w1,w2); 3-wheel score = mean(w1,w2,proxy)
    s2=[(r[2]+r[3])/2 for r in T]
    s3=[(r[2]+r[3]+r[4])/3 for r in T]
    yT=[r[1] for r in T]
    p2,r2,n2 = coverage_at_precision(s2,yT,0.90)
    p3,r3,n3 = coverage_at_precision(s3,yT,0.90)
    rows_un.append(dict(organism=org, n=len(T), ess_total=ess_tot,
                        w2_P=p2, w2_essCov=r2, w2_calls=n2,
                        w3_P=p3, w3_essCov=r3, w3_calls=n3,
                        delta_essCov=round(r3-r2,3)))
rows_un.sort(key=lambda r:-r["n"])
for r in rows_un[:25]:
    print(f"  {r['organism']:<20}{r['ess_total']:>8}  | {r['w2_P']:.3f}  {r['w2_essCov']:.3f}"
          f"   | {r['w3_P']:.3f}  {r['w3_essCov']:.3f}    {r['delta_essCov']:+.3f}")
m2=np.mean([r["w2_essCov"] for r in rows_un])
m3=np.mean([r["w3_essCov"] for r in rows_un])
mp2=np.mean([r["w2_P"] for r in rows_un])
mp3=np.mean([r["w3_P"] for r in rows_un])
print(f"\n  MEAN over {len(rows_un)} unmodeled orgs (@ P>=0.90):")
print(f"    2-wheel: P={mp2:.3f}  essCov={m2:.3f}")
print(f"    3-wheel(proxy): P={mp3:.3f}  essCov={m3:.3f}   delta_essCov={(m3-m2)*100:+.1f}pp")

json.dump(dict(modeled=rows_mod, unmodeled=rows_un,
               mean_2wheel_essCov=float(m2), mean_3wheel_essCov=float(m3),
               mean_2wheel_P=float(mp2),     mean_3wheel_P=float(mp3)),
          open(OUT/"fba_free_coverage.json","w"), indent=2)
print(f"\nwrote outputs/orphan/fba_free_coverage.json")
