"""Build a learned W3' that REPLACES the FBA wheel using only features
available for ALL 60 organisms (no metabolic model required).

Training labels: fba_essential per gene on the 4 organisms with native FBA runs
                 (beril_Putida, beril_Keio via b-bridge, mtub, beril_Koxy via
                 gene-name bridge — same set that fed metabolic_transfer.py).
Train mode    : Leave-one-organism-out — for each modeled org, train on the
                 other 3 and predict its FBA-essential.
Features      : codon, orthology STRUCTURE, regulator booleans, cooccurrence
                 STRUCTURE, and gene-product keyword counts. Deliberately
                 EXCLUDES any *_frac_essential field so the proxy stays
                 orthogonal to W2 (conservation).

Then drop the proxy into the 3-wheel system in place of real FBA and compare
the TTT-consensus precision against the actual FBA wheel (the 4-organism
bigg_3wheel result).
"""
import csv, json, time, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
cobra.Configuration().solver = "glpk"

# --- minimal numpy logistic regression (no sklearn) ---
class Scaler:
    def fit(self, X):
        self.mu = X.mean(0); self.sd = X.std(0); self.sd[self.sd<1e-9] = 1.0; return self
    def transform(self, X): return (X - self.mu)/self.sd
class Logit:
    def __init__(self, C=1.0, max_iter=400, class_weight=None):
        self.C=C; self.max_iter=max_iter; self.class_weight=class_weight
    def fit(self, X, y):
        n, d = X.shape
        Xb = np.c_[X, np.ones(n)]
        w = np.zeros(d+1)
        if self.class_weight == "balanced":
            p1 = y.mean(); cw = np.where(y==1, 0.5/max(p1,1e-6), 0.5/max(1-p1,1e-6))
        else:
            cw = np.ones(n)
        lr = 0.1
        for it in range(self.max_iter):
            z = Xb @ w
            z = np.clip(z, -30, 30)
            p = 1.0/(1.0+np.exp(-z))
            grad = Xb.T @ ((p - y) * cw) / n + (w / self.C) * np.r_[np.ones(d), 0]
            # adaptive step: decay
            w -= lr * grad
            if it % 80 == 79: lr *= 0.5
        self.w = w; return self
    def predict_proba(self, X):
        Xb = np.c_[X, np.ones(len(X))]
        z = np.clip(Xb @ self.w, -30, 30)
        p = 1.0/(1.0+np.exp(-z))
        return np.c_[1-p, p]
    @property
    def coef_(self): return self.w[:-1].reshape(1,-1)
StandardScaler = Scaler
LogisticRegression = Logit

OUT = Path("outputs/orphan")
MODELS = Path("data/external_data/bigg_models")

# ---- shared data ----
ALL = list(csv.DictReader(open("data/drive_import/labels/labels.csv")))
clade_of = {r["organism"]:r["clade"] for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
og_of = {(r["organism"], r["locus_tag"]):r["og_id"]
         for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")) if r.get("og_id")}

# ---- Wheel 1 (preds) and Wheel 2 (leak-clean conservation) helpers ----
P = np.load(OUT/"af_torch_preds_aug.npz", allow_pickle=True)
_cl, _lt, _p = P["clade"], P["lt"], P["p"]
w1 = {(str(_cl[i]), str(_lt[i])): float(_p[i]) for i in range(len(_p)) if not np.isnan(_p[i])}
print(f"[load] w1 preds: {len(w1)}", flush=True)

def cons_rate(held):
    n=defaultdict(int); e=defaultdict(int)
    for r in ALL:
        if clade_of.get(r["organism"])==held: continue
        og = og_of.get((r["organism"], r["locus_tag"]))
        if og: n[og]+=1; e[og]+=int(r["essential"])
    return {og: e[og]/n[og] for og in n if n[og]>=5}

# ---- run FBA on the same 4 models to get training labels ----
RICH = ["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly",
        "his__L","ile__L","leu__L","lys__L","met__L","phe__L","pro__L","ser__L",
        "thr__L","trp__L","tyr__L","val__L","ade","gua","ura","thymd","cytd",
        "ins","adn","gsn","uri","thm","ribflv","nac","pnto__R","pydx","4abz","btn","fol"]
def fba_ess(path, label):
    M = cobra.io.read_sbml_model(str(path))
    for x in RICH:
        rid=f"EX_{x}_e"
        if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
    g0 = M.slim_optimize()
    print(f"  [{label}] growth={g0:.3f} running deletion...", flush=True)
    de = single_gene_deletion(M)
    out={}
    for _,row in de.iterrows():
        gid = list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
        out[gid] = int(row["growth"]<0.01*g0 or np.isnan(row["growth"]))
    print(f"    {sum(out.values())} ess / {len(out)}", flush=True)
    return out, M

print("Step 1: FBA on 4 models (this is the TRAINING-LABEL source only)", flush=True)
t0=time.time()
fba_pu, _ = fba_ess(MODELS/"iJN1463.xml.gz", "iJN1463")
fba_ec, _ = fba_ess(MODELS/"iML1515.xml.gz", "iML1515")
fba_mt, _ = fba_ess(MODELS/"iEK1008.xml.gz", "iEK1008")
fba_ko, M_kox = fba_ess(MODELS/"iYL1228.xml.gz", "iYL1228")
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# bridges
b_to_keio = {r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
mtub_og = {}
for r in csv.DictReader(open("colab/work/og_assignments.csv")):
    if r["organism"].startswith("deg_Mycobacterium_tuberculosis_H37Rv"):
        mtub_og.setdefault(r["locus_tag"], r["og_id"])
# koxy gene-name bridge
koxy_annot={r["locus_tag"]:r["product"]
            for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))
            if r["organism"]=="beril_Koxy" and r.get("product")}
koxy_tok=defaultdict(list)
for lt, prod in koxy_annot.items():
    for tk in re.findall(r"[A-Za-z0-9]{4,}", prod): koxy_tok[tk.lower()].append(lt)
kpn_to_koxy={}
for g in M_kox.genes:
    gn=(g.name or "").strip()
    if gn and g.id!=gn:
        mm=koxy_tok.get(gn.lower(), [])
        if len(mm)==1: kpn_to_koxy[g.id]=mm[0]

# fba per (our_org, our_locus) — only where we have a bridge
fba_truth = {}
for g, e in fba_pu.items():
    fba_truth[("beril_Putida", g)] = e
for b, e in fba_ec.items():
    if b in b_to_keio: fba_truth[("beril_Keio", b_to_keio[b])] = e
# mtub: model gene = Rv tag (try direct first) else via OG
truth_org = "mtub"
truth_lts = {r["locus_tag"] for r in ALL if r["organism"]==truth_org}
for g, e in fba_mt.items():
    if g in truth_lts:
        fba_truth[(truth_org, g)] = e
for g, e in fba_ko.items():
    if g in kpn_to_koxy: fba_truth[("beril_Koxy", kpn_to_koxy[g])] = e
print(f"\n[labels] fba_truth: {len(fba_truth)} genes across 4 orgs", flush=True)
for org in ["beril_Putida","beril_Keio","mtub","beril_Koxy"]:
    sub = [v for (o,lt),v in fba_truth.items() if o==org]
    print(f"  {org:<15} n={len(sub)} fba_ess={sum(sub)}", flush=True)

# ---- feature tables (all orgs) ----
codon = {(r["organism"], r["locus_tag"]):r
         for r in csv.DictReader(open("data/drive_import/labels/codon_features.csv"))}
orth = {(r["organism"], r["locus_tag"]):r
         for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv"))}
coocc = {(r["organism"], r["locus_tag"]):r
         for r in csv.DictReader(open("data/drive_import/labels/cooccurrence_features.csv"))}
reg = {(r["organism"], r["locus_tag"]):r
         for r in csv.DictReader(open("data/drive_import/labels/regulator_features.csv"))}
annot = {(r["organism"], r["locus_tag"]):(r.get("product") or "").lower()
         for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))}

# product-string keyword features — coarse but powerful for metabolic shape
KEYS = ["synthase","synthetase","ligase","decarboxylase","dehydrogenase",
        "transferase","reductase","hydrolase","kinase","phosphatase",
        "isomerase","biosynth","ribosom","trna","rrna","metabolism",
        "cofactor","cell wall","peptidoglycan","membrane","atp","gtp",
        "transport","permease","abc","secretion","regulator","sensor",
        "histidine kinase","conjugat","hypothetical","unknown"]

def fv(o, lt):
    f = []
    # codon
    c = codon.get((o,lt))
    f.append(float(c["cai"]) if c and c["cai"] else 0.0)
    f.append(float(c["gc"])  if c and c["gc"]  else 0.0)
    f.append(float(c["gc3"]) if c and c["gc3"] else 0.0)
    f.append(np.log1p(float(c["cds_length"])) if c and c["cds_length"] else 0.0)
    f.append(np.log1p(abs(float(c["intergenic_prev"]))) if c and c["intergenic_prev"] else 0.0)
    f.append(np.log1p(abs(float(c["intergenic_next"]))) if c and c["intergenic_next"] else 0.0)
    f.append(int(c["same_strand_prev"]=="True") if c else 0)
    f.append(int(c["same_strand_next"]=="True") if c else 0)
    # orthology structure (NO *_frac_essential)
    h = orth.get((o,lt))
    f.append(int(h["own_fold"]) if h and h["own_fold"] else 0)
    f.append(np.log1p(float(h["n_paralogs_in_genome"])) if h else 0.0)
    f.append(np.log1p(float(h["family_size_total"])) if h else 0.0)
    f.append(np.log1p(float(h["family_n_organisms"])) if h else 0.0)
    f.append(int(h["is_orphan"]=="True") if h else 1)
    # cooccurrence (NO essential_neighbor_frac)
    k = coocc.get((o,lt))
    f.append(float(k["cooccur_max_jaccard"]) if k and k["cooccur_max_jaccard"] else 0.0)
    f.append(np.log1p(float(k["cooccur_n_neighbors_50"])) if k else 0.0)
    f.append(np.log1p(float(k["cooccur_n_neighbors_80"])) if k else 0.0)
    # regulator booleans
    g = reg.get((o,lt))
    f.append(int(g["is_regulator"]=="True") if g else 0)
    f.append(int(g["is_signaling"]=="True") if g else 0)
    f.append(int(g["is_transporter"]=="True") if g else 0)
    f.append(int(g["is_conditional"]=="True") if g else 0)
    # product keywords
    p = annot.get((o,lt), "")
    for kw in KEYS:
        f.append(int(kw in p))
    return f

FEAT_NAMES = (["cai","gc","gc3","log_cdslen","log_intergenic_prev","log_intergenic_next",
               "same_strand_prev","same_strand_next",
               "own_fold","log_paralogs","log_family_size","log_family_n_orgs","is_orphan",
               "cooccur_jaccard","log_n_neigh50","log_n_neigh80",
               "is_regulator","is_signaling","is_transporter","is_conditional"]
              + [f"kw_{k.replace(' ','_')}" for k in KEYS])
print(f"\n[features] {len(FEAT_NAMES)} per gene", flush=True)

# ---- assemble training set & run leave-one-out ----
def build_set(org_filter):
    X, y, lts = [], [], []
    for (o, lt), e in fba_truth.items():
        if org_filter and o not in org_filter: continue
        X.append(fv(o, lt)); y.append(e); lts.append((o, lt))
    return np.array(X), np.array(y), lts

MODELED4 = ["beril_Putida","beril_Keio","mtub","beril_Koxy"]
print("\nStep 2: leave-one-org-out FBA-prediction AUC (proxy vs real FBA)", flush=True)

def auc(s,y):
    s=np.array(s); y=np.array(y)
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s))
    pos=y==1
    if pos.sum()==0 or (~pos).sum()==0: return 0.5
    return float((r[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*(~pos).sum()))

loo_aucs = []
proxy_predict = {}   # (org, lt) -> proxy_score (only for held-out org)
for held in MODELED4:
    Xtr, ytr, _   = build_set(set(MODELED4)-{held})
    Xte, yte, lts = build_set({held})
    if len(yte)<20: print(f"  {held}: too few held-out ({len(yte)}) -- skip"); continue
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    m.fit(sc.transform(Xtr), ytr)
    s = m.predict_proba(sc.transform(Xte))[:,1]
    a = auc(s, yte)
    loo_aucs.append((held, len(yte), int(sum(yte)), a))
    for (k, lt), sv in zip(lts, s):
        proxy_predict[(k, lt)] = float(sv)
    print(f"  held={held:<15} n={len(yte):>5} ess={int(sum(yte)):>4}  proxy_AUC_vs_FBA={a:.3f}", flush=True)

# train one FINAL model on all 4 modeled orgs (for use on unmodeled orgs)
Xall, yall, _ = build_set(set(MODELED4))
sc_all = StandardScaler().fit(Xall)
mf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(sc_all.transform(Xall), yall)
# log feature importance
coef_sorted = sorted(zip(FEAT_NAMES, mf.coef_[0]), key=lambda kv:-abs(kv[1]))
print("\n[final model] top features by |coef|:")
for n,c in coef_sorted[:12]: print(f"  {n:<25}{c:+.3f}")

# ---- 3-wheel comparison: proxy-W3' vs real-W3 ----
# Use the SAME 4 organisms, and for each, do the TTT-consensus test exactly
# like bigg_3wheel.py: W1>=0.5, W2>=0.5, W3'>=0.5 OR W3>=0.5 (real)
print("\nStep 3: 3-wheel TTT precision: REAL FBA vs PROXY-W3'", flush=True)
print(f"{'organism':<18}{'n':>6}{'ess':>5}"
      f"{'TTT_real_n':>11}{'TTT_real_P':>12}"
      f"{'TTT_prox_n':>11}{'TTT_prox_P':>12}{'delta':>8}", flush=True)

def prec(g,t): return sum(1 for r in g if r["truth"]==t)/max(len(g),1)

results=[]
truth_by = {(r["organism"], r["locus_tag"]): int(r["essential"]) for r in ALL}
for our_org in MODELED4:
    held_clade = clade_of.get(our_org,"?")
    rate = cons_rate(held_clade)
    # only consider genes with all 3 signals (real-FBA labelled = in fba_truth)
    table=[]
    for (o, lt), real_fba in fba_truth.items():
        if o != our_org: continue
        if (o, lt) not in truth_by: continue
        og = og_of.get((o, lt))
        w1v = w1.get((held_clade, lt), np.nan)
        w2v = rate.get(og, np.nan) if og else np.nan
        proxy = proxy_predict.get((o, lt), np.nan)
        if np.isnan(w1v) or np.isnan(w2v) or np.isnan(proxy): continue
        table.append(dict(lt=lt, truth=truth_by[(o,lt)],
                          w1=int(w1v>=0.5), w2=int(w2v>=0.5),
                          w3_real=int(real_fba>=0.5),
                          w3_proxy=int(proxy>=0.5)))
    if not table: continue
    ttt_real  = [r for r in table if r["w1"]==r["w2"]==r["w3_real"]==1]
    ttt_proxy = [r for r in table if r["w1"]==r["w2"]==r["w3_proxy"]==1]
    pR = prec(ttt_real,1); pP = prec(ttt_proxy,1)
    n_ess=sum(r["truth"] for r in table)
    results.append(dict(organism=our_org, n=len(table), ess=n_ess,
                        ttt_real_n=len(ttt_real), ttt_real_P=round(pR,3),
                        ttt_proxy_n=len(ttt_proxy), ttt_proxy_P=round(pP,3),
                        delta=round(pP-pR,3)))
    print(f"  {our_org:<16}{len(table):>6}{n_ess:>5}"
          f"{len(ttt_real):>11}{pR:>12.3f}"
          f"{len(ttt_proxy):>11}{pP:>12.3f}{pP-pR:>+8.3f}", flush=True)

mean_real = np.mean([r["ttt_real_P"] for r in results])
mean_prox = np.mean([r["ttt_proxy_P"] for r in results])
print(f"\n  MEAN: TTT precision REAL-FBA={mean_real:.3f}  PROXY-W3'={mean_prox:.3f}  delta={mean_prox-mean_real:+.3f}")

# ---- Apply proxy to ALL 60 orgs as a portable W3' ----
proxy_all = {}
all_keys = list(truth_by.keys())
X60 = np.array([fv(o, lt) for (o, lt) in all_keys])
S60 = mf.predict_proba(sc_all.transform(X60))[:,1]
for (k, sv) in zip(all_keys, S60):
    proxy_all[k] = float(sv)
print(f"\n[proxy_all] computed for {len(proxy_all)} (org, lt) pairs (all 60 orgs)")

# Validate as a wheel on UNMODELED orgs: AUC of proxy vs truth + 3-wheel TTT precision
MODELED15 = {"beril_Putida","beril_Keio","ecoli_BW25113_tradis","mtub","beril_Koxy",
             "beril_WCS417","beril_PS","beril_psRCH2","beril_SyringaeB728a",
             "beril_SyringaeB728a_mexBdelta","beril_pseudo1_N1B4","beril_pseudo3_N2E3",
             "beril_pseudo5_N2C3_1","beril_pseudo6_N2E2","beril_pseudo13_GW456_L13"}
unmodeled = sorted({o for (o,_) in truth_by} - MODELED15)
print(f"\nStep 4: proxy-W3' as a wheel on {len(unmodeled)} UNMODELED orgs", flush=True)
print(f"{'organism':<22}{'n':>6}{'ess':>5}{'proxy_AUC':>11}{'W1+W2_P':>10}{'TTT_prox_P':>13}{'delta_pp':>11}")
rows4=[]
for org in unmodeled:
    held = clade_of.get(org, "?")
    rate = cons_rate(held)
    table=[]
    for (o, lt), tv in truth_by.items():
        if o != org: continue
        og = og_of.get((o, lt))
        w1v = w1.get((held, lt), np.nan)
        w2v = rate.get(og, np.nan) if og else np.nan
        prx = proxy_all.get((o, lt), np.nan)
        if np.isnan(w1v) or np.isnan(w2v) or np.isnan(prx): continue
        table.append(dict(lt=lt, truth=tv,
                          w1=int(w1v>=0.5), w2=int(w2v>=0.5), w3p=int(prx>=0.5),
                          prxs=prx))
    if len(table)<30: continue
    n_ess = sum(r["truth"] for r in table)
    a_prx = auc([r["prxs"] for r in table], [r["truth"] for r in table])
    w1w2 = [r for r in table if r["w1"]==1 and r["w2"]==1]
    ttt = [r for r in table if r["w1"]==r["w2"]==r["w3p"]==1]
    pw = prec(w1w2,1); pT = prec(ttt,1)
    rows4.append(dict(organism=org, n=len(table), ess=n_ess,
                      proxy_AUC=round(a_prx,3),
                      w1w2_P=round(pw,3), ttt_proxy_P=round(pT,3),
                      delta_pp=round((pT-pw)*100,1)))
rows4.sort(key=lambda r:-r["n"])
for r in rows4[:25]:
    print(f"  {r['organism']:<20}{r['n']:>6}{r['ess']:>5}{r['proxy_AUC']:>11.3f}"
          f"{r['w1w2_P']:>10.3f}{r['ttt_proxy_P']:>13.3f}{r['delta_pp']:>+10.1f}pp")
mA = np.mean([r["proxy_AUC"] for r in rows4])
mW = np.mean([r["w1w2_P"] for r in rows4])
mT = np.mean([r["ttt_proxy_P"] for r in rows4])
print(f"\n  MEAN over {len(rows4)} unmodeled orgs: proxy_AUC={mA:.3f}  "
      f"W1+W2_P={mW:.3f}  TTT_proxy_P={mT:.3f}  proxy lift={(mT-mW)*100:+.1f}pp")

json.dump(dict(loo=loo_aucs, three_wheel_modeled=results,
               three_wheel_unmodeled=rows4,
               feat_importance=[(n, float(c)) for n,c in coef_sorted],
               n_features=len(FEAT_NAMES), n_train_genes=len(fba_truth)),
          open(OUT/"fba_free_wheel.json","w"), indent=2)
print(f"\nwrote outputs/orphan/fba_free_wheel.json", flush=True)
