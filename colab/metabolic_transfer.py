"""FBA-necessity propagation: extend the metabolic wheel to organisms WITHOUT
a metabolic model, by transferring FBA-essentiality across orthogroups.

Only 15 / 60 of our organisms have (or can borrow) a BiGG model. The other 45
are stuck on 2 wheels. But metabolic necessity is a property of the REACTION /
orthogroup, not the organism: if gene G is FBA-essential in the modeled
organisms that carry orthogroup OG, then a gene in OG in an UNMODELED organism
is likely metabolically essential too.

Step 1: run single-gene-deletion FBA (rich medium) on 4 DISTINCT models
        covering 4 distinct organisms:
          iJN1463  -> beril_Putida   (native PP_ ids)
          iML1515  -> beril_Keio     (b-number, via our bridge)
          iEK1008  -> mtub           (native Rv ids, OG via og_assignments)
          iYL1228  -> beril_Koxy     (gene-name bridge)
Step 2: map each FBA-essential gene -> orthogroup -> build og_fba_rate
Step 3: propagate og_fba_rate to ALL 60 organisms (a per-gene metabolic prior)
Step 4: VALIDATE on the 45 organisms WITHOUT their own model:
          - AUC of og_fba_rate vs truth
          - does it add precision on top of conservation (Wheel 2)?

Output: outputs/orphan/metabolic_transfer.json / .png
"""
import csv, time, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
cobra.Configuration().solver = "glpk"

OUT = Path("outputs/orphan")
MODELS = Path("data/external_data/bigg_models")
RICH = ["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly",
        "his__L","ile__L","leu__L","lys__L","met__L","phe__L","pro__L","ser__L",
        "thr__L","trp__L","tyr__L","val__L","ade","gua","ura","thymd","cytd",
        "ins","adn","gsn","uri","thm","ribflv","nac","pnto__R","pydx","4abz","btn","fol"]
def fba_ess(path, label):
    M = cobra.io.read_sbml_model(str(path))
    for x in RICH:
        rid=f"EX_{x}_e"
        if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
    g0=M.slim_optimize()
    print(f"  [{label}] growth={g0:.3f}  running deletion...", flush=True)
    de=single_gene_deletion(M)
    out={}
    for _,row in de.iterrows():
        gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
        out[gid]=(row["growth"]<0.01*g0 or np.isnan(row["growth"]))
    print(f"    {sum(out.values())} essential / {len(out)} genes", flush=True)
    return out

# ---- shared ----
ALL = list(csv.DictReader(open("data/drive_import/labels/labels.csv")))
clade_of = {r["organism"]:r["clade"] for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
og_of = {}
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"): og_of[(r["organism"],r["locus_tag"])]=r["og_id"]
# mtub OGs come from og_assignments (mmseqs), keyed by deg_Mycobacterium...
mtub_og = {}
for r in csv.DictReader(open("colab/work/og_assignments.csv")):
    if r["organism"].startswith("deg_Mycobacterium_tuberculosis_H37Rv"):
        mtub_og.setdefault(r["locus_tag"], r["og_id"])
keio_bridge = {r["locus_tag"]:r["b_number"]
               for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
b_to_keio = {v:k for k,v in keio_bridge.items()}

# ---- Step 1: run 4 models ----
print("Step 1: FBA gene-deletion on 4 distinct models", flush=True)
t0=time.time()
fba = {
    "putida": fba_ess(MODELS/"iJN1463.xml.gz", "iJN1463"),
    "keio":   fba_ess(MODELS/"iML1515.xml.gz", "iML1515"),
    "mtub":   fba_ess(MODELS/"iEK1008.xml.gz", "iEK1008"),
    "koxy":   fba_ess(MODELS/"iYL1228.xml.gz", "iYL1228"),
}
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ---- Step 2: map FBA-essential genes -> OG; build og_fba_rate ----
# Klebsiella gene-name bridge (rebuild)
import re
koxy_annot = {r["locus_tag"]:r["product"]
              for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))
              if r["organism"]=="beril_Koxy" and r.get("product")}
koxy_tok = defaultdict(list)
for lt, prod in koxy_annot.items():
    for tk in re.findall(r"[A-Za-z0-9]{4,}", prod): koxy_tok[tk.lower()].append(lt)
M_kox = cobra.io.read_sbml_model(str(MODELS/"iYL1228.xml.gz"))
kpn_to_koxy = {}
for g in M_kox.genes:
    gn=(g.name or "").strip()
    if gn and g.id!=gn:
        mm=koxy_tok.get(gn.lower(), [])
        if len(mm)==1: kpn_to_koxy[g.id]=mm[0]

# og -> [fba calls], one per modeled organism
og_n = defaultdict(int); og_e = defaultdict(int)
def add(modeled_org_lookup, fba_dict):
    for model_gene, ess in fba_dict.items():
        og = modeled_org_lookup(model_gene)
        if og:
            og_n[og]+=1; og_e[og]+=int(ess)
add(lambda g: og_of.get(("beril_Putida", g)), fba["putida"])
add(lambda g: og_of.get(("beril_Keio", b_to_keio.get(g,""))), fba["keio"])
add(lambda g: mtub_og.get(g), fba["mtub"])
add(lambda g: og_of.get(("beril_Koxy", kpn_to_koxy.get(g,""))), fba["koxy"])
og_fba_rate = {og: og_e[og]/og_n[og] for og in og_n}
print(f"\nStep 2: og_fba_rate built for {len(og_fba_rate)} orthogroups "
      f"(from 4 modeled organisms)", flush=True)

# ---- Step 3+4: validate on UNMODELED organisms ----
MODELED = {"beril_Putida","beril_Keio","ecoli_BW25113_tradis","mtub","beril_Koxy",
           "beril_WCS417","beril_PS","beril_psRCH2","beril_SyringaeB728a",
           "beril_SyringaeB728a_mexBdelta","beril_pseudo1_N1B4","beril_pseudo3_N2E3",
           "beril_pseudo5_N2C3_1","beril_pseudo6_N2E2","beril_pseudo13_GW456_L13"}
# conservation rate (leak-clean per clade) reused for the W2 baseline
def cons_rate(held):
    n=defaultdict(int); e=defaultdict(int)
    for r in ALL:
        if clade_of.get(r["organism"])==held: continue
        og=og_of.get((r["organism"],r["locus_tag"]))
        if og: n[og]+=1; e[og]+=int(r["essential"])
    return {og:e[og]/n[og] for og in n if n[og]>=5}

truth_by_org = defaultdict(dict)
for r in ALL: truth_by_org[r["organism"]][r["locus_tag"]]=int(r["essential"])

def auc(scores, ys):
    s=np.array(scores); y=np.array(ys)
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); pos=y==1
    if pos.sum()==0 or (~pos).sum()==0: return 0.5
    return float((r[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*(~pos).sum()))

unmodeled = [o for o in truth_by_org if o not in MODELED and sum(truth_by_org[o].values())>=30]
print(f"\nStep 4: validating on {len(unmodeled)} UNMODELED organisms", flush=True)
rows=[]
for org in unmodeled:
    truth = truth_by_org[org]
    held = clade_of.get(org,"?")
    rate = cons_rate(held)
    keys=[lt for lt in truth if og_of.get((org,lt)) in og_fba_rate]
    if len(keys)<30: continue
    y=[truth[lt] for lt in keys]
    fba_score=[og_fba_rate[og_of[(org,lt)]] for lt in keys]
    cons_score=[rate.get(og_of[(org,lt)], 0.0) for lt in keys]
    # combined: max of the two priors
    comb=[max(a,b) for a,b in zip(fba_score,cons_score)]
    rows.append(dict(organism=org, n=len(keys), ess=int(sum(y)),
                     fba_transfer_auc=round(auc(fba_score,y),3),
                     conservation_auc=round(auc(cons_score,y),3),
                     combined_auc=round(auc(comb,y),3)))

rows.sort(key=lambda r:-r["n"])
print(f"\n{'organism':<26}{'n':>6}{'ess':>6}{'FBA_xfer':>9}{'cons':>7}{'comb':>7}{'FBA-cons':>9}")
for r in rows:
    d=r["fba_transfer_auc"]-r["conservation_auc"]
    print(f"  {r['organism']:<24}{r['n']:>6}{r['ess']:>6}{r['fba_transfer_auc']:>9.3f}"
          f"{r['conservation_auc']:>7.3f}{r['combined_auc']:>7.3f}{d:>+9.3f}")
mean_fba=np.mean([r["fba_transfer_auc"] for r in rows])
mean_cons=np.mean([r["conservation_auc"] for r in rows])
mean_comb=np.mean([r["combined_auc"] for r in rows])
print(f"\n  MEAN over {len(rows)} unmodeled orgs: "
      f"FBA-transfer={mean_fba:.3f}  conservation={mean_cons:.3f}  combined={mean_comb:.3f}")
print(f"  combined lift over conservation: {(mean_comb-mean_cons)*100:+.1f}pp")

json.dump(dict(n_modeled=len(MODELED), n_og_with_fba=len(og_fba_rate),
               n_unmodeled_validated=len(rows),
               mean_fba_transfer_auc=round(float(mean_fba),3),
               mean_conservation_auc=round(float(mean_cons),3),
               mean_combined_auc=round(float(mean_comb),3),
               per_org=rows), open(OUT/"metabolic_transfer.json","w"), indent=2)
print(f"\nwrote outputs/orphan/metabolic_transfer.json", flush=True)
