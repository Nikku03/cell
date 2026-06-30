"""FULL STACK: fuse every method into one integrated E. coli cell model, measure
how far it gets, and export all data for the interactive HTML app.
Signals fused for essentiality: FBA metabolic necessity (per medium) + essential
core (informational machinery) + conservation (feba ortholog breadth) + feba
fitness. Plus: regulatory network, conditional KO outcomes, SOS dynamics params.
"""
import csv, json, warnings, sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models")
m=cobra.io.read_sbml_model(str(MODELS/"iJO1366.xml.gz")); model_b={g.id for g in m.genes}
CARBON={"glucose":"EX_glc__D_e","arabinose":"EX_arab__L_e","glycerol":"EX_glyc_e","maltose":"EX_malt_e","succinate":"EX_succ_e","acetate":"EX_ac_e"}
exset={r.id for r in m.reactions}
name2b={}; prod={}
for r in csv.DictReader(open(REG/"TRN.csv")):
    if r.get("gene_name") and r.get("gene_id"): name2b[r["gene_name"].lower()]=r["gene_id"]
b2name={b:n for n,b in name2b.items()}
for l in open(REG/"GeneProductSet.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=3: prod[p[1].lower()]=p[2]
# regulatory net
activators=defaultdict(list); edges_sign={}
tf_targets=defaultdict(list); is_tf=set()
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); tf,tg,s=p[0].lower(),p[1].lower(),p[2].lower(); is_tf.add(tf)
    tf_targets[tf].append([tg, 1 if s=="activator" else (-1 if s=="repressor" else 0)])
    if s=="activator": activators[tg].append(tf)
auto={}
for tf in tf_targets:
    for tg,s in tf_targets[tf]:
        if tg==tf: auto[tf]=s
# Keio truth
b_of={r["locus_tag"]:r["b_number"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
ess_b={}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"]=="beril_Keio":
        b=b_of.get(r["locus_tag"])
        if b: ess_b[b]=int(r["essential"])
# essential core (informational)
KW=["ribosomal protein","rna polymerase","dna polymerase","trna ligase","aminoacyl","dna gyrase","dna topoisomerase","primase","replicative dna helicase","elongation factor","initiation factor","peptide chain release","rrna","ribosome","dna primase","cell division protein fts"]
PFX=("rpl","rps","rpm","rpo","tuf","fus","frr","infa","infb","infc","prf","dnaa","dnab","dnae","dnag","dnan","dnax","gyra","gyrb","nusa","nusg","lig")
core_names=set(g for g,p in prod.items() if any(k in p.lower() for k in KW) or g.startswith(PFX))
core_b=set(name2b[g] for g in core_names if g in name2b)
# conservation + fitness from feba Keio
con=sqlite3.connect("data/external_data/feba/feba.db"); cur=con.cursor()
b2kl={sn:l for l,sn in cur.execute("SELECT locusId,sysName FROM Gene WHERE orgId='Keio' AND sysName LIKE 'b%'")}
north=defaultdict(set)
for o2,l1 in cur.execute("SELECT orgId2,locusId1 FROM Ortholog WHERE orgId1='Keio'"): north[l1].add(o2)
P=cur.execute("SELECT COUNT(DISTINCT orgId2) FROM Ortholog WHERE orgId1='Keio'").fetchone()[0]
minfit={}
for lid,fit in cur.execute("SELECT locusId,fit FROM GeneFitness WHERE orgId='Keio'"):
    if fit is None: continue
    if lid not in minfit or fit<minfit[lid]: minfit[lid]=fit
def conservation(b):
    kl=b2kl.get(b); return len(north.get(kl,()))/P if kl else 0.0

# ---- KO x medium FBA ----
print("running KO x medium FBA ...",flush=True)
def set_medium(mm,carbon):
    med=mm.medium.copy()
    for cs,ex in CARBON.items():
        if ex in exset: med[ex]=(10.0 if cs==carbon else 0.0)
    mm.medium=med
ko_growth=defaultdict(dict); wt_growth={}
for carbon in CARBON:
    with m as mm:
        set_medium(mm,carbon); wt=mm.slim_optimize(); wt_growth[carbon]=round(float(wt),3)
        if not wt or wt<1e-6: continue
        df=single_gene_deletion(mm,processes=1)
        for _,row in df.iterrows():
            ids=row["ids"]; b=list(ids)[0] if isinstance(ids,(set,frozenset)) else ids
            gr=row["growth"]; g=0.0 if gr is None or (isinstance(gr,float) and np.isnan(gr)) else float(gr)
            ko_growth[b][carbon]=round(g/wt,3)
    print(f"  {carbon}: wt={wt_growth[carbon]}",flush=True)

# ---- fuse essentiality signals + measure ----
fba_core=set(b for b in model_b if all(ko_growth.get(b,{}).get(c,1)<0.01 for c in CARBON if wt_growth.get(c,0)>1e-6))
def cons_ess(b): return conservation(b)>=0.85
def fit_ess(b):
    kl=b2kl.get(b); return (kl is not None) and (kl not in minfit or minfit.get(kl,0)< -3)
uni=list(ess_b); truth=set(b for b in uni if ess_b[b]==1)
def metr(pred):
    Pp=pred&set(uni); tp=len(Pp&truth); fp=len(Pp-truth); fn=len(truth-Pp)
    return dict(precision=round(tp/max(tp+fp,1),3),recall=round(tp/max(tp+fn,1),3),
               accuracy=round(sum(1 for b in uni if (b in pred)==(b in truth))/len(uni),3),n_pred=len(Pp))
S_fba=fba_core&set(uni); S_core=core_b&set(uni)
S_cons=set(b for b in uni if cons_ess(b)); S_fit=set(b for b in uni if fit_ess(b))
fused=S_fba|S_core|S_cons
print("\n=== FULL-STACK essentiality (vs Keio, full genome) ===")
print(f"  FBA metabolic        : {metr(S_fba)}")
print(f"  essential core       : {metr(S_core)}")
print(f"  conservation>=85%    : {metr(S_cons)}")
print(f"  feba fitness         : {metr(S_fit)}")
print(f"  FUSED (FBA|core|cons): {metr(fused)}")
print(f"  FUSED + fitness      : {metr(fused|S_fit)}")

# ---- per-gene records for the app ----
allnames=set(prod)|set(name2b)|set(tf_targets)
genes=[]
for g in sorted(allnames):
    b=name2b.get(g,"")
    rs=len(tf_targets.get(g,[]))
    genes.append(dict(name=g,b=b,product=(prod.get(g,"") or "")[:60],
        essential=ess_b.get(b,None),
        pred_essential=int(b in fused) if b else 0,
        metabolic=int(b in model_b),
        conservation=round(conservation(b),2) if b else 0,
        is_tf=int(g in tf_targets), regulon=rs,
        tf_class=("global" if rs>=80 else "specific" if rs>=12 else ("minor" if rs>0 else "")),
        autoreg=({1:"pos",-1:"neg"}.get(auto.get(g),"") if g in tf_targets else ""),
        core=int(b in core_b)))
# KO lookup (metabolic genes only have FBA; others viable unless core)
ko=dict()
for b in model_b: ko[b]=ko_growth.get(b,{})
# regulatory adjacency (cap targets per TF for size)
net={tf:[[t,s] for t,s in tf_targets[tf][:60]] for tf in tf_targets}
# SOS dynamics params for JS
sos=dict(genes=["lexA","recA","sulA","uvrA","dinB","umuD"],t_damage=120,t_repair=260,T=400)
data=dict(organism="E. coli K-12 MG1655",
          metrics=dict(fba=metr(S_fba),core=metr(S_core),conservation=metr(S_cons),
                       fitness=metr(S_fit),fused=metr(fused),fused_fit=metr(fused|S_fit)),
          wt_growth=wt_growth, carbons=list(CARBON),
          n_genes=len(genes), genes=genes, ko=ko, net=net,
          core_b=sorted(core_b), crp_targets=[t for t,s in tf_targets.get("crp",[])][:400],
          sos=sos)
json.dump(data,open(OUT/"cell_app_data.json","w"))
print(f"\nwrote {OUT}/cell_app_data.json  ({len(genes)} genes, {len(net)} TFs)")
