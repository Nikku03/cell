"""Integrated E. coli cell SIMULATOR: the laid-out cell as a runnable network.
Perturb (knock out / mutate genes, set condition) -> predict the outcome by
combining the layers we built:
  METABOLIC  : FBA on iJO1366 (viability/growth, per medium)
  REGULATORY : RegulonDB network propagation (KO a TF -> targets lose regulation)
  CONDITIONAL: medium sets which genes are needed
Validated against Keio single-gene-deletion truth.
"""
import csv, json, warnings
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models")
m=cobra.io.read_sbml_model(str(MODELS/"iJO1366.xml.gz"))
model_b={g.id for g in m.genes}
CARBON={"glucose":"EX_glc__D_e","arabinose":"EX_arab__L_e","glycerol":"EX_glyc_e","maltose":"EX_malt_e","succinate":"EX_succ_e","acetate":"EX_ac_e"}

# regulatory network (gene name space) + name<->b
name2b={}; b2name={}
for r in csv.DictReader(open(REG/"TRN.csv")):
    if r.get("gene_name") and r.get("gene_id"): name2b[r["gene_name"].lower()]=r["gene_id"]; b2name[r["gene_id"]]=r["gene_name"].lower()
edges=defaultdict(list)            # tf -> [(target, sign)]
SIGN={"activator":1,"repressor":-1}
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); edges[p[0].lower()].append((p[1].lower(),SIGN.get(p[2].lower(),0)))
# Keio truth
b_of={r["locus_tag"]:r["b_number"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
ess_b={}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"]=="beril_Keio":
        b=b_of.get(r["locus_tag"])
        if b: ess_b[b]=int(r["essential"])

def set_medium(mm,carbon):
    med=mm.medium.copy()
    for cs,ex in CARBON.items():
        if ex in med or ex in {r.id for r in mm.reactions}: med[ex]= (10.0 if cs==carbon else 0.0)
    mm.medium=med

def simulate(knockouts=(), carbon="glucose", mutate=None):
    """knockouts: gene names or b-numbers; mutate: {gene: fraction in [0,1]} reduces flux capacity.
    returns viability, growth, and regulatory downstream effects."""
    kos_b=set()
    for g in knockouts:
        b=g if g in model_b else name2b.get(g.lower())
        if b: kos_b.add(b)
    with m as mm:
        set_medium(mm,carbon)
        wt=mm.slim_optimize()
        for b in kos_b:
            try: mm.genes.get_by_id(b).knock_out()
            except KeyError: pass
        if mutate:
            for g,frac in mutate.items():
                b=g if g in model_b else name2b.get(g.lower())
                if b and b in model_b:
                    for rx in mm.genes.get_by_id(b).reactions:
                        rx.lower_bound*=frac; rx.upper_bound*=frac
        gr=mm.slim_optimize(); gr=float(gr) if gr and gr==gr else 0.0
    # regulatory propagation: KO of a TF -> direct targets lose their regulator
    reg_eff=[]
    for g in knockouts:
        gl=g.lower() if isinstance(g,str) else g
        if gl in edges:
            tgts=edges[gl]
            reg_eff.append(dict(tf=gl, n_targets=len(tgts),
                up=[t for t,s in tgts if s==1][:8], down=[t for t,s in tgts if s==-1][:8]))
    return dict(carbon=carbon, knockouts=list(knockouts), wt_growth=round(wt,3),
                growth=round(gr,3), viable=gr>0.01*wt, regulatory_effects=reg_eff)

# ============ VALIDATION: single-gene KO vs Keio truth (glucose) ============
print("VALIDATION: predict viability of every single-gene KO vs Keio truth (glucose minimal)\n")
with m as mm:
    set_medium(mm,"glucose"); wt=mm.slim_optimize()
    df=single_gene_deletion(mm,processes=1)
pred_lethal=set()
for _,row in df.iterrows():
    ids=row["ids"]; b=list(ids)[0] if isinstance(ids,(set,frozenset)) else ids
    gr=row["growth"]
    if gr is None or (isinstance(gr,float) and np.isnan(gr)) or gr<0.01*wt: pred_lethal.add(b)
uni=[b for b in model_b if b in ess_b]
truth=set(b for b in uni if ess_b[b]==1)
P=pred_lethal&set(uni); tp=len(P&truth); fp=len(P-truth); fn=len(truth-P)
prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); acc=sum(1 for b in uni if (b in pred_lethal)==(b in truth))/len(uni)
print(f"  genes scored: {len(uni)}  truth-essential: {len(truth)}")
print(f"  simulator lethal-KO prediction: precision={prec:.3f} recall={rec:.3f} accuracy={acc:.3f}")
print(f"  (this is the iJO1366 metabolic ceiling; regulatory KOs add reach below)")

# ============ WORKED PERTURBATIONS (the 'what happens if' interface) ============
print("\n=== worked perturbations ===")
def show(r):
    v="VIABLE" if r["viable"] else "*** LETHAL ***"
    s=f"  KO {r['knockouts']} on {r['carbon']}: growth {r['growth']}/{r['wt_growth']}  -> {v}"
    for e in r["regulatory_effects"]:
        s+=f"\n      [reg] {e['tf']} KO -> {e['n_targets']} targets lose regulation; e.g. activated:{e['up'][:5]} repressed-now-derepressed:{e['down'][:5]}"
    return s
print(show(simulate(["pyrB"])))                       # essential metabolic
print(show(simulate(["sdhA"],carbon="glucose")))       # dispensable on glucose
print(show(simulate(["sdhA"],carbon="succinate")))     # essential on succinate (conditional!)
print(show(simulate(["malQ"],carbon="maltose")))       # conditional on maltose
print(show(simulate(["crp"])))                         # global TF: downstream cascade
print(show(simulate(["fnr"])))                         # essential global TF
print(show(simulate([],carbon="glucose",mutate={"pgi":0.0})))  # 'mutation': null pgi
print(show(simulate(["aceA","aceB"],carbon="acetate")))# double KO on acetate (glyoxylate shunt)

json.dump(dict(validation=dict(n=len(uni),precision=round(prec,3),recall=round(rec,3),accuracy=round(acc,3)),
               note="metabolic viability = FBA(iJO1366); regulatory KO = network propagation; conditional = medium"),
          open(OUT/"cell_simulator.json","w"),indent=2)
print(f"\nwrote {OUT}/cell_simulator.json")
