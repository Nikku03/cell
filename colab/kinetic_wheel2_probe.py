"""Probe WHY kinetics didn't change essentiality, properly:
  (1) continuous AUC of knockout growth-drop vs Keio truth: plain vs kinetic
      (does kinetics re-RANK genes even if it doesn't flip binary calls?)
  (2) budget sweep P in {0.99,0.97,0.95,0.90,0.80}*WT: do kinetic-only
      essentials appear, and are they TRUE (precision)?
"""
import csv, json, warnings, time
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
from optlang.symbolics import Zero
warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"
OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models"); t0=time.time()
EC_KCAT={"1":13.0,"2":25.0,"3":50.0,"4":15.0,"5":13.0,"6":8.0}; KD=14.0
def kc(r):
    a=r.annotation.get("ec-code")
    if a:
        ec=a[0] if isinstance(a,list) else a; c=ec.split(".")[0]
        if c in EC_KCAT: return EC_KCAT[c]
    return KD
m=cobra.io.read_sbml_model(str(MODELS/"iJO1366.xml.gz")); wt0=m.slim_optimize()
b_of={r["locus_tag"]:r["b_number"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
ess_b={}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"]=="beril_Keio":
        b=b_of.get(r["locus_tag"]);
        if b: ess_b[b]=int(r["essential"])
universe=sorted({g.id for g in m.genes}&set(ess_b))
y=np.array([ess_b[b] for b in universe]);
glist=[m.genes.get_by_id(b) for b in universe]
def pool(model,P):
    c=model.problem.Constraint(Zero,name="pp",ub=P); model.add_cons_vars([c]); model.solver.update()
    co={}
    for r in model.reactions:
        if r.genes and not r.id.startswith(("EX_","DM_","BIOMASS")):
            x=1.0/(kc(r)*3600.0); co[r.forward_variable]=x; co[r.reverse_variable]=x
    c.set_linear_coefficients(co); model.solver.update(); return c
def parse(df):
    out={}
    for _,row in df.iterrows():
        i=row["ids"]; g=list(i)[0] if isinstance(i,(set,frozenset)) else i
        out[g]=0.0 if row["growth"] is None or np.isnan(row["growth"]) else row["growth"]
    return out
def auc(s,yy):
    s=np.asarray(s,float)
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); mm=yy==1; pos=mm.sum()
    return float((r[mm].sum()-pos*(pos-1)/2)/(pos*(~mm).sum()))

# free pool usage
with m as mm:
    pool(mm,1e9); mm.optimize()
    used=sum((r.forward_variable.primal+r.reverse_variable.primal)/(kc(r)*3600.0)
             for r in mm.reactions if r.genes and not r.id.startswith(("EX_","DM_","BIOMASS")))

# plain deletion -> continuous score = wt - growth
plain=parse(single_gene_deletion(m,gene_list=glist,processes=1)); wtp=m.slim_optimize()
score_plain=np.array([wtp-plain.get(b,0) for b in universe])
print(f"plain FBA: continuous AUC(growth-drop vs truth) = {auc(score_plain,y):.3f}")

def budget_for(frac):
    lo,hi=used*0.2,used*1.0; tgt=frac*wt0
    for _ in range(20):
        mid=(lo+hi)/2
        with m as mm:
            pool(mm,mid); g=mm.slim_optimize() or 0
        if g>=tgt: hi=mid
        else: lo=mid
    return hi

truth=set(b for b in universe if ess_b[b]==1)
plain_ess=set(b for b in universe if plain.get(b,0)<0.01*wtp)
rows=[]
print(f"\n{'WTfrac':>7}{'P':>11}{'wt_ec':>8}{'AUC':>7}{'ess':>6}{'kin_only':>9}{'kin_true':>9}{'kin_prec':>9}")
for frac in [0.99,0.97,0.95,0.90,0.80]:
    P=budget_for(frac)
    with m as mm:
        pool(mm,P); wtec=mm.slim_optimize()
        ded=parse(single_gene_deletion(mm,gene_list=glist,processes=1))
    sc=np.array([wtec-ded.get(b,0) for b in universe])
    a=auc(sc,y)
    ec_ess=set(b for b in universe if ded.get(b,0)<0.01*wtec)
    ko=ec_ess-plain_ess; kot=ko&truth
    rows.append(dict(frac=frac,P=P,wt_ec=round(wtec,4),auc=round(a,3),ess=len(ec_ess),
                     kin_only=len(ko),kin_only_true=len(kot),
                     kin_prec=round(len(kot)/max(len(ko),1),3)))
    print(f"{frac:>7.2f}{P:>11.3e}{wtec:>8.4f}{a:>7.3f}{len(ec_ess):>6}{len(ko):>9}{len(kot):>9}{len(kot)/max(len(ko),1):>9.2f}")

json.dump(dict(plain_auc=round(auc(score_plain,y),3),free_pool=used,wt=wt0,sweep=rows),
          open(OUT/"kinetic_wheel2_probe.json","w"),indent=2)
print(f"\n[{time.time()-t0:.0f}s] wrote {OUT}/kinetic_wheel2_probe.json")
