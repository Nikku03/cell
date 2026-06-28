"""KINETIC Wheel 2 — add active-site/kinetics to the metabolic network.

The idea you wanted: an enzyme's active site fixes WHICH reaction it runs and
WHICH metabolites it touches (its partners in the graph). Its kinetics (kcat)
fix HOW MUCH flux it can carry. So a knockout that forces flux onto a SLOW
bypass should be infeasible even when a bypass exists topologically -> the gene
is kinetically essential. Plain FBA/topology misses these (it treats any bypass
as free). This is enzyme-constrained FBA (sMOMENT/GECKO), realized on Wheel 2.

Method (iJO1366, E. coli, validated vs Keio truth via b-number bridge):
  - each gene-associated reaction draws on a SHARED proteome pool:
        sum_j (v_fwd_j + v_rev_j) / kcat_j  <=  P
    so high flux through a LOW-kcat enzyme is expensive.
  - kcat assigned by EC class (coarse literature medians, Bar-Even 2011) — the
    ONLY transferable part of kinetics; we do NOT claim per-enzyme exact rates.
    The thing we validate is essentiality (Keio truth), not the rates.
  - calibrate P to the tightest budget that still allows ~WT growth, then run
    single-gene-deletion WITH vs WITHOUT the pool constraint.

Output: which genes become essential ONLY when kinetics is on (kinetic
bottlenecks), and whether that improves precision/recall vs plain FBA.
"""
import csv, json, warnings, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
from optlang.symbolics import Zero
warnings.filterwarnings("ignore")
cobra.Configuration().solver = "glpk"
OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models")
t0=time.time()

# EC-class median kcat (1/s), coarse literature priors (Bar-Even 2011 medians).
# Used ONLY as order-of-magnitude capacity priors, clearly not per-enzyme truth.
EC_KCAT={"1":13.0,"2":25.0,"3":50.0,"4":15.0,"5":13.0,"6":8.0}
KCAT_DEFAULT=14.0
def rxn_kcat(r):
    a=r.annotation.get("ec-code")
    if a:
        ec=a[0] if isinstance(a,list) else a
        c=ec.split(".")[0]
        if c in EC_KCAT: return EC_KCAT[c]
    return KCAT_DEFAULT

print(f"[{time.time()-t0:.0f}s] load iJO1366",flush=True)
m=cobra.io.read_sbml_model(str(MODELS/"iJO1366.xml.gz"))
wt_uncon=m.slim_optimize()
print(f"  unconstrained WT growth = {wt_uncon:.4f}")

# ---- truth via Keio bridge ----
b_of={}  # keio locus -> b
for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv")):
    b_of[r["locus_tag"]]=r["b_number"]
ess_b={}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"]!="beril_Keio": continue
    b=b_of.get(r["locus_tag"])
    if b: ess_b[b]=int(r["essential"])
model_b={g.id for g in m.genes}
universe=sorted(model_b & set(ess_b))   # genes we can both model and score
print(f"  eval universe (model gene & Keio truth): {len(universe)}  essential={sum(ess_b[b] for b in universe)}")

def build_pool(model,P):
    """add shared proteome-pool constraint sum (fwd+rev)/(kcat*3600) <= P."""
    cons=model.problem.Constraint(Zero,name="prot_pool",ub=P)
    model.add_cons_vars([cons]); model.solver.update()
    coeffs={}
    for r in model.reactions:
        if not r.genes: continue
        if r.id.startswith("EX_") or r.id.startswith("DM_") or r.id.startswith("BIOMASS"): continue
        c=1.0/(rxn_kcat(r)*3600.0)
        coeffs[r.forward_variable]=c; coeffs[r.reverse_variable]=c
    cons.set_linear_coefficients(coeffs)
    model.solver.update()
    return cons

# ---- calibrate P: tightest budget keeping WT >= 0.99*max ----
print(f"[{time.time()-t0:.0f}s] calibrate proteome budget P",flush=True)
def wt_growth_at(P):
    with m as mm:
        build_pool(mm,P); return mm.slim_optimize() or 0.0
# find scale: compute WT pool usage at unconstrained optimum
with m as mm:
    cons=build_pool(mm,1e9); sol=mm.optimize()
    used=sum((r.forward_variable.primal+r.reverse_variable.primal)/(rxn_kcat(r)*3600.0)
             for r in mm.reactions if r.genes and not (r.id.startswith(("EX_","DM_","BIOMASS"))))
print(f"  pool usage at free optimum ~ {used:.3e}")
lo,hi=used*0.3,used*1.0; target=0.99*wt_uncon
for _ in range(18):
    mid=(lo+hi)/2; g=wt_growth_at(mid)
    if g>=target: hi=mid
    else: lo=mid
P=hi; print(f"  chosen P={P:.3e}  WT growth at P = {wt_growth_at(P):.4f} ({wt_growth_at(P)/wt_uncon:.1%} of max)")

# ---- plain FBA single-gene-deletion ----
print(f"[{time.time()-t0:.0f}s] plain FBA single-gene-deletion",flush=True)
plain=single_gene_deletion(m,gene_list=[m.genes.get_by_id(b) for b in universe],processes=1)
def parse(df):
    out={}
    for _,row in df.iterrows():
        ids=row["ids"]; gid=list(ids)[0] if isinstance(ids,(set,frozenset)) else ids
        out[gid]=0.0 if row["growth"] is None or np.isnan(row["growth"]) else row["growth"]
    return out
plain_g=parse(plain); wt_plain=m.slim_optimize()

# ---- enzyme-constrained single-gene-deletion ----
print(f"[{time.time()-t0:.0f}s] enzyme-constrained single-gene-deletion",flush=True)
with m as mm:
    build_pool(mm,P); wt_ec=mm.slim_optimize()
    ec=single_gene_deletion(mm,gene_list=[mm.genes.get_by_id(b) for b in universe],processes=1)
ec_g=parse(ec)
print(f"  WT growth plain={wt_plain:.4f}  ec={wt_ec:.4f}")

# ---- essential calls: growth < 1% of WT ----
def calls(gmap,wt): return {b for b in universe if gmap.get(b,0.0) < 0.01*wt}
plain_ess=calls(plain_g,wt_plain); ec_ess=calls(ec_g,wt_ec)
truth_ess={b for b in universe if ess_b[b]==1}

def prf(pred):
    tp=len(pred&truth_ess); fp=len(pred-truth_ess); fn=len(truth_ess-pred)
    p=tp/max(tp+fp,1); r=tp/max(tp+fn,1); f=2*p*r/max(p+r,1e-9)
    return dict(n_pred=len(pred),tp=tp,fp=fp,precision=round(p,3),recall=round(r,3),f1=round(f,3))
P_plain=prf(plain_ess); P_ec=prf(ec_ess)

# genes essential ONLY with kinetics on
kin_only=ec_ess-plain_ess
kin_only_true=kin_only&truth_ess
print(f"\n=== essentiality vs Keio truth (universe n={len(universe)}, {len(truth_ess)} essential) ===")
print(f"  plain FBA : {P_plain}")
print(f"  kinetic   : {P_ec}")
print(f"\n  genes essential ONLY when kinetics ON: {len(kin_only)}  of which TRUE-essential: {len(kin_only_true)}  (precision {len(kin_only_true)/max(len(kin_only),1):.2f})")

# worked examples: kinetic-only true essentials, with their reaction/EC/kcat
examples=[]
for b in list(kin_only_true)[:8]:
    g=m.genes.get_by_id(b)
    rs=[(r.id,r.annotation.get("ec-code"),rxn_kcat(r)) for r in g.reactions][:3]
    examples.append(dict(b=b,name=g.name,growth_plain=round(plain_g.get(b,0),3),
                         growth_kinetic=round(ec_g.get(b,0),3),reactions=rs))
print("\n=== worked kinetic-bottleneck essentials (topology bypassable, kinetically not) ===")
for e in examples:
    print(f"  {e['b']} {e['name']:<14} plain_growth={e['growth_plain']:.3f} -> kinetic={e['growth_kinetic']:.3f}  {e['reactions'][:2]}")

json.dump(dict(universe=len(universe),n_truth_ess=len(truth_ess),
               wt_uncon=round(wt_uncon,4),P=P,wt_plain=round(wt_plain,4),wt_ec=round(wt_ec,4),
               plain=P_plain,kinetic=P_ec,
               kin_only=len(kin_only),kin_only_true=len(kin_only_true),
               kin_only_precision=round(len(kin_only_true)/max(len(kin_only),1),3),
               examples=examples,ec_kcat=EC_KCAT,kcat_default=KCAT_DEFAULT),
          open(OUT/"kinetic_wheel2.json","w"),indent=2)
print(f"\n[{time.time()-t0:.0f}s] wrote {OUT}/kinetic_wheel2.json")
