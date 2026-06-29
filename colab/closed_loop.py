"""CLOSE THE LOOP: environment -> effector -> TF activity -> expression -> flux.

Textbook circuit: carbon source -> cAMP/CRP -> catabolic operons -> growth.
  - glucose present  -> cAMP low  -> CRP inactive -> catabolic operons OFF
  - glucose absent   -> cAMP high -> CRP active   -> catabolic operon ON *if* its
                        specific inducer (the carbon source) is present.
Operon ON/OFF gates the metabolic genes: OFF genes are knocked out in iJO1366,
then FBA gives growth + conditional essentiality.

Key test: REGULATION-AWARE vs REGULATION-BLIND FBA. Pure FBA says "the network
*can* use arabinose"; the closed loop says "only if CRP turned the genes on."
That difference is the loop.

Honest labels: topology/sign + catabolite-repression logic are textbook/RegulonDB;
rate params illustrative; FBA is iJO1366.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cobra, warnings; warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"
OUT=Path("outputs/orphan")
rng=np.random.default_rng(1)
m=cobra.io.read_sbml_model("data/external_data/bigg_models/iJO1366.xml.gz")

CARBON={"glucose":"EX_glc__D_e","arabinose":"EX_arab__L_e","glycerol":"EX_glyc_e","maltose":"EX_malt_e"}
# operon -> (metabolic gene b-numbers, inducer carbon)
OPERONS={"araBAD":(["b0062","b0063"],"arabinose"),
         "glpFK" :(["b3926","b2241","b3927"],"glycerol"),
         "malPQ" :(["b3416","b3417"],"maltose")}
GENENAME={"b0062":"araA","b0063":"araB","b3926":"glpK","b2241":"glpA","b3927":"glpF","b3416":"malQ","b3417":"malP"}

def regulatory_state(env, crp_override=None):
    """env: set of carbon names present. returns crp_active, {operon:on}, off_genes."""
    crp_active = ("glucose" not in env) if crp_override is None else crp_override
    onoff={}; off=[]
    for op,(genes,inducer) in OPERONS.items():
        on = crp_active and (inducer in env)
        onoff[op]=on
        if not on: off+=genes
    return crp_active,onoff,off

def grow(env, regulated=True, crp_override=None, ko=()):
    with m:
        med=m.medium.copy()
        med["EX_glc__D_e"]=10.0 if "glucose" in env else 0.0
        for cs,ex in CARBON.items():
            if cs=="glucose": continue
            med[ex]=10.0 if cs in env else 0.0
        m.medium=med
        off=[]
        if regulated:
            _,_,off=regulatory_state(env,crp_override)
        for g in set(off)|set(ko):
            try: m.genes.get_by_id(g).knock_out()
            except KeyError: pass
        v=m.slim_optimize()
        return float(v) if v and v==v else 0.0

# ---------- 1. growth table: regulated vs blind ----------
envs=[{"glucose"},{"arabinose"},{"glycerol"},{"maltose"},{"glucose","arabinose"}]
print(f"{'environment':<22}{'CRP':>5}{'operonsON':>22}{'growth(reg)':>12}{'growth(blind)':>14}")
table=[]
for env in envs:
    crp,onoff,off=regulatory_state(env)
    on=[k for k,v in onoff.items() if v]
    gr=grow(env,regulated=True); gb=grow(env,regulated=False)
    table.append(dict(env="+".join(sorted(env)),crp=crp,on=on,growth_reg=round(gr,3),growth_blind=round(gb,3)))
    print(f"  {'+'.join(sorted(env)):<20}{str(crp):>5}{','.join(on) or '-':>22}{gr:>12.3f}{gb:>14.3f}")

# ---------- 2. the decisive loop case: arabinose, CRP forced inactive ----------
g_on  = grow({"arabinose"}, regulated=True)                    # CRP active (no glucose) -> araBAD ON
g_off = grow({"arabinose"}, regulated=True, crp_override=False)# CRP inactive (crp mutant / repression) -> OFF
g_fba = grow({"arabinose"}, regulated=False)                   # regulation-blind FBA
print(f"\nDECISIVE: sole carbon = arabinose")
print(f"  CRP active (loop)      growth = {g_on:.3f}")
print(f"  CRP inactive (loop)    growth = {g_off:.3f}   <- metabolically capable but regulation blocks")
print(f"  regulation-blind FBA   growth = {g_fba:.3f}   <- pure FBA misses the block")

# ---------- 3. conditional essentiality of catabolic genes ----------
print(f"\nconditional essentiality (deletion -> growth lost?):")
ess={}
for env in [{"glucose"},{"arabinose"},{"glycerol"},{"maltose"}]:
    base=grow(env,regulated=True)
    row={}
    for op,(genes,ind) in OPERONS.items():
        for g in genes:
            if base>1e-4:
                gd=grow(env,regulated=True,ko=(g,)); row[GENENAME[g]]= "ESS" if gd<0.01*base else "ne"
            else: row[GENENAME[g]]="(nogrow)"
    ess["+".join(sorted(env))]=row
    print(f"  {('+'.join(sorted(env))):<12} "+"  ".join(f"{k}:{v}" for k,v in row.items()))

# ---------- 4. dynamic trace: shift to arabinose turns araBAD on (Gillespie) ----------
def hill_act(x,K,n): return x**n/(K**n+x**n)
def sim_arabad(T=300, shift=80):
    # araB protein; activated by CRP active-conc; before shift glucose(CRP off), after arabinose(CRP on)
    dm=np.log(2)/5; dp=np.log(2)/40; ktl=5.0; beta=30.0; K=40.0; n=2; leak=0.03
    CRP=120.0
    mR=0.0; pR=0.0; t=0.0; ts=[]; ps=[]; nr=0.0
    while t<T:
        phi=0.08 if t<shift else 1.0          # CRP active fraction: low (glucose) -> high (arabinose)
        rif=leak+(1-leak)*hill_act(CRP*phi,K,n)
        a=np.array([beta*rif, ktl*mR, dm*mR, dp*pR]); a0=a.sum()
        tau=rng.exponential(1/a0); t+=tau
        while nr<=t and nr<=T: ts.append(nr); ps.append(pR); nr+=1.0
        r=rng.random()*a0; c=np.cumsum(a); k=np.searchsorted(c,r)
        if k==0: mR+=1
        elif k==1: pR+=1
        elif k==2: mR=max(0,mR-1)
        else: pR=max(0,pR-1)
    return np.array(ts),np.array(ps)
tA,pA=sim_arabad()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
ax1.plot(tA,pA,color="#2ca02c",lw=1.8); ax1.axvline(80,color="gray",ls="--")
ax1.text(84,ax1.get_ylim()[1]*0.5,"shift glucose->arabinose\n(CRP activates)",fontsize=9)
ax1.set_title("Loop step 1-3: carbon shift -> CRP active -> araBAD induced (Gillespie)")
ax1.set_xlabel("time (min)"); ax1.set_ylabel("araBAD protein (molecules)")
labels=[t["env"] for t in table]; gr=[t["growth_reg"] for t in table]; gb=[t["growth_blind"] for t in table]
x=np.arange(len(labels)); w=0.38
ax2.bar(x-w/2,gr,w,label="regulation-aware loop",color="#1f77b4")
ax2.bar(x+w/2,gb,w,label="regulation-blind FBA",color="#ff7f0e",alpha=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(labels,rotation=30,ha="right",fontsize=8)
ax2.set_title("Loop step 4-5: growth, regulation-aware vs blind"); ax2.set_ylabel("growth rate (1/h)"); ax2.legend(fontsize=9)
plt.tight_layout(); fig.savefig(OUT/"closed_loop.png",dpi=110); print(f"\nwrote {OUT}/closed_loop.png")

json.dump(dict(growth_table=table,
               arabinose_decisive=dict(crp_active=round(g_on,3),crp_inactive=round(g_off,3),fba_blind=round(g_fba,3)),
               conditional_essentiality=ess),open(OUT/"closed_loop.json","w"),indent=2)
print(f"wrote {OUT}/closed_loop.json")
