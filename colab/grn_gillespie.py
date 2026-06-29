"""Dynamic regulatory layer for the cell model: TF occupancy -> production rate
-> mRNA/protein concentrations, simulated stochastically (Gillespie SSA), under
changing scenarios (effector/condition knobs).

The honest division of labor (given everything we proved):
  TOPOLOGY + SIGN (which TF regulates what, activator/repressor)  -> MEASURED
        (RegulonDB) - we do NOT predict the binding site.
  OCCUPANCY -> RATE: thermodynamic/Hill input function of the regulator's ACTIVE
        concentration (concentration is the variable that decides binding).
  RATE / CONC PARAMETERS: E. coli literature-typical defaults (mRNA t1/2 ~5 min,
        protein diluted by growth ~40 min, Hill n=2). Qualitatively correct;
        absolute numbers illustrative (per-gene kinetics are not predictable).

Per gene g: species mRNA_g, protein_g. Reactions:
   transcription  ->mRNA_g   prop = beta_g * RIF_g(active regulator concs)
   translation    ->prot_g   prop = k_tl * mRNA_g
   mRNA decay     mRNA_g->    prop = d_m * mRNA_g
   protein dilute prot_g->    prop = d_p * prot_g
RIF = product over regulators of Hill activation/repression. A scenario sets
each TF's ACTIVE FRACTION phi(t) (effector present -> repressor inactive, etc.).

Demos:
  A. SOS response  - real LexA regulon (RegulonDB). DNA damage inactivates LexA
     -> SOS genes derepress; repair restores repression. Conditional genes ON/OFF.
  B. Coherent feed-forward loop (AND) - sign-sensitive DELAY / response time
     (Alon, 'An Introduction to Systems Biology').
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT=Path("outputs/orphan"); REG=Path("data/external_data/regulon")
rng=np.random.default_rng(7)

# ---------- general Gillespie GRN engine ----------
def hill_act(x,K,n): return x**n/(K**n+x**n)
def hill_rep(x,K,n): return K**n/(K**n+x**n)

def simulate(genes, regulators, phi, T, dt_rec=1.0, params=None, init=None, max_steps=8_000_000):
    """genes: list of names. regulators: {gene:[(reg,sign)]} sign in {'activator','repressor'}.
       phi(reg,t)->active fraction [0,1]. returns times, protein-trajectory dict."""
    p=dict(beta=18.0, k_tl=5.0, d_m=np.log(2)/5, d_p=np.log(2)/40, K=40.0, n=2, leak=0.02)
    if params: p.update(params)
    gi={g:i for i,g in enumerate(genes)}; N=len(genes)
    m=np.zeros(N); pr=np.zeros(N)
    if init:
        for g,(mi,pi) in init.items(): m[gi[g]]=mi; pr[gi[g]]=pi
    t=0.0; rec_t=[]; rec_p=[[] for _ in range(N)]; next_rec=0.0; steps=0
    def rif(g,t):
        regs=regulators.get(g,[])
        if not regs: return 1.0
        f=1.0
        for reg,sign in regs:
            if reg not in gi: continue
            x=pr[gi[reg]]*phi(reg,t)
            h=hill_act(x,p["K"],p["n"]) if sign=="activator" else hill_rep(x,p["K"],p["n"])
            f*=(p["leak"]+(1-p["leak"])*h)
        return f
    while t<T and steps<max_steps:
        # propensities: per gene [transcription, translation, mdecay, pdecay]
        a=np.empty(4*N)
        for i,g in enumerate(genes):
            a[4*i]=p["beta"]*rif(g,t); a[4*i+1]=p["k_tl"]*m[i]
            a[4*i+2]=p["d_m"]*m[i];     a[4*i+3]=p["d_p"]*pr[i]
        a0=a.sum()
        if a0<=0: break
        tau=rng.exponential(1.0/a0); t+=tau
        # record
        while next_rec<=t and next_rec<=T:
            rec_t.append(next_rec)
            for i in range(N): rec_p[i].append(pr[i])
            next_rec+=dt_rec
        r=rng.random()*a0; idx=np.searchsorted(np.cumsum(a),r); g=idx//4; k=idx%4
        if   k==0: m[g]+=1
        elif k==1: pr[g]+=1
        elif k==2: m[g]=max(0,m[g]-1)
        else:      pr[g]=max(0,pr[g]-1)
        steps+=1
    return np.array(rec_t), {genes[i]:np.array(rec_p[i]) for i in range(N)}

# ---------- Demo A: SOS response from real LexA regulon ----------
tf=defaultdict(list)
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    pp=l.rstrip().split("\t"); tf[pp[0].lower()].append((pp[1].lower(),pp[2]))
lexa_targets=[g for g,s in tf["lexa"] if s=="repressor"]
canon=["reca","sula","uvra","dinb","ruva","umud","lexa"]
sos=[g for g in canon if g in set(lexa_targets) or g=="lexa"]
if "lexa" not in sos: sos=["lexa"]+sos
sos=list(dict.fromkeys(sos))[:7]
genesA=sos
regA={g:[("lexa","repressor")] for g in genesA}     # LexA represses every SOS gene (incl itself)
print(f"SOS network ({len(genesA)} genes): {genesA}")
T=400.0; t_damage,t_repair=120.0,260.0
def phiA(reg,t):
    if reg=="lexa": return 0.05 if (t_damage<=t<t_repair) else 1.0   # damage inactivates LexA
    return 1.0
# start at repressed steady state (LexA protein high, targets low)
initA={g:(0,5) for g in genesA}; initA["lexa"]=(2,120)
tA,trA=simulate(genesA,regA,phiA,T,params=dict(beta=20.0,K=40.0),init=initA)

# ---------- Demo B: coherent feed-forward loop (AND) -> delay ----------
genesB=["X","Y","Z"]
regB={"Y":[("X","activator")],"Z":[("X","activator"),("Y","activator")]}  # AND at Z
def phiB(reg,t): return 1.0 if (reg!="X") else (1.0 if t>=40 else 0.0)     # X signal turns ON at t=40
# but X protein itself must rise: model X as constitutively producible, gated by signal via phi on X-as-regulator
initB={"X":(0,0),"Y":(0,0),"Z":(0,0)}
# X is produced constitutively once "present": emulate by an input gene with no regulator but phi gating its OWN activity as regulator
regBfull={"X":[], **regB}
tB,trB=simulate(genesB,regBfull,phiB,200.0,params=dict(beta=25.0,K=50.0,n=2),init=initB)

# ---------- plots ----------
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
for g in genesA:
    ax1.plot(tA,trA[g],label=g,lw=1.6)
ax1.axvspan(t_damage,t_repair,color="red",alpha=0.08)
ax1.axvline(t_damage,color="red",ls="--",lw=1); ax1.axvline(t_repair,color="green",ls="--",lw=1)
ax1.text(t_damage+3,ax1.get_ylim()[1]*0.95,"DNA damage\n(LexA inactivated)",fontsize=8,color="red",va="top")
ax1.set_title("A. SOS response (real LexA regulon, Gillespie SSA)")
ax1.set_xlabel("time (min)"); ax1.set_ylabel("protein (molecules)"); ax1.legend(fontsize=8,ncol=2)
for g,c in zip(genesB,["#1f77b4","#ff7f0e","#2ca02c"]):
    ax2.plot(tB,trB[g],label=g,lw=1.8,color=c)
ax2.axvline(40,color="gray",ls="--",lw=1); ax2.text(42,ax2.get_ylim()[1]*0.9,"X signal ON",fontsize=8)
ax2.set_title("B. Coherent FFL (AND): Z delayed after X (response time)")
ax2.set_xlabel("time (min)"); ax2.set_ylabel("protein (molecules)"); ax2.legend(fontsize=9)
plt.tight_layout(); fig.savefig(OUT/"grn_gillespie.png",dpi=110); print(f"wrote {OUT}/grn_gillespie.png")

# ---------- quantify ----------
def mean_after(tr,t,lo,hi):
    msk=(t>=lo)&(t<hi); return float(tr[msk].mean()) if msk.any() else float("nan")
sos_summary={}
for g in genesA:
    base=mean_after(trA[g],tA,t_damage-40,t_damage); ind=mean_after(trA[g],tA,t_repair-40,t_repair)
    sos_summary[g]=dict(basal=round(base,1),induced=round(ind,1),fold=round(ind/max(base,1),1))
# FFL delay: time for Z to reach 50% of its plateau, vs Y
def t_half(tr,t,t0):
    plateau=tr[t>0.8*t.max()].mean()
    for i in range(len(t)):
        if t[i]>=t0 and tr[i]>=0.5*plateau: return round(float(t[i]-t0),1)
    return None
ffl=dict(Y_t_half=t_half(trB["Y"],tB,40),Z_t_half=t_half(trB["Z"],tB,40))
print("\nSOS fold-induction (basal->induced):")
for g,v in sos_summary.items(): print(f"  {g:<6} {v['basal']:>6} -> {v['induced']:>6}  ({v['fold']}x)")
print(f"\nFFL response delay: Y reaches 50% at +{ffl['Y_t_half']} min, Z at +{ffl['Z_t_half']} min (Z lags Y -> delay)")
json.dump(dict(sos_genes=genesA,sos=sos_summary,ffl=ffl,
               params="beta20-25/min, mRNA t1/2 5min, protein dilution 40min, Hill n=2, K=40",
               note="topology+sign from RegulonDB; rate params literature-typical defaults"),
          open(OUT/"grn_gillespie.json","w"),indent=2)
print(f"wrote {OUT}/grn_gillespie.json")
