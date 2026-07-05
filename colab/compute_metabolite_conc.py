"""PREDICT metabolite concentrations (Problem 5, prediction tier).

Two principled routes, no new measurement:
  IONS      -> physiological constants (K+, Na+, Ca2+, Cl-, Mg2+, HCO3- are known, not predicted).
  CENTRAL   -> thermodynamic NET analysis (Kummel 2006 / Noor 2014 MDF): every reaction carrying flux
               must be downhill, dG = dG0' + RT*ln(Q) < 0. With flux directions (our FBA), curated dG0'
               for central carbon, and the measured anchors, that is a set of linear inequalities on the
               log-concentrations. Solve an LP for each free metabolite's feasible MIN/MAX -> a
               concentration RANGE + geometric-mean estimate. Predicts intermediates we never measured
               (1,3-BPG, 2PG, DHAP, GAP, ...) and cross-checks the ones we did.

Enriches metabolites.json in place (adds concentration_uM/conc_tier/conc_range_uM where missing).
Skips gracefully if metabolites.json or scipy absent. -> metabolites.json (updated)
"""
import json, math
from pathlib import Path
OUT=Path("outputs/orphan")
RT_ln10 = 8.314e-3*298.15*math.log(10)      # kJ/mol per log10 unit ~ 5.708

# intracellular ion concentrations (uM) — physiological constants (cytosolic, mammalian; Alberts/BioNumbers)
IONS_UM={"K+":140000,"Na+":12000,"Cl-":10000,"Ca2+":0.1,"Mg2+":1000,"HCO3-":15000,
         "Fe2+":1,"Zn2+":1,"Cu2+":0.01,"phosphate":10000,"Pi":10000}

# central glycolysis (forward), dG0' (kJ/mol, pH7) and stoichiometry over base metabolite names.
# H2O/H+ folded into dG0'. Textbook values (Lehninger / eQuilibrator component-contribution).
RXN=[
 ("HK",   -16.7, {"glucose":-1,"ATP":-1,"glucose-6-phosphate":1,"ADP":1}),
 ("PGI",   +1.7, {"glucose-6-phosphate":-1,"fructose-6-phosphate":1}),
 ("PFK",  -14.2, {"fructose-6-phosphate":-1,"ATP":-1,"fructose-1,6-bisphosphate":1,"ADP":1}),
 ("ALD",  +23.8, {"fructose-1,6-bisphosphate":-1,"DHAP":1,"GAP":1}),
 ("TPI",   +7.5, {"DHAP":-1,"GAP":1}),
 ("GAPDH", +6.3, {"GAP":-1,"NAD+":-1,"Pi":-1,"1,3-bisphosphoglycerate":1,"NADH":1}),
 ("PGK",  -18.8, {"1,3-bisphosphoglycerate":-1,"ADP":-1,"3-phosphoglycerate":1,"ATP":1}),
 ("PGM",   +4.4, {"3-phosphoglycerate":-1,"2-phosphoglycerate":1}),
 ("ENO",   +1.8, {"2-phosphoglycerate":-1,"phosphoenolpyruvate":1}),
 ("PK",   -31.4, {"phosphoenolpyruvate":-1,"ADP":-1,"pyruvate":1,"ATP":1}),
 ("LDH",  -25.1, {"pyruvate":-1,"NADH":-1,"L-lactate":1,"NAD+":1}),
]
# metabolites fixed to their measured/curated value (anchors); the rest are predicted
ANCHORS=["glucose","ATP","ADP","NAD+","NADH","pyruvate","L-lactate","Pi"]
LOG_LO,LOG_HI=-7.0,-1.0                      # 0.1 uM .. 100 mM physiological window (log10 M)
# Park2016 metabolite name -> our Human-GEM name, where spelling differs (central metabolites)
CONC_SYN={"a-ketoglutarate":"2-oxoglutarate","dihydroxyacetonephosphate":"dihydroxyacetone phosphate",
          "sn-glycerol-3-phosphate":"glycerol-3-phosphate","6-phospho-d-gluconate":"6-phosphogluconate"}

def load_measured_conc():
    """Park2016 measured absolute concentrations (fetch_metabolite_conc.py): metabolite name -> (uM, bigg)."""
    import csv
    f=OUT/"metabolite_conc_measured.tsv"; m={}
    if not f.exists(): return m
    for r in csv.DictReader(open(f), delimiter="\t"):
        try: uM=float(r["conc_uM"])
        except (KeyError, ValueError, TypeError): continue
        if r.get("metabolite") and uM>0: m[r["metabolite"].strip()]=(uM, (r.get("bigg") or "").strip())
    return m

def net_estimate(anchor_uM):
    """LP feasible min/max of log10[M] for each free central metabolite given anchors + dG0' + forward flux."""
    from scipy.optimize import linprog
    mets=sorted({m for _,_,s in RXN for m in s})
    free=[m for m in mets if m not in ANCHORS]
    idx={m:i for i,m in enumerate(free)}
    A=[]; b=[]
    for _,dg0,s in RXN:
        row=[0.0]*len(free); rhs=-dg0/RT_ln10
        for m,c in s.items():
            if m in idx: row[idx[m]]=c
            else:                                       # anchored -> constant to RHS
                a=anchor_uM.get(m)
                if a is None: row=None; break
                rhs-=c*math.log10(a*1e-6)
        if row is None: continue
        A.append(row); b.append(rhs-1e-6)               # sum(coef*x) <= rhs  (dG<0)
    bounds=[(LOG_LO,LOG_HI)]*len(free)
    out={}
    for m,i in idx.items():
        c=[0.0]*len(free); c[i]=1.0
        lo=linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        hi=linprog([-x for x in c], A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if lo.success and hi.success:
            xmin,xmax=lo.x[i],hi.x[i]
            out[m]=(10**((xmin+xmax)/2)*1e6, 10**xmin*1e6, 10**xmax*1e6)   # est,min,max in uM
    return out

def main():
    f=OUT/"metabolites.json"
    if not f.exists():
        print("metabolites.json absent -> run compute_metabolites first"); return
    try: import scipy  # noqa
    except Exception: print("scipy absent -> ion constants only"); scipy=None
    D=json.load(open(f)); M=D["metabolites"]
    lower={k.lower():k for k in M}
    def put(name, uM, tier, rng=None):
        k=lower.get(name.lower())
        if k is None: return False
        M[k]["concentration_uM"]=round(float(uM),4); M[k]["conc_tier"]=tier
        if rng: M[k]["conc_range_uM"]=[round(rng[0],4),round(rng[1],4)]
        return True
    # 0) MEASURED absolute concentrations (Park 2016) -> anchor at measured-literature tier. Matches by
    #    normalized name (spaces/hyphens/chirality-insensitive) or an explicit synonym. Highest-priority tier.
    import re
    def _norm(s): return re.sub(r"[\s\-,]", "", re.sub(r"^[ld]-","", s.lower()))
    nlookup={}
    for k in M: nlookup.setdefault(_norm(k), k)
    n_meas=0
    for pname,(uM,bigg) in load_measured_conc().items():
        syn=CONC_SYN.get(pname.lower(), pname)
        k=lower.get(syn.lower()) or lower.get(pname.lower()) or nlookup.get(_norm(syn)) or nlookup.get(_norm(pname))
        if k is None: continue
        if M[k].get("conc_tier")=="measured": continue          # don't override a real measured value
        M[k]["concentration_uM"]=round(float(uM),4); M[k]["conc_tier"]="measured-literature"
        M[k]["conc_source"]="Park2016"; n_meas+=1
    # 1) ions -> physiological constants (only set if not already measured)
    n_ion=0
    for ion,uM in IONS_UM.items():
        k=lower.get(ion.lower())
        if k and M[k].get("conc_tier")!="measured":
            if put(ion,uM,"physiological-constant"): n_ion+=1
    # 2) central metabolites -> thermodynamic NET estimate
    n_thermo=0; checks=[]
    anchor_uM={}
    for a in ANCHORS:
        k=lower.get(a.lower())
        if k and M[k].get("concentration_uM"): anchor_uM[a]=M[k]["concentration_uM"]
    if 'scipy' in dir() and scipy is not None and len(anchor_uM)>=4:
        est=net_estimate(anchor_uM)
        for m,(e,mn,mx) in est.items():
            k=lower.get(m.lower())
            prev=M[k].get("concentration_uM") if k else None
            prev_tier=M[k].get("conc_tier") if k else None
            if prev and prev_tier in ("measured","measured-literature","curated-literature"):
                checks.append((m,prev,e))               # cross-check vs measured/curated, don't overwrite
            elif put(m,e,"thermodynamic-estimate",(mn,mx)):
                n_thermo+=1
    json.dump(D, open(f,"w"))
    print(f"metabolite concentrations: +{n_meas} MEASURED (Park2016 absolute, measured-literature tier), "
          f"+{n_ion} ions (physiological constants), +{n_thermo} central metabolites (thermodynamic NET estimate)")
    if checks:
        fold=[abs(math.log10(e/p)) for m,p,e in checks if p>0 and e>0]
        med=10**(sorted(fold)[len(fold)//2]) if fold else None
        print(f"  cross-check vs curated on {len(checks)} anchored intermediates: median fold-diff "
              f"{med:.2f}x" if med else "  (no overlap to cross-check)")
        for m,p,e in checks[:5]: print(f"    {m:26} curated {p:.3g} uM vs thermo {e:.3g} uM")

if __name__=="__main__":
    main()
