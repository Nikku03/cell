"""METABOLITES as first-class species (Problem 5). Everything else in the model is gene/protein-centric;
this makes metabolites, lipids and ions queryable entities in their own right.

For each metabolite (Human-GEM, compartment-collapsed to a base name) we record:
  - the reactions that PRODUCE and CONSUME it, its compartments, degree, and hub/currency flag;
  - its FLUX turnover (from flux.json) -> how much material actually moves through it;
  - its absolute intracellular CONCENTRATION (curated central-metabolome literature, or a measured file),
    with an honest tier -- the metabolite analogue of the PaxDb protein-abundance layer.

Reuses compute_flux.parse_gem (proper stoichiometry + compartments). Skips gracefully if Human-GEM
absent. -> metabolites.json {name: {compartments, produced_by, consumed_by, n_rxn, hub, turnover,
concentration_uM, conc_tier}}
"""
import json, os, re
from collections import defaultdict
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")

# currency / hub metabolites (not tracked as pathway intermediates; flagged so queries can skip them)
HUB={"H2O","H+","H","CO2","O2","NH3","NH4+","H2O2","Pi","PPi","phosphate","diphosphate",
     "NAD+","NADH","NADP+","NADPH","FAD","FADH2","FMN","ATP","ADP","AMP","GTP","GDP","GMP",
     "CTP","CDP","UTP","UDP","CoA","ACP","Na+","K+","Cl-","Ca2+","Mg2+","Fe2+","Fe3+","HCO3-",
     "SAM","SAH","GSH","GSSG","H2S","SO4","sulfate"}

# curated absolute intracellular concentrations (uM), central metabolism — order-of-magnitude, mammalian
# where known else general (Park et al. 2016 Nat Chem Biol; BioNumbers). tier = 'curated-literature'.
CURATED_UM={
 "ATP":3000,"ADP":500,"AMP":100,"GTP":300,"GDP":60,"NAD+":500,"NADH":50,"NADP+":3,"NADPH":120,
 "FAD":150,"CoA":140,"acetyl-CoA":40,"glutathione":8000,"GSH":8000,"GSSG":200,
 "L-glutamate":15000,"glutamate":15000,"L-glutamine":4000,"glutamine":4000,"L-aspartate":4000,
 "aspartate":4000,"L-alanine":4000,"alanine":4000,"glycine":3000,"L-serine":1000,"serine":1000,
 "glucose":2000,"glucose-6-phosphate":300,"G6P":300,"fructose-6-phosphate":80,"F6P":80,
 "fructose-1,6-bisphosphate":15000,"FBP":15000,"3-phosphoglycerate":500,"3PG":500,
 "phosphoenolpyruvate":100,"PEP":100,"pyruvate":100,"L-lactate":3000,"lactate":3000,
 "citrate":300,"cis-aconitate":20,"isocitrate":30,"2-oxoglutarate":100,"alpha-ketoglutarate":100,
 "AKG":100,"succinyl-CoA":30,"succinate":500,"fumarate":80,"L-malate":2000,"malate":2000,
 "oxaloacetate":5,"Pi":10000,"phosphate":10000,"PPi":300,"6-phosphogluconate":30,
 "ribose-5-phosphate":100,"UDP-glucose":600,"UTP":2000,"CTP":500,"S-adenosylmethionine":50,
 "SAM":50,"creatine":10000,"taurine":20000,"myo-inositol":2000,"carnitine":500,
}
SYN={"g6p":"glucose-6-phosphate","f6p":"fructose-6-phosphate","fbp":"fructose-1,6-bisphosphate",
     "3pg":"3-phosphoglycerate","pep":"phosphoenolpyruvate","akg":"2-oxoglutarate",
     "alpha-ketoglutarate":"2-oxoglutarate","oxoglutarate":"2-oxoglutarate","gsh":"glutathione"}

def norm(name):
    n=name.strip(); low=n.lower()
    return SYN.get(low, low)

def load_conc():
    """absolute intracellular metabolite concentrations (uM). Optional measured file (METAB_CONC_FILE:
    tsv 'metabolite<TAB>conc_uM'), else curated literature set. -> {norm_name: (uM, tier)}."""
    out={}
    f=os.environ.get("METAB_CONC_FILE","")
    if f and Path(f).exists():
        for l in open(f):
            p=l.rstrip("\n").split("\t")
            if len(p)>=2 and p[0].lower() not in ("metabolite","name"):
                try: out[norm(p[0])]=(float(p[1]),"measured")
                except: pass
    for name,uM in CURATED_UM.items():
        k=norm(name)
        if k not in out: out[k]=(float(uM),"curated-literature")
    return out

def main():
    try:
        from compute_flux import parse_gem, base_met
    except Exception as e:
        print("cannot import compute_flux.parse_gem:",repr(e)[:80]); return
    rxns, met_idx = parse_gem()
    if not rxns:
        print("Human-GEM absent -> metabolite layer skipped (runs on Colab)"); return
    inv={i:n for n,i in met_idx.items()}
    flux=(json.load(open(OUT/"flux.json")).get("flux",{}) if (OUT/"flux.json").exists() else {})
    conc=load_conc()
    # aggregate per base metabolite name (collapse compartments)
    prod=defaultdict(set); cons=defaultdict(set); comps=defaultdict(set); nrxn=defaultdict(int)
    turnover=defaultdict(float)
    for r in rxns:
        rid=r["id"]; v=flux.get(rid,{}).get("v")
        for mi,coef in r["stoich"].items():
            full=inv[mi]; bn=base_met(full)
            m=re.search(r"\[([^\]]*)\]\s*$", full)
            if m: comps[bn].add(m.group(1))
            nrxn[bn]+=1
            # producer/consumer by stoichiometric sign; reversible reactions count as both
            if coef>0 or r["rev"]: prod[bn].add(rid)
            if coef<0 or r["rev"]: cons[bn].add(rid)
            if v is not None: turnover[bn]+=abs(coef*v)
    out={}
    for bn in nrxn:
        c=conc.get(norm(bn))
        rec=dict(compartments=sorted(comps[bn]),
                 produced_by=sorted(prod[bn])[:15], consumed_by=sorted(cons[bn])[:15],
                 n_producers=len(prod[bn]), n_consumers=len(cons[bn]), n_rxn=nrxn[bn],
                 hub=bn in HUB)
        if turnover.get(bn): rec["turnover"]=round(turnover[bn]/2.0,4)   # steady-state throughput
        if c: rec["concentration_uM"]=c[0]; rec["conc_tier"]=c[1]
        out[bn]=rec
    n_conc=sum(1 for r in out.values() if "concentration_uM" in r)
    n_flux=sum(1 for r in out.values() if "turnover" in r)
    payload=dict(metabolites=out, summary=dict(metabolites=len(out),
                 with_concentration=n_conc, with_turnover=n_flux, hubs=sum(1 for r in out.values() if r["hub"]),
                 conc_measured=sum(1 for r in out.values() if r.get("conc_tier")=="measured"),
                 conc_curated=sum(1 for r in out.values() if r.get("conc_tier")=="curated-literature")))
    json.dump(payload, open(OUT/"metabolites.json","w"))
    s=payload["summary"]
    print(f"metabolites: {s['metabolites']} species first-class | {s['with_concentration']} with concentration "
          f"({s['conc_measured']} measured, {s['conc_curated']} curated) | {s['with_turnover']} with flux turnover | {s['hubs']} hubs")

if __name__=="__main__":
    main()
