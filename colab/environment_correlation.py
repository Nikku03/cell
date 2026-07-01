"""Part E: link adaptation-variant allele frequencies to ENVIRONMENTAL variables.
gnomAD ancestry groups -> representative ancestral-environment values (UV, latitude,
winter cold, dairy history, malaria burden). Correlate each adaptation variant's AF
across populations with each environment axis. n=6 ancestry groups -> ILLUSTRATIVE, not
a biobank GxE study (stated honestly)."""
import json
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan")
pc=json.load(open(OUT/"population_constraint.json"))
# representative ancestral-environment values per gnomAD population (published/typical)
POPS=["afr","amr","eas","sas","nfe","fin"]
ENV={  # higher = more of that pressure
 "UV_exposure":   dict(afr=9,amr=7,eas=5,sas=9,nfe=3,fin=2),
 "latitude_abs":  dict(afr=5,amr=15,eas=35,sas=20,nfe=50,fin=62),
 "winter_cold":   dict(afr=1,amr=3,eas=6,sas=2,nfe=7,fin=9),
 "dairy_history": dict(afr=2,amr=4,eas=1,sas=5,nfe=9,fin=9),
 "malaria_burden":dict(afr=9,amr=4,eas=2,sas=7,nfe=1,fin=0),
}
def pear(a,b):
    a=np.array(a,float); b=np.array(b,float)
    if a.std()<1e-9 or b.std()<1e-9: return 0.0
    return float(np.corrcoef(a,b)[0,1])
print("=== adaptation variant AF vs environmental axes (|r| across 6 ancestry groups) ===")
rows=[]
for g in pc["genes"]:
    if g["category"]!="adaptation" or not g["top_diff_variant"]: continue
    afs=g["top_diff_variant"]["afs"]
    pops=[p for p in POPS if p in afs]
    if len(pops)<4: continue
    af=[afs[p] for p in pops]
    best=None
    corrs={}
    for env,vals in ENV.items():
        r=pear(af,[vals[p] for p in pops]); corrs[env]=round(r,2)
        if best is None or abs(r)>abs(best[1]): best=(env,r)
    rows.append(dict(gene=g["sym"],variant=g["top_diff_variant"]["hgvsp"],
        expected=g["environment"].split("->")[0].strip() if "->" in g["environment"] else g["environment"],
        best_env=best[0],r=round(best[1],2),all_corr=corrs))
    print(f"  {g['sym']:8s} {g['top_diff_variant']['hgvsp']:14s} best={best[0]:15s} r={best[1]:+.2f}   expected: {rows[-1]['expected']}")
json.dump(dict(env_table=ENV,results=rows,
    caveat="n=6 ancestry groups with representative ancestral-environment values; "
           "illustrative of the approach, not a measured biobank GxE analysis. Direction "
           "depends on ancestral/derived polarization."),
    open(OUT/"environment_correlation.json","w"),indent=2)
# quick hit-rate: does best_env match expected pressure category?
KEY={"UV_exposure":["UV"],"latitude_abs":["UV","latitude"],"winter_cold":["cold","temperature"],
     "dairy_history":["diet","lactase"],"malaria_burden":["pathogen","malaria"]}
hit=0
for r in rows:
    kws=KEY.get(r["best_env"],[])
    if any(k.lower() in r["expected"].lower() for k in kws): hit+=1
print(f"\n  best-env matches expected pressure: {hit}/{len(rows)} (illustrative, n=6)")
print("wrote environment_correlation.json")
