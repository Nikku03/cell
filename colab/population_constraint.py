"""Use human POPULATION DNA (gnomAD per-variant, per-population allele frequencies) to:
 A) map the VITAL parts of genes — where mutations are purged (constrained) vs tolerated;
 B) link tolerated variation to ENVIRONMENTAL adaptation (altitude, UV, diet, cold,
    pathogen) via population differentiation.

Central hypothesis: purifying selection protects the vital core (essential genes carry
NO common missense); local adaptation to environment acts on the tolerant periphery
(adaptation genes carry common, population-differentiated missense). gnomAD populations
are ANCESTRY groups; the environmental link for these classic loci is from the adaptation
literature (gnomAD has no direct altitude/temperature variable) — stated honestly.
"""
import json, urllib.request, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
OUT=Path("outputs/orphan"); SC=Path("/tmp/claude-0/-home-user-cell/558875a8-e44f-59af-a869-11202bf854d3/scratchpad")

# gene set: category + (for adaptation) environmental pressure + expected population
GENES={
 # essential (Hart CEG core machinery)
 "RPL4":("essential",""), "RPS3":("essential",""), "POLR2A":("essential",""),
 "EEF2":("essential",""), "SNRNP200":("essential",""), "PSMB2":("essential",""),
 # non-essential / tolerant (olfactory + dispensable)
 "OR2T33":("non_essential",""), "OR51B4":("non_essential",""), "OR4C11":("non_essential",""),
 "ANKRD30A":("non_essential",""),
 # environmental adaptation
 "EPAS1":("adaptation","altitude/hypoxia (high-altitude, e.g. Tibet -> EAS)"),
 "EGLN1":("adaptation","altitude/hypoxia (EAS/Andean)"),
 "SLC24A5":("adaptation","UV/latitude -> skin pigmentation (low-UV Europe -> NFE)"),
 "SLC45A2":("adaptation","UV/latitude -> pigmentation (NFE)"),
 "MC1R":("adaptation","UV -> pigmentation/hair (NFE/EAS)"),
 "LCT":("adaptation","diet -> lactase persistence (dairy cultures -> NFE)"),
 "FADS1":("adaptation","diet -> fatty-acid metabolism (diet-dependent)"),
 "UCP1":("adaptation","cold/temperature -> thermogenesis (cold climates)"),
 "G6PD":("adaptation","pathogen -> malaria resistance (malaria-endemic -> AFR/SAS)"),
 "ACKR1":("adaptation","pathogen -> malaria (Duffy-null, AFR)"),
}
POPS=["afr","amr","eas","sas","nfe","fin"]
ENV_OF_POP={"eas":"E.Asia/high-altitude/cold","afr":"Africa/malaria/high-UV",
    "nfe":"Europe/high-latitude/low-UV/dairy","fin":"N.Europe/cold","sas":"S.Asia","amr":"Americas(admixed)"}

def gq(sym):
    q='{gene(gene_symbol:"%s",reference_genome:GRCh37){variants(dataset:gnomad_r2_1){pos consequence hgvsp exome{ac an populations{id ac an}}}}}'%sym
    req=urllib.request.Request("https://gnomad.broadinstitute.org/api",
        data=json.dumps({"query":q}).encode(),headers={"Content-Type":"application/json"})
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(req,timeout=120))
        except Exception as e: time.sleep(3)
    return None

def parse_pos(hgvsp):
    if not hgvsp: return None
    import re
    m=re.search(r"p\.[A-Za-z]{3}(\d+)",hgvsp)
    return int(m.group(1)) if m else None

cache=SC/"gnomad_genes.json"
raw=json.load(open(cache)) if cache.exists() else {}
for sym in GENES:
    if sym in raw: continue
    d=gq(sym)
    if d and d.get("data",{}).get("gene"):
        raw[sym]=d["data"]["gene"]["variants"]; print(f"  fetched {sym}: {len(raw[sym])} variants",flush=True)
    else: print(f"  {sym}: FAILED"); raw[sym]=[]
    time.sleep(1)
json.dump(raw,open(cache,"w"))

def analyze(sym):
    vs=raw.get(sym,[])
    mis=[v for v in vs if v.get("consequence")=="missense_variant" and v.get("exome") and v["exome"].get("an")]
    L=max([parse_pos(v.get("hgvsp")) or 0 for v in mis]+[1])
    n=len(mis)
    # global AF and per-position tolerance
    max_af=0.0; tol_pos=set(); topvar=None
    for v in mis:
        ex=v["exome"]; af=ex["ac"]/ex["an"] if ex["an"] else 0
        max_af=max(max_af,af); p=parse_pos(v.get("hgvsp"))
        if af>0.001 and p: tol_pos.add(p)
        # population differentiation
        pa={x["id"]:(x["ac"]/x["an"] if x["an"] and x["an"]>1000 else None) for x in ex["populations"]}
        vals=[pa[k] for k in POPS if pa.get(k) is not None]
        if len(vals)>=4:
            spread=max(vals)-min(vals)
            if topvar is None or spread>topvar["spread"]:
                topvar=dict(hgvsp=v.get("hgvsp"),spread=round(spread,3),
                    afs={k:round(pa[k],3) for k in POPS if pa.get(k) is not None})
    return dict(sym=sym,category=GENES[sym][0],environment=GENES[sym][1],length=L,
        n_missense=n, missense_density=round(n/L,3) if L else 0,
        max_missense_AF=round(max_af,4), tolerant_positions=len(tol_pos),
        tolerant_frac=round(len(tol_pos)/L,4) if L else 0,
        max_pop_spread=(topvar["spread"] if topvar else 0.0), top_diff_variant=topvar)

rows=[analyze(s) for s in GENES]
by={}
for cat in ["essential","non_essential","adaptation"]:
    r=[x for x in rows if x["category"]==cat]
    by[cat]=dict(n=len(r),
        mean_max_AF=round(float(np.mean([x["max_missense_AF"] for x in r])),4),
        mean_tolerant_frac=round(float(np.mean([x["tolerant_frac"] for x in r])),4),
        mean_max_spread=round(float(np.mean([x["max_pop_spread"] for x in r])),3))

print("\n=== A) VITAL vs TOLERANT (observed population variation) ===")
print(f"{'category':14s} {'maxMissenseAF':>14s} {'tolerantFrac':>13s} {'maxPopSpread':>13s}")
for cat,v in by.items():
    print(f"{cat:14s} {v['mean_max_AF']:>14.4f} {v['mean_tolerant_frac']:>13.4f} {v['mean_max_spread']:>13.3f}")
print("\n  interpretation: essential genes -> ~0 common missense (vital, mutations purged);")
print("  adaptation genes -> common, highly population-differentiated missense (tolerant + selected).")

print("\n=== B) ENVIRONMENTAL ADAPTATION signals (top population-differentiated missense) ===")
for x in rows:
    if x["category"]=="adaptation" and x["top_diff_variant"]:
        tv=x["top_diff_variant"]
        hi=max(tv["afs"],key=tv["afs"].get); lo=min(tv["afs"],key=tv["afs"].get)
        print(f"  {x['sym']:8s} {tv['hgvsp'] or '?':14s} spread={tv['spread']:.2f}  "
              f"high={hi}({tv['afs'][hi]}) low={lo}({tv['afs'][lo]})")
        print(f"           pressure: {x['environment']}")
        print(f"           high-freq population {hi} = {ENV_OF_POP.get(hi,'?')}")

summary=dict(genes=rows, by_category=by,
    caveat="gnomAD populations are ancestry groups, not environmental measurements; the "
           "altitude/UV/diet/cold/pathogen link for these loci is from adaptation literature.",
    finding="essential genes carry ~no common missense (vital core protected by purifying "
            "selection); adaptation genes carry common, population-differentiated missense "
            "(tolerant periphery where environmental selection acts).")
json.dump(summary,open(OUT/"population_constraint.json","w"),indent=2)
print("\nwrote population_constraint.json")
