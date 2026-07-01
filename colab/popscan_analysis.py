"""Genome-wide-ish differentiation scan (Part 1) + ancestral polarization (Part 2).
Rank the ~400-gene panel by population differentiation; flag NOVEL candidates (not in the
known-adaptation list); polarize direction via AFR-as-ancestral baseline (Africans closest
to ancestral) to name the DERIVED allele + the ADAPTED population/environment."""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
SC=Path("/tmp/claude-0/-home-user-cell/558875a8-e44f-59af-a869-11202bf854d3/scratchpad")
raw=json.load(open(SC/"gnomad_genes.json"))
POPS=["afr","amr","eas","sas","nfe","fin"]
ENV_OF_POP={"eas":"E.Asia (cold/high-alt subset)","afr":"Africa (malaria/high-UV)",
    "nfe":"Europe (high-lat/low-UV/dairy)","fin":"N.Europe (cold/high-lat)","sas":"S.Asia (UV/malaria)","amr":"Americas (admixed)"}
KNOWN={"EPAS1","EGLN1","SLC24A5","SLC45A2","MC1R","LCT","FADS1","UCP1","G6PD","ACKR1"}
TFs=set()
for l in open(H/"trrust_human.tsv"):
    p=l.split("\t")
    if len(p)>=3: TFs.add(p[0])

def top_diff(sym):
    best=None
    for v in raw.get(sym,[]):
        ex=v.get("exome");
        if not ex or not ex.get("an"): continue
        pa={x["id"]:(x["ac"]/x["an"] if x["an"] and x["an"]>2000 else None) for x in ex["populations"]}
        vals={k:pa[k] for k in POPS if pa.get(k) is not None}
        if len(vals)<4: continue
        spread=max(vals.values())-min(vals.values())
        if best is None or spread>best["spread"]:
            best=dict(hgvsp=v.get("hgvsp"),spread=round(spread,3),afs={k:round(x,3) for k,x in vals.items()})
    return best
def polarize(afs):
    afr=afs.get("afr")
    if afr is None: return None
    if afr<0.5:   # alt allele rare in Africa -> alt is derived
        adapted=max(afs,key=afs.get); return dict(derived="alt",adapted_pop=adapted,derived_freq=afs[adapted])
    else:         # alt common in Africa -> alt is ancestral, ref is derived
        adapted=min(afs,key=afs.get); return dict(derived="ref",adapted_pop=adapted,derived_freq=round(1-afs[adapted],3))

rows=[]
for sym in raw:
    td=top_diff(sym)
    if not td: continue
    cat=("known_adaptation" if sym in KNOWN else "TF" if sym in TFs else "background")
    pol=polarize(td["afs"])
    rows.append(dict(gene=sym,category=cat,spread=td["spread"],variant=td["hgvsp"],
        afs=td["afs"],polar=pol))
rows.sort(key=lambda r:-r["spread"])
print(f"scanned {len(rows)} genes for population differentiation\n")
print("=== TOP 20 most population-differentiated (Part 1 scan) ===")
for r in rows[:20]:
    ap=r["polar"]["adapted_pop"] if r["polar"] else "?"
    tag={"known_adaptation":"[known]","TF":"[TF]","background":"[novel?]"}[r["category"]]
    print(f"  {r['gene']:9s} {tag:9s} spread={r['spread']:.2f} {str(r['variant']):14s} "
          f"-> derived enriched in {ap} ({ENV_OF_POP.get(ap,'')})")
# novel candidates = high differentiation, not known adaptation
novel=[r for r in rows if r["category"]!="known_adaptation" and r["spread"]>=0.30]
print(f"\n=== NOVEL high-differentiation candidates (not in known list, spread>=0.30): {len(novel)} ===")
for r in novel[:15]:
    ap=r["polar"]["adapted_pop"] if r["polar"] else "?"
    print(f"  {r['gene']:9s} [{r['category']}] spread={r['spread']:.2f} {r['variant']} -> {ap}")
known_ranks={r["gene"]:i+1 for i,r in enumerate(rows) if r["gene"] in KNOWN}
print(f"\n  known-adaptation genes recovered in top ranks: {sorted(known_ranks.items(),key=lambda x:x[1])[:8]}")
json.dump(dict(n_scanned=len(rows),top=rows[:40],
    novel_candidates=[dict(gene=r["gene"],category=r["category"],spread=r["spread"],
        variant=r["variant"],adapted_pop=(r["polar"]["adapted_pop"] if r["polar"] else None)) for r in novel],
    known_ranks=known_ranks,
    method="differentiation=max between-population AF gap; polarization=AFR-as-ancestral baseline",
    caveat="~400-gene targeted panel (TFs+background+controls), not all 20k; AFR-baseline "
           "polarization is a heuristic (validate top hits with an outgroup ancestral allele)."),
    open(OUT/"popscan_analysis.json","w"),indent=2)
print("\nwrote popscan_analysis.json")
