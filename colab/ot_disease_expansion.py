"""Expand mutation/disease data via Open Targets (integrates GWAS + ClinVar + somatic +
literature + drugs into evidence-scored gene->disease links). For disease/master TFs: pull
associated diseases + evidence-type composition (genetic/somatic/drug) + whether druggable."""
import json, urllib.request, warnings, socket, time
socket.setdefaulttimeout(35); warnings.filterwarnings("ignore")
from collections import defaultdict, Counter
from pathlib import Path
OUT=Path("outputs/orphan")
GENES=["TP53","MYC","MYCN","E2F1","SP1","NFKB1","RELA","STAT3","HIF1A","PAX6","PAX3","PAX8",
 "GATA4","GATA1","GATA3","TBX5","SOX9","SOX2","SOX10","RUNX1","RUNX2","HNF1A","HNF1B","HNF4A",
 "PDX1","NKX2-5","NKX2-1","CRX","NRL","FOXP2","FOXP3","FOXA2","MECP2","WT1","CEBPA","SPI1",
 "IKZF1","MEF2C","TBX1","GLI3","ATRX","SMAD4","CTCF","KLF1","CREBBP","EP300","TCF4","ZEB2"]
OT="https://api.platform.opentargets.org/api/v4/graphql"
def gq(q):
    req=urllib.request.Request(OT,data=json.dumps({"query":q}).encode(),headers={"Content-Type":"application/json"})
    for _ in range(2):
        try: return json.load(urllib.request.urlopen(req,timeout=30))
        except Exception: time.sleep(2)
    return None
def ensg(sym):
    d=gq('{search(queryString:"%s",entityNames:["target"]){hits{id name}}}'%sym)
    for h in ((d or {}).get("data",{}).get("search",{}) or {}).get("hits",[]):
        if h["name"].upper()==sym.upper(): return h["id"]
    hits=((d or {}).get("data",{}).get("search",{}) or {}).get("hits",[])
    return hits[0]["id"] if hits else None
rows=[]; dt_tot=Counter()
for sym in GENES:
    eid=ensg(sym)
    if not eid: print(f"{sym}: no id"); continue
    d=gq('{target(ensemblId:"%s"){associatedDiseases(page:{index:0,size:8}){count rows{score disease{name} datatypeScores{id score}}}}}'%eid)
    ad=((d or {}).get("data",{}).get("target",{}) or {}).get("associatedDiseases")
    if not ad: print(f"{sym}: no assoc"); continue
    n=ad["count"]; top=[]
    dtsum=defaultdict(float)
    for r in ad["rows"]:
        top.append((r["disease"]["name"],round(r["score"],2)))
        for dt in r.get("datatypeScores",[]): dtsum[dt["id"]]+=dt["score"]
    dom=max(dtsum,key=dtsum.get) if dtsum else None
    drug = dtsum.get("known_drug",0)>0
    somatic = dtsum.get("somatic_mutation",0)
    genetic = dtsum.get("genetic_association",0)
    for k,v in dtsum.items(): dt_tot[k]+=v
    rows.append(dict(gene=sym,n_diseases=n,top=top[:3],dominant_evidence=dom,druggable=drug,
        somatic=round(somatic,1),genetic=round(genetic,1)))
    print(f"  {sym:7s} diseases={n:5d}  top={top[0][0][:34]:34s} evidence={dom:18s} drug={'Y' if drug else '-'}",flush=True)
    time.sleep(0.25)
tot_assoc=sum(r["n_diseases"] for r in rows)
drug_tfs=[r["gene"] for r in rows if r["druggable"]]
print(f"\n=== summary ===")
print(f"  {len(rows)} TFs -> {tot_assoc} gene-disease associations (Open Targets; HPO had ~hundreds)")
print(f"  TFs that are DRUG targets (known_drug evidence): {len(drug_tfs)} -> {drug_tfs}")
print(f"  evidence-type totals (mutation = genetic+somatic dominate): {dict(dt_tot.most_common())}")
cancer_tfs=sorted([r for r in rows if r['somatic']>0.5],key=lambda x:-x['somatic'])[:8]
print(f"  most cancer/somatic-driven TFs: {[(r['gene'],round(r['somatic'],1)) for r in cancer_tfs]}")
json.dump(dict(n_tfs=len(rows),total_associations=tot_assoc,drug_target_tfs=drug_tfs,
    evidence_totals=dict(dt_tot),per_tf=rows),open(OUT/"ot_disease_expansion.json","w"),indent=2)
print("\nwrote ot_disease_expansion.json")
