"""Target Intelligence: turn the per-gene attributes into the ranked answers pharma actually asks.
Produces target-priority, white-space, and combination tables. -> ti_*.json"""
import csv, json
from pathlib import Path
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")
dep={r["gene"]:float(r["frac_dep"]) for r in csv.DictReader(open(OUT/"depmap_essentiality.csv"))}
loeuf={}
for r in csv.DictReader(open(OUT/"integrated_cell_human.csv")):
    try: loeuf[r["gene"]]=float(r["loeuf"])
    except: pass
drugged={}; rd=csv.reader(open(H/"dgidb.tsv"),delimiter="\t"); next(rd)
for row in rd:
    if len(row)>9 and row[2] and row[9] and not row[9].lower().startswith("chembl"):
        drugged.setdefault(row[2],set()).add(row[9].title())
lit=json.load(open(OUT/"literature_counts.json"))
codep=json.load(open(OUT/"depmap_codep.json"))
# 1. TARGET PRIORITY — cancer-essential x germline-tolerant (safety window) x druggable
# selective dependency (essential in a SUBSET of lines) beats pan-essential (toxic to normal cells)
tp=[]
for g,fd in dep.items():
    lo=loeuf.get(g)
    if fd<0.35 or lo is None: continue
    selective = fd<=0.85
    tp.append(dict(gene=g,frac_dep=round(fd,2),loeuf=round(lo,2),druggable=g in drugged,
        drugs=sorted(drugged.get(g,[]))[:4],pubs=lit.get(g,0),selective=selective,
        window=round(fd*min(lo,2)*(1.3 if selective else 0.6),3)))
tp.sort(key=lambda x:-x["window"])
json.dump(tp[:600],open(OUT/"ti_target_priority.json","w"))
# 2. WHITE SPACE — high dependency, NOT druggable, low literature
ws=[dict(gene=g,frac_dep=round(fd,2),loeuf=round(loeuf.get(g,-1),2),pubs=lit.get(g,0))
    for g,fd in dep.items() if fd>=0.5 and g not in drugged]
ws.sort(key=lambda x:(x["pubs"],-x["frac_dep"]))
json.dump(ws[:600],open(OUT/"ti_whitespace.json","w"))
# 3. COMBOS — druggable co-essential partners per gene
combos={g:[[p,r] for p,r in parts if p in drugged][:5] for g,parts in codep.items()}
combos={g:v for g,v in combos.items() if v}
json.dump(combos,open(OUT/"ti_combos.json","w"))
print("target_priority:",len(tp),"| druggable:",sum(1 for t in tp if t['druggable']),"| whitespace:",len(ws),"| combos:",len(combos))
print("TOP DRUGGABLE SAFETY-WINDOW TARGETS:")
for t in [x for x in tp if x['druggable']][:10]: print("  ",t['gene'],"window",t['window'],"| dep",t['frac_dep'],"loeuf",t['loeuf'],"|",t['drugs'][:2])
print("TOP WHITE-SPACE (essential, undrugged, understudied):")
for w in ws[:10]: print("  ",w['gene'],"pubs",w['pubs'],"dep",w['frac_dep'])
