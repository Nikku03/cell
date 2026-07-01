"""Merge the MEASURED TF networks (CollecTRI, DoRothEA-tiered) with curated TRRUST into one
evidence-tagged human regulatory network, then redo: coverage of essential genes, motifs,
master regulators of the essential core, TF constraint, TF->disease."""
import json, csv, gzip, warnings
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); rng=np.random.default_rng(0)
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
def readset(fn):
    s=set()
    for i,l in enumerate(open(H/fn)):
        if i>0 and l.split("\t")[0].strip().strip('"'): s.add(l.split("\t")[0].strip().strip('"'))
    return s
CEG=readset("CEGv2.txt")
loeuf={}
with gzip.open(H/"gnomad_constraint.txt.bgz","rt") as f:
    hh=f.readline().split("\t"); gi,li=hh.index("gene"),hh.index("oe_lof_upper")
    for l in f:
        p=l.split("\t")
        try: v=float(p[li])
        except: continue
        if p[gi] not in loeuf or v<loeuf[p[gi]]: loeuf[p[gi]]=v
# HPO disease
g2dis=defaultdict(set)
for i,l in enumerate(open(H/"genes_to_phenotype.txt")):
    if i==0: continue
    p=l.rstrip("\n").split("\t")
    if len(p)>=6: g2dis[p[1]].add(p[5])
# edges: (tf,target) -> set(evidence)
edges=defaultdict(set); TFset=set()
for l in open(H/"trrust_human.tsv"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=3: edges[(p[0],p[1])].add("TRRUST"); TFset.add(p[0])
for r in csv.DictReader(open(H/"collectri.tsv"),delimiter="\t"):
    a,b=r["source_genesymbol"],r["target_genesymbol"]
    if a and b: edges[(a,b)].add("CollecTRI"); TFset.add(a)
for r in csv.DictReader(open(H/"dorothea.tsv"),delimiter="\t"):
    a,b=r["source_genesymbol"],r["target_genesymbol"]; lv=(r.get("dorothea_level") or "").split(";")[0]
    if a and b: edges[(a,b)].add("DoRothEA-"+lv); TFset.add(a)
def highconf(ev): return ("TRRUST" in ev) or ("CollecTRI" in ev) or any(e in ev for e in("DoRothEA-A","DoRothEA-B"))
HC={k:v for k,v in edges.items() if highconf(v)}
print(f"=== merged regulatory network ===")
print(f"  total edges {len(edges)} (TRRUST-only had 9396)  |  high-confidence {len(HC)}  |  TFs {len(TFset)}")
trr=set(k for k,v in edges.items() if "TRRUST" in v); col=set(k for k,v in edges.items() if "CollecTRI" in v)
print(f"  TRRUST {len(trr)}, CollecTRI {len(col)}, overlap {len(trr&col)}")
# coverage of essential genes as targets
def cover(eset):
    tgt=set(b for (a,b) in eset); return len(set(CEG)&tgt)
print(f"  essential genes (CEG) with >=1 regulator: TRRUST {cover(trr)}/{len(CEG)} -> merged HC {cover(HC)}/{len(CEG)}")

# out-degree map (high-confidence)
out=defaultdict(set)
for (a,b) in HC: out[a].add(b)
TFs=set(out)
# TF constraint
def med(gs): v=[loeuf[g] for g in gs if g in loeuf]; return round(float(np.median(v)),3) if v else None
print(f"\n=== TFs vital ===  median LOEUF: TFs {med(TFs)} vs all genes {med(loeuf)}")

# motifs on high-confidence TF->TF subnetwork
tt=[(a,b) for (a,b) in HC if b in TFs and a!=b]
tto=defaultdict(set)
for a,b in tt: tto[a].add(b)
def ffl(g):
    n=0
    for a in g:
        for b in g[a]:
            if b in g: n+=len(g[a]&g[b])
    return n
def mutual(g):
    s=set()
    for a in g:
        for b in g[a]:
            if b in g and a in g[b] and a<b: s.add((a,b))
    return len(s)
def rand(g):
    e=[[a,b] for a in g for b in g[a]]
    for _ in range(len(e)*3):
        i,j=rng.integers(len(e)),rng.integers(len(e))
        if i!=j and e[i][0]!=e[j][0]: e[i][1],e[j][1]=e[j][1],e[i][1]
    g2=defaultdict(set)
    for a,b in e:
        if a!=b: g2[a].add(b)
    return g2
of=ffl(tto); om=mutual(tto)
fn=[];mn=[]
for _ in range(30):
    g2=rand(tto); fn.append(ffl(g2)); mn.append(mutual(g2))
def z(o,d): d=np.array(d); return round((o-d.mean())/(d.std()+1e-9),1)
print(f"\n=== motifs (high-conf TF->TF, {len(tt)} edges) ===")
print(f"  feed-forward loops: {of} (Z={z(of,fn)})   mutual: {om} (Z={z(om,mn)})")

# master regulators of essential core (merged)
mr=[]
for tf in out:
    ne=len(out[tf]&set(CEG))
    if ne>=8: mr.append((tf,ne,len(out[tf]),loeuf.get(tf)))
mr.sort(key=lambda x:-x[1])
print(f"\n=== master regulators of essential core (merged) — essential targets / total ===")
for tf,ne,n,lf in mr[:12]: print(f"  {tf:8s} {ne:4d}/{n:5d}  LOEUF {lf}")

# TF -> disease with measured targets
tf_dis_targets=[]
for tf in TFs:
    if tf in g2dis and g2dis[tf]:
        shared=[b for b in out[tf] if b in g2dis and g2dis[tf]&g2dis[b]]
        if shared: tf_dis_targets.append((tf,len(g2dis[tf]),len(shared)))
tf_dis_targets.sort(key=lambda x:-x[2])
print(f"\n=== disease TFs whose MEASURED targets share their disease (candidate causal regulon) ===")
for tf,nd,ns in tf_dis_targets[:10]: print(f"  {tf:8s} {nd} diseases, {ns} measured targets co-diseased")
json.dump(dict(total_edges=len(edges),high_conf=len(HC),tfs=len(TFset),
    trrust_edges=len(trr),collectri_edges=len(col),overlap=len(trr&col),
    ceg_covered_trrust=cover(trr),ceg_covered_merged=cover(HC),ceg_total=len(CEG),
    tf_median_loeuf=med(TFs),ffl=dict(obs=of,z=z(of,fn)),mutual=dict(obs=om,z=z(om,mn)),
    master_regulators=[dict(tf=t,ess_targets=ne,targets=n,loeuf=lf) for t,ne,n,lf in mr[:20]]),
    open(OUT/"merged_tf_network.json","w"),indent=2)
print("\nwrote merged_tf_network.json")
