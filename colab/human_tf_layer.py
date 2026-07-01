"""Human regulatory (W3) layer: TRRUST TF->target network integrated with essentiality
(Hart CEG/NEG) and population constraint (gnomAD LOEUF). Questions: are TFs vital? are
essential genes more regulated? which TFs are master regulators of the essential core?
which TFs govern the environmental-adaptation genes?"""
import gzip, json
from collections import defaultdict
from pathlib import Path
import numpy as np
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
def readset(fn):
    s=set()
    for i,l in enumerate(open(H/fn)):
        if i>0 and l.split("\t")[0].strip().strip('"'): s.add(l.split("\t")[0].strip().strip('"'))
    return s
CEG=readset("CEGv2.txt"); NEG=readset("NEGv1.txt")
loeuf={}
with gzip.open(H/"gnomad_constraint.txt.bgz","rt") as f:
    h=f.readline().split("\t"); gi,li=h.index("gene"),h.index("oe_lof_upper")
    for l in f:
        p=l.split("\t")
        try: v=float(p[li])
        except: continue
        if p[gi] not in loeuf or v<loeuf[p[gi]]: loeuf[p[gi]]=v
# TRRUST
tf_targets=defaultdict(list); regs_of=defaultdict(set); TFs=set()
for l in open(H/"trrust_human.tsv"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=3:
        tf_targets[p[0]].append((p[1],p[2])); regs_of[p[1]].add(p[0]); TFs.add(p[0])
allgenes=set(loeuf)
def med(gs): 
    v=[loeuf[g] for g in gs if g in loeuf]; return round(float(np.median(v)),3) if v else None

print("=== TFs are VITAL (dosage-sensitive) ===")
print(f"  median LOEUF: TFs {med(TFs)}  vs all genes {med(allgenes)}  (lower = more constrained)")
print(f"  TFs that are core-essential (CEG): {len(TFs&CEG)}/{len(TFs)} ({100*len(TFs&CEG)/len(TFs):.0f}%)  "
      f"vs genome baseline {100*len(CEG&allgenes)/len(allgenes):.1f}%")

print("\n=== essential genes are MORE regulated ===")
ceg_ind=[len(regs_of[g]) for g in CEG if g in regs_of or g in NEG or True]
def indeg(gs): 
    v=[len(regs_of[g]) for g in gs]; return round(float(np.mean(v)),2)
print(f"  mean #regulators (TRRUST in-degree): CEG {indeg(CEG)}  vs NEG {indeg(NEG)}")

print("\n=== master regulators of the essential core ===")
mr=[]
for tf,ts in tf_targets.items():
    tgt=set(t for t,_ in ts); ne=len(tgt&CEG)
    if ne>=3: mr.append((tf,ne,len(tgt),round(ne/len(tgt),2),loeuf.get(tf)))
mr.sort(key=lambda x:-x[1])
for tf,ne,n,frac,lf in mr[:12]:
    print(f"  {tf:8s} essential targets {ne:3d}/{n:3d} (frac {frac})  TF LOEUF {lf}")

print("\n=== TFs governing environmental-adaptation genes ===")
ADAPT=["EPAS1","EGLN1","SLC24A5","SLC45A2","MC1R","LCT","FADS1","UCP1","G6PD","ACKR1"]
for g in ADAPT:
    r=sorted(regs_of.get(g,[]))
    if r: print(f"  {g:8s} <- {', '.join(r[:8])}")

summary=dict(
    tf_median_loeuf=med(TFs), all_median_loeuf=med(allgenes),
    tf_essential_frac=round(len(TFs&CEG)/len(TFs),3), genome_essential_frac=round(len(CEG&allgenes)/len(allgenes),3),
    ceg_mean_regulators=indeg(CEG), neg_mean_regulators=indeg(NEG),
    master_regulators=[dict(tf=tf,ess_targets=ne,targets=n,frac=frac,loeuf=lf) for tf,ne,n,frac,lf in mr[:20]],
    adaptation_regulators={g:sorted(regs_of.get(g,[])) for g in ADAPT})
json.dump(summary,open(OUT/"human_tf_layer.json","w"),indent=2)
print("\nwrote human_tf_layer.json")
