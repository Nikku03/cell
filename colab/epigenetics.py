"""Epigenetic regulatory architecture per gene: CpG-island promoter status + counts of
ENCODE cCREs (promoter-like/enhancer-like/CTCF) near the TSS. Integrate with essentiality,
TFs, and disease. Hypothesis: housekeeping/essential genes = CpG-island promoter + few
enhancers (constitutive); developmental/disease TFs = many enhancers (complex, cell-type-
specific control) -- the epigenetic basis of differentiation and of non-coding disease."""
import gzip, csv, json, bisect, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
H=Path("data/external_data/human")
def readset(fn):
    s=set()
    for i,l in enumerate(open(H/fn)):
        if i>0 and l.split("\t")[0].strip().strip('"'): s.add(l.split("\t")[0].strip().strip('"'))
    return s
CEG=readset("CEGv2.txt"); NEG=readset("NEGv1.txt")
loeuf={}
with gzip.open(H/"gnomad_constraint.txt.bgz","rt") as f:
    hh=f.readline().split("\t"); gi,li=hh.index("gene"),hh.index("oe_lof_upper")
    for l in f:
        p=l.split("\t")
        try: v=float(p[li])
        except: continue
        if p[gi] not in loeuf or v<loeuf[p[gi]]: loeuf[p[gi]]=v
# TFs (merged) + disease TFs (HPO)
TFs=set()
for l in open(H/"trrust_human.tsv"):
    p=l.split("\t");
    if len(p)>=3: TFs.add(p[0])
for r in csv.DictReader(open(H/"collectri.tsv"),delimiter="\t"):
    if r["source_genesymbol"]: TFs.add(r["source_genesymbol"])
g2dis=defaultdict(set)
for i,l in enumerate(open(H/"genes_to_phenotype.txt")):
    if i==0: continue
    p=l.rstrip("\n").split("\t")
    if len(p)>=6: g2dis[p[1]].add(p[5])
disTF=set(t for t in TFs if t in g2dis)
# gene TSS from refGene (longest tx per symbol)
tss={}
for l in gzip.open(H/"refGene.txt.gz","rt"):
    p=l.rstrip("\n").split("\t")
    if len(p)<13: continue
    sym=p[12]; chrom=p[2]; strand=p[3]; txs=int(p[4]); txe=int(p[5])
    if "_" in chrom: continue
    t=txs if strand=="+" else txe
    if sym not in tss or (txe-txs)>tss[sym][2]:
        tss[sym]=(chrom,t,txe-txs)
# CpG islands per chrom
cpg=defaultdict(list)
for l in gzip.open(H/"cpgIslandExt.txt.gz","rt"):
    p=l.rstrip("\n").split("\t")
    if len(p)<4: continue
    cpg[p[1]].append((int(p[2]),int(p[3])))
for c in cpg: cpg[c].sort()
cpg_starts={c:[a for a,_ in v] for c,v in cpg.items()}
def has_cpg(chrom,pos,pad=1000):
    v=cpg.get(chrom); 
    if not v: return False
    st=cpg_starts[chrom]; i=bisect.bisect_right(st,pos+pad)
    for j in range(max(0,i-6),i):
        a,b=v[j]
        if b>=pos-pad and a<=pos+pad: return True
    return False
# cCREs per chrom by category
print("indexing 2.3M cCREs ...",flush=True)
ccre=defaultdict(lambda: defaultdict(list))  # chrom -> cat -> [midpoints]
for l in open(H/"cCREs.bed"):
    p=l.rstrip("\n").split("\t")
    if len(p)<6: continue
    chrom=p[0]; mid=(int(p[1])+int(p[2]))//2; t=p[5]
    cat="promoter" if "PLS" in t else ("enhancer" if "ELS" in t else ("ctcf" if "CTCF" in t else "other"))
    ccre[chrom][cat].append(mid)
for c in ccre:
    for cat in ccre[c]: ccre[c][cat].sort()
def count_win(chrom,pos,cat,win=50000):
    arr=ccre.get(chrom,{}).get(cat)
    if not arr: return 0
    return bisect.bisect_right(arr,pos+win)-bisect.bisect_left(arr,pos-win)

rows={}
for sym,(chrom,t,_) in tss.items():
    rows[sym]=dict(cpg_promoter=int(has_cpg(chrom,t)),
        enhancers=count_win(chrom,t,"enhancer"), promoters=count_win(chrom,t,"promoter"),
        ctcf=count_win(chrom,t,"ctcf"))
def stats(names,key):
    v=[rows[g][key] for g in names if g in rows]
    return round(float(np.mean(v)),2) if v else None
def frac_cpg(names):
    v=[rows[g]["cpg_promoter"] for g in names if g in rows]
    return round(float(np.mean(v)),2) if v else None
allg=set(rows)
groups={"essential (CEG)":CEG,"non-essential (NEG)":NEG,"all genes":allg,
        "TFs":TFs&allg,"disease TFs":disTF&allg}
print("\n=== epigenetic regulatory architecture ===")
print(f"{'group':22s} {'CpG-promoter':>12s} {'enhancers':>10s} {'promoters':>10s} {'CTCF':>6s}")
out={}
for g,s in groups.items():
    out[g]=dict(n=len(s&allg),cpg_promoter=frac_cpg(s),enhancers=stats(s,"enhancers"),
        promoters=stats(s,"promoters"),ctcf=stats(s,"ctcf"))
    print(f"  {g:20s} {frac_cpg(s):>12} {stats(s,'enhancers'):>10} {stats(s,'promoters'):>10} {stats(s,'ctcf'):>6}")
# most enhancer-dense genes (complex regulation) -- are they developmental TFs?
dense=sorted(allg,key=lambda g:-rows[g]["enhancers"])[:20]
print(f"\n  most enhancer-dense genes (complex regulation): {[d for d in dense[:15]]}")
print(f"  ... of top 50, how many are TFs: {sum(1 for g in sorted(allg,key=lambda g:-rows[g]['enhancers'])[:50] if g in TFs)}/50")
import json
json.dump(dict(groups=out,n_genes=len(rows),
    top_enhancer_dense=dense[:30],
    top50_tf_fraction=round(sum(1 for g in sorted(allg,key=lambda g:-rows[g]['enhancers'])[:50] if g in TFs)/50,2)),
    open("outputs/orphan/epigenetics.json","w"),indent=2)
print("\nwrote epigenetics.json")
