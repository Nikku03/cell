"""Per-gene literature coverage from NCBI gene2pubmed (human), mapped to the cell's HGNC symbols
(via synonyms for the ~37 genes whose gene_info Symbol is a full name). -> literature_counts.json"""
import gzip, csv, json
from collections import Counter
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
cell=set(r["gene"] for r in csv.DictReader(open(OUT/"integrated_cell_human.csv")) if r["gene"])
c=Counter()
with gzip.open(H/"gene2pubmed.gz","rt") as f:
    next(f)
    for l in f:
        p=l.split("\t")
        if len(p)>2 and p[0]=="9606": c[p[1]]+=1
e2sym={}
with gzip.open(H/"gene_info.gz","rt") as f:
    next(f)
    for l in f:
        p=l.split("\t")
        if len(p)<6 or p[0]!="9606": continue
        if p[2] in cell: e2sym[p[1]]=p[2]
        else:
            for s in p[4].split("|"):
                if s in cell: e2sym[p[1]]=s; break
lit={}
for ent,n in c.items():
    s=e2sym.get(ent)
    if s: lit[s]=lit.get(s,0)+n
json.dump(lit,open(OUT/"literature_counts.json","w"))
print("literature: %d cell genes with publications"%len(lit))
