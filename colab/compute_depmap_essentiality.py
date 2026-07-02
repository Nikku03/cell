"""Reduce DepMap CRISPRGeneEffect.csv (cell lines x genes, Chronos scores) to per-gene MEASURED
essentiality: mean effect + fraction of lines dependent (Chronos < -0.5). -> depmap_essentiality.csv
(gene,mean_effect,frac_dep,essential). Essential = dependent in >=50% of lines."""
import csv
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
src=H/"CRISPRGeneEffect.csv"
import numpy as np
rows=[]; genes=None
with open(src) as f:
    rd=csv.reader(f); header=next(rd)
    genes=[g.split(" (")[0] for g in header[1:]]
    for r in rd:
        rows.append([float(x) if x else np.nan for x in r[1:]])
M=np.array(rows,dtype=np.float32)              # lines x genes
print("matrix:",M.shape)
mean_eff=np.nanmean(M,axis=0)
frac_dep=np.nanmean(M< -0.5,axis=0)
with open(OUT/"depmap_essentiality.csv","w",newline="") as o:
    w=csv.writer(o); w.writerow(["gene","mean_effect","frac_dep","essential"])
    ess=0
    for g,me,fd in zip(genes,mean_eff,frac_dep):
        e=1 if fd>=0.5 else 0; ess+=e
        w.writerow([g,round(float(me),3),round(float(fd),3),e])
print("wrote depmap_essentiality.csv | measured-essential genes:",ess,"of",len(genes))
