"""The TF's real job = modulate RNAP recruitment to the promoter. Test whether the
RNAP-recruitment signal (sigma70 -35/-10 promoter strength) is a usable,
sequence-derivable (universal) signal:
  does promoter score predict (a) baseline expression, (b) essentiality,
  (c) how regulated a gene is (in-degree of TFs)?
E. coli: genome + PRECISE expression + RegulonDB + Keio truth.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); COE=Path("data/external_data/coexpression")
B={"A":0,"C":1,"G":2,"T":3}
def enc(s): return np.array([B.get(c,-1) for c in s],dtype=np.int8)
def rc(a): comp=np.array([3,2,1,0,-1]); return comp[a][::-1]
seq=[l.strip() for l in open("data/external_data/ecoli_genome/ecoli_K12.fna") if not l.startswith(">")]
G="".join(seq).upper(); GLEN=len(G); Genc=enc(G)
genes={}
for l in open(REG/"GeneProductSet.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit() and p[5] in ("forward","reverse"):
        genes[p[1].lower()]=(int(p[3]),int(p[4]),p[5])
name2b={r["gene_name"].lower():r["gene_id"] for r in csv.DictReader(open(REG/"TRN.csv")) if r.get("gene_name") and r.get("gene_id")}
# RegulonDB TF in-degree (how many TFs regulate each gene)
indeg=defaultdict(int)
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); indeg[p[1].lower()]+=1
# PRECISE mean expression by b
expr={}
with open(COE/"log_tpm.csv") as f:
    r=csv.reader(f); next(r)
    for line in r: expr[line[0]]=float(np.mean([float(x) for x in line[1:]]))
# Keio essentiality by b
b_of={r["locus_tag"]:r["b_number"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
ess_b={}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"]=="beril_Keio":
        b=b_of.get(r["locus_tag"]);
        if b: ess_b[b]=int(r["essential"])
# sigma70 promoter score: best -35(TTGACA) ... spacer 15-19 ... -10(TATAAT) near gene start
M35=enc("TTGACA"); M10=enc("TATAAT")
def matchrun(a,m):
    n=len(a)-len(m)+1
    return np.array([int((a[i:i+len(m)]==m).sum()) for i in range(n)]) if n>0 else np.zeros(0)
def promoter_score(name,pad=120):
    if name not in genes: return None
    s,e,st=genes[name]
    a=Genc[max(0,s-pad-1):s-1] if st=="forward" else rc(Genc[e:min(GLEN,e+pad)])
    if len(a)<30 or not (a>=0).all(): return None
    s35=matchrun(a,M35); s10=matchrun(a,M10); best=0
    for i in range(len(s35)):
        for sp in range(15,20):
            j=i+6+sp
            if j<len(s10): best=max(best,s35[i]+s10[j])   # max 6+6=12
    return best

rows=[]
for nm in genes:
    b=name2b.get(nm); ps=promoter_score(nm)
    if b is None or ps is None: continue
    rows.append((nm,b,ps,expr.get(b),ess_b.get(b),indeg.get(nm,0)))
def corr(xs,ys):
    x=np.array(xs,float); y=np.array(ys,float); return float(np.corrcoef(x,y)[0,1])
ps_e=[(r[2],r[3]) for r in rows if r[3] is not None]
ps_x=[(r[2],r[3]) for r in rows if r[3] is not None]
print(f"genes scored: {len(rows)}")
print(f"corr(promoter sigma-score, mean expression)   = {corr([a for a,b in ps_e],[b for a,b in ps_e]):+.3f}  (n={len(ps_e)})")
# essentiality: mean promoter score essential vs non
es=[r[2] for r in rows if r[4]==1]; ne=[r[2] for r in rows if r[4]==0]
print(f"promoter score: essential mean={np.mean(es):.2f} (n={len(es)})  non-ess mean={np.mean(ne):.2f} (n={len(ne)})")
# AUC promoter-score -> essential
def auc(pos,neg):
    s=np.array(pos+neg); y=np.array([1]*len(pos)+[0]*len(neg))
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))
print(f"  AUC(promoter score -> essential) = {auc(es,ne):.3f}")
# regulated-ness: promoter score vs number of TF regulators
ps_reg=[(r[2],r[5]) for r in rows]
print(f"corr(promoter score, #TF regulators)          = {corr([a for a,b in ps_reg],[b for a,b in ps_reg]):+.3f}")
# expression vs essentiality (is high baseline expression tied to essential?)
ex_e=[r[3] for r in rows if r[4]==1 and r[3] is not None]; ex_ne=[r[3] for r in rows if r[4]==0 and r[3] is not None]
print(f"\nmean expression: essential={np.mean(ex_e):.2f}  non-ess={np.mean(ex_ne):.2f}  AUC(expr->ess)={auc(ex_e,ex_ne):.3f}")
