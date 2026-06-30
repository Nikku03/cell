"""Build + verify the GLOBAL CONDITION MULTIPLIER as master-regulator program
activities, and test whether they (a) capture the global condition variance and
(b) are predictable from the condition's EFFECTORS (metadata).

programs = master-regulator regulons (FNR, Fur, CRP, ArcA, ...) + growth (ribosomal)
observed activity(prog, cond) = mean z-expr of the regulon in that condition
verify (a): how much condition variance do the ~K programs reconstruct?
verify (b): do effectors predict activity?  growth-rate->ribosomal; anaerobic->FNR;
            iron-limited(dpd)->Fur; carbon->CRP.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
COE=Path("data/external_data/coexpression"); REG=Path("data/external_data/regulon")
# expression
cols=None; bs=[]; rows=[]
with open(COE/"log_tpm.csv") as f:
    r=csv.reader(f); cols=next(r)[1:]
    for line in r: bs.append(line[0]); rows.append([float(x) for x in line[1:]])
M=np.array(rows); bidx={b:i for i,b in enumerate(bs)}
Z=(M-M.mean(1,keepdims=True))/(M.std(1,keepdims=True)+1e-9)   # gene z across conditions
sidx={s:i for i,s in enumerate(cols)}
# metadata
meta={}
for r in csv.DictReader(open(COE/"metadata.csv")):
    meta[r["Sample ID"]]=r
# names
name2b={r["gene_name"].lower():r["gene_id"] for r in csv.DictReader(open(REG/"TRN.csv")) if r.get("gene_name") and r.get("gene_id")}
# master-regulator regulons (RegulonDB) -> b-numbers present in expression
tf_t=defaultdict(set)
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); tf_t[p[0].lower()].add(p[1].lower())
MASTERS=["crp","fnr","arca","fur","lexa","fis","ihf","hns","lrp","narl","soxs","oxyr","phob","ompr","nac"]
reg_b={}
for tf in MASTERS:
    b=[name2b.get(t) for t in tf_t.get(tf,[]) ]
    b=[x for x in b if x in bidx]
    if len(b)>=8: reg_b[tf]=b
# growth program = ribosomal proteins (names rpl*/rps*/rpm*)
ribo=[name2b.get(n) for n in name2b if n.startswith(("rpl","rps","rpm"))]
ribo=[x for x in ribo if x in bidx]
reg_b["growth_ribo"]=ribo
print(f"programs: {list(reg_b)}  (sizes {[len(v) for v in reg_b.values()]})")
# activity matrix: program x condition  (mean z over regulon)
progs=list(reg_b)
A=np.array([Z[[bidx[b] for b in reg_b[p]]].mean(0) for p in progs])   # K x C
print(f"activity matrix: {A.shape} (programs x conditions)")

# ---- verify (a): variance of condition effects reconstructed by the programs ----
R=M-M.mean(1,keepdims=True)                      # gene-centered (condition effects)
At=A.T                                            # C x K
At=np.hstack([At,np.ones((At.shape[0],1))])
# least squares per gene: R_gene ~ At ; R2
coef,_,_,_=np.linalg.lstsq(At,R.T,rcond=None)     # (K+1) x genes
pred=(At@coef).T
ss=((R-pred)**2).sum(); st=(R**2).sum()
print(f"\n(a) condition variance reconstructed by {len(progs)} master programs: R^2 = {1-ss/st:.3f}")

# ---- verify (b): effectors predict program activity ----
def act(prog): return A[progs.index(prog)]
def get(s,k): return meta.get(s,{}).get(k,"")
# growth: metadata growth rate vs ribosomal activity
gr=[]; ra=[]
for s in cols:
    v=get(s,"Growth Rate (1/hr)")
    try: g=float(v)
    except: continue
    gr.append(g); ra.append(act("growth_ribo")[sidx[s]])
print(f"\n(b) growth rate -> ribosomal-program activity: corr = {np.corrcoef(gr,ra)[0,1]:+.2f} (n={len(gr)})")
# FNR: anaerobic vs aerobic
ea=lambda s: get(s,"Electron Acceptor").lower()
anaer=[act("fnr")[sidx[s]] for s in cols if ea(s) and "o2" not in ea(s) and "oxygen" not in ea(s) and "aerobic"!=ea(s) and ea(s) not in ("","air")]
aer=[act("fnr")[sidx[s]] for s in cols if ea(s) in ("o2","oxygen","air","aerobic")]
if "fnr" in progs:
    print(f"    FNR activity: anaerobic mean={np.mean(anaer) if anaer else float('nan'):+.2f} (n={len(anaer)})  aerobic mean={np.mean(aer) if aer else float('nan'):+.2f} (n={len(aer)})")
# Fur: iron-limited (dpd in condition id) vs others
if "fur" in progs:
    fur=act("fur")
    dpd=[fur[sidx[s]] for s in cols if "dpd" in get(s,"Condition ID").lower() or "dpd" in s.lower()]
    rest=[fur[sidx[s]] for s in cols if not ("dpd" in get(s,"Condition ID").lower() or "dpd" in s.lower())]
    print(f"    Fur activity: iron-limited(dpd) mean={np.mean(dpd) if dpd else float('nan'):+.2f} (n={len(dpd)})  other mean={np.mean(rest):+.2f}")
# CRP: glucose vs non-glucose carbon
if "crp" in progs:
    crp=act("crp")
    glc=[crp[sidx[s]] for s in cols if "glucose" in get(s,"Carbon Source (g/L)").lower() or "glc" in get(s,"Condition ID").lower()]
    non=[crp[sidx[s]] for s in cols if get(s,"Carbon Source (g/L)") and not ("glucose" in get(s,"Carbon Source (g/L)").lower() or "glc" in get(s,"Condition ID").lower())]
    print(f"    CRP activity: glucose mean={np.mean(glc) if glc else float('nan'):+.2f} (n={len(glc)})  non-glucose mean={np.mean(non) if non else float('nan'):+.2f} (n={len(non)})")
import json
json.dump(dict(programs=progs,recon_r2=round(float(1-ss/st),3),
               growth_corr=round(float(np.corrcoef(gr,ra)[0,1]),3)),
          open("outputs/orphan/condition_multiplier.json","w"),indent=2)
print("\nwrote outputs/orphan/condition_multiplier.json")
