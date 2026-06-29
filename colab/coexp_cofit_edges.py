"""Regulatory EDGE prediction from FUNCTIONAL data: co-expression + co-fitness.
The lever that actually carries signal (vs the operator-sequence wall).

Predict, for each E. coli TF, which genes it regulates (RegulonDB truth), from:
  COEXP : |Pearson r| of the TF's mRNA profile vs each gene, across PRECISE
          (278 RNA-seq conditions). + or - regulation -> use |r|.
  COFIT : |co-fitness| from feba (Keio), TF vs each gene.
  COMB  : z(COEXP) + z(COFIT).
Blind setting: rank ALL genes per TF; score vs RegulonDB targets.
Metrics: AUC (target vs non-target), recall@K & precision@K (K=#targets),
enrichment over base rate. Head-to-head + per protein family.

Caveat (stated, not hidden): TF mRNA often does NOT track TF ACTIVITY
(ligand/post-translational control), so TF->target co-expression is the weak
direction; co-fitness is orthogonal. We measure what each really delivers.

DATA (PRECISE RNA-seq compendium, SBRG; not committed - re-fetch with):
  base=https://raw.githubusercontent.com/SBRG/precise-db/master/data
  curl -L -o data/external_data/coexpression/log_tpm.csv   $base/log_tpm.csv
  curl -L -o data/external_data/coexpression/gene_info.csv $base/gene_info.csv
  curl -L -o data/external_data/coexpression/metadata.csv  $base/metadata.csv
"""
import csv, json, sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
COE=Path("data/external_data/coexpression"); REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")
con=sqlite3.connect("data/external_data/feba/feba.db"); cur=con.cursor()

# ---- load PRECISE expression (b-number x samples) ----
hdr=None; bs=[]; rows=[]
with open(COE/"log_tpm.csv") as f:
    r=csv.reader(f); hdr=next(r)
    for line in r:
        bs.append(line[0]); rows.append([float(x) for x in line[1:]])
X=np.array(rows,float); bidx={b:i for i,b in enumerate(bs)}
# standardize rows for fast correlation
Z=(X-X.mean(1,keepdims=True))/ (X.std(1,keepdims=True)+1e-9); n_s=X.shape[1]
print(f"expression: {X.shape[0]} genes x {n_s} samples")

# gene name <-> b
name2b={}
for r in csv.DictReader(open(COE/"gene_info.csv")):
    b=r[list(r.keys())[0]]
    if r.get("gene_name"): name2b[r["gene_name"].lower()]=b
# RegulonDB TF -> targets (names)
tf_t=defaultdict(set)
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); tf_t[p[0].lower()].add(p[1].lower())
# feba cofit (b-space)
kb={l:sn for l,sn in cur.execute("SELECT locusId,sysName FROM Gene WHERE orgId='Keio' AND sysName LIKE 'b%'")}
cofit=defaultdict(dict)
for l,h,cf in cur.execute("SELECT locusId,hitId,cofit FROM Cofit WHERE orgId='Keio'"):
    if l in kb and h in kb: cofit[kb[l]][kb[h]]=abs(cf)
# family (Pfam DBD) per TF b
dom=defaultdict(list)
for l,dn in cur.execute("SELECT locusId,domainName FROM GeneDomain WHERE orgId='Keio' AND domainDb='PFam'"):
    if l in kb: dom[kb[l]].append(dn or "")
FAMP=[("HTH_AraC","AraC"),("LysR_substrate","LysR"),("HTH_1","LysR"),("LacI","LacI"),("Trans_reg_C","OmpR"),
      ("GerE","NarL"),("Crp","Crp"),("GntR","GntR"),("FCD","GntR"),("HTH_18","MarR"),("MarR","MarR"),
      ("TetR","TetR"),("Sigma54","bEBP"),("Lrp","Lrp"),("AsnC","Lrp"),("IclR","IclR"),("DeoR","DeoR")]
def fam_of(b):
    ds=" ".join(dom.get(b,[]))
    for k,f in FAMP:
        if k.lower() in ds.lower(): return f
    return "other"

def auc(score,ypos):
    order=np.argsort(-score); y=ypos[order]
    P=y.sum(); N=len(y)-P
    if P==0 or N==0: return None
    ranks=np.empty(len(y)); ranks[np.argsort(-score)]=np.arange(len(y))
    # rank-sum AUC
    r=np.empty(len(score)); o=np.argsort(score); r[o]=np.arange(len(score)); m=ypos==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))

allb=bs
rows_out=[]
for tf,tgts in tf_t.items():
    tb=name2b.get(tf)
    if tb is None or tb not in bidx: continue
    tgt_b=set(name2b.get(t) for t in tgts) & set(allb); tgt_b.discard(tb)
    if len(tgt_b)<10: continue
    # coexp score over all genes
    co=(Z @ Z[bidx[tb]])/n_s; coexp=np.abs(co); coexp[bidx[tb]]=0
    # cofit score over all genes (0 if absent)
    cf=np.array([cofit.get(tb,{}).get(b,0.0) for b in allb])
    ypos=np.array([1 if b in tgt_b else 0 for b in allb])
    K=int(ypos.sum()); base=K/len(allb)
    def z(a):
        s=a.std(); return (a-a.mean())/(s if s>1e-9 else 1)
    comb=z(coexp)+z(cf)
    def recK(sc):
        top=set(np.argsort(-sc)[:K]); return len(top & set(np.where(ypos==1)[0]))/K
    a_co=auc(coexp,ypos); a_cf=auc(cf,ypos); a_cb=auc(comb,ypos)
    rows_out.append(dict(tf=tf,fam=fam_of(tb),n=K,
        auc_coexp=a_co,auc_cofit=a_cf,auc_comb=a_cb,
        rec_coexp=round(recK(coexp),3),rec_cofit=round(recK(cf),3),rec_comb=round(recK(comb),3),
        enr_comb=round(recK(comb)/base,1) if base>0 else 0))

def mean(k):
    v=[r[k] for r in rows_out if r[k] is not None]; return round(float(np.mean(v)),3) if v else None
print(f"\nTFs evaluated: {len(rows_out)}")
print(f"{'signal':<10}{'mean AUC':>10}{'mean recall@K':>15}")
print(f"{'co-express':<10}{mean('auc_coexp'):>10}{mean('rec_coexp'):>15}")
print(f"{'co-fitness':<10}{mean('auc_cofit'):>10}{mean('rec_cofit'):>15}")
print(f"{'COMBINED':<10}{mean('auc_comb'):>10}{mean('rec_comb'):>15}")
print(f"\nmean enrichment (combined recall@K / base): {round(float(np.mean([r['enr_comb'] for r in rows_out])),1)}x")

# top TFs by combined AUC
rows_out.sort(key=lambda r:-(r['auc_comb'] or 0))
print(f"\n{'TF':<8}{'fam':<8}{'n':>4}{'coexp':>7}{'cofit':>7}{'COMB':>7}")
for r in rows_out[:15]:
    print(f"  {r['tf']:<6}{r['fam']:<8}{r['n']:>4}{(r['auc_coexp'] or 0):>7.3f}{(r['auc_cofit'] or 0):>7.3f}{(r['auc_comb'] or 0):>7.3f}")

# per family
byf=defaultdict(list)
for r in rows_out: byf[r['fam']].append(r)
print(f"\n=== by family (mean combined AUC) ===")
for f,rs in sorted(byf.items(),key=lambda x:-np.mean([r['auc_comb'] or 0 for r in x[1]])):
    if len(rs)>=2:
        print(f"  {f:<8} n_tf={len(rs):>2}  coexp={np.mean([r['auc_coexp'] or 0 for r in rs]):.3f}  cofit={np.mean([r['auc_cofit'] or 0 for r in rs]):.3f}  COMB={np.mean([r['auc_comb'] or 0 for r in rs]):.3f}")
json.dump(dict(n_tf=len(rows_out),mean=dict(coexp=mean('auc_coexp'),cofit=mean('auc_cofit'),combined=mean('auc_comb')),
               mean_recallK=dict(coexp=mean('rec_coexp'),cofit=mean('rec_cofit'),combined=mean('rec_comb')),
               per_tf=rows_out),open(OUT/"coexp_cofit_edges.json","w"),indent=2)
print(f"\nwrote {OUT}/coexp_cofit_edges.json")
