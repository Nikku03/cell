"""Combined activator->target matcher: ADJACENCY + CO-FITNESS + FAMILY-EFFECTOR.
Head-to-head vs each signal alone, on Keio, validated on RegulonDB, by family.

For each candidate (activator A, target gene g):
  s_cofit  = |cofit(A,g)|                       (functional; feba Cofit)
  s_adj    = 1/genomic_distance(A,g) (+divergent bonus)   (positional)
  s_eff    = 1 if g's product matches A-family effector keywords else 0  (chemistry)
Rank all genes per activator by each signal and by the SUM (z-scored). Metric:
recall@K and precision@K (K=#true targets), enrichment over base rate, per family.
"""
import sqlite3, json, re
from collections import defaultdict, Counter
import numpy as np
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db"); cur=con.cursor()
OUT="/home/user/cell/outputs/orphan"

# maps
loc2name={}; name2loc={}; loc2coord={}; loc2prod={}
for lid,gn,sc,b,e,st,desc in cur.execute("SELECT locusId,gene,scaffoldId,begin,end,strand,desc FROM Gene WHERE orgId='Keio' AND type=1"):
    if gn: loc2name[lid]=gn.lower(); name2loc[gn.lower()]=lid
    loc2coord[lid]=(sc,(b+e)//2,st); loc2prod[lid]=(desc or "").lower()
allloc=sorted(loc2coord, key=lambda l:(loc2coord[l][0],loc2coord[l][1]))
order={l:i for i,l in enumerate(allloc)}
nG=len(loc2name)

# family from Pfam
PF=[("Crp","Crp/FNR"),("TetR","TetR"),("LysR","LysR"),("HTH_1","LysR"),("GntR","GntR"),
    ("MarR","MarR"),("LacI","LacI"),("AraC","AraC"),("HTH_AraC","AraC"),("IclR","IclR"),
    ("DeoR","DeoR"),("Lrp","Lrp/AsnC"),("AsnC","Lrp/AsnC"),("MerR","MerR"),("Sigma","Sigma"),
    ("Trans_reg_C","OmpR"),("LuxR","LuxR"),("Rrf2","Rrf2"),("Fis","Fis")]
loc2fam={}
for lid,dn in cur.execute("SELECT locusId,domainName FROM GeneDomain WHERE orgId='Keio' AND domainDb='PFam'"):
    if lid in loc2fam: continue
    for k,f in PF:
        if k.lower() in (dn or "").lower(): loc2fam[lid]=f; break
# family effector keywords (target product should mention these)
EFF={"LysR":["amino acid","aromatic","biosynth","oxid","catabol"],
     "GntR":["acid","sugar","catabol","dehydrogenase","fatty","metabol"],
     "TetR":["efflux","transport","drug","resist","membrane"],
     "AraC":["sugar","arabinose","catabol","transport"],
     "LacI":["sugar","transport","catabol"],
     "IclR":["acetate","glyoxylate","catabol"],
     "DeoR":["sugar","nucleoside","deoxy","catabol"],
     "MarR":["resist","efflux","oxidative","stress"]}

cofit=defaultdict(dict)
for lid,hit,cf in cur.execute("SELECT locusId,hitId,cofit FROM Cofit WHERE orgId='Keio'"):
    cofit[lid][hit]=abs(cf)

act_tgt=defaultdict(set)
for l in open("/home/user/cell/data/external_data/regulon/network_tf_gene.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t")
    if p[2]=="activator": act_tgt[p[0].lower()].add(p[1].lower())

def candidate_scores(lA, fam):
    """return dict gene_loc -> (s_cofit, s_adj, s_eff) for plausible candidates."""
    out={}
    # cofit candidates
    for h,cf in cofit.get(lA,{}).items():
        out.setdefault(h,[0,0,0])[0]=cf
    # adjacency candidates (window +-5)
    iA=order.get(lA)
    if iA is not None:
        for d in range(1,6):
            for j in (iA-d,iA+d):
                if 0<=j<len(allloc):
                    g=allloc[j]; s=1.0/d
                    # divergent bonus
                    if loc2coord[g][2]!=loc2coord[lA][2]: s+=0.5
                    out.setdefault(g,[0,0,0])[1]=max(out.get(g,[0,0,0])[1],s)
    # effector: among union candidates, mark product match
    kws=EFF.get(fam,[])
    for g in list(out):
        if kws and any(k in loc2prod.get(g,"") for k in kws): out[g][2]=1.0
    return out

def topK_hits(scored, truth, K, keyfn):
    ranked=sorted(scored, key=lambda g:-keyfn(scored[g]))
    top=ranked[:K] if K>0 else []
    return sum(1 for g in top if g in truth)

rows=[]
for act,tgts in act_tgt.items():
    if act not in name2loc: continue
    lA=name2loc[act]; fam=loc2fam.get(lA,"other")
    tloc={name2loc[t] for t in tgts if t in name2loc}
    if len(tloc)<5: continue
    sc=candidate_scores(lA,fam)
    if not sc: continue
    # z-score each signal across candidates for fair sum
    arr=np.array([sc[g] for g in sc],float)
    mu=arr.mean(0); sd=arr.std(0); sd[sd<1e-9]=1
    zc={g:(np.array(sc[g])-mu)/sd for g in sc}
    K=len(tloc)
    h_cofit=topK_hits(sc,tloc,K,lambda v:v[0])
    h_adj  =topK_hits(sc,tloc,K,lambda v:v[1])
    h_comb =topK_hits(zc,tloc,K,lambda v:v[0]+v[1]+0.5*v[2])
    base=K/nG
    rows.append(dict(activator=act,family=fam,n_targets=K,n_cand=len(sc),
                     prec_cofit=round(h_cofit/K,3),prec_adj=round(h_adj/K,3),prec_comb=round(h_comb/K,3),
                     enrich_comb=round((h_comb/K)/base,1)))

rows.sort(key=lambda r:-r["n_targets"])
print(f"activators tested: {len(rows)}\n")
print(f"{'activator':<9}{'family':<10}{'n':>4}{'cofit':>7}{'adj':>7}{'COMB':>7}{'enrich':>8}")
for r in rows[:22]:
    print(f"  {r['activator']:<7}{r['family']:<10}{r['n_targets']:>4}{r['prec_cofit']:>7.2f}{r['prec_adj']:>7.2f}{r['prec_comb']:>7.2f}{r['enrich_comb']:>7.0f}x")

import numpy as np
def m(k): return round(float(np.mean([r[k] for r in rows])),3)
print(f"\nOVERALL prec@K: cofit={m('prec_cofit')}  adj={m('prec_adj')}  COMBINED={m('prec_comb')}")
print("\n=== by family (precision@K) ===")
byf=defaultdict(list)
for r in rows: byf[r["family"]].append(r)
print(f"{'family':<11}{'n':>4}{'cofit':>7}{'adj':>7}{'COMB':>7}")
fam_sum={}
for fam,rs in sorted(byf.items(),key=lambda x:-len(x[1])):
    c=np.mean([r['prec_cofit'] for r in rs]); a=np.mean([r['prec_adj'] for r in rs]); cb=np.mean([r['prec_comb'] for r in rs])
    fam_sum[fam]=dict(n=len(rs),cofit=round(c,3),adj=round(a,3),comb=round(cb,3))
    print(f"  {fam:<9}{len(rs):>4}{c:>7.2f}{a:>7.2f}{cb:>7.2f}")
json.dump(dict(overall=dict(cofit=m('prec_cofit'),adj=m('prec_adj'),combined=m('prec_comb')),
               by_family=fam_sum,per_activator=rows),open(OUT+"/activator_combined.json","w"),indent=2)
print(f"\nwrote {OUT}/activator_combined.json")
