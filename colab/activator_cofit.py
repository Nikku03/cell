"""Match each ACTIVATOR (TF) to its targets via CO-FITNESS, validated on
RegulonDB, broken down by PROTEIN FAMILY.

Idea: an activator and the genes it activates have correlated fitness profiles
(knock out the activator -> lose the target's function -> same conditions hurt).
Co-fitness is a DATA-DRIVEN functional link that may recover activator->target
edges where sequence/motif failed (Wunderlich-Mirny ceiling).

For each Keio activator (RegulonDB) present in feba with >=5 in-Keio targets:
  candidate targets = its top co-fit partners (feba Cofit, up to 76/gene)
  metrics: recall@76 (true targets found among cofit partners),
           precision@K (K = #true targets), enrichment over base rate
  family   = from Pfam domain (HTH_Crp/TetR_N/HTH_LysR/GntR/MarR/...)
Aggregate by family: which activator families are recoverable by co-fitness?
Baseline: random gene ranking (enrichment 1.0).
"""
import sqlite3, json
from collections import defaultdict, Counter
import numpy as np
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db"); cur=con.cursor()
OUT="/home/user/cell/outputs/orphan"

# feba Keio maps
loc2name={}; name2loc={}
for lid,gn in cur.execute("SELECT locusId,gene FROM Gene WHERE orgId='Keio' AND type=1 AND gene!=''"):
    loc2name[lid]=gn.lower(); name2loc[gn.lower()]=lid
nG=len(loc2name)
# protein family from Pfam domain name
PF_FAMILY=[("Crp","Crp/FNR"),("TetR","TetR"),("LysR","LysR"),("HTH_1","LysR"),
           ("GntR","GntR"),("MarR","MarR"),("LacI","LacI"),("AraC","AraC"),
           ("HTH_AraC","AraC"),("Trans_reg_C","OmpR"),("Sigma","Sigma"),
           ("IclR","IclR"),("DeoR","DeoR"),("Fis","Fis"),("HTH_18","MarR"),
           ("LuxR","LuxR"),("Lrp","Lrp/AsnC"),("AsnC","Lrp/AsnC"),("MerR","MerR"),
           ("ArsR","ArsR"),("Rrf2","Rrf2"),("HTH_AsnC","Lrp/AsnC")]
loc2fam={}
for lid,dn in cur.execute("SELECT locusId,domainName FROM GeneDomain WHERE orgId='Keio' AND domainDb='PFam'"):
    if lid in loc2fam: continue
    for key,fam in PF_FAMILY:
        if key.lower() in (dn or "").lower(): loc2fam[lid]=fam; break

# cofit partners per locus
cofit=defaultdict(list)
for lid,hit,cf in cur.execute("SELECT locusId,hitId,cofit FROM Cofit WHERE orgId='Keio'"):
    cofit[lid].append((hit,cf))

# RegulonDB activator -> target names
act_tgt=defaultdict(set)
for l in open("/home/user/cell/data/external_data/regulon/network_tf_gene.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t")
    if p[2]=="activator": act_tgt[p[0].lower()].add(p[1].lower())

base_rate_per_tf=lambda nt: nt/max(nG,1)
rows=[]
for act,tgts in act_tgt.items():
    if act not in name2loc: continue
    lA=name2loc[act]
    tgt_loc={name2loc[t] for t in tgts if t in name2loc}
    if len(tgt_loc)<5: continue
    partners=cofit.get(lA,[])
    if not partners: continue
    part_loc=[h for h,_ in partners]
    part_set=set(part_loc)
    # recall@76
    found=tgt_loc & part_set
    recall=len(found)/len(tgt_loc)
    # precision@K (K = #targets): top-K cofit partners by |cofit|
    ranked=[h for h,_ in sorted(partners,key=lambda x:-abs(x[1]))]
    K=len(tgt_loc); topK=set(ranked[:K])
    precK=len(topK & tgt_loc)/max(K,1)
    base=base_rate_per_tf(len(tgt_loc))
    enrich=precK/base if base>0 else 0
    fam=loc2fam.get(lA,"other")
    rows.append(dict(activator=act, family=fam, n_targets=len(tgt_loc),
                     recall_at_cofit=round(recall,3), precision_at_K=round(precK,3),
                     base_rate=round(base,4), enrichment=round(enrich,1)))

rows.sort(key=lambda r:-r["n_targets"])
print(f"activators tested (>=5 in-Keio targets, has cofit): {len(rows)}\n")
print(f"{'activator':<10}{'family':<11}{'n_tgt':>6}{'recall@cofit':>13}{'prec@K':>8}{'enrich':>8}")
for r in rows[:25]:
    print(f"  {r['activator']:<8}{r['family']:<11}{r['n_targets']:>6}{r['recall_at_cofit']:>13.3f}{r['precision_at_K']:>8.3f}{r['enrichment']:>7.1f}x")

# aggregate by family
print("\n=== by protein family ===")
byf=defaultdict(list)
for r in rows: byf[r["family"]].append(r)
print(f"{'family':<12}{'n_act':>6}{'mean_recall':>12}{'mean_precK':>11}{'mean_enrich':>12}")
fam_summary={}
for fam,rs in sorted(byf.items(),key=lambda x:-len(x[1])):
    mr=np.mean([r["recall_at_cofit"] for r in rs]); mp=np.mean([r["precision_at_K"] for r in rs]); me=np.mean([r["enrichment"] for r in rs])
    fam_summary[fam]=dict(n=len(rs),mean_recall=round(mr,3),mean_precK=round(mp,3),mean_enrich=round(me,1))
    print(f"  {fam:<10}{len(rs):>6}{mr:>12.3f}{mp:>11.3f}{me:>11.1f}x")

mr=np.mean([r["recall_at_cofit"] for r in rows]); mp=np.mean([r["precision_at_K"] for r in rows]); me=np.mean([r["enrichment"] for r in rows])
print(f"\nOVERALL: mean recall@cofit={mr:.3f}  mean precision@K={mp:.3f}  mean enrichment={me:.1f}x over random")
json.dump(dict(n_activators=len(rows),overall_recall=round(mr,3),overall_precK=round(mp,3),
               overall_enrichment=round(me,1),by_family=fam_summary,per_activator=rows),
          open(OUT+"/activator_cofit.json","w"),indent=2)
print(f"\nwrote {OUT}/activator_cofit.json")
