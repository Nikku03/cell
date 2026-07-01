"""Complete the hand-curated TF network (TRRUST) using DISEASE data. Principle: a disease
mutation is a natural perturbation. Genes that cause the SAME disease/phenotype as a TF are
candidate targets of that TF. Validate co-disease as an edge signal against held-out TRRUST
edges, then propose new edges. + annotate TFs with the diseases they cause."""
import json, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
H=Path("data/external_data/human"); OUT=Path("outputs/orphan"); rng=np.random.default_rng(0)
# HPO gene<->phenotype/disease
g2hpo=defaultdict(set); g2dis=defaultdict(set); dis2g=defaultdict(set); dis_name={}
for i,l in enumerate(open(H/"genes_to_phenotype.txt")):
    if i==0: continue
    p=l.rstrip("\n").split("\t")
    if len(p)<6: continue
    g=p[1]; hpo=p[2]; dis=p[5]
    g2hpo[g].add(hpo); g2dis[g].add(dis); dis2g[dis].add(g)
# TRRUST
tf_t=defaultdict(set); TFs=set(); known=set(); sign={}
for l in open(H/"trrust_human.tsv"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=3:
        tf_t[p[0]].add(p[1]); TFs.add(p[0]); known.add((p[0],p[1])); sign[(p[0],p[1])]=p[2]
hpo_genes=set(g2hpo)
TF_h=[t for t in TFs if t in hpo_genes]
print(f"TRRUST: {len(TFs)} TFs, {len(known)} edges. HPO: {len(hpo_genes)} disease genes, {len(dis2g)} diseases.")
print(f"TFs with disease data: {len(TF_h)}")

# --- validate: do known edges share a disease/phenotype more than random? ---
def share_disease(a,b): return len(g2dis[a]&g2dis[b])>0
def jacc(a,b):
    u=g2hpo[a]|g2hpo[b]; return len(g2hpo[a]&g2hpo[b])/len(u) if u else 0.0
known_h=[(a,b) for (a,b) in known if a in hpo_genes and b in hpo_genes]
# random TF-gene non-edges
allg=list(hpo_genes); neg=[]
while len(neg)<len(known_h):
    a=TF_h[rng.integers(len(TF_h))]; b=allg[rng.integers(len(allg))]
    if (a,b) not in known and a!=b: neg.append((a,b))
pos_share=np.mean([share_disease(a,b) for a,b in known_h])
neg_share=np.mean([share_disease(a,b) for a,b in neg])
pos_j=np.mean([jacc(a,b) for a,b in known_h]); neg_j=np.mean([jacc(a,b) for a,b in neg])
# AUC of jaccard separating known edges from random
sc=[jacc(a,b) for a,b in known_h]+[jacc(a,b) for a,b in neg]; y=[1]*len(known_h)+[0]*len(neg)
sc=np.array(sc);y=np.array(y);o=np.argsort(sc);r=np.empty(len(sc));r[o]=np.arange(len(sc));m=y==1
auc=float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))
print(f"\n=== validation: known TRRUST edges vs random (both disease genes, n={len(known_h)}) ===")
print(f"  share a disease: known {pos_share:.2f} vs random {neg_share:.2f}  ({pos_share/max(neg_share,1e-6):.1f}x)")
print(f"  phenotype Jaccard: known {pos_j:.3f} vs random {neg_j:.3f};  edge-recovery AUC = {auc:.3f}")

# --- complete the network: co-disease candidate edges ---
cand=defaultdict(lambda: defaultdict(float))   # tf -> gene -> best jaccard
for dis,genes in dis2g.items():
    tfs_in=[g for g in genes if g in TFs]
    for tf in tfs_in:
        for gene in genes:
            if gene!=tf and (tf,gene) not in known:
                cand[tf][gene]=max(cand[tf][gene], jacc(tf,gene))
# keep candidates with decent phenotype similarity
NEW=[]
for tf in cand:
    for gene,j in cand[tf].items():
        if j>=0.20: NEW.append((tf,gene,round(j,2)))
NEW.sort(key=lambda x:-x[2])
print(f"\n=== network completion (co-disease + phenotype Jaccard>=0.20) ===")
print(f"  candidate NEW TF->gene edges: {len(NEW)}  (TRRUST had {len(known)})")
print(f"  network grows {len(known)} -> {len(known)+len(NEW)} edges (+{100*len(NEW)/len(known):.0f}%)")
print("  top candidate new edges (TF -> gene, shared-disease, phenotype similarity):")
for tf,gene,j in NEW[:15]:
    d=list(g2dis[tf]&g2dis[gene])[:1]
    print(f"    {tf:8s} -> {gene:8s}  J={j}  shared disease {d}")

# --- TF disease annotation ---
tf_dis={t:sorted(g2dis[t])[:5] for t in TF_h if g2dis[t]}
print(f"\n=== example TF disease annotations ===")
for t in ["TP53","PAX6","TBX5","GATA4","SOX9","RUNX1"]:
    if t in g2dis: print(f"  {t:6s}: {len(g2dis[t])} diseases e.g. {sorted(g2dis[t])[:3]}")

json.dump(dict(trrust_edges=len(known), tfs=len(TFs), disease_genes=len(hpo_genes),
    validation=dict(known_share_disease=round(float(pos_share),3),random_share_disease=round(float(neg_share),3),
        enrichment=round(float(pos_share/max(neg_share,1e-6)),1),jaccard_auc=round(auc,3)),
    new_candidate_edges=len(NEW), grown_network_edges=len(known)+len(NEW),
    top_new_edges=[dict(tf=a,gene=b,jaccard=j,shared_disease=list(g2dis[a]&g2dis[b])[:2]) for a,b,j in NEW[:40]]),
    open(OUT/"human_network_completion.json","w"),indent=2)
print("\nwrote human_network_completion.json")
