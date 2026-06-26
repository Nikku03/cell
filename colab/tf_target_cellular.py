"""Cellular-logic TF->target predictor, validated on E. coli RegulonDB.

We predict which gene each TF regulates using the cell's OWN tricks, not a naive
genome-wide motif scan:

  (A) ADJACENCY  -- local regulators sit beside their target operon, usually
      divergently transcribed. Predict: TF -> the gene(s) immediately flanking
      it, especially a divergent neighbor sharing an intergenic promoter.
  (B) FAMILY-EFFECTOR -- a TF family implies an effector chemistry class; its
      real targets tend to be in pathways handling that chemistry. We approximate
      with annotation-keyword compatibility between TF family and target product.
  (C) CONVERGENCE -- the cell makes degenerate sites decisive by clustering
      several TFs on one promoter. We credit predictions where a target is
      flanked by / near multiple TFs.

Then VALIDATE against RegulonDB (TF gene -> target gene), split by local vs
global regulators (global = CRP/FNR/IHF/Fis/H-NS/ArcA/Fur/Lrp... the hubs that
do NOT sit next to targets).

Output: outputs/orphan/tf_target_cellular.json
"""
import csv, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")

# ---- gene coordinates (ordered along chromosome) ----
genes=[]   # (bnum, name, start, end, strand)
for l in open(REG/"GeneProductSet.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit() and p[5] in ("forward","reverse"):
        genes.append((p[2], p[1], int(p[3]), int(p[4]), p[5]))
genes.sort(key=lambda g:g[2])
name2idx={g[1].lower():i for i,g in enumerate(genes)}
bnum2name={g[0]:g[1] for g in genes}
N=len(genes)
print(f"genes with coords (chromosome order): {N}", flush=True)

# ---- RegulonDB ground truth: TF gene name -> set of target gene names ----
truth_edges=set(); tf_targets=defaultdict(set)
tf_outdeg=Counter()
for l in open(REG/"network_tf_gene.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t")
    tf, tgt = p[0].lower(), p[1].lower()
    truth_edges.add((tf,tgt)); tf_targets[tf].add(tgt); tf_outdeg[tf]+=1
print(f"RegulonDB truth: {len(truth_edges)} edges, {len(tf_targets)} TFs", flush=True)

# global regulators = top-degree hubs (do NOT sit next to targets)
GLOBAL=set([t for t,_ in tf_outdeg.most_common(15)])
print(f"global (hub) TFs: {sorted(GLOBAL)}", flush=True)

# which TF names have a chromosomal position?
tf_names=[t for t in tf_targets if t in name2idx]
print(f"TFs with a chromosome position: {len(tf_names)}/{len(tf_targets)}", flush=True)

# ---- (A) adjacency prediction ----
# For a TF at index i, predict targets = immediate neighbors within a window,
# with strongest weight on a DIVERGENT neighbor (TF and neighbor point away from
# a shared intergenic region) and same-strand immediate downstream (co-operonic).
def neighbors(i, window=3):
    out=[]
    for d in range(1, window+1):
        for j in (i-d, i+d):
            if 0<=j<N: out.append((j, d))
    return out

def divergent(a, b):
    # a is upstream-ish; divergent if a on reverse and b on forward with a left of b,
    # OR generally strands point away from the gap between them
    ga, gb = genes[a], genes[b]
    left, right = (ga, gb) if ga[2]<gb[2] else (gb, ga)
    return left[4]=="reverse" and right[4]=="forward"

pred_adj=defaultdict(dict)   # tf -> {target_name: score}
for tf in tf_names:
    i=name2idx[tf]
    for j,d in neighbors(i, window=3):
        tgt=genes[j][1].lower()
        if tgt==tf: continue
        score = 1.0/d
        if divergent(i,j): score += 1.5    # divergent pair = classic local-regulator architecture
        # intergenic gap small -> shared promoter more likely
        gap = abs(genes[j][2]-genes[i][3]) if j>i else abs(genes[i][2]-genes[j][3])
        if gap < 400: score += 0.5
        pred_adj[tf][tgt]=max(pred_adj[tf].get(tgt,0), score)

# ---- validate adjacency ----
def evaluate(pred, thresh, restrict=None):
    tp=fp=fn=0; pos=0
    for tf in tf_names:
        if restrict=="local" and tf in GLOBAL: continue
        if restrict=="global" and tf not in GLOBAL: continue
        real=tf_targets[tf]
        predicted={t for t,s in pred.get(tf,{}).items() if s>=thresh}
        tp+=len(predicted & real)
        fp+=len(predicted - real)
        # recall denominator: real targets that even COULD be predicted (have coords)
        real_coord={t for t in real if t in name2idx}
        fn+=len(real_coord - predicted)
        pos+=len(real_coord)
    P=tp/max(tp+fp,1); R=tp/max(pos,1)
    return dict(tp=tp,fp=fp,pos=pos,precision=round(P,3),recall=round(R,3),
                F1=round(2*P*R/max(P+R,1e-9),3))

print("\n=== (A) ADJACENCY prediction vs RegulonDB ===")
for label,restrict in [("ALL TFs",None),("LOCAL (non-hub)","local"),("GLOBAL (hub)","global")]:
    for th in [1.0, 2.0, 2.5]:
        r=evaluate(pred_adj, th, restrict)
        print(f"  {label:<18} thresh>={th}: P={r['precision']} R={r['recall']} F1={r['F1']}  (tp={r['tp']} fp={r['fp']} of {r['pos']} coord-targets)")
    print()

# ---- (C) CONVERGENCE: credit targets flanked by multiple TFs ----
# count how many TFs are within window of each gene
tf_idx={name2idx[t] for t in tf_names}
conv_count=Counter()
for gi in range(N):
    for j,_ in neighbors(gi, window=3):
        if j in tf_idx: conv_count[gi]+=1

# precision of adjacency predictions where target has >=2 flanking TFs
def eval_convergence(pred, thresh, min_tf):
    tp=fp=0
    for tf in tf_names:
        if tf in GLOBAL: continue
        real=tf_targets[tf]
        for t,s in pred.get(tf,{}).items():
            if s<thresh: continue
            ti=name2idx.get(t)
            if ti is None or conv_count[ti]<min_tf: continue
            if t in real: tp+=1
            else: fp+=1
    return tp, fp, round(tp/max(tp+fp,1),3)

print("=== (C) CONVERGENCE filter (local TFs, adjacency thresh>=1.0) ===")
for m in [1,2,3]:
    tp,fp,P=eval_convergence(pred_adj, 1.0, m)
    print(f"  require >={m} flanking TFs on target: P={P} (tp={tp} fp={fp})")

# ---- (B) FAMILY-EFFECTOR compatibility (annotation-keyword proxy) ----
# load product annotations for E. coli genes (use GeneProductSet col6 desc)
desc={}
for l in open(REG/"GeneProductSet.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t")
    if len(p)>=7: desc[p[1].lower()]=p[6].lower()
# family effector chemistry -> target keyword classes
EFFECTOR={
 "lysr":["amino acid","aromatic","catabol","oxid"],
 "tetr":["efflux","transport","drug","resist","membrane"],
 "gntr":["acid","sugar","catabol","dehydrogenase","metabol"],
 "arac":["sugar","arabinose","catabol","transport"],
 "marr":["resist","efflux","oxidative","stress","drug"],
 "lacl":["sugar","transport","catabol","glyco"],
 "iclr":["acetate","glyoxylate","catabol"],
}
def tf_family(tf):
    d=desc.get(tf,"")
    for fam in EFFECTOR:
        if fam in d.replace("-",""): return fam
    return None
# precision of adjacency predictions where target product matches TF effector class
def eval_effector(pred, thresh):
    tp=fp=0; n_applic=0
    for tf in tf_names:
        if tf in GLOBAL: continue
        fam=tf_family(tf)
        if not fam: continue
        kws=EFFECTOR[fam]
        for t,s in pred.get(tf,{}).items():
            if s<thresh: continue
            td=desc.get(t,"")
            if not any(k in td for k in kws): continue
            n_applic+=1
            if t in tf_targets[tf]: tp+=1
            else: fp+=1
    return tp, fp, round(tp/max(tp+fp,1),3), n_applic

print("\n=== (B) FAMILY-EFFECTOR filter (local TFs, adjacency thresh>=1.0) ===")
tp,fp,P,na=eval_effector(pred_adj,1.0)
print(f"  target product matches TF effector class: P={P} (tp={tp} fp={fp}, {na} predictions had a known family)")

# ---- COMBINED: adjacency + (convergence OR effector) ----
print("\n=== COMBINED cellular-logic (local TFs) ===")
base=evaluate(pred_adj, 1.0, "local")
print(f"  adjacency alone (local): P={base['precision']} R={base['recall']} F1={base['F1']}")

json.dump(dict(
   n_genes=N, n_truth_edges=len(truth_edges), n_tf_with_coords=len(tf_names),
   global_tfs=sorted(GLOBAL),
   adjacency_all=evaluate(pred_adj,1.0,None),
   adjacency_local=evaluate(pred_adj,1.0,"local"),
   adjacency_global=evaluate(pred_adj,1.0,"global"),
   adjacency_local_strict=evaluate(pred_adj,2.5,"local"),
   convergence={m:eval_convergence(pred_adj,1.0,m)[2] for m in (1,2,3)},
   effector_precision=P,
), open(OUT/"tf_target_cellular.json","w"), indent=2)
print("\nwrote outputs/orphan/tf_target_cellular.json")
